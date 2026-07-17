from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b61_requires import b66_immune_malaise_corridor_gate_result


def b67_glial_energy_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b66_immune_malaise_corridor_gate_result(results)
    explicit_decision_set = set(B67_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    glial_reserve_episodes = 0
    lactate_support_episodes = 0
    fatigue_pressure_episodes = 0
    lock_or_release_episodes = 0
    corridor_safety_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        primitive_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
        predator_contacts = result_predator_contacts(result)
        decisions = [
            str(item.get("b67_decision"))
            for item in trace
            if item.get("b67_decision") is not None
        ]
        glial_reserves = [
            float(item.get("b67_glial_reserve", 0.0) or 0.0)
            for item in trace
            if item.get("b67_glial_reserve") is not None
        ]
        lactate_supports = [
            float(item.get("b67_lactate_support", 0.0) or 0.0)
            for item in trace
            if item.get("b67_lactate_support") is not None
        ]
        fatigue_pressures = [
            float(item.get("b67_fatigue_pressure", 0.0) or 0.0)
            for item in trace
            if item.get("b67_fatigue_pressure") is not None
        ]
        locks = [
            int(item.get("b67_recovery_lock", 0) or 0)
            for item in trace
            if item.get("b67_recovery_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        glial_reserve = any(abs(value) > 0.0 for value in glial_reserves)
        lactate_support = any(abs(value) > 0.0 for value in lactate_supports)
        fatigue_pressure = any(abs(value) > 0.0 for value in fatigue_pressures)
        lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "glial_recovery_support",
                "continue_glial_recovery_lock",
                "glial_energy_release",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if glial_reserve:
            glial_reserve_episodes += 1
        if lactate_support:
            lactate_support_episodes += 1
        if fatigue_pressure:
            fatigue_pressure_episodes += 1
        if lock_or_release:
            lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b67_decision": bool(explicit_decision),
                    "glial_reserve": bool(glial_reserve),
                    "lactate_support": bool(lactate_support),
                    "fatigue_pressure": bool(fatigue_pressure),
                    "lock_or_release": bool(lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "glial_reserves": glial_reserves,
                "lactate_supports": lactate_supports,
                "fatigue_pressures": fatigue_pressures,
                "recovery_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b66_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b67_decision_episodes": explicit_decision_episodes >= 2,
        "glial_reserve_episodes": glial_reserve_episodes >= 2,
        "lactate_support_episodes": lactate_support_episodes >= 2,
        "fatigue_pressure_episodes": fatigue_pressure_episodes >= 2,
        "lock_or_release_episodes": lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b67_aggregate:" + name
        for name, ok in aggregate_checks.items()
        if not ok
    )
    passed = not failures
    return {
        "scenario": B6_CORRIDOR_SCENARIO,
        "status": "accepted" if passed else "discarded",
        "passed": passed,
        "base_gate": base_gate,
        "aggregate": {
            "base_b66_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "glial_reserve_episodes": int(glial_reserve_episodes),
            "lactate_support_episodes": int(lactate_support_episodes),
            "fatigue_pressure_episodes": int(fatigue_pressure_episodes),
            "lock_or_release_episodes": int(lock_or_release_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
