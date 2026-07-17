from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b72_requires import b73_reticular_inhibition_corridor_gate_result


def b74_thalamic_rebound_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b73_reticular_inhibition_corridor_gate_result(results)
    explicit_decision_set = set(B74_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    rebound_potential_episodes = 0
    inhibition_aftereffect_episodes = 0
    release_window_episodes = 0
    rebound_lock_or_release_episodes = 0
    corridor_safety_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        primitive_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
        predator_contacts = result_predator_contacts(result)
        decisions = [
            str(item.get("b74_decision"))
            for item in trace
            if item.get("b74_decision") is not None
        ]
        rebound_potentials = [
            float(item.get("b74_rebound_potential", 0.0) or 0.0)
            for item in trace
            if item.get("b74_rebound_potential") is not None
        ]
        inhibition_aftereffects = [
            float(item.get("b74_inhibition_aftereffect", 0.0) or 0.0)
            for item in trace
            if item.get("b74_inhibition_aftereffect") is not None
        ]
        release_windows = [
            float(item.get("b74_release_window", 0.0) or 0.0)
            for item in trace
            if item.get("b74_release_window") is not None
        ]
        locks = [
            int(item.get("b74_rebound_lock", 0) or 0)
            for item in trace
            if item.get("b74_rebound_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        rebound_potential = any(abs(value) > 0.0 for value in rebound_potentials)
        inhibition_aftereffect = any(
            abs(value) > 0.0 for value in inhibition_aftereffects
        )
        release_window = any(abs(value) > 0.0 for value in release_windows)
        rebound_lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "thalamic_rebound_hold",
                "continue_rebound_lock",
                "rebound_release_stride",
                "post_inhibition_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if rebound_potential:
            rebound_potential_episodes += 1
        if inhibition_aftereffect:
            inhibition_aftereffect_episodes += 1
        if release_window:
            release_window_episodes += 1
        if rebound_lock_or_release:
            rebound_lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b74_decision": bool(explicit_decision),
                    "rebound_potential": bool(rebound_potential),
                    "inhibition_aftereffect": bool(inhibition_aftereffect),
                    "release_window": bool(release_window),
                    "rebound_lock_or_release": bool(rebound_lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "rebound_potentials": rebound_potentials,
                "inhibition_aftereffects": inhibition_aftereffects,
                "release_windows": release_windows,
                "rebound_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b73_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b74_decision_episodes": explicit_decision_episodes >= 2,
        "rebound_potential_episodes": rebound_potential_episodes >= 2,
        "inhibition_aftereffect_episodes": inhibition_aftereffect_episodes >= 2,
        "release_window_episodes": release_window_episodes >= 2,
        "rebound_lock_or_release_episodes": rebound_lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b74_aggregate:" + name
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
            "base_b73_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "rebound_potential_episodes": int(rebound_potential_episodes),
            "inhibition_aftereffect_episodes": int(inhibition_aftereffect_episodes),
            "release_window_episodes": int(release_window_episodes),
            "rebound_lock_or_release_episodes": int(
                rebound_lock_or_release_episodes
            ),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
