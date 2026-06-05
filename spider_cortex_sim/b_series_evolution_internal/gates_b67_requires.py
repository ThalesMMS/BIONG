from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b66_requires import b67_glial_energy_corridor_gate_result


def b68_motor_pacing_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b67_glial_energy_corridor_gate_result(results)
    explicit_decision_set = set(B68_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    motor_reserve_episodes = 0
    stride_pacing_episodes = 0
    overexertion_risk_episodes = 0
    lock_or_release_episodes = 0
    corridor_safety_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        metrics = result.get("metrics", {})
        primitive_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
        predator_contacts = int(
            metrics.get("predator_contacts", result.get("predator_contacts", 0)) or 0
        )
        decisions = [
            str(item.get("b68_decision"))
            for item in trace
            if item.get("b68_decision") is not None
        ]
        motor_reserves = [
            float(item.get("b68_motor_reserve", 0.0) or 0.0)
            for item in trace
            if item.get("b68_motor_reserve") is not None
        ]
        stride_pacings = [
            float(item.get("b68_stride_pacing", 0.0) or 0.0)
            for item in trace
            if item.get("b68_stride_pacing") is not None
        ]
        overexertion_risks = [
            float(item.get("b68_overexertion_risk", 0.0) or 0.0)
            for item in trace
            if item.get("b68_overexertion_risk") is not None
        ]
        locks = [
            int(item.get("b68_pacing_lock", 0) or 0)
            for item in trace
            if item.get("b68_pacing_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        motor_reserve = any(abs(value) > 0.0 for value in motor_reserves)
        stride_pacing = any(abs(value) > 0.0 for value in stride_pacings)
        overexertion_risk = any(abs(value) > 0.0 for value in overexertion_risks)
        lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "motor_recovery_pace",
                "continue_motor_pacing_lock",
                "motor_safe_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if motor_reserve:
            motor_reserve_episodes += 1
        if stride_pacing:
            stride_pacing_episodes += 1
        if overexertion_risk:
            overexertion_risk_episodes += 1
        if lock_or_release:
            lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b68_decision": bool(explicit_decision),
                    "motor_reserve": bool(motor_reserve),
                    "stride_pacing": bool(stride_pacing),
                    "overexertion_risk": bool(overexertion_risk),
                    "lock_or_release": bool(lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "motor_reserves": motor_reserves,
                "stride_pacings": stride_pacings,
                "overexertion_risks": overexertion_risks,
                "pacing_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b67_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b68_decision_episodes": explicit_decision_episodes >= 2,
        "motor_reserve_episodes": motor_reserve_episodes >= 2,
        "stride_pacing_episodes": stride_pacing_episodes >= 2,
        "overexertion_risk_episodes": overexertion_risk_episodes >= 2,
        "lock_or_release_episodes": lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b68_aggregate:" + name
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
            "base_b67_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "motor_reserve_episodes": int(motor_reserve_episodes),
            "stride_pacing_episodes": int(stride_pacing_episodes),
            "overexertion_risk_episodes": int(overexertion_risk_episodes),
            "lock_or_release_episodes": int(lock_or_release_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
