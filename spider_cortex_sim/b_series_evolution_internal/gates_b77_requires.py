from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b76_requires import b77_olivary_error_corridor_gate_result


def b78_vestibular_balance_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b77_olivary_error_corridor_gate_result(results)
    explicit_decision_set = set(B78_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    balance_error_episodes = 0
    head_stabilization_episodes = 0
    locomotor_confidence_episodes = 0
    slip_risk_episodes = 0
    balance_lock_or_release_episodes = 0
    corridor_safety_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        primitive_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
        predator_contacts = result_predator_contacts(result)
        decisions = [
            str(item.get("b78_decision"))
            for item in trace
            if item.get("b78_decision") is not None
        ]
        balance_errors = [
            float(item.get("b78_balance_error", 0.0) or 0.0)
            for item in trace
            if item.get("b78_balance_error") is not None
        ]
        head_stabilizations = [
            float(item.get("b78_head_stabilization", 0.0) or 0.0)
            for item in trace
            if item.get("b78_head_stabilization") is not None
        ]
        locomotor_confidences = [
            float(item.get("b78_locomotor_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b78_locomotor_confidence") is not None
        ]
        slip_risks = [
            float(item.get("b78_slip_risk", 0.0) or 0.0)
            for item in trace
            if item.get("b78_slip_risk") is not None
        ]
        locks = [
            int(item.get("b78_balance_lock", 0) or 0)
            for item in trace
            if item.get("b78_balance_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        balance_error = any(abs(value) > 0.0 for value in balance_errors)
        head_stabilization = any(abs(value) > 0.0 for value in head_stabilizations)
        locomotor_confidence = any(abs(value) > 0.0 for value in locomotor_confidences)
        slip_risk = any(abs(value) > 0.0 for value in slip_risks)
        balance_lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "vestibular_balance_hold",
                "continue_balance_lock",
                "stabilized_corridor_release",
                "balance_recovery_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if balance_error:
            balance_error_episodes += 1
        if head_stabilization:
            head_stabilization_episodes += 1
        if locomotor_confidence:
            locomotor_confidence_episodes += 1
        if slip_risk:
            slip_risk_episodes += 1
        if balance_lock_or_release:
            balance_lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b78_decision": bool(explicit_decision),
                    "balance_error": bool(balance_error),
                    "head_stabilization": bool(head_stabilization),
                    "locomotor_confidence": bool(locomotor_confidence),
                    "slip_risk": bool(slip_risk),
                    "balance_lock_or_release": bool(balance_lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "balance_errors": balance_errors,
                "head_stabilizations": head_stabilizations,
                "locomotor_confidences": locomotor_confidences,
                "slip_risks": slip_risks,
                "balance_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b77_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b78_decision_episodes": explicit_decision_episodes >= 2,
        "balance_error_episodes": balance_error_episodes >= 2,
        "head_stabilization_episodes": head_stabilization_episodes >= 2,
        "locomotor_confidence_episodes": locomotor_confidence_episodes >= 2,
        "slip_risk_episodes": slip_risk_episodes >= 2,
        "balance_lock_or_release_episodes": balance_lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b78_aggregate:" + name
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
            "base_b77_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "balance_error_episodes": int(balance_error_episodes),
            "head_stabilization_episodes": int(head_stabilization_episodes),
            "locomotor_confidence_episodes": int(locomotor_confidence_episodes),
            "slip_risk_episodes": int(slip_risk_episodes),
            "balance_lock_or_release_episodes": int(
                balance_lock_or_release_episodes
            ),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
