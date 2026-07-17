from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b68_requires import b69_vestibular_orientation_corridor_gate_result


def b70_optic_flow_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b69_vestibular_orientation_corridor_gate_result(results)
    explicit_decision_set = set(B70_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    flow_confidence_episodes = 0
    lateral_drift_episodes = 0
    looming_risk_episodes = 0
    lock_or_release_episodes = 0
    corridor_safety_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        primitive_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
        predator_contacts = result_predator_contacts(result)
        decisions = [
            str(item.get("b70_decision"))
            for item in trace
            if item.get("b70_decision") is not None
        ]
        flow_confidences = [
            float(item.get("b70_flow_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b70_flow_confidence") is not None
        ]
        lateral_drifts = [
            float(item.get("b70_lateral_drift", 0.0) or 0.0)
            for item in trace
            if item.get("b70_lateral_drift") is not None
        ]
        looming_risks = [
            float(item.get("b70_looming_risk", 0.0) or 0.0)
            for item in trace
            if item.get("b70_looming_risk") is not None
        ]
        locks = [
            int(item.get("b70_flow_lock", 0) or 0)
            for item in trace
            if item.get("b70_flow_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        flow_confidence = any(abs(value) > 0.0 for value in flow_confidences)
        lateral_drift = any(abs(value) > 0.0 for value in lateral_drifts)
        looming_risk = any(abs(value) > 0.0 for value in looming_risks)
        lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "optic_flow_recenter",
                "continue_flow_lock",
                "clear_path_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if flow_confidence:
            flow_confidence_episodes += 1
        if lateral_drift:
            lateral_drift_episodes += 1
        if looming_risk:
            looming_risk_episodes += 1
        if lock_or_release:
            lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b70_decision": bool(explicit_decision),
                    "flow_confidence": bool(flow_confidence),
                    "lateral_drift": bool(lateral_drift),
                    "looming_risk": bool(looming_risk),
                    "lock_or_release": bool(lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "flow_confidences": flow_confidences,
                "lateral_drifts": lateral_drifts,
                "looming_risks": looming_risks,
                "flow_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b69_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b70_decision_episodes": explicit_decision_episodes >= 2,
        "flow_confidence_episodes": flow_confidence_episodes >= 2,
        "lateral_drift_episodes": lateral_drift_episodes >= 2,
        "looming_risk_episodes": looming_risk_episodes >= 2,
        "lock_or_release_episodes": lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b70_aggregate:" + name
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
            "base_b69_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "flow_confidence_episodes": int(flow_confidence_episodes),
            "lateral_drift_episodes": int(lateral_drift_episodes),
            "looming_risk_episodes": int(looming_risk_episodes),
            "lock_or_release_episodes": int(lock_or_release_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
