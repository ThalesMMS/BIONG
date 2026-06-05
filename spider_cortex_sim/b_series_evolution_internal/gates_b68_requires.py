from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b67_requires import b68_motor_pacing_corridor_gate_result


def b69_vestibular_orientation_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b68_motor_pacing_corridor_gate_result(results)
    explicit_decision_set = set(B69_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    heading_confidence_episodes = 0
    turn_error_episodes = 0
    orientation_stability_episodes = 0
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
            str(item.get("b69_decision"))
            for item in trace
            if item.get("b69_decision") is not None
        ]
        heading_confidences = [
            float(item.get("b69_heading_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b69_heading_confidence") is not None
        ]
        turn_errors = [
            float(item.get("b69_turn_error", 0.0) or 0.0)
            for item in trace
            if item.get("b69_turn_error") is not None
        ]
        orientation_stabilities = [
            float(item.get("b69_orientation_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b69_orientation_stability") is not None
        ]
        locks = [
            int(item.get("b69_orientation_lock", 0) or 0)
            for item in trace
            if item.get("b69_orientation_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        heading_confidence = any(abs(value) > 0.0 for value in heading_confidences)
        turn_error = any(abs(value) > 0.0 for value in turn_errors)
        orientation_stability = any(abs(value) > 0.0 for value in orientation_stabilities)
        lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "vestibular_recenter",
                "continue_orientation_lock",
                "stable_heading_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if heading_confidence:
            heading_confidence_episodes += 1
        if turn_error:
            turn_error_episodes += 1
        if orientation_stability:
            orientation_stability_episodes += 1
        if lock_or_release:
            lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b69_decision": bool(explicit_decision),
                    "heading_confidence": bool(heading_confidence),
                    "turn_error": bool(turn_error),
                    "orientation_stability": bool(orientation_stability),
                    "lock_or_release": bool(lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "heading_confidences": heading_confidences,
                "turn_errors": turn_errors,
                "orientation_stabilities": orientation_stabilities,
                "orientation_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b68_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b69_decision_episodes": explicit_decision_episodes >= 2,
        "heading_confidence_episodes": heading_confidence_episodes >= 2,
        "turn_error_episodes": turn_error_episodes >= 2,
        "orientation_stability_episodes": orientation_stability_episodes >= 2,
        "lock_or_release_episodes": lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b69_aggregate:" + name
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
            "base_b68_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "heading_confidence_episodes": int(heading_confidence_episodes),
            "turn_error_episodes": int(turn_error_episodes),
            "orientation_stability_episodes": int(orientation_stability_episodes),
            "lock_or_release_episodes": int(lock_or_release_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
