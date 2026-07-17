from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b75_requires import b76_cerebellar_stride_corridor_gate_result


def b77_olivary_error_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b76_cerebellar_stride_corridor_gate_result(results)
    explicit_decision_set = set(B77_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    prediction_error_episodes = 0
    climbing_fiber_drive_episodes = 0
    corrective_timing_episodes = 0
    stability_confidence_episodes = 0
    error_lock_or_release_episodes = 0
    corridor_safety_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        primitive_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
        predator_contacts = result_predator_contacts(result)
        decisions = [
            str(item.get("b77_decision"))
            for item in trace
            if item.get("b77_decision") is not None
        ]
        prediction_errors = [
            float(item.get("b77_prediction_error", 0.0) or 0.0)
            for item in trace
            if item.get("b77_prediction_error") is not None
        ]
        climbing_drives = [
            float(item.get("b77_climbing_fiber_drive", 0.0) or 0.0)
            for item in trace
            if item.get("b77_climbing_fiber_drive") is not None
        ]
        corrective_timings = [
            float(item.get("b77_corrective_timing", 0.0) or 0.0)
            for item in trace
            if item.get("b77_corrective_timing") is not None
        ]
        stability_confidences = [
            float(item.get("b77_stability_confidence", 0.0) or 0.0)
            for item in trace
            if item.get("b77_stability_confidence") is not None
        ]
        locks = [
            int(item.get("b77_error_lock", 0) or 0)
            for item in trace
            if item.get("b77_error_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        prediction_error = any(abs(value) > 0.0 for value in prediction_errors)
        climbing_fiber_drive = any(abs(value) > 0.0 for value in climbing_drives)
        corrective_timing = any(abs(value) > 0.0 for value in corrective_timings)
        stability_confidence = any(abs(value) > 0.0 for value in stability_confidences)
        error_lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "olivary_error_hold",
                "continue_error_lock",
                "corrective_stride_release",
                "stabilized_stride_release",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if prediction_error:
            prediction_error_episodes += 1
        if climbing_fiber_drive:
            climbing_fiber_drive_episodes += 1
        if corrective_timing:
            corrective_timing_episodes += 1
        if stability_confidence:
            stability_confidence_episodes += 1
        if error_lock_or_release:
            error_lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b77_decision": bool(explicit_decision),
                    "prediction_error": bool(prediction_error),
                    "climbing_fiber_drive": bool(climbing_fiber_drive),
                    "corrective_timing": bool(corrective_timing),
                    "stability_confidence": bool(stability_confidence),
                    "error_lock_or_release": bool(error_lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "prediction_errors": prediction_errors,
                "climbing_fiber_drives": climbing_drives,
                "corrective_timings": corrective_timings,
                "stability_confidences": stability_confidences,
                "error_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b76_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b77_decision_episodes": explicit_decision_episodes >= 2,
        "prediction_error_episodes": prediction_error_episodes >= 2,
        "climbing_fiber_drive_episodes": climbing_fiber_drive_episodes >= 2,
        "corrective_timing_episodes": corrective_timing_episodes >= 2,
        "stability_confidence_episodes": stability_confidence_episodes >= 2,
        "error_lock_or_release_episodes": error_lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b77_aggregate:" + name
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
            "base_b76_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "prediction_error_episodes": int(prediction_error_episodes),
            "climbing_fiber_drive_episodes": int(climbing_fiber_drive_episodes),
            "corrective_timing_episodes": int(corrective_timing_episodes),
            "stability_confidence_episodes": int(stability_confidence_episodes),
            "error_lock_or_release_episodes": int(
                error_lock_or_release_episodes
            ),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
