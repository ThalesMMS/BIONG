from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b74_requires import b75_basal_thalamic_release_corridor_gate_result


def b76_cerebellar_stride_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b75_basal_thalamic_release_corridor_gate_result(results)
    explicit_decision_set = set(B76_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    stride_timing_signal_episodes = 0
    error_correction_episodes = 0
    burst_smoothing_episodes = 0
    stride_gate_episodes = 0
    stride_lock_or_release_episodes = 0
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
            str(item.get("b76_decision"))
            for item in trace
            if item.get("b76_decision") is not None
        ]
        stride_timings = [
            float(item.get("b76_stride_timing_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b76_stride_timing_signal") is not None
        ]
        error_corrections = [
            float(item.get("b76_error_correction", 0.0) or 0.0)
            for item in trace
            if item.get("b76_error_correction") is not None
        ]
        burst_smoothings = [
            float(item.get("b76_burst_smoothing", 0.0) or 0.0)
            for item in trace
            if item.get("b76_burst_smoothing") is not None
        ]
        stride_gates = [
            float(item.get("b76_stride_gate", 0.0) or 0.0)
            for item in trace
            if item.get("b76_stride_gate") is not None
        ]
        locks = [
            int(item.get("b76_stride_lock", 0) or 0)
            for item in trace
            if item.get("b76_stride_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        stride_timing_signal = any(abs(value) > 0.0 for value in stride_timings)
        error_correction = any(abs(value) > 0.0 for value in error_corrections)
        burst_smoothing = any(abs(value) > 0.0 for value in burst_smoothings)
        stride_gate = any(abs(value) > 0.0 for value in stride_gates)
        stride_lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "cerebellar_stride_hold",
                "continue_stride_lock",
                "timed_stride_release",
                "smoothed_burst_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if stride_timing_signal:
            stride_timing_signal_episodes += 1
        if error_correction:
            error_correction_episodes += 1
        if burst_smoothing:
            burst_smoothing_episodes += 1
        if stride_gate:
            stride_gate_episodes += 1
        if stride_lock_or_release:
            stride_lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b76_decision": bool(explicit_decision),
                    "stride_timing_signal": bool(stride_timing_signal),
                    "error_correction": bool(error_correction),
                    "burst_smoothing": bool(burst_smoothing),
                    "stride_gate": bool(stride_gate),
                    "stride_lock_or_release": bool(stride_lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "stride_timing_signals": stride_timings,
                "error_corrections": error_corrections,
                "burst_smoothings": burst_smoothings,
                "stride_gates": stride_gates,
                "stride_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b75_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b76_decision_episodes": explicit_decision_episodes >= 2,
        "stride_timing_signal_episodes": stride_timing_signal_episodes >= 2,
        "error_correction_episodes": error_correction_episodes >= 2,
        "burst_smoothing_episodes": burst_smoothing_episodes >= 2,
        "stride_gate_episodes": stride_gate_episodes >= 2,
        "stride_lock_or_release_episodes": stride_lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b76_aggregate:" + name
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
            "base_b75_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "stride_timing_signal_episodes": int(stride_timing_signal_episodes),
            "error_correction_episodes": int(error_correction_episodes),
            "burst_smoothing_episodes": int(burst_smoothing_episodes),
            "stride_gate_episodes": int(stride_gate_episodes),
            "stride_lock_or_release_episodes": int(
                stride_lock_or_release_episodes
            ),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
