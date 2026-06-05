from __future__ import annotations

from .shared import *
from .constants import *

from .gates_b1_b6 import trace_uses_only_primitive_actions
from .gates_b70_requires import b71_tectal_orienting_corridor_gate_result


def b72_pulvinar_attention_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b71_tectal_orienting_corridor_gate_result(results)
    explicit_decision_set = set(B72_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    focus_signal_episodes = 0
    distractor_load_episodes = 0
    filter_gain_episodes = 0
    attention_lock_or_release_episodes = 0
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
            str(item.get("b72_decision"))
            for item in trace
            if item.get("b72_decision") is not None
        ]
        focus_signals = [
            float(item.get("b72_focus_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b72_focus_signal") is not None
        ]
        distractor_loads = [
            float(item.get("b72_distractor_load", 0.0) or 0.0)
            for item in trace
            if item.get("b72_distractor_load") is not None
        ]
        filter_gains = [
            float(item.get("b72_filter_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b72_filter_gain") is not None
        ]
        locks = [
            int(item.get("b72_attention_lock", 0) or 0)
            for item in trace
            if item.get("b72_attention_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        focus_signal = any(abs(value) > 0.0 for value in focus_signals)
        distractor_load = any(abs(value) > 0.0 for value in distractor_loads)
        filter_gain = any(abs(value) > 0.0 for value in filter_gains)
        attention_lock_or_release = any(lock > 0 for lock in locks) or any(
            decision
            in {
                "pulvinar_filter_recenter",
                "continue_attention_lock",
                "filtered_target_stride",
                "selective_focus_stride",
            }
            for decision in decisions
        )
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if focus_signal:
            focus_signal_episodes += 1
        if distractor_load:
            distractor_load_episodes += 1
        if filter_gain:
            filter_gain_episodes += 1
        if attention_lock_or_release:
            attention_lock_or_release_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b72_decision": bool(explicit_decision),
                    "focus_signal": bool(focus_signal),
                    "distractor_load": bool(distractor_load),
                    "filter_gain": bool(filter_gain),
                    "attention_lock_or_release": bool(attention_lock_or_release),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "focus_signals": focus_signals,
                "distractor_loads": distractor_loads,
                "filter_gains": filter_gains,
                "attention_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b71_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b72_decision_episodes": explicit_decision_episodes >= 2,
        "focus_signal_episodes": focus_signal_episodes >= 2,
        "distractor_load_episodes": distractor_load_episodes >= 2,
        "filter_gain_episodes": filter_gain_episodes >= 2,
        "attention_lock_or_release_episodes": attention_lock_or_release_episodes >= 2,
    }
    failures.extend(
        "corridor_b72_aggregate:" + name
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
            "base_b71_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "focus_signal_episodes": int(focus_signal_episodes),
            "distractor_load_episodes": int(distractor_load_episodes),
            "filter_gain_episodes": int(filter_gain_episodes),
            "attention_lock_or_release_episodes": int(attention_lock_or_release_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
