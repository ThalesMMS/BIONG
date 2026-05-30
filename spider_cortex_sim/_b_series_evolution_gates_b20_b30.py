from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_gates_b1_b6 import (
    trace_uses_only_primitive_actions,
)

from ._b_series_evolution_gates_b7_b19 import (
    b19_episodic_corridor_gate_result,
)

def b20_working_memory_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b19_episodic_corridor_gate_result(results)
    explicit_decision_set = set(B20_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    buffer_state_episodes = 0
    buffer_lock_episodes = 0
    buffer_signal_episodes = 0
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
            str(item.get("b20_decision"))
            for item in trace
            if item.get("b20_decision") is not None
        ]
        states = [
            str(item.get("b20_buffer_state"))
            for item in trace
            if item.get("b20_buffer_state") is not None
        ]
        locks = [
            int(item.get("b20_buffer_lock", 0) or 0)
            for item in trace
            if item.get("b20_buffer_lock") is not None
        ]
        signals = [
            float(item.get("b20_working_buffer", 0.0) or 0.0)
            + float(item.get("b20_context_binding", 0.0) or 0.0)
            + float(item.get("b20_gate_vote", 0.0) or 0.0)
            + float(item.get("b20_release_vote", 0.0) or 0.0)
            for item in trace
            if item.get("b20_working_buffer") is not None
            or item.get("b20_context_binding") is not None
            or item.get("b20_gate_vote") is not None
            or item.get("b20_release_vote") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        buffer_state = any(state != "non_corridor" for state in states)
        buffer_lock = explicit_decision and any(lock > 0 for lock in locks)
        buffer_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if buffer_state:
            buffer_state_episodes += 1
        if buffer_lock:
            buffer_lock_episodes += 1
        if buffer_signal:
            buffer_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b20_decision": bool(explicit_decision),
                    "buffer_state": bool(buffer_state),
                    "buffer_lock": bool(buffer_lock),
                    "buffer_signal": bool(buffer_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "buffer_states": states,
                "buffer_locks": locks,
                "buffer_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b20_decision_episodes": explicit_decision_episodes >= 2,
        "buffer_state_episodes": buffer_state_episodes >= 2,
        "buffer_lock_episodes": buffer_lock_episodes >= 2,
        "buffer_signal_episodes": buffer_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b20_aggregate:" + name
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
            "base_b19_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "buffer_state_episodes": int(buffer_state_episodes),
            "buffer_lock_episodes": int(buffer_lock_episodes),
            "buffer_signal_episodes": int(buffer_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b21_hippocampal_replay_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b20_working_memory_corridor_gate_result(results)
    explicit_decision_set = set(B21_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    replay_state_episodes = 0
    replay_lock_episodes = 0
    replay_signal_episodes = 0
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
            str(item.get("b21_decision"))
            for item in trace
            if item.get("b21_decision") is not None
        ]
        states = [
            str(item.get("b21_replay_state"))
            for item in trace
            if item.get("b21_replay_state") is not None
        ]
        locks = [
            int(item.get("b21_replay_lock", 0) or 0)
            for item in trace
            if item.get("b21_replay_lock") is not None
        ]
        signals = [
            float(item.get("b21_sequence_memory", 0.0) or 0.0)
            + float(item.get("b21_replay_score", 0.0) or 0.0)
            + float(item.get("b21_route_commitment", 0.0) or 0.0)
            + float(item.get("b21_abort_prediction", 0.0) or 0.0)
            for item in trace
            if item.get("b21_sequence_memory") is not None
            or item.get("b21_replay_score") is not None
            or item.get("b21_route_commitment") is not None
            or item.get("b21_abort_prediction") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        replay_state = any(state != "non_corridor" for state in states)
        replay_lock = explicit_decision and any(lock > 0 for lock in locks)
        replay_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if replay_state:
            replay_state_episodes += 1
        if replay_lock:
            replay_lock_episodes += 1
        if replay_signal:
            replay_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b21_decision": bool(explicit_decision),
                    "replay_state": bool(replay_state),
                    "replay_lock": bool(replay_lock),
                    "replay_signal": bool(replay_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "replay_states": states,
                "replay_locks": locks,
                "replay_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b21_decision_episodes": explicit_decision_episodes >= 2,
        "replay_state_episodes": replay_state_episodes >= 2,
        "replay_lock_episodes": replay_lock_episodes >= 2,
        "replay_signal_episodes": replay_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b21_aggregate:" + name
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
            "base_b20_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "replay_state_episodes": int(replay_state_episodes),
            "replay_lock_episodes": int(replay_lock_episodes),
            "replay_signal_episodes": int(replay_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b22_prospective_replay_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b21_hippocampal_replay_corridor_gate_result(results)
    explicit_decision_set = set(B22_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    sim_state_episodes = 0
    sim_lock_episodes = 0
    sim_signal_episodes = 0
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
            str(item.get("b22_decision"))
            for item in trace
            if item.get("b22_decision") is not None
        ]
        states = [
            str(item.get("b22_sim_state"))
            for item in trace
            if item.get("b22_sim_state") is not None
        ]
        locks = [
            int(item.get("b22_sim_lock", 0) or 0)
            for item in trace
            if item.get("b22_sim_lock") is not None
        ]
        signals = [
            float(item.get("b22_prospective_sim", 0.0) or 0.0)
            + float(item.get("b22_forward_model_score", 0.0) or 0.0)
            + float(item.get("b22_viability_projection", 0.0) or 0.0)
            + float(item.get("b22_abort_projection", 0.0) or 0.0)
            for item in trace
            if item.get("b22_prospective_sim") is not None
            or item.get("b22_forward_model_score") is not None
            or item.get("b22_viability_projection") is not None
            or item.get("b22_abort_projection") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        sim_state = any(state != "non_corridor" for state in states)
        sim_lock = explicit_decision and any(lock > 0 for lock in locks)
        sim_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if sim_state:
            sim_state_episodes += 1
        if sim_lock:
            sim_lock_episodes += 1
        if sim_signal:
            sim_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b22_decision": bool(explicit_decision),
                    "sim_state": bool(sim_state),
                    "sim_lock": bool(sim_lock),
                    "sim_signal": bool(sim_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "sim_states": states,
                "sim_locks": locks,
                "sim_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b22_decision_episodes": explicit_decision_episodes >= 2,
        "sim_state_episodes": sim_state_episodes >= 2,
        "sim_lock_episodes": sim_lock_episodes >= 2,
        "sim_signal_episodes": sim_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b22_aggregate:" + name
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
            "base_b21_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "sim_state_episodes": int(sim_state_episodes),
            "sim_lock_episodes": int(sim_lock_episodes),
            "sim_signal_episodes": int(sim_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b23_conflict_monitor_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b22_prospective_replay_corridor_gate_result(results)
    explicit_decision_set = set(B23_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    conflict_state_episodes = 0
    monitor_lock_episodes = 0
    conflict_signal_episodes = 0
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
            str(item.get("b23_decision"))
            for item in trace
            if item.get("b23_decision") is not None
        ]
        states = [
            str(item.get("b23_conflict_state"))
            for item in trace
            if item.get("b23_conflict_state") is not None
        ]
        locks = [
            int(item.get("b23_monitor_lock", 0) or 0)
            for item in trace
            if item.get("b23_monitor_lock") is not None
        ]
        signals = [
            float(item.get("b23_prediction_error", 0.0) or 0.0)
            + float(item.get("b23_conflict_memory", 0.0) or 0.0)
            + float(item.get("b23_stability_vote", 0.0) or 0.0)
            + float(item.get("b23_abort_bias", 0.0) or 0.0)
            for item in trace
            if item.get("b23_prediction_error") is not None
            or item.get("b23_conflict_memory") is not None
            or item.get("b23_stability_vote") is not None
            or item.get("b23_abort_bias") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        conflict_state = any(state != "non_corridor" for state in states)
        monitor_lock = explicit_decision and any(lock > 0 for lock in locks)
        conflict_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if conflict_state:
            conflict_state_episodes += 1
        if monitor_lock:
            monitor_lock_episodes += 1
        if conflict_signal:
            conflict_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b23_decision": bool(explicit_decision),
                    "conflict_state": bool(conflict_state),
                    "monitor_lock": bool(monitor_lock),
                    "conflict_signal": bool(conflict_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "conflict_states": states,
                "monitor_locks": locks,
                "conflict_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b23_decision_episodes": explicit_decision_episodes >= 2,
        "conflict_state_episodes": conflict_state_episodes >= 2,
        "monitor_lock_episodes": monitor_lock_episodes >= 2,
        "conflict_signal_episodes": conflict_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b23_aggregate:" + name
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
            "base_b22_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "conflict_state_episodes": int(conflict_state_episodes),
            "monitor_lock_episodes": int(monitor_lock_episodes),
            "conflict_signal_episodes": int(conflict_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b24_precision_conflict_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b23_conflict_monitor_corridor_gate_result(results)
    explicit_decision_set = set(B24_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    precision_state_episodes = 0
    precision_lock_episodes = 0
    precision_signal_episodes = 0
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
            str(item.get("b24_decision"))
            for item in trace
            if item.get("b24_decision") is not None
        ]
        states = [
            str(item.get("b24_precision_state"))
            for item in trace
            if item.get("b24_precision_state") is not None
        ]
        locks = [
            int(item.get("b24_precision_lock", 0) or 0)
            for item in trace
            if item.get("b24_precision_lock") is not None
        ]
        signals = [
            float(item.get("b24_precision_memory", 0.0) or 0.0)
            + float(item.get("b24_precision_vote", 0.0) or 0.0)
            + float(item.get("b24_uncertainty_pressure", 0.0) or 0.0)
            + float(item.get("b24_abort_precision", 0.0) or 0.0)
            for item in trace
            if item.get("b24_precision_memory") is not None
            or item.get("b24_precision_vote") is not None
            or item.get("b24_uncertainty_pressure") is not None
            or item.get("b24_abort_precision") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        precision_state = any(state != "non_corridor" for state in states)
        precision_lock = explicit_decision and any(lock > 0 for lock in locks)
        precision_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if precision_state:
            precision_state_episodes += 1
        if precision_lock:
            precision_lock_episodes += 1
        if precision_signal:
            precision_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b24_decision": bool(explicit_decision),
                    "precision_state": bool(precision_state),
                    "precision_lock": bool(precision_lock),
                    "precision_signal": bool(precision_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "precision_states": states,
                "precision_locks": locks,
                "precision_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b24_decision_episodes": explicit_decision_episodes >= 2,
        "precision_state_episodes": precision_state_episodes >= 2,
        "precision_lock_episodes": precision_lock_episodes >= 2,
        "precision_signal_episodes": precision_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b24_aggregate:" + name
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
            "base_b23_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "precision_state_episodes": int(precision_state_episodes),
            "precision_lock_episodes": int(precision_lock_episodes),
            "precision_signal_episodes": int(precision_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b25_metacognitive_confidence_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b24_precision_conflict_corridor_gate_result(results)
    explicit_decision_set = set(B25_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    metacognitive_state_episodes = 0
    meta_lock_episodes = 0
    metacognitive_signal_episodes = 0
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
            str(item.get("b25_decision"))
            for item in trace
            if item.get("b25_decision") is not None
        ]
        states = [
            str(item.get("b25_metacognitive_state"))
            for item in trace
            if item.get("b25_metacognitive_state") is not None
        ]
        locks = [
            int(item.get("b25_meta_lock", 0) or 0)
            for item in trace
            if item.get("b25_meta_lock") is not None
        ]
        signals = [
            float(item.get("b25_confidence_memory", 0.0) or 0.0)
            + float(item.get("b25_confidence_vote", 0.0) or 0.0)
            + float(item.get("b25_doubt_pressure", 0.0) or 0.0)
            + float(item.get("b25_control_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b25_confidence_memory") is not None
            or item.get("b25_confidence_vote") is not None
            or item.get("b25_doubt_pressure") is not None
            or item.get("b25_control_gain") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        metacognitive_state = any(state != "non_corridor" for state in states)
        meta_lock = explicit_decision and any(lock > 0 for lock in locks)
        metacognitive_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if metacognitive_state:
            metacognitive_state_episodes += 1
        if meta_lock:
            meta_lock_episodes += 1
        if metacognitive_signal:
            metacognitive_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b25_decision": bool(explicit_decision),
                    "metacognitive_state": bool(metacognitive_state),
                    "meta_lock": bool(meta_lock),
                    "metacognitive_signal": bool(metacognitive_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "metacognitive_states": states,
                "meta_locks": locks,
                "metacognitive_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b25_decision_episodes": explicit_decision_episodes >= 2,
        "metacognitive_state_episodes": metacognitive_state_episodes >= 2,
        "meta_lock_episodes": meta_lock_episodes >= 2,
        "metacognitive_signal_episodes": metacognitive_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b25_aggregate:" + name
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
            "base_b24_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "metacognitive_state_episodes": int(metacognitive_state_episodes),
            "meta_lock_episodes": int(meta_lock_episodes),
            "metacognitive_signal_episodes": int(metacognitive_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b26_allostatic_prediction_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b25_metacognitive_confidence_corridor_gate_result(results)
    explicit_decision_set = set(B26_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    allostatic_state_episodes = 0
    stability_lock_episodes = 0
    allostatic_signal_episodes = 0
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
            str(item.get("b26_decision"))
            for item in trace
            if item.get("b26_decision") is not None
        ]
        states = [
            str(item.get("b26_allostatic_state"))
            for item in trace
            if item.get("b26_allostatic_state") is not None
        ]
        locks = [
            int(item.get("b26_stability_lock", 0) or 0)
            for item in trace
            if item.get("b26_stability_lock") is not None
        ]
        signals = [
            float(item.get("b26_prediction_error", 0.0) or 0.0)
            + float(item.get("b26_setpoint_pressure", 0.0) or 0.0)
            + float(item.get("b26_control_vote", 0.0) or 0.0)
            for item in trace
            if item.get("b26_prediction_error") is not None
            or item.get("b26_setpoint_pressure") is not None
            or item.get("b26_control_vote") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        allostatic_state = any(state != "non_corridor" for state in states)
        stability_lock = explicit_decision and any(lock > 0 for lock in locks)
        allostatic_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if allostatic_state:
            allostatic_state_episodes += 1
        if stability_lock:
            stability_lock_episodes += 1
        if allostatic_signal:
            allostatic_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b26_decision": bool(explicit_decision),
                    "allostatic_state": bool(allostatic_state),
                    "stability_lock": bool(stability_lock),
                    "allostatic_signal": bool(allostatic_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "allostatic_states": states,
                "stability_locks": locks,
                "allostatic_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b26_decision_episodes": explicit_decision_episodes >= 2,
        "allostatic_state_episodes": allostatic_state_episodes >= 2,
        "stability_lock_episodes": stability_lock_episodes >= 2,
        "allostatic_signal_episodes": allostatic_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b26_aggregate:" + name
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
            "base_b25_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "allostatic_state_episodes": int(allostatic_state_episodes),
            "stability_lock_episodes": int(stability_lock_episodes),
            "allostatic_signal_episodes": int(allostatic_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b27_arousal_gain_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b26_allostatic_prediction_corridor_gate_result(results)
    explicit_decision_set = set(B27_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    arousal_state_episodes = 0
    arousal_lock_episodes = 0
    arousal_signal_episodes = 0
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
            str(item.get("b27_decision"))
            for item in trace
            if item.get("b27_decision") is not None
        ]
        states = [
            str(item.get("b27_arousal_state"))
            for item in trace
            if item.get("b27_arousal_state") is not None
        ]
        locks = [
            int(item.get("b27_arousal_lock", 0) or 0)
            for item in trace
            if item.get("b27_arousal_lock") is not None
        ]
        signals = [
            float(item.get("b27_arousal_level", 0.0) or 0.0)
            + float(item.get("b27_gain_modulation", 0.0) or 0.0)
            + float(item.get("b27_stress_pressure", 0.0) or 0.0)
            for item in trace
            if item.get("b27_arousal_level") is not None
            or item.get("b27_gain_modulation") is not None
            or item.get("b27_stress_pressure") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        arousal_state = any(state != "non_corridor" for state in states)
        arousal_lock = explicit_decision and any(lock > 0 for lock in locks)
        arousal_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if arousal_state:
            arousal_state_episodes += 1
        if arousal_lock:
            arousal_lock_episodes += 1
        if arousal_signal:
            arousal_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b27_decision": bool(explicit_decision),
                    "arousal_state": bool(arousal_state),
                    "arousal_lock": bool(arousal_lock),
                    "arousal_signal": bool(arousal_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "arousal_states": states,
                "arousal_locks": locks,
                "arousal_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b27_decision_episodes": explicit_decision_episodes >= 2,
        "arousal_state_episodes": arousal_state_episodes >= 2,
        "arousal_lock_episodes": arousal_lock_episodes >= 2,
        "arousal_signal_episodes": arousal_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b27_aggregate:" + name
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
            "base_b26_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "arousal_state_episodes": int(arousal_state_episodes),
            "arousal_lock_episodes": int(arousal_lock_episodes),
            "arousal_signal_episodes": int(arousal_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b28_interoceptive_attention_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b27_arousal_gain_corridor_gate_result(results)
    explicit_decision_set = set(B28_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    attention_state_episodes = 0
    attention_lock_episodes = 0
    attention_signal_episodes = 0
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
            str(item.get("b28_decision"))
            for item in trace
            if item.get("b28_decision") is not None
        ]
        states = [
            str(item.get("b28_attention_state"))
            for item in trace
            if item.get("b28_attention_state") is not None
        ]
        locks = [
            int(item.get("b28_attention_lock", 0) or 0)
            for item in trace
            if item.get("b28_attention_lock") is not None
        ]
        signals = [
            float(item.get("b28_interoceptive_focus", 0.0) or 0.0)
            + float(item.get("b28_attention_gain", 0.0) or 0.0)
            + float(item.get("b28_distractor_pressure", 0.0) or 0.0)
            for item in trace
            if item.get("b28_interoceptive_focus") is not None
            or item.get("b28_attention_gain") is not None
            or item.get("b28_distractor_pressure") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        attention_state = any(state != "non_corridor" for state in states)
        attention_lock = explicit_decision and any(lock > 0 for lock in locks)
        attention_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if attention_state:
            attention_state_episodes += 1
        if attention_lock:
            attention_lock_episodes += 1
        if attention_signal:
            attention_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b28_decision": bool(explicit_decision),
                    "attention_state": bool(attention_state),
                    "attention_lock": bool(attention_lock),
                    "attention_signal": bool(attention_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "attention_states": states,
                "attention_locks": locks,
                "attention_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b28_decision_episodes": explicit_decision_episodes >= 2,
        "attention_state_episodes": attention_state_episodes >= 2,
        "attention_lock_episodes": attention_lock_episodes >= 2,
        "attention_signal_episodes": attention_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b28_aggregate:" + name
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
            "base_b27_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "attention_state_episodes": int(attention_state_episodes),
            "attention_lock_episodes": int(attention_lock_episodes),
            "attention_signal_episodes": int(attention_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b29_salience_competition_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b28_interoceptive_attention_corridor_gate_result(results)
    explicit_decision_set = set(B29_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    salience_state_episodes = 0
    salience_lock_episodes = 0
    salience_signal_episodes = 0
    winner_channel_episodes = 0
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
            str(item.get("b29_decision"))
            for item in trace
            if item.get("b29_decision") is not None
        ]
        states = [
            str(item.get("b29_salience_state"))
            for item in trace
            if item.get("b29_salience_state") is not None
        ]
        locks = [
            int(item.get("b29_salience_lock", 0) or 0)
            for item in trace
            if item.get("b29_salience_lock") is not None
        ]
        winners = [
            str(item.get("b29_winner_channel"))
            for item in trace
            if item.get("b29_winner_channel") is not None
        ]
        signals = [
            float(item.get("b29_threat_salience", 0.0) or 0.0)
            + float(item.get("b29_homeostatic_salience", 0.0) or 0.0)
            + float(item.get("b29_corridor_salience", 0.0) or 0.0)
            for item in trace
            if item.get("b29_threat_salience") is not None
            or item.get("b29_homeostatic_salience") is not None
            or item.get("b29_corridor_salience") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        salience_state = any(state != "non_corridor" for state in states)
        salience_lock = explicit_decision and any(lock > 0 for lock in locks)
        salience_signal = any(signal > 0.0 for signal in signals)
        winner_channel = any(winner in {"corridor", "homeostasis", "threat"} for winner in winners)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if salience_state:
            salience_state_episodes += 1
        if salience_lock:
            salience_lock_episodes += 1
        if salience_signal:
            salience_signal_episodes += 1
        if winner_channel:
            winner_channel_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b29_decision": bool(explicit_decision),
                    "salience_state": bool(salience_state),
                    "salience_lock": bool(salience_lock),
                    "salience_signal": bool(salience_signal),
                    "winner_channel": bool(winner_channel),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "salience_states": states,
                "salience_locks": locks,
                "winner_channels": winners,
                "salience_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b29_decision_episodes": explicit_decision_episodes >= 2,
        "salience_state_episodes": salience_state_episodes >= 2,
        "salience_lock_episodes": salience_lock_episodes >= 2,
        "salience_signal_episodes": salience_signal_episodes >= 2,
        "winner_channel_episodes": winner_channel_episodes >= 2,
    }
    failures.extend(
        "corridor_b29_aggregate:" + name
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
            "base_b28_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "salience_state_episodes": int(salience_state_episodes),
            "salience_lock_episodes": int(salience_lock_episodes),
            "salience_signal_episodes": int(salience_signal_episodes),
            "winner_channel_episodes": int(winner_channel_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b30_basal_ganglia_gate_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b29_salience_competition_corridor_gate_result(results)
    explicit_decision_set = set(B30_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    gate_state_episodes = 0
    gate_lock_episodes = 0
    gate_signal_episodes = 0
    action_gate_episodes = 0
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
            str(item.get("b30_decision"))
            for item in trace
            if item.get("b30_decision") is not None
        ]
        states = [
            str(item.get("b30_gate_state"))
            for item in trace
            if item.get("b30_gate_state") is not None
        ]
        locks = [
            int(item.get("b30_gate_lock", 0) or 0)
            for item in trace
            if item.get("b30_gate_lock") is not None
        ]
        gates = [
            str(item.get("b30_action_gate"))
            for item in trace
            if item.get("b30_action_gate") is not None
        ]
        signals = [
            float(item.get("b30_go_signal", 0.0) or 0.0)
            + float(item.get("b30_no_go_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b30_go_signal") is not None
            or item.get("b30_no_go_signal") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        gate_state = any(state != "non_corridor" for state in states)
        gate_lock = explicit_decision and any(lock > 0 for lock in locks)
        gate_signal = any(signal > 0.0 for signal in signals)
        action_gate = any(gate in {"go", "no_go"} for gate in gates)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if gate_state:
            gate_state_episodes += 1
        if gate_lock:
            gate_lock_episodes += 1
        if gate_signal:
            gate_signal_episodes += 1
        if action_gate:
            action_gate_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b30_decision": bool(explicit_decision),
                    "gate_state": bool(gate_state),
                    "gate_lock": bool(gate_lock),
                    "gate_signal": bool(gate_signal),
                    "action_gate": bool(action_gate),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "gate_states": states,
                "gate_locks": locks,
                "action_gates": gates,
                "gate_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b30_decision_episodes": explicit_decision_episodes >= 2,
        "gate_state_episodes": gate_state_episodes >= 2,
        "gate_lock_episodes": gate_lock_episodes >= 2,
        "gate_signal_episodes": gate_signal_episodes >= 2,
        "action_gate_episodes": action_gate_episodes >= 2,
    }
    failures.extend(
        "corridor_b30_aggregate:" + name
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
            "base_b29_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "gate_state_episodes": int(gate_state_episodes),
            "gate_lock_episodes": int(gate_lock_episodes),
            "gate_signal_episodes": int(gate_signal_episodes),
            "action_gate_episodes": int(action_gate_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
