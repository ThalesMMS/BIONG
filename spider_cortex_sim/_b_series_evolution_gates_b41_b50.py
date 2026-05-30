from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_gates_b1_b6 import (
    trace_uses_only_primitive_actions,
)

from ._b_series_evolution_gates_b31_b40 import (
    b40_global_workspace_corridor_gate_result,
)

def b41_executive_workspace_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b40_global_workspace_corridor_gate_result(results)
    explicit_decision_set = set(B41_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    selection_episodes = 0
    inhibition_episodes = 0
    goal_context_episodes = 0
    stability_episodes = 0
    lock_episodes = 0
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
            str(item.get("b41_decision"))
            for item in trace
            if item.get("b41_decision") is not None
        ]
        selections = [
            float(item.get("b41_executive_selection", 0.0) or 0.0)
            for item in trace
            if item.get("b41_executive_selection") is not None
        ]
        inhibitions = [
            float(item.get("b41_inhibitory_pressure", 0.0) or 0.0)
            for item in trace
            if item.get("b41_inhibitory_pressure") is not None
        ]
        contexts = [
            float(item.get("b41_goal_context", 0.0) or 0.0)
            for item in trace
            if item.get("b41_goal_context") is not None
        ]
        stabilities = [
            float(item.get("b41_executive_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b41_executive_stability") is not None
        ]
        locks = [
            int(item.get("b41_executive_lock", 0) or 0)
            for item in trace
            if item.get("b41_executive_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        selection = any(abs(value) > 0.0 for value in selections)
        inhibition = any(value >= 0.0 for value in inhibitions) and len(inhibitions) > 0
        goal_context = any(abs(value) > 0.0 for value in contexts)
        stability = any(abs(value) > 0.0 for value in stabilities)
        executive_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if selection:
            selection_episodes += 1
        if inhibition:
            inhibition_episodes += 1
        if goal_context:
            goal_context_episodes += 1
        if stability:
            stability_episodes += 1
        if executive_lock:
            lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b41_decision": bool(explicit_decision),
                    "executive_selection": bool(selection),
                    "inhibitory_pressure": bool(inhibition),
                    "goal_context": bool(goal_context),
                    "executive_stability": bool(stability),
                    "executive_lock": bool(executive_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "executive_selections": selections,
                "inhibitory_pressures": inhibitions,
                "goal_contexts": contexts,
                "executive_stabilities": stabilities,
                "executive_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b41_decision_episodes": explicit_decision_episodes >= 2,
        "executive_selection_episodes": selection_episodes >= 2,
        "inhibitory_pressure_episodes": inhibition_episodes >= 2,
        "goal_context_episodes": goal_context_episodes >= 2,
        "executive_stability_episodes": stability_episodes >= 2,
        "executive_lock_episodes": lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b41_aggregate:" + name
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
            "base_b40_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "executive_selection_episodes": int(selection_episodes),
            "inhibitory_pressure_episodes": int(inhibition_episodes),
            "goal_context_episodes": int(goal_context_episodes),
            "executive_stability_episodes": int(stability_episodes),
            "executive_lock_episodes": int(lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b42_error_monitor_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b41_executive_workspace_corridor_gate_result(results)
    explicit_decision_set = set(B42_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    error_episodes = 0
    conflict_episodes = 0
    performance_episodes = 0
    stability_episodes = 0
    lock_episodes = 0
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
            str(item.get("b42_decision"))
            for item in trace
            if item.get("b42_decision") is not None
        ]
        errors = [
            float(item.get("b42_error_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b42_error_signal") is not None
        ]
        conflicts = [
            float(item.get("b42_conflict_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b42_conflict_signal") is not None
        ]
        contexts = [
            float(item.get("b42_performance_context", 0.0) or 0.0)
            for item in trace
            if item.get("b42_performance_context") is not None
        ]
        stabilities = [
            float(item.get("b42_monitor_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b42_monitor_stability") is not None
        ]
        locks = [
            int(item.get("b42_monitor_lock", 0) or 0)
            for item in trace
            if item.get("b42_monitor_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        error_signal = any(value >= 0.0 for value in errors) and len(errors) > 0
        conflict_signal = any(value >= 0.0 for value in conflicts) and len(conflicts) > 0
        performance_context = any(abs(value) > 0.0 for value in contexts)
        monitor_stability = any(abs(value) > 0.0 for value in stabilities)
        monitor_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if error_signal:
            error_episodes += 1
        if conflict_signal:
            conflict_episodes += 1
        if performance_context:
            performance_episodes += 1
        if monitor_stability:
            stability_episodes += 1
        if monitor_lock:
            lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b42_decision": bool(explicit_decision),
                    "error_signal": bool(error_signal),
                    "conflict_signal": bool(conflict_signal),
                    "performance_context": bool(performance_context),
                    "monitor_stability": bool(monitor_stability),
                    "monitor_lock": bool(monitor_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "error_signals": errors,
                "conflict_signals": conflicts,
                "performance_contexts": contexts,
                "monitor_stabilities": stabilities,
                "monitor_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b42_decision_episodes": explicit_decision_episodes >= 2,
        "error_signal_episodes": error_episodes >= 2,
        "conflict_signal_episodes": conflict_episodes >= 2,
        "performance_context_episodes": performance_episodes >= 2,
        "monitor_stability_episodes": stability_episodes >= 2,
        "monitor_lock_episodes": lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b42_aggregate:" + name
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
            "base_b41_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "error_signal_episodes": int(error_episodes),
            "conflict_signal_episodes": int(conflict_episodes),
            "performance_context_episodes": int(performance_episodes),
            "monitor_stability_episodes": int(stability_episodes),
            "monitor_lock_episodes": int(lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b43_adaptive_precision_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b42_error_monitor_corridor_gate_result(results)
    explicit_decision_set = set(B43_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    precision_episodes = 0
    threshold_episodes = 0
    arousal_episodes = 0
    stability_episodes = 0
    lock_episodes = 0
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
            str(item.get("b43_decision"))
            for item in trace
            if item.get("b43_decision") is not None
        ]
        precisions = [
            float(item.get("b43_precision_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b43_precision_signal") is not None
        ]
        thresholds = [
            float(item.get("b43_adaptive_threshold", 0.0) or 0.0)
            for item in trace
            if item.get("b43_adaptive_threshold") is not None
        ]
        arousals = [
            float(item.get("b43_arousal_context", 0.0) or 0.0)
            for item in trace
            if item.get("b43_arousal_context") is not None
        ]
        stabilities = [
            float(item.get("b43_control_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b43_control_stability") is not None
        ]
        locks = [
            int(item.get("b43_precision_lock", 0) or 0)
            for item in trace
            if item.get("b43_precision_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        precision_signal = any(abs(value) > 0.0 for value in precisions)
        adaptive_threshold = len(thresholds) > 0
        arousal_context = len(arousals) > 0
        control_stability = any(abs(value) > 0.0 for value in stabilities)
        precision_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if precision_signal:
            precision_episodes += 1
        if adaptive_threshold:
            threshold_episodes += 1
        if arousal_context:
            arousal_episodes += 1
        if control_stability:
            stability_episodes += 1
        if precision_lock:
            lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b43_decision": bool(explicit_decision),
                    "precision_signal": bool(precision_signal),
                    "adaptive_threshold": bool(adaptive_threshold),
                    "arousal_context": bool(arousal_context),
                    "control_stability": bool(control_stability),
                    "precision_lock": bool(precision_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "precision_signals": precisions,
                "adaptive_thresholds": thresholds,
                "arousal_contexts": arousals,
                "control_stabilities": stabilities,
                "precision_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b43_decision_episodes": explicit_decision_episodes >= 2,
        "precision_signal_episodes": precision_episodes >= 2,
        "adaptive_threshold_episodes": threshold_episodes >= 2,
        "arousal_context_episodes": arousal_episodes >= 2,
        "control_stability_episodes": stability_episodes >= 2,
        "precision_lock_episodes": lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b43_aggregate:" + name
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
            "base_b42_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "precision_signal_episodes": int(precision_episodes),
            "adaptive_threshold_episodes": int(threshold_episodes),
            "arousal_context_episodes": int(arousal_episodes),
            "control_stability_episodes": int(stability_episodes),
            "precision_lock_episodes": int(lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b44_thalamic_relay_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b43_adaptive_precision_corridor_gate_result(results)
    explicit_decision_set = set(B44_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    relay_gate_episodes = 0
    sensory_episodes = 0
    context_episodes = 0
    stability_episodes = 0
    lock_episodes = 0
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
            str(item.get("b44_decision"))
            for item in trace
            if item.get("b44_decision") is not None
        ]
        relay_gates = [
            float(item.get("b44_relay_gate", 0.0) or 0.0)
            for item in trace
            if item.get("b44_relay_gate") is not None
        ]
        sensory_values = [
            float(item.get("b44_sensory_precision", 0.0) or 0.0)
            for item in trace
            if item.get("b44_sensory_precision") is not None
        ]
        context_values = [
            float(item.get("b44_context_relay", 0.0) or 0.0)
            for item in trace
            if item.get("b44_context_relay") is not None
        ]
        stabilities = [
            float(item.get("b44_gate_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b44_gate_stability") is not None
        ]
        locks = [
            int(item.get("b44_relay_lock", 0) or 0)
            for item in trace
            if item.get("b44_relay_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        relay_gate = any(abs(value) > 0.0 for value in relay_gates)
        sensory_precision = any(abs(value) > 0.0 for value in sensory_values)
        context_relay = any(abs(value) > 0.0 for value in context_values)
        gate_stability = any(abs(value) > 0.0 for value in stabilities)
        relay_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if relay_gate:
            relay_gate_episodes += 1
        if sensory_precision:
            sensory_episodes += 1
        if context_relay:
            context_episodes += 1
        if gate_stability:
            stability_episodes += 1
        if relay_lock:
            lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b44_decision": bool(explicit_decision),
                    "relay_gate": bool(relay_gate),
                    "sensory_precision": bool(sensory_precision),
                    "context_relay": bool(context_relay),
                    "gate_stability": bool(gate_stability),
                    "relay_lock": bool(relay_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "relay_gates": relay_gates,
                "sensory_precisions": sensory_values,
                "context_relays": context_values,
                "gate_stabilities": stabilities,
                "relay_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b44_decision_episodes": explicit_decision_episodes >= 2,
        "relay_gate_episodes": relay_gate_episodes >= 2,
        "sensory_precision_episodes": sensory_episodes >= 2,
        "context_relay_episodes": context_episodes >= 2,
        "gate_stability_episodes": stability_episodes >= 2,
        "relay_lock_episodes": lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b44_aggregate:" + name
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
            "base_b43_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "relay_gate_episodes": int(relay_gate_episodes),
            "sensory_precision_episodes": int(sensory_episodes),
            "context_relay_episodes": int(context_episodes),
            "gate_stability_episodes": int(stability_episodes),
            "relay_lock_episodes": int(lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b45_reticular_inhibition_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b44_thalamic_relay_corridor_gate_result(results)
    explicit_decision_set = set(B45_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    inhibitory_gate_episodes = 0
    sensory_filter_episodes = 0
    context_suppression_episodes = 0
    stability_episodes = 0
    lock_episodes = 0
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
            str(item.get("b45_decision"))
            for item in trace
            if item.get("b45_decision") is not None
        ]
        inhibitory_gates = [
            float(item.get("b45_inhibitory_gate", 0.0) or 0.0)
            for item in trace
            if item.get("b45_inhibitory_gate") is not None
        ]
        sensory_filters = [
            float(item.get("b45_sensory_filter", 0.0) or 0.0)
            for item in trace
            if item.get("b45_sensory_filter") is not None
        ]
        context_suppressions = [
            float(item.get("b45_context_suppression", 0.0) or 0.0)
            for item in trace
            if item.get("b45_context_suppression") is not None
        ]
        stabilities = [
            float(item.get("b45_loop_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b45_loop_stability") is not None
        ]
        locks = [
            int(item.get("b45_inhibition_lock", 0) or 0)
            for item in trace
            if item.get("b45_inhibition_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        inhibitory_gate = any(abs(value) > 0.0 for value in inhibitory_gates)
        sensory_filter = any(abs(value) > 0.0 for value in sensory_filters)
        context_suppression = any(abs(value) > 0.0 for value in context_suppressions)
        loop_stability = any(abs(value) > 0.0 for value in stabilities)
        inhibition_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if inhibitory_gate:
            inhibitory_gate_episodes += 1
        if sensory_filter:
            sensory_filter_episodes += 1
        if context_suppression:
            context_suppression_episodes += 1
        if loop_stability:
            stability_episodes += 1
        if inhibition_lock:
            lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b45_decision": bool(explicit_decision),
                    "inhibitory_gate": bool(inhibitory_gate),
                    "sensory_filter": bool(sensory_filter),
                    "context_suppression": bool(context_suppression),
                    "loop_stability": bool(loop_stability),
                    "inhibition_lock": bool(inhibition_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "inhibitory_gates": inhibitory_gates,
                "sensory_filters": sensory_filters,
                "context_suppressions": context_suppressions,
                "loop_stabilities": stabilities,
                "inhibition_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b45_decision_episodes": explicit_decision_episodes >= 2,
        "inhibitory_gate_episodes": inhibitory_gate_episodes >= 2,
        "sensory_filter_episodes": sensory_filter_episodes >= 2,
        "context_suppression_episodes": context_suppression_episodes >= 2,
        "loop_stability_episodes": stability_episodes >= 2,
        "inhibition_lock_episodes": lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b45_aggregate:" + name
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
            "base_b44_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "inhibitory_gate_episodes": int(inhibitory_gate_episodes),
            "sensory_filter_episodes": int(sensory_filter_episodes),
            "context_suppression_episodes": int(context_suppression_episodes),
            "loop_stability_episodes": int(stability_episodes),
            "inhibition_lock_episodes": int(lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b46_corticothalamic_feedback_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b45_reticular_inhibition_corridor_gate_result(results)
    explicit_decision_set = set(B46_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    feedback_gain_episodes = 0
    topdown_context_episodes = 0
    prediction_match_episodes = 0
    stability_episodes = 0
    lock_episodes = 0
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
            str(item.get("b46_decision"))
            for item in trace
            if item.get("b46_decision") is not None
        ]
        feedback_gains = [
            float(item.get("b46_feedback_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b46_feedback_gain") is not None
        ]
        topdown_contexts = [
            float(item.get("b46_topdown_context", 0.0) or 0.0)
            for item in trace
            if item.get("b46_topdown_context") is not None
        ]
        prediction_matches = [
            float(item.get("b46_prediction_match", 0.0) or 0.0)
            for item in trace
            if item.get("b46_prediction_match") is not None
        ]
        stabilities = [
            float(item.get("b46_feedback_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b46_feedback_stability") is not None
        ]
        locks = [
            int(item.get("b46_feedback_lock", 0) or 0)
            for item in trace
            if item.get("b46_feedback_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        feedback_gain = any(abs(value) > 0.0 for value in feedback_gains)
        topdown_context = any(abs(value) > 0.0 for value in topdown_contexts)
        prediction_match = any(abs(value) > 0.0 for value in prediction_matches)
        feedback_stability = any(abs(value) > 0.0 for value in stabilities)
        feedback_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if feedback_gain:
            feedback_gain_episodes += 1
        if topdown_context:
            topdown_context_episodes += 1
        if prediction_match:
            prediction_match_episodes += 1
        if feedback_stability:
            stability_episodes += 1
        if feedback_lock:
            lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b46_decision": bool(explicit_decision),
                    "feedback_gain": bool(feedback_gain),
                    "topdown_context": bool(topdown_context),
                    "prediction_match": bool(prediction_match),
                    "feedback_stability": bool(feedback_stability),
                    "feedback_lock": bool(feedback_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "feedback_gains": feedback_gains,
                "topdown_contexts": topdown_contexts,
                "prediction_matches": prediction_matches,
                "feedback_stabilities": stabilities,
                "feedback_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b46_decision_episodes": explicit_decision_episodes >= 2,
        "feedback_gain_episodes": feedback_gain_episodes >= 2,
        "topdown_context_episodes": topdown_context_episodes >= 2,
        "prediction_match_episodes": prediction_match_episodes >= 2,
        "feedback_stability_episodes": stability_episodes >= 2,
        "feedback_lock_episodes": lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b46_aggregate:" + name
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
            "base_b45_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "feedback_gain_episodes": int(feedback_gain_episodes),
            "topdown_context_episodes": int(topdown_context_episodes),
            "prediction_match_episodes": int(prediction_match_episodes),
            "feedback_stability_episodes": int(stability_episodes),
            "feedback_lock_episodes": int(lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b47_oscillatory_synchrony_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b46_corticothalamic_feedback_corridor_gate_result(results)
    explicit_decision_set = set(B47_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    phase_alignment_episodes = 0
    synchrony_gain_episodes = 0
    coherence_episodes = 0
    phase_lock_episodes = 0
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
            str(item.get("b47_decision"))
            for item in trace
            if item.get("b47_decision") is not None
        ]
        phase_alignments = [
            float(item.get("b47_phase_alignment", 0.0) or 0.0)
            for item in trace
            if item.get("b47_phase_alignment") is not None
        ]
        synchrony_gains = [
            float(item.get("b47_synchrony_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b47_synchrony_gain") is not None
        ]
        coherences = [
            float(item.get("b47_cross_loop_coherence", 0.0) or 0.0)
            for item in trace
            if item.get("b47_cross_loop_coherence") is not None
        ]
        locks = [
            int(item.get("b47_phase_lock", 0) or 0)
            for item in trace
            if item.get("b47_phase_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        phase_alignment = any(abs(value) > 0.0 for value in phase_alignments)
        synchrony_gain = any(abs(value) > 0.0 for value in synchrony_gains)
        coherence = any(abs(value) > 0.0 for value in coherences)
        phase_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if phase_alignment:
            phase_alignment_episodes += 1
        if synchrony_gain:
            synchrony_gain_episodes += 1
        if coherence:
            coherence_episodes += 1
        if phase_lock:
            phase_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b47_decision": bool(explicit_decision),
                    "phase_alignment": bool(phase_alignment),
                    "synchrony_gain": bool(synchrony_gain),
                    "cross_loop_coherence": bool(coherence),
                    "phase_lock": bool(phase_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "phase_alignments": phase_alignments,
                "synchrony_gains": synchrony_gains,
                "cross_loop_coherences": coherences,
                "phase_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b47_decision_episodes": explicit_decision_episodes >= 2,
        "phase_alignment_episodes": phase_alignment_episodes >= 2,
        "synchrony_gain_episodes": synchrony_gain_episodes >= 2,
        "cross_loop_coherence_episodes": coherence_episodes >= 2,
        "phase_lock_episodes": phase_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b47_aggregate:" + name
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
            "base_b46_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "phase_alignment_episodes": int(phase_alignment_episodes),
            "synchrony_gain_episodes": int(synchrony_gain_episodes),
            "cross_loop_coherence_episodes": int(coherence_episodes),
            "phase_lock_episodes": int(phase_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b48_cerebellar_timing_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b47_oscillatory_synchrony_corridor_gate_result(results)
    explicit_decision_set = set(B48_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    timing_error_episodes = 0
    predictive_timing_episodes = 0
    corrective_gain_episodes = 0
    calibration_lock_episodes = 0
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
            str(item.get("b48_decision"))
            for item in trace
            if item.get("b48_decision") is not None
        ]
        timing_errors = [
            float(item.get("b48_timing_error", 0.0) or 0.0)
            for item in trace
            if item.get("b48_timing_error") is not None
        ]
        predictive_timings = [
            float(item.get("b48_predictive_timing", 0.0) or 0.0)
            for item in trace
            if item.get("b48_predictive_timing") is not None
        ]
        corrective_gains = [
            float(item.get("b48_corrective_gain", 0.0) or 0.0)
            for item in trace
            if item.get("b48_corrective_gain") is not None
        ]
        locks = [
            int(item.get("b48_calibration_lock", 0) or 0)
            for item in trace
            if item.get("b48_calibration_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        timing_error = any(abs(value) > 0.0 for value in timing_errors)
        predictive_timing = any(abs(value) > 0.0 for value in predictive_timings)
        corrective_gain = any(abs(value) > 0.0 for value in corrective_gains)
        calibration_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if timing_error:
            timing_error_episodes += 1
        if predictive_timing:
            predictive_timing_episodes += 1
        if corrective_gain:
            corrective_gain_episodes += 1
        if calibration_lock:
            calibration_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b48_decision": bool(explicit_decision),
                    "timing_error": bool(timing_error),
                    "predictive_timing": bool(predictive_timing),
                    "corrective_gain": bool(corrective_gain),
                    "calibration_lock": bool(calibration_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "timing_errors": timing_errors,
                "predictive_timings": predictive_timings,
                "corrective_gains": corrective_gains,
                "calibration_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b48_decision_episodes": explicit_decision_episodes >= 2,
        "timing_error_episodes": timing_error_episodes >= 2,
        "predictive_timing_episodes": predictive_timing_episodes >= 2,
        "corrective_gain_episodes": corrective_gain_episodes >= 2,
        "calibration_lock_episodes": calibration_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b48_aggregate:" + name
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
            "base_b47_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "timing_error_episodes": int(timing_error_episodes),
            "predictive_timing_episodes": int(predictive_timing_episodes),
            "corrective_gain_episodes": int(corrective_gain_episodes),
            "calibration_lock_episodes": int(calibration_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b49_striatal_action_gate_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b48_cerebellar_timing_corridor_gate_result(results)
    explicit_decision_set = set(B49_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    go_signal_episodes = 0
    no_go_signal_episodes = 0
    balance_episodes = 0
    selection_lock_episodes = 0
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
            str(item.get("b49_decision"))
            for item in trace
            if item.get("b49_decision") is not None
        ]
        go_signals = [
            float(item.get("b49_go_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b49_go_signal") is not None
        ]
        no_go_signals = [
            float(item.get("b49_no_go_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b49_no_go_signal") is not None
        ]
        balances = [
            float(item.get("b49_action_gate_balance", 0.0) or 0.0)
            for item in trace
            if item.get("b49_action_gate_balance") is not None
        ]
        locks = [
            int(item.get("b49_selection_lock", 0) or 0)
            for item in trace
            if item.get("b49_selection_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        go_signal = any(abs(value) > 0.0 for value in go_signals)
        no_go_signal = any(abs(value) > 0.0 for value in no_go_signals)
        balance = any(abs(value) > 0.0 for value in balances)
        selection_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if go_signal:
            go_signal_episodes += 1
        if no_go_signal:
            no_go_signal_episodes += 1
        if balance:
            balance_episodes += 1
        if selection_lock:
            selection_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b49_decision": bool(explicit_decision),
                    "go_signal": bool(go_signal),
                    "no_go_signal": bool(no_go_signal),
                    "action_gate_balance": bool(balance),
                    "selection_lock": bool(selection_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "go_signals": go_signals,
                "no_go_signals": no_go_signals,
                "action_gate_balances": balances,
                "selection_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b48_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b49_decision_episodes": explicit_decision_episodes >= 2,
        "go_signal_episodes": go_signal_episodes >= 2,
        "no_go_signal_episodes": no_go_signal_episodes >= 2,
        "action_gate_balance_episodes": balance_episodes >= 2,
        "selection_lock_episodes": selection_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b49_aggregate:" + name
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
            "base_b48_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "go_signal_episodes": int(go_signal_episodes),
            "no_go_signal_episodes": int(no_go_signal_episodes),
            "action_gate_balance_episodes": int(balance_episodes),
            "selection_lock_episodes": int(selection_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b50_habit_chunking_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b49_striatal_action_gate_corridor_gate_result(results)
    explicit_decision_set = set(B50_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    habit_strength_episodes = 0
    chunk_value_episodes = 0
    habit_stability_episodes = 0
    chunk_lock_episodes = 0
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
            str(item.get("b50_decision"))
            for item in trace
            if item.get("b50_decision") is not None
        ]
        habit_strengths = [
            float(item.get("b50_habit_strength", 0.0) or 0.0)
            for item in trace
            if item.get("b50_habit_strength") is not None
        ]
        chunk_values = [
            float(item.get("b50_chunk_value", 0.0) or 0.0)
            for item in trace
            if item.get("b50_chunk_value") is not None
        ]
        habit_stabilities = [
            float(item.get("b50_habit_stability", 0.0) or 0.0)
            for item in trace
            if item.get("b50_habit_stability") is not None
        ]
        locks = [
            int(item.get("b50_chunk_lock", 0) or 0)
            for item in trace
            if item.get("b50_chunk_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        habit_strength = any(abs(value) > 0.0 for value in habit_strengths)
        chunk_value = any(abs(value) > 0.0 for value in chunk_values)
        habit_stability = any(abs(value) > 0.0 for value in habit_stabilities)
        chunk_lock = explicit_decision and any(lock > 0 for lock in locks)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if habit_strength:
            habit_strength_episodes += 1
        if chunk_value:
            chunk_value_episodes += 1
        if habit_stability:
            habit_stability_episodes += 1
        if chunk_lock:
            chunk_lock_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b50_decision": bool(explicit_decision),
                    "habit_strength": bool(habit_strength),
                    "chunk_value": bool(chunk_value),
                    "habit_stability": bool(habit_stability),
                    "chunk_lock": bool(chunk_lock),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "habit_strengths": habit_strengths,
                "chunk_values": chunk_values,
                "habit_stabilities": habit_stabilities,
                "chunk_locks": locks,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "base_b49_corridor_diagnostic": bool(base_gate["passed"]),
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b50_decision_episodes": explicit_decision_episodes >= 2,
        "habit_strength_episodes": habit_strength_episodes >= 2,
        "chunk_value_episodes": chunk_value_episodes >= 2,
        "habit_stability_episodes": habit_stability_episodes >= 2,
        "chunk_lock_episodes": chunk_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b50_aggregate:" + name
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
            "base_b49_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "habit_strength_episodes": int(habit_strength_episodes),
            "chunk_value_episodes": int(chunk_value_episodes),
            "habit_stability_episodes": int(habit_stability_episodes),
            "chunk_lock_episodes": int(chunk_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
