from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

from ._b_series_evolution_gates_b1_b6 import (
    _behavior_score_payload,
    _stats_payload,
    trace_uses_only_primitive_actions,
)

def b7_corridor_viability_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    scenario = get_scenario(B6_CORRIDOR_SCENARIO)
    explicit_decision_set = set(B7_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    food_progress_episodes = 0
    explicit_decision_episodes = 0
    improvement_episodes = 0
    full_success_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        stats = result["stats"]
        trace = result["trace"]
        score = scenario.score_episode(stats, trace)
        score_payload = _behavior_score_payload(score)
        primitive_trace_ok, primitive_violations = trace_uses_only_primitive_actions(
            trace
        )
        decisions = [
            str(item.get("b7_decision"))
            for item in trace
            if item.get("b7_decision") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        abort_or_recover = any(
            decision in {"abort_return_unviable", "recover_before_crossing"}
            for decision in decisions
        )
        food_distance_delta = float(getattr(stats, "food_distance_delta", 0.0) or 0.0)
        food_progress = food_distance_delta >= B7_CORRIDOR_REQUIRED_FOOD_DISTANCE_DELTA
        improvement = (
            food_distance_delta > B7_CORRIDOR_REQUIRED_FOOD_DISTANCE_DELTA
            or int(stats.steps) > B6_CORRIDOR_BASELINE_STEPS
            or float(getattr(stats, "final_health", 0.0) or 0.0) > 0.0
            or bool(stats.alive)
            or int(getattr(stats, "food_eaten", 0) or 0) > 0
            or abort_or_recover
        )
        if food_progress:
            food_progress_episodes += 1
        if explicit_decision:
            explicit_decision_episodes += 1
        if improvement:
            improvement_episodes += 1
        if bool(score_payload["success"]):
            full_success_episodes += 1
        checks = {
            "primitive_trace": primitive_trace_ok,
            "predator_contacts": int(stats.predator_contacts) == 0,
        }
        episode_failures = [name for name, ok in checks.items() if not ok]
        if episode_failures:
            failures.append(
                "corridor_ep" + str(episode) + ":" + ",".join(episode_failures)
            )
        episode_results.append(
            {
                "evaluation_episode": episode,
                "gate": {
                    "scenario": B6_CORRIDOR_SCENARIO,
                    "status": "accepted" if not episode_failures else "discarded",
                    "passed": not episode_failures,
                    "checks": {
                        **checks,
                        "food_distance_delta_floor": food_progress,
                        "explicit_b7_decision": explicit_decision,
                        "improvement_over_b6_axis": improvement,
                    },
                    "decisions": decisions,
                    "failures": episode_failures,
                    "primitive_violations": primitive_violations,
                },
                "metrics": {
                    **_stats_payload(stats),
                    "food_distance_delta": round(food_distance_delta, 6),
                    "explicit_b7_decision": bool(explicit_decision),
                    "abort_or_recover_recorded": bool(abort_or_recover),
                },
                "behavior_score": score_payload,
            }
        )
    aggregate_checks = {
        "food_progress_episodes": food_progress_episodes >= 2,
        "explicit_decision_episodes": explicit_decision_episodes >= 2,
        "improvement_episodes": improvement_episodes >= 2,
    }
    if full_success_episodes >= 2:
        aggregate_checks = {
            "score_success_episodes": True,
            **aggregate_checks,
        }
        aggregate_checks["food_progress_episodes"] = True
        aggregate_checks["explicit_decision_episodes"] = True
        aggregate_checks["improvement_episodes"] = True
    failures.extend(
        "corridor_aggregate:" + name
        for name, ok in aggregate_checks.items()
        if not ok
    )
    passed = not failures
    return {
        "scenario": B6_CORRIDOR_SCENARIO,
        "status": "accepted" if passed else "discarded",
        "passed": passed,
        "baseline": {
            "b6_steps": B6_CORRIDOR_BASELINE_STEPS,
            "b6_food_distance_delta": B7_CORRIDOR_REQUIRED_FOOD_DISTANCE_DELTA,
        },
        "aggregate": {
            "food_progress_episodes": int(food_progress_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "improvement_episodes": int(improvement_episodes),
            "full_success_episodes": int(full_success_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b8_spatial_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b7_corridor_viability_gate_result(results)
    explicit_decision_set = set(B8_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    spatial_map_episodes = 0
    mapped_progress_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        stats = result["stats"]
        trace = result["trace"]
        decisions = [
            str(item.get("b8_decision"))
            for item in trace
            if item.get("b8_decision") is not None
        ]
        map_states = [
            str(item.get("b8_spatial_map_state"))
            for item in trace
            if item.get("b8_spatial_map_state") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        spatial_map = any(state != "non_corridor" for state in map_states)
        mapped_progress = explicit_decision and float(stats.food_distance_delta) >= B7_CORRIDOR_REQUIRED_FOOD_DISTANCE_DELTA
        if explicit_decision:
            explicit_decision_episodes += 1
        if spatial_map:
            spatial_map_episodes += 1
        if mapped_progress:
            mapped_progress_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b8_decision": bool(explicit_decision),
                    "spatial_map_state": bool(spatial_map),
                    "mapped_progress": bool(mapped_progress),
                },
                "decisions": decisions,
                "map_states": map_states,
            }
        )
    aggregate_checks = {
        "base_b7_corridor_gate": bool(base_gate["passed"]),
        "explicit_b8_decision_episodes": explicit_decision_episodes >= 2,
        "spatial_map_episodes": spatial_map_episodes >= 2,
        "mapped_progress_episodes": mapped_progress_episodes >= 2,
    }
    failures.extend(
        "corridor_b8_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "spatial_map_episodes": int(spatial_map_episodes),
            "mapped_progress_episodes": int(mapped_progress_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b9_waypoint_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b8_spatial_corridor_gate_result(results)
    explicit_decision_set = set(B9_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    route_state_episodes = 0
    locked_waypoint_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b9_decision"))
            for item in trace
            if item.get("b9_decision") is not None
        ]
        route_states = [
            str(item.get("b9_route_state"))
            for item in trace
            if item.get("b9_route_state") is not None
        ]
        waypoint_locks = [
            int(item.get("b9_waypoint_lock", 0) or 0)
            for item in trace
            if item.get("b9_waypoint_lock") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        route_state = any(state != "non_corridor" for state in route_states)
        waypoint_locked = explicit_decision and any(lock > 0 for lock in waypoint_locks)
        if explicit_decision:
            explicit_decision_episodes += 1
        if route_state:
            route_state_episodes += 1
        if waypoint_locked:
            locked_waypoint_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b9_decision": bool(explicit_decision),
                    "route_state": bool(route_state),
                    "waypoint_locked": bool(waypoint_locked),
                },
                "decisions": decisions,
                "route_states": route_states,
                "waypoint_locks": waypoint_locks,
            }
        )
    aggregate_checks = {
        "base_b8_corridor_gate": bool(base_gate["passed"]),
        "explicit_b9_decision_episodes": explicit_decision_episodes >= 2,
        "route_state_episodes": route_state_episodes >= 2,
        "locked_waypoint_episodes": locked_waypoint_episodes >= 2,
    }
    failures.extend(
        "corridor_b9_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "route_state_episodes": int(route_state_episodes),
            "locked_waypoint_episodes": int(locked_waypoint_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b10_prospective_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b9_waypoint_corridor_gate_result(results)
    explicit_decision_set = set(B10_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    replay_state_episodes = 0
    committed_plan_episodes = 0
    value_signal_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b10_decision"))
            for item in trace
            if item.get("b10_decision") is not None
        ]
        replay_states = [
            str(item.get("b10_replay_state"))
            for item in trace
            if item.get("b10_replay_state") is not None
        ]
        commitments = [
            int(item.get("b10_plan_commitment", 0) or 0)
            for item in trace
            if item.get("b10_plan_commitment") is not None
        ]
        values = [
            float(item.get("b10_prospective_value", 0.0) or 0.0)
            for item in trace
            if item.get("b10_prospective_value") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        replay_state = any(state != "non_corridor" for state in replay_states)
        committed_plan = explicit_decision and any(lock > 0 for lock in commitments)
        value_signal = any(value > 0.0 for value in values)
        if explicit_decision:
            explicit_decision_episodes += 1
        if replay_state:
            replay_state_episodes += 1
        if committed_plan:
            committed_plan_episodes += 1
        if value_signal:
            value_signal_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b10_decision": bool(explicit_decision),
                    "replay_state": bool(replay_state),
                    "committed_plan": bool(committed_plan),
                    "value_signal": bool(value_signal),
                },
                "decisions": decisions,
                "replay_states": replay_states,
                "commitments": commitments,
                "prospective_values": values,
            }
        )
    aggregate_checks = {
        "base_b9_corridor_gate": bool(base_gate["passed"]),
        "explicit_b10_decision_episodes": explicit_decision_episodes >= 2,
        "replay_state_episodes": replay_state_episodes >= 2,
        "committed_plan_episodes": committed_plan_episodes >= 2,
        "value_signal_episodes": value_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b10_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "replay_state_episodes": int(replay_state_episodes),
            "committed_plan_episodes": int(committed_plan_episodes),
            "value_signal_episodes": int(value_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b11_confidence_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b10_prospective_corridor_gate_result(results)
    explicit_decision_set = set(B11_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    confidence_state_episodes = 0
    confidence_lock_episodes = 0
    neuromod_signal_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b11_decision"))
            for item in trace
            if item.get("b11_decision") is not None
        ]
        confidence_states = [
            str(item.get("b11_confidence_state"))
            for item in trace
            if item.get("b11_confidence_state") is not None
        ]
        locks = [
            int(item.get("b11_confidence_lock", 0) or 0)
            for item in trace
            if item.get("b11_confidence_lock") is not None
        ]
        signals = [
            float(item.get("b11_neuromod_signal", 0.0) or 0.0)
            for item in trace
            if item.get("b11_neuromod_signal") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        confidence_state = any(state != "non_corridor" for state in confidence_states)
        confidence_locked = explicit_decision and any(lock > 0 for lock in locks)
        neuromod_signal = any(signal > 0.0 for signal in signals)
        if explicit_decision:
            explicit_decision_episodes += 1
        if confidence_state:
            confidence_state_episodes += 1
        if confidence_locked:
            confidence_lock_episodes += 1
        if neuromod_signal:
            neuromod_signal_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b11_decision": bool(explicit_decision),
                    "confidence_state": bool(confidence_state),
                    "confidence_locked": bool(confidence_locked),
                    "neuromod_signal": bool(neuromod_signal),
                },
                "decisions": decisions,
                "confidence_states": confidence_states,
                "confidence_locks": locks,
                "neuromod_signals": signals,
            }
        )
    aggregate_checks = {
        "base_b10_corridor_gate": bool(base_gate["passed"]),
        "explicit_b11_decision_episodes": explicit_decision_episodes >= 2,
        "confidence_state_episodes": confidence_state_episodes >= 2,
        "confidence_lock_episodes": confidence_lock_episodes >= 2,
        "neuromod_signal_episodes": neuromod_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b11_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "confidence_state_episodes": int(confidence_state_episodes),
            "confidence_lock_episodes": int(confidence_lock_episodes),
            "neuromod_signal_episodes": int(neuromod_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b12_attention_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b11_confidence_corridor_gate_result(results)
    explicit_decision_set = set(B12_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    attention_state_episodes = 0
    attention_lock_episodes = 0
    prediction_signal_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b12_decision"))
            for item in trace
            if item.get("b12_decision") is not None
        ]
        attention_states = [
            str(item.get("b12_attention_state"))
            for item in trace
            if item.get("b12_attention_state") is not None
        ]
        locks = [
            int(item.get("b12_search_lock", 0) or 0)
            for item in trace
            if item.get("b12_search_lock") is not None
        ]
        signals = [
            float(item.get("b12_attention_gain", 0.0) or 0.0)
            + float(item.get("b12_prediction_error", 0.0) or 0.0)
            for item in trace
            if item.get("b12_attention_gain") is not None
            or item.get("b12_prediction_error") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        attention_state = any(state != "non_corridor" for state in attention_states)
        attention_locked = explicit_decision and any(lock > 0 for lock in locks)
        prediction_signal = any(signal > 0.0 for signal in signals)
        if explicit_decision:
            explicit_decision_episodes += 1
        if attention_state:
            attention_state_episodes += 1
        if attention_locked:
            attention_lock_episodes += 1
        if prediction_signal:
            prediction_signal_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b12_decision": bool(explicit_decision),
                    "attention_state": bool(attention_state),
                    "attention_locked": bool(attention_locked),
                    "prediction_signal": bool(prediction_signal),
                },
                "decisions": decisions,
                "attention_states": attention_states,
                "search_locks": locks,
                "prediction_signals": signals,
            }
        )
    aggregate_checks = {
        "base_b11_corridor_gate": bool(base_gate["passed"]),
        "explicit_b12_decision_episodes": explicit_decision_episodes >= 2,
        "attention_state_episodes": attention_state_episodes >= 2,
        "attention_lock_episodes": attention_lock_episodes >= 2,
        "prediction_signal_episodes": prediction_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b12_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "attention_state_episodes": int(attention_state_episodes),
            "attention_lock_episodes": int(attention_lock_episodes),
            "prediction_signal_episodes": int(prediction_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b13_local_search_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b12_attention_corridor_gate_result(results)
    explicit_decision_set = set(B13_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    search_state_episodes = 0
    search_lock_episodes = 0
    local_signal_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b13_decision"))
            for item in trace
            if item.get("b13_decision") is not None
        ]
        search_states = [
            str(item.get("b13_search_state"))
            for item in trace
            if item.get("b13_search_state") is not None
        ]
        locks = [
            int(item.get("b13_search_lock", 0) or 0)
            for item in trace
            if item.get("b13_search_lock") is not None
        ]
        signals = [
            float(item.get("b13_local_route_score", 0.0) or 0.0)
            + float(item.get("b13_affordance_samples", 0.0) or 0.0)
            + float(item.get("b13_dead_end_score", 0.0) or 0.0)
            for item in trace
            if item.get("b13_local_route_score") is not None
            or item.get("b13_affordance_samples") is not None
            or item.get("b13_dead_end_score") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        search_state = any(state != "non_corridor" for state in search_states)
        search_locked = explicit_decision and any(lock > 0 for lock in locks)
        local_signal = any(signal > 0.0 for signal in signals)
        if explicit_decision:
            explicit_decision_episodes += 1
        if search_state:
            search_state_episodes += 1
        if search_locked:
            search_lock_episodes += 1
        if local_signal:
            local_signal_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b13_decision": bool(explicit_decision),
                    "search_state": bool(search_state),
                    "search_locked": bool(search_locked),
                    "local_search_signal": bool(local_signal),
                },
                "decisions": decisions,
                "search_states": search_states,
                "search_locks": locks,
                "local_search_signals": signals,
            }
        )
    aggregate_checks = {
        "base_b12_corridor_gate": bool(base_gate["passed"]),
        "explicit_b13_decision_episodes": explicit_decision_episodes >= 2,
        "search_state_episodes": search_state_episodes >= 2,
        "search_lock_episodes": search_lock_episodes >= 2,
        "local_search_signal_episodes": local_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b13_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "search_state_episodes": int(search_state_episodes),
            "search_lock_episodes": int(search_lock_episodes),
            "local_search_signal_episodes": int(local_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b14_uncertainty_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b13_local_search_corridor_gate_result(results)
    explicit_decision_set = set(B14_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    uncertainty_state_episodes = 0
    confidence_signal_episodes = 0
    commitment_lock_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b14_decision"))
            for item in trace
            if item.get("b14_decision") is not None
        ]
        states = [
            str(item.get("b14_uncertainty_state"))
            for item in trace
            if item.get("b14_uncertainty_state") is not None
        ]
        locks = [
            int(item.get("b14_commitment_lock", 0) or 0)
            for item in trace
            if item.get("b14_commitment_lock") is not None
        ]
        signals = [
            float(item.get("b14_affordance_confidence", 0.0) or 0.0)
            + float(item.get("b14_uncertainty", 0.0) or 0.0)
            + float(item.get("b14_risk_adjusted_score", 0.0) or 0.0)
            for item in trace
            if item.get("b14_affordance_confidence") is not None
            or item.get("b14_uncertainty") is not None
            or item.get("b14_risk_adjusted_score") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        uncertainty_state = any(state != "non_corridor" for state in states)
        confidence_signal = any(signal > 0.0 for signal in signals)
        commitment_lock = explicit_decision and any(lock > 0 for lock in locks)
        if explicit_decision:
            explicit_decision_episodes += 1
        if uncertainty_state:
            uncertainty_state_episodes += 1
        if confidence_signal:
            confidence_signal_episodes += 1
        if commitment_lock:
            commitment_lock_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b14_decision": bool(explicit_decision),
                    "uncertainty_state": bool(uncertainty_state),
                    "confidence_signal": bool(confidence_signal),
                    "commitment_lock": bool(commitment_lock),
                },
                "decisions": decisions,
                "uncertainty_states": states,
                "commitment_locks": locks,
                "confidence_signals": signals,
            }
        )
    aggregate_checks = {
        "base_b13_corridor_gate": bool(base_gate["passed"]),
        "explicit_b14_decision_episodes": explicit_decision_episodes >= 2,
        "uncertainty_state_episodes": uncertainty_state_episodes >= 2,
        "confidence_signal_episodes": confidence_signal_episodes >= 2,
        "commitment_lock_episodes": commitment_lock_episodes >= 2,
    }
    failures.extend(
        "corridor_b14_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "uncertainty_state_episodes": int(uncertainty_state_episodes),
            "confidence_signal_episodes": int(confidence_signal_episodes),
            "commitment_lock_episodes": int(commitment_lock_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b15_option_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b14_uncertainty_corridor_gate_result(results)
    explicit_decision_set = set(B15_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    option_state_episodes = 0
    option_lock_episodes = 0
    option_value_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b15_decision"))
            for item in trace
            if item.get("b15_decision") is not None
        ]
        states = [
            str(item.get("b15_option_state"))
            for item in trace
            if item.get("b15_option_state") is not None
        ]
        locks = [
            int(item.get("b15_option_lock", 0) or 0)
            for item in trace
            if item.get("b15_option_lock") is not None
        ]
        signals = [
            float(item.get("b15_option_value", 0.0) or 0.0)
            + float(item.get("b15_termination_pressure", 0.0) or 0.0)
            + float(item.get("b15_persistence_score", 0.0) or 0.0)
            for item in trace
            if item.get("b15_option_value") is not None
            or item.get("b15_termination_pressure") is not None
            or item.get("b15_persistence_score") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        option_state = any(state != "non_corridor" for state in states)
        option_lock = explicit_decision and any(lock > 0 for lock in locks)
        option_value = any(signal > 0.0 for signal in signals)
        if explicit_decision:
            explicit_decision_episodes += 1
        if option_state:
            option_state_episodes += 1
        if option_lock:
            option_lock_episodes += 1
        if option_value:
            option_value_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b15_decision": bool(explicit_decision),
                    "option_state": bool(option_state),
                    "option_lock": bool(option_lock),
                    "option_value_signal": bool(option_value),
                },
                "decisions": decisions,
                "option_states": states,
                "option_locks": locks,
                "option_value_signals": signals,
            }
        )
    aggregate_checks = {
        "base_b14_corridor_gate": bool(base_gate["passed"]),
        "explicit_b15_decision_episodes": explicit_decision_episodes >= 2,
        "option_state_episodes": option_state_episodes >= 2,
        "option_lock_episodes": option_lock_episodes >= 2,
        "option_value_signal_episodes": option_value_episodes >= 2,
    }
    failures.extend(
        "corridor_b15_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "option_state_episodes": int(option_state_episodes),
            "option_lock_episodes": int(option_lock_episodes),
            "option_value_signal_episodes": int(option_value_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b16_option_ensemble_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b15_option_corridor_gate_result(results)
    explicit_decision_set = set(B16_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    ensemble_state_episodes = 0
    ensemble_lock_episodes = 0
    ensemble_signal_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b16_decision"))
            for item in trace
            if item.get("b16_decision") is not None
        ]
        states = [
            str(item.get("b16_ensemble_state"))
            for item in trace
            if item.get("b16_ensemble_state") is not None
        ]
        locks = [
            int(item.get("b16_ensemble_lock", 0) or 0)
            for item in trace
            if item.get("b16_ensemble_lock") is not None
        ]
        signals = [
            float(item.get("b16_continue_vote", 0.0) or 0.0)
            + float(item.get("b16_return_vote", 0.0) or 0.0)
            + float(item.get("b16_consensus_score", 0.0) or 0.0)
            + float(item.get("b16_conflict_score", 0.0) or 0.0)
            for item in trace
            if item.get("b16_continue_vote") is not None
            or item.get("b16_return_vote") is not None
            or item.get("b16_consensus_score") is not None
            or item.get("b16_conflict_score") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        ensemble_state = any(state != "non_corridor" for state in states)
        ensemble_lock = explicit_decision and any(lock > 0 for lock in locks)
        ensemble_signal = any(signal > 0.0 for signal in signals)
        if explicit_decision:
            explicit_decision_episodes += 1
        if ensemble_state:
            ensemble_state_episodes += 1
        if ensemble_lock:
            ensemble_lock_episodes += 1
        if ensemble_signal:
            ensemble_signal_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b16_decision": bool(explicit_decision),
                    "ensemble_state": bool(ensemble_state),
                    "ensemble_lock": bool(ensemble_lock),
                    "ensemble_signal": bool(ensemble_signal),
                },
                "decisions": decisions,
                "ensemble_states": states,
                "ensemble_locks": locks,
                "ensemble_signals": signals,
            }
        )
    aggregate_checks = {
        "base_b15_corridor_gate": bool(base_gate["passed"]),
        "explicit_b16_decision_episodes": explicit_decision_episodes >= 2,
        "ensemble_state_episodes": ensemble_state_episodes >= 2,
        "ensemble_lock_episodes": ensemble_lock_episodes >= 2,
        "ensemble_signal_episodes": ensemble_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b16_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "ensemble_state_episodes": int(ensemble_state_episodes),
            "ensemble_lock_episodes": int(ensemble_lock_episodes),
            "ensemble_signal_episodes": int(ensemble_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b17_neuromodulated_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b16_option_ensemble_corridor_gate_result(results)
    explicit_decision_set = set(B17_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    modulator_state_episodes = 0
    modulation_lock_episodes = 0
    modulation_signal_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b17_decision"))
            for item in trace
            if item.get("b17_decision") is not None
        ]
        states = [
            str(item.get("b17_modulator_state"))
            for item in trace
            if item.get("b17_modulator_state") is not None
        ]
        locks = [
            int(item.get("b17_modulation_lock", 0) or 0)
            for item in trace
            if item.get("b17_modulation_lock") is not None
        ]
        signals = [
            float(item.get("b17_arousal_signal", 0.0) or 0.0)
            + float(item.get("b17_homeostatic_gain", 0.0) or 0.0)
            + float(item.get("b17_option_gain", 0.0) or 0.0)
            + float(item.get("b17_conflict_release", 0.0) or 0.0)
            for item in trace
            if item.get("b17_arousal_signal") is not None
            or item.get("b17_homeostatic_gain") is not None
            or item.get("b17_option_gain") is not None
            or item.get("b17_conflict_release") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        modulator_state = any(state != "non_corridor" for state in states)
        modulation_lock = explicit_decision and any(lock > 0 for lock in locks)
        modulation_signal = any(signal > 0.0 for signal in signals)
        if explicit_decision:
            explicit_decision_episodes += 1
        if modulator_state:
            modulator_state_episodes += 1
        if modulation_lock:
            modulation_lock_episodes += 1
        if modulation_signal:
            modulation_signal_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b17_decision": bool(explicit_decision),
                    "modulator_state": bool(modulator_state),
                    "modulation_lock": bool(modulation_lock),
                    "modulation_signal": bool(modulation_signal),
                },
                "decisions": decisions,
                "modulator_states": states,
                "modulation_locks": locks,
                "modulation_signals": signals,
            }
        )
    aggregate_checks = {
        "base_b16_corridor_gate": bool(base_gate["passed"]),
        "explicit_b17_decision_episodes": explicit_decision_episodes >= 2,
        "modulator_state_episodes": modulator_state_episodes >= 2,
        "modulation_lock_episodes": modulation_lock_episodes >= 2,
        "modulation_signal_episodes": modulation_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b17_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "modulator_state_episodes": int(modulator_state_episodes),
            "modulation_lock_episodes": int(modulation_lock_episodes),
            "modulation_signal_episodes": int(modulation_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b18_eligibility_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b17_neuromodulated_corridor_gate_result(results)
    explicit_decision_set = set(B18_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = list(base_gate["failures"])
    episode_results = []
    explicit_decision_episodes = 0
    trace_state_episodes = 0
    trace_lock_episodes = 0
    trace_signal_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        trace = result["trace"]
        decisions = [
            str(item.get("b18_decision"))
            for item in trace
            if item.get("b18_decision") is not None
        ]
        states = [
            str(item.get("b18_trace_state"))
            for item in trace
            if item.get("b18_trace_state") is not None
        ]
        locks = [
            int(item.get("b18_trace_lock", 0) or 0)
            for item in trace
            if item.get("b18_trace_lock") is not None
        ]
        signals = [
            float(item.get("b18_eligibility_trace", 0.0) or 0.0)
            + float(item.get("b18_reward_prediction_proxy", 0.0) or 0.0)
            + float(item.get("b18_stability_bias", 0.0) or 0.0)
            + float(item.get("b18_switch_pressure", 0.0) or 0.0)
            for item in trace
            if item.get("b18_eligibility_trace") is not None
            or item.get("b18_reward_prediction_proxy") is not None
            or item.get("b18_stability_bias") is not None
            or item.get("b18_switch_pressure") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        trace_state = any(state != "non_corridor" for state in states)
        trace_lock = explicit_decision and any(lock > 0 for lock in locks)
        trace_signal = any(signal > 0.0 for signal in signals)
        if explicit_decision:
            explicit_decision_episodes += 1
        if trace_state:
            trace_state_episodes += 1
        if trace_lock:
            trace_lock_episodes += 1
        if trace_signal:
            trace_signal_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b18_decision": bool(explicit_decision),
                    "trace_state": bool(trace_state),
                    "trace_lock": bool(trace_lock),
                    "trace_signal": bool(trace_signal),
                },
                "decisions": decisions,
                "trace_states": states,
                "trace_locks": locks,
                "trace_signals": signals,
            }
        )
    aggregate_checks = {
        "base_b17_corridor_gate": bool(base_gate["passed"]),
        "explicit_b18_decision_episodes": explicit_decision_episodes >= 2,
        "trace_state_episodes": trace_state_episodes >= 2,
        "trace_lock_episodes": trace_lock_episodes >= 2,
        "trace_signal_episodes": trace_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b18_aggregate:" + name
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
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "trace_state_episodes": int(trace_state_episodes),
            "trace_lock_episodes": int(trace_lock_episodes),
            "trace_signal_episodes": int(trace_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b19_episodic_corridor_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    base_gate = b18_eligibility_corridor_gate_result(results)
    explicit_decision_set = set(B19_CORRIDOR_EXPLICIT_DECISIONS)
    failures: list[str] = []
    episode_results = []
    explicit_decision_episodes = 0
    memory_state_episodes = 0
    memory_lock_episodes = 0
    memory_signal_episodes = 0
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
            str(item.get("b19_decision"))
            for item in trace
            if item.get("b19_decision") is not None
        ]
        states = [
            str(item.get("b19_memory_state"))
            for item in trace
            if item.get("b19_memory_state") is not None
        ]
        locks = [
            int(item.get("b19_memory_lock", 0) or 0)
            for item in trace
            if item.get("b19_memory_lock") is not None
        ]
        signals = [
            float(item.get("b19_episode_memory", 0.0) or 0.0)
            + float(item.get("b19_consolidation_score", 0.0) or 0.0)
            + float(item.get("b19_stability_vote", 0.0) or 0.0)
            + float(item.get("b19_switch_suppression", 0.0) or 0.0)
            for item in trace
            if item.get("b19_episode_memory") is not None
            or item.get("b19_consolidation_score") is not None
            or item.get("b19_stability_vote") is not None
            or item.get("b19_switch_suppression") is not None
        ]
        explicit_decision = any(decision in explicit_decision_set for decision in decisions)
        memory_state = any(state != "non_corridor" for state in states)
        memory_lock = explicit_decision and any(lock > 0 for lock in locks)
        memory_signal = any(signal > 0.0 for signal in signals)
        corridor_safety = primitive_ok and predator_contacts == 0
        if explicit_decision:
            explicit_decision_episodes += 1
        if memory_state:
            memory_state_episodes += 1
        if memory_lock:
            memory_lock_episodes += 1
        if memory_signal:
            memory_signal_episodes += 1
        if corridor_safety:
            corridor_safety_episodes += 1
        episode_results.append(
            {
                "evaluation_episode": episode,
                "checks": {
                    "explicit_b19_decision": bool(explicit_decision),
                    "memory_state": bool(memory_state),
                    "memory_lock": bool(memory_lock),
                    "memory_signal": bool(memory_signal),
                    "corridor_safety": bool(corridor_safety),
                },
                "decisions": decisions,
                "memory_states": states,
                "memory_locks": locks,
                "memory_signals": signals,
                "predator_contacts": predator_contacts,
                "primitive_violations": primitive_violations,
            }
        )
    aggregate_checks = {
        "corridor_safety_episodes": corridor_safety_episodes == len(results),
        "explicit_b19_decision_episodes": explicit_decision_episodes >= 2,
        "memory_state_episodes": memory_state_episodes >= 2,
        "memory_lock_episodes": memory_lock_episodes >= 2,
        "memory_signal_episodes": memory_signal_episodes >= 2,
    }
    failures.extend(
        "corridor_b19_aggregate:" + name
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
            "base_b18_corridor_diagnostic": bool(base_gate["passed"]),
            "corridor_safety_episodes": int(corridor_safety_episodes),
            "explicit_decision_episodes": int(explicit_decision_episodes),
            "memory_state_episodes": int(memory_state_episodes),
            "memory_lock_episodes": int(memory_lock_episodes),
            "memory_signal_episodes": int(memory_signal_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
