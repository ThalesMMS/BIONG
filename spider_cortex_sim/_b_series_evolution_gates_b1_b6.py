from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

def _stats_payload(stats: EpisodeStats) -> dict[str, object]:
    return {
        "scenario": stats.scenario,
        "steps": int(stats.steps),
        "alive": bool(stats.alive),
        "food_eaten": int(stats.food_eaten),
        "sleep_events": int(stats.sleep_events),
        "shelter_entries": int(stats.shelter_entries),
        "predator_contacts": int(stats.predator_contacts),
        "final_health": round(float(stats.final_health), 6),
        "total_reward": round(float(stats.total_reward), 6),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _behavior_score_payload(score: object) -> dict[str, object]:
    checks = getattr(score, "checks", {})
    return {
        "success": bool(getattr(score, "success", False)),
        "failures": list(getattr(score, "failures", [])),
        "behavior_metrics": _json_safe(getattr(score, "behavior_metrics", {})),
        "checks": {
            str(name): {
                "passed": bool(getattr(check, "passed", False)),
                "value": _json_safe(getattr(check, "value", None)),
            }
            for name, check in dict(checks).items()
        },
    }


def trace_uses_only_primitive_actions(
    trace: Sequence[dict[str, object]],
) -> tuple[bool, list[dict[str, object]]]:
    primitive_actions = set(ACTIONS)
    violations: list[dict[str, object]] = []
    for item in trace:
        tick = int(item.get("tick", -1))
        for key in ("action", "intended_action", "executed_action", "bridge_primitive_action"):
            value = item.get(key)
            if value is None:
                continue
            if str(value) not in primitive_actions:
                violations.append({"tick": tick, "field": key, "value": value})
    return not violations, violations


def b1_easy_gate_result(
    stats: EpisodeStats,
    trace: Sequence[dict[str, object]],
    *,
    scenario_name: str = B1_EASY_SCENARIO,
) -> dict[str, object]:
    scenario = get_scenario(scenario_name)
    primitive_trace_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
    checks = {
        "alive": bool(stats.alive),
        "completed_horizon": int(stats.steps) >= int(scenario.max_steps),
        "food_events": int(stats.food_eaten) > 0,
        "sleep_events": int(stats.sleep_events) > 0,
        "shelter_entries": int(stats.shelter_entries) > 0,
        "primitive_trace": primitive_trace_ok,
    }
    passed = all(checks.values())
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "scenario": scenario_name,
        "status": "accepted" if passed else "discarded",
        "passed": bool(passed),
        "checks": checks,
        "failures": failures,
        "primitive_violations": primitive_violations,
    }


def b2_canonical_progress_gate_result(
    stats: EpisodeStats,
    trace: Sequence[dict[str, object]],
    *,
    scenario_name: str = B1_CANONICAL_SCENARIO,
) -> dict[str, object]:
    scenario = get_scenario(scenario_name)
    primitive_trace_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
    completed_horizon = bool(stats.alive) and int(stats.steps) >= int(scenario.max_steps)
    checks = {
        "completed_horizon_or_min_steps": completed_horizon
        or int(stats.steps) >= B2_CANONICAL_MIN_STEPS,
        "food_events": completed_horizon
        or int(stats.food_eaten) >= int(B2_CANONICAL_BASELINE["food_eaten"]),
        "sleep_events": completed_horizon
        or int(stats.sleep_events) >= int(B2_CANONICAL_BASELINE["sleep_events"]),
        "shelter_entries": completed_horizon
        or int(stats.shelter_entries) >= int(B2_CANONICAL_BASELINE["shelter_entries"]),
        "predator_contacts": completed_horizon
        or int(stats.predator_contacts) <= B2_CANONICAL_MAX_PREDATOR_CONTACTS,
        "primitive_trace": primitive_trace_ok,
    }
    passed = all(checks.values())
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "scenario": scenario_name,
        "status": "accepted" if passed else "discarded",
        "passed": bool(passed),
        "baseline": dict(B2_CANONICAL_BASELINE),
        "checks": checks,
        "failures": failures,
        "primitive_violations": primitive_violations,
    }


def b3_canonical_robust_gate_result(
    stats: EpisodeStats,
    trace: Sequence[dict[str, object]],
    *,
    evaluation_episode: int,
    scenario_name: str = B1_CANONICAL_SCENARIO,
) -> dict[str, object]:
    scenario = get_scenario(scenario_name)
    primitive_trace_ok, primitive_violations = trace_uses_only_primitive_actions(trace)
    if int(evaluation_episode) == 0:
        checks = {
            "alive": bool(stats.alive),
            "completed_horizon": int(stats.steps) >= int(scenario.max_steps),
            "food_events": int(stats.food_eaten) > 0,
            "sleep_events": int(stats.sleep_events) > 0,
            "shelter_entries": int(stats.shelter_entries) > 0,
            "predator_contacts": int(stats.predator_contacts)
            <= B3_CANONICAL_MAX_PREDATOR_CONTACTS_EP0,
            "primitive_trace": primitive_trace_ok,
        }
    else:
        checks = {
            "min_steps": int(stats.steps) >= 100,
            "food_events": int(stats.food_eaten) >= 3,
            "sleep_events": int(stats.sleep_events) >= 3,
            "shelter_entries": int(stats.shelter_entries) >= 5,
            "predator_contacts": int(stats.predator_contacts)
            <= B3_CANONICAL_MAX_PREDATOR_CONTACTS_EP1,
            "primitive_trace": primitive_trace_ok,
        }
    passed = all(checks.values())
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "scenario": scenario_name,
        "evaluation_episode": int(evaluation_episode),
        "status": "accepted" if passed else "discarded",
        "passed": bool(passed),
        "baseline": dict(B3_CANONICAL_BASELINES.get(int(evaluation_episode), {})),
        "checks": checks,
        "failures": failures,
        "primitive_violations": primitive_violations,
    }


def b4_easy_multi_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    episode_gates = []
    for result in results:
        episode = int(result["evaluation_episode"])
        stats = result["stats"]
        trace = result["trace"]
        gate = b1_easy_gate_result(
            stats,
            trace,
            scenario_name=B1_EASY_SCENARIO,
        )
        episode_gates.append(
            {
                "evaluation_episode": episode,
                "gate": gate,
                "metrics": _stats_payload(stats),
            }
        )
        if not gate["passed"]:
            failures.append("easy_ep" + str(episode) + ":" + ",".join(gate["failures"]))
    return {
        "scenario": B1_EASY_SCENARIO,
        "status": "accepted" if not failures else "discarded",
        "passed": not failures,
        "failures": failures,
        "episode_results": episode_gates,
    }


def b4_canonical_multi_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    episode_results = []
    completed_horizons = 0
    min_steps: int | None = None
    total_contacts = 0
    food_cycle_episodes = 0
    sleep_cycle_episodes = 0
    shelter_cycle_episodes = 0
    scenario = get_scenario(B1_CANONICAL_SCENARIO)
    for result in results:
        episode = int(result["evaluation_episode"])
        stats = result["stats"]
        trace = result["trace"]
        primitive_trace_ok, primitive_violations = trace_uses_only_primitive_actions(
            trace
        )
        completed_horizon = bool(stats.alive) and int(stats.steps) >= int(
            scenario.max_steps
        )
        if completed_horizon:
            completed_horizons += 1
        min_steps = (
            int(stats.steps)
            if min_steps is None
            else min(min_steps, int(stats.steps))
        )
        total_contacts += int(stats.predator_contacts)
        food_cycle = int(stats.food_eaten) >= 3
        sleep_cycle = int(stats.sleep_events) >= 3
        shelter_cycle = int(stats.shelter_entries) >= 2
        if food_cycle:
            food_cycle_episodes += 1
        if sleep_cycle:
            sleep_cycle_episodes += 1
        if shelter_cycle:
            shelter_cycle_episodes += 1
        checks = {
            "survival_floor": int(stats.steps) >= B4_CANONICAL_MIN_STEPS,
            "predator_contacts": int(stats.predator_contacts) <= 5,
            "primitive_trace": primitive_trace_ok,
        }
        if episode in {0, 1}:
            checks["b3_anchor_completed_horizon"] = completed_horizon
            checks["b3_anchor_predator_contacts"] = int(stats.predator_contacts) <= 4
            checks["b3_anchor_food_events"] = food_cycle
            checks["b3_anchor_sleep_events"] = sleep_cycle
            checks["b3_anchor_shelter_entries"] = shelter_cycle
        passed = all(checks.values())
        episode_failures = [name for name, ok in checks.items() if not ok]
        if not passed:
            failures.append(
                "canonical_ep" + str(episode) + ":" + ",".join(episode_failures)
            )
        episode_results.append(
            {
                "evaluation_episode": episode,
                "gate": {
                    "scenario": B1_CANONICAL_SCENARIO,
                    "evaluation_episode": episode,
                    "status": "accepted" if passed else "discarded",
                    "passed": passed,
                    "checks": checks,
                    "failures": episode_failures,
                    "primitive_violations": primitive_violations,
                },
                "metrics": _stats_payload(stats),
            }
        )
    aggregate_checks = {
        "completed_horizons": completed_horizons
        >= B4_CANONICAL_MIN_COMPLETED_HORIZONS,
        "min_steps": int(min_steps or 0) >= B4_CANONICAL_MIN_STEPS,
        "total_predator_contacts": total_contacts
        <= B4_CANONICAL_MAX_TOTAL_PREDATOR_CONTACTS,
        "food_cycle_episodes": food_cycle_episodes
        >= B4_CANONICAL_MIN_FOOD_CYCLE_EPISODES,
        "sleep_cycle_episodes": sleep_cycle_episodes
        >= B4_CANONICAL_MIN_SLEEP_CYCLE_EPISODES,
        "shelter_cycle_episodes": shelter_cycle_episodes
        >= B4_CANONICAL_MIN_SHELTER_CYCLE_EPISODES,
    }
    aggregate_failures = [
        "aggregate:" + name for name, ok in aggregate_checks.items() if not ok
    ]
    failures.extend(aggregate_failures)
    passed = not failures
    return {
        "scenario": B1_CANONICAL_SCENARIO,
        "status": "accepted" if passed else "discarded",
        "passed": passed,
        "baseline": dict(B4_CANONICAL_BASELINE),
        "objective": "retain B3 survival while adding recovery-balance modular control",
        "aggregate": {
            "completed_horizons": int(completed_horizons),
            "min_steps": int(min_steps or 0),
            "total_predator_contacts": int(total_contacts),
            "food_cycle_episodes": int(food_cycle_episodes),
            "sleep_cycle_episodes": int(sleep_cycle_episodes),
            "shelter_cycle_episodes": int(shelter_cycle_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b5_food_deprivation_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    scenario = get_scenario(B5_FOOD_DEPRIVATION_SCENARIO)
    failures: list[str] = []
    episode_results = []
    progress_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        stats = result["stats"]
        trace = result["trace"]
        score = scenario.score_episode(stats, trace)
        primitive_trace_ok, primitive_violations = trace_uses_only_primitive_actions(
            trace
        )
        progress = bool(float(stats.food_distance_delta) > 0.0 or int(stats.food_eaten) >= 2)
        if progress:
            progress_episodes += 1
        checks = {
            "alive": bool(stats.alive),
            "completed_horizon": int(stats.steps) >= int(scenario.max_steps),
            "food_eaten": int(stats.food_eaten) >= 1,
            "final_hunger": float(stats.final_hunger) <= 0.90,
            "primitive_trace": primitive_trace_ok,
        }
        passed = all(checks.values())
        episode_failures = [name for name, ok in checks.items() if not ok]
        if not passed:
            failures.append(
                "food_deprivation_ep" + str(episode) + ":" + ",".join(episode_failures)
            )
        episode_results.append(
            {
                "evaluation_episode": episode,
                "gate": {
                    "scenario": B5_FOOD_DEPRIVATION_SCENARIO,
                    "status": "accepted" if passed else "discarded",
                    "passed": passed,
                    "checks": checks,
                    "progress": progress,
                    "failures": episode_failures,
                    "primitive_violations": primitive_violations,
                },
                "metrics": {
                    **_stats_payload(stats),
                    "final_hunger": round(float(stats.final_hunger), 6),
                    "food_distance_delta": round(float(stats.food_distance_delta), 6),
                },
                "behavior_score": _behavior_score_payload(score),
            }
        )
    aggregate_checks = {"progress_episodes": int(progress_episodes) >= 2}
    failures.extend(
        "food_deprivation_aggregate:" + name
        for name, ok in aggregate_checks.items()
        if not ok
    )
    passed = not failures
    return {
        "scenario": B5_FOOD_DEPRIVATION_SCENARIO,
        "status": "accepted" if passed else "discarded",
        "passed": passed,
        "aggregate": {
            "progress_episodes": int(progress_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b5_sleep_conflict_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    scenario = get_scenario(B5_SLEEP_CONFLICT_SCENARIO)
    failures: list[str] = []
    episode_results = []
    movement_metric_exposed = False
    post_recovery_movement_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        stats = result["stats"]
        trace = result["trace"]
        score = scenario.score_episode(stats, trace)
        score_payload = _behavior_score_payload(score)
        behavior_metrics = score_payload["behavior_metrics"]
        post_recovery_tick = None
        if isinstance(behavior_metrics, dict):
            movement_metric_exposed = (
                movement_metric_exposed
                or "post_recovery_movement_tick" in behavior_metrics
            )
            post_recovery_tick = behavior_metrics.get("post_recovery_movement_tick")
        if post_recovery_tick is not None:
            post_recovery_movement_episodes += 1
        primitive_trace_ok, primitive_violations = trace_uses_only_primitive_actions(
            trace
        )
        checks = {
            "alive": bool(stats.alive),
            "completed_horizon": int(stats.steps) >= int(scenario.max_steps),
            "sleep_events": int(stats.sleep_events) >= 6,
            "final_sleep_debt": float(stats.final_sleep_debt) <= 0.08,
            "primitive_trace": primitive_trace_ok,
        }
        passed = all(checks.values())
        episode_failures = [name for name, ok in checks.items() if not ok]
        if not passed:
            failures.append(
                "sleep_conflict_ep" + str(episode) + ":" + ",".join(episode_failures)
            )
        episode_results.append(
            {
                "evaluation_episode": episode,
                "gate": {
                    "scenario": B5_SLEEP_CONFLICT_SCENARIO,
                    "status": "accepted" if passed else "discarded",
                    "passed": passed,
                    "checks": checks,
                    "post_recovery_movement_tick": post_recovery_tick,
                    "failures": episode_failures,
                    "primitive_violations": primitive_violations,
                },
                "metrics": {
                    **_stats_payload(stats),
                    "final_sleep_debt": round(float(stats.final_sleep_debt), 6),
                },
                "behavior_score": score_payload,
            }
        )
    aggregate_checks = {
        "post_recovery_movement_episodes": (
            int(post_recovery_movement_episodes) >= 2
            if movement_metric_exposed
            else True
        )
    }
    failures.extend(
        "sleep_conflict_aggregate:" + name
        for name, ok in aggregate_checks.items()
        if not ok
    )
    passed = not failures
    return {
        "scenario": B5_SLEEP_CONFLICT_SCENARIO,
        "status": "accepted" if passed else "discarded",
        "passed": passed,
        "aggregate": {
            "movement_metric_exposed": bool(movement_metric_exposed),
            "post_recovery_movement_episodes": int(post_recovery_movement_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b5_diagnostic_probe_result(
    scenario_name: str,
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    scenario = get_scenario(scenario_name)
    episode_results = []
    for result in results:
        stats = result["stats"]
        trace = result["trace"]
        score = scenario.score_episode(stats, trace)
        primitive_trace_ok, primitive_violations = trace_uses_only_primitive_actions(
            trace
        )
        episode_results.append(
            {
                "evaluation_episode": int(result["evaluation_episode"]),
                "metrics": _stats_payload(stats),
                "primitive_trace": primitive_trace_ok,
                "primitive_violations": primitive_violations,
                "behavior_score": _behavior_score_payload(score),
            }
        )
    return {
        "scenario": scenario_name,
        "blocking": False,
        "episode_results": episode_results,
    }


def b6_food_predator_conflict_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    scenario = get_scenario(B6_FOOD_PREDATOR_SCENARIO)
    failures: list[str] = []
    episode_results = []
    threat_exposure_episodes = 0
    threat_priority_or_suppression_episodes = 0
    full_success_episodes = 0
    for result in results:
        episode = int(result["evaluation_episode"])
        stats = result["stats"]
        trace = result["trace"]
        score = scenario.score_episode(stats, trace)
        score_payload = _behavior_score_payload(score)
        metrics = score_payload["behavior_metrics"]
        metrics = metrics if isinstance(metrics, dict) else {}
        danger_tick_count = int(metrics.get("danger_tick_count", 0) or 0)
        threat_priority_rate = float(metrics.get("threat_priority_rate", 0.0) or 0.0)
        foraging_suppressed_rate = float(
            metrics.get("foraging_suppressed_rate", 0.0) or 0.0
        )
        if danger_tick_count >= 1:
            threat_exposure_episodes += 1
        if threat_priority_rate > 0.0 or foraging_suppressed_rate > 0.0:
            threat_priority_or_suppression_episodes += 1
        if bool(score_payload["success"]):
            full_success_episodes += 1
        primitive_trace_ok, primitive_violations = trace_uses_only_primitive_actions(
            trace
        )
        checks = {
            "survives_without_contact": bool(stats.alive)
            and int(stats.predator_contacts) == 0,
            "primitive_trace": primitive_trace_ok,
        }
        episode_failures = [name for name, ok in checks.items() if not ok]
        if episode_failures:
            failures.append(
                "food_predator_ep" + str(episode) + ":" + ",".join(episode_failures)
            )
        episode_results.append(
            {
                "evaluation_episode": episode,
                "gate": {
                    "scenario": B6_FOOD_PREDATOR_SCENARIO,
                    "status": "accepted" if not episode_failures else "discarded",
                    "passed": not episode_failures,
                    "checks": checks,
                    "failures": episode_failures,
                    "primitive_violations": primitive_violations,
                },
                "metrics": _stats_payload(stats),
                "behavior_score": score_payload,
            }
        )
    aggregate_checks = {
        "threat_exposure_episodes": threat_exposure_episodes >= 2,
        "threat_priority_or_suppression_episodes": (
            threat_priority_or_suppression_episodes >= 2
        ),
    }
    if full_success_episodes >= 2:
        aggregate_checks = {
            "score_success_episodes": True,
            **aggregate_checks,
        }
        aggregate_checks["threat_exposure_episodes"] = True
        aggregate_checks["threat_priority_or_suppression_episodes"] = True
    failures.extend(
        "food_predator_aggregate:" + name
        for name, ok in aggregate_checks.items()
        if not ok
    )
    passed = not failures
    return {
        "scenario": B6_FOOD_PREDATOR_SCENARIO,
        "status": "accepted" if passed else "discarded",
        "passed": passed,
        "aggregate": {
            "threat_exposure_episodes": int(threat_exposure_episodes),
            "threat_priority_or_suppression_episodes": int(
                threat_priority_or_suppression_episodes
            ),
            "full_success_episodes": int(full_success_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }


def b6_corridor_progress_gate_result(
    results: Sequence[dict[str, object]],
) -> dict[str, object]:
    scenario = get_scenario(B6_CORRIDOR_SCENARIO)
    failures: list[str] = []
    episode_results = []
    progress_episodes = 0
    survival_progress_episodes = 0
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
        progress = float(stats.food_distance_delta) > 0.0
        food_progress_over_b5 = (
            int(stats.steps) >= B6_CORRIDOR_BASELINE_STEPS
            and float(stats.food_distance_delta)
            > B6_CORRIDOR_BASELINE_FOOD_DISTANCE_DELTA
        )
        survival_progress = (
            bool(stats.alive)
            or int(stats.steps) >= 16
            or food_progress_over_b5
        )
        if progress:
            progress_episodes += 1
        if survival_progress:
            survival_progress_episodes += 1
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
                        "food_distance_delta": progress,
                        "survival_progress": survival_progress,
                        "food_progress_over_b5": food_progress_over_b5,
                    },
                    "failures": episode_failures,
                    "primitive_violations": primitive_violations,
                },
                "metrics": {
                    **_stats_payload(stats),
                    "food_distance_delta": round(float(stats.food_distance_delta), 6),
                },
                "behavior_score": score_payload,
            }
        )
    aggregate_checks = {
        "progress_episodes": progress_episodes >= 2,
        "survival_progress_episodes": survival_progress_episodes >= 2,
    }
    if full_success_episodes >= 2:
        aggregate_checks = {
            "score_success_episodes": True,
            **aggregate_checks,
        }
        aggregate_checks["progress_episodes"] = True
        aggregate_checks["survival_progress_episodes"] = True
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
            "steps": B6_CORRIDOR_BASELINE_STEPS,
            "food_distance_delta": B6_CORRIDOR_BASELINE_FOOD_DISTANCE_DELTA,
        },
        "aggregate": {
            "progress_episodes": int(progress_episodes),
            "survival_progress_episodes": int(survival_progress_episodes),
            "full_success_episodes": int(full_success_episodes),
            "checks": aggregate_checks,
        },
        "failures": failures,
        "episode_results": episode_results,
    }
