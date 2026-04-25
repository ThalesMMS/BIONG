from __future__ import annotations

from .common import *

def _score_night_rest(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Score night-rest behavior based on deep-shelter occupancy, presence of deep sleep, and reduction in sleep debt.
    
    Builds three behavior checks (deep-shelter occupancy rate, presence of a "DEEP_SLEEP" phase, and sleep-debt reduction) and returns a BehavioralEpisodeScore whose behavior_metrics include: "deep_night_rate", "night_stillness_rate", "sleep_debt_reduction", "sleep_events", "deep_sleep_reached", "sleep_pressure_tick_count", "sleep_priority_rate", "left_shelter", "shelter_exit_tick", "predator_visible_ticks", and "predator_contacts".
    
    Returns:
        BehavioralEpisodeScore: Score with the three night-rest checks and the assembled behavior_metrics described above.
    """
    deep_night_rate = float(stats.night_role_distribution.get("deep", 0.0))
    sleep_debt_reduction = float(max(0.0, NIGHT_REST_INITIAL_SLEEP_DEBT - stats.final_sleep_debt))
    deep_sleep_reached = _trace_any_sleep_phase(trace, "DEEP_SLEEP")
    left_shelter, shelter_exit_tick = _trace_shelter_exit(trace)
    predator_visible_ticks = (
        sum(1 for item in trace if _trace_predator_visible(item))
        if trace
        else 0
    )
    payloads = _trace_action_selection_payloads(trace)
    sleepy_payloads = []
    for payload in payloads:
        sleep_debt = _payload_float(payload, "evidence", "sleep", "sleep_debt")
        fatigue = _payload_float(payload, "evidence", "sleep", "fatigue")
        if sleep_debt is None or fatigue is None:
            continue
        if sleep_debt >= 0.5 and fatigue >= 0.6:
            sleepy_payloads.append(payload)
    sleep_priority_rate = float(
        sum(
            1.0
            for payload in sleepy_payloads
            if _payload_text(payload, "winning_valence") == "sleep"
        ) / max(1, len(sleepy_payloads))
    )
    checks = (
        build_behavior_check(NIGHT_REST_CHECKS[0], passed=deep_night_rate >= 0.95, value=deep_night_rate),
        build_behavior_check(NIGHT_REST_CHECKS[1], passed=deep_sleep_reached, value=deep_sleep_reached),
        build_behavior_check(NIGHT_REST_CHECKS[2], passed=sleep_debt_reduction >= 0.45, value=sleep_debt_reduction),
    )
    full_success = all(check.passed for check in checks)
    behavior_metrics = {
        "deep_night_rate": deep_night_rate,
        "night_stillness_rate": float(stats.night_stillness_rate),
        "sleep_debt_reduction": sleep_debt_reduction,
        "sleep_events": int(stats.sleep_events),
        "deep_sleep_reached": deep_sleep_reached,
        "sleep_pressure_tick_count": len(sleepy_payloads),
        "sleep_priority_rate": sleep_priority_rate,
        "left_shelter": bool(left_shelter),
        "shelter_exit_tick": shelter_exit_tick,
        "predator_visible_ticks": int(predator_visible_ticks),
        "predator_contacts": int(stats.predator_contacts),
    }
    behavior_metrics["failure_mode"] = _classify_night_rest_failure(
        {
            **behavior_metrics,
            "checks_passed": full_success,
        }
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate safe, reproducible rest in deep shelter during the night.",
        checks=checks,
        behavior_metrics=behavior_metrics,
    )

def _score_open_field_foraging(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Score open-field foraging performance by producing progress, viability, and survival checks plus trace-derived diagnostics and a failure-mode classification.

    Computes three behavior checks (food distance progress, foraging viability, and alive), extracts trace-backed diagnostics (initial and min food distances, shelter exit tick, death tick, hunger valence rate, per-tick food-signal strengths), derives weak-scenario diagnostic bands, and classifies a `failure_mode` from the aggregated metrics.

    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute food progress, food eaten, survival, and predator contacts.
        trace (Sequence[Dict[str, object]]): Execution trace used to derive commitment/orientation diagnostics and food-cue signals.

    Returns:
        BehavioralEpisodeScore: A score whose checks cover food progress, foraging viability, and survival. The returned `behavior_metrics` includes at least:
            - `food_distance_delta`, `food_eaten`, `alive`, `predator_contacts`
            - `initial_food_distance`, `min_food_distance_reached`
            - `left_shelter`, `shelter_exit_tick`, `death_tick`
            - `hunger_valence_rate`
            - `initial_food_signal_strength`, `max_food_signal_strength`, `food_signal_tick_rate`
            - `predator_visible_ticks`
          plus weak-scenario diagnostic fields and a computed `failure_mode`.
    """
    food_progress = float(stats.food_distance_delta)
    foraging_viable = bool(stats.food_eaten > 0 or food_progress >= 2.0)
    checks = (
        build_behavior_check(OPEN_FIELD_FORAGING_CHECKS[0], passed=food_progress > 0.0, value=food_progress),
        build_behavior_check(OPEN_FIELD_FORAGING_CHECKS[1], passed=foraging_viable, value=bool(foraging_viable)),
        build_behavior_check(OPEN_FIELD_FORAGING_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    full_success = all(check.passed for check in checks)
    initial_food_distance = _resolve_initial_food_distance(trace)
    food_distances = _trace_food_distances(trace)
    min_food_distance_reached = min(food_distances) if food_distances else None
    left_shelter, shelter_exit_tick = _trace_shelter_exit(trace)
    death_tick = _trace_death_tick(trace)
    hunger_valence_rate = _hunger_valence_rate(trace)
    predator_visible_ticks = (
        sum(1 for item in trace if _trace_predator_visible(item))
        if trace
        else None
    )
    food_signal_strengths = _trace_food_signal_strengths(trace)
    initial_food_signal_strength = (
        food_signal_strengths[0] if food_signal_strengths else 0.0
    )
    max_food_signal_strength = max(food_signal_strengths, default=0.0)
    food_signal_tick_rate = (
        sum(1 for value in food_signal_strengths if value > 0.0)
        / len(food_signal_strengths)
        if food_signal_strengths
        else 0.0
    )
    diagnostics = _weak_scenario_diagnostics(
        food_distance_delta=food_progress,
        food_eaten=int(stats.food_eaten),
        alive=bool(stats.alive),
        predator_contacts=int(stats.predator_contacts),
        full_success=full_success,
    )
    behavior_metrics = {
        "food_distance_delta": food_progress,
        "food_eaten": int(stats.food_eaten),
        "alive": bool(stats.alive),
        "predator_contacts": int(stats.predator_contacts),
        "initial_food_distance": initial_food_distance,
        "min_food_distance_reached": min_food_distance_reached,
        "left_shelter": bool(left_shelter),
        "shelter_exit_tick": shelter_exit_tick,
        "death_tick": death_tick,
        "hunger_valence_rate": hunger_valence_rate,
        "predator_visible_ticks": predator_visible_ticks,
        "initial_food_signal_strength": initial_food_signal_strength,
        "max_food_signal_strength": max_food_signal_strength,
        "food_signal_tick_rate": food_signal_tick_rate,
        **diagnostics,
    }
    behavior_metrics["failure_mode"] = _classify_open_field_foraging_failure(
        {
            **behavior_metrics,
            "checks_passed": full_success,
        }
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate reproducible open-field foraging with an explicit progress metric.",
        checks=checks,
        behavior_metrics=behavior_metrics,
    )

def _score_exposed_day_foraging(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Score an exposed-day foraging episode and attach trace-derived diagnostics and a failure-mode label.

    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to evaluate objective checks and numeric metrics.
        trace (Sequence[Dict[str, object]]): Ordered trace items used to extract shelter/exit, food-distance, and predator-visibility diagnostics.

    Returns:
        BehavioralEpisodeScore: The scenario score including three behavioral checks (food-distance progress, zero predator contacts, survival) and a `behavior_metrics` mapping. `behavior_metrics` contains numeric and boolean summaries from `stats` (e.g., `food_distance_delta`, `food_eaten`, `predator_contacts`, `alive`, `final_health`), trace-derived diagnostics (e.g., `left_shelter`, `shelter_exit_tick`, `peak_food_progress`, `predator_visible_ticks`), weaker diagnostic bands, and a `failure_mode` string produced by the exposed-day failure classifier.
    """
    food_progress = float(stats.food_distance_delta)
    checks = (
        build_behavior_check(EXPOSED_DAY_FORAGING_CHECKS[0], passed=food_progress > 0.0, value=food_progress),
        build_behavior_check(EXPOSED_DAY_FORAGING_CHECKS[1], passed=stats.predator_contacts == 0, value=int(stats.predator_contacts)),
        build_behavior_check(EXPOSED_DAY_FORAGING_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    full_success = all(check.passed for check in checks)
    diagnostics = _weak_scenario_diagnostics(
        food_distance_delta=food_progress,
        food_eaten=int(stats.food_eaten),
        alive=bool(stats.alive),
        predator_contacts=int(stats.predator_contacts),
        full_success=full_success,
    )
    trace_metrics = _extract_exposed_day_trace_metrics(trace)
    behavior_metrics = {
        "food_distance_delta": food_progress,
        "food_eaten": int(stats.food_eaten),
        "predator_contacts": int(stats.predator_contacts),
        "alive": bool(stats.alive),
        "final_health": float(stats.final_health),
        **trace_metrics,
        **diagnostics,
    }
    behavior_metrics["failure_mode"] = _classify_exposed_day_foraging_failure(
        {
            **behavior_metrics,
            "checks_passed": full_success,
        }
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate exposed daytime foraging with a reproducible progress metric.",
        checks=checks,
        behavior_metrics=behavior_metrics,
    )

def _score_food_deprivation(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Compute the scenario score for the food deprivation scenario, producing pass/fail checks and behavior metrics describing recovery and foraging progress.

    Parameters:
        stats: Aggregated episode metrics used to evaluate checks and populate metrics.
        trace: Ordered execution trace used to derive per-tick diagnostics (distances, shelter exit, death tick, action-selection payloads).

    Returns:
        BehavioralEpisodeScore: Contains the scenario objective, per-check results, and a behavior_metrics mapping. behavior_metrics includes:
            - food_eaten: number of food items consumed.
            - food_distance_delta: net change in distance to food (positive indicates progress).
            - hunger_reduction: reduction in hunger from the scenario's initial value (clamped at zero).
            - alive: whether the agent survived the episode.
            - min_food_distance_reached: smallest trace-derived distance to food or None.
            - left_shelter: whether the spider exited the scenario shelter.
            - shelter_exit_tick: tick of first shelter exit, or None.
            - death_tick: tick where the spider first became not alive, or None.
            - hunger_valence_rate: fraction of action-selection ticks where hunger was the winning valence.
            - failure_mode: classifier label describing the dominant failure mode.
            - plus additional diagnostic fields produced by the scenario diagnostic helper.
    """
    hunger_reduction = float(max(0.0, FOOD_DEPRIVATION_INITIAL_HUNGER - stats.final_hunger))
    initial_food_distance = _resolve_initial_food_distance(trace)
    food_distances = _trace_food_distances(trace)
    min_food_distance_reached = min(food_distances) if food_distances else None
    left_shelter, shelter_exit_tick = _trace_shelter_exit(trace)
    death_tick = _trace_death_tick(trace)
    hunger_valence_rate = _hunger_valence_rate(trace)
    commits_to_foraging = bool(left_shelter and hunger_valence_rate >= 0.5)
    checks = (
        build_behavior_check(FOOD_DEPRIVATION_CHECKS[0], passed=(stats.food_eaten > 0 or hunger_reduction >= 0.18), value=hunger_reduction),
        build_behavior_check(FOOD_DEPRIVATION_CHECKS[1], passed=stats.food_distance_delta > 0.0, value=float(stats.food_distance_delta)),
        build_behavior_check(
            FOOD_DEPRIVATION_CHECKS[2],
            passed=commits_to_foraging,
            value={
                "left_shelter": bool(left_shelter),
                "hunger_valence_rate": hunger_valence_rate,
            },
        ),
        build_behavior_check(FOOD_DEPRIVATION_CHECKS[3], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    full_success = all(check.passed for check in checks)
    diagnostics = _weak_scenario_diagnostics(
        food_distance_delta=float(stats.food_distance_delta),
        food_eaten=int(stats.food_eaten),
        hunger_reduction=hunger_reduction,
        alive=bool(stats.alive),
        predator_contacts=int(stats.predator_contacts),
        full_success=full_success,
    )
    behavior_metrics = {
        "food_eaten": int(stats.food_eaten),
        "food_distance_delta": float(stats.food_distance_delta),
        "hunger_reduction": hunger_reduction,
        "alive": bool(stats.alive),
        "min_food_distance_reached": min_food_distance_reached,
        "left_shelter": bool(left_shelter),
        "shelter_exit_tick": shelter_exit_tick,
        "death_tick": death_tick,
        "hunger_valence_rate": hunger_valence_rate,
        **diagnostics,
    }
    behavior_metrics["failure_mode"] = _classify_food_deprivation_failure(
        {
            **behavior_metrics,
            "checks_passed": full_success,
            "initial_food_distance": initial_food_distance,
        }
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate measurable recovery or food progress under homeostatic deprivation.",
        checks=checks,
        behavior_metrics=behavior_metrics,
    )

def _score_sleep_vs_exploration_conflict(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate whether sleep motivation overrides exploration in a nocturnal, low-threat context.

    Analyzes action-selection trace payloads to detect ticks with high sleep evidence and low predator visibility, then builds three behavioral checks:
    - whether sleep valence was selected when sleep pressure was high,
    - whether exploration-related gates (visual and sensory) were suppressed under those sleep-priority ticks,
    - whether the agent exhibited resting behavior (sleep event or sufficient sleep-debt reduction).

    Parameters:
        stats (EpisodeStats): Aggregate statistics collected for the episode (e.g., sleep events, final sleep debt).
        trace (Sequence[Dict[str, object]]): Execution trace of messages and states produced during the episode.

    Returns:
        BehavioralEpisodeScore: Score object containing the scenario objective, the three checks above, and behavior_metrics including:
            - `sleep_pressure_tick_count`: count of ticks meeting sleep-pressure criteria,
            - `sleep_priority_rate`: fraction of those ticks where `sleep` was the winning valence,
            - `mean_visual_gate_under_sleep`: mean visual gate value during sleep-pressure ticks,
            - `sleep_events`: number of sleep events recorded,
            - `sleep_debt_reduction`: amount of sleep-debt reduced from the scenario initial value.
    """
    payloads = _trace_action_selection_payloads(trace)
    sleepy_payloads: list[Dict[str, object]] = []
    for payload in payloads:
        sleep_debt = _payload_float(payload, "evidence", "sleep", "sleep_debt")
        fatigue = _payload_float(payload, "evidence", "sleep", "fatigue")
        predator_visible = _payload_float(payload, "evidence", "threat", "predator_visible")
        if sleep_debt is None or fatigue is None or predator_visible is None:
            continue
        if sleep_debt >= 0.6 and fatigue >= 0.6 and predator_visible < 0.5:
            sleepy_payloads.append(payload)
    sleep_priority_rate = float(
        sum(
            1.0
            for payload in sleepy_payloads
            if _payload_text(payload, "winning_valence") == "sleep"
        ) / max(1, len(sleepy_payloads))
    )
    sleepy_gate_payloads = [
        payload
        for payload in sleepy_payloads
        if _payload_float(payload, "module_gates", "visual_cortex") is not None
        and _payload_float(payload, "module_gates", "sensory_cortex") is not None
    ]
    sleep_priority_gate_payloads = [
        payload
        for payload in sleepy_gate_payloads
        if _payload_text(payload, "winning_valence") == "sleep"
    ]
    exploration_suppressed_rate = float(
        sum(
            1.0
            for payload in sleep_priority_gate_payloads
            if (visual_gate := _payload_float(payload, "module_gates", "visual_cortex")) is not None
            and (sensory_gate := _payload_float(payload, "module_gates", "sensory_cortex")) is not None
            and visual_gate < 0.6
            and sensory_gate < 0.7
        ) / max(1, len(sleep_priority_gate_payloads))
    )
    visual_gates_under_sleep = [
        visual_gate
        for payload in sleep_priority_gate_payloads
        if (visual_gate := _payload_float(payload, "module_gates", "visual_cortex")) is not None
    ]
    sleep_priority = bool(
        len(sleepy_payloads) >= 1 and sleep_priority_rate >= CONFLICT_PASS_RATE
    )
    exploration_suppressed = bool(
        len(sleep_priority_gate_payloads) >= 1
        and exploration_suppressed_rate >= CONFLICT_PASS_RATE
    )
    resting_behavior = bool(
        stats.sleep_events > 0
        or stats.final_sleep_debt <= (SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - 0.12)
    )
    checks = (
        build_behavior_check(SLEEP_VS_EXPLORATION_CONFLICT_CHECKS[0], passed=sleep_priority, value=sleep_priority),
        build_behavior_check(
            SLEEP_VS_EXPLORATION_CONFLICT_CHECKS[1],
            passed=exploration_suppressed,
            value=exploration_suppressed,
        ),
        build_behavior_check(
            SLEEP_VS_EXPLORATION_CONFLICT_CHECKS[2],
            passed=resting_behavior,
            value={
                "sleep_events": int(stats.sleep_events),
                "sleep_debt_reduction": float(
                    max(0.0, SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - stats.final_sleep_debt)
                ),
            },
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate that sleep pressure overrides residual exploration in a safe nighttime context.",
        checks=checks,
        behavior_metrics={
            "sleep_pressure_tick_count": len(sleepy_payloads),
            "sleep_priority_rate": sleep_priority_rate,
            "exploration_suppressed_rate": exploration_suppressed_rate,
            "mean_visual_gate_under_sleep": float(
                sum(visual_gates_under_sleep) / max(1, len(visual_gates_under_sleep))
            ),
            "sleep_events": int(stats.sleep_events),
            "sleep_debt_reduction": float(
                max(0.0, SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - stats.final_sleep_debt)
            ),
        },
    )

__all__ = [
    "_score_exposed_day_foraging",
    "_score_food_deprivation",
    "_score_night_rest",
    "_score_open_field_foraging",
    "_score_sleep_vs_exploration_conflict",
]
