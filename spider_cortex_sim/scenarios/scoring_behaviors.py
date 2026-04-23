from __future__ import annotations

from typing import Dict, Mapping, Sequence

from ..metrics import (
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeStats,
    build_behavior_check,
    build_behavior_score,
)
from .specs import (
    NIGHT_REST_INITIAL_SLEEP_DEBT,
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
)
from .trace import (
    _trace_any_mode,
    _trace_any_sleep_phase,
    _trace_predator_memory_seen,
    _trace_escape_seen,
    _trace_action_selection_payloads,
    _trace_max_predator_threat,
    _trace_dominant_predator_types,
    _float_or_none,
    _int_or_none,
    _trace_food_distances,
    _resolve_initial_food_distance,
    _trace_shelter_exit,
    _trace_death_tick,
    _hunger_valence_rate,
    _trace_predator_visible,
    _trace_corridor_metrics,
    _extract_exposed_day_trace_metrics,
    _trace_food_signal_strengths,
    _payload_float,
    _payload_text,
)

from .scoring_checks import CONFLICT_PASS_RATE, CORRIDOR_GAUNTLET_CHECKS, ENTRANCE_AMBUSH_CHECKS, EXPOSED_DAY_FORAGING_CHECKS, FOOD_DEPRIVATION_CHECKS, FOOD_VS_PREDATOR_CONFLICT_CHECKS, NIGHT_REST_CHECKS, OLFACTORY_AMBUSH_CHECKS, OLFACTORY_AMBUSH_WINDOW_TICKS, OPEN_FIELD_FORAGING_CHECKS, PREDATOR_EDGE_CHECKS, RECOVER_AFTER_FAILED_CHASE_CHECKS, SHELTER_BLOCKADE_CHECKS, SLEEP_VS_EXPLORATION_CONFLICT_CHECKS, TWO_SHELTER_TRADEOFF_CHECKS, VISUAL_HUNTER_OPEN_FIELD_CHECKS, VISUAL_OLFACTORY_PINCER_CHECKS
from .scoring_diagnostics import _classify_corridor_gauntlet_failure, _classify_exposed_day_foraging_failure, _classify_food_deprivation_failure, _classify_night_rest_failure, _classify_open_field_foraging_failure, _classify_two_shelter_tradeoff_failure, _module_response_for_type, _module_share_for_type, _weak_scenario_diagnostics

def _score_night_rest(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Assess night-rest performance for the "night rest" scenario.

    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute occupancy, stillness, sleep events, and final sleep debt.
        trace (Sequence[Dict[str, object]]): Execution trace entries used to detect sleep-phase events (e.g., "DEEP_SLEEP").

    Returns:
        BehavioralEpisodeScore: Score containing three checks (deep-shelter occupancy rate, presence of a DEEP_SLEEP phase, and sleep-debt reduction) and behavior_metrics with keys:
            - "deep_night_rate": proportion of nights spent in deep shelter
            - "night_stillness_rate": recorded stillness rate during night
            - "sleep_debt_reduction": reduction in sleep debt compared to the scenario baseline
            - "sleep_events": count of sleep events
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

def _score_predator_edge(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Evaluate an episode according to the "predator edge" scenario and produce a behavioral score with per-check results and metrics.

    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute detections, alerts, responses, and mode transitions.
        trace (Sequence[Dict[str, object]]): Execution trace of state snapshots used to detect predator memory records and escape events.

    Returns:
        BehavioralEpisodeScore: Score object containing three behavior checks (predator detected, predator memory recorded, predator reacted) and a metrics dictionary with keys:
        - `predator_sightings`: number of predator sightings,
        - `alert_events`: number of alert events,
        - `predator_response_events`: number of predator response events,
        - `predator_mode_transitions`: number of predator mode transitions.
    """
    predator_detected = bool(stats.predator_sightings > 0 or stats.alert_events > 0)
    predator_memory_recorded = _trace_predator_memory_seen(trace)
    predator_reacted = bool(stats.predator_response_events > 0 or stats.predator_mode_transitions > 0 or _trace_escape_seen(trace))
    checks = (
        build_behavior_check(PREDATOR_EDGE_CHECKS[0], passed=predator_detected, value=predator_detected),
        build_behavior_check(PREDATOR_EDGE_CHECKS[1], passed=predator_memory_recorded, value=predator_memory_recorded),
        build_behavior_check(PREDATOR_EDGE_CHECKS[2], passed=predator_reacted, value=predator_reacted),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate predator detection and response during a controlled encounter at the visual edge.",
        checks=checks,
        behavior_metrics={
            "predator_sightings": int(stats.predator_sightings),
            "alert_events": int(stats.alert_events),
            "predator_response_events": int(stats.predator_response_events),
            "predator_mode_transitions": int(stats.predator_mode_transitions),
        },
    )

def _score_entrance_ambush(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Evaluate an episode for the "entrance ambush" scenario and produce per-check results and behavior metrics.

    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to derive checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace entries used to detect predator escape events.

    Returns:
        BehavioralEpisodeScore: Score containing three behavior checks (survival, zero predator contacts, shelter safety)
        and metrics: `survival`, `predator_contacts`, `night_shelter_occupancy_rate`, and `predator_escapes`.
    """
    shelter_safety = float(max(stats.night_shelter_occupancy_rate, 1.0 if _trace_escape_seen(trace) else 0.0))
    checks = (
        build_behavior_check(ENTRANCE_AMBUSH_CHECKS[0], passed=bool(stats.alive), value=bool(stats.alive)),
        build_behavior_check(ENTRANCE_AMBUSH_CHECKS[1], passed=stats.predator_contacts == 0, value=int(stats.predator_contacts)),
        build_behavior_check(ENTRANCE_AMBUSH_CHECKS[2], passed=shelter_safety >= 0.75, value=shelter_safety),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate survival and preserved safety under an ambush at the shelter entrance.",
        checks=checks,
        behavior_metrics={
            "survival": bool(stats.alive),
            "predator_contacts": int(stats.predator_contacts),
            "night_shelter_occupancy_rate": float(stats.night_shelter_occupancy_rate),
            "predator_escapes": int(stats.predator_escapes),
        },
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

def _score_shelter_blockade(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Assess the episode for the shelter-blockade scenario and build a BehavioralEpisodeScore.

    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to evaluate survival, contacts, and occupancy.
        trace (Sequence[Dict[str, object]]): Execution trace states inspected for predator escape events that influence shelter safety.

    Returns:
        BehavioralEpisodeScore: Score object containing three behavior checks (alive, zero predator contacts, shelter safety threshold) and metrics:
            - night_shelter_occupancy_rate (float)
            - predator_escapes (int)
            - predator_contacts (int)
            - alive (bool)
    """
    safety_score = float(max(stats.night_shelter_occupancy_rate, 1.0 if _trace_escape_seen(trace) else 0.0))
    checks = (
        build_behavior_check(SHELTER_BLOCKADE_CHECKS[0], passed=bool(stats.alive), value=bool(stats.alive)),
        build_behavior_check(SHELTER_BLOCKADE_CHECKS[1], passed=stats.predator_contacts == 0, value=int(stats.predator_contacts)),
        build_behavior_check(SHELTER_BLOCKADE_CHECKS[2], passed=safety_score >= 0.75, value=safety_score),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate safe, reproducible behavior when the shelter entrance is blocked.",
        checks=checks,
        behavior_metrics={
            "night_shelter_occupancy_rate": float(stats.night_shelter_occupancy_rate),
            "predator_escapes": int(stats.predator_escapes),
            "predator_contacts": int(stats.predator_contacts),
            "alive": bool(stats.alive),
        },
    )

def _score_recover_after_failed_chase(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Assess whether the predator entered RECOVER and subsequently returned to WAIT after a failed chase.

    Builds three behavior checks (entered RECOVER, returned to WAIT, spider alive) and aggregates related behavior metrics.

    Returns:
        BehavioralEpisodeScore: Score object containing the three checks and a `behavior_metrics` mapping with keys:
            - "entered_recover" (bool)
            - "returned_to_wait" (bool)
            - "predator_mode_transitions" (int)
            - "alive" (bool)
    """
    entered_recover = _trace_any_mode(trace, "RECOVER")
    returned_to_wait = _trace_any_mode(trace, "WAIT")
    checks = (
        build_behavior_check(RECOVER_AFTER_FAILED_CHASE_CHECKS[0], passed=entered_recover, value=entered_recover),
        build_behavior_check(RECOVER_AFTER_FAILED_CHASE_CHECKS[1], passed=returned_to_wait, value=returned_to_wait),
        build_behavior_check(RECOVER_AFTER_FAILED_CHASE_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate the predator's deterministic transition into RECOVER and WAIT after a failed chase.",
        checks=checks,
        behavior_metrics={
            "entered_recover": entered_recover,
            "returned_to_wait": returned_to_wait,
            "predator_mode_transitions": int(stats.predator_mode_transitions),
            "alive": bool(stats.alive),
        },
    )

def _score_corridor_gauntlet(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Evaluate corridor gauntlet episode performance with trace-backed diagnostics.

    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to compute checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace used to derive shelter exit, predator visibility, progress, and death diagnostics.

    Returns:
        BehavioralEpisodeScore: Score object containing three behavior checks (food progress, zero predator contacts, alive)
        and metrics including `failure_mode`, `left_shelter`, `shelter_exit_tick`, `predator_visible_ticks`, `peak_food_progress`, and `death_tick`.
    """
    food_progress = float(stats.food_distance_delta)
    checks = (
        build_behavior_check(CORRIDOR_GAUNTLET_CHECKS[0], passed=food_progress > 0.0, value=food_progress),
        build_behavior_check(CORRIDOR_GAUNTLET_CHECKS[1], passed=stats.predator_contacts == 0, value=int(stats.predator_contacts)),
        build_behavior_check(CORRIDOR_GAUNTLET_CHECKS[2], passed=bool(stats.alive), value=bool(stats.alive)),
    )
    full_success = all(check.passed for check in checks)
    diagnostics = _weak_scenario_diagnostics(
        food_distance_delta=food_progress,
        food_eaten=int(stats.food_eaten),
        alive=bool(stats.alive),
        predator_contacts=int(stats.predator_contacts),
        full_success=full_success,
    )
    trace_metrics = _trace_corridor_metrics(trace)
    behavior_metrics = {
        "food_distance_delta": food_progress,
        "predator_contacts": int(stats.predator_contacts),
        "alive": bool(stats.alive),
        "predator_mode_transitions": int(stats.predator_mode_transitions),
        **trace_metrics,
        **diagnostics,
    }
    behavior_metrics["failure_mode"] = _classify_corridor_gauntlet_failure(
        stats,
        trace_metrics,
        full_success,
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate narrow-corridor progress without relying only on aggregated reward.",
        checks=checks,
        behavior_metrics=behavior_metrics,
    )

def _score_two_shelter_tradeoff(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Compute the behavioral score and check results for the two-shelter tradeoff scenario.

    Builds three checks: survival (alive), progress toward food or shelter occupancy (passes if food distance improved or occupancy >= 0.9), and high shelter occupancy (occupancy >= 0.9). Returns a BehavioralEpisodeScore containing those checks and metrics derived from EpisodeStats.

    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to evaluate checks and metrics.
        trace (Sequence[Dict[str, object]]): Execution trace (not used by this scorer).

    Returns:
        BehavioralEpisodeScore: Score bundle including the three check results and the following behavior_metrics:
            - "food_distance_delta" (float): change in distance to food.
            - "night_shelter_occupancy_rate" (float): fraction of nights spent in the alternative shelter.
            - "alive" (bool): whether the agent finished the episode alive.
            - "final_health" (float): agent's final health value.
    """
    progress_score = float(max(stats.food_distance_delta, stats.night_shelter_occupancy_rate))
    checks = (
        build_behavior_check(TWO_SHELTER_TRADEOFF_CHECKS[0], passed=bool(stats.alive), value=bool(stats.alive)),
        build_behavior_check(TWO_SHELTER_TRADEOFF_CHECKS[1], passed=(stats.food_distance_delta > 0.0 or stats.night_shelter_occupancy_rate >= 0.9), value=progress_score),
        build_behavior_check(TWO_SHELTER_TRADEOFF_CHECKS[2], passed=stats.night_shelter_occupancy_rate >= 0.9, value=float(stats.night_shelter_occupancy_rate)),
    )
    full_success = all(check.passed for check in checks)
    initial_food_distance = _resolve_initial_food_distance(trace)
    food_distances = _trace_food_distances(trace)
    min_food_distance_reached = min(food_distances) if food_distances else None
    left_shelter, shelter_exit_tick = _trace_shelter_exit(trace)
    predator_visible_ticks = (
        sum(1 for item in trace if _trace_predator_visible(item))
        if trace
        else 0
    )
    behavior_metrics = {
        "food_distance_delta": float(stats.food_distance_delta),
        "initial_food_distance": initial_food_distance,
        "min_food_distance_reached": min_food_distance_reached,
        "night_shelter_occupancy_rate": float(stats.night_shelter_occupancy_rate),
        "left_shelter": bool(left_shelter),
        "shelter_exit_tick": shelter_exit_tick,
        "predator_visible_ticks": int(predator_visible_ticks),
        "predator_contacts": int(stats.predator_contacts),
        "alive": bool(stats.alive),
        "final_health": float(stats.final_health),
    }
    behavior_metrics["failure_mode"] = _classify_two_shelter_tradeoff_failure(
        {
            **behavior_metrics,
            "checks_passed": full_success,
        }
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate the behavioral trade-off between progress toward food and safe occupancy of an alternative shelter.",
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

def _score_visual_olfactory_pincer(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate the agent's response when visual and olfactory predator threats occur simultaneously.

    Builds three checks (dual detection, type-specific module engagement, and surviving without predator contact) and returns a BehaviorEpisodeScore populated with those checks and behavior_metrics that include visual/olfactory threat peaks, dominant predator types seen, per-type module share values, predator contact and response maps, and the final alive flag.

    Returns:
        BehavioralEpisodeScore: Score object containing the composed checks and a behavior_metrics mapping with keys such as
        "visual_predator_threat_peak", "olfactory_predator_threat_peak", "dominant_predator_types_seen",
        per-type cortex share values, predator contact/response maps, and "alive".
    """
    visual_peak = _trace_max_predator_threat(trace, "visual")
    olfactory_peak = _trace_max_predator_threat(trace, "olfactory")
    dominant_types_seen = sorted(_trace_dominant_predator_types(trace))
    visual_response = _module_response_for_type(stats, "visual")
    olfactory_response = _module_response_for_type(stats, "olfactory")
    visual_visual_share = _module_share_for_type(stats, "visual", "visual_cortex")
    visual_sensory_share = _module_share_for_type(stats, "visual", "sensory_cortex")
    olfactory_visual_share = _module_share_for_type(stats, "olfactory", "visual_cortex")
    olfactory_sensory_share = _module_share_for_type(stats, "olfactory", "sensory_cortex")
    dual_detected = bool(
        (visual_peak > 0.0 and olfactory_peak > 0.0)
        or (
            sum(float(value) for value in visual_response.values()) > 0.0
            and sum(float(value) for value in olfactory_response.values()) > 0.0
        )
    )
    type_specific_response = bool(
        dual_detected
        and visual_visual_share > 0.0
        and olfactory_sensory_share > 0.0
        and visual_visual_share > visual_sensory_share
        and olfactory_sensory_share > olfactory_visual_share
    )
    survives_cleanly = bool(stats.alive and stats.predator_contacts == 0)
    checks = (
        build_behavior_check(
            VISUAL_OLFACTORY_PINCER_CHECKS[0],
            passed=dual_detected,
            value={"visual_peak": visual_peak, "olfactory_peak": olfactory_peak},
        ),
        build_behavior_check(
            VISUAL_OLFACTORY_PINCER_CHECKS[1],
            passed=type_specific_response,
            value={
                "visual_visual_cortex": visual_visual_share,
                "visual_sensory_cortex": visual_sensory_share,
                "olfactory_visual_cortex": olfactory_visual_share,
                "olfactory_sensory_cortex": olfactory_sensory_share,
            },
        ),
        build_behavior_check(
            VISUAL_OLFACTORY_PINCER_CHECKS[2],
            passed=survives_cleanly,
            value={
                "alive": bool(stats.alive),
                "predator_contacts": int(stats.predator_contacts),
            },
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Measure whether the agent distinguishes simultaneous visual and olfactory predator pressure.",
        checks=checks,
        behavior_metrics={
            "visual_predator_threat_peak": visual_peak,
            "olfactory_predator_threat_peak": olfactory_peak,
            "dominant_predator_types_seen": dominant_types_seen,
            "visual_visual_cortex_share": visual_visual_share,
            "visual_sensory_cortex_share": visual_sensory_share,
            "olfactory_visual_cortex_share": olfactory_visual_share,
            "olfactory_sensory_cortex_share": olfactory_sensory_share,
            "module_response_by_predator_type": {
                "visual": dict(visual_response),
                "olfactory": dict(olfactory_response),
            },
            "predator_contacts_by_type": dict(stats.predator_contacts_by_type),
            "predator_response_latency_by_type": dict(stats.predator_response_latency_by_type),
            "alive": bool(stats.alive),
        },
    )

def _score_olfactory_ambush(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate an olfactory ambush scenario for whether olfactory threat drives sensory-cortex-led responses and safe survival.

    Parameters:
        stats (EpisodeStats): Aggregated episode statistics (module responses, contacts, survival, etc.).
        trace (Sequence[Dict[str, object]]): Ordered trace records from the episode used to extract time-series diagnostics.

    Returns:
        BehavioralEpisodeScore: Scoring object containing pass/fail checks, derived behavior metrics, and failure-mode diagnostics for the olfactory-ambush scenario.
    """
    initial_trace = trace[:OLFACTORY_AMBUSH_WINDOW_TICKS]
    visual_peak = _trace_max_predator_threat(trace, "visual")
    visual_peak_initial = _trace_max_predator_threat(initial_trace, "visual")
    olfactory_peak = _trace_max_predator_threat(trace, "olfactory")
    olfactory_peak_initial = _trace_max_predator_threat(initial_trace, "olfactory")
    sensory_share = _module_share_for_type(stats, "olfactory", "sensory_cortex")
    visual_share = _module_share_for_type(stats, "olfactory", "visual_cortex")
    predator_visible_ticks = sum(1 for item in trace if _trace_predator_visible(item))
    predator_visible_ticks_initial = sum(
        1 for item in initial_trace if _trace_predator_visible(item)
    )
    olfactory_detected = bool(
        olfactory_peak_initial > 0.0
        or sum(
            float(value)
            for value in _module_response_for_type(stats, "olfactory").values()
        )
        > 0.0
    )
    sensory_engaged = bool(
        olfactory_detected
        and sensory_share > 0.0
        and sensory_share > visual_share
        and (not initial_trace or visual_peak_initial <= 0.05)
    )
    survives_cleanly = bool(stats.alive and stats.predator_contacts == 0)
    checks = (
        build_behavior_check(
            OLFACTORY_AMBUSH_CHECKS[0],
            passed=olfactory_detected
            and (not initial_trace or predator_visible_ticks_initial == 0),
            value={
                "olfactory_peak": olfactory_peak,
                "olfactory_peak_initial": olfactory_peak_initial,
                "visual_peak": visual_peak,
                "visual_peak_initial": visual_peak_initial,
                "predator_visible_ticks": predator_visible_ticks,
                "predator_visible_ticks_initial": predator_visible_ticks_initial,
            },
        ),
        build_behavior_check(
            OLFACTORY_AMBUSH_CHECKS[1],
            passed=sensory_engaged,
            value={
                "sensory_cortex_share": sensory_share,
                "visual_cortex_share": visual_share,
            },
        ),
        build_behavior_check(
            OLFACTORY_AMBUSH_CHECKS[2],
            passed=survives_cleanly,
            value={
                "alive": bool(stats.alive),
                "predator_contacts": int(stats.predator_contacts),
            },
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Measure whether hidden olfactory threat recruits sensory-cortex-led response without visual contact.",
        checks=checks,
        behavior_metrics={
            "olfactory_predator_threat_peak": olfactory_peak,
            "olfactory_predator_threat_peak_initial": olfactory_peak_initial,
            "visual_predator_threat_peak": visual_peak,
            "visual_predator_threat_peak_initial": visual_peak_initial,
            "predator_visible_ticks": predator_visible_ticks,
            "predator_visible_ticks_initial": predator_visible_ticks_initial,
            "olfactory_sensory_cortex_share": sensory_share,
            "olfactory_visual_cortex_share": visual_share,
            "predator_contacts_by_type": dict(stats.predator_contacts_by_type),
            "alive": bool(stats.alive),
        },
    )

def _score_visual_hunter_open_field(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate episode behavior in an open-field scenario with a fast visual predator.

    Builds three checks that detect (1) whether a visual predator signal was present, (2) whether the visual-cortex module dominated the sensory module for the visual predator, and (3) whether the spider survived without predator contacts. Returns a BehaviorScore containing those checks and metrics including the visual threat peak, module share values, per-type contact and response-latency maps, and final alive status.

    Returns:
        BehavioralEpisodeScore: Aggregated score, checks, and behavior metrics for the visual-hunter open-field scenario.
    """
    visual_peak = _trace_max_predator_threat(trace, "visual")
    visual_share = _module_share_for_type(stats, "visual", "visual_cortex")
    sensory_share = _module_share_for_type(stats, "visual", "sensory_cortex")
    visual_detected = bool(
        visual_peak > 0.0
        or sum(
            float(value)
            for value in _module_response_for_type(stats, "visual").values()
        )
        > 0.0
    )
    visual_engaged = bool(
        visual_detected
        and visual_share > 0.0
        and visual_share > sensory_share
    )
    survives_cleanly = bool(stats.alive and stats.predator_contacts == 0)
    checks = (
        build_behavior_check(
            VISUAL_HUNTER_OPEN_FIELD_CHECKS[0],
            passed=visual_detected,
            value=visual_peak,
        ),
        build_behavior_check(
            VISUAL_HUNTER_OPEN_FIELD_CHECKS[1],
            passed=visual_engaged,
            value={
                "visual_cortex_share": visual_share,
                "sensory_cortex_share": sensory_share,
            },
        ),
        build_behavior_check(
            VISUAL_HUNTER_OPEN_FIELD_CHECKS[2],
            passed=survives_cleanly,
            value={
                "alive": bool(stats.alive),
                "predator_contacts": int(stats.predator_contacts),
            },
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Measure whether open-field visual threat recruits visual-cortex-led predator response.",
        checks=checks,
        behavior_metrics={
            "visual_predator_threat_peak": visual_peak,
            "visual_visual_cortex_share": visual_share,
            "visual_sensory_cortex_share": sensory_share,
            "predator_contacts_by_type": dict(stats.predator_contacts_by_type),
            "predator_response_latency_by_type": dict(stats.predator_response_latency_by_type),
            "alive": bool(stats.alive),
        },
    )

def _score_food_vs_predator_conflict(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate whether predator threat overrides food-seeking and whether the agent survives without predator contact.

    Identifies action-selection payloads that signify a dangerous predator (predator visibility ≥ 0.5 and either proximity ≥ 0.3 or certainty ≥ 0.35), then computes three checks:
    - whether the selected valence favors `threat`,
    - whether hunger gating is suppressed under threat,
    - whether the agent survives with zero predator contacts.

    The returned score includes these checks and behavior metrics: `danger_tick_count`, `threat_priority_rate`, `mean_hunger_gate_under_threat`, `predator_contacts`, and `alive`.

    Parameters:
        stats (EpisodeStats): Episode summary statistics used for survival and contact checks.
        trace (Sequence[Dict[str, object]]): Execution trace from which action-selection payloads are extracted.

    Returns:
        BehavioralEpisodeScore: A score object containing the scenario objective, the three checks, and the computed behavior metrics.
    """
    payloads = _trace_action_selection_payloads(trace)
    dangerous_payloads: list[Dict[str, object]] = []
    for payload in payloads:
        predator_visible = _payload_float(payload, "evidence", "threat", "predator_visible")
        predator_proximity = _payload_float(payload, "evidence", "threat", "predator_proximity")
        predator_certainty = _payload_float(payload, "evidence", "threat", "predator_certainty")
        if predator_visible is None or predator_proximity is None or predator_certainty is None:
            continue
        if predator_visible >= 0.5 and (
            predator_proximity >= 0.3 or predator_certainty >= 0.35
        ):
            dangerous_payloads.append(payload)
    threat_priority_rate = float(
        sum(
            1.0
            for payload in dangerous_payloads
            if _payload_text(payload, "winning_valence") == "threat"
        ) / max(1, len(dangerous_payloads))
    )
    dangerous_hunger_gates = [
        hunger_gate
        for payload in dangerous_payloads
        if (hunger_gate := _payload_float(payload, "module_gates", "hunger_center")) is not None
    ]
    threat_wins = sum(
        1
        for payload in dangerous_payloads
        if _payload_text(payload, "winning_valence") == "threat"
    )
    foraging_suppressed_rate = float(
        sum(
            1.0
            for payload in dangerous_payloads
            if _payload_text(payload, "winning_valence") == "threat"
            and (hunger_gate := _payload_float(payload, "module_gates", "hunger_center")) is not None
            and hunger_gate < 0.5
        ) / max(1, threat_wins)
    )
    threat_priority = bool(
        len(dangerous_payloads) >= 1 and threat_priority_rate >= CONFLICT_PASS_RATE
    )
    foraging_suppressed = bool(
        len(dangerous_hunger_gates) >= 1
        and foraging_suppressed_rate >= CONFLICT_PASS_RATE
    )
    survives_without_contact = bool(stats.alive and stats.predator_contacts == 0)
    checks = (
        build_behavior_check(FOOD_VS_PREDATOR_CONFLICT_CHECKS[0], passed=threat_priority, value=threat_priority),
        build_behavior_check(FOOD_VS_PREDATOR_CONFLICT_CHECKS[1], passed=foraging_suppressed, value=foraging_suppressed),
        build_behavior_check(
            FOOD_VS_PREDATOR_CONFLICT_CHECKS[2],
            passed=survives_without_contact,
            value={"alive": bool(stats.alive), "predator_contacts": int(stats.predator_contacts)},
        ),
    )
    return build_behavior_score(
        stats=stats,
        objective="Validate that threat overrides food drive when both compete at the same moment.",
        checks=checks,
        behavior_metrics={
            "danger_tick_count": len(dangerous_payloads),
            "threat_priority_rate": threat_priority_rate,
            "foraging_suppressed_rate": foraging_suppressed_rate,
            "mean_hunger_gate_under_threat": float(
                sum(dangerous_hunger_gates) / max(1, len(dangerous_hunger_gates))
            ),
            "predator_contacts": int(stats.predator_contacts),
            "alive": bool(stats.alive),
        },
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
