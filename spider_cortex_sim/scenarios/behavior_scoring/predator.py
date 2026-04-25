from __future__ import annotations

from typing import Dict, Sequence

from .common import (
    CORRIDOR_GAUNTLET_CHECKS,
    ENTRANCE_AMBUSH_CHECKS,
    OLFACTORY_AMBUSH_CHECKS,
    OLFACTORY_AMBUSH_WINDOW_TICKS,
    PREDATOR_EDGE_CHECKS,
    RECOVER_AFTER_FAILED_CHASE_CHECKS,
    SHELTER_BLOCKADE_CHECKS,
    TWO_SHELTER_TRADEOFF_CHECKS,
    VISUAL_HUNTER_OPEN_FIELD_CHECKS,
    VISUAL_OLFACTORY_PINCER_CHECKS,
    BehavioralEpisodeScore,
    EpisodeStats,
    _classify_corridor_gauntlet_failure,
    _classify_two_shelter_tradeoff_failure,
    _module_response_for_type,
    _module_share_for_type,
    _resolve_initial_food_distance,
    _trace_any_mode,
    _trace_corridor_metrics,
    _trace_dominant_predator_types,
    _trace_escape_seen,
    _trace_food_distances,
    _trace_max_predator_threat,
    _trace_predator_memory_seen,
    _trace_predator_visible,
    _trace_shelter_exit,
    _weak_scenario_diagnostics,
    build_behavior_check,
    build_behavior_score,
)

def _score_predator_edge(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Score an episode for the "predator edge" scenario by evaluating detection, memory recording, and reaction checks.
    
    Returns:
        BehavioralEpisodeScore: A score object containing three behavior checks (predator detected, predator memory recorded, predator reacted) and a metrics dictionary with keys:
            - `predator_sightings`: int number of predator sightings
            - `alert_events`: int number of alert events
            - `predator_response_events`: int number of predator response events
            - `predator_mode_transitions`: int number of predator mode transitions
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
    Score an episode for the entrance-ambush scenario and produce per-check results and behavior metrics.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics for the episode.
        trace (Sequence[Dict[str, object]]): Per-tick trace entries used for trace-derived detections.
    
    Returns:
        BehavioralEpisodeScore: Contains three behavior checks (survival, zero predator contacts, shelter safety)
        and a behavior_metrics mapping with keys: `survival`, `predator_contacts`, `night_shelter_occupancy_rate`, and `predator_escapes`.
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

def _score_shelter_blockade(stats: EpisodeStats, trace: Sequence[Dict[str, object]]) -> BehavioralEpisodeScore:
    """
    Score an episode for the shelter-blockade scenario.
    
    Builds three behavior checks: alive, zero predator contacts, and a shelter safety threshold derived from night shelter occupancy or observed predator escapes.
    
    Parameters:
        stats (EpisodeStats): Aggregated episode statistics used to evaluate survival, predator contacts, escapes, and occupancy.
        trace (Sequence[Dict[str, object]]): Per-tick trace inspected for predator escape events that affect shelter safety.
    
    Returns:
        BehavioralEpisodeScore: Composed score containing the three checks and a metrics dictionary with:
            - night_shelter_occupancy_rate (float): Final occupancy rate for night shelters.
            - predator_escapes (int): Count of predator escape events.
            - predator_contacts (int): Count of predator contacts.
            - alive (bool): Whether the agent survived the episode.
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

def _trace_wait_after_recover(trace: Sequence[Dict[str, object]]) -> bool:
    """
    Detect whether a trace contains a "RECOVER" lizard_mode followed later by a "WAIT" lizard_mode.
    
    Parameters:
        trace (Sequence[Dict[str, object]]): Per-tick trace entries; each entry may include a "state" dict with a "lizard_mode" key.
    
    Returns:
        True if a "RECOVER" mode is seen and a subsequent tick has mode "WAIT", False otherwise.
    """
    saw_recover = False
    for item in trace:
        state = item.get("state")
        if not isinstance(state, dict):
            continue
        mode = state.get("lizard_mode")
        if saw_recover and mode == "WAIT":
            return True
        if mode == "RECOVER":
            saw_recover = True
    return False


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
    returned_to_wait = _trace_wait_after_recover(trace)
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
    Score the two-shelter tradeoff scenario and produce its behavioral checks and metrics.
    
    Returns:
        BehavioralEpisodeScore: Score bundle containing three behavior checks (survival, progress toward food or shelter occupancy, and high shelter occupancy) and a `behavior_metrics` mapping with at least the following entries:
            - "food_distance_delta" (float): change in distance to food.
            - "initial_food_distance" (float | None): resolved initial distance to food from the trace, or None if unavailable.
            - "min_food_distance_reached" (float | None): minimum food distance observed in the trace, or None if unavailable.
            - "night_shelter_occupancy_rate" (float): fraction of nights spent in the alternative shelter.
            - "left_shelter" (bool): whether the agent left the shelter during the episode.
            - "shelter_exit_tick" (int | None): tick index when the shelter was exited, or None.
            - "predator_visible_ticks" (int): number of ticks a predator was visible.
            - "predator_contacts" (int): total predator contact count.
            - "alive" (bool): whether the agent finished the episode alive.
            - "final_health" (float): agent's final health value.
            - "failure_mode" (str): classified failure mode for the scenario.
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

def _score_visual_olfactory_pincer(
    stats: EpisodeStats,
    trace: Sequence[Dict[str, object]],
) -> BehavioralEpisodeScore:
    """
    Evaluate whether the agent distinguishes and responds to simultaneous visual and olfactory predator threats.
    
    Builds three behavior checks: dual detection of visual and olfactory threats, type-specific module engagement (visual cortex for visual threats and sensory cortex for olfactory threats), and surviving without predator contact. Populates behavior_metrics with threat peaks, dominant predator types, per-modality module shares and response maps, predator contact/latency maps, and the final alive flag.
    
    Returns:
        BehavioralEpisodeScore: Score containing the composed checks and a behavior_metrics mapping with keys including
        "visual_predator_threat_peak", "olfactory_predator_threat_peak", "dominant_predator_types_seen",
        "visual_visual_cortex_share", "visual_sensory_cortex_share", "olfactory_visual_cortex_share",
        "olfactory_sensory_cortex_share", "module_response_by_predator_type", "predator_contacts_by_type",
        "predator_response_latency_by_type", and "alive".
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

__all__ = [
    "_score_corridor_gauntlet",
    "_score_entrance_ambush",
    "_score_olfactory_ambush",
    "_score_predator_edge",
    "_score_recover_after_failed_chase",
    "_score_shelter_blockade",
    "_score_two_shelter_tradeoff",
    "_score_visual_hunter_open_field",
    "_score_visual_olfactory_pincer",
]
