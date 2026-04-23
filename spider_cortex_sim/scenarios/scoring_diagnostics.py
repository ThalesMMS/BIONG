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

def _food_approached(metrics: Mapping[str, object]) -> bool:
    """
    Determine whether the agent made an approach toward the target food during the episode.

    Considers `min_food_distance_reached` relative to `initial_food_distance` when both are present; if either is missing, falls back to the sign of `food_distance_delta` to infer approach.

    Parameters:
        metrics (Mapping[str, object]): Episode-derived metrics. Expected keys (optional):
            `min_food_distance_reached`, `initial_food_distance`, `food_distance_delta`.

    Returns:
        bool: `True` if the agent made a measurable approach toward the food, `False` otherwise.
    """
    min_food_distance = _float_or_none(metrics.get("min_food_distance_reached"))
    initial_food_distance = _float_or_none(metrics.get("initial_food_distance"))
    if min_food_distance is not None and initial_food_distance is not None:
        return min_food_distance < initial_food_distance
    food_distance_delta = _float_or_none(metrics.get("food_distance_delta"))
    return food_distance_delta is not None and food_distance_delta > 0.0

def _classify_food_deprivation_failure(metrics: Mapping[str, object]) -> str:
    """
    Assigns a failure or outcome label for a food-deprivation episode from derived diagnostic metrics.

    Parameters:
        metrics (Mapping[str, object]): Derived episode metrics used to classify outcome. Recognized keys:
            - checks_passed: truthy indicates the scenario checks were satisfied.
            - left_shelter: whether the agent left its shelter.
            - hunger_valence_rate: fraction of action-selection ticks favoring hunger (numeric).
            - min_food_distance_reached: minimum observed distance to the target food (numeric).
            - initial_food_distance: initial distance to the target food (numeric).
            - food_distance_delta: change in food distance; positive indicates progress toward food (numeric).
            - alive: whether the agent survived the episode.
            - food_eaten: number of food items consumed (int).

    Returns:
        str: One of the outcome labels:
            - "success": scenario checks passed.
            - "no_commitment": agent did not leave shelter or showed insufficient hunger-driven commitment.
            - "orientation_failure": agent left shelter but did not approach the target food.
            - "timing_failure": agent died before consuming food or died despite eating without meeting checks.
            - "scoring_mismatch": agent survived and approached/acted but failed the scenario checks.
    """
    if metrics.get("checks_passed", False):
        return "success"

    left_shelter = bool(metrics.get("left_shelter", False))
    hunger_valence_rate = _float_or_none(metrics.get("hunger_valence_rate")) or 0.0
    approached_food = _food_approached(metrics)

    if not left_shelter or hunger_valence_rate < 0.5:
        return "no_commitment"
    if not approached_food:
        return "orientation_failure"

    alive = bool(metrics.get("alive", False))
    food_eaten = _int_or_none(metrics.get("food_eaten")) or 0
    if not alive and food_eaten <= 0:
        return "timing_failure"
    if alive:
        return "scoring_mismatch"
    return "timing_failure"

def _classify_open_field_foraging_failure(metrics: Mapping[str, object]) -> str:
    """
    Classify an open-field foraging episode into a single outcome label based on diagnostic metrics.

    Parameters:
        metrics (Mapping[str, object]): Diagnostic values computed for an episode. Recognized keys include:
            - checks_passed: truthy when scenario checks indicate full success.
            - left_shelter: whether the agent left shelter.
            - hunger_valence_rate: numeric measure of hunger-driven action preference.
            - initial_food_signal_strength, max_food_signal_strength: numeric food-cue strength estimates.
            - initial_food_distance, min_food_distance_reached, food_distance_delta: numeric distance/progress metrics.
            - food_eaten: count of food items consumed.
            - alive: final alive status.
        Missing or unknown keys are treated as absent/zero where appropriate.

    Returns:
        str: One of:
            - "success": checks indicate full success.
            - "never_left_shelter": agent did not leave shelter.
            - "no_hunger_commitment": left shelter but hunger-driven commitment was insufficient.
            - "left_without_food_signal": left shelter with no detectable food cue.
            - "orientation_failure": food cues were present but the agent did not approach food.
            - "progressed_then_died": the agent approached food but died before completion.
            - "stall": the agent left shelter, trace-confirmed no predator visibility, made no food progress, and survived.
            - "scoring_mismatch": outcome did not match any above category (ambiguous/unexpected).
    """
    if metrics.get("checks_passed", False):
        return "success"

    if not bool(metrics.get("left_shelter", False)):
        return "never_left_shelter"

    hunger_valence_rate = _float_or_none(metrics.get("hunger_valence_rate")) or 0.0
    if hunger_valence_rate < 0.5:
        return "no_hunger_commitment"

    initial_food_signal_strength = _float_or_none(
        metrics.get("initial_food_signal_strength")
    ) or 0.0
    if initial_food_signal_strength <= 0.0:
        return "left_without_food_signal"

    food_eaten = _int_or_none(metrics.get("food_eaten")) or 0
    approached_food = food_eaten > 0 or _food_approached(metrics)
    predator_visible_ticks = _int_or_none(metrics.get("predator_visible_ticks"))
    alive = bool(metrics.get("alive", False))

    if (
        not approached_food
        and "predator_visible_ticks" in metrics
        and predator_visible_ticks == 0
        and alive
    ):
        return "stall"
    if not approached_food:
        return "orientation_failure"
    if not alive:
        return "progressed_then_died"
    return "scoring_mismatch"


def _classify_night_rest_failure(metrics: Mapping[str, object]) -> str:
    """
    Assign a night-rest failure label from trace-backed rest diagnostics.
    """
    if metrics.get("checks_passed", False):
        return "success"

    if bool(metrics.get("predator_contacts", 0)) or (_int_or_none(metrics.get("predator_visible_ticks")) or 0) > 0:
        return "predator_interrupted"

    if bool(metrics.get("left_shelter", False)):
        return "leaves_shelter"

    deep_sleep_reached = bool(metrics.get("deep_sleep_reached", False))
    sleep_debt_reduction = _float_or_none(metrics.get("sleep_debt_reduction")) or 0.0
    sleep_events = _int_or_none(metrics.get("sleep_events")) or 0
    sleep_priority_rate = _float_or_none(metrics.get("sleep_priority_rate")) or 0.0

    if deep_sleep_reached and sleep_debt_reduction < 0.45:
        return "deep_without_recovery"
    if deep_sleep_reached or sleep_events > 0 or sleep_priority_rate >= 0.5:
        return "settles_but_not_deep"
    return "never_settles"


def _classify_two_shelter_tradeoff_failure(metrics: Mapping[str, object]) -> str:
    """
    Assign a two-shelter tradeoff failure label from movement, shelter, and threat diagnostics.
    """
    if metrics.get("checks_passed", False):
        return "success"

    if not bool(metrics.get("left_shelter", False)):
        return "frozen_in_left_shelter"

    predator_contacts = _int_or_none(metrics.get("predator_contacts")) or 0
    if predator_contacts > 0:
        return "central_chokepoint_contact"

    if not bool(metrics.get("alive", False)) and _food_approached(metrics):
        return "progress_then_dies"

    night_shelter_occupancy_rate = (
        _float_or_none(metrics.get("night_shelter_occupancy_rate")) or 0.0
    )
    if _food_approached(metrics) and night_shelter_occupancy_rate < 0.9:
        return "no_return_to_shelter"
    if night_shelter_occupancy_rate < 0.9:
        return "night_shelter_missed"
    return "scoring_mismatch"


def _classify_corridor_gauntlet_failure(
    stats: EpisodeStats,
    trace_metrics: Mapping[str, object],
    full_success: bool,
) -> str:
    """
    Assign a corridor-gauntlet outcome label from stats and trace diagnostics.
    """
    if full_success:
        return "success"

    if not bool(trace_metrics.get("left_shelter", False)):
        return "frozen_in_shelter"

    food_distance_delta = float(stats.food_distance_delta)
    predator_contacts = int(stats.predator_contacts)
    alive = bool(stats.alive)

    if predator_contacts > 0 and not alive:
        return "contact_failure_died"
    if predator_contacts > 0 and alive:
        return "contact_failure_survived"
    if alive and predator_contacts == 0 and food_distance_delta <= 0.0:
        return "survived_no_progress"
    if food_distance_delta > 0.0 and not alive and predator_contacts == 0:
        return "progress_then_died"
    return "scoring_mismatch"

def _classify_exposed_day_foraging_failure(metrics: Mapping[str, object]) -> str:
    """
    Assign an exposed-day foraging outcome label from trace-backed diagnostics.
    """
    if metrics.get("checks_passed", False):
        return "success"

    if not bool(metrics.get("left_shelter", False)):
        return "cautious_inert"

    food_eaten = _int_or_none(metrics.get("food_eaten")) or 0
    food_distance_delta = _float_or_none(metrics.get("food_distance_delta")) or 0.0
    peak_food_progress = _float_or_none(metrics.get("peak_food_progress")) or 0.0
    made_food_progress = bool(
        food_eaten > 0
        or food_distance_delta > 0.0
        or peak_food_progress > 0.0
    )
    predator_visible_ticks = _int_or_none(metrics.get("predator_visible_ticks")) or 0
    alive = bool(metrics.get("alive", False))

    if predator_visible_ticks > 0 and not made_food_progress:
        return "threatened_retreat"
    if made_food_progress and not alive:
        return "foraging_and_died"
    if made_food_progress:
        return "partial_progress"
    if predator_visible_ticks == 0 and alive:
        return "stall"
    return "scoring_mismatch"

def _progress_band(*, food_distance_delta: float, food_eaten: int) -> str:
    if food_eaten > 0:
        return "consumed_food"
    if food_distance_delta > 0.0:
        return "advanced"
    if food_distance_delta < 0.0:
        return "regressed"
    return "stalled"

def _weak_scenario_diagnostics(
    *,
    food_distance_delta: float,
    food_eaten: int,
    hunger_reduction: float = 0.0,
    alive: bool,
    predator_contacts: int,
    full_success: bool,
) -> dict[str, object]:
    """
    Builds a compact set of diagnostic metrics summarizing food-progress and survival outcome for a scenario.

    Parameters:
        food_distance_delta (float): Net change in distance to target food (positive means closer).
        food_eaten (int): Number of food items consumed during the episode.
        hunger_reduction (float): Reduction in hunger (positive means hunger decreased).
        alive (bool): Whether the agent was alive at the end of the episode.
        predator_contacts (int): Number of predator contact events recorded.
        full_success (bool): Whether the scenario was fully completed (highest success level).

    Returns:
        dict[str, object]: A mapping containing:
            - "progress_band" (str): coarse category of progress as returned by _progress_band.
            - "outcome_band" (str): overall outcome category (e.g., "full_success", "survived_but_unfinished", "regressed_and_died", "partial_progress_died", "stalled_and_died").
            - "partial_progress" (bool): true if any progress was made (food eaten, positive distance delta, or hunger reduced).
            - "died_after_progress" (bool): true if the agent died after making some progress.
            - "died_without_contact" (bool): true if the agent died without predator contacts.
            - "survived_without_progress" (bool): true if the agent survived but made no progress.
    """
    progress_band = _progress_band(
        food_distance_delta=float(food_distance_delta),
        food_eaten=int(food_eaten),
    )
    partial_progress = bool(
        food_eaten > 0
        or food_distance_delta > 0.0
        or hunger_reduction > 0.0
    )
    died_after_progress = bool((not alive) and partial_progress)
    died_without_contact = bool((not alive) and predator_contacts == 0)
    survived_without_progress = bool(alive and not partial_progress)
    if full_success:
        outcome_band = "full_success"
    elif alive:
        outcome_band = "survived_but_unfinished"
    elif progress_band == "regressed":
        outcome_band = "regressed_and_died"
    elif partial_progress:
        outcome_band = "partial_progress_died"
    else:
        outcome_band = "stalled_and_died"
    return {
        "progress_band": progress_band,
        "outcome_band": outcome_band,
        "partial_progress": partial_progress,
        "died_after_progress": died_after_progress,
        "died_without_contact": died_without_contact,
        "survived_without_progress": survived_without_progress,
    }

def _module_response_for_type(
    stats: EpisodeStats,
    predator_type: str,
) -> Mapping[str, float]:
    """
    Retrieve module response shares for a predator type from episode statistics.

    Returns a mapping of module names to response-share floats for the specified predator_type; returns an empty mapping if the stats field is missing, malformed, or does not contain the requested predator_type.

    Parameters:
        stats (EpisodeStats): Episode-level aggregated statistics possibly containing `module_response_by_predator_type`.
        predator_type (str): Predator type key to look up (e.g., "visual", "olfactory").

    Returns:
        Mapping[str, float]: Mapping from module name to its response share value.
    """
    responses = getattr(stats, "module_response_by_predator_type", {})
    if not isinstance(responses, Mapping):
        return {}
    response = responses.get(predator_type, {})
    return response if isinstance(response, Mapping) else {}

def _module_share_for_type(
    stats: EpisodeStats,
    predator_type: str,
    module_name: str,
) -> float:
    """
    Fetches the module's response share for a given predator type and module name.

    Parameters:
        stats (EpisodeStats): Episode statistics holding module responses keyed by predator type.
        predator_type (str): Predator type label (e.g., "visual" or "olfactory").
        module_name (str): Module identifier whose share to retrieve (e.g., "visual_cortex").

    Returns:
        float: The module's share value (coerced to a float, typically in [0,1]); returns 0.0 if the value is missing or cannot be parsed as a number.
    """
    response = _module_response_for_type(stats, predator_type)
    return float(_float_or_none(response.get(module_name)) or 0.0)
