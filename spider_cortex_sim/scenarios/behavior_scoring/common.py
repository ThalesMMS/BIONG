from __future__ import annotations

from typing import Dict, Mapping, Sequence

from ...metrics import (
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeStats,
    build_behavior_check,
    build_behavior_score,
)
from ..specs import (
    NIGHT_REST_INITIAL_SLEEP_DEBT,
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
)
from ..trace import (
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

from ..scoring_checks import CONFLICT_PASS_RATE, CORRIDOR_GAUNTLET_CHECKS, ENTRANCE_AMBUSH_CHECKS, EXPOSED_DAY_FORAGING_CHECKS, FOOD_DEPRIVATION_CHECKS, FOOD_VS_PREDATOR_CONFLICT_CHECKS, NIGHT_REST_CHECKS, OLFACTORY_AMBUSH_CHECKS, OLFACTORY_AMBUSH_WINDOW_TICKS, OPEN_FIELD_FORAGING_CHECKS, PREDATOR_EDGE_CHECKS, RECOVER_AFTER_FAILED_CHASE_CHECKS, SHELTER_BLOCKADE_CHECKS, SLEEP_VS_EXPLORATION_CONFLICT_CHECKS, TWO_SHELTER_TRADEOFF_CHECKS, VISUAL_HUNTER_OPEN_FIELD_CHECKS, VISUAL_OLFACTORY_PINCER_CHECKS
from ..scoring_diagnostics import _classify_corridor_gauntlet_failure, _classify_exposed_day_foraging_failure, _classify_food_deprivation_failure, _classify_night_rest_failure, _classify_open_field_foraging_failure, _classify_two_shelter_tradeoff_failure, _module_response_for_type, _module_share_for_type, _weak_scenario_diagnostics

__all__ = [name for name in globals() if not name.startswith("__")]
