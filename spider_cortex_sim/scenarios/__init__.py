from __future__ import annotations

from typing import Dict, Sequence

from . import scoring as _scoring
from . import setup as _setup
from . import specs as _specs
from . import trace as _trace
from .scoring import (
    CONFLICT_PASS_RATE,
    CORRIDOR_GAUNTLET_CHECKS,
    ENTRANCE_AMBUSH_CHECKS,
    EXPOSED_DAY_FORAGING_CHECKS,
    FOOD_DEPRIVATION_CHECKS,
    FOOD_VS_PREDATOR_CONFLICT_CHECKS,
    NIGHT_REST_CHECKS,
    OLFACTORY_AMBUSH_CHECKS,
    OLFACTORY_AMBUSH_WINDOW_TICKS,
    OPEN_FIELD_FORAGING_CHECKS,
    PREDATOR_EDGE_CHECKS,
    RECOVER_AFTER_FAILED_CHASE_CHECKS,
    SHELTER_BLOCKADE_CHECKS,
    SLEEP_VS_EXPLORATION_CONFLICT_CHECKS,
    TWO_SHELTER_TRADEOFF_CHECKS,
    VISUAL_HUNTER_OPEN_FIELD_CHECKS,
    VISUAL_OLFACTORY_PINCER_CHECKS,
    _classify_corridor_gauntlet_failure,
    _classify_exposed_day_foraging_failure,
    _classify_food_deprivation_failure,
    _classify_open_field_foraging_failure,
    _food_approached,
    _module_response_for_type,
    _module_share_for_type,
    _progress_band,
    _score_corridor_gauntlet,
    _score_entrance_ambush,
    _score_exposed_day_foraging,
    _score_food_deprivation,
    _score_food_vs_predator_conflict,
    _score_night_rest,
    _score_olfactory_ambush,
    _score_open_field_foraging,
    _score_predator_edge,
    _score_recover_after_failed_chase,
    _score_shelter_blockade,
    _score_sleep_vs_exploration_conflict,
    _score_two_shelter_tradeoff,
    _score_visual_hunter_open_field,
    _score_visual_olfactory_pincer,
    _weak_scenario_diagnostics,
)
from .setup import (
    _corridor_gauntlet,
    _entrance_ambush,
    _entrance_ambush_cell,
    _exposed_day_foraging,
    _first_cell,
    _first_food_visible_frontier,
    _food_deprivation,
    _food_visible_from_cell,
    _food_vs_predator_conflict,
    _night_rest,
    _olfactory_ambush,
    _open_field_foraging,
    _open_field_foraging_food_cell,
    _open_field_foraging_has_initial_food_signal,
    _predator_edge,
    _recover_after_failed_chase,
    _safe_lizard_cell,
    _set_predators,
    _shelter_blockade,
    _sleep_vs_exploration_conflict,
    _teleport_spider,
    _two_shelter_tradeoff,
    _visual_hunter_open_field,
    _visual_olfactory_pincer,
)
from .specs import (
    FAST_VISUAL_HUNTER_PROFILE,
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    NIGHT_REST_INITIAL_SLEEP_DEBT,
    SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
    TRACE_MAP_HEIGHT_FALLBACK,
    TRACE_MAP_WIDTH_FALLBACK,
    BenchmarkTier,
    BehaviorScoreFn,
    ProbeType,
    ScenarioSpec,
)
from .trace import (
    _clamp01,
    _environment_observation_food_distances,
    _extract_exposed_day_trace_metrics,
    _float_or_none,
    _food_signal_strength,
    _hunger_valence_rate,
    _int_or_none,
    _mapping_predator_visible,
    _memory_vector_freshness,
    _observation_food_distances,
    _observation_meta_food_distance,
    _payload_float,
    _payload_text,
    _predator_visible_flag,
    _state_alive,
    _state_food_distance,
    _state_position,
    _trace_action_selection_payloads,
    _trace_any_mode,
    _trace_any_sleep_phase,
    _trace_cell,
    _trace_corridor_metrics,
    _trace_death_tick,
    _trace_distance_deltas,
    _trace_dominant_predator_types,
    _trace_escape_seen,
    _trace_food_distances,
    _trace_food_positions,
    _trace_food_signal_strengths,
    _trace_initial_food_distance,
    _trace_max_predator_threat,
    _trace_meta_mappings,
    _trace_peak_food_progress,
    _trace_post_observation_food_distances,
    _trace_pre_observation_food_distances,
    _trace_pre_tick_snapshot_payload,
    _trace_predator_memory_seen,
    _trace_predator_visible,
    _trace_previous_position,
    _trace_reconstructed_initial_food_distance,
    _trace_shelter_cells,
    _trace_shelter_exit,
    _trace_shelter_template_key,
    _trace_states,
    _trace_tick,
)

SCENARIOS: Dict[str, ScenarioSpec] = {
    "night_rest": ScenarioSpec(
        name="night_rest",
        description="Tired spider resting safely in deep shelter at night.",
        objective="Confirm safe rest with deep occupancy and reduced sleep debt.",
        behavior_checks=NIGHT_REST_CHECKS,
        diagnostic_focus="Deep rest, nighttime stillness, and reduced sleep debt.",
        success_interpretation="Success requires consistent deep occupancy and clear sleep recovery.",
        failure_interpretation="Failure suggests unstable rest, persistent sleep debt, or loss of deep shelter occupancy.",
        budget_note="This scenario already tends to pass under the short budget and serves as a positive rest control.",
        max_steps=12,
        map_template="central_burrow",
        setup=_night_rest,
        score_episode=_score_night_rest,
    ),
    "predator_edge": ScenarioSpec(
        name="predator_edge",
        description="Predator appears at the edge of the visual field.",
        objective="Confirm predator detection and response with reproducible explicit memory.",
        behavior_checks=PREDATOR_EDGE_CHECKS,
        diagnostic_focus="Predator detection, explicit memory, and observable encounter response.",
        success_interpretation="Success requires a perceived encounter, recorded memory, and a behavioral response.",
        failure_interpretation="Failure indicates a problem with detection, explicit memory, or encounter response.",
        budget_note="Works as a short sanity-check scenario for perception and predator response.",
        max_steps=18,
        map_template="central_burrow",
        setup=_predator_edge,
        score_episode=_score_predator_edge,
    ),
    "entrance_ambush": ScenarioSpec(
        name="entrance_ambush",
        description="Lizard waiting near the shelter entrance.",
        objective="Measure survival and shelter safety under an entrance ambush.",
        behavior_checks=ENTRANCE_AMBUSH_CHECKS,
        diagnostic_focus="Survival, avoided contact, and preserved shelter safety.",
        success_interpretation="Success requires surviving without direct contact while preserving a safe shelter or escape route.",
        failure_interpretation="Failure suggests a breach in entrance safety or an inability to sustain refuge.",
        budget_note="This scenario already tends to pass under the short budget and acts as a shelter-safety control.",
        max_steps=18,
        map_template="entrance_funnel",
        setup=_entrance_ambush,
        score_episode=_score_entrance_ambush,
    ),
    "open_field_foraging": ScenarioSpec(
        name="open_field_foraging",
        description="Food placed far from shelter in exposed terrain.",
        objective="Measure progress or success in a controlled open-field foraging task.",
        behavior_checks=OPEN_FIELD_FORAGING_CHECKS,
        diagnostic_focus=(
            "Probes food-vector acquisition in exposed terrain with an initial food signal; "
            "left_without_food_signal indicates setup regression, while regressed_and_died "
            "indicates capability or arbitration failure"
        ),
        success_interpretation="Success requires viable progress toward food while surviving exposure.",
        failure_interpretation="Failure may mean regression away from food, stalling, or dying before consolidating foraging.",
        budget_note="Under the short budget, this scenario often exposes spatial regression and contact-free death as distinct failure signals.",
        max_steps=20,
        map_template="exposed_feeding_ground",
        setup=_open_field_foraging,
        score_episode=_score_open_field_foraging,
        probe_type=ProbeType.CAPABILITY_PROBE.value,
        target_skill="food_vector_acquisition_exposed",
        geometry_assumptions=(
            "Food spawn is selected to provide a positive initial food signal from shelter "
            "via smell or visibility"
        ),
        benchmark_tier=BenchmarkTier.CAPABILITY.value,
        acceptable_partial_progress=(
            "Any positive food_distance_delta or left_shelter with food signal present indicates "
            "food-vector acquisition capability even if foraging_viable fails"
        ),
    ),
    "shelter_blockade": ScenarioSpec(
        name="shelter_blockade",
        description="Lizard blocking the shelter entrance at night.",
        objective="Validate preserved safety or escape under a nighttime route blockade.",
        behavior_checks=SHELTER_BLOCKADE_CHECKS,
        diagnostic_focus="Nighttime survival, avoided contact, and preserved shelter safety.",
        success_interpretation="Success requires survival and either preserved nighttime safety or an observable escape.",
        failure_interpretation="Failure suggests contact under blockade or loss of shelter safety during the night.",
        budget_note="Intermediate nighttime-pressure scenario that is useful for comparing sheltering versus escape.",
        max_steps=16,
        map_template="entrance_funnel",
        setup=_shelter_blockade,
        score_episode=_score_shelter_blockade,
    ),
    "recover_after_failed_chase": ScenarioSpec(
        name="recover_after_failed_chase",
        description="Lizard must exit ambush and enter recovery after losing the spider.",
        objective="Measure the predator's behavioral transition after a failed chase.",
        behavior_checks=RECOVER_AFTER_FAILED_CHASE_CHECKS,
        diagnostic_focus="Predator RECOVER -> WAIT sequence and spider survival after the chase.",
        success_interpretation="Success requires the predator's deterministic transition and episode survival.",
        failure_interpretation="Failure indicates a predator FSM regression or a lethal outcome for the spider.",
        budget_note="Short regression scenario for the predator FSM, less sensitive to training budget.",
        max_steps=14,
        map_template="entrance_funnel",
        setup=_recover_after_failed_chase,
        score_episode=_score_recover_after_failed_chase,
    ),
    "corridor_gauntlet": ScenarioSpec(
        name="corridor_gauntlet",
        description="Hungry spider crosses a narrow corridor with the lizard on the escape axis.",
        objective="Measure food progress and safety in a controlled risk corridor.",
        behavior_checks=CORRIDOR_GAUNTLET_CHECKS,
        diagnostic_focus="Trace-backed failure_mode classification for corridor traversal, shelter exit, predator visibility, peak food progress, contact, and death timing.",
        success_interpretation=(
            "Full success on this compound probe requires shelter exit, positive food-distance "
            "progress through the corridor, zero predator contacts, and survival to episode end; "
            "the scenario tests navigation, threat avoidance, and food commitment together."
        ),
        failure_interpretation=(
            'Use behavior_metrics.failure_mode to separate sub-capability failures in this compound probe: '
            '"success" means all corridor checks passed; '
            '"frozen_in_shelter" means no trace-confirmed shelter exit; '
            '"contact_failure_died" means predator contact occurred and the spider died; '
            '"contact_failure_survived" means predator contact occurred but the spider survived; '
            '"survived_no_progress" means the spider survived without contact but made no food-distance progress; '
            '"progress_then_died" means the spider made food-distance progress, avoided contact, then died; '
            '"scoring_mismatch" means stats and trace diagnostics reached an unexpected combination.'
        ),
        budget_note="Under the short budget, this scenario often shows avoided contact without enough progress for full success.",
        max_steps=20,
        map_template="corridor_escape",
        setup=_corridor_gauntlet,
        score_episode=_score_corridor_gauntlet,
        probe_type=ProbeType.CAPABILITY_PROBE.value,
        target_skill="corridor_navigation_under_threat",
        geometry_assumptions="Lizard positioned on corridor row; food at farthest spawn; requires shelter exit and directional commitment",
        benchmark_tier=BenchmarkTier.CAPABILITY.value,
        acceptable_partial_progress=(
            "Any positive food_distance_delta without contact indicates corridor navigation capability; "
            "survived_no_progress means the spider left shelter but made no measurable progress "
            "(navigation failed after exiting shelter); frozen_in_shelter is the shelter-exit failure"
        ),
    ),
    "two_shelter_tradeoff": ScenarioSpec(
        name="two_shelter_tradeoff",
        description="Map with two shelters forcing a choice between food and a refuge route.",
        objective="Measure the trade-off between food exploration and safety across two shelters.",
        behavior_checks=TWO_SHELTER_TRADEOFF_CHECKS,
        diagnostic_focus="Trade-off between food progress and maintaining an alternative shelter.",
        success_interpretation="Success requires survival, preserved nighttime shelter use, and useful progress.",
        failure_interpretation="Failure suggests a poor choice between exploration and safety or inability to sustain shelter.",
        budget_note="Useful scenario for comparing policies even when the short budget remains limited.",
        max_steps=24,
        map_template="two_shelters",
        setup=_two_shelter_tradeoff,
        score_episode=_score_two_shelter_tradeoff,
    ),
    "exposed_day_foraging": ScenarioSpec(
        name="exposed_day_foraging",
        description="Daytime foraging with food in open terrain and a nearby lizard patrol.",
        objective="Measure daytime foraging progress in open terrain under a nearby patrol.",
        behavior_checks=EXPOSED_DAY_FORAGING_CHECKS,
        diagnostic_focus=(
            "Hypothesis: previous geometry placed lizard on food, forcing spider into "
            "threat override before food signal was available; separate retreat, "
            "stalling, partial progress, and death during exposed daytime foraging."
        ),
        success_interpretation="Success requires progress toward food without contact while staying alive under the nearby patrol.",
        failure_interpretation="Use behavior_metrics.failure_mode to distinguish cautious_inert, threatened_retreat, foraging_and_died, partial_progress, stall, and scoring_mismatch under the heuristic lizard-food/frontier spacing; partial_progress is acceptable probe behavior showing capability under pressure.",
        budget_note="Geometry fix applies a threat-radius heuristic around food and the first food-visible frontier; remaining failures indicate arbitration or training limits.",
        max_steps=20,
        map_template="exposed_feeding_ground",
        setup=_exposed_day_foraging,
        score_episode=_score_exposed_day_foraging,
        probe_type=ProbeType.CAPABILITY_PROBE.value,
        target_skill="daytime_foraging_under_patrol",
        geometry_assumptions=(
            "Lizard spawned using frontier-separation heuristic to avoid placing predator "
            "directly on food; remaining failures indicate arbitration limits not geometry bugs"
        ),
        benchmark_tier=BenchmarkTier.CAPABILITY.value,
        acceptable_partial_progress=(
            "Any positive food_distance_delta indicates foraging capability even under threat; "
            "cautious_inert indicates arbitration chose safety over food"
        ),
    ),
    "food_deprivation": ScenarioSpec(
        name="food_deprivation",
        description="Spider starts with acute hunger and must recover homeostasis with distant food.",
        objective="Measure homeostatic recovery or explicit food progress under deprivation.",
        behavior_checks=FOOD_DEPRIVATION_CHECKS,
        diagnostic_focus=(
            "Separates hunger-driven commitment failures from timing failures and orientation "
            "failures during acute food deprivation"
        ),
        success_interpretation="Success requires measurable hunger reduction, progress toward food, and survival.",
        failure_interpretation="Use behavior_metrics.failure_mode to distinguish no_commitment, orientation_failure, timing_failure, and scoring_mismatch during reproducible diagnosis.",
        budget_note="Tests whether learned behavior can beat the calibrated geometric race condition instead of asking it to survive an unreachable far-food timer.",
        max_steps=22,
        map_template="central_burrow",
        setup=_food_deprivation,
        score_episode=_score_food_deprivation,
        probe_type=ProbeType.CAPABILITY_PROBE.value,
        target_skill="hunger_driven_commitment",
        geometry_assumptions="Food at calibrated 4-6 Manhattan distance; homeostatic death timer ~12-15 ticks at hunger 0.96; tests timing not just direction",
        benchmark_tier=BenchmarkTier.CAPABILITY.value,
        acceptable_partial_progress=(
            "commits_to_foraging=True with approaches_food=True indicates commitment capability "
            "even if timing_failure prevents full success"
        ),
    ),
    "visual_olfactory_pincer": ScenarioSpec(
        name="visual_olfactory_pincer",
        description="Spider is pinned between a visible hunter ahead and an olfactory hunter behind.",
        objective="Test whether predator-type-specific signals recruit distinct module responses under dual threat.",
        behavior_checks=VISUAL_OLFACTORY_PINCER_CHECKS,
        diagnostic_focus="Dual-threat detection, predator-type differentiation, and split visual-vs-sensory response.",
        success_interpretation="Success requires representing both predator types, showing type-specific module preference, and surviving without contact.",
        failure_interpretation="Failure suggests collapsed threat representation, weak specialization, or inability to escape the pincer safely.",
        budget_note="Short specialization probe for the new multi-predator ecology: both threat types are present on the opening frame.",
        max_steps=16,
        map_template="exposed_feeding_ground",
        setup=_visual_olfactory_pincer,
        score_episode=_score_visual_olfactory_pincer,
    ),
    "olfactory_ambush": ScenarioSpec(
        name="olfactory_ambush",
        description="An olfactory predator waits outside the shelter entrance while the spider faces inward.",
        objective="Test sensory-cortex-led response to hidden olfactory threat near refuge geometry.",
        behavior_checks=OLFACTORY_AMBUSH_CHECKS,
        diagnostic_focus="Olfactory-only threat perception, suppressed visual evidence, and sensory-cortex recruitment.",
        success_interpretation="Success requires detecting the hidden threat, preferring sensory-cortex response, and surviving the ambush cleanly.",
        failure_interpretation="Failure suggests the hidden predator is missed, treated as a generic threat, or still produces avoidable contact.",
        budget_note="Designed to isolate olfactory pressure without visible predator evidence on the opening frame.",
        max_steps=18,
        map_template="entrance_funnel",
        setup=_olfactory_ambush,
        score_episode=_score_olfactory_ambush,
    ),
    "visual_hunter_open_field": ScenarioSpec(
        name="visual_hunter_open_field",
        description="A fast visual hunter pressures the spider in exposed terrain.",
        objective="Test visual-cortex-led response to a fast visual predator in open terrain.",
        behavior_checks=VISUAL_HUNTER_OPEN_FIELD_CHECKS,
        diagnostic_focus="Visual threat detection, visual-cortex engagement, and survival in exposed terrain.",
        success_interpretation="Success requires explicit visual threat detection, stronger visual than sensory response, and survival without contact.",
        failure_interpretation="Failure suggests weak visual specialization or inability to stabilize behavior under open-field pursuit pressure.",
        budget_note="Pairs with olfactory_ambush to contrast visual-specialist and olfactory-specialist predator ecologies.",
        max_steps=16,
        map_template="exposed_feeding_ground",
        setup=_visual_hunter_open_field,
        score_episode=_score_visual_hunter_open_field,
    ),
    "food_vs_predator_conflict": ScenarioSpec(
        name="food_vs_predator_conflict",
        description="Nearby food competes with a visible predator in open terrain.",
        objective="Validate that threat overrides foraging when the conflict is explicit and immediate.",
        behavior_checks=FOOD_VS_PREDATOR_CONFLICT_CHECKS,
        diagnostic_focus="Threat-versus-hunger arbitration, down-weighted foraging, and contact-free survival.",
        success_interpretation="Success requires threat priority, suppressed food drive, and survival without contact.",
        failure_interpretation="Failure suggests opaque arbitration or persistent foraging under visible risk.",
        budget_note="Short deterministic scenario focused on checking priority gating in the food-versus-predator conflict.",
        max_steps=16,
        map_template="exposed_feeding_ground",
        setup=_food_vs_predator_conflict,
        score_episode=_score_food_vs_predator_conflict,
    ),
    "sleep_vs_exploration_conflict": ScenarioSpec(
        name="sleep_vs_exploration_conflict",
        description="Safe night with high fatigue and sleep debt, but residual exploration is still possible.",
        objective="Validate that sleep overrides residual exploration when the context is safe and homeostatic pressure is high.",
        behavior_checks=SLEEP_VS_EXPLORATION_CONFLICT_CHECKS,
        diagnostic_focus="Sleep-versus-exploration arbitration, reduced exploration gates, and useful shelter rest.",
        success_interpretation="Success requires sleep priority, suppressed residual exploration, and observable rest.",
        failure_interpretation="Failure suggests residual exploration still dominates despite strong sleep pressure.",
        budget_note="Short homeostatic-conflict scenario for validating the semantics of sleep gating.",
        max_steps=14,
        map_template="central_burrow",
        setup=_sleep_vs_exploration_conflict,
        score_episode=_score_sleep_vs_exploration_conflict,
    ),
}

CAPABILITY_PROBE_SCENARIOS: tuple[str, ...] = tuple(
    name
    for name, scenario in SCENARIOS.items()
    if scenario.probe_type == ProbeType.CAPABILITY_PROBE.value
)
# Capability probes are scenarios where a 0.00 score is interpretable through
# behavior_metrics.failure_mode and behavior_metrics.outcome_band, not a blocker
# for benchmark-of-record progress.

SCENARIO_NAMES: Sequence[str] = tuple(SCENARIOS.keys())


def get_scenario(name: str) -> ScenarioSpec:
    """
    Look up a scenario specification by its registry name.
    
    Parameters:
        name (str): Registry key identifying the scenario to retrieve.
    
    Returns:
        ScenarioSpec: The specification object associated with `name`.
    
    Raises:
        ValueError: If `name` is not a known scenario key.
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    return SCENARIOS[name]

__all__ = [
    'CAPABILITY_PROBE_SCENARIOS',
    'CONFLICT_PASS_RATE',
    'CORRIDOR_GAUNTLET_CHECKS',
    'ENTRANCE_AMBUSH_CHECKS',
    'EXPOSED_DAY_FORAGING_CHECKS',
    'FAST_VISUAL_HUNTER_PROFILE',
    'FOOD_DEPRIVATION_CHECKS',
    'FOOD_DEPRIVATION_INITIAL_HUNGER',
    'FOOD_VS_PREDATOR_CONFLICT_CHECKS',
    'NIGHT_REST_CHECKS',
    'NIGHT_REST_INITIAL_SLEEP_DEBT',
    'OLFACTORY_AMBUSH_CHECKS',
    'OLFACTORY_AMBUSH_WINDOW_TICKS',
    'OPEN_FIELD_FORAGING_CHECKS',
    'PREDATOR_EDGE_CHECKS',
    'RECOVER_AFTER_FAILED_CHASE_CHECKS',
    'SCENARIOS',
    'SCENARIO_NAMES',
    'SHELTER_BLOCKADE_CHECKS',
    'SLEEP_VS_EXPLORATION_CONFLICT_CHECKS',
    'SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT',
    'TRACE_MAP_HEIGHT_FALLBACK',
    'TRACE_MAP_WIDTH_FALLBACK',
    'TWO_SHELTER_TRADEOFF_CHECKS',
    'VISUAL_HUNTER_OPEN_FIELD_CHECKS',
    'VISUAL_OLFACTORY_PINCER_CHECKS',
    'BehaviorScoreFn',
    'BenchmarkTier',
    'ProbeType',
    'ScenarioSpec',
    '_clamp01',
    '_classify_corridor_gauntlet_failure',
    '_classify_exposed_day_foraging_failure',
    '_classify_food_deprivation_failure',
    '_classify_open_field_foraging_failure',
    '_corridor_gauntlet',
    '_entrance_ambush',
    '_entrance_ambush_cell',
    '_environment_observation_food_distances',
    '_exposed_day_foraging',
    '_extract_exposed_day_trace_metrics',
    '_first_cell',
    '_first_food_visible_frontier',
    '_float_or_none',
    '_food_approached',
    '_food_deprivation',
    '_food_signal_strength',
    '_food_visible_from_cell',
    '_food_vs_predator_conflict',
    '_hunger_valence_rate',
    '_int_or_none',
    '_mapping_predator_visible',
    '_memory_vector_freshness',
    '_module_response_for_type',
    '_module_share_for_type',
    '_night_rest',
    '_observation_food_distances',
    '_observation_meta_food_distance',
    '_olfactory_ambush',
    '_open_field_foraging',
    '_open_field_foraging_food_cell',
    '_open_field_foraging_has_initial_food_signal',
    '_payload_float',
    '_payload_text',
    '_predator_edge',
    '_predator_visible_flag',
    '_progress_band',
    '_recover_after_failed_chase',
    '_safe_lizard_cell',
    '_score_corridor_gauntlet',
    '_score_entrance_ambush',
    '_score_exposed_day_foraging',
    '_score_food_deprivation',
    '_score_food_vs_predator_conflict',
    '_score_night_rest',
    '_score_olfactory_ambush',
    '_score_open_field_foraging',
    '_score_predator_edge',
    '_score_recover_after_failed_chase',
    '_score_shelter_blockade',
    '_score_sleep_vs_exploration_conflict',
    '_score_two_shelter_tradeoff',
    '_score_visual_hunter_open_field',
    '_score_visual_olfactory_pincer',
    '_set_predators',
    '_shelter_blockade',
    '_sleep_vs_exploration_conflict',
    '_state_alive',
    '_state_food_distance',
    '_state_position',
    '_teleport_spider',
    '_trace_action_selection_payloads',
    '_trace_any_mode',
    '_trace_any_sleep_phase',
    '_trace_cell',
    '_trace_corridor_metrics',
    '_trace_death_tick',
    '_trace_distance_deltas',
    '_trace_dominant_predator_types',
    '_trace_escape_seen',
    '_trace_food_distances',
    '_trace_food_positions',
    '_trace_food_signal_strengths',
    '_trace_initial_food_distance',
    '_trace_max_predator_threat',
    '_trace_meta_mappings',
    '_trace_peak_food_progress',
    '_trace_post_observation_food_distances',
    '_trace_pre_observation_food_distances',
    '_trace_pre_tick_snapshot_payload',
    '_trace_predator_memory_seen',
    '_trace_predator_visible',
    '_trace_previous_position',
    '_trace_reconstructed_initial_food_distance',
    '_trace_shelter_cells',
    '_trace_shelter_exit',
    '_trace_shelter_template_key',
    '_trace_states',
    '_trace_tick',
    '_two_shelter_tradeoff',
    '_visual_hunter_open_field',
    '_visual_olfactory_pincer',
    '_weak_scenario_diagnostics',
    'get_scenario',
]

del _scoring
del _setup
del _specs
del _trace
