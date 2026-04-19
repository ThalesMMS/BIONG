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

NIGHT_REST_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("deep_night_shelter", "Maintains deep shelter occupancy during the night.", ">= 0.95"),
    BehaviorCheckSpec("deep_sleep_reached", "Reaches deep sleep during the scenario.", "true"),
    BehaviorCheckSpec("sleep_debt_reduced", "Reduces sleep debt relative to the initial state.", ">= 0.45"),
)

PREDATOR_EDGE_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("predator_detected", "Detects the predator at the edge of the visual field.", "true"),
    BehaviorCheckSpec("predator_memory_recorded", "Records explicit predator memory after the encounter.", "true"),
    BehaviorCheckSpec("predator_reacted", "Shows an observable behavioral response to the encounter.", "true"),
)

ENTRANCE_AMBUSH_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("survives_ambush", "Survives the entrance ambush.", "true"),
    BehaviorCheckSpec("avoids_contact", "Avoids direct contact with the predator at the entrance.", "true"),
    BehaviorCheckSpec("keeps_shelter_safety", "Preserves shelter safety or survival under entrance pressure.", ">= 0.75 occupancy or escape"),
)

OPEN_FIELD_FORAGING_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("made_food_progress", "Reduces distance to food in exposed terrain.", "> 0"),
    BehaviorCheckSpec("foraging_viable", "Eats or substantially reduces distance to food.", "eat or delta >= 2"),
    BehaviorCheckSpec("survives_exposure", "Survives the exposed foraging scenario.", "true"),
)

SHELTER_BLOCKADE_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("survives_blockade", "Survives the nighttime entrance blockade.", "true"),
    BehaviorCheckSpec("avoids_blockade_contact", "Avoids direct contact during the blockade.", "true"),
    BehaviorCheckSpec("preserves_safety_or_escapes", "Preserves nighttime safety or records an escape from the predator.", "occupancy >= 0.75 or escape"),
)

RECOVER_AFTER_FAILED_CHASE_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("predator_enters_recover", "The predator enters RECOVER after failing the chase.", "true"),
    BehaviorCheckSpec("predator_returns_to_wait", "The predator returns to WAIT after the chase.", "true"),
    BehaviorCheckSpec("spider_survives", "The spider survives the post-chase sequence.", "true"),
)

CORRIDOR_GAUNTLET_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("corridor_food_progress", "Reduces distance to food inside the corridor.", "> 0"),
    BehaviorCheckSpec("corridor_avoids_contact", "Avoids direct contact in the narrow corridor.", "true"),
    BehaviorCheckSpec("corridor_survives", "Survives the risk corridor.", "true"),
)

TWO_SHELTER_TRADEOFF_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("tradeoff_survives", "Survives the two-shelter choice scenario.", "true"),
    BehaviorCheckSpec("tradeoff_makes_progress", "Makes progress toward food or safety.", "food delta > 0 or occupancy >= 0.9"),
    BehaviorCheckSpec("tradeoff_night_shelter", "Maintains high nighttime shelter occupancy.", ">= 0.9"),
)

EXPOSED_DAY_FORAGING_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("day_food_progress", "Reduces distance to food during daytime.", "> 0"),
    BehaviorCheckSpec("day_avoids_contact", "Avoids direct contact with the predator while foraging.", "true"),
    BehaviorCheckSpec("day_survives", "Survives exposed daytime foraging.", "true"),
)

FOOD_DEPRIVATION_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("hunger_reduced", "Reduces hunger or manages to eat under deprivation.", "eat or hunger reduction >= 0.18"),
    BehaviorCheckSpec("approaches_food", "Makes progress toward food.", "> 0"),
    BehaviorCheckSpec(
        "commits_to_foraging",
        "Spider leaves shelter and prioritizes hunger.",
        "left_shelter and hunger_valence_rate >= 0.5",
    ),
    BehaviorCheckSpec("survives_deprivation", "Survives the deprivation episode.", "true"),
)

VISUAL_OLFACTORY_PINCER_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("dual_threat_detected", "Both visual and olfactory predator pressures are represented.", "visual and olfactory threats > 0"),
    BehaviorCheckSpec("type_specific_response", "Visual threat favors visual_cortex while olfactory threat favors sensory_cortex.", "module preference splits by predator type"),
    BehaviorCheckSpec("survives_pincer", "Survives the dual-threat pincer without direct contact.", "alive and predator_contacts=0"),
)

OLFACTORY_AMBUSH_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("olfactory_threat_detected", "Registers the olfactory predator despite no visual contact.", "olfactory threat > 0 and predator_visible=false"),
    BehaviorCheckSpec("sensory_cortex_engaged", "Sensory cortex dominates the response to the olfactory ambush.", "sensory_cortex >= visual_cortex"),
    BehaviorCheckSpec("survives_olfactory_ambush", "Survives the ambush without direct contact.", "alive and predator_contacts=0"),
)

OLFACTORY_AMBUSH_WINDOW_TICKS = 4

VISUAL_HUNTER_OPEN_FIELD_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("visual_threat_detected", "Registers the fast visual hunter in open terrain.", "visual threat > 0"),
    BehaviorCheckSpec("visual_cortex_engaged", "Visual cortex dominates the response to the open-field hunter.", "visual_cortex >= sensory_cortex"),
    BehaviorCheckSpec("survives_visual_hunter", "Survives the open-field visual-hunter encounter.", "alive and predator_contacts=0"),
)

CONFLICT_PASS_RATE = 0.8

FOOD_VS_PREDATOR_CONFLICT_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("threat_priority", "Threat wins arbitration when the predator is visible and dangerous.", "winning_valence=threat"),
    BehaviorCheckSpec("foraging_suppressed_under_threat", "Foraging drive is suppressed while threat dominates.", "hunger gate < 0.5"),
    BehaviorCheckSpec("survives_without_contact", "Remains alive and avoids direct contact during the conflict.", "alive and predator_contacts=0"),
)

SLEEP_VS_EXPLORATION_CONFLICT_CHECKS: Sequence[BehaviorCheckSpec] = (
    BehaviorCheckSpec("sleep_priority", "Sleep wins arbitration in a safe nighttime context with high sleep pressure.", "winning_valence=sleep"),
    BehaviorCheckSpec("exploration_suppressed_under_sleep_pressure", "Residual exploration is suppressed while sleep pressure remains high.", "visual/sensory gates reduced"),
    BehaviorCheckSpec("resting_behavior_emerges", "The spider stays in or returns to the shelter and enters useful rest.", "sleep event or sleep debt reduction"),
)
