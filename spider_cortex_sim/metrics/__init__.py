"""Compatibility exports for behavior metrics."""

from .aggregation import (
    aggregate_behavior_scores,
    aggregate_episode_stats,
    build_behavior_check,
    build_behavior_score,
    flatten_behavior_rows,
    summarize_behavior_suite,
)
from .aggregation import _aggregate_values, _mean_like, _mean_map, _mean_scalar
from .accumulator import EpisodeMetricAccumulator, jensen_shannon_divergence
from .accumulator import (
    _clamp_unit_interval,
    _contact_predator_types,
    _diagnostic_predator_distance,
    _diagnostic_predator_distance_for_type,
    _dominant_predator_type,
    _first_active_predator_type,
    _normalize_counts,
    _normalize_distribution,
    _predator_type_threat,
    _softmax_probabilities,
)
from .types import (
    ACTION_CENTER_REPRESENTATION_FIELDS,
    COMPETENCE_LABELS,
    PREDATOR_RESPONSE_END_THRESHOLD,
    PREDATOR_TYPE_NAMES,
    PRIMARY_REPRESENTATION_READOUT_MODULES,
    PROPOSAL_SOURCE_NAMES,
    PROPOSER_REPRESENTATION_LOGIT_FIELD,
    REFLEX_MODULE_NAMES,
    SHELTER_ROLES,
    BehavioralEpisodeScore,
    BehaviorCheckResult,
    BehaviorCheckSpec,
    EpisodeStats,
    competence_label_from_eval_reflex_scale,
    normalize_competence_label,
)

__all__ = [
    "ACTION_CENTER_REPRESENTATION_FIELDS",
    "BehaviorCheckResult",
    "BehaviorCheckSpec",
    "BehavioralEpisodeScore",
    "COMPETENCE_LABELS",
    "EpisodeMetricAccumulator",
    "EpisodeStats",
    "PREDATOR_RESPONSE_END_THRESHOLD",
    "PREDATOR_TYPE_NAMES",
    "PRIMARY_REPRESENTATION_READOUT_MODULES",
    "PROPOSAL_SOURCE_NAMES",
    "PROPOSER_REPRESENTATION_LOGIT_FIELD",
    "REFLEX_MODULE_NAMES",
    "SHELTER_ROLES",
    "aggregate_behavior_scores",
    "aggregate_episode_stats",
    "build_behavior_check",
    "build_behavior_score",
    "competence_label_from_eval_reflex_scale",
    "flatten_behavior_rows",
    "jensen_shannon_divergence",
    "normalize_competence_label",
    "summarize_behavior_suite",
]
