"""Episode metric accumulation and representation-specialization helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Sequence

from ..ablations import PROPOSAL_SOURCE_NAMES, REFLEX_MODULE_NAMES
from .types import (
    EpisodeStats,
    PREDATOR_RESPONSE_END_THRESHOLD,
    PREDATOR_TYPE_NAMES,
    PRIMARY_REPRESENTATION_READOUT_MODULES,
    PROPOSER_REPRESENTATION_LOGIT_FIELD,
    SHELTER_ROLES,
)

from .accumulator_math import _clamp_unit_interval, _softmax_probabilities, jensen_shannon_divergence, _normalize_counts, _normalize_distribution, _predator_type_threat, _dominant_predator_type, _diagnostic_predator_distance, _diagnostic_predator_distance_for_type, _first_active_predator_type, _coerce_grid_coordinate, _contact_predator_types
from .accumulator_recording import AccumulatorRecordingMixin
from .accumulator_snapshot import AccumulatorSnapshotMixin


@dataclass
class EpisodeMetricAccumulator(AccumulatorRecordingMixin, AccumulatorSnapshotMixin):
    reward_component_names: Sequence[str]
    predator_states: Sequence[str]
    night_ticks: int = 0
    night_shelter_ticks: int = 0
    night_still_ticks: int = 0
    night_role_ticks: Dict[str, int] = field(default_factory=dict)
    predator_response_latencies: List[int] = field(default_factory=list)
    active_predator_response: Dict[str, int] | None = None
    predator_contacts_by_type: Dict[str, int] = field(default_factory=dict)
    predator_escapes_by_type: Dict[str, int] = field(default_factory=dict)
    predator_response_latencies_by_type: Dict[str, List[int]] = field(default_factory=dict)
    active_predator_responses_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reward_component_totals: Dict[str, float] = field(default_factory=dict)
    predator_state_ticks: Dict[str, int] = field(default_factory=dict)
    sleep_debt_samples: List[float] = field(default_factory=list)
    predator_mode_transitions: int = 0
    initial_food_dist: int | None = None
    final_food_dist: int | None = None
    initial_shelter_dist: int | None = None
    final_shelter_dist: int | None = None
    decision_steps: int = 0
    reflex_steps: int = 0
    final_reflex_override_steps: int = 0
    module_reflex_usage_steps: Dict[str, int] = field(default_factory=dict)
    module_reflex_override_steps: Dict[str, int] = field(default_factory=dict)
    module_reflex_dominance_sums: Dict[str, float] = field(default_factory=dict)
    module_contribution_share_sums: Dict[str, float] = field(default_factory=dict)
    module_response_by_predator_type_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    proposer_logits_by_predator_type: Dict[str, Dict[str, List[float]]] = field(
        default_factory=dict
    )
    proposer_probs_by_predator_type: Dict[str, Dict[str, List[float]]] = field(
        default_factory=dict
    )
    proposer_steps_by_predator_type: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    action_center_gate_sums_by_predator_type: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    action_center_gate_counts_by_predator_type: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    action_center_contribution_sums_by_predator_type: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    action_center_contribution_counts_by_predator_type: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    dominant_module_counts: Dict[str, int] = field(default_factory=dict)
    current_dominant_module: str = ""
    current_proposer_post_reflex_logits: Dict[str, List[float]] = field(default_factory=dict)
    current_proposer_probs: Dict[str, List[float]] = field(default_factory=dict)
    current_action_center_gates: Dict[str, float] = field(default_factory=dict)
    current_action_center_contribution_share: Dict[str, float] = field(default_factory=dict)
    dominant_module_share_sum: float = 0.0
    effective_module_count_sum: float = 0.0
    module_agreement_rate_sum: float = 0.0
    module_disagreement_rate_sum: float = 0.0
    learning_steps: int = 0
    module_credit_weight_sums: Dict[str, float] = field(default_factory=dict)
    module_gradient_norm_sums: Dict[str, float] = field(default_factory=dict)
    counterfactual_credit_weight_sums: Dict[str, float] = field(default_factory=dict)
    motor_execution_steps: int = 0
    motor_slip_steps: int = 0
    orientation_alignment_samples: List[float] = field(default_factory=list)
    terrain_difficulty_samples: List[float] = field(default_factory=list)
    terrain_execution_counts: Dict[str, int] = field(default_factory=dict)
    terrain_slip_counts: Dict[str, int] = field(default_factory=dict)
