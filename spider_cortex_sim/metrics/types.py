"""Metric result dataclasses, constants, and label helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..ablations import PROPOSAL_SOURCE_NAMES, REFLEX_MODULE_NAMES

SHELTER_ROLES: Sequence[str] = ("outside", "entrance", "inside", "deep")
PREDATOR_TYPE_NAMES: Sequence[str] = ("visual", "olfactory")
# Threats at or below this level are treated as resolved for predator-response timing.
PREDATOR_RESPONSE_END_THRESHOLD: float = 0.05
COMPETENCE_LABELS: Sequence[str] = ("self_sufficient", "scaffolded", "mixed")
# Representation specialization compares the canonical proposer readout
# pipeline, `post_reflex_logits -> softmax -> probabilities`, for the primary
# sensory proposer modules. We intentionally anchor this readout on
# `visual_cortex` and `sensory_cortex`: they are the modality-specific cortical
# pathways we want to compare directly, while recurrent hidden state is
# history-dependent and therefore not a stable per-step probability readout
# across feed-forward and recurrent variants. Action-center specialization uses
# the per-module `module_gates` and `module_contribution_share` readouts.
PRIMARY_REPRESENTATION_READOUT_MODULES: Sequence[str] = (
    "visual_cortex",
    "sensory_cortex",
)
PROPOSER_REPRESENTATION_LOGIT_FIELD: str = "post_reflex_logits"
ACTION_CENTER_REPRESENTATION_FIELDS: Sequence[str] = (
    "module_gates",
    "module_contribution_share",
)


def normalize_competence_label(label: str) -> str:
    """
    Validate and return a competence label from the allowed set.
    
    Converts the input to `str` and verifies it is a member of `COMPETENCE_LABELS`; otherwise an error is raised.
    
    Returns:
        The validated competence label (one of `COMPETENCE_LABELS`).
    
    Raises:
        ValueError: If the normalized label is not one of the allowed `COMPETENCE_LABELS`.
    """
    normalized = str(label)
    if normalized not in COMPETENCE_LABELS:
        available = ", ".join(repr(item) for item in COMPETENCE_LABELS)
        raise ValueError(f"Invalid competence_label. Available labels: {available}.")
    return normalized


def competence_label_from_eval_reflex_scale(
    eval_reflex_scale: float | None,
) -> str:
    """
    Map an evaluation reflex scale to one of the competence labels.
    
    Parameters:
        eval_reflex_scale (float | None): Optional numeric scale indicating evaluation reflex strength.
    
    Returns:
        str: `"mixed"` if `eval_reflex_scale` is `None`, `"self_sufficient"` if the value is approximately `0.0`, otherwise `"scaffolded"`.
    """
    if eval_reflex_scale is None:
        return "mixed"
    return (
        "self_sufficient"
        if math.isclose(float(eval_reflex_scale), 0.0, abs_tol=1e-12)
        else "scaffolded"
    )


@dataclass(frozen=True)
class BehaviorCheckSpec:
    name: str
    description: str
    expected: str


@dataclass(frozen=True)
class BehaviorCheckResult:
    name: str
    description: str
    expected: str
    passed: bool
    value: Any


@dataclass
class BehavioralEpisodeScore:
    episode: int
    seed: int
    scenario: str
    objective: str
    success: bool
    checks: Dict[str, BehaviorCheckResult]
    behavior_metrics: Dict[str, Any]
    failures: List[str]


@dataclass
class EpisodeStats:
    episode: int
    seed: int
    training: bool
    scenario: str | None
    total_reward: float
    steps: int
    food_eaten: int
    sleep_events: int
    shelter_entries: int
    alert_events: int
    predator_contacts: int
    predator_sightings: int
    predator_escapes: int
    night_ticks: int
    night_shelter_ticks: int
    night_still_ticks: int
    night_role_ticks: Dict[str, int]
    night_shelter_occupancy_rate: float
    night_stillness_rate: float
    night_role_distribution: Dict[str, float]
    predator_response_events: int
    mean_predator_response_latency: float
    mean_sleep_debt: float
    food_distance_delta: float
    shelter_distance_delta: float
    final_hunger: float
    final_fatigue: float
    final_sleep_debt: float
    final_health: float
    alive: bool
    reward_component_totals: Dict[str, float]
    predator_state_ticks: Dict[str, int]
    predator_mode_transitions: int
    dominant_predator_state: str
    predator_contacts_by_type: Dict[str, int] = field(default_factory=dict)
    predator_escapes_by_type: Dict[str, int] = field(default_factory=dict)
    predator_response_latency_by_type: Dict[str, float] = field(default_factory=dict)
    module_response_by_predator_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    proposer_divergence_by_module: Dict[str, float] = field(default_factory=dict)
    action_center_gate_differential: Dict[str, float] = field(default_factory=dict)
    action_center_contribution_differential: Dict[str, float] = field(default_factory=dict)
    representation_specialization_score: float = 0.0
    mean_proposer_probs_by_predator_type: Dict[str, Dict[str, List[float]]] = field(
        default_factory=dict
    )
    reflex_usage_rate: float = 0.0
    final_reflex_override_rate: float = 0.0
    mean_reflex_dominance: float = 0.0
    module_reflex_usage_rates: Dict[str, float] = field(default_factory=dict)
    module_reflex_override_rates: Dict[str, float] = field(default_factory=dict)
    module_reflex_dominance: Dict[str, float] = field(default_factory=dict)
    module_contribution_share: Dict[str, float] = field(default_factory=dict)
    dominant_module: str = ""
    dominant_module_share: float = 0.0
    effective_module_count: float = 0.0
    module_agreement_rate: float = 0.0
    module_disagreement_rate: float = 0.0
    mean_module_credit_weights: Dict[str, float] = field(default_factory=dict)
    module_gradient_norm_means: Dict[str, float] = field(default_factory=dict)
    motor_slip_rate: float = 0.0
    mean_orientation_alignment: float = 0.0
    mean_terrain_difficulty: float = 0.0
    terrain_slip_rates: Dict[str, float] = field(default_factory=dict)
