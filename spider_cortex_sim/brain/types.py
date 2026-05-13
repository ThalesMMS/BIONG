from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Sequence

import numpy as np

from ..ablations import BrainAblationConfig, default_brain_config
from ..arbitration import (
    ARBITRATION_EVIDENCE_FIELDS as DEFAULT_ARBITRATION_EVIDENCE_FIELDS,
    ARBITRATION_GATE_MODULE_ORDER as DEFAULT_ARBITRATION_GATE_MODULE_ORDER,
    ARBITRATION_NETWORK_NAME as DEFAULT_ARBITRATION_NETWORK_NAME,
    MONOLITHIC_POLICY_NAME as DEFAULT_MONOLITHIC_POLICY_NAME,
    PRIORITY_GATING_WEIGHTS as DEFAULT_PRIORITY_GATING_WEIGHTS,
    VALENCE_EVIDENCE_WEIGHTS as DEFAULT_VALENCE_EVIDENCE_WEIGHTS,
    VALENCE_ORDER as DEFAULT_VALENCE_ORDER,
    ArbitrationDecision,
    ValenceScore,
    apply_priority_gating,
    arbitration_evidence_input_dim,
    arbitration_gate_weight_for,
    clamp_unit,
    compute_arbitration,
    fixed_formula_valence_scores_from_evidence,
    warm_start_arbitration_network,
)
from ..bus import MessageBus
from ..interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    architecture_signature,
    interface_registry,
)
from ..modules import MODULE_HIDDEN_DIMS, CorticalModuleBank, ModuleResult, ReflexDecision
from ..nn import ArbitrationNetwork, MotorNetwork, ProposalNetwork, one_hot, softmax
from ..noise import _compute_execution_difficulty_core
from ..operational_profiles import OperationalProfile, runtime_operational_profile
from ..reflexes import (
    _apply_reflex_path as apply_reflex_path,
    _direction_action as direction_action,
    _module_reflex_decision as module_reflex_decision,
)
from ..world import ACTIONS

@dataclass
class BrainStep:
    module_results: List[ModuleResult]
    action_center_logits: np.ndarray
    action_center_policy: np.ndarray
    motor_correction_logits: np.ndarray
    observation: Dict[str, np.ndarray] = field(default_factory=dict)
    total_logits_without_reflex: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    total_logits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    policy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    value: float = 0.0
    action_intent_idx: int = 0
    motor_action_idx: int = 0
    action_idx: int = 0
    orientation_alignment: float = 1.0
    terrain_difficulty: float = 0.0
    momentum: float = 0.0
    execution_difficulty: float = 0.0
    execution_slip_occurred: bool = False
    motor_slip_occurred: bool = False
    motor_noise_applied: bool = False
    slip_reason: str = "none"
    motor_override: bool = False
    final_reflex_override: bool = False
    action_center_input: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    motor_input: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    policy_mode: str = "normal"
    arbitration_decision: ArbitrationDecision | None = None
    phase_logits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    phase_prediction: str | None = None
    phase_prediction_confidence: float = 0.0
    phase_target: str | None = None
    phase_target_idx: int = -1
    scenario_name: str | None = None
    event_attention_top_type: str | None = None
    event_attention_top_age: int = -1
    event_attention_entropy: float = 0.0
    selected_option: str | None = None
    option_age: int = -1
    option_termination_reason: str = "none"
    option_logits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    option_leaf_logits: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    option_owned_action: str | None = None
    safety_mask_applied: bool = False
    safety_masked_actions: tuple[str, ...] = ()
    external_override_count: int = 0
    affordance_blocked_logits: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    affordance_blocked_targets: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    affordance_role_logits: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    affordance_role_targets: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    geometry_logits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    geometry_targets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    shelter_column_logits: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    shelter_column_targets: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    shelter_position_logits: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    shelter_position_targets: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    transition_prediction_logits: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    transition_prediction_targets: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    transition_rollout_prediction_logits: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    transition_rollout_prediction_targets: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    teacher_action_target_idx: int = -1
    teacher_action_target_name: str | None = None
    teacher_action_target_stage: str | None = None
    teacher_option_target_idx: int = -1
    teacher_option_target_name: str | None = None
    teacher_option_target_stage: str | None = None
    b_level: int = -1
    b_effective_level: str | None = None
    b_mode: str | None = None
    semantic_action: str | None = None
    semantic_action_idx: int = -1
    learned_semantic_action: str | None = None
    learned_semantic_action_idx: int = -1
    semantic_action_source: str | None = None
    semantic_action_reason: str | None = None
    semantic_override_count: int = 0
    semantic_logits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    semantic_policy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    bridge_primitive_action: str | None = None
    bridge_reason: str | None = None
    blocked_mask: dict[str, bool] = field(default_factory=dict)
    food_delta_used: float = 0.0
    shelter_delta_used: float = 0.0
