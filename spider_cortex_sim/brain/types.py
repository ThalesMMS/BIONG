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
    b_parent_level: int | None = None
    b_transfer_source_checkpoint: str | None = None
    b_transfer_coverage: float | None = None
    b_current_threat_pressure: float | None = None
    b_temporal_threat_pressure: float | None = None
    b_predator_memory_pressure: float | None = None
    b_predator_trace_pressure: float | None = None
    b3_contact_cooldown: int | None = None
    b3_post_food_cooldown: int | None = None
    b3_hunger_drop: float | None = None
    b3_controller_profile: str | None = None
    b4_controller_profile: str | None = None
    b4_recovery_pressure: float | None = None
    b4_sleep_hold: bool | None = None
    b4_exit_blocked: bool | None = None
    b4_hunger_release: float | None = None
    b4_genetic_generation: int | None = None
    b4_genetic_candidate: int | None = None
    b5_controller_profile: str | None = None
    b5_hunger_urgency: float | None = None
    b5_sleep_pressure: float | None = None
    b5_recovery_debt: float | None = None
    b5_threat_gate: float | None = None
    b5_sleep_bout_lock: int | None = None
    b5_forage_commitment_lock: int | None = None
    b5_homeostatic_decision: str | None = None
    b5_genetic_generation: int | None = None
    b5_genetic_candidate: int | None = None
    b6_controller_family: str | None = None
    b6_controller_profile: str | None = None
    b6_risk_pressure: float | None = None
    b6_threat_priority: float | None = None
    b6_forage_suppressed: float | None = None
    b6_corridor_commitment: int | None = None
    b6_corridor_progress_memory: float | None = None
    b6_recurrent_state: str | None = None
    b6_return_lock: int | None = None
    b6_decision: str | None = None
    b6_genetic_generation: int | None = None
    b6_genetic_candidate: int | None = None
    b7_controller_profile: str | None = None
    b7_affordance_state: str | None = None
    b7_energy_budget: float | None = None
    b7_budget_margin: float | None = None
    b7_food_steps_estimate: float | None = None
    b7_return_steps_estimate: float | None = None
    b7_corridor_viability: float | None = None
    b7_abort_return: bool | None = None
    b7_commitment_lock: int | None = None
    b7_decision: str | None = None
    b7_genetic_generation: int | None = None
    b7_genetic_candidate: int | None = None
    b8_controller_profile: str | None = None
    b8_spatial_map_state: str | None = None
    b8_local_affordance_score: float | None = None
    b8_return_vector_strength: float | None = None
    b8_corridor_dead_end_risk: float | None = None
    b8_abort_executed: bool | None = None
    b8_place_memory: float | None = None
    b8_decision: str | None = None
    b8_genetic_generation: int | None = None
    b8_genetic_candidate: int | None = None
    b9_controller_profile: str | None = None
    b9_route_state: str | None = None
    b9_route_confidence: float | None = None
    b9_waypoint_lock: int | None = None
    b9_path_integrator: float | None = None
    b9_replan_signal: float | None = None
    b9_decision: str | None = None
    b9_genetic_generation: int | None = None
    b9_genetic_candidate: int | None = None
    b10_controller_profile: str | None = None
    b10_replay_state: str | None = None
    b10_prospective_value: float | None = None
    b10_rollout_depth: int | None = None
    b10_replay_memory: float | None = None
    b10_plan_commitment: int | None = None
    b10_abort_signal: float | None = None
    b10_decision: str | None = None
    b10_genetic_generation: int | None = None
    b10_genetic_candidate: int | None = None
    b11_controller_profile: str | None = None
    b11_confidence_state: str | None = None
    b11_plan_confidence: float | None = None
    b11_uncertainty: float | None = None
    b11_neuromod_signal: float | None = None
    b11_confidence_lock: int | None = None
    b11_decision: str | None = None
    b11_genetic_generation: int | None = None
    b11_genetic_candidate: int | None = None
    b12_controller_profile: str | None = None
    b12_attention_state: str | None = None
    b12_prediction_error: float | None = None
    b12_attention_gain: float | None = None
    b12_expected_progress: float | None = None
    b12_search_lock: int | None = None
    b12_decision: str | None = None
    b12_genetic_generation: int | None = None
    b12_genetic_candidate: int | None = None
    b13_controller_profile: str | None = None
    b13_search_state: str | None = None
    b13_local_route_score: float | None = None
    b13_affordance_samples: float | None = None
    b13_search_memory: float | None = None
    b13_dead_end_score: float | None = None
    b13_search_lock: int | None = None
    b13_decision: str | None = None
    b13_genetic_generation: int | None = None
    b13_genetic_candidate: int | None = None
    b14_controller_profile: str | None = None
    b14_uncertainty_state: str | None = None
    b14_affordance_confidence: float | None = None
    b14_uncertainty: float | None = None
    b14_risk_adjusted_score: float | None = None
    b14_commitment_lock: int | None = None
    b14_decision: str | None = None
    b14_genetic_generation: int | None = None
    b14_genetic_candidate: int | None = None
    b15_controller_profile: str | None = None
    b15_option_state: str | None = None
    b15_option_value: float | None = None
    b15_termination_pressure: float | None = None
    b15_persistence_score: float | None = None
    b15_option_lock: int | None = None
    b15_decision: str | None = None
    b15_genetic_generation: int | None = None
    b15_genetic_candidate: int | None = None
    b16_controller_profile: str | None = None
    b16_ensemble_state: str | None = None
    b16_continue_vote: float | None = None
    b16_return_vote: float | None = None
    b16_option_votes: float | None = None
    b16_consensus_score: float | None = None
    b16_conflict_score: float | None = None
    b16_ensemble_lock: int | None = None
    b16_decision: str | None = None
    b16_genetic_generation: int | None = None
    b16_genetic_candidate: int | None = None
    b17_controller_profile: str | None = None
    b17_modulator_state: str | None = None
    b17_arousal_signal: float | None = None
    b17_homeostatic_gain: float | None = None
    b17_option_gain: float | None = None
    b17_conflict_release: float | None = None
    b17_modulation_lock: int | None = None
    b17_decision: str | None = None
    b17_genetic_generation: int | None = None
    b17_genetic_candidate: int | None = None
    b18_controller_profile: str | None = None
    b18_trace_state: str | None = None
    b18_eligibility_trace: float | None = None
    b18_reward_prediction_proxy: float | None = None
    b18_stability_bias: float | None = None
    b18_switch_pressure: float | None = None
    b18_trace_lock: int | None = None
    b18_decision: str | None = None
    b18_genetic_generation: int | None = None
    b18_genetic_candidate: int | None = None
    b19_controller_profile: str | None = None
    b19_memory_state: str | None = None
    b19_episode_memory: float | None = None
    b19_consolidation_score: float | None = None
    b19_stability_vote: float | None = None
    b19_switch_suppression: float | None = None
    b19_memory_lock: int | None = None
    b19_decision: str | None = None
    b19_genetic_generation: int | None = None
    b19_genetic_candidate: int | None = None
    b20_controller_profile: str | None = None
    b20_buffer_state: str | None = None
    b20_working_buffer: float | None = None
    b20_context_binding: float | None = None
    b20_gate_vote: float | None = None
    b20_release_vote: float | None = None
    b20_buffer_lock: int | None = None
    b20_decision: str | None = None
    b20_genetic_generation: int | None = None
    b20_genetic_candidate: int | None = None
    b21_controller_profile: str | None = None
    b21_replay_state: str | None = None
    b21_sequence_memory: float | None = None
    b21_replay_score: float | None = None
    b21_route_commitment: float | None = None
    b21_abort_prediction: float | None = None
    b21_replay_lock: int | None = None
    b21_decision: str | None = None
    b21_genetic_generation: int | None = None
    b21_genetic_candidate: int | None = None
    b22_controller_profile: str | None = None
    b22_sim_state: str | None = None
    b22_prospective_sim: float | None = None
    b22_forward_model_score: float | None = None
    b22_viability_projection: float | None = None
    b22_abort_projection: float | None = None
    b22_sim_lock: int | None = None
    b22_decision: str | None = None
    b22_genetic_generation: int | None = None
    b22_genetic_candidate: int | None = None
    b23_controller_profile: str | None = None
    b23_conflict_state: str | None = None
    b23_prediction_error: float | None = None
    b23_conflict_memory: float | None = None
    b23_stability_vote: float | None = None
    b23_abort_bias: float | None = None
    b23_monitor_lock: int | None = None
    b23_decision: str | None = None
    b23_genetic_generation: int | None = None
    b23_genetic_candidate: int | None = None
    b24_controller_profile: str | None = None
    b24_precision_state: str | None = None
    b24_precision_memory: float | None = None
    b24_precision_vote: float | None = None
    b24_uncertainty_pressure: float | None = None
    b24_abort_precision: float | None = None
    b24_precision_lock: int | None = None
    b24_decision: str | None = None
    b24_genetic_generation: int | None = None
    b24_genetic_candidate: int | None = None
    b25_controller_profile: str | None = None
    b25_metacognitive_state: str | None = None
    b25_confidence_memory: float | None = None
    b25_confidence_vote: float | None = None
    b25_doubt_pressure: float | None = None
    b25_control_gain: float | None = None
    b25_meta_lock: int | None = None
    b25_decision: str | None = None
    b25_genetic_generation: int | None = None
    b25_genetic_candidate: int | None = None
    b26_controller_profile: str | None = None
    b26_allostatic_state: str | None = None
    b26_prediction_error: float | None = None
    b26_setpoint_pressure: float | None = None
    b26_control_vote: float | None = None
    b26_stability_lock: int | None = None
    b26_decision: str | None = None
    b26_genetic_generation: int | None = None
    b26_genetic_candidate: int | None = None
    b27_controller_profile: str | None = None
    b27_arousal_state: str | None = None
    b27_arousal_level: float | None = None
    b27_gain_modulation: float | None = None
    b27_stress_pressure: float | None = None
    b27_arousal_lock: int | None = None
    b27_decision: str | None = None
    b27_genetic_generation: int | None = None
    b27_genetic_candidate: int | None = None
    b28_controller_profile: str | None = None
    b28_attention_state: str | None = None
    b28_interoceptive_focus: float | None = None
    b28_attention_gain: float | None = None
    b28_distractor_pressure: float | None = None
    b28_attention_lock: int | None = None
    b28_decision: str | None = None
    b28_genetic_generation: int | None = None
    b28_genetic_candidate: int | None = None
    b29_controller_profile: str | None = None
    b29_salience_state: str | None = None
    b29_threat_salience: float | None = None
    b29_homeostatic_salience: float | None = None
    b29_corridor_salience: float | None = None
    b29_winner_channel: str | None = None
    b29_salience_lock: int | None = None
    b29_decision: str | None = None
    b29_genetic_generation: int | None = None
    b29_genetic_candidate: int | None = None
    b30_controller_profile: str | None = None
    b30_gate_state: str | None = None
    b30_go_signal: float | None = None
    b30_no_go_signal: float | None = None
    b30_action_gate: str | None = None
    b30_gate_lock: int | None = None
    b30_decision: str | None = None
    b30_genetic_generation: int | None = None
    b30_genetic_candidate: int | None = None
    b31_controller_profile: str | None = None
    b31_dopamine_state: str | None = None
    b31_reward_prediction_error: float | None = None
    b31_tonic_dopamine: float | None = None
    b31_phasic_dopamine: float | None = None
    b31_gate_bias: float | None = None
    b31_dopamine_lock: int | None = None
    b31_decision: str | None = None
    b31_genetic_generation: int | None = None
    b31_genetic_candidate: int | None = None
    b32_controller_profile: str | None = None
    b32_critic_value: float | None = None
    b32_actor_advantage: float | None = None
    b32_value_error: float | None = None
    b32_policy_bias: float | None = None
    b32_value_lock: int | None = None
    b32_decision: str | None = None
    b32_genetic_generation: int | None = None
    b32_genetic_candidate: int | None = None
    b33_controller_profile: str | None = None
    b33_td_error: float | None = None
    b33_bootstrap_value: float | None = None
    b33_reward_trace: float | None = None
    b33_actor_update: float | None = None
    b33_td_lock: int | None = None
    b33_decision: str | None = None
    b33_genetic_generation: int | None = None
    b33_genetic_candidate: int | None = None
    b34_controller_profile: str | None = None
    b34_eligibility_trace: float | None = None
    b34_credit_assignment: float | None = None
    b34_synaptic_tag: float | None = None
    b34_decay_memory: float | None = None
    b34_credit_lock: int | None = None
    b34_decision: str | None = None
    b34_genetic_generation: int | None = None
    b34_genetic_candidate: int | None = None
    b35_controller_profile: str | None = None
    b35_forward_value: float | None = None
    b35_transition_error: float | None = None
    b35_model_confidence: float | None = None
    b35_prediction_memory: float | None = None
    b35_model_lock: int | None = None
    b35_decision: str | None = None
    b35_genetic_generation: int | None = None
    b35_genetic_candidate: int | None = None
    b36_controller_profile: str | None = None
    b36_latent_state: float | None = None
    b36_belief_error: float | None = None
    b36_state_confidence: float | None = None
    b36_context_memory: float | None = None
    b36_belief_lock: int | None = None
    b36_decision: str | None = None
    b36_genetic_generation: int | None = None
    b36_genetic_candidate: int | None = None
    b37_controller_profile: str | None = None
    b37_external_state_factor: float | None = None
    b37_internal_state_factor: float | None = None
    b37_factor_alignment: float | None = None
    b37_factor_confidence: float | None = None
    b37_factor_lock: int | None = None
    b37_decision: str | None = None
    b37_genetic_generation: int | None = None
    b37_genetic_candidate: int | None = None
    b38_controller_profile: str | None = None
    b38_external_attention: float | None = None
    b38_internal_attention: float | None = None
    b38_attention_balance: float | None = None
    b38_attention_gain: float | None = None
    b38_attention_lock: int | None = None
    b38_decision: str | None = None
    b38_genetic_generation: int | None = None
    b38_genetic_candidate: int | None = None
    b39_controller_profile: str | None = None
    b39_binding_strength: float | None = None
    b39_cross_factor_coherence: float | None = None
    b39_bound_context: float | None = None
    b39_binding_gain: float | None = None
    b39_binding_lock: int | None = None
    b39_decision: str | None = None
    b39_genetic_generation: int | None = None
    b39_genetic_candidate: int | None = None
    b40_controller_profile: str | None = None
    b40_workspace_activation: float | None = None
    b40_broadcast_gain: float | None = None
    b40_context_availability: float | None = None
    b40_workspace_stability: float | None = None
    b40_workspace_lock: int | None = None
    b40_decision: str | None = None
    b40_genetic_generation: int | None = None
    b40_genetic_candidate: int | None = None
    b41_controller_profile: str | None = None
    b41_executive_selection: float | None = None
    b41_inhibitory_pressure: float | None = None
    b41_goal_context: float | None = None
    b41_executive_stability: float | None = None
    b41_executive_lock: int | None = None
    b41_decision: str | None = None
    b41_genetic_generation: int | None = None
    b41_genetic_candidate: int | None = None
    b42_controller_profile: str | None = None
    b42_error_signal: float | None = None
    b42_conflict_signal: float | None = None
    b42_performance_context: float | None = None
    b42_monitor_stability: float | None = None
    b42_monitor_lock: int | None = None
    b42_decision: str | None = None
    b42_genetic_generation: int | None = None
    b42_genetic_candidate: int | None = None
    b43_controller_profile: str | None = None
    b43_precision_signal: float | None = None
    b43_adaptive_threshold: float | None = None
    b43_arousal_context: float | None = None
    b43_control_stability: float | None = None
    b43_precision_lock: int | None = None
    b43_decision: str | None = None
    b43_genetic_generation: int | None = None
    b43_genetic_candidate: int | None = None
    b44_controller_profile: str | None = None
    b44_relay_gate: float | None = None
    b44_sensory_precision: float | None = None
    b44_context_relay: float | None = None
    b44_gate_stability: float | None = None
    b44_relay_lock: int | None = None
    b44_decision: str | None = None
    b44_genetic_generation: int | None = None
    b44_genetic_candidate: int | None = None
    b45_controller_profile: str | None = None
    b45_inhibitory_gate: float | None = None
    b45_sensory_filter: float | None = None
    b45_context_suppression: float | None = None
    b45_loop_stability: float | None = None
    b45_inhibition_lock: int | None = None
    b45_decision: str | None = None
    b45_genetic_generation: int | None = None
    b45_genetic_candidate: int | None = None
    b46_controller_profile: str | None = None
    b46_feedback_gain: float | None = None
    b46_topdown_context: float | None = None
    b46_prediction_match: float | None = None
    b46_feedback_stability: float | None = None
    b46_feedback_lock: int | None = None
    b46_decision: str | None = None
    b46_genetic_generation: int | None = None
    b46_genetic_candidate: int | None = None
    b47_controller_profile: str | None = None
    b47_phase_alignment: float | None = None
    b47_synchrony_gain: float | None = None
    b47_cross_loop_coherence: float | None = None
    b47_phase_lock: int | None = None
    b47_decision: str | None = None
    b47_genetic_generation: int | None = None
    b47_genetic_candidate: int | None = None
    b48_controller_profile: str | None = None
    b48_timing_error: float | None = None
    b48_predictive_timing: float | None = None
    b48_corrective_gain: float | None = None
    b48_calibration_lock: int | None = None
    b48_decision: str | None = None
    b48_genetic_generation: int | None = None
    b48_genetic_candidate: int | None = None
    b49_controller_profile: str | None = None
    b49_go_signal: float | None = None
    b49_no_go_signal: float | None = None
    b49_action_gate_balance: float | None = None
    b49_selection_lock: int | None = None
    b49_decision: str | None = None
    b49_genetic_generation: int | None = None
    b49_genetic_candidate: int | None = None
    b50_controller_profile: str | None = None
    b50_habit_strength: float | None = None
    b50_chunk_value: float | None = None
    b50_habit_stability: float | None = None
    b50_chunk_lock: int | None = None
    b50_decision: str | None = None
    b50_genetic_generation: int | None = None
    b50_genetic_candidate: int | None = None
    b51_controller_profile: str | None = None
    b51_prediction_error: float | None = None
    b51_dopamine_gain: float | None = None
    b51_habit_modulation: float | None = None
    b51_modulation_lock: int | None = None
    b51_decision: str | None = None
    b51_genetic_generation: int | None = None
    b51_genetic_candidate: int | None = None
    b52_controller_profile: str | None = None
    b52_acetylcholine_level: float | None = None
    b52_precision_gain: float | None = None
    b52_uncertainty_signal: float | None = None
    b52_attention_lock: int | None = None
    b52_decision: str | None = None
    b52_genetic_generation: int | None = None
    b52_genetic_candidate: int | None = None
    b53_controller_profile: str | None = None
    b53_norepinephrine_level: float | None = None
    b53_arousal_gain: float | None = None
    b53_surprise_signal: float | None = None
    b53_gain_lock: int | None = None
    b53_decision: str | None = None
    b53_genetic_generation: int | None = None
    b53_genetic_candidate: int | None = None
    b54_controller_profile: str | None = None
    b54_serotonin_level: float | None = None
    b54_patience_signal: float | None = None
    b54_impulse_suppression: float | None = None
    b54_patience_lock: int | None = None
    b54_decision: str | None = None
    b54_genetic_generation: int | None = None
    b54_genetic_candidate: int | None = None
    b55_controller_profile: str | None = None
    b55_hypothalamic_drive: float | None = None
    b55_satiety_signal: float | None = None
    b55_recovery_bias: float | None = None
    b55_drive_balance: float | None = None
    b55_drive_lock: int | None = None
    b55_decision: str | None = None
    b55_genetic_generation: int | None = None
    b55_genetic_candidate: int | None = None
    b56_controller_profile: str | None = None
    b56_cortisol_level: float | None = None
    b56_stress_load: float | None = None
    b56_recovery_signal: float | None = None
    b56_endocrine_balance: float | None = None
    b56_stress_lock: int | None = None
    b56_decision: str | None = None
    b56_genetic_generation: int | None = None
    b56_genetic_candidate: int | None = None
    b57_controller_profile: str | None = None
    b57_interoceptive_awareness: float | None = None
    b57_visceral_salience: float | None = None
    b57_body_state_confidence: float | None = None
    b57_awareness_balance: float | None = None
    b57_awareness_lock: int | None = None
    b57_decision: str | None = None
    b57_genetic_generation: int | None = None
    b57_genetic_candidate: int | None = None
    b58_controller_profile: str | None = None
    b58_conflict_signal: float | None = None
    b58_error_likelihood: float | None = None
    b58_control_allocation: float | None = None
    b58_resolution_balance: float | None = None
    b58_conflict_lock: int | None = None
    b58_decision: str | None = None
    b58_genetic_generation: int | None = None
    b58_genetic_candidate: int | None = None
    b59_controller_profile: str | None = None
    b59_goal_context: float | None = None
    b59_working_set_stability: float | None = None
    b59_task_set_confidence: float | None = None
    b59_executive_balance: float | None = None
    b59_executive_lock: int | None = None
    b59_decision: str | None = None
    b59_genetic_generation: int | None = None
    b59_genetic_candidate: int | None = None
    b60_controller_profile: str | None = None
    b60_outcome_value: float | None = None
    b60_reversal_signal: float | None = None
    b60_goal_value_confidence: float | None = None
    b60_value_balance: float | None = None
    b60_value_lock: int | None = None
    b60_decision: str | None = None
    b60_genetic_generation: int | None = None
    b60_genetic_candidate: int | None = None
    b61_controller_profile: str | None = None
    b61_safety_value: float | None = None
    b61_threat_value: float | None = None
    b61_safety_confidence: float | None = None
    b61_affective_balance: float | None = None
    b61_safety_lock: int | None = None
    b61_decision: str | None = None
    b61_genetic_generation: int | None = None
    b61_genetic_candidate: int | None = None
    b62_controller_profile: str | None = None
    b62_defensive_mode: str | None = None
    b62_freeze_pressure: float | None = None
    b62_flee_pressure: float | None = None
    b62_shelter_bias: float | None = None
    b62_defense_balance: float | None = None
    b62_defense_lock: int | None = None
    b62_decision: str | None = None
    b62_genetic_generation: int | None = None
    b62_genetic_candidate: int | None = None
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
