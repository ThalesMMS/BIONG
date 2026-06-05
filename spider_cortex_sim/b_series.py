from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .local_ecology_observation import LocalEcologyObservationAdapter


B_SERIES_POLICY_NAME = "b_series_policy"
B0_CURRENT_BRIDGE_POLICY_NAME = "b0_current_bridge_policy"
B1_CAPACITY_H48_POLICY_NAME = "b1_capacity_h48_bridge_policy"
B1_CAPACITY_H64_POLICY_NAME = "b1_capacity_h64_bridge_policy"
B1_THREAT_GUARD_POLICY_NAME = "b1_threat_guard_bridge_policy"
B2_TEMPORAL_THREAT_H48_POLICY_NAME = "b2_temporal_threat_h48_bridge_policy"
B2_TEMPORAL_THREAT_H56_POLICY_NAME = "b2_temporal_threat_h56_bridge_policy"
B2_TEMPORAL_THREAT_H64_POLICY_NAME = "b2_temporal_threat_h64_bridge_policy"
B3_CONTACT_MEMORY_H48_POLICY_NAME = "b3_contact_memory_h48_bridge_policy"
B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME = (
    "b3_contact_memory_strict_h48_bridge_policy"
)
B3_CONTACT_MEMORY_H56_POLICY_NAME = "b3_contact_memory_h56_bridge_policy"
B3_RECURRENT_GUARD_H48_POLICY_NAME = "b3_recurrent_guard_h48_bridge_policy"
B4_RECOVERY_BALANCE_H48_POLICY_NAME = "b4_recovery_balance_h48_bridge_policy"
B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME = (
    "b4_predator_exit_memory_h48_bridge_policy"
)
B4_RECOVERY_BALANCE_H56_POLICY_NAME = "b4_recovery_balance_h56_bridge_policy"
B4_GENETIC_RECOVERY_H48_POLICY_NAME = "b4_genetic_recovery_h48_bridge_policy"
B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME = (
    "b5_homeostatic_arbiter_h48_bridge_policy"
)
B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME = "b5_circadian_recovery_h48_bridge_policy"
B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME = (
    "b5_homeostatic_arbiter_h56_bridge_policy"
)
B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME = "b5_genetic_homeostasis_h48_bridge_policy"
B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME = "b6_risk_forage_arbiter_h48_bridge_policy"
B6_CORRIDOR_SURVIVAL_GUARD_H48_POLICY_NAME = (
    "b6_corridor_survival_guard_h48_bridge_policy"
)
B6_THREAT_PRIORITY_MEMORY_H48_POLICY_NAME = (
    "b6_threat_priority_memory_h48_bridge_policy"
)
B6_RISK_CORRIDOR_H56_POLICY_NAME = "b6_risk_corridor_h56_bridge_policy"
B6_GENETIC_RISK_CORRIDOR_H48_POLICY_NAME = (
    "b6_genetic_risk_corridor_h48_bridge_policy"
)
B6_RECURRENT_CONTEXT_H48_POLICY_NAME = "b6_recurrent_context_h48_bridge_policy"
B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME = (
    "b6_recurrent_threat_homeostasis_h48_bridge_policy"
)
B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME = (
    "b6_recurrent_corridor_guard_h48_bridge_policy"
)
B6_RECURRENT_CONTEXT_H56_POLICY_NAME = "b6_recurrent_context_h56_bridge_policy"
B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME = (
    "b6_genetic_recurrent_memory_h48_bridge_policy"
)
B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME = "b6_fused_risk_recurrent_h48_bridge_policy"
B7_AFFORDANCE_BUDGET_H48_POLICY_NAME = "b7_affordance_budget_h48_bridge_policy"
B7_ENERGY_BUDGET_CORRIDOR_H48_POLICY_NAME = (
    "b7_energy_budget_corridor_h48_bridge_policy"
)
B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME = "b7_recurrent_affordance_h48_bridge_policy"
B7_AFFORDANCE_BUDGET_H56_POLICY_NAME = "b7_affordance_budget_h56_bridge_policy"
B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME = (
    "b7_genetic_affordance_budget_h48_bridge_policy"
)
B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME = (
    "b8_spatial_affordance_map_h48_bridge_policy"
)
B8_RETURN_VECTOR_H48_POLICY_NAME = "b8_return_vector_h48_bridge_policy"
B8_CORRIDOR_PLACE_MEMORY_H48_POLICY_NAME = (
    "b8_corridor_place_memory_h48_bridge_policy"
)
B8_SPATIAL_AFFORDANCE_MAP_H56_POLICY_NAME = (
    "b8_spatial_affordance_map_h56_bridge_policy"
)
B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME = (
    "b8_genetic_spatial_affordance_h48_bridge_policy"
)
B9_WAYPOINT_PLANNER_H48_POLICY_NAME = "b9_waypoint_planner_h48_bridge_policy"
B9_PATH_INTEGRATION_H48_POLICY_NAME = "b9_path_integration_h48_bridge_policy"
B9_ROUTE_MEMORY_H48_POLICY_NAME = "b9_route_memory_h48_bridge_policy"
B9_WAYPOINT_PLANNER_H56_POLICY_NAME = "b9_waypoint_planner_h56_bridge_policy"
B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME = (
    "b9_genetic_waypoint_planner_h48_bridge_policy"
)
B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME = "b10_prospective_replay_h48_bridge_policy"
B10_VALUE_ROUTE_EVALUATOR_H48_POLICY_NAME = (
    "b10_value_route_evaluator_h48_bridge_policy"
)
B10_REPLAY_PLANNER_H48_POLICY_NAME = "b10_replay_planner_h48_bridge_policy"
B10_PROSPECTIVE_REPLAY_H56_POLICY_NAME = "b10_prospective_replay_h56_bridge_policy"
B10_GENETIC_REPLAY_PLANNER_H48_POLICY_NAME = (
    "b10_genetic_replay_planner_h48_bridge_policy"
)
B11_CONFIDENCE_ARBITER_H48_POLICY_NAME = "b11_confidence_arbiter_h48_bridge_policy"
B11_UNCERTAINTY_GATE_H48_POLICY_NAME = "b11_uncertainty_gate_h48_bridge_policy"
B11_NEUROMODULATED_REPLAY_H48_POLICY_NAME = (
    "b11_neuromodulated_replay_h48_bridge_policy"
)
B11_CONFIDENCE_ARBITER_H56_POLICY_NAME = "b11_confidence_arbiter_h56_bridge_policy"
B11_GENETIC_CONFIDENCE_GATE_H48_POLICY_NAME = (
    "b11_genetic_confidence_gate_h48_bridge_policy"
)
B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME = (
    "b12_predictive_attention_h48_bridge_policy"
)
B12_ACTIVE_INFERENCE_GATE_H48_POLICY_NAME = (
    "b12_active_inference_gate_h48_bridge_policy"
)
B12_AFFORDANCE_ATTENTION_H48_POLICY_NAME = (
    "b12_affordance_attention_h48_bridge_policy"
)
B12_PREDICTIVE_ATTENTION_H56_POLICY_NAME = (
    "b12_predictive_attention_h56_bridge_policy"
)
B12_GENETIC_ATTENTION_GATE_H48_POLICY_NAME = (
    "b12_genetic_attention_gate_h48_bridge_policy"
)
B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME = (
    "b13_local_affordance_search_h48_bridge_policy"
)
B13_COUNTERFACTUAL_ROUTE_H48_POLICY_NAME = (
    "b13_counterfactual_route_h48_bridge_policy"
)
B13_AFFORDANCE_SAMPLER_H48_POLICY_NAME = (
    "b13_affordance_sampler_h48_bridge_policy"
)
B13_LOCAL_AFFORDANCE_SEARCH_H56_POLICY_NAME = (
    "b13_local_affordance_search_h56_bridge_policy"
)
B13_GENETIC_LOCAL_SEARCH_H48_POLICY_NAME = (
    "b13_genetic_local_search_h48_bridge_policy"
)
B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME = (
    "b14_affordance_uncertainty_h48_bridge_policy"
)
B14_RISK_CALIBRATED_SEARCH_H48_POLICY_NAME = (
    "b14_risk_calibrated_search_h48_bridge_policy"
)
B14_CONFIDENCE_WEIGHTED_ROUTE_H48_POLICY_NAME = (
    "b14_confidence_weighted_route_h48_bridge_policy"
)
B14_AFFORDANCE_UNCERTAINTY_H56_POLICY_NAME = (
    "b14_affordance_uncertainty_h56_bridge_policy"
)
B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME = (
    "b14_genetic_uncertainty_search_h48_bridge_policy"
)
B15_OPTION_CRITIC_H48_POLICY_NAME = "b15_option_critic_h48_bridge_policy"
B15_PERSISTENCE_GATE_H48_POLICY_NAME = "b15_persistence_gate_h48_bridge_policy"
B15_VALUE_GATED_OPTION_H48_POLICY_NAME = "b15_value_gated_option_h48_bridge_policy"
B15_OPTION_CRITIC_H56_POLICY_NAME = "b15_option_critic_h56_bridge_policy"
B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME = (
    "b15_genetic_option_critic_h48_bridge_policy"
)
B16_OPTION_ENSEMBLE_H48_POLICY_NAME = "b16_option_ensemble_h48_bridge_policy"
B16_COMPETING_OPTIONS_H48_POLICY_NAME = "b16_competing_options_h48_bridge_policy"
B16_ACTION_SET_VOTER_H48_POLICY_NAME = "b16_action_set_voter_h48_bridge_policy"
B16_OPTION_ENSEMBLE_H56_POLICY_NAME = "b16_option_ensemble_h56_bridge_policy"
B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME = (
    "b16_genetic_option_ensemble_h48_bridge_policy"
)
B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME = (
    "b17_neuromodulated_ensemble_h48_bridge_policy"
)
B17_AROUSAL_GATED_OPTIONS_H48_POLICY_NAME = (
    "b17_arousal_gated_options_h48_bridge_policy"
)
B17_HOMEOSTATIC_MODULATOR_H48_POLICY_NAME = (
    "b17_homeostatic_modulator_h48_bridge_policy"
)
B17_NEUROMODULATED_ENSEMBLE_H56_POLICY_NAME = (
    "b17_neuromodulated_ensemble_h56_bridge_policy"
)
B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME = (
    "b17_genetic_neuromodulated_ensemble_h48_bridge_policy"
)
B18_ELIGIBILITY_TRACE_H48_POLICY_NAME = "b18_eligibility_trace_h48_bridge_policy"
B18_METASTABLE_AROUSAL_H48_POLICY_NAME = "b18_metastable_arousal_h48_bridge_policy"
B18_SYNAPTIC_TRACE_MODULATOR_H48_POLICY_NAME = (
    "b18_synaptic_trace_modulator_h48_bridge_policy"
)
B18_ELIGIBILITY_TRACE_H56_POLICY_NAME = "b18_eligibility_trace_h56_bridge_policy"
B18_GENETIC_ELIGIBILITY_TRACE_H48_POLICY_NAME = (
    "b18_genetic_eligibility_trace_h48_bridge_policy"
)
B19_EPISODIC_META_MEMORY_H48_POLICY_NAME = (
    "b19_episodic_meta_memory_h48_bridge_policy"
)
B19_STABILITY_MEMORY_H48_POLICY_NAME = "b19_stability_memory_h48_bridge_policy"
B19_SWITCH_SUPPRESSION_H48_POLICY_NAME = (
    "b19_switch_suppression_h48_bridge_policy"
)
B19_EPISODIC_META_MEMORY_H56_POLICY_NAME = (
    "b19_episodic_meta_memory_h56_bridge_policy"
)
B19_GENETIC_META_MEMORY_H48_POLICY_NAME = (
    "b19_genetic_meta_memory_h48_bridge_policy"
)
B20_WORKING_MEMORY_GATE_H48_POLICY_NAME = (
    "b20_working_memory_gate_h48_bridge_policy"
)
B20_CONTEXT_BINDING_H48_POLICY_NAME = "b20_context_binding_h48_bridge_policy"
B20_STABILITY_BUFFER_H48_POLICY_NAME = "b20_stability_buffer_h48_bridge_policy"
B20_WORKING_MEMORY_GATE_H56_POLICY_NAME = (
    "b20_working_memory_gate_h56_bridge_policy"
)
B20_GENETIC_WORKING_MEMORY_H48_POLICY_NAME = (
    "b20_genetic_working_memory_h48_bridge_policy"
)
B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME = (
    "b21_hippocampal_replay_h48_bridge_policy"
)
B21_SEQUENCE_BINDING_H48_POLICY_NAME = "b21_sequence_binding_h48_bridge_policy"
B21_ROUTE_REHEARSAL_H48_POLICY_NAME = "b21_route_rehearsal_h48_bridge_policy"
B21_HIPPOCAMPAL_REPLAY_H56_POLICY_NAME = (
    "b21_hippocampal_replay_h56_bridge_policy"
)
B21_GENETIC_REPLAY_GATE_H48_POLICY_NAME = (
    "b21_genetic_replay_gate_h48_bridge_policy"
)
B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME = (
    "b22_prospective_map_replay_h48_bridge_policy"
)
B22_FORWARD_MODEL_GATE_H48_POLICY_NAME = "b22_forward_model_gate_h48_bridge_policy"
B22_ROUTE_VIABILITY_SIM_H48_POLICY_NAME = (
    "b22_route_viability_sim_h48_bridge_policy"
)
B22_PROSPECTIVE_MAP_REPLAY_H56_POLICY_NAME = (
    "b22_prospective_map_replay_h56_bridge_policy"
)
B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME = (
    "b22_genetic_prospective_replay_h48_bridge_policy"
)
B23_CONFLICT_MONITOR_H48_POLICY_NAME = "b23_conflict_monitor_h48_bridge_policy"
B23_ERROR_GATED_REPLAY_H48_POLICY_NAME = "b23_error_gated_replay_h48_bridge_policy"
B23_ABORT_CONFLICT_ARBITER_H48_POLICY_NAME = (
    "b23_abort_conflict_arbiter_h48_bridge_policy"
)
B23_CONFLICT_MONITOR_H56_POLICY_NAME = "b23_conflict_monitor_h56_bridge_policy"
B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME = (
    "b23_genetic_conflict_monitor_h48_bridge_policy"
)
B24_PRECISION_CONFLICT_H48_POLICY_NAME = "b24_precision_conflict_h48_bridge_policy"
B24_PREDICTION_PRECISION_GATE_H48_POLICY_NAME = (
    "b24_prediction_precision_gate_h48_bridge_policy"
)
B24_RELIABILITY_ABORT_H48_POLICY_NAME = "b24_reliability_abort_h48_bridge_policy"
B24_PRECISION_CONFLICT_H56_POLICY_NAME = "b24_precision_conflict_h56_bridge_policy"
B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME = (
    "b24_genetic_precision_conflict_h48_bridge_policy"
)
B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME = (
    "b25_metacognitive_confidence_h48_bridge_policy"
)
B25_CONFIDENCE_CALIBRATION_H48_POLICY_NAME = (
    "b25_confidence_calibration_h48_bridge_policy"
)
B25_UNCERTAINTY_INTEGRATOR_H48_POLICY_NAME = (
    "b25_uncertainty_integrator_h48_bridge_policy"
)
B25_METACOGNITIVE_CONFIDENCE_H56_POLICY_NAME = (
    "b25_metacognitive_confidence_h56_bridge_policy"
)
B25_GENETIC_METACOGNITION_H48_POLICY_NAME = (
    "b25_genetic_metacognition_h48_bridge_policy"
)
B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME = (
    "b26_allostatic_prediction_h48_bridge_policy"
)
B26_SETPOINT_DRIFT_H48_POLICY_NAME = "b26_setpoint_drift_h48_bridge_policy"
B26_ERROR_SUPPRESSION_H48_POLICY_NAME = "b26_error_suppression_h48_bridge_policy"
B26_ALLOSTATIC_PREDICTION_H56_POLICY_NAME = (
    "b26_allostatic_prediction_h56_bridge_policy"
)
B26_GENETIC_ALLOSTASIS_H48_POLICY_NAME = "b26_genetic_allostasis_h48_bridge_policy"
B27_AROUSAL_GAIN_H48_POLICY_NAME = "b27_arousal_gain_h48_bridge_policy"
B27_STRESS_MODULATION_H48_POLICY_NAME = "b27_stress_modulation_h48_bridge_policy"
B27_ENERGY_AROUSAL_H48_POLICY_NAME = "b27_energy_arousal_h48_bridge_policy"
B27_AROUSAL_GAIN_H56_POLICY_NAME = "b27_arousal_gain_h56_bridge_policy"
B27_GENETIC_AROUSAL_H48_POLICY_NAME = "b27_genetic_arousal_h48_bridge_policy"
B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME = (
    "b28_interoceptive_attention_h48_bridge_policy"
)
B28_THREAT_FOCUS_ATTENTION_H48_POLICY_NAME = (
    "b28_threat_focus_attention_h48_bridge_policy"
)
B28_HOMEOSTATIC_ATTENTION_H48_POLICY_NAME = (
    "b28_homeostatic_attention_h48_bridge_policy"
)
B28_INTEROCEPTIVE_ATTENTION_H56_POLICY_NAME = (
    "b28_interoceptive_attention_h56_bridge_policy"
)
B28_GENETIC_ATTENTION_H48_POLICY_NAME = "b28_genetic_attention_h48_bridge_policy"
B29_SALIENCE_COMPETITION_H48_POLICY_NAME = (
    "b29_salience_competition_h48_bridge_policy"
)
B29_THREAT_SALIENCE_GATE_H48_POLICY_NAME = (
    "b29_threat_salience_gate_h48_bridge_policy"
)
B29_HOMEOSTATIC_SALIENCE_GATE_H48_POLICY_NAME = (
    "b29_homeostatic_salience_gate_h48_bridge_policy"
)
B29_SALIENCE_COMPETITION_H56_POLICY_NAME = (
    "b29_salience_competition_h56_bridge_policy"
)
B29_GENETIC_SALIENCE_H48_POLICY_NAME = "b29_genetic_salience_h48_bridge_policy"
B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME = (
    "b30_basal_ganglia_gate_h48_bridge_policy"
)
B30_GO_NOGO_BALANCE_H48_POLICY_NAME = "b30_go_nogo_balance_h48_bridge_policy"
B30_THREAT_INHIBITION_GATE_H48_POLICY_NAME = (
    "b30_threat_inhibition_gate_h48_bridge_policy"
)
B30_BASAL_GANGLIA_GATE_H56_POLICY_NAME = (
    "b30_basal_ganglia_gate_h56_bridge_policy"
)
B30_GENETIC_ACTION_GATE_H48_POLICY_NAME = "b30_genetic_action_gate_h48_bridge_policy"
B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME = (
    "b31_dopamine_prediction_error_h48_bridge_policy"
)
B31_TONIC_DOPAMINE_GATE_H48_POLICY_NAME = (
    "b31_tonic_dopamine_gate_h48_bridge_policy"
)
B31_PHASIC_DOPAMINE_GATE_H48_POLICY_NAME = (
    "b31_phasic_dopamine_gate_h48_bridge_policy"
)
B31_DOPAMINE_PREDICTION_ERROR_H56_POLICY_NAME = (
    "b31_dopamine_prediction_error_h56_bridge_policy"
)
B31_GENETIC_DOPAMINE_GATE_H48_POLICY_NAME = (
    "b31_genetic_dopamine_gate_h48_bridge_policy"
)
B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME = "b32_actor_critic_value_h48_bridge_policy"
B32_ADVANTAGE_VALUE_GATE_H48_POLICY_NAME = (
    "b32_advantage_value_gate_h48_bridge_policy"
)
B32_CRITIC_STABILITY_H48_POLICY_NAME = "b32_critic_stability_h48_bridge_policy"
B32_ACTOR_CRITIC_VALUE_H56_POLICY_NAME = "b32_actor_critic_value_h56_bridge_policy"
B32_GENETIC_ACTOR_CRITIC_H48_POLICY_NAME = (
    "b32_genetic_actor_critic_h48_bridge_policy"
)
B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME = (
    "b33_td_error_decomposition_h48_bridge_policy"
)
B33_BOOTSTRAPPED_VALUE_GATE_H48_POLICY_NAME = (
    "b33_bootstrapped_value_gate_h48_bridge_policy"
)
B33_REWARD_TRACE_CRITIC_H48_POLICY_NAME = "b33_reward_trace_critic_h48_bridge_policy"
B33_TD_ERROR_DECOMPOSITION_H56_POLICY_NAME = (
    "b33_td_error_decomposition_h56_bridge_policy"
)
B33_GENETIC_TD_VALUE_H48_POLICY_NAME = "b33_genetic_td_value_h48_bridge_policy"
B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME = (
    "b34_eligibility_credit_h48_bridge_policy"
)
B34_DELAYED_CREDIT_GATE_H48_POLICY_NAME = (
    "b34_delayed_credit_gate_h48_bridge_policy"
)
B34_SYNAPTIC_TAGGING_H48_POLICY_NAME = "b34_synaptic_tagging_h48_bridge_policy"
B34_ELIGIBILITY_CREDIT_H56_POLICY_NAME = (
    "b34_eligibility_credit_h56_bridge_policy"
)
B34_GENETIC_ELIGIBILITY_H48_POLICY_NAME = "b34_genetic_eligibility_h48_bridge_policy"
B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME = (
    "b35_forward_model_value_h48_bridge_policy"
)
B35_TRANSITION_ERROR_GATE_H48_POLICY_NAME = (
    "b35_transition_error_gate_h48_bridge_policy"
)
B35_MODEL_CONFIDENCE_H48_POLICY_NAME = "b35_model_confidence_h48_bridge_policy"
B35_FORWARD_MODEL_VALUE_H56_POLICY_NAME = (
    "b35_forward_model_value_h56_bridge_policy"
)
B35_GENETIC_FORWARD_MODEL_H48_POLICY_NAME = (
    "b35_genetic_forward_model_h48_bridge_policy"
)
B36_LATENT_BELIEF_STATE_H48_POLICY_NAME = (
    "b36_latent_belief_state_h48_bridge_policy"
)
B36_BELIEF_ERROR_GATE_H48_POLICY_NAME = "b36_belief_error_gate_h48_bridge_policy"
B36_CONTEXT_INFERENCE_H48_POLICY_NAME = "b36_context_inference_h48_bridge_policy"
B36_LATENT_BELIEF_STATE_H56_POLICY_NAME = (
    "b36_latent_belief_state_h56_bridge_policy"
)
B36_GENETIC_BELIEF_STATE_H48_POLICY_NAME = (
    "b36_genetic_belief_state_h48_bridge_policy"
)
B37_STATE_FACTOR_GATE_H48_POLICY_NAME = (
    "b37_state_factor_gate_h48_bridge_policy"
)
B37_INTERO_EXTERO_FACTOR_H48_POLICY_NAME = (
    "b37_intero_extero_factor_h48_bridge_policy"
)
B37_FACTOR_CONFIDENCE_H48_POLICY_NAME = "b37_factor_confidence_h48_bridge_policy"
B37_STATE_FACTOR_GATE_H56_POLICY_NAME = (
    "b37_state_factor_gate_h56_bridge_policy"
)
B37_GENETIC_STATE_FACTOR_H48_POLICY_NAME = (
    "b37_genetic_state_factor_h48_bridge_policy"
)
B38_FACTOR_ATTENTION_H48_POLICY_NAME = "b38_factor_attention_h48_bridge_policy"
B38_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME = (
    "b38_interoceptive_attention_h48_bridge_policy"
)
B38_CONFIDENCE_ATTENTION_H48_POLICY_NAME = (
    "b38_confidence_attention_h48_bridge_policy"
)
B38_FACTOR_ATTENTION_H56_POLICY_NAME = "b38_factor_attention_h56_bridge_policy"
B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME = (
    "b38_genetic_factor_attention_h48_bridge_policy"
)
B39_ATTENTION_BINDING_H48_POLICY_NAME = "b39_attention_binding_h48_bridge_policy"
B39_CROSS_FACTOR_BINDING_H48_POLICY_NAME = (
    "b39_cross_factor_binding_h48_bridge_policy"
)
B39_CONTEXT_BINDING_ATTENTION_H48_POLICY_NAME = (
    "b39_context_binding_attention_h48_bridge_policy"
)
B39_ATTENTION_BINDING_H56_POLICY_NAME = "b39_attention_binding_h56_bridge_policy"
B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME = (
    "b39_genetic_attention_binding_h48_bridge_policy"
)
B40_GLOBAL_WORKSPACE_H48_POLICY_NAME = "b40_global_workspace_h48_bridge_policy"
B40_SENSORY_WORKSPACE_H48_POLICY_NAME = "b40_sensory_workspace_h48_bridge_policy"
B40_CONTEXT_WORKSPACE_H48_POLICY_NAME = "b40_context_workspace_h48_bridge_policy"
B40_GLOBAL_WORKSPACE_H56_POLICY_NAME = "b40_global_workspace_h56_bridge_policy"
B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME = (
    "b40_genetic_global_workspace_h48_bridge_policy"
)
B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME = "b41_executive_workspace_h48_bridge_policy"
B41_INHIBITORY_CONTROL_H48_POLICY_NAME = "b41_inhibitory_control_h48_bridge_policy"
B41_GOAL_CONTEXT_SELECTOR_H48_POLICY_NAME = "b41_goal_context_selector_h48_bridge_policy"
B41_EXECUTIVE_WORKSPACE_H56_POLICY_NAME = "b41_executive_workspace_h56_bridge_policy"
B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME = (
    "b41_genetic_executive_workspace_h48_bridge_policy"
)
B42_ERROR_MONITOR_H48_POLICY_NAME = "b42_error_monitor_h48_bridge_policy"
B42_CONFLICT_MONITOR_H48_POLICY_NAME = "b42_conflict_monitor_h48_bridge_policy"
B42_PERFORMANCE_MONITOR_H48_POLICY_NAME = "b42_performance_monitor_h48_bridge_policy"
B42_ERROR_MONITOR_H56_POLICY_NAME = "b42_error_monitor_h56_bridge_policy"
B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME = (
    "b42_genetic_error_monitor_h48_bridge_policy"
)
B43_ADAPTIVE_PRECISION_H48_POLICY_NAME = "b43_adaptive_precision_h48_bridge_policy"
B43_AROUSAL_PRECISION_H48_POLICY_NAME = "b43_arousal_precision_h48_bridge_policy"
B43_THRESHOLD_ADAPTATION_H48_POLICY_NAME = (
    "b43_threshold_adaptation_h48_bridge_policy"
)
B43_ADAPTIVE_PRECISION_H56_POLICY_NAME = "b43_adaptive_precision_h56_bridge_policy"
B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME = (
    "b43_genetic_adaptive_precision_h48_bridge_policy"
)
B44_THALAMIC_RELAY_H48_POLICY_NAME = "b44_thalamic_relay_h48_bridge_policy"
B44_SENSORY_RELAY_H48_POLICY_NAME = "b44_sensory_relay_h48_bridge_policy"
B44_CONTEXT_RELAY_H48_POLICY_NAME = "b44_context_relay_h48_bridge_policy"
B44_THALAMIC_RELAY_H56_POLICY_NAME = "b44_thalamic_relay_h56_bridge_policy"
B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME = (
    "b44_genetic_thalamic_relay_h48_bridge_policy"
)
B45_RETICULAR_INHIBITION_H48_POLICY_NAME = "b45_reticular_inhibition_h48_bridge_policy"
B45_SENSORY_INHIBITION_H48_POLICY_NAME = "b45_sensory_inhibition_h48_bridge_policy"
B45_CONTEXT_INHIBITION_H48_POLICY_NAME = "b45_context_inhibition_h48_bridge_policy"
B45_RETICULAR_INHIBITION_H56_POLICY_NAME = "b45_reticular_inhibition_h56_bridge_policy"
B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME = (
    "b45_genetic_reticular_inhibition_h48_bridge_policy"
)
B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME = (
    "b46_corticothalamic_feedback_h48_bridge_policy"
)
B46_FEEDBACK_GAIN_H48_POLICY_NAME = "b46_feedback_gain_h48_bridge_policy"
B46_CONTEXT_FEEDBACK_H48_POLICY_NAME = "b46_context_feedback_h48_bridge_policy"
B46_CORTICOTHALAMIC_FEEDBACK_H56_POLICY_NAME = (
    "b46_corticothalamic_feedback_h56_bridge_policy"
)
B46_GENETIC_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME = (
    "b46_genetic_corticothalamic_feedback_h48_bridge_policy"
)
B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME = (
    "b47_oscillatory_synchrony_h48_bridge_policy"
)
B47_PHASE_LOCKING_H48_POLICY_NAME = "b47_phase_locking_h48_bridge_policy"
B47_COHERENCE_GATE_H48_POLICY_NAME = "b47_coherence_gate_h48_bridge_policy"
B47_OSCILLATORY_SYNCHRONY_H56_POLICY_NAME = (
    "b47_oscillatory_synchrony_h56_bridge_policy"
)
B47_GENETIC_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME = (
    "b47_genetic_oscillatory_synchrony_h48_bridge_policy"
)
B48_CEREBELLAR_TIMING_H48_POLICY_NAME = "b48_cerebellar_timing_h48_bridge_policy"
B48_TIMING_ERROR_CORRECTION_H48_POLICY_NAME = (
    "b48_timing_error_correction_h48_bridge_policy"
)
B48_PREDICTIVE_TIMING_H48_POLICY_NAME = "b48_predictive_timing_h48_bridge_policy"
B48_CEREBELLAR_TIMING_H56_POLICY_NAME = "b48_cerebellar_timing_h56_bridge_policy"
B48_GENETIC_CEREBELLAR_TIMING_H48_POLICY_NAME = (
    "b48_genetic_cerebellar_timing_h48_bridge_policy"
)
B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME = "b49_striatal_action_gate_h48_bridge_policy"
B49_DIRECT_PATH_FACILITATION_H48_POLICY_NAME = (
    "b49_direct_path_facilitation_h48_bridge_policy"
)
B49_INDIRECT_PATH_SUPPRESSION_H48_POLICY_NAME = (
    "b49_indirect_path_suppression_h48_bridge_policy"
)
B49_STRIATAL_ACTION_GATE_H56_POLICY_NAME = "b49_striatal_action_gate_h56_bridge_policy"
B49_GENETIC_STRIATAL_GATE_H48_POLICY_NAME = (
    "b49_genetic_striatal_gate_h48_bridge_policy"
)
B50_HABIT_CHUNKING_H48_POLICY_NAME = "b50_habit_chunking_h48_bridge_policy"
B50_ACTION_CHUNK_VALUE_H48_POLICY_NAME = "b50_action_chunk_value_h48_bridge_policy"
B50_HABIT_STABILITY_H48_POLICY_NAME = "b50_habit_stability_h48_bridge_policy"
B50_HABIT_CHUNKING_H56_POLICY_NAME = "b50_habit_chunking_h56_bridge_policy"
B50_GENETIC_HABIT_CHUNKING_H48_POLICY_NAME = (
    "b50_genetic_habit_chunking_h48_bridge_policy"
)
B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME = (
    "b51_dopaminergic_habit_modulation_h48_bridge_policy"
)
B51_REWARD_PREDICTION_GAIN_H48_POLICY_NAME = (
    "b51_reward_prediction_gain_h48_bridge_policy"
)
B51_NOVELTY_MODULATED_HABIT_H48_POLICY_NAME = (
    "b51_novelty_modulated_habit_h48_bridge_policy"
)
B51_DOPAMINERGIC_HABIT_MODULATION_H56_POLICY_NAME = (
    "b51_dopaminergic_habit_modulation_h56_bridge_policy"
)
B51_GENETIC_DOPAMINE_HABIT_H48_POLICY_NAME = (
    "b51_genetic_dopamine_habit_h48_bridge_policy"
)
B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME = (
    "b52_cholinergic_precision_gate_h48_bridge_policy"
)
B52_ATTENTION_GAIN_H48_POLICY_NAME = "b52_attention_gain_h48_bridge_policy"
B52_UNCERTAINTY_RELEASE_H48_POLICY_NAME = (
    "b52_uncertainty_release_h48_bridge_policy"
)
B52_CHOLINERGIC_PRECISION_GATE_H56_POLICY_NAME = (
    "b52_cholinergic_precision_gate_h56_bridge_policy"
)
B52_GENETIC_CHOLINERGIC_PRECISION_H48_POLICY_NAME = (
    "b52_genetic_cholinergic_precision_h48_bridge_policy"
)
B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME = (
    "b53_noradrenergic_arousal_gain_h48_bridge_policy"
)
B53_SURPRISE_GAIN_H48_POLICY_NAME = "b53_surprise_gain_h48_bridge_policy"
B53_STRESS_PRECISION_H48_POLICY_NAME = "b53_stress_precision_h48_bridge_policy"
B53_NORADRENERGIC_AROUSAL_GAIN_H56_POLICY_NAME = (
    "b53_noradrenergic_arousal_gain_h56_bridge_policy"
)
B53_GENETIC_AROUSAL_PRECISION_H48_POLICY_NAME = (
    "b53_genetic_arousal_precision_h48_bridge_policy"
)
B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME = (
    "b54_serotonergic_patience_gate_h48_bridge_policy"
)
B54_IMPULSE_SUPPRESSION_H48_POLICY_NAME = "b54_impulse_suppression_h48_bridge_policy"
B54_PATIENCE_BALANCE_H48_POLICY_NAME = "b54_patience_balance_h48_bridge_policy"
B54_SEROTONERGIC_PATIENCE_GATE_H56_POLICY_NAME = (
    "b54_serotonergic_patience_gate_h56_bridge_policy"
)
B54_GENETIC_SEROTONIN_PATIENCE_H48_POLICY_NAME = (
    "b54_genetic_serotonin_patience_h48_bridge_policy"
)
B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME = (
    "b55_hypothalamic_drive_coupling_h48_bridge_policy"
)
B55_SATIETY_RECOVERY_BALANCE_H48_POLICY_NAME = (
    "b55_satiety_recovery_balance_h48_bridge_policy"
)
B55_SLEEP_HUNGER_ARBITER_H48_POLICY_NAME = (
    "b55_sleep_hunger_arbiter_h48_bridge_policy"
)
B55_HYPOTHALAMIC_DRIVE_COUPLING_H56_POLICY_NAME = (
    "b55_hypothalamic_drive_coupling_h56_bridge_policy"
)
B55_GENETIC_HYPOTHALAMIC_DRIVE_H48_POLICY_NAME = (
    "b55_genetic_hypothalamic_drive_h48_bridge_policy"
)
B56_HPA_STRESS_AXIS_H48_POLICY_NAME = "b56_hpa_stress_axis_h48_bridge_policy"
B56_CORTISOL_RECOVERY_BALANCE_H48_POLICY_NAME = (
    "b56_cortisol_recovery_balance_h48_bridge_policy"
)
B56_STRESS_LOAD_GATE_H48_POLICY_NAME = "b56_stress_load_gate_h48_bridge_policy"
B56_HPA_STRESS_AXIS_H56_POLICY_NAME = "b56_hpa_stress_axis_h56_bridge_policy"
B56_GENETIC_HPA_STRESS_H48_POLICY_NAME = (
    "b56_genetic_hpa_stress_h48_bridge_policy"
)
B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME = (
    "b57_insular_interoceptive_awareness_h48_bridge_policy"
)
B57_VISCERAL_SALIENCE_GATE_H48_POLICY_NAME = (
    "b57_visceral_salience_gate_h48_bridge_policy"
)
B57_STRESS_DRIVE_AWARENESS_H48_POLICY_NAME = (
    "b57_stress_drive_awareness_h48_bridge_policy"
)
B57_INSULAR_INTEROCEPTIVE_AWARENESS_H56_POLICY_NAME = (
    "b57_insular_interoceptive_awareness_h56_bridge_policy"
)
B57_GENETIC_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME = (
    "b57_genetic_interoceptive_awareness_h48_bridge_policy"
)
B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME = "b58_acc_conflict_monitor_h48_bridge_policy"
B58_ERROR_SALIENCE_GATE_H48_POLICY_NAME = "b58_error_salience_gate_h48_bridge_policy"
B58_CONFLICT_RESOLUTION_BALANCE_H48_POLICY_NAME = (
    "b58_conflict_resolution_balance_h48_bridge_policy"
)
B58_ACC_CONFLICT_MONITOR_H56_POLICY_NAME = "b58_acc_conflict_monitor_h56_bridge_policy"
B58_GENETIC_ACC_CONFLICT_H48_POLICY_NAME = (
    "b58_genetic_acc_conflict_h48_bridge_policy"
)
B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME = (
    "b59_prefrontal_goal_context_h48_bridge_policy"
)
B59_WORKING_SET_STABILITY_H48_POLICY_NAME = (
    "b59_working_set_stability_h48_bridge_policy"
)
B59_EXECUTIVE_TASK_SET_H48_POLICY_NAME = "b59_executive_task_set_h48_bridge_policy"
B59_PREFRONTAL_GOAL_CONTEXT_H56_POLICY_NAME = (
    "b59_prefrontal_goal_context_h56_bridge_policy"
)
B59_GENETIC_PREFRONTAL_CONTROL_H48_POLICY_NAME = (
    "b59_genetic_prefrontal_control_h48_bridge_policy"
)
B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME = (
    "b60_orbitofrontal_outcome_value_h48_bridge_policy"
)
B60_REVERSAL_VALUE_GATE_H48_POLICY_NAME = "b60_reversal_value_gate_h48_bridge_policy"
B60_GOAL_OUTCOME_PREDICTION_H48_POLICY_NAME = (
    "b60_goal_outcome_prediction_h48_bridge_policy"
)
B60_ORBITOFRONTAL_OUTCOME_VALUE_H56_POLICY_NAME = (
    "b60_orbitofrontal_outcome_value_h56_bridge_policy"
)
B60_GENETIC_ORBITOFRONTAL_VALUE_H48_POLICY_NAME = (
    "b60_genetic_orbitofrontal_value_h48_bridge_policy"
)
B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME = (
    "b61_amygdala_safety_value_h48_bridge_policy"
)
B61_THREAT_VALUE_TAG_H48_POLICY_NAME = "b61_threat_value_tag_h48_bridge_policy"
B61_SAFETY_PREDICTION_GATE_H48_POLICY_NAME = (
    "b61_safety_prediction_gate_h48_bridge_policy"
)
B61_AMYGDALA_SAFETY_VALUE_H56_POLICY_NAME = (
    "b61_amygdala_safety_value_h56_bridge_policy"
)
B61_GENETIC_AMYGDALA_SAFETY_H48_POLICY_NAME = (
    "b61_genetic_amygdala_safety_h48_bridge_policy"
)
B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME = (
    "b62_defensive_mode_selector_h48_bridge_policy"
)
B62_FREEZE_FLEE_BALANCE_H48_POLICY_NAME = (
    "b62_freeze_flee_balance_h48_bridge_policy"
)
B62_SHELTER_DEFENSE_GATE_H48_POLICY_NAME = (
    "b62_shelter_defense_gate_h48_bridge_policy"
)
B62_DEFENSIVE_MODE_SELECTOR_H56_POLICY_NAME = (
    "b62_defensive_mode_selector_h56_bridge_policy"
)
B62_GENETIC_DEFENSIVE_MODE_H48_POLICY_NAME = (
    "b62_genetic_defensive_mode_h48_bridge_policy"
)
B63_PERIAQUEDUCTAL_ESCAPE_SEQUENCE_H48_POLICY_NAME = (
    "b63_periaqueductal_escape_sequence_h48_bridge_policy"
)
B63_FREEZE_RELEASE_ESCAPE_H48_POLICY_NAME = (
    "b63_freeze_release_escape_h48_bridge_policy"
)
B63_SHELTER_VECTOR_SEQUENCE_H48_POLICY_NAME = (
    "b63_shelter_vector_sequence_h48_bridge_policy"
)
B63_PERIAQUEDUCTAL_ESCAPE_SEQUENCE_H56_POLICY_NAME = (
    "b63_periaqueductal_escape_sequence_h56_bridge_policy"
)
B63_GENETIC_ESCAPE_SEQUENCE_H48_POLICY_NAME = (
    "b63_genetic_escape_sequence_h48_bridge_policy"
)
B64_VAGAL_RECOVERY_BRAKE_H48_POLICY_NAME = (
    "b64_vagal_recovery_brake_h48_bridge_policy"
)
B64_POST_ESCAPE_REST_GATE_H48_POLICY_NAME = (
    "b64_post_escape_rest_gate_h48_bridge_policy"
)
B64_SHELTER_RECOVERY_COUPLING_H48_POLICY_NAME = (
    "b64_shelter_recovery_coupling_h48_bridge_policy"
)
B64_VAGAL_RECOVERY_BRAKE_H56_POLICY_NAME = (
    "b64_vagal_recovery_brake_h56_bridge_policy"
)
B64_GENETIC_RECOVERY_BRAKE_H48_POLICY_NAME = (
    "b64_genetic_recovery_brake_h48_bridge_policy"
)
B65_ENTERIC_ASSIMILATION_GATE_H48_POLICY_NAME = (
    "b65_enteric_assimilation_gate_h48_bridge_policy"
)
B65_POST_RECOVERY_FORAGE_RELEASE_H48_POLICY_NAME = (
    "b65_post_recovery_forage_release_h48_bridge_policy"
)
B65_SHELTER_DIGESTIVE_COUPLING_H48_POLICY_NAME = (
    "b65_shelter_digestive_coupling_h48_bridge_policy"
)
B65_ENTERIC_ASSIMILATION_GATE_H56_POLICY_NAME = (
    "b65_enteric_assimilation_gate_h56_bridge_policy"
)
B65_GENETIC_ENTERIC_ASSIMILATION_H48_POLICY_NAME = (
    "b65_genetic_enteric_assimilation_h48_bridge_policy"
)
B66_IMMUNE_MALAISE_GATE_H48_POLICY_NAME = (
    "b66_immune_malaise_gate_h48_bridge_policy"
)
B66_INFLAMMATORY_RECOVERY_HOLD_H48_POLICY_NAME = (
    "b66_inflammatory_recovery_hold_h48_bridge_policy"
)
B66_DAMAGE_AWARE_FORAGE_RELEASE_H48_POLICY_NAME = (
    "b66_damage_aware_forage_release_h48_bridge_policy"
)
B66_IMMUNE_MALAISE_GATE_H56_POLICY_NAME = (
    "b66_immune_malaise_gate_h56_bridge_policy"
)
B66_GENETIC_IMMUNE_MALAISE_H48_POLICY_NAME = (
    "b66_genetic_immune_malaise_h48_bridge_policy"
)
B67_GLIAL_ENERGY_GATE_H48_POLICY_NAME = (
    "b67_glial_energy_gate_h48_bridge_policy"
)
B67_LACTATE_RECOVERY_SUPPORT_H48_POLICY_NAME = (
    "b67_lactate_recovery_support_h48_bridge_policy"
)
B67_FATIGUE_AWARE_RELEASE_H48_POLICY_NAME = (
    "b67_fatigue_aware_release_h48_bridge_policy"
)
B67_GLIAL_ENERGY_GATE_H56_POLICY_NAME = (
    "b67_glial_energy_gate_h56_bridge_policy"
)
B67_GENETIC_GLIAL_ENERGY_H48_POLICY_NAME = (
    "b67_genetic_glial_energy_h48_bridge_policy"
)
B68_PROPRIOCEPTIVE_PACING_GATE_H48_POLICY_NAME = (
    "b68_proprioceptive_pacing_gate_h48_bridge_policy"
)
B68_STRIDE_RECOVERY_PACING_H48_POLICY_NAME = (
    "b68_stride_recovery_pacing_h48_bridge_policy"
)
B68_OVEREXERTION_AWARE_RELEASE_H48_POLICY_NAME = (
    "b68_overexertion_aware_release_h48_bridge_policy"
)
B68_PROPRIOCEPTIVE_PACING_GATE_H56_POLICY_NAME = (
    "b68_proprioceptive_pacing_gate_h56_bridge_policy"
)
B68_GENETIC_MOTOR_PACING_H48_POLICY_NAME = (
    "b68_genetic_motor_pacing_h48_bridge_policy"
)
B69_VESTIBULAR_ORIENTATION_GATE_H48_POLICY_NAME = (
    "b69_vestibular_orientation_gate_h48_bridge_policy"
)
B69_HEADING_STABILITY_PACING_H48_POLICY_NAME = (
    "b69_heading_stability_pacing_h48_bridge_policy"
)
B69_TURN_ERROR_RECOVERY_H48_POLICY_NAME = (
    "b69_turn_error_recovery_h48_bridge_policy"
)
B69_VESTIBULAR_ORIENTATION_GATE_H56_POLICY_NAME = (
    "b69_vestibular_orientation_gate_h56_bridge_policy"
)
B69_GENETIC_ORIENTATION_GATE_H48_POLICY_NAME = (
    "b69_genetic_orientation_gate_h48_bridge_policy"
)
B70_OPTIC_FLOW_STABILIZATION_GATE_H48_POLICY_NAME = (
    "b70_optic_flow_stabilization_gate_h48_bridge_policy"
)
B70_LATERAL_FLOW_PACING_H48_POLICY_NAME = (
    "b70_lateral_flow_pacing_h48_bridge_policy"
)
B70_LOOMING_RISK_RECOVERY_H48_POLICY_NAME = (
    "b70_looming_risk_recovery_h48_bridge_policy"
)
B70_OPTIC_FLOW_STABILIZATION_GATE_H56_POLICY_NAME = (
    "b70_optic_flow_stabilization_gate_h56_bridge_policy"
)
B70_GENETIC_OPTIC_FLOW_GATE_H48_POLICY_NAME = (
    "b70_genetic_optic_flow_gate_h48_bridge_policy"
)
B71_TECTAL_ORIENTING_GATE_H48_POLICY_NAME = (
    "b71_tectal_orienting_gate_h48_bridge_policy"
)
B71_SALIENCE_MAP_PACING_H48_POLICY_NAME = (
    "b71_salience_map_pacing_h48_bridge_policy"
)
B71_COLLISION_VETO_RECOVERY_H48_POLICY_NAME = (
    "b71_collision_veto_recovery_h48_bridge_policy"
)
B71_TECTAL_ORIENTING_GATE_H56_POLICY_NAME = (
    "b71_tectal_orienting_gate_h56_bridge_policy"
)
B71_GENETIC_TECTAL_ORIENTING_H48_POLICY_NAME = (
    "b71_genetic_tectal_orienting_h48_bridge_policy"
)
B72_PULVINAR_ATTENTION_GATE_H48_POLICY_NAME = (
    "b72_pulvinar_attention_gate_h48_bridge_policy"
)
B72_DISTRACTOR_FILTER_PACING_H48_POLICY_NAME = (
    "b72_distractor_filter_pacing_h48_bridge_policy"
)
B72_FOCUS_LOCK_RECOVERY_H48_POLICY_NAME = (
    "b72_focus_lock_recovery_h48_bridge_policy"
)
B72_PULVINAR_ATTENTION_GATE_H56_POLICY_NAME = (
    "b72_pulvinar_attention_gate_h56_bridge_policy"
)
B72_GENETIC_PULVINAR_ATTENTION_H48_POLICY_NAME = (
    "b72_genetic_pulvinar_attention_h48_bridge_policy"
)
B73_RETICULAR_INHIBITION_GATE_H48_POLICY_NAME = (
    "b73_reticular_inhibition_gate_h48_bridge_policy"
)
B73_SURROUND_SUPPRESSION_PACING_H48_POLICY_NAME = (
    "b73_surround_suppression_pacing_h48_bridge_policy"
)
B73_FOCUS_RELEASE_RECOVERY_H48_POLICY_NAME = (
    "b73_focus_release_recovery_h48_bridge_policy"
)
B73_RETICULAR_INHIBITION_GATE_H56_POLICY_NAME = (
    "b73_reticular_inhibition_gate_h56_bridge_policy"
)
B73_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME = (
    "b73_genetic_reticular_inhibition_h48_bridge_policy"
)
B74_THALAMIC_REBOUND_GATE_H48_POLICY_NAME = (
    "b74_thalamic_rebound_gate_h48_bridge_policy"
)
B74_REBOUND_RELEASE_PACING_H48_POLICY_NAME = (
    "b74_rebound_release_pacing_h48_bridge_policy"
)
B74_POST_INHIBITION_RECOVERY_H48_POLICY_NAME = (
    "b74_post_inhibition_recovery_h48_bridge_policy"
)
B74_THALAMIC_REBOUND_GATE_H56_POLICY_NAME = (
    "b74_thalamic_rebound_gate_h56_bridge_policy"
)
B74_GENETIC_THALAMIC_REBOUND_H48_POLICY_NAME = (
    "b74_genetic_thalamic_rebound_h48_bridge_policy"
)
B75_BASAL_THALAMIC_RELEASE_H48_POLICY_NAME = (
    "b75_basal_thalamic_release_h48_bridge_policy"
)
B75_RELEASE_BURST_PACING_H48_POLICY_NAME = (
    "b75_release_burst_pacing_h48_bridge_policy"
)
B75_GO_NOGO_REBOUND_TIMING_H48_POLICY_NAME = (
    "b75_go_nogo_rebound_timing_h48_bridge_policy"
)
B75_BASAL_THALAMIC_RELEASE_H56_POLICY_NAME = (
    "b75_basal_thalamic_release_h56_bridge_policy"
)
B75_GENETIC_BASAL_THALAMIC_RELEASE_H48_POLICY_NAME = (
    "b75_genetic_basal_thalamic_release_h48_bridge_policy"
)
B76_CEREBELLAR_STRIDE_GATE_H48_POLICY_NAME = (
    "b76_cerebellar_stride_gate_h48_bridge_policy"
)
B76_STRIDE_ERROR_PACING_H48_POLICY_NAME = (
    "b76_stride_error_pacing_h48_bridge_policy"
)
B76_BURST_SMOOTHING_RECOVERY_H48_POLICY_NAME = (
    "b76_burst_smoothing_recovery_h48_bridge_policy"
)
B76_CEREBELLAR_STRIDE_GATE_H56_POLICY_NAME = (
    "b76_cerebellar_stride_gate_h56_bridge_policy"
)
B76_GENETIC_CEREBELLAR_STRIDE_H48_POLICY_NAME = (
    "b76_genetic_cerebellar_stride_h48_bridge_policy"
)
B77_OLIVARY_ERROR_CORRECTION_H48_POLICY_NAME = (
    "b77_olivary_error_correction_h48_bridge_policy"
)
B77_ERROR_PREDICTION_PACING_H48_POLICY_NAME = (
    "b77_error_prediction_pacing_h48_bridge_policy"
)
B77_CLIMBING_FIBER_RECOVERY_H48_POLICY_NAME = (
    "b77_climbing_fiber_recovery_h48_bridge_policy"
)
B77_OLIVARY_ERROR_CORRECTION_H56_POLICY_NAME = (
    "b77_olivary_error_correction_h56_bridge_policy"
)
B77_GENETIC_OLIVARY_ERROR_H48_POLICY_NAME = (
    "b77_genetic_olivary_error_h48_bridge_policy"
)
B78_VESTIBULAR_BALANCE_H48_POLICY_NAME = (
    "b78_vestibular_balance_h48_bridge_policy"
)
B78_HEAD_STABILIZATION_H48_POLICY_NAME = (
    "b78_head_stabilization_h48_bridge_policy"
)
B78_LOCOMOTOR_CONFIDENCE_H48_POLICY_NAME = (
    "b78_locomotor_confidence_h48_bridge_policy"
)
B78_VESTIBULAR_BALANCE_H56_POLICY_NAME = (
    "b78_vestibular_balance_h56_bridge_policy"
)
B78_GENETIC_VESTIBULAR_BALANCE_H48_POLICY_NAME = (
    "b78_genetic_vestibular_balance_h48_bridge_policy"
)
B0_CURRENT_BRIDGE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b0_current_bridge_policy/seed_7/best"
)
B1_THREAT_GUARD_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b1_threat_guard_bridge_policy/seed_7/best"
)
B2_TEMPORAL_THREAT_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b2_temporal_threat_h48_bridge_policy/seed_7/best"
)
B3_RECURRENT_GUARD_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b3_recurrent_guard_h48_bridge_policy/seed_7/best"
)
B4_GENETIC_RECOVERY_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b4_genetic_recovery_h48_bridge_policy/seed_7/best"
)
B5_GENETIC_HOMEOSTASIS_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b5_genetic_homeostasis_h48_bridge_policy/seed_7/best"
)
B6_FUSED_RISK_RECURRENT_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b6_fused_risk_recurrent_h48_bridge_policy/seed_7/best"
)
B7_AFFORDANCE_BUDGET_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b7_affordance_budget_h48_bridge_policy/seed_7/best"
)
B8_SPATIAL_AFFORDANCE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b8_spatial_affordance_map_h48_bridge_policy/seed_7/best"
)
B9_WAYPOINT_PLANNER_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b9_waypoint_planner_h48_bridge_policy/seed_7/best"
)
B10_PROSPECTIVE_REPLAY_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b10_prospective_replay_h48_bridge_policy/seed_7/best"
)
B11_CONFIDENCE_ARBITER_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b11_confidence_arbiter_h48_bridge_policy/seed_7/best"
)
B12_PREDICTIVE_ATTENTION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b12_predictive_attention_h48_bridge_policy/seed_7/best"
)
B13_LOCAL_AFFORDANCE_SEARCH_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b13_local_affordance_search_h48_bridge_policy/seed_7/best"
)
B14_AFFORDANCE_UNCERTAINTY_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b14_affordance_uncertainty_h48_bridge_policy/seed_7/best"
)
B15_OPTION_CRITIC_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b15_option_critic_h48_bridge_policy/seed_7/best"
)
B16_OPTION_ENSEMBLE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b16_option_ensemble_h48_bridge_policy/seed_7/best"
)
B17_NEUROMODULATED_ENSEMBLE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b17_neuromodulated_ensemble_h48_bridge_policy/seed_7/best"
)
B18_ELIGIBILITY_TRACE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b18_eligibility_trace_h48_bridge_policy/seed_7/best"
)
B19_EPISODIC_META_MEMORY_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b19_episodic_meta_memory_h48_bridge_policy/seed_7/best"
)
B20_WORKING_MEMORY_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b20_working_memory_gate_h48_bridge_policy/seed_7/best"
)
B21_HIPPOCAMPAL_REPLAY_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b21_hippocampal_replay_h48_bridge_policy/seed_7/best"
)
B22_PROSPECTIVE_REPLAY_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b22_prospective_map_replay_h48_bridge_policy/seed_7/best"
)
B23_CONFLICT_MONITOR_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b23_conflict_monitor_h48_bridge_policy/seed_7/best"
)
B24_PRECISION_CONFLICT_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b24_precision_conflict_h48_bridge_policy/seed_7/best"
)
B25_METACOGNITIVE_CONFIDENCE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b25_metacognitive_confidence_h48_bridge_policy/seed_7/best"
)
B26_ALLOSTATIC_PREDICTION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b26_allostatic_prediction_h48_bridge_policy/seed_7/best"
)
B27_AROUSAL_GAIN_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b27_arousal_gain_h48_bridge_policy/seed_7/best"
)
B28_INTEROCEPTIVE_ATTENTION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b28_interoceptive_attention_h48_bridge_policy/seed_7/best"
)
B29_SALIENCE_COMPETITION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b29_salience_competition_h48_bridge_policy/seed_7/best"
)
B30_BASAL_GANGLIA_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b30_basal_ganglia_gate_h48_bridge_policy/seed_7/best"
)
B31_DOPAMINE_PREDICTION_ERROR_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b31_dopamine_prediction_error_h48_bridge_policy/seed_7/best"
)
B32_ACTOR_CRITIC_VALUE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b32_actor_critic_value_h48_bridge_policy/seed_7/best"
)
B33_TD_ERROR_DECOMPOSITION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b33_td_error_decomposition_h48_bridge_policy/seed_7/best"
)
B34_ELIGIBILITY_CREDIT_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b34_eligibility_credit_h48_bridge_policy/seed_7/best"
)
B35_FORWARD_MODEL_VALUE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b35_forward_model_value_h48_bridge_policy/seed_7/best"
)
B36_LATENT_BELIEF_STATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b36_latent_belief_state_h48_bridge_policy/seed_7/best"
)
B37_STATE_FACTOR_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b37_state_factor_gate_h48_bridge_policy/seed_7/best"
)
B38_FACTOR_ATTENTION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b38_factor_attention_h48_bridge_policy/seed_7/best"
)
B39_ATTENTION_BINDING_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b39_attention_binding_h48_bridge_policy/seed_7/best"
)
B40_GLOBAL_WORKSPACE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b40_global_workspace_h48_bridge_policy/seed_7/best"
)
B41_EXECUTIVE_WORKSPACE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b41_executive_workspace_h48_bridge_policy/seed_7/best"
)
B42_ERROR_MONITOR_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b42_error_monitor_h48_bridge_policy/seed_7/best"
)
B43_ADAPTIVE_PRECISION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b43_adaptive_precision_h48_bridge_policy/seed_7/best"
)
B44_THALAMIC_RELAY_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b44_thalamic_relay_h48_bridge_policy/seed_7/best"
)
B45_RETICULAR_INHIBITION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b45_reticular_inhibition_h48_bridge_policy/seed_7/best"
)
B46_CORTICOTHALAMIC_FEEDBACK_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b46_corticothalamic_feedback_h48_bridge_policy/seed_7/best"
)
B47_OSCILLATORY_SYNCHRONY_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b47_oscillatory_synchrony_h48_bridge_policy/seed_7/best"
)
B48_CEREBELLAR_TIMING_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b48_cerebellar_timing_h48_bridge_policy/seed_7/best"
)
B49_STRIATAL_ACTION_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b49_striatal_action_gate_h48_bridge_policy/seed_7/best"
)
B50_HABIT_CHUNKING_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b50_habit_chunking_h48_bridge_policy/seed_7/best"
)
B51_DOPAMINERGIC_HABIT_MODULATION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b51_dopaminergic_habit_modulation_h48_bridge_policy/seed_7/best"
)
B52_CHOLINERGIC_PRECISION_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b52_cholinergic_precision_gate_h48_bridge_policy/seed_7/best"
)
B53_NORADRENERGIC_AROUSAL_GAIN_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b53_noradrenergic_arousal_gain_h48_bridge_policy/seed_7/best"
)
B54_SEROTONERGIC_PATIENCE_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b54_serotonergic_patience_gate_h48_bridge_policy/seed_7/best"
)
B55_HYPOTHALAMIC_DRIVE_COUPLING_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b55_hypothalamic_drive_coupling_h48_bridge_policy/seed_7/best"
)
B56_HPA_STRESS_AXIS_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b56_hpa_stress_axis_h48_bridge_policy/seed_7/best"
)
B57_INSULAR_INTEROCEPTIVE_AWARENESS_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b57_insular_interoceptive_awareness_h48_bridge_policy/seed_7/best"
)
B58_ACC_CONFLICT_MONITOR_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b58_acc_conflict_monitor_h48_bridge_policy/seed_7/best"
)
B59_PREFRONTAL_GOAL_CONTEXT_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b59_prefrontal_goal_context_h48_bridge_policy/seed_7/best"
)
B60_ORBITOFRONTAL_OUTCOME_VALUE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b60_orbitofrontal_outcome_value_h48_bridge_policy/seed_7/best"
)
B61_AMYGDALA_SAFETY_VALUE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b61_amygdala_safety_value_h48_bridge_policy/seed_7/best"
)
B62_DEFENSIVE_MODE_SELECTOR_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b62_defensive_mode_selector_h48_bridge_policy/seed_7/best"
)
B63_PERIAQUEDUCTAL_ESCAPE_SEQUENCE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b63_periaqueductal_escape_sequence_h48_bridge_policy/seed_7/best"
)
B64_VAGAL_RECOVERY_BRAKE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b64_vagal_recovery_brake_h48_bridge_policy/seed_7/best"
)
B65_ENTERIC_ASSIMILATION_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b65_enteric_assimilation_gate_h48_bridge_policy/seed_7/best"
)
B66_IMMUNE_MALAISE_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b66_immune_malaise_gate_h48_bridge_policy/seed_7/best"
)
B67_GLIAL_ENERGY_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b67_glial_energy_gate_h48_bridge_policy/seed_7/best"
)
B68_PROPRIOCEPTIVE_PACING_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b68_proprioceptive_pacing_gate_h48_bridge_policy/seed_7/best"
)
B69_VESTIBULAR_ORIENTATION_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b69_vestibular_orientation_gate_h48_bridge_policy/seed_7/best"
)
B70_OPTIC_FLOW_STABILIZATION_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b70_optic_flow_stabilization_gate_h48_bridge_policy/seed_7/best"
)
B71_TECTAL_ORIENTING_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b71_tectal_orienting_gate_h48_bridge_policy/seed_7/best"
)
B72_PULVINAR_ATTENTION_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b72_pulvinar_attention_gate_h48_bridge_policy/seed_7/best"
)
B73_RETICULAR_INHIBITION_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b73_reticular_inhibition_gate_h48_bridge_policy/seed_7/best"
)
B74_THALAMIC_REBOUND_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b74_thalamic_rebound_gate_h48_bridge_policy/seed_7/best"
)
B75_BASAL_THALAMIC_RELEASE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b75_basal_thalamic_release_h48_bridge_policy/seed_7/best"
)
B76_CEREBELLAR_STRIDE_GATE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b76_cerebellar_stride_gate_h48_bridge_policy/seed_7/best"
)
B77_OLIVARY_ERROR_CORRECTION_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b77_olivary_error_correction_h48_bridge_policy/seed_7/best"
)
B78_VESTIBULAR_BALANCE_DEFAULT_CHECKPOINT = (
    "artifacts/b_series/evolution/b78_vestibular_balance_h48_bridge_policy/seed_7/best"
)
B_CURRENT_BRIDGE_EFFECTIVE_LEVEL = "B0-current-simple"
B_CURRENT_BRIDGE_SELECTION_SOURCE = "legacy_direct_controller"
B1_THREAT_GUARD_EFFECTIVE_LEVEL = "B1-threat-guard"
B1_THREAT_GUARD_SELECTION_SOURCE = "b1_threat_guard_controller"
B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL = "B2-temporal-threat"
B2_TEMPORAL_THREAT_SELECTION_SOURCE = "b2_temporal_threat_controller"
B3_CONTACT_MEMORY_EFFECTIVE_LEVEL = "B3-contact-memory"
B3_CONTACT_MEMORY_SELECTION_SOURCE = "b3_contact_memory_controller"
B3_RECURRENT_GUARD_EFFECTIVE_LEVEL = "B3-recurrent-guard"
B3_RECURRENT_GUARD_SELECTION_SOURCE = "b3_recurrent_guard_controller"
B4_RECOVERY_BALANCE_EFFECTIVE_LEVEL = "B4-recovery-balance"
B4_RECOVERY_BALANCE_SELECTION_SOURCE = "b4_recovery_balance_controller"
B4_GENETIC_RECOVERY_SELECTION_SOURCE = "b4_genetic_recovery_controller"
B5_HOMEOSTATIC_ARBITER_EFFECTIVE_LEVEL = "B5-homeostatic-arbiter"
B5_HOMEOSTATIC_ARBITER_SELECTION_SOURCE = "b5_homeostatic_arbiter_controller"
B5_GENETIC_HOMEOSTASIS_SELECTION_SOURCE = "b5_genetic_homeostasis_controller"
B6_RISK_CORRIDOR_EFFECTIVE_LEVEL = "B6-risk-corridor"
B6_RISK_CORRIDOR_SELECTION_SOURCE = "b6_risk_corridor_controller"
B6_RECURRENT_MEMORY_EFFECTIVE_LEVEL = "B6-recurrent-memory"
B6_RECURRENT_MEMORY_SELECTION_SOURCE = "b6_recurrent_memory_controller"
B6_FUSED_RISK_RECURRENT_EFFECTIVE_LEVEL = "B6-fused-risk-recurrent"
B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE = "b6_fused_risk_recurrent_controller"
B7_AFFORDANCE_BUDGET_EFFECTIVE_LEVEL = "B7-affordance-budget"
B7_AFFORDANCE_BUDGET_SELECTION_SOURCE = "b7_affordance_budget_controller"
B8_SPATIAL_AFFORDANCE_EFFECTIVE_LEVEL = "B8-spatial-affordance-map"
B8_SPATIAL_AFFORDANCE_SELECTION_SOURCE = "b8_spatial_affordance_controller"
B9_WAYPOINT_PLANNER_EFFECTIVE_LEVEL = "B9-waypoint-planner"
B9_WAYPOINT_PLANNER_SELECTION_SOURCE = "b9_waypoint_planner_controller"
B10_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL = "B10-prospective-replay"
B10_PROSPECTIVE_REPLAY_SELECTION_SOURCE = "b10_prospective_replay_controller"
B11_CONFIDENCE_ARBITER_EFFECTIVE_LEVEL = "B11-confidence-arbiter"
B11_CONFIDENCE_ARBITER_SELECTION_SOURCE = "b11_confidence_arbiter_controller"
B12_PREDICTIVE_ATTENTION_EFFECTIVE_LEVEL = "B12-predictive-attention"
B12_PREDICTIVE_ATTENTION_SELECTION_SOURCE = "b12_predictive_attention_controller"
B13_LOCAL_SEARCH_EFFECTIVE_LEVEL = "B13-local-affordance-search"
B13_LOCAL_SEARCH_SELECTION_SOURCE = "b13_local_affordance_search_controller"
B14_AFFORDANCE_UNCERTAINTY_EFFECTIVE_LEVEL = "B14-affordance-uncertainty"
B14_AFFORDANCE_UNCERTAINTY_SELECTION_SOURCE = "b14_affordance_uncertainty_controller"
B15_OPTION_CRITIC_EFFECTIVE_LEVEL = "B15-option-critic"
B15_OPTION_CRITIC_SELECTION_SOURCE = "b15_option_critic_controller"
B16_OPTION_ENSEMBLE_EFFECTIVE_LEVEL = "B16-option-ensemble"
B16_OPTION_ENSEMBLE_SELECTION_SOURCE = "b16_option_ensemble_controller"
B17_NEUROMODULATED_ENSEMBLE_EFFECTIVE_LEVEL = "B17-neuromodulated-ensemble"
B17_NEUROMODULATED_ENSEMBLE_SELECTION_SOURCE = (
    "b17_neuromodulated_ensemble_controller"
)
B18_ELIGIBILITY_TRACE_EFFECTIVE_LEVEL = "B18-eligibility-trace"
B18_ELIGIBILITY_TRACE_SELECTION_SOURCE = "b18_eligibility_trace_controller"
B19_EPISODIC_META_MEMORY_EFFECTIVE_LEVEL = "B19-episodic-meta-memory"
B19_EPISODIC_META_MEMORY_SELECTION_SOURCE = "b19_episodic_meta_memory_controller"
B20_WORKING_MEMORY_GATE_EFFECTIVE_LEVEL = "B20-working-memory-gate"
B20_WORKING_MEMORY_GATE_SELECTION_SOURCE = "b20_working_memory_gate_controller"
B21_HIPPOCAMPAL_REPLAY_EFFECTIVE_LEVEL = "B21-hippocampal-replay"
B21_HIPPOCAMPAL_REPLAY_SELECTION_SOURCE = "b21_hippocampal_replay_controller"
B22_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL = "B22-prospective-map-replay"
B22_PROSPECTIVE_REPLAY_SELECTION_SOURCE = "b22_prospective_replay_controller"
B23_CONFLICT_MONITOR_EFFECTIVE_LEVEL = "B23-conflict-monitor"
B23_CONFLICT_MONITOR_SELECTION_SOURCE = "b23_conflict_monitor_controller"
B24_PRECISION_CONFLICT_EFFECTIVE_LEVEL = "B24-precision-conflict"
B24_PRECISION_CONFLICT_SELECTION_SOURCE = "b24_precision_conflict_controller"
B25_METACOGNITIVE_CONFIDENCE_EFFECTIVE_LEVEL = "B25-metacognitive-confidence"
B25_METACOGNITIVE_CONFIDENCE_SELECTION_SOURCE = (
    "b25_metacognitive_confidence_controller"
)
B26_ALLOSTATIC_PREDICTION_EFFECTIVE_LEVEL = "B26-allostatic-prediction"
B26_ALLOSTATIC_PREDICTION_SELECTION_SOURCE = "b26_allostatic_prediction_controller"
B27_AROUSAL_GAIN_EFFECTIVE_LEVEL = "B27-arousal-gain"
B27_AROUSAL_GAIN_SELECTION_SOURCE = "b27_arousal_gain_controller"
B28_INTEROCEPTIVE_ATTENTION_EFFECTIVE_LEVEL = "B28-interoceptive-attention"
B28_INTEROCEPTIVE_ATTENTION_SELECTION_SOURCE = (
    "b28_interoceptive_attention_controller"
)
B29_SALIENCE_COMPETITION_EFFECTIVE_LEVEL = "B29-salience-competition"
B29_SALIENCE_COMPETITION_SELECTION_SOURCE = "b29_salience_competition_controller"
B30_BASAL_GANGLIA_GATE_EFFECTIVE_LEVEL = "B30-basal-ganglia-gate"
B30_BASAL_GANGLIA_GATE_SELECTION_SOURCE = "b30_basal_ganglia_gate_controller"
B31_DOPAMINE_PREDICTION_ERROR_EFFECTIVE_LEVEL = "B31-dopamine-prediction-error"
B31_DOPAMINE_PREDICTION_ERROR_SELECTION_SOURCE = (
    "b31_dopamine_prediction_error_controller"
)
B32_ACTOR_CRITIC_VALUE_EFFECTIVE_LEVEL = "B32-actor-critic-value"
B32_ACTOR_CRITIC_VALUE_SELECTION_SOURCE = "b32_actor_critic_value_controller"
B33_TD_ERROR_DECOMPOSITION_EFFECTIVE_LEVEL = "B33-td-error-decomposition"
B33_TD_ERROR_DECOMPOSITION_SELECTION_SOURCE = "b33_td_error_decomposition_controller"
B34_ELIGIBILITY_CREDIT_EFFECTIVE_LEVEL = "B34-eligibility-credit"
B34_ELIGIBILITY_CREDIT_SELECTION_SOURCE = "b34_eligibility_credit_controller"
B35_FORWARD_MODEL_VALUE_EFFECTIVE_LEVEL = "B35-forward-model-value"
B35_FORWARD_MODEL_VALUE_SELECTION_SOURCE = "b35_forward_model_value_controller"
B36_LATENT_BELIEF_STATE_EFFECTIVE_LEVEL = "B36-latent-belief-state"
B36_LATENT_BELIEF_STATE_SELECTION_SOURCE = "b36_latent_belief_state_controller"
B37_STATE_FACTOR_GATE_EFFECTIVE_LEVEL = "B37-state-factor-gate"
B37_STATE_FACTOR_GATE_SELECTION_SOURCE = "b37_state_factor_gate_controller"
B38_FACTOR_ATTENTION_EFFECTIVE_LEVEL = "B38-factor-attention"
B38_FACTOR_ATTENTION_SELECTION_SOURCE = "b38_factor_attention_controller"
B39_ATTENTION_BINDING_EFFECTIVE_LEVEL = "B39-attention-binding"
B39_ATTENTION_BINDING_SELECTION_SOURCE = "b39_attention_binding_controller"
B40_GLOBAL_WORKSPACE_EFFECTIVE_LEVEL = "B40-global-workspace"
B40_GLOBAL_WORKSPACE_SELECTION_SOURCE = "b40_global_workspace_controller"
B41_EXECUTIVE_WORKSPACE_EFFECTIVE_LEVEL = "B41-executive-workspace"
B41_EXECUTIVE_WORKSPACE_SELECTION_SOURCE = "b41_executive_workspace_controller"
B42_ERROR_MONITOR_EFFECTIVE_LEVEL = "B42-error-monitor"
B42_ERROR_MONITOR_SELECTION_SOURCE = "b42_error_monitor_controller"
B43_ADAPTIVE_PRECISION_EFFECTIVE_LEVEL = "B43-adaptive-precision"
B43_ADAPTIVE_PRECISION_SELECTION_SOURCE = "b43_adaptive_precision_controller"
B44_THALAMIC_RELAY_EFFECTIVE_LEVEL = "B44-thalamic-relay"
B44_THALAMIC_RELAY_SELECTION_SOURCE = "b44_thalamic_relay_controller"
B45_RETICULAR_INHIBITION_EFFECTIVE_LEVEL = "B45-reticular-inhibition"
B45_RETICULAR_INHIBITION_SELECTION_SOURCE = "b45_reticular_inhibition_controller"
B46_CORTICOTHALAMIC_FEEDBACK_EFFECTIVE_LEVEL = "B46-corticothalamic-feedback"
B46_CORTICOTHALAMIC_FEEDBACK_SELECTION_SOURCE = (
    "b46_corticothalamic_feedback_controller"
)
B47_OSCILLATORY_SYNCHRONY_EFFECTIVE_LEVEL = "B47-oscillatory-synchrony"
B47_OSCILLATORY_SYNCHRONY_SELECTION_SOURCE = (
    "b47_oscillatory_synchrony_controller"
)
B48_CEREBELLAR_TIMING_EFFECTIVE_LEVEL = "B48-cerebellar-timing"
B48_CEREBELLAR_TIMING_SELECTION_SOURCE = "b48_cerebellar_timing_controller"
B49_STRIATAL_ACTION_GATE_EFFECTIVE_LEVEL = "B49-striatal-action-gate"
B49_STRIATAL_ACTION_GATE_SELECTION_SOURCE = "b49_striatal_action_gate_controller"
B50_HABIT_CHUNKING_EFFECTIVE_LEVEL = "B50-habit-chunking"
B50_HABIT_CHUNKING_SELECTION_SOURCE = "b50_habit_chunking_controller"
B51_DOPAMINERGIC_HABIT_MODULATION_EFFECTIVE_LEVEL = "B51-dopaminergic-habit-modulation"
B51_DOPAMINERGIC_HABIT_MODULATION_SELECTION_SOURCE = (
    "b51_dopaminergic_habit_modulation_controller"
)
B52_CHOLINERGIC_PRECISION_GATE_EFFECTIVE_LEVEL = "B52-cholinergic-precision-gate"
B52_CHOLINERGIC_PRECISION_GATE_SELECTION_SOURCE = (
    "b52_cholinergic_precision_gate_controller"
)
B53_NORADRENERGIC_AROUSAL_GAIN_EFFECTIVE_LEVEL = "B53-noradrenergic-arousal-gain"
B53_NORADRENERGIC_AROUSAL_GAIN_SELECTION_SOURCE = (
    "b53_noradrenergic_arousal_gain_controller"
)
B54_SEROTONERGIC_PATIENCE_GATE_EFFECTIVE_LEVEL = "B54-serotonergic-patience-gate"
B54_SEROTONERGIC_PATIENCE_GATE_SELECTION_SOURCE = (
    "b54_serotonergic_patience_gate_controller"
)
B55_HYPOTHALAMIC_DRIVE_COUPLING_EFFECTIVE_LEVEL = "B55-hypothalamic-drive-coupling"
B55_HYPOTHALAMIC_DRIVE_COUPLING_SELECTION_SOURCE = (
    "b55_hypothalamic_drive_coupling_controller"
)
B56_HPA_STRESS_AXIS_EFFECTIVE_LEVEL = "B56-hpa-stress-axis"
B56_HPA_STRESS_AXIS_SELECTION_SOURCE = "b56_hpa_stress_axis_controller"
B57_INSULAR_INTEROCEPTIVE_AWARENESS_EFFECTIVE_LEVEL = (
    "B57-insular-interoceptive-awareness"
)
B57_INSULAR_INTEROCEPTIVE_AWARENESS_SELECTION_SOURCE = (
    "b57_insular_interoceptive_awareness_controller"
)
B58_ACC_CONFLICT_MONITOR_EFFECTIVE_LEVEL = "B58-acc-conflict-monitor"
B58_ACC_CONFLICT_MONITOR_SELECTION_SOURCE = "b58_acc_conflict_monitor_controller"
B59_PREFRONTAL_GOAL_CONTEXT_EFFECTIVE_LEVEL = "B59-prefrontal-goal-context"
B59_PREFRONTAL_GOAL_CONTEXT_SELECTION_SOURCE = "b59_prefrontal_goal_context_controller"
B60_ORBITOFRONTAL_OUTCOME_VALUE_EFFECTIVE_LEVEL = "B60-orbitofrontal-outcome-value"
B60_ORBITOFRONTAL_OUTCOME_VALUE_SELECTION_SOURCE = (
    "b60_orbitofrontal_outcome_value_controller"
)
B61_AMYGDALA_SAFETY_VALUE_EFFECTIVE_LEVEL = "B61-amygdala-safety-value"
B61_AMYGDALA_SAFETY_VALUE_SELECTION_SOURCE = "b61_amygdala_safety_value_controller"
B62_DEFENSIVE_MODE_SELECTOR_EFFECTIVE_LEVEL = "B62-defensive-mode-selector"
B62_DEFENSIVE_MODE_SELECTOR_SELECTION_SOURCE = "b62_defensive_mode_selector_controller"
B63_PERIAQUEDUCTAL_ESCAPE_EFFECTIVE_LEVEL = "B63-periaqueductal-escape"
B63_PERIAQUEDUCTAL_ESCAPE_SELECTION_SOURCE = "b63_periaqueductal_escape_controller"
B64_VAGAL_RECOVERY_BRAKE_EFFECTIVE_LEVEL = "B64-vagal-recovery-brake"
B64_VAGAL_RECOVERY_BRAKE_SELECTION_SOURCE = "b64_vagal_recovery_brake_controller"
B65_ENTERIC_ASSIMILATION_EFFECTIVE_LEVEL = "B65-enteric-assimilation-gate"
B65_ENTERIC_ASSIMILATION_SELECTION_SOURCE = "b65_enteric_assimilation_controller"
B66_IMMUNE_MALAISE_EFFECTIVE_LEVEL = "B66-immune-malaise-gate"
B66_IMMUNE_MALAISE_SELECTION_SOURCE = "b66_immune_malaise_controller"
B67_GLIAL_ENERGY_EFFECTIVE_LEVEL = "B67-glial-energy-gate"
B67_GLIAL_ENERGY_SELECTION_SOURCE = "b67_glial_homeostasis_controller"
B68_MOTOR_PACING_EFFECTIVE_LEVEL = "B68-motor-pacing-gate"
B68_MOTOR_PACING_SELECTION_SOURCE = "b68_proprioceptive_pacing_controller"
B69_VESTIBULAR_ORIENTATION_EFFECTIVE_LEVEL = "B69-vestibular-orientation-gate"
B69_VESTIBULAR_ORIENTATION_SELECTION_SOURCE = "b69_vestibular_orientation_controller"
B70_OPTIC_FLOW_EFFECTIVE_LEVEL = "B70-optic-flow-stabilization-gate"
B70_OPTIC_FLOW_SELECTION_SOURCE = "b70_optic_flow_controller"
B71_TECTAL_ORIENTING_EFFECTIVE_LEVEL = "B71-tectal-orienting-gate"
B71_TECTAL_ORIENTING_SELECTION_SOURCE = "b71_tectal_orienting_controller"
B72_PULVINAR_ATTENTION_EFFECTIVE_LEVEL = "B72-pulvinar-attention-gate"
B72_PULVINAR_ATTENTION_SELECTION_SOURCE = "b72_pulvinar_attention_controller"
B73_RETICULAR_INHIBITION_EFFECTIVE_LEVEL = "B73-reticular-inhibition-gate"
B73_RETICULAR_INHIBITION_SELECTION_SOURCE = "b73_reticular_inhibition_controller"
B74_THALAMIC_REBOUND_EFFECTIVE_LEVEL = "B74-thalamic-rebound-gate"
B74_THALAMIC_REBOUND_SELECTION_SOURCE = "b74_thalamic_rebound_controller"
B75_BASAL_THALAMIC_RELEASE_EFFECTIVE_LEVEL = "B75-basal-thalamic-release"
B75_BASAL_THALAMIC_RELEASE_SELECTION_SOURCE = "b75_basal_thalamic_release_controller"
B76_CEREBELLAR_STRIDE_EFFECTIVE_LEVEL = "B76-cerebellar-stride-gate"
B76_CEREBELLAR_STRIDE_SELECTION_SOURCE = "b76_cerebellar_stride_controller"
B77_OLIVARY_ERROR_EFFECTIVE_LEVEL = "B77-olivary-error-correction"
B77_OLIVARY_ERROR_SELECTION_SOURCE = "b77_olivary_error_controller"
B78_VESTIBULAR_BALANCE_EFFECTIVE_LEVEL = "B78-vestibular-balance"
B78_VESTIBULAR_BALANCE_SELECTION_SOURCE = "b78_vestibular_balance_controller"
LOCOMOTION_ACTIONS: tuple[str, ...] = (
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "STAY",
    "ORIENT_UP",
    "ORIENT_DOWN",
    "ORIENT_LEFT",
    "ORIENT_RIGHT",
)
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(LOCOMOTION_ACTIONS)}
B_SEMANTIC_ACTIONS: tuple[str, ...] = (
    "MOVE_TO_FOOD",
    "MOVE_TO_SHELTER",
    "EXPLORE",
    "STAY",
    "EAT",
    "SLEEP",
)
B_SEMANTIC_ACTION_TO_INDEX = {
    name: idx for idx, name in enumerate(B_SEMANTIC_ACTIONS)
}
B_SERIES_MODES: tuple[str, ...] = ("legacy_semantic", "current_bridge")
BRIDGE_MOVE_ACTIONS: tuple[str, ...] = (
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
)
BRIDGE_ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "MOVE_UP": (0, -1),
    "MOVE_DOWN": (0, 1),
    "MOVE_LEFT": (-1, 0),
    "MOVE_RIGHT": (1, 0),
}
BRIDGE_EXPLORE_ORDER: tuple[str, ...] = (
    "MOVE_RIGHT",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_UP",
)


@dataclass(frozen=True)
class BSeriesBridgeDecision:
    semantic_action: str
    primitive_action: str
    reason: str
    blocked_mask: dict[str, bool]
    food_delta_used: float
    shelter_delta_used: float
    external_override_count: int = 0

    @property
    def primitive_action_idx(self) -> int:
        return int(ACTION_TO_INDEX[self.primitive_action])


def _meta_from_observation(observation: Mapping[str, object]) -> Mapping[str, object]:
    meta = observation.get("meta")
    return meta if isinstance(meta, Mapping) else {}


def _submapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def b_series_blocked_mask(
    observation: Mapping[str, object],
) -> dict[str, bool]:
    local_ecology = LocalEcologyObservationAdapter.from_observation(observation)
    blocked: dict[str, bool] = {}
    for action_name in LOCOMOTION_ACTIONS:
        if action_name in BRIDGE_MOVE_ACTIONS:
            blocked[action_name] = local_ecology.affordance_for(action_name).blocked
        else:
            blocked[action_name] = False
    return blocked


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return float(numeric)


def _movement_candidates(blocked_mask: Mapping[str, bool]) -> list[str]:
    return [
        action_name
        for action_name in BRIDGE_MOVE_ACTIONS
        if not bool(blocked_mask.get(action_name, False))
    ]


def _transition_for(
    local_ecology: LocalEcologyObservationAdapter,
    action_name: str,
):
    return local_ecology.transition_for(action_name)


def _geodesic_for(
    local_ecology: LocalEcologyObservationAdapter,
    action_name: str,
):
    return local_ecology.geodesic_for(action_name)


def _affordance_for(
    local_ecology: LocalEcologyObservationAdapter,
    action_name: str,
):
    return local_ecology.affordance_for(action_name)


def _target_vector(
    meta: Mapping[str, object],
    target_name: str,
) -> tuple[float, float, str]:
    vision = _submapping(_submapping(meta.get("vision")).get(target_name))
    if (
        _float_value(vision.get("visible"), 0.0) > 0.0
        and _float_value(vision.get("certainty"), 0.0) > 0.0
    ):
        dx = _float_value(vision.get("dx"), 0.0)
        dy = _float_value(vision.get("dy"), 0.0)
        if abs(dx) + abs(dy) >= 0.05:
            return dx, dy, f"{target_name}_vision_vector"

    memory = _submapping(_submapping(meta.get("memory_vectors")).get(target_name))
    if _float_value(memory.get("age"), 1.0) < 1.0:
        dx = _float_value(memory.get("dx"), 0.0)
        dy = _float_value(memory.get("dy"), 0.0)
        if abs(dx) + abs(dy) >= 0.05:
            return dx, dy, f"{target_name}_memory_vector"

    trace = _submapping(_submapping(meta.get("percept_traces")).get(target_name))
    dx = _float_value(trace.get("dx"), 0.0)
    dy = _float_value(trace.get("dy"), 0.0)
    strength = max(
        _float_value(trace.get("strength"), 0.0),
        _float_value(trace.get("freshness"), 0.0),
        _float_value(trace.get("confidence"), 0.0),
    )
    if strength > 0.0 and abs(dx) + abs(dy) >= 0.05:
        return dx, dy, f"{target_name}_trace_vector"

    return 0.0, 0.0, "no_target_vector"


def _predator_pressure(meta: Mapping[str, object]) -> float:
    return max(
        _float_value(meta.get("predator_smell_strength"), 0.0),
        _float_value(meta.get("visual_predator_threat"), 0.0),
        _float_value(meta.get("olfactory_predator_threat"), 0.0),
        _float_value(meta.get("predator_motion_salience"), 0.0),
        1.0 if bool(meta.get("predator_visible", False)) else 0.0,
    )


def _best_exit_action(
    meta: Mapping[str, object],
    local_ecology: LocalEcologyObservationAdapter,
    candidates: Sequence[str],
) -> tuple[str, float, str]:
    best_exit_action = "STAY"
    best_exit_delta = 0.0
    best_exit_score = -1e9
    predator_pressure = _predator_pressure(meta)
    for action_name in candidates:
        transition = _transition_for(local_ecology, action_name)
        geodesic = _geodesic_for(local_ecology, action_name)
        exit_delta = geodesic.exit_geodesic_delta
        predator_delta = transition.predator_dist_delta
        score = exit_delta + (1.00 * predator_pressure * predator_delta)
        if predator_pressure >= 0.50 and predator_delta <= -0.5:
            score -= 4.0
        if geodesic.next_on_exit_target:
            score += 0.75
        if score > best_exit_score:
            best_exit_score = score
            best_exit_action = action_name
            best_exit_delta = transition.food_dist_delta
    if best_exit_score > 0.0:
        return best_exit_action, best_exit_delta, "food_exit_to_outside"
    return "STAY", 0.0, "no_exit_progress"


def _vector_guided_food_action(
    meta: Mapping[str, object],
    local_ecology: LocalEcologyObservationAdapter,
    candidates: Sequence[str],
) -> tuple[str, float, str] | None:
    target_dx, target_dy, source = _target_vector(meta, "food")
    if source == "no_target_vector":
        return None

    best_action = "STAY"
    best_delta = 0.0
    best_score = -1e9
    current_role = str(meta.get("shelter_role", "outside"))
    predator_pressure = _predator_pressure(meta)
    for action_name in candidates:
        action_dx, action_dy = BRIDGE_ACTION_DELTAS[action_name]
        transition = _transition_for(local_ecology, action_name)
        affordance = _affordance_for(local_ecology, action_name)
        next_role = affordance.next_role
        next_has_food = transition.next_cell_has_food
        food_delta = transition.food_dist_delta
        predator_delta = transition.predator_dist_delta
        score = (
            action_dx * target_dx
            + action_dy * target_dy
            + 0.20 * food_delta
            + (0.60 * predator_pressure * predator_delta)
        )
        if next_has_food:
            score += 4.0
        if predator_pressure >= 0.50 and predator_delta <= -0.5:
            score -= 5.0
        if not next_has_food and current_role == "outside" and next_role != "outside":
            score -= 1.25
        if (
            not next_has_food
            and current_role == "entrance"
            and next_role in {"inside", "deep"}
        ):
            score -= 1.75
        if score > best_score:
            best_score = score
            best_action = action_name
            best_delta = food_delta

    if best_action == "STAY" or best_score <= 0.0:
        return None
    return best_action, best_delta, source


def _best_food_action(
    meta: Mapping[str, object],
    local_ecology: LocalEcologyObservationAdapter,
    candidates: Sequence[str],
) -> tuple[str, float, str]:
    if bool(meta.get("on_shelter", False)) and str(meta.get("shelter_role", "outside")) not in {
        "outside",
        "entrance",
    }:
        exit_action = _best_exit_action(meta, local_ecology, candidates)
        if exit_action[0] != "STAY":
            return exit_action
        if _predator_pressure(meta) >= 0.50:
            return exit_action

    vector_action = _vector_guided_food_action(meta, local_ecology, candidates)
    if vector_action is not None:
        return vector_action

    best_action = "STAY"
    best_delta = 0.0
    best_score = -1e9
    current_role = str(meta.get("shelter_role", "outside"))
    for action_name in candidates:
        transition = _transition_for(local_ecology, action_name)
        affordance = _affordance_for(local_ecology, action_name)
        delta = transition.food_dist_delta
        predator_delta = transition.predator_dist_delta
        next_has_food = transition.next_cell_has_food
        score = delta + 0.40 * predator_delta
        next_role = affordance.next_role
        if not next_has_food and current_role == "outside" and next_role != "outside":
            score -= 1.50
        elif (
            not next_has_food
            and current_role == "entrance"
            and next_role in {"inside", "deep"}
        ):
            score -= 2.00
        if next_has_food:
            score += 2.0
        if score > best_score:
            best_score = score
            best_action = action_name
            best_delta = delta
    if best_score <= -1e8:
        return "STAY", 0.0, "no_food_candidate"
    return best_action, best_delta, "food_progress"


def _best_shelter_action(
    meta: Mapping[str, object],
    local_ecology: LocalEcologyObservationAdapter,
    candidates: Sequence[str],
) -> tuple[str, float, str]:
    best_action = "STAY"
    best_delta = 0.0
    best_score = -1e9
    predator_pressure = _predator_pressure(meta)
    for action_name in candidates:
        transition = _transition_for(local_ecology, action_name)
        geodesic = _geodesic_for(local_ecology, action_name)
        shelter_delta = transition.shelter_dist_delta
        predator_delta = transition.predator_dist_delta
        exit_delta = geodesic.exit_geodesic_delta
        deep_delta = geodesic.deep_geodesic_delta
        score = (
            shelter_delta
            + 0.35 * exit_delta
            + 0.50 * deep_delta
            + (1.25 * predator_pressure * predator_delta)
        )
        if predator_pressure >= 0.50 and predator_delta <= -0.5:
            score -= 4.0
        if geodesic.next_on_deep_target:
            score += 1.0
        elif geodesic.next_on_exit_target:
            score += 0.5
        if score > best_score:
            best_score = score
            best_action = action_name
            best_delta = shelter_delta
    if best_score <= -1e8:
        return "STAY", 0.0, "no_shelter_candidate"
    return best_action, best_delta, "shelter_progress"


def _explore_action(
    candidates: Sequence[str],
    *,
    rng: np.random.Generator | None,
    sample: bool,
) -> tuple[str, str]:
    if not candidates:
        return "STAY", "explore_no_unblocked_move"
    if sample and rng is not None:
        order = list(candidates)
        rng.shuffle(order)
        return str(order[0]), "explore_seeded_shuffle"
    for action_name in BRIDGE_EXPLORE_ORDER:
        if action_name in candidates:
            return action_name, "explore_deterministic_order"
    return str(candidates[0]), "explore_first_unblocked"


def bridge_b_semantic_action(
    semantic_action: str,
    observation: Mapping[str, object],
    *,
    rng: np.random.Generator | None = None,
    sample: bool = False,
) -> BSeriesBridgeDecision:
    if semantic_action not in B_SEMANTIC_ACTION_TO_INDEX:
        raise ValueError(f"Unknown B-series semantic action: {semantic_action!r}.")
    meta = _meta_from_observation(observation)
    local_ecology = LocalEcologyObservationAdapter.from_meta(meta)
    blocked_mask = b_series_blocked_mask(observation)
    candidates = _movement_candidates(blocked_mask)
    food_delta = 0.0
    shelter_delta = 0.0

    if semantic_action == "MOVE_TO_FOOD":
        if bool(meta.get("on_food", False)):
            primitive_action = "STAY"
            reason = "already_on_food"
        else:
            primitive_action, food_delta, reason = _best_food_action(
                meta,
                local_ecology,
                candidates,
            )
    elif semantic_action == "MOVE_TO_SHELTER":
        shelter_role = str(meta.get("shelter_role", "outside"))
        shelter_role_level = _float_value(meta.get("shelter_role_level"), 0.0)
        if bool(meta.get("on_shelter", False)) and (
            shelter_role == "deep" or shelter_role_level >= 0.95
        ):
            primitive_action = "STAY"
            reason = "already_deep_shelter"
        elif bool(meta.get("on_shelter", False)) and not candidates:
            primitive_action = "STAY"
            reason = "shelter_hold_no_unblocked_move"
        else:
            primitive_action, shelter_delta, reason = _best_shelter_action(
                meta,
                local_ecology,
                candidates,
            )
    elif semantic_action == "EXPLORE":
        primitive_action, reason = _explore_action(
            candidates,
            rng=rng,
            sample=sample,
        )
    elif semantic_action in {"STAY", "EAT", "SLEEP"}:
        primitive_action = "STAY"
        reason = f"{semantic_action.lower()}_maps_to_stay"
    else:
        primitive_action = "STAY"
        reason = "fallback_stay"

    if bool(blocked_mask.get(primitive_action, False)):
        primitive_action = "STAY"
        reason = f"{reason}_blocked_to_stay"
    return BSeriesBridgeDecision(
        semantic_action=semantic_action,
        primitive_action=primitive_action,
        reason=reason,
        blocked_mask=dict(blocked_mask),
        food_delta_used=float(food_delta),
        shelter_delta_used=float(shelter_delta),
        external_override_count=0,
    )


__all__ = [
    *(name for name in globals() if name.isupper()),
    "BSeriesBridgeDecision",
    "bridge_b_semantic_action",
    "b_series_blocked_mask",
]
