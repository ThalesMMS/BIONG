from __future__ import annotations

from .config import *
from .config import _arbitration_fields
from ..b_series import (
    B0_CURRENT_BRIDGE_DEFAULT_CHECKPOINT,
    B0_CURRENT_BRIDGE_POLICY_NAME,
    B1_CAPACITY_H48_POLICY_NAME,
    B1_CAPACITY_H64_POLICY_NAME,
    B1_THREAT_GUARD_DEFAULT_CHECKPOINT,
    B1_THREAT_GUARD_POLICY_NAME,
    B2_TEMPORAL_THREAT_DEFAULT_CHECKPOINT,
    B2_TEMPORAL_THREAT_H48_POLICY_NAME,
    B2_TEMPORAL_THREAT_H56_POLICY_NAME,
    B2_TEMPORAL_THREAT_H64_POLICY_NAME,
    B3_CONTACT_MEMORY_H48_POLICY_NAME,
    B3_CONTACT_MEMORY_H56_POLICY_NAME,
    B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME,
    B3_RECURRENT_GUARD_DEFAULT_CHECKPOINT,
    B3_RECURRENT_GUARD_H48_POLICY_NAME,
    B4_GENETIC_RECOVERY_H48_POLICY_NAME,
    B4_GENETIC_RECOVERY_DEFAULT_CHECKPOINT,
    B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME,
    B4_RECOVERY_BALANCE_H48_POLICY_NAME,
    B4_RECOVERY_BALANCE_H56_POLICY_NAME,
    B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME,
    B5_GENETIC_HOMEOSTASIS_DEFAULT_CHECKPOINT,
    B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
    B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
    B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME,
    B6_CORRIDOR_SURVIVAL_GUARD_H48_POLICY_NAME,
    B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
    B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME,
    B6_GENETIC_RISK_CORRIDOR_H48_POLICY_NAME,
    B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
    B6_RECURRENT_CONTEXT_H56_POLICY_NAME,
    B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME,
    B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME,
    B6_RISK_CORRIDOR_H56_POLICY_NAME,
    B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
    B6_THREAT_PRIORITY_MEMORY_H48_POLICY_NAME,
    B6_FUSED_RISK_RECURRENT_DEFAULT_CHECKPOINT,
    B7_AFFORDANCE_BUDGET_DEFAULT_CHECKPOINT,
    B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
    B7_AFFORDANCE_BUDGET_H56_POLICY_NAME,
    B7_ENERGY_BUDGET_CORRIDOR_H48_POLICY_NAME,
    B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME,
    B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME,
    B8_CORRIDOR_PLACE_MEMORY_H48_POLICY_NAME,
    B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME,
    B8_RETURN_VECTOR_H48_POLICY_NAME,
    B8_SPATIAL_AFFORDANCE_DEFAULT_CHECKPOINT,
    B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
    B8_SPATIAL_AFFORDANCE_MAP_H56_POLICY_NAME,
    B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME,
    B9_PATH_INTEGRATION_H48_POLICY_NAME,
    B9_ROUTE_MEMORY_H48_POLICY_NAME,
    B9_WAYPOINT_PLANNER_DEFAULT_CHECKPOINT,
    B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
    B9_WAYPOINT_PLANNER_H56_POLICY_NAME,
    B10_GENETIC_REPLAY_PLANNER_H48_POLICY_NAME,
    B10_PROSPECTIVE_REPLAY_DEFAULT_CHECKPOINT,
    B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
    B10_PROSPECTIVE_REPLAY_H56_POLICY_NAME,
    B10_REPLAY_PLANNER_H48_POLICY_NAME,
    B10_VALUE_ROUTE_EVALUATOR_H48_POLICY_NAME,
    B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
    B11_CONFIDENCE_ARBITER_H56_POLICY_NAME,
    B11_CONFIDENCE_ARBITER_DEFAULT_CHECKPOINT,
    B11_GENETIC_CONFIDENCE_GATE_H48_POLICY_NAME,
    B11_NEUROMODULATED_REPLAY_H48_POLICY_NAME,
    B11_UNCERTAINTY_GATE_H48_POLICY_NAME,
    B12_ACTIVE_INFERENCE_GATE_H48_POLICY_NAME,
    B12_AFFORDANCE_ATTENTION_H48_POLICY_NAME,
    B12_GENETIC_ATTENTION_GATE_H48_POLICY_NAME,
    B12_PREDICTIVE_ATTENTION_DEFAULT_CHECKPOINT,
    B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
    B12_PREDICTIVE_ATTENTION_H56_POLICY_NAME,
    B13_AFFORDANCE_SAMPLER_H48_POLICY_NAME,
    B13_COUNTERFACTUAL_ROUTE_H48_POLICY_NAME,
    B13_GENETIC_LOCAL_SEARCH_H48_POLICY_NAME,
    B13_LOCAL_AFFORDANCE_SEARCH_DEFAULT_CHECKPOINT,
    B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
    B13_LOCAL_AFFORDANCE_SEARCH_H56_POLICY_NAME,
    B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
    B14_AFFORDANCE_UNCERTAINTY_H56_POLICY_NAME,
    B14_CONFIDENCE_WEIGHTED_ROUTE_H48_POLICY_NAME,
    B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME,
    B14_RISK_CALIBRATED_SEARCH_H48_POLICY_NAME,
    B14_AFFORDANCE_UNCERTAINTY_DEFAULT_CHECKPOINT,
    B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME,
    B15_OPTION_CRITIC_DEFAULT_CHECKPOINT,
    B15_OPTION_CRITIC_H48_POLICY_NAME,
    B15_OPTION_CRITIC_H56_POLICY_NAME,
    B15_PERSISTENCE_GATE_H48_POLICY_NAME,
    B15_VALUE_GATED_OPTION_H48_POLICY_NAME,
    B16_ACTION_SET_VOTER_H48_POLICY_NAME,
    B16_COMPETING_OPTIONS_H48_POLICY_NAME,
    B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME,
    B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
    B16_OPTION_ENSEMBLE_H56_POLICY_NAME,
    B16_OPTION_ENSEMBLE_DEFAULT_CHECKPOINT,
    B17_AROUSAL_GATED_OPTIONS_H48_POLICY_NAME,
    B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
    B17_HOMEOSTATIC_MODULATOR_H48_POLICY_NAME,
    B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
    B17_NEUROMODULATED_ENSEMBLE_H56_POLICY_NAME,
    B17_NEUROMODULATED_ENSEMBLE_DEFAULT_CHECKPOINT,
    B18_ELIGIBILITY_TRACE_DEFAULT_CHECKPOINT,
    B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
    B18_ELIGIBILITY_TRACE_H56_POLICY_NAME,
    B18_GENETIC_ELIGIBILITY_TRACE_H48_POLICY_NAME,
    B18_METASTABLE_AROUSAL_H48_POLICY_NAME,
    B18_SYNAPTIC_TRACE_MODULATOR_H48_POLICY_NAME,
    B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
    B19_EPISODIC_META_MEMORY_H56_POLICY_NAME,
    B19_GENETIC_META_MEMORY_H48_POLICY_NAME,
    B19_STABILITY_MEMORY_H48_POLICY_NAME,
    B19_SWITCH_SUPPRESSION_H48_POLICY_NAME,
    B19_EPISODIC_META_MEMORY_DEFAULT_CHECKPOINT,
    B20_CONTEXT_BINDING_H48_POLICY_NAME,
    B20_GENETIC_WORKING_MEMORY_H48_POLICY_NAME,
    B20_STABILITY_BUFFER_H48_POLICY_NAME,
    B20_WORKING_MEMORY_GATE_DEFAULT_CHECKPOINT,
    B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
    B20_WORKING_MEMORY_GATE_H56_POLICY_NAME,
    B21_GENETIC_REPLAY_GATE_H48_POLICY_NAME,
    B21_HIPPOCAMPAL_REPLAY_DEFAULT_CHECKPOINT,
    B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
    B21_HIPPOCAMPAL_REPLAY_H56_POLICY_NAME,
    B21_ROUTE_REHEARSAL_H48_POLICY_NAME,
    B21_SEQUENCE_BINDING_H48_POLICY_NAME,
    B22_FORWARD_MODEL_GATE_H48_POLICY_NAME,
    B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
    B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
    B22_PROSPECTIVE_MAP_REPLAY_H56_POLICY_NAME,
    B22_PROSPECTIVE_REPLAY_DEFAULT_CHECKPOINT,
    B22_ROUTE_VIABILITY_SIM_H48_POLICY_NAME,
    B23_ABORT_CONFLICT_ARBITER_H48_POLICY_NAME,
    B23_CONFLICT_MONITOR_DEFAULT_CHECKPOINT,
    B23_CONFLICT_MONITOR_H48_POLICY_NAME,
    B23_CONFLICT_MONITOR_H56_POLICY_NAME,
    B23_ERROR_GATED_REPLAY_H48_POLICY_NAME,
    B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME,
    B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME,
    B24_PRECISION_CONFLICT_H48_POLICY_NAME,
    B24_PRECISION_CONFLICT_H56_POLICY_NAME,
    B24_PRECISION_CONFLICT_DEFAULT_CHECKPOINT,
    B24_PREDICTION_PRECISION_GATE_H48_POLICY_NAME,
    B24_RELIABILITY_ABORT_H48_POLICY_NAME,
    B25_CONFIDENCE_CALIBRATION_H48_POLICY_NAME,
    B25_GENETIC_METACOGNITION_H48_POLICY_NAME,
    B25_METACOGNITIVE_CONFIDENCE_DEFAULT_CHECKPOINT,
    B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
    B25_METACOGNITIVE_CONFIDENCE_H56_POLICY_NAME,
    B25_UNCERTAINTY_INTEGRATOR_H48_POLICY_NAME,
    B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
    B26_ALLOSTATIC_PREDICTION_H56_POLICY_NAME,
    B26_ALLOSTATIC_PREDICTION_DEFAULT_CHECKPOINT,
    B26_ERROR_SUPPRESSION_H48_POLICY_NAME,
    B26_GENETIC_ALLOSTASIS_H48_POLICY_NAME,
    B26_SETPOINT_DRIFT_H48_POLICY_NAME,
    B27_AROUSAL_GAIN_H48_POLICY_NAME,
    B27_AROUSAL_GAIN_H56_POLICY_NAME,
    B27_AROUSAL_GAIN_DEFAULT_CHECKPOINT,
    B27_ENERGY_AROUSAL_H48_POLICY_NAME,
    B27_GENETIC_AROUSAL_H48_POLICY_NAME,
    B27_STRESS_MODULATION_H48_POLICY_NAME,
    B28_GENETIC_ATTENTION_H48_POLICY_NAME,
    B28_HOMEOSTATIC_ATTENTION_H48_POLICY_NAME,
    B28_INTEROCEPTIVE_ATTENTION_DEFAULT_CHECKPOINT,
    B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
    B28_INTEROCEPTIVE_ATTENTION_H56_POLICY_NAME,
    B28_THREAT_FOCUS_ATTENTION_H48_POLICY_NAME,
    B29_GENETIC_SALIENCE_H48_POLICY_NAME,
    B29_HOMEOSTATIC_SALIENCE_GATE_H48_POLICY_NAME,
    B29_SALIENCE_COMPETITION_DEFAULT_CHECKPOINT,
    B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
    B29_SALIENCE_COMPETITION_H56_POLICY_NAME,
    B29_THREAT_SALIENCE_GATE_H48_POLICY_NAME,
    B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
    B30_BASAL_GANGLIA_GATE_H56_POLICY_NAME,
    B30_BASAL_GANGLIA_GATE_DEFAULT_CHECKPOINT,
    B30_GENETIC_ACTION_GATE_H48_POLICY_NAME,
    B30_GO_NOGO_BALANCE_H48_POLICY_NAME,
    B30_THREAT_INHIBITION_GATE_H48_POLICY_NAME,
    B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
    B31_DOPAMINE_PREDICTION_ERROR_H56_POLICY_NAME,
    B31_DOPAMINE_PREDICTION_ERROR_DEFAULT_CHECKPOINT,
    B31_GENETIC_DOPAMINE_GATE_H48_POLICY_NAME,
    B31_PHASIC_DOPAMINE_GATE_H48_POLICY_NAME,
    B31_TONIC_DOPAMINE_GATE_H48_POLICY_NAME,
    B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
    B32_ACTOR_CRITIC_VALUE_H56_POLICY_NAME,
    B32_ACTOR_CRITIC_VALUE_DEFAULT_CHECKPOINT,
    B32_ADVANTAGE_VALUE_GATE_H48_POLICY_NAME,
    B32_CRITIC_STABILITY_H48_POLICY_NAME,
    B32_GENETIC_ACTOR_CRITIC_H48_POLICY_NAME,
    B33_BOOTSTRAPPED_VALUE_GATE_H48_POLICY_NAME,
    B33_GENETIC_TD_VALUE_H48_POLICY_NAME,
    B33_REWARD_TRACE_CRITIC_H48_POLICY_NAME,
    B33_TD_ERROR_DECOMPOSITION_DEFAULT_CHECKPOINT,
    B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
    B33_TD_ERROR_DECOMPOSITION_H56_POLICY_NAME,
    B34_DELAYED_CREDIT_GATE_H48_POLICY_NAME,
    B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
    B34_ELIGIBILITY_CREDIT_H56_POLICY_NAME,
    B34_ELIGIBILITY_CREDIT_DEFAULT_CHECKPOINT,
    B34_GENETIC_ELIGIBILITY_H48_POLICY_NAME,
    B34_SYNAPTIC_TAGGING_H48_POLICY_NAME,
    B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
    B35_FORWARD_MODEL_VALUE_H56_POLICY_NAME,
    B35_FORWARD_MODEL_VALUE_DEFAULT_CHECKPOINT,
    B35_GENETIC_FORWARD_MODEL_H48_POLICY_NAME,
    B35_MODEL_CONFIDENCE_H48_POLICY_NAME,
    B35_TRANSITION_ERROR_GATE_H48_POLICY_NAME,
    B36_BELIEF_ERROR_GATE_H48_POLICY_NAME,
    B36_CONTEXT_INFERENCE_H48_POLICY_NAME,
    B36_GENETIC_BELIEF_STATE_H48_POLICY_NAME,
    B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
    B36_LATENT_BELIEF_STATE_H56_POLICY_NAME,
    B36_LATENT_BELIEF_STATE_DEFAULT_CHECKPOINT,
    B37_FACTOR_CONFIDENCE_H48_POLICY_NAME,
    B37_GENETIC_STATE_FACTOR_H48_POLICY_NAME,
    B37_INTERO_EXTERO_FACTOR_H48_POLICY_NAME,
    B37_STATE_FACTOR_GATE_DEFAULT_CHECKPOINT,
    B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
    B37_STATE_FACTOR_GATE_H56_POLICY_NAME,
    B38_CONFIDENCE_ATTENTION_H48_POLICY_NAME,
    B38_FACTOR_ATTENTION_H48_POLICY_NAME,
    B38_FACTOR_ATTENTION_H56_POLICY_NAME,
    B38_FACTOR_ATTENTION_DEFAULT_CHECKPOINT,
    B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME,
    B38_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
    B39_ATTENTION_BINDING_H48_POLICY_NAME,
    B39_ATTENTION_BINDING_H56_POLICY_NAME,
    B39_ATTENTION_BINDING_DEFAULT_CHECKPOINT,
    B39_CONTEXT_BINDING_ATTENTION_H48_POLICY_NAME,
    B39_CROSS_FACTOR_BINDING_H48_POLICY_NAME,
    B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME,
    B40_CONTEXT_WORKSPACE_H48_POLICY_NAME,
    B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME,
    B40_GLOBAL_WORKSPACE_DEFAULT_CHECKPOINT,
    B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
    B40_GLOBAL_WORKSPACE_H56_POLICY_NAME,
    B40_SENSORY_WORKSPACE_H48_POLICY_NAME,
    B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
    B41_EXECUTIVE_WORKSPACE_H56_POLICY_NAME,
    B41_EXECUTIVE_WORKSPACE_DEFAULT_CHECKPOINT,
    B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
    B41_GOAL_CONTEXT_SELECTOR_H48_POLICY_NAME,
    B41_INHIBITORY_CONTROL_H48_POLICY_NAME,
    B42_CONFLICT_MONITOR_H48_POLICY_NAME,
    B42_ERROR_MONITOR_H48_POLICY_NAME,
    B42_ERROR_MONITOR_H56_POLICY_NAME,
    B42_ERROR_MONITOR_DEFAULT_CHECKPOINT,
    B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME,
    B42_PERFORMANCE_MONITOR_H48_POLICY_NAME,
    B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
    B43_ADAPTIVE_PRECISION_H56_POLICY_NAME,
    B43_AROUSAL_PRECISION_H48_POLICY_NAME,
    B43_ADAPTIVE_PRECISION_DEFAULT_CHECKPOINT,
    B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME,
    B43_THRESHOLD_ADAPTATION_H48_POLICY_NAME,
    B44_CONTEXT_RELAY_H48_POLICY_NAME,
    B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME,
    B44_SENSORY_RELAY_H48_POLICY_NAME,
    B44_THALAMIC_RELAY_DEFAULT_CHECKPOINT,
    B44_THALAMIC_RELAY_H48_POLICY_NAME,
    B44_THALAMIC_RELAY_H56_POLICY_NAME,
    B45_CONTEXT_INHIBITION_H48_POLICY_NAME,
    B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME,
    B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
    B45_RETICULAR_INHIBITION_H56_POLICY_NAME,
    B45_SENSORY_INHIBITION_H48_POLICY_NAME,
    B45_RETICULAR_INHIBITION_DEFAULT_CHECKPOINT,
    B46_CONTEXT_FEEDBACK_H48_POLICY_NAME,
    B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
    B46_CORTICOTHALAMIC_FEEDBACK_H56_POLICY_NAME,
    B46_CORTICOTHALAMIC_FEEDBACK_DEFAULT_CHECKPOINT,
    B46_FEEDBACK_GAIN_H48_POLICY_NAME,
    B46_GENETIC_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
    B47_COHERENCE_GATE_H48_POLICY_NAME,
    B47_GENETIC_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
    B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
    B47_OSCILLATORY_SYNCHRONY_H56_POLICY_NAME,
    B47_OSCILLATORY_SYNCHRONY_DEFAULT_CHECKPOINT,
    B47_PHASE_LOCKING_H48_POLICY_NAME,
    B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
    B48_CEREBELLAR_TIMING_H56_POLICY_NAME,
    B48_GENETIC_CEREBELLAR_TIMING_H48_POLICY_NAME,
    B48_PREDICTIVE_TIMING_H48_POLICY_NAME,
    B48_TIMING_ERROR_CORRECTION_H48_POLICY_NAME,
    B48_CEREBELLAR_TIMING_DEFAULT_CHECKPOINT,
    B49_DIRECT_PATH_FACILITATION_H48_POLICY_NAME,
    B49_GENETIC_STRIATAL_GATE_H48_POLICY_NAME,
    B49_INDIRECT_PATH_SUPPRESSION_H48_POLICY_NAME,
    B49_STRIATAL_ACTION_GATE_DEFAULT_CHECKPOINT,
    B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
    B49_STRIATAL_ACTION_GATE_H56_POLICY_NAME,
    B50_ACTION_CHUNK_VALUE_H48_POLICY_NAME,
    B50_GENETIC_HABIT_CHUNKING_H48_POLICY_NAME,
    B50_HABIT_CHUNKING_DEFAULT_CHECKPOINT,
    B50_HABIT_CHUNKING_H48_POLICY_NAME,
    B50_HABIT_CHUNKING_H56_POLICY_NAME,
    B50_HABIT_STABILITY_H48_POLICY_NAME,
    B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
    B51_DOPAMINERGIC_HABIT_MODULATION_H56_POLICY_NAME,
    B51_GENETIC_DOPAMINE_HABIT_H48_POLICY_NAME,
    B51_NOVELTY_MODULATED_HABIT_H48_POLICY_NAME,
    B51_REWARD_PREDICTION_GAIN_H48_POLICY_NAME,
    B52_ATTENTION_GAIN_H48_POLICY_NAME,
    B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
    B52_CHOLINERGIC_PRECISION_GATE_H56_POLICY_NAME,
    B52_GENETIC_CHOLINERGIC_PRECISION_H48_POLICY_NAME,
    B52_UNCERTAINTY_RELEASE_H48_POLICY_NAME,
    B53_GENETIC_AROUSAL_PRECISION_H48_POLICY_NAME,
    B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
    B53_NORADRENERGIC_AROUSAL_GAIN_H56_POLICY_NAME,
    B53_STRESS_PRECISION_H48_POLICY_NAME,
    B53_SURPRISE_GAIN_H48_POLICY_NAME,
    B54_GENETIC_SEROTONIN_PATIENCE_H48_POLICY_NAME,
    B54_IMPULSE_SUPPRESSION_H48_POLICY_NAME,
    B54_PATIENCE_BALANCE_H48_POLICY_NAME,
    B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
    B54_SEROTONERGIC_PATIENCE_GATE_H56_POLICY_NAME,
    B55_GENETIC_HYPOTHALAMIC_DRIVE_H48_POLICY_NAME,
    B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
    B55_HYPOTHALAMIC_DRIVE_COUPLING_H56_POLICY_NAME,
    B55_SATIETY_RECOVERY_BALANCE_H48_POLICY_NAME,
    B55_SLEEP_HUNGER_ARBITER_H48_POLICY_NAME,
    B56_CORTISOL_RECOVERY_BALANCE_H48_POLICY_NAME,
    B56_GENETIC_HPA_STRESS_H48_POLICY_NAME,
    B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
    B56_HPA_STRESS_AXIS_H56_POLICY_NAME,
    B56_STRESS_LOAD_GATE_H48_POLICY_NAME,
    B57_GENETIC_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
    B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
    B57_INSULAR_INTEROCEPTIVE_AWARENESS_H56_POLICY_NAME,
    B57_STRESS_DRIVE_AWARENESS_H48_POLICY_NAME,
    B57_VISCERAL_SALIENCE_GATE_H48_POLICY_NAME,
    B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
    B58_ACC_CONFLICT_MONITOR_H56_POLICY_NAME,
    B58_CONFLICT_RESOLUTION_BALANCE_H48_POLICY_NAME,
    B58_ERROR_SALIENCE_GATE_H48_POLICY_NAME,
    B58_GENETIC_ACC_CONFLICT_H48_POLICY_NAME,
    B59_EXECUTIVE_TASK_SET_H48_POLICY_NAME,
    B59_GENETIC_PREFRONTAL_CONTROL_H48_POLICY_NAME,
    B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
    B59_PREFRONTAL_GOAL_CONTEXT_H56_POLICY_NAME,
    B59_WORKING_SET_STABILITY_H48_POLICY_NAME,
    B60_GENETIC_ORBITOFRONTAL_VALUE_H48_POLICY_NAME,
    B60_GOAL_OUTCOME_PREDICTION_H48_POLICY_NAME,
    B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
    B60_ORBITOFRONTAL_OUTCOME_VALUE_H56_POLICY_NAME,
    B60_REVERSAL_VALUE_GATE_H48_POLICY_NAME,
    B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
    B61_AMYGDALA_SAFETY_VALUE_H56_POLICY_NAME,
    B61_GENETIC_AMYGDALA_SAFETY_H48_POLICY_NAME,
    B61_SAFETY_PREDICTION_GATE_H48_POLICY_NAME,
    B61_THREAT_VALUE_TAG_H48_POLICY_NAME,
    B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
    B62_DEFENSIVE_MODE_SELECTOR_H56_POLICY_NAME,
    B62_FREEZE_FLEE_BALANCE_H48_POLICY_NAME,
    B62_GENETIC_DEFENSIVE_MODE_H48_POLICY_NAME,
    B62_SHELTER_DEFENSE_GATE_H48_POLICY_NAME,
)


B5_ACCEPTED_HOMEOSTASIS_PARAMS: dict[str, float] = {
    "emergency_hunger_release": 0.961294,
    "exit_recovery_debt_max": 0.49905,
    "exit_sleep_pressure_max": 0.57323,
    "exit_threat_max": 0.65131,
    "forage_lock_ticks": 5.075901,
    "forage_threat_max": 0.503494,
    "hunger_release": 0.899621,
    "return_hunger_max": 0.786839,
    "sleep_hunger_max": 0.779747,
    "sleep_lock_ticks": 10.580013,
    "sleep_pressure_threshold": 0.733657,
    "sleep_threat_max": 0.438163,
}

B6_ACCEPTED_FUSED_PARAMS: dict[str, float] = B5_ACCEPTED_HOMEOSTASIS_PARAMS | {
    "b6_family": 3.0,
    "b6_risk_threshold": 0.35,
    "b6_corridor_hunger": 0.86,
    "b6_corridor_lock_ticks": 16.0,
    "b6_threat_memory_ticks": 12.0,
    "b6_return_lock_ticks": 8.0,
    "b6_recurrent_decay": 0.70,
}

B7_BASE_AFFORDANCE_PARAMS: dict[str, float] = B6_ACCEPTED_FUSED_PARAMS | {
    "b7_budget_step_cost": 0.085,
    "b7_viability_margin": -0.08,
    "b7_abort_health": 0.36,
    "b7_recover_health": 0.42,
    "b7_food_commit_distance": 13.0,
    "b7_commitment_ticks": 8.0,
    "b7_recurrent_decay": 0.72,
}

B8_BASE_SPATIAL_PARAMS: dict[str, float] = B7_BASE_AFFORDANCE_PARAMS | {
    "b8_place_memory_decay": 0.78,
    "b8_dead_end_risk_threshold": 0.62,
    "b8_return_vector_threshold": 0.18,
    "b8_abort_health": 0.18,
    "b8_hold_health": 0.10,
    "b8_food_progress_floor": 13.0,
}

B9_BASE_WAYPOINT_PARAMS: dict[str, float] = B8_BASE_SPATIAL_PARAMS | {
    "b9_route_memory_decay": 0.82,
    "b9_waypoint_commit_ticks": 6.0,
    "b9_route_confidence_threshold": 0.18,
    "b9_path_integrator_gain": 0.50,
    "b9_replan_dead_end_threshold": 0.72,
    "b9_progress_floor": 13.0,
}

B10_BASE_REPLAY_PARAMS: dict[str, float] = B9_BASE_WAYPOINT_PARAMS | {
    "b10_replay_memory_decay": 0.84,
    "b10_rollout_gain": 0.55,
    "b10_value_threshold": 0.20,
    "b10_replay_commit_ticks": 5.0,
    "b10_abort_threshold": 0.64,
    "b10_progress_floor": 13.0,
}

B11_BASE_CONFIDENCE_PARAMS: dict[str, float] = B10_BASE_REPLAY_PARAMS | {
    "b11_confidence_decay": 0.86,
    "b11_confidence_threshold": 0.24,
    "b11_uncertainty_threshold": 0.70,
    "b11_neuromod_gain": 0.50,
    "b11_confidence_commit_ticks": 5.0,
}

B12_BASE_ATTENTION_PARAMS: dict[str, float] = B11_BASE_CONFIDENCE_PARAMS | {
    "b12_attention_decay": 0.84,
    "b12_attention_threshold": 0.18,
    "b12_prediction_error_threshold": 0.66,
    "b12_affordance_gain": 0.45,
    "b12_search_commit_ticks": 5.0,
}

B13_BASE_LOCAL_SEARCH_PARAMS: dict[str, float] = B12_BASE_ATTENTION_PARAMS | {
    "b13_search_memory_decay": 0.86,
    "b13_candidate_gain": 0.50,
    "b13_search_threshold": 0.20,
    "b13_dead_end_threshold": 0.68,
    "b13_local_commit_ticks": 5.0,
}

B14_BASE_UNCERTAINTY_PARAMS: dict[str, float] = B13_BASE_LOCAL_SEARCH_PARAMS | {
    "b14_uncertainty_decay": 0.82,
    "b14_confidence_threshold": 0.42,
    "b14_uncertainty_threshold": 0.58,
    "b14_risk_gain": 0.40,
    "b14_commit_ticks": 4.0,
}

B15_BASE_OPTION_PARAMS: dict[str, float] = B14_BASE_UNCERTAINTY_PARAMS | {
    "b15_option_memory_decay": 0.84,
    "b15_option_value_threshold": 0.28,
    "b15_termination_threshold": 0.62,
    "b15_persistence_gain": 0.45,
    "b15_option_commit_ticks": 5.0,
}

B16_BASE_ENSEMBLE_PARAMS: dict[str, float] = B15_BASE_OPTION_PARAMS | {
    "b16_ensemble_decay": 0.82,
    "b16_consensus_threshold": 0.30,
    "b16_conflict_threshold": 0.56,
    "b16_vote_gain": 0.50,
    "b16_commit_ticks": 5.0,
}

B17_BASE_NEUROMOD_PARAMS: dict[str, float] = B16_BASE_ENSEMBLE_PARAMS | {
    "b17_arousal_decay": 0.84,
    "b17_gain_threshold": 0.34,
    "b17_conflict_release": 0.52,
    "b17_homeostasis_gain": 0.44,
    "b17_commit_ticks": 5.0,
}

B18_BASE_TRACE_PARAMS: dict[str, float] = B17_BASE_NEUROMOD_PARAMS | {
    "b18_trace_decay": 0.86,
    "b18_stability_threshold": 0.30,
    "b18_switch_threshold": 0.58,
    "b18_prediction_gain": 0.42,
    "b18_trace_commit_ticks": 5.0,
}

B19_BASE_META_MEMORY_PARAMS: dict[str, float] = B18_BASE_TRACE_PARAMS | {
    "b19_memory_decay": 0.88,
    "b19_consolidation_threshold": 0.30,
    "b19_switch_suppression_threshold": 0.58,
    "b19_stability_gain": 0.42,
    "b19_memory_commit_ticks": 5.0,
}

B20_BASE_WORKING_MEMORY_PARAMS: dict[str, float] = B19_BASE_META_MEMORY_PARAMS | {
    "b20_buffer_decay": 0.86,
    "b20_gate_threshold": 0.30,
    "b20_release_threshold": 0.58,
    "b20_context_gain": 0.44,
    "b20_buffer_commit_ticks": 5.0,
}

B21_BASE_REPLAY_PARAMS: dict[str, float] = B20_BASE_WORKING_MEMORY_PARAMS | {
    "b21_replay_decay": 0.84,
    "b21_replay_threshold": 0.30,
    "b21_abort_threshold": 0.58,
    "b21_sequence_gain": 0.46,
    "b21_replay_commit_ticks": 5.0,
}

B22_BASE_PROSPECTIVE_PARAMS: dict[str, float] = B21_BASE_REPLAY_PARAMS | {
    "b22_sim_decay": 0.84,
    "b22_viability_threshold": 0.30,
    "b22_abort_threshold": 0.58,
    "b22_forward_gain": 0.48,
    "b22_sim_commit_ticks": 5.0,
}

B23_BASE_CONFLICT_PARAMS: dict[str, float] = B22_BASE_PROSPECTIVE_PARAMS | {
    "b23_conflict_decay": 0.86,
    "b23_conflict_threshold": 0.24,
    "b23_abort_bias_threshold": 0.62,
    "b23_error_gain": 0.46,
    "b23_monitor_commit_ticks": 5.0,
}

B24_BASE_PRECISION_PARAMS: dict[str, float] = B23_BASE_CONFLICT_PARAMS | {
    "b24_precision_decay": 0.88,
    "b24_precision_threshold": 0.26,
    "b24_uncertainty_threshold": 0.64,
    "b24_precision_gain": 0.45,
    "b24_precision_commit_ticks": 5.0,
}

B25_BASE_METACOG_PARAMS: dict[str, float] = B24_BASE_PRECISION_PARAMS | {
    "b25_confidence_decay": 0.90,
    "b25_confidence_threshold": 0.28,
    "b25_doubt_threshold": 0.66,
    "b25_control_gain": 0.44,
    "b25_meta_commit_ticks": 5.0,
}

B26_BASE_ALLOSTATIC_PARAMS: dict[str, float] = B25_BASE_METACOG_PARAMS | {
    "b26_error_decay": 0.88,
    "b26_prediction_threshold": 0.24,
    "b26_abort_threshold": 0.68,
    "b26_control_gain": 0.46,
    "b26_stability_commit_ticks": 5.0,
}

B27_BASE_AROUSAL_PARAMS: dict[str, float] = B26_BASE_ALLOSTATIC_PARAMS | {
    "b27_arousal_decay": 0.86,
    "b27_gain_threshold": 0.24,
    "b27_stress_threshold": 0.70,
    "b27_modulation_gain": 0.48,
    "b27_arousal_commit_ticks": 5.0,
}

B28_BASE_ATTENTION_PARAMS: dict[str, float] = B27_BASE_AROUSAL_PARAMS | {
    "b28_attention_decay": 0.86,
    "b28_focus_threshold": 0.24,
    "b28_distractor_threshold": 0.70,
    "b28_attention_gain": 0.48,
    "b28_attention_commit_ticks": 5.0,
}

B29_BASE_SALIENCE_PARAMS: dict[str, float] = B28_BASE_ATTENTION_PARAMS | {
    "b29_salience_decay": 0.86,
    "b29_corridor_threshold": 0.24,
    "b29_threat_threshold": 0.70,
    "b29_homeostatic_gain": 0.40,
    "b29_competition_gain": 0.52,
    "b29_salience_commit_ticks": 5.0,
}

B30_BASE_GATE_PARAMS: dict[str, float] = B29_BASE_SALIENCE_PARAMS | {
    "b30_gate_decay": 0.86,
    "b30_go_threshold": 0.24,
    "b30_nogo_threshold": 0.70,
    "b30_go_gain": 0.52,
    "b30_nogo_gain": 0.44,
    "b30_gate_commit_ticks": 5.0,
}

B31_BASE_DOPAMINE_PARAMS: dict[str, float] = B30_BASE_GATE_PARAMS | {
    "b31_dopamine_decay": 0.86,
    "b31_go_threshold": 0.24,
    "b31_nogo_threshold": 0.70,
    "b31_prediction_gain": 0.46,
    "b31_phasic_gain": 0.52,
    "b31_tonic_gain": 0.42,
    "b31_dopamine_commit_ticks": 5.0,
}

B32_BASE_ACTOR_CRITIC_PARAMS: dict[str, float] = B31_BASE_DOPAMINE_PARAMS | {
    "b32_value_decay": 0.88,
    "b32_advantage_threshold": 0.20,
    "b32_abort_threshold": -0.18,
    "b32_critic_gain": 0.46,
    "b32_actor_gain": 0.50,
    "b32_value_commit_ticks": 5.0,
}

B33_BASE_TD_ERROR_PARAMS: dict[str, float] = B32_BASE_ACTOR_CRITIC_PARAMS | {
    "b33_td_decay": 0.88,
    "b33_bootstrap_gain": 0.42,
    "b33_reward_trace_gain": 0.36,
    "b33_td_commit_ticks": 5.0,
    "b33_td_threshold": 0.16,
    "b33_abort_threshold": -0.20,
}
B34_BASE_ELIGIBILITY_PARAMS: dict[str, float] = B33_BASE_TD_ERROR_PARAMS | {
    "b34_eligibility_decay": 0.86,
    "b34_credit_gain": 0.38,
    "b34_tag_gain": 0.32,
    "b34_credit_threshold": 0.12,
    "b34_abort_threshold": -0.18,
    "b34_credit_lock_ticks": 5.0,
}
B35_BASE_FORWARD_MODEL_PARAMS: dict[str, float] = B34_BASE_ELIGIBILITY_PARAMS | {
    "b35_model_decay": 0.84,
    "b35_prediction_gain": 0.40,
    "b35_transition_gain": 0.34,
    "b35_confidence_gain": 0.30,
    "b35_commit_threshold": 0.14,
    "b35_abort_threshold": -0.16,
    "b35_model_lock_ticks": 5.0,
}
B36_BASE_BELIEF_STATE_PARAMS: dict[str, float] = B35_BASE_FORWARD_MODEL_PARAMS | {
    "b36_belief_decay": 0.86,
    "b36_latent_gain": 0.38,
    "b36_context_gain": 0.32,
    "b36_error_gain": 0.34,
    "b36_commit_threshold": 0.14,
    "b36_abort_threshold": -0.16,
    "b36_belief_lock_ticks": 5.0,
}
B37_BASE_STATE_FACTOR_PARAMS: dict[str, float] = B36_BASE_BELIEF_STATE_PARAMS | {
    "b37_factor_decay": 0.86,
    "b37_external_gain": 0.36,
    "b37_internal_gain": 0.34,
    "b37_factor_balance_gain": 0.32,
    "b37_commit_threshold": 0.14,
    "b37_abort_threshold": -0.16,
    "b37_factor_lock_ticks": 5.0,
}
B38_BASE_FACTOR_ATTENTION_PARAMS: dict[str, float] = B37_BASE_STATE_FACTOR_PARAMS | {
    "b38_attention_decay": 0.86,
    "b38_external_attention_gain": 0.34,
    "b38_internal_attention_gain": 0.36,
    "b38_confidence_attention_gain": 0.32,
    "b38_attention_threshold": 0.13,
    "b38_abort_threshold": -0.17,
    "b38_attention_lock_ticks": 5.0,
}
B39_BASE_ATTENTION_BINDING_PARAMS: dict[str, float] = B38_BASE_FACTOR_ATTENTION_PARAMS | {
    "b39_binding_decay": 0.86,
    "b39_external_binding_gain": 0.32,
    "b39_internal_binding_gain": 0.34,
    "b39_context_binding_gain": 0.36,
    "b39_binding_threshold": 0.12,
    "b39_abort_threshold": -0.18,
    "b39_binding_lock_ticks": 5.0,
}
B40_BASE_GLOBAL_WORKSPACE_PARAMS: dict[str, float] = B39_BASE_ATTENTION_BINDING_PARAMS | {
    "b40_workspace_decay": 0.86,
    "b40_activation_gain": 0.34,
    "b40_broadcast_gain": 0.32,
    "b40_context_gain": 0.36,
    "b40_workspace_threshold": 0.12,
    "b40_abort_threshold": -0.18,
    "b40_workspace_lock_ticks": 5.0,
}
B41_BASE_EXECUTIVE_WORKSPACE_PARAMS: dict[str, float] = B40_BASE_GLOBAL_WORKSPACE_PARAMS | {
    "b41_executive_decay": 0.86,
    "b41_selection_gain": 0.34,
    "b41_inhibition_gain": 0.30,
    "b41_goal_context_gain": 0.32,
    "b41_selection_threshold": 0.12,
    "b41_abort_threshold": -0.18,
    "b41_executive_lock_ticks": 5.0,
}
B42_BASE_ERROR_MONITOR_PARAMS: dict[str, float] = B41_BASE_EXECUTIVE_WORKSPACE_PARAMS | {
    "b42_monitor_decay": 0.86,
    "b42_error_gain": 0.34,
    "b42_conflict_gain": 0.30,
    "b42_performance_gain": 0.32,
    "b42_commit_threshold": 0.12,
    "b42_abort_threshold": -0.18,
    "b42_monitor_lock_ticks": 5.0,
}
B43_BASE_ADAPTIVE_PRECISION_PARAMS: dict[str, float] = B42_BASE_ERROR_MONITOR_PARAMS | {
    "b43_precision_decay": 0.86,
    "b43_precision_gain": 0.34,
    "b43_arousal_gain": 0.30,
    "b43_threshold_gain": 0.32,
    "b43_commit_threshold": 0.12,
    "b43_abort_threshold": -0.18,
    "b43_precision_lock_ticks": 5.0,
}
B44_BASE_THALAMIC_RELAY_PARAMS: dict[str, float] = B43_BASE_ADAPTIVE_PRECISION_PARAMS | {
    "b44_relay_decay": 0.86,
    "b44_gate_gain": 0.34,
    "b44_sensory_gain": 0.30,
    "b44_context_gain": 0.32,
    "b44_relay_threshold": 0.12,
    "b44_abort_threshold": -0.18,
    "b44_relay_lock_ticks": 5.0,
}
B45_BASE_RETICULAR_INHIBITION_PARAMS: dict[str, float] = B44_BASE_THALAMIC_RELAY_PARAMS | {
    "b45_inhibition_decay": 0.86,
    "b45_inhibitory_gain": 0.34,
    "b45_sensory_filter_gain": 0.30,
    "b45_context_suppression_gain": 0.32,
    "b45_commit_threshold": 0.12,
    "b45_abort_threshold": -0.18,
    "b45_inhibition_lock_ticks": 5.0,
}
B46_BASE_CORTICOTHALAMIC_FEEDBACK_PARAMS: dict[str, float] = B45_BASE_RETICULAR_INHIBITION_PARAMS | {
    "b46_feedback_decay": 0.86,
    "b46_feedback_gain": 0.34,
    "b46_topdown_gain": 0.30,
    "b46_prediction_gain": 0.32,
    "b46_commit_threshold": 0.08,
    "b46_abort_threshold": -0.18,
    "b46_feedback_lock_ticks": 5.0,
}
B47_BASE_OSCILLATORY_SYNCHRONY_PARAMS: dict[str, float] = B46_BASE_CORTICOTHALAMIC_FEEDBACK_PARAMS | {
    "b47_phase_decay": 0.86,
    "b47_phase_gain": 0.32,
    "b47_synchrony_gain": 0.34,
    "b47_coherence_gain": 0.30,
    "b47_commit_threshold": 0.08,
    "b47_abort_threshold": -0.18,
    "b47_phase_lock_ticks": 5.0,
}
B48_BASE_CEREBELLAR_TIMING_PARAMS: dict[str, float] = B47_BASE_OSCILLATORY_SYNCHRONY_PARAMS | {
    "b48_timing_decay": 0.86,
    "b48_error_gain": 0.30,
    "b48_prediction_gain": 0.34,
    "b48_corrective_gain": 0.32,
    "b48_commit_threshold": 0.08,
    "b48_abort_threshold": -0.18,
    "b48_calibration_lock_ticks": 5.0,
}
B49_BASE_STRIATAL_GATE_PARAMS: dict[str, float] = B48_BASE_CEREBELLAR_TIMING_PARAMS | {
    "b49_gate_decay": 0.86,
    "b49_go_gain": 0.34,
    "b49_no_go_gain": 0.30,
    "b49_balance_gain": 0.32,
    "b49_commit_threshold": 0.08,
    "b49_abort_threshold": -0.18,
    "b49_selection_lock_ticks": 5.0,
}
B50_BASE_HABIT_CHUNKING_PARAMS: dict[str, float] = B49_BASE_STRIATAL_GATE_PARAMS | {
    "b50_habit_decay": 0.86,
    "b50_habit_gain": 0.34,
    "b50_chunk_value_gain": 0.30,
    "b50_stability_gain": 0.32,
    "b50_commit_threshold": 0.08,
    "b50_abort_threshold": -0.18,
    "b50_chunk_lock_ticks": 5.0,
}
B51_BASE_DOPAMINERGIC_HABIT_PARAMS: dict[str, float] = B50_BASE_HABIT_CHUNKING_PARAMS | {
    "b51_dopamine_decay": 0.86,
    "b51_prediction_error_gain": 0.32,
    "b51_dopamine_gain": 0.34,
    "b51_habit_modulation_gain": 0.30,
    "b51_commit_threshold": 0.08,
    "b51_abort_threshold": -0.18,
    "b51_modulation_lock_ticks": 5.0,
}
B52_BASE_CHOLINERGIC_PRECISION_PARAMS: dict[str, float] = (
    B51_BASE_DOPAMINERGIC_HABIT_PARAMS
    | {
        "b52_acetylcholine_decay": 0.86,
        "b52_uncertainty_gain": 0.30,
        "b52_precision_gain": 0.34,
        "b52_attention_gain": 0.32,
        "b52_commit_threshold": 0.08,
        "b52_abort_threshold": -0.20,
        "b52_attention_lock_ticks": 5.0,
    }
)
B53_BASE_NORADRENERGIC_AROUSAL_PARAMS: dict[str, float] = (
    B52_BASE_CHOLINERGIC_PRECISION_PARAMS
    | {
        "b53_norepinephrine_decay": 0.86,
        "b53_surprise_gain": 0.30,
        "b53_arousal_gain": 0.34,
        "b53_precision_mod_gain": 0.32,
        "b53_commit_threshold": 0.08,
        "b53_abort_threshold": -0.20,
        "b53_gain_lock_ticks": 5.0,
    }
)
B54_BASE_SEROTONERGIC_PATIENCE_PARAMS: dict[str, float] = (
    B53_BASE_NORADRENERGIC_AROUSAL_PARAMS
    | {
        "b54_serotonin_decay": 0.86,
        "b54_patience_gain": 0.34,
        "b54_impulse_suppression_gain": 0.32,
        "b54_arousal_balance_gain": 0.30,
        "b54_commit_threshold": 0.08,
        "b54_abort_threshold": -0.20,
        "b54_patience_lock_ticks": 5.0,
    }
)
B55_BASE_HYPOTHALAMIC_DRIVE_PARAMS: dict[str, float] = (
    B54_BASE_SEROTONERGIC_PATIENCE_PARAMS
    | {
        "b55_drive_decay": 0.86,
        "b55_hunger_gain": 0.34,
        "b55_satiety_gain": 0.28,
        "b55_recovery_gain": 0.30,
        "b55_threat_gate_gain": 0.26,
        "b55_commit_threshold": 0.08,
        "b55_abort_threshold": -0.20,
        "b55_drive_lock_ticks": 5.0,
    }
)
B56_BASE_HPA_STRESS_PARAMS: dict[str, float] = (
    B55_BASE_HYPOTHALAMIC_DRIVE_PARAMS
    | {
        "b56_endocrine_decay": 0.88,
        "b56_cortisol_gain": 0.30,
        "b56_stress_load_gain": 0.32,
        "b56_recovery_signal_gain": 0.30,
        "b56_drive_mod_gain": 0.28,
        "b56_commit_threshold": 0.08,
        "b56_abort_threshold": -0.20,
        "b56_stress_lock_ticks": 5.0,
    }
)
B57_BASE_INSULAR_INTEROCEPTION_PARAMS: dict[str, float] = (
    B56_BASE_HPA_STRESS_PARAMS
    | {
        "b57_awareness_decay": 0.87,
        "b57_visceral_salience_gain": 0.32,
        "b57_body_confidence_gain": 0.30,
        "b57_stress_awareness_gain": 0.28,
        "b57_drive_awareness_gain": 0.28,
        "b57_commit_threshold": 0.08,
        "b57_abort_threshold": -0.20,
        "b57_awareness_lock_ticks": 5.0,
    }
)
B58_BASE_ACC_CONFLICT_PARAMS: dict[str, float] = (
    B57_BASE_INSULAR_INTEROCEPTION_PARAMS
    | {
        "b58_conflict_decay": 0.87,
        "b58_conflict_gain": 0.32,
        "b58_error_likelihood_gain": 0.30,
        "b58_control_allocation_gain": 0.30,
        "b58_awareness_mod_gain": 0.28,
        "b58_commit_threshold": 0.08,
        "b58_abort_threshold": -0.20,
        "b58_conflict_lock_ticks": 5.0,
    }
)
B59_BASE_PREFRONTAL_CONTEXT_PARAMS: dict[str, float] = (
    B58_BASE_ACC_CONFLICT_PARAMS
    | {
        "b59_context_decay": 0.87,
        "b59_goal_context_gain": 0.32,
        "b59_working_set_gain": 0.30,
        "b59_task_confidence_gain": 0.30,
        "b59_control_mod_gain": 0.28,
        "b59_commit_threshold": 0.08,
        "b59_abort_threshold": -0.20,
        "b59_executive_lock_ticks": 5.0,
    }
)
B60_BASE_ORBITOFRONTAL_VALUE_PARAMS: dict[str, float] = (
    B59_BASE_PREFRONTAL_CONTEXT_PARAMS
    | {
        "b60_value_decay": 0.88,
        "b60_outcome_value_gain": 0.32,
        "b60_reversal_signal_gain": 0.28,
        "b60_goal_value_confidence_gain": 0.30,
        "b60_prefrontal_mod_gain": 0.26,
        "b60_commit_threshold": 0.07,
        "b60_reversal_threshold": 0.24,
        "b60_value_lock_ticks": 5.0,
    }
)
B61_BASE_AMYGDALA_SAFETY_PARAMS: dict[str, float] = (
    B60_BASE_ORBITOFRONTAL_VALUE_PARAMS
    | {
        "b61_affect_decay": 0.88,
        "b61_safety_value_gain": 0.32,
        "b61_threat_value_gain": 0.28,
        "b61_safety_confidence_gain": 0.30,
        "b61_ofc_mod_gain": 0.26,
        "b61_commit_threshold": 0.07,
        "b61_threat_threshold": 0.26,
        "b61_safety_lock_ticks": 5.0,
    }
)
B62_BASE_DEFENSIVE_MODE_PARAMS: dict[str, float] = (
    B61_BASE_AMYGDALA_SAFETY_PARAMS
    | {
        "b62_defense_decay": 0.88,
        "b62_freeze_gain": 0.30,
        "b62_flee_gain": 0.34,
        "b62_shelter_bias_gain": 0.28,
        "b62_balance_gain": 0.26,
        "b62_freeze_threshold": 0.30,
        "b62_flee_threshold": 0.23,
        "b62_defense_lock_ticks": 4.0,
    }
)

__all__ = [name for name in globals() if not name.startswith("_")]
__all__.append("_arbitration_fields")
