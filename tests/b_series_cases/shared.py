import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig, resolve_ablation_configs
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.b_series import (
    B_CURRENT_BRIDGE_EFFECTIVE_LEVEL,
    B_CURRENT_BRIDGE_SELECTION_SOURCE,
    B0_CURRENT_BRIDGE_DEFAULT_CHECKPOINT,
    B1_CAPACITY_H48_POLICY_NAME,
    B1_CAPACITY_H64_POLICY_NAME,
    B1_THREAT_GUARD_DEFAULT_CHECKPOINT,
    B1_THREAT_GUARD_EFFECTIVE_LEVEL,
    B1_THREAT_GUARD_POLICY_NAME,
    B1_THREAT_GUARD_SELECTION_SOURCE,
    B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL,
    B2_TEMPORAL_THREAT_H48_POLICY_NAME,
    B2_TEMPORAL_THREAT_H56_POLICY_NAME,
    B2_TEMPORAL_THREAT_H64_POLICY_NAME,
    B2_TEMPORAL_THREAT_SELECTION_SOURCE,
    B2_TEMPORAL_THREAT_DEFAULT_CHECKPOINT,
    B3_CONTACT_MEMORY_EFFECTIVE_LEVEL,
    B3_CONTACT_MEMORY_H48_POLICY_NAME,
    B3_CONTACT_MEMORY_H56_POLICY_NAME,
    B3_CONTACT_MEMORY_SELECTION_SOURCE,
    B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME,
    B3_RECURRENT_GUARD_EFFECTIVE_LEVEL,
    B3_RECURRENT_GUARD_H48_POLICY_NAME,
    B3_RECURRENT_GUARD_SELECTION_SOURCE,
    B4_GENETIC_RECOVERY_H48_POLICY_NAME,
    B4_GENETIC_RECOVERY_SELECTION_SOURCE,
    B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME,
    B4_RECOVERY_BALANCE_EFFECTIVE_LEVEL,
    B4_RECOVERY_BALANCE_H48_POLICY_NAME,
    B4_RECOVERY_BALANCE_H56_POLICY_NAME,
    B4_RECOVERY_BALANCE_SELECTION_SOURCE,
    B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME,
    B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
    B5_GENETIC_HOMEOSTASIS_SELECTION_SOURCE,
    B5_GENETIC_HOMEOSTASIS_DEFAULT_CHECKPOINT,
    B5_HOMEOSTATIC_ARBITER_EFFECTIVE_LEVEL,
    B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
    B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME,
    B5_HOMEOSTATIC_ARBITER_SELECTION_SOURCE,
    B6_CORRIDOR_SURVIVAL_GUARD_H48_POLICY_NAME,
    B6_FUSED_RISK_RECURRENT_DEFAULT_CHECKPOINT,
    B6_FUSED_RISK_RECURRENT_EFFECTIVE_LEVEL,
    B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
    B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE,
    B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME,
    B6_GENETIC_RISK_CORRIDOR_H48_POLICY_NAME,
    B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
    B6_RECURRENT_CONTEXT_H56_POLICY_NAME,
    B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME,
    B6_RECURRENT_MEMORY_EFFECTIVE_LEVEL,
    B6_RECURRENT_MEMORY_SELECTION_SOURCE,
    B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME,
    B6_RISK_CORRIDOR_EFFECTIVE_LEVEL,
    B6_RISK_CORRIDOR_H56_POLICY_NAME,
    B6_RISK_CORRIDOR_SELECTION_SOURCE,
    B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
    B6_THREAT_PRIORITY_MEMORY_H48_POLICY_NAME,
    B7_AFFORDANCE_BUDGET_EFFECTIVE_LEVEL,
    B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
    B7_AFFORDANCE_BUDGET_H56_POLICY_NAME,
    B7_AFFORDANCE_BUDGET_SELECTION_SOURCE,
    B7_ENERGY_BUDGET_CORRIDOR_H48_POLICY_NAME,
    B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME,
    B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME,
    B8_CORRIDOR_PLACE_MEMORY_H48_POLICY_NAME,
    B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME,
    B8_RETURN_VECTOR_H48_POLICY_NAME,
    B8_SPATIAL_AFFORDANCE_EFFECTIVE_LEVEL,
    B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
    B8_SPATIAL_AFFORDANCE_MAP_H56_POLICY_NAME,
    B8_SPATIAL_AFFORDANCE_SELECTION_SOURCE,
    B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME,
    B9_PATH_INTEGRATION_H48_POLICY_NAME,
    B9_ROUTE_MEMORY_H48_POLICY_NAME,
    B9_WAYPOINT_PLANNER_EFFECTIVE_LEVEL,
    B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
    B9_WAYPOINT_PLANNER_H56_POLICY_NAME,
    B9_WAYPOINT_PLANNER_SELECTION_SOURCE,
    B10_GENETIC_REPLAY_PLANNER_H48_POLICY_NAME,
    B10_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL,
    B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
    B10_PROSPECTIVE_REPLAY_H56_POLICY_NAME,
    B10_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
    B10_REPLAY_PLANNER_H48_POLICY_NAME,
    B10_VALUE_ROUTE_EVALUATOR_H48_POLICY_NAME,
    B11_CONFIDENCE_ARBITER_EFFECTIVE_LEVEL,
    B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
    B11_CONFIDENCE_ARBITER_H56_POLICY_NAME,
    B11_CONFIDENCE_ARBITER_SELECTION_SOURCE,
    B11_GENETIC_CONFIDENCE_GATE_H48_POLICY_NAME,
    B11_NEUROMODULATED_REPLAY_H48_POLICY_NAME,
    B11_UNCERTAINTY_GATE_H48_POLICY_NAME,
    B12_ACTIVE_INFERENCE_GATE_H48_POLICY_NAME,
    B12_AFFORDANCE_ATTENTION_H48_POLICY_NAME,
    B12_GENETIC_ATTENTION_GATE_H48_POLICY_NAME,
    B12_PREDICTIVE_ATTENTION_EFFECTIVE_LEVEL,
    B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
    B12_PREDICTIVE_ATTENTION_H56_POLICY_NAME,
    B12_PREDICTIVE_ATTENTION_SELECTION_SOURCE,
    B13_AFFORDANCE_SAMPLER_H48_POLICY_NAME,
    B13_COUNTERFACTUAL_ROUTE_H48_POLICY_NAME,
    B13_GENETIC_LOCAL_SEARCH_H48_POLICY_NAME,
    B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
    B13_LOCAL_AFFORDANCE_SEARCH_H56_POLICY_NAME,
    B13_LOCAL_SEARCH_EFFECTIVE_LEVEL,
    B13_LOCAL_SEARCH_SELECTION_SOURCE,
    B14_AFFORDANCE_UNCERTAINTY_EFFECTIVE_LEVEL,
    B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
    B14_AFFORDANCE_UNCERTAINTY_H56_POLICY_NAME,
    B14_AFFORDANCE_UNCERTAINTY_SELECTION_SOURCE,
    B14_CONFIDENCE_WEIGHTED_ROUTE_H48_POLICY_NAME,
    B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME,
    B14_RISK_CALIBRATED_SEARCH_H48_POLICY_NAME,
    B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME,
    B15_OPTION_CRITIC_EFFECTIVE_LEVEL,
    B15_OPTION_CRITIC_H48_POLICY_NAME,
    B15_OPTION_CRITIC_H56_POLICY_NAME,
    B15_OPTION_CRITIC_SELECTION_SOURCE,
    B15_PERSISTENCE_GATE_H48_POLICY_NAME,
    B15_VALUE_GATED_OPTION_H48_POLICY_NAME,
    B16_ACTION_SET_VOTER_H48_POLICY_NAME,
    B16_COMPETING_OPTIONS_H48_POLICY_NAME,
    B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME,
    B16_OPTION_ENSEMBLE_EFFECTIVE_LEVEL,
    B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
    B16_OPTION_ENSEMBLE_H56_POLICY_NAME,
    B16_OPTION_ENSEMBLE_SELECTION_SOURCE,
    B17_AROUSAL_GATED_OPTIONS_H48_POLICY_NAME,
    B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
    B17_HOMEOSTATIC_MODULATOR_H48_POLICY_NAME,
    B17_NEUROMODULATED_ENSEMBLE_EFFECTIVE_LEVEL,
    B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
    B17_NEUROMODULATED_ENSEMBLE_H56_POLICY_NAME,
    B17_NEUROMODULATED_ENSEMBLE_SELECTION_SOURCE,
    B18_ELIGIBILITY_TRACE_EFFECTIVE_LEVEL,
    B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
    B18_ELIGIBILITY_TRACE_H56_POLICY_NAME,
    B18_ELIGIBILITY_TRACE_SELECTION_SOURCE,
    B18_GENETIC_ELIGIBILITY_TRACE_H48_POLICY_NAME,
    B18_METASTABLE_AROUSAL_H48_POLICY_NAME,
    B18_SYNAPTIC_TRACE_MODULATOR_H48_POLICY_NAME,
    B19_EPISODIC_META_MEMORY_EFFECTIVE_LEVEL,
    B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
    B19_EPISODIC_META_MEMORY_H56_POLICY_NAME,
    B19_EPISODIC_META_MEMORY_SELECTION_SOURCE,
    B19_GENETIC_META_MEMORY_H48_POLICY_NAME,
    B19_STABILITY_MEMORY_H48_POLICY_NAME,
    B19_SWITCH_SUPPRESSION_H48_POLICY_NAME,
    B20_CONTEXT_BINDING_H48_POLICY_NAME,
    B20_GENETIC_WORKING_MEMORY_H48_POLICY_NAME,
    B20_STABILITY_BUFFER_H48_POLICY_NAME,
    B20_WORKING_MEMORY_GATE_EFFECTIVE_LEVEL,
    B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
    B20_WORKING_MEMORY_GATE_H56_POLICY_NAME,
    B20_WORKING_MEMORY_GATE_SELECTION_SOURCE,
    B21_GENETIC_REPLAY_GATE_H48_POLICY_NAME,
    B21_HIPPOCAMPAL_REPLAY_EFFECTIVE_LEVEL,
    B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
    B21_HIPPOCAMPAL_REPLAY_H56_POLICY_NAME,
    B21_HIPPOCAMPAL_REPLAY_SELECTION_SOURCE,
    B21_ROUTE_REHEARSAL_H48_POLICY_NAME,
    B21_SEQUENCE_BINDING_H48_POLICY_NAME,
    B22_FORWARD_MODEL_GATE_H48_POLICY_NAME,
    B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
    B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
    B22_PROSPECTIVE_MAP_REPLAY_H56_POLICY_NAME,
    B22_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL,
    B22_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
    B22_ROUTE_VIABILITY_SIM_H48_POLICY_NAME,
    B23_ABORT_CONFLICT_ARBITER_H48_POLICY_NAME,
    B23_CONFLICT_MONITOR_EFFECTIVE_LEVEL,
    B23_CONFLICT_MONITOR_H48_POLICY_NAME,
    B23_CONFLICT_MONITOR_H56_POLICY_NAME,
    B23_CONFLICT_MONITOR_SELECTION_SOURCE,
    B23_ERROR_GATED_REPLAY_H48_POLICY_NAME,
    B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME,
    B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME,
    B24_PRECISION_CONFLICT_EFFECTIVE_LEVEL,
    B24_PRECISION_CONFLICT_H48_POLICY_NAME,
    B24_PRECISION_CONFLICT_H56_POLICY_NAME,
    B24_PRECISION_CONFLICT_SELECTION_SOURCE,
    B24_PREDICTION_PRECISION_GATE_H48_POLICY_NAME,
    B24_RELIABILITY_ABORT_H48_POLICY_NAME,
    B25_CONFIDENCE_CALIBRATION_H48_POLICY_NAME,
    B25_GENETIC_METACOGNITION_H48_POLICY_NAME,
    B25_METACOGNITIVE_CONFIDENCE_EFFECTIVE_LEVEL,
    B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
    B25_METACOGNITIVE_CONFIDENCE_H56_POLICY_NAME,
    B25_METACOGNITIVE_CONFIDENCE_SELECTION_SOURCE,
    B25_UNCERTAINTY_INTEGRATOR_H48_POLICY_NAME,
    B26_ALLOSTATIC_PREDICTION_EFFECTIVE_LEVEL,
    B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
    B26_ALLOSTATIC_PREDICTION_H56_POLICY_NAME,
    B26_ALLOSTATIC_PREDICTION_SELECTION_SOURCE,
    B26_ERROR_SUPPRESSION_H48_POLICY_NAME,
    B26_GENETIC_ALLOSTASIS_H48_POLICY_NAME,
    B26_SETPOINT_DRIFT_H48_POLICY_NAME,
    B27_AROUSAL_GAIN_EFFECTIVE_LEVEL,
    B27_AROUSAL_GAIN_H48_POLICY_NAME,
    B27_AROUSAL_GAIN_H56_POLICY_NAME,
    B27_AROUSAL_GAIN_SELECTION_SOURCE,
    B27_ENERGY_AROUSAL_H48_POLICY_NAME,
    B27_GENETIC_AROUSAL_H48_POLICY_NAME,
    B27_STRESS_MODULATION_H48_POLICY_NAME,
    B28_GENETIC_ATTENTION_H48_POLICY_NAME,
    B28_HOMEOSTATIC_ATTENTION_H48_POLICY_NAME,
    B28_INTEROCEPTIVE_ATTENTION_EFFECTIVE_LEVEL,
    B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
    B28_INTEROCEPTIVE_ATTENTION_H56_POLICY_NAME,
    B28_INTEROCEPTIVE_ATTENTION_SELECTION_SOURCE,
    B28_THREAT_FOCUS_ATTENTION_H48_POLICY_NAME,
    B29_GENETIC_SALIENCE_H48_POLICY_NAME,
    B29_HOMEOSTATIC_SALIENCE_GATE_H48_POLICY_NAME,
    B29_SALIENCE_COMPETITION_EFFECTIVE_LEVEL,
    B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
    B29_SALIENCE_COMPETITION_H56_POLICY_NAME,
    B29_SALIENCE_COMPETITION_SELECTION_SOURCE,
    B29_THREAT_SALIENCE_GATE_H48_POLICY_NAME,
    B30_BASAL_GANGLIA_GATE_EFFECTIVE_LEVEL,
    B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
    B30_BASAL_GANGLIA_GATE_H56_POLICY_NAME,
    B30_BASAL_GANGLIA_GATE_SELECTION_SOURCE,
    B30_GENETIC_ACTION_GATE_H48_POLICY_NAME,
    B30_GO_NOGO_BALANCE_H48_POLICY_NAME,
    B30_THREAT_INHIBITION_GATE_H48_POLICY_NAME,
    B31_DOPAMINE_PREDICTION_ERROR_EFFECTIVE_LEVEL,
    B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
    B31_DOPAMINE_PREDICTION_ERROR_H56_POLICY_NAME,
    B31_DOPAMINE_PREDICTION_ERROR_SELECTION_SOURCE,
    B31_GENETIC_DOPAMINE_GATE_H48_POLICY_NAME,
    B31_PHASIC_DOPAMINE_GATE_H48_POLICY_NAME,
    B31_TONIC_DOPAMINE_GATE_H48_POLICY_NAME,
    B32_ACTOR_CRITIC_VALUE_EFFECTIVE_LEVEL,
    B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
    B32_ACTOR_CRITIC_VALUE_H56_POLICY_NAME,
    B32_ACTOR_CRITIC_VALUE_SELECTION_SOURCE,
    B32_ADVANTAGE_VALUE_GATE_H48_POLICY_NAME,
    B32_CRITIC_STABILITY_H48_POLICY_NAME,
    B32_GENETIC_ACTOR_CRITIC_H48_POLICY_NAME,
    B33_BOOTSTRAPPED_VALUE_GATE_H48_POLICY_NAME,
    B33_GENETIC_TD_VALUE_H48_POLICY_NAME,
    B33_REWARD_TRACE_CRITIC_H48_POLICY_NAME,
    B33_TD_ERROR_DECOMPOSITION_EFFECTIVE_LEVEL,
    B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
    B33_TD_ERROR_DECOMPOSITION_H56_POLICY_NAME,
    B33_TD_ERROR_DECOMPOSITION_SELECTION_SOURCE,
    B34_DELAYED_CREDIT_GATE_H48_POLICY_NAME,
    B34_ELIGIBILITY_CREDIT_EFFECTIVE_LEVEL,
    B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
    B34_ELIGIBILITY_CREDIT_H56_POLICY_NAME,
    B34_ELIGIBILITY_CREDIT_SELECTION_SOURCE,
    B34_GENETIC_ELIGIBILITY_H48_POLICY_NAME,
    B34_SYNAPTIC_TAGGING_H48_POLICY_NAME,
    B35_FORWARD_MODEL_VALUE_EFFECTIVE_LEVEL,
    B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
    B35_FORWARD_MODEL_VALUE_H56_POLICY_NAME,
    B35_FORWARD_MODEL_VALUE_SELECTION_SOURCE,
    B35_GENETIC_FORWARD_MODEL_H48_POLICY_NAME,
    B35_MODEL_CONFIDENCE_H48_POLICY_NAME,
    B35_TRANSITION_ERROR_GATE_H48_POLICY_NAME,
    B36_BELIEF_ERROR_GATE_H48_POLICY_NAME,
    B36_CONTEXT_INFERENCE_H48_POLICY_NAME,
    B36_GENETIC_BELIEF_STATE_H48_POLICY_NAME,
    B36_LATENT_BELIEF_STATE_EFFECTIVE_LEVEL,
    B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
    B36_LATENT_BELIEF_STATE_H56_POLICY_NAME,
    B36_LATENT_BELIEF_STATE_SELECTION_SOURCE,
    B37_FACTOR_CONFIDENCE_H48_POLICY_NAME,
    B37_GENETIC_STATE_FACTOR_H48_POLICY_NAME,
    B37_INTERO_EXTERO_FACTOR_H48_POLICY_NAME,
    B37_STATE_FACTOR_GATE_EFFECTIVE_LEVEL,
    B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
    B37_STATE_FACTOR_GATE_H56_POLICY_NAME,
    B37_STATE_FACTOR_GATE_SELECTION_SOURCE,
    B38_CONFIDENCE_ATTENTION_H48_POLICY_NAME,
    B38_FACTOR_ATTENTION_EFFECTIVE_LEVEL,
    B38_FACTOR_ATTENTION_H48_POLICY_NAME,
    B38_FACTOR_ATTENTION_H56_POLICY_NAME,
    B38_FACTOR_ATTENTION_SELECTION_SOURCE,
    B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME,
    B38_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
    B39_ATTENTION_BINDING_EFFECTIVE_LEVEL,
    B39_ATTENTION_BINDING_H48_POLICY_NAME,
    B39_ATTENTION_BINDING_H56_POLICY_NAME,
    B39_ATTENTION_BINDING_SELECTION_SOURCE,
    B39_CONTEXT_BINDING_ATTENTION_H48_POLICY_NAME,
    B39_CROSS_FACTOR_BINDING_H48_POLICY_NAME,
    B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME,
    B40_CONTEXT_WORKSPACE_H48_POLICY_NAME,
    B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME,
    B40_GLOBAL_WORKSPACE_EFFECTIVE_LEVEL,
    B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
    B40_GLOBAL_WORKSPACE_H56_POLICY_NAME,
    B40_GLOBAL_WORKSPACE_SELECTION_SOURCE,
    B40_SENSORY_WORKSPACE_H48_POLICY_NAME,
    B41_EXECUTIVE_WORKSPACE_EFFECTIVE_LEVEL,
    B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
    B41_EXECUTIVE_WORKSPACE_H56_POLICY_NAME,
    B41_EXECUTIVE_WORKSPACE_SELECTION_SOURCE,
    B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
    B41_GOAL_CONTEXT_SELECTOR_H48_POLICY_NAME,
    B41_INHIBITORY_CONTROL_H48_POLICY_NAME,
    B42_CONFLICT_MONITOR_H48_POLICY_NAME,
    B42_ERROR_MONITOR_EFFECTIVE_LEVEL,
    B42_ERROR_MONITOR_H48_POLICY_NAME,
    B42_ERROR_MONITOR_H56_POLICY_NAME,
    B42_ERROR_MONITOR_SELECTION_SOURCE,
    B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME,
    B42_PERFORMANCE_MONITOR_H48_POLICY_NAME,
    B43_ADAPTIVE_PRECISION_EFFECTIVE_LEVEL,
    B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
    B43_ADAPTIVE_PRECISION_H56_POLICY_NAME,
    B43_ADAPTIVE_PRECISION_SELECTION_SOURCE,
    B43_AROUSAL_PRECISION_H48_POLICY_NAME,
    B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME,
    B43_THRESHOLD_ADAPTATION_H48_POLICY_NAME,
    B44_CONTEXT_RELAY_H48_POLICY_NAME,
    B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME,
    B44_SENSORY_RELAY_H48_POLICY_NAME,
    B44_THALAMIC_RELAY_EFFECTIVE_LEVEL,
    B44_THALAMIC_RELAY_H48_POLICY_NAME,
    B44_THALAMIC_RELAY_H56_POLICY_NAME,
    B44_THALAMIC_RELAY_SELECTION_SOURCE,
    B45_CONTEXT_INHIBITION_H48_POLICY_NAME,
    B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME,
    B45_RETICULAR_INHIBITION_EFFECTIVE_LEVEL,
    B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
    B45_RETICULAR_INHIBITION_H56_POLICY_NAME,
    B45_RETICULAR_INHIBITION_SELECTION_SOURCE,
    B45_SENSORY_INHIBITION_H48_POLICY_NAME,
    B46_CONTEXT_FEEDBACK_H48_POLICY_NAME,
    B46_CORTICOTHALAMIC_FEEDBACK_EFFECTIVE_LEVEL,
    B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
    B46_CORTICOTHALAMIC_FEEDBACK_H56_POLICY_NAME,
    B46_CORTICOTHALAMIC_FEEDBACK_SELECTION_SOURCE,
    B46_FEEDBACK_GAIN_H48_POLICY_NAME,
    B46_GENETIC_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
    B47_COHERENCE_GATE_H48_POLICY_NAME,
    B47_GENETIC_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
    B47_OSCILLATORY_SYNCHRONY_EFFECTIVE_LEVEL,
    B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
    B47_OSCILLATORY_SYNCHRONY_H56_POLICY_NAME,
    B47_OSCILLATORY_SYNCHRONY_SELECTION_SOURCE,
    B47_PHASE_LOCKING_H48_POLICY_NAME,
    B48_CEREBELLAR_TIMING_EFFECTIVE_LEVEL,
    B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
    B48_CEREBELLAR_TIMING_H56_POLICY_NAME,
    B48_CEREBELLAR_TIMING_SELECTION_SOURCE,
    B48_GENETIC_CEREBELLAR_TIMING_H48_POLICY_NAME,
    B48_PREDICTIVE_TIMING_H48_POLICY_NAME,
    B48_TIMING_ERROR_CORRECTION_H48_POLICY_NAME,
    B49_DIRECT_PATH_FACILITATION_H48_POLICY_NAME,
    B49_GENETIC_STRIATAL_GATE_H48_POLICY_NAME,
    B49_INDIRECT_PATH_SUPPRESSION_H48_POLICY_NAME,
    B49_STRIATAL_ACTION_GATE_EFFECTIVE_LEVEL,
    B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
    B49_STRIATAL_ACTION_GATE_H56_POLICY_NAME,
    B49_STRIATAL_ACTION_GATE_SELECTION_SOURCE,
    B50_ACTION_CHUNK_VALUE_H48_POLICY_NAME,
    B50_GENETIC_HABIT_CHUNKING_H48_POLICY_NAME,
    B50_HABIT_CHUNKING_EFFECTIVE_LEVEL,
    B50_HABIT_CHUNKING_H48_POLICY_NAME,
    B50_HABIT_CHUNKING_H56_POLICY_NAME,
    B50_HABIT_CHUNKING_SELECTION_SOURCE,
    B50_HABIT_STABILITY_H48_POLICY_NAME,
    B51_DOPAMINERGIC_HABIT_MODULATION_EFFECTIVE_LEVEL,
    B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
    B51_DOPAMINERGIC_HABIT_MODULATION_H56_POLICY_NAME,
    B51_DOPAMINERGIC_HABIT_MODULATION_SELECTION_SOURCE,
    B51_GENETIC_DOPAMINE_HABIT_H48_POLICY_NAME,
    B51_NOVELTY_MODULATED_HABIT_H48_POLICY_NAME,
    B51_REWARD_PREDICTION_GAIN_H48_POLICY_NAME,
    B52_ATTENTION_GAIN_H48_POLICY_NAME,
    B52_CHOLINERGIC_PRECISION_GATE_EFFECTIVE_LEVEL,
    B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
    B52_CHOLINERGIC_PRECISION_GATE_H56_POLICY_NAME,
    B52_CHOLINERGIC_PRECISION_GATE_SELECTION_SOURCE,
    B52_GENETIC_CHOLINERGIC_PRECISION_H48_POLICY_NAME,
    B52_UNCERTAINTY_RELEASE_H48_POLICY_NAME,
    B53_GENETIC_AROUSAL_PRECISION_H48_POLICY_NAME,
    B53_NORADRENERGIC_AROUSAL_GAIN_EFFECTIVE_LEVEL,
    B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
    B53_NORADRENERGIC_AROUSAL_GAIN_H56_POLICY_NAME,
    B53_NORADRENERGIC_AROUSAL_GAIN_SELECTION_SOURCE,
    B53_STRESS_PRECISION_H48_POLICY_NAME,
    B53_SURPRISE_GAIN_H48_POLICY_NAME,
    B54_GENETIC_SEROTONIN_PATIENCE_H48_POLICY_NAME,
    B54_IMPULSE_SUPPRESSION_H48_POLICY_NAME,
    B54_PATIENCE_BALANCE_H48_POLICY_NAME,
    B54_SEROTONERGIC_PATIENCE_GATE_EFFECTIVE_LEVEL,
    B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
    B54_SEROTONERGIC_PATIENCE_GATE_H56_POLICY_NAME,
    B54_SEROTONERGIC_PATIENCE_GATE_SELECTION_SOURCE,
    B55_GENETIC_HYPOTHALAMIC_DRIVE_H48_POLICY_NAME,
    B55_HYPOTHALAMIC_DRIVE_COUPLING_EFFECTIVE_LEVEL,
    B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
    B55_HYPOTHALAMIC_DRIVE_COUPLING_H56_POLICY_NAME,
    B55_HYPOTHALAMIC_DRIVE_COUPLING_SELECTION_SOURCE,
    B55_SATIETY_RECOVERY_BALANCE_H48_POLICY_NAME,
    B55_SLEEP_HUNGER_ARBITER_H48_POLICY_NAME,
    B56_CORTISOL_RECOVERY_BALANCE_H48_POLICY_NAME,
    B56_GENETIC_HPA_STRESS_H48_POLICY_NAME,
    B56_HPA_STRESS_AXIS_EFFECTIVE_LEVEL,
    B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
    B56_HPA_STRESS_AXIS_H56_POLICY_NAME,
    B56_HPA_STRESS_AXIS_SELECTION_SOURCE,
    B56_STRESS_LOAD_GATE_H48_POLICY_NAME,
    B57_GENETIC_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
    B57_INSULAR_INTEROCEPTIVE_AWARENESS_EFFECTIVE_LEVEL,
    B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
    B57_INSULAR_INTEROCEPTIVE_AWARENESS_H56_POLICY_NAME,
    B57_INSULAR_INTEROCEPTIVE_AWARENESS_SELECTION_SOURCE,
    B57_STRESS_DRIVE_AWARENESS_H48_POLICY_NAME,
    B57_VISCERAL_SALIENCE_GATE_H48_POLICY_NAME,
    B58_ACC_CONFLICT_MONITOR_EFFECTIVE_LEVEL,
    B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
    B58_ACC_CONFLICT_MONITOR_H56_POLICY_NAME,
    B58_ACC_CONFLICT_MONITOR_SELECTION_SOURCE,
    B58_CONFLICT_RESOLUTION_BALANCE_H48_POLICY_NAME,
    B58_ERROR_SALIENCE_GATE_H48_POLICY_NAME,
    B58_GENETIC_ACC_CONFLICT_H48_POLICY_NAME,
    B59_EXECUTIVE_TASK_SET_H48_POLICY_NAME,
    B59_GENETIC_PREFRONTAL_CONTROL_H48_POLICY_NAME,
    B59_PREFRONTAL_GOAL_CONTEXT_EFFECTIVE_LEVEL,
    B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
    B59_PREFRONTAL_GOAL_CONTEXT_H56_POLICY_NAME,
    B59_PREFRONTAL_GOAL_CONTEXT_SELECTION_SOURCE,
    B59_WORKING_SET_STABILITY_H48_POLICY_NAME,
    B60_GENETIC_ORBITOFRONTAL_VALUE_H48_POLICY_NAME,
    B60_GOAL_OUTCOME_PREDICTION_H48_POLICY_NAME,
    B60_ORBITOFRONTAL_OUTCOME_VALUE_EFFECTIVE_LEVEL,
    B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
    B60_ORBITOFRONTAL_OUTCOME_VALUE_H56_POLICY_NAME,
    B60_ORBITOFRONTAL_OUTCOME_VALUE_SELECTION_SOURCE,
    B60_REVERSAL_VALUE_GATE_H48_POLICY_NAME,
    B61_AMYGDALA_SAFETY_VALUE_EFFECTIVE_LEVEL,
    B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
    B61_AMYGDALA_SAFETY_VALUE_H56_POLICY_NAME,
    B61_AMYGDALA_SAFETY_VALUE_SELECTION_SOURCE,
    B61_GENETIC_AMYGDALA_SAFETY_H48_POLICY_NAME,
    B61_SAFETY_PREDICTION_GATE_H48_POLICY_NAME,
    B61_THREAT_VALUE_TAG_H48_POLICY_NAME,
    B62_DEFENSIVE_MODE_SELECTOR_EFFECTIVE_LEVEL,
    B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
    B62_DEFENSIVE_MODE_SELECTOR_H56_POLICY_NAME,
    B62_DEFENSIVE_MODE_SELECTOR_SELECTION_SOURCE,
    B62_FREEZE_FLEE_BALANCE_H48_POLICY_NAME,
    B62_GENETIC_DEFENSIVE_MODE_H48_POLICY_NAME,
    B62_SHELTER_DEFENSE_GATE_H48_POLICY_NAME,
    B_SEMANTIC_ACTIONS,
    B_SEMANTIC_ACTION_TO_INDEX,
    bridge_b_semantic_action,
)
from spider_cortex_sim.b_series_evolution import (
    build_b1_capacity_config,
    build_b2_temporal_threat_config,
    build_b3_contact_memory_config,
    build_b4_recovery_balance_config,
    build_b5_homeostatic_arbiter_config,
    build_b6_risk_corridor_config,
    build_b7_affordance_budget_config,
    build_b8_spatial_affordance_config,
    build_b9_waypoint_planner_config,
    build_b10_prospective_replay_config,
    build_b11_confidence_arbiter_config,
    build_b12_predictive_attention_config,
    build_b13_local_affordance_search_config,
    build_b14_affordance_uncertainty_config,
    build_b15_option_critic_config,
    build_b16_option_ensemble_config,
    build_b17_neuromodulated_ensemble_config,
    build_b18_eligibility_trace_config,
    build_b19_episodic_meta_memory_config,
    build_b20_working_memory_gate_config,
    build_b21_hippocampal_replay_config,
    build_b22_prospective_replay_config,
    build_b23_conflict_monitor_config,
    build_b24_precision_conflict_config,
    build_b25_metacognitive_confidence_config,
    build_b26_allostatic_prediction_config,
    build_b27_arousal_gain_config,
    build_b28_interoceptive_attention_config,
    build_b29_salience_competition_config,
    build_b30_basal_ganglia_gate_config,
    build_b31_dopamine_prediction_error_config,
    build_b32_actor_critic_value_config,
    build_b33_td_error_decomposition_config,
    build_b34_eligibility_credit_config,
    build_b35_forward_model_value_config,
    build_b36_latent_belief_state_config,
    build_b37_state_factor_gate_config,
    build_b38_factor_attention_config,
    build_b39_attention_binding_config,
    build_b40_global_workspace_config,
    build_b41_executive_workspace_config,
    build_b42_error_monitor_config,
    build_b43_adaptive_precision_config,
    build_b44_thalamic_relay_config,
    build_b45_reticular_inhibition_config,
    build_b46_corticothalamic_feedback_config,
    build_b47_oscillatory_synchrony_config,
    build_b48_cerebellar_timing_config,
    build_b49_striatal_action_gate_config,
    build_b50_habit_chunking_config,
    build_b51_dopaminergic_habit_modulation_config,
    build_b52_cholinergic_precision_gate_config,
    build_b53_noradrenergic_arousal_gain_config,
    build_b54_serotonergic_patience_gate_config,
    build_b55_hypothalamic_drive_coupling_config,
    build_b56_hpa_stress_axis_config,
    build_b57_insular_interoceptive_awareness_config,
    build_b58_acc_conflict_monitor_config,
    build_b59_prefrontal_goal_context_config,
    build_b60_orbitofrontal_outcome_value_config,
    build_b61_amygdala_safety_value_config,
    build_b62_defensive_mode_selector_config,
    b4_canonical_multi_gate_result,
    b5_food_deprivation_gate_result,
    b5_sleep_conflict_gate_result,
    b6_corridor_progress_gate_result,
    b6_food_predator_conflict_gate_result,
    b7_corridor_viability_gate_result,
    b8_spatial_corridor_gate_result,
    b9_waypoint_corridor_gate_result,
    b10_prospective_corridor_gate_result,
    b11_confidence_corridor_gate_result,
    b12_attention_corridor_gate_result,
    b13_local_search_corridor_gate_result,
    b14_uncertainty_corridor_gate_result,
    b15_option_corridor_gate_result,
    b16_option_ensemble_corridor_gate_result,
    b17_neuromodulated_corridor_gate_result,
    b18_eligibility_corridor_gate_result,
    b19_episodic_corridor_gate_result,
    b20_working_memory_corridor_gate_result,
    b21_hippocampal_replay_corridor_gate_result,
    b22_prospective_replay_corridor_gate_result,
    b23_conflict_monitor_corridor_gate_result,
    b24_precision_conflict_corridor_gate_result,
    b25_metacognitive_confidence_corridor_gate_result,
    b26_allostatic_prediction_corridor_gate_result,
    b27_arousal_gain_corridor_gate_result,
    b28_interoceptive_attention_corridor_gate_result,
    b29_salience_competition_corridor_gate_result,
    b30_basal_ganglia_gate_corridor_gate_result,
    b31_dopamine_prediction_error_corridor_gate_result,
    b32_actor_critic_value_corridor_gate_result,
    b33_td_error_decomposition_corridor_gate_result,
    b34_eligibility_credit_corridor_gate_result,
    b35_forward_model_value_corridor_gate_result,
    b36_latent_belief_state_corridor_gate_result,
    b37_state_factor_gate_corridor_gate_result,
    b38_factor_attention_corridor_gate_result,
    b39_attention_binding_corridor_gate_result,
    b40_global_workspace_corridor_gate_result,
    b41_executive_workspace_corridor_gate_result,
    b42_error_monitor_corridor_gate_result,
    b43_adaptive_precision_corridor_gate_result,
    b44_thalamic_relay_corridor_gate_result,
    b45_reticular_inhibition_corridor_gate_result,
    b46_corticothalamic_feedback_corridor_gate_result,
    b47_oscillatory_synchrony_corridor_gate_result,
    b48_cerebellar_timing_corridor_gate_result,
    b49_striatal_action_gate_corridor_gate_result,
    b50_habit_chunking_corridor_gate_result,
    b51_dopaminergic_habit_corridor_gate_result,
    b52_cholinergic_precision_corridor_gate_result,
    b53_noradrenergic_arousal_corridor_gate_result,
    b54_serotonergic_patience_corridor_gate_result,
    b55_hypothalamic_drive_corridor_gate_result,
    b56_hpa_stress_corridor_gate_result,
    b57_insular_interoceptive_corridor_gate_result,
    b58_acc_conflict_corridor_gate_result,
    b59_prefrontal_goal_corridor_gate_result,
    b60_orbitofrontal_value_corridor_gate_result,
    b61_amygdala_safety_corridor_gate_result,
    b62_defensive_mode_corridor_gate_result,
    select_b6_promotion,
    trace_uses_only_primitive_actions,
)
from spider_cortex_sim.bus import MessageBus
from spider_cortex_sim.b_series_legacy import (
    LEGACY_B0_ACTIONS,
    LegacyB0Simulation,
)
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_TO_INDEX,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.simulation import SpiderSimulation
from spider_cortex_sim.world import ACTIONS


def _bridge_observation() -> dict[str, object]:
    return {
        "meta": {
            "on_food": False,
            "on_shelter": False,
            "local_affordances": {
                "MOVE_UP": {"blocked": False},
                "MOVE_DOWN": {"blocked": False},
                "MOVE_LEFT": {"blocked": False},
                "MOVE_RIGHT": {"blocked": False},
            },
            "local_transition_consequences": {
                "MOVE_UP": {
                    "food_dist_delta": -1.0,
                    "shelter_dist_delta": -1.0,
                },
                "MOVE_DOWN": {
                    "food_dist_delta": 0.0,
                    "shelter_dist_delta": 0.0,
                },
                "MOVE_LEFT": {
                    "food_dist_delta": 0.25,
                    "shelter_dist_delta": 1.5,
                },
                "MOVE_RIGHT": {
                    "food_dist_delta": 1.0,
                    "shelter_dist_delta": 0.2,
                },
            },
            "local_geodesic_consequences": {
                "MOVE_LEFT": {
                    "exit_geodesic_delta": 1.0,
                    "deep_geodesic_delta": 1.0,
                },
                "MOVE_RIGHT": {
                    "exit_geodesic_delta": 0.0,
                    "deep_geodesic_delta": 0.0,
                },
            },
        }
    }


def _set_module_values(
    observation: dict[str, object],
    module_name: str,
    values: dict[str, float],
) -> None:
    spec = MODULE_INTERFACE_BY_NAME[module_name]
    vector = np.asarray(observation[spec.observation_key], dtype=float).copy()
    signal_to_index = {name: idx for idx, name in enumerate(spec.signal_names)}
    for name, value in values.items():
        vector[signal_to_index[name]] = float(value)
    observation[spec.observation_key] = vector


def _brain_observation(
    meta: dict[str, object] | None = None,
    *,
    hunger: dict[str, float] | None = None,
    sleep: dict[str, float] | None = None,
    threat: dict[str, float] | None = None,
) -> dict[str, object]:
    observation: dict[str, object] = {
        spec.observation_key: np.zeros(spec.input_dim, dtype=float)
        for spec in MODULE_INTERFACES
    }
    observation[ACTION_CONTEXT_INTERFACE.observation_key] = np.zeros(
        ACTION_CONTEXT_INTERFACE.input_dim,
        dtype=float,
    )
    observation[MOTOR_CONTEXT_INTERFACE.observation_key] = np.zeros(
        MOTOR_CONTEXT_INTERFACE.input_dim,
        dtype=float,
    )
    observation["meta"] = meta or _bridge_observation()["meta"]
    if hunger:
        _set_module_values(observation, "hunger_center", hunger)
    if sleep:
        _set_module_values(observation, "sleep_center", sleep)
    if threat:
        _set_module_values(observation, "threat_center", threat)
    return observation


def _b0_config(name: str = "b0_current_bridge_policy") -> BrainAblationConfig:
    return resolve_ablation_configs([name])[0]


def _save_b1_threat_guard_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b0_source = SpiderBrain(seed=101, module_dropout=0.0, config=_b0_config())
    b0_checkpoint = b0_source.save(tmp_path / "b0")
    b1_config = build_b1_capacity_config(
        B1_THREAT_GUARD_POLICY_NAME,
        source_checkpoint=b0_checkpoint,
    )
    b1_source = SpiderBrain(seed=102, module_dropout=0.0, config=b1_config)
    return b1_source.save(tmp_path / "b1")


def _save_b2_temporal_threat_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b1_checkpoint = _save_b1_threat_guard_source(tmp_path)
    b2_config = build_b2_temporal_threat_config(
        B2_TEMPORAL_THREAT_H48_POLICY_NAME,
        source_checkpoint=b1_checkpoint,
    )
    b2_source = SpiderBrain(seed=103, module_dropout=0.0, config=b2_config)
    return b2_source.save(tmp_path / "b2")


def _save_b3_recurrent_guard_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b2_checkpoint = _save_b2_temporal_threat_source(tmp_path)
    b3_config = build_b3_contact_memory_config(
        B3_RECURRENT_GUARD_H48_POLICY_NAME,
        source_checkpoint=b2_checkpoint,
    )
    b3_source = SpiderBrain(seed=104, module_dropout=0.0, config=b3_config)
    return b3_source.save(tmp_path / "b3")


def _save_b4_genetic_recovery_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b3_checkpoint = _save_b3_recurrent_guard_source(tmp_path)
    b4_config = build_b4_recovery_balance_config(
        B4_GENETIC_RECOVERY_H48_POLICY_NAME,
        source_checkpoint=b3_checkpoint,
        controller_profile="genetic_recovery",
    )
    b4_source = SpiderBrain(seed=105, module_dropout=0.0, config=b4_config)
    return b4_source.save(tmp_path / "b4")


def _save_b5_genetic_homeostasis_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b4_checkpoint = _save_b4_genetic_recovery_source(tmp_path)
    b5_config = build_b5_homeostatic_arbiter_config(
        B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
        source_checkpoint=b4_checkpoint,
        controller_profile="genetic_homeostasis",
    )
    b5_source = SpiderBrain(seed=106, module_dropout=0.0, config=b5_config)
    return b5_source.save(tmp_path / "b5")


def _save_b6_fused_risk_recurrent_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b5_checkpoint = _save_b5_genetic_homeostasis_source(tmp_path)
    b6_config = build_b6_risk_corridor_config(
        B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
        source_checkpoint=b5_checkpoint,
        controller_profile="fused_risk_recurrent",
    )
    b6_source = SpiderBrain(seed=107, module_dropout=0.0, config=b6_config)
    return b6_source.save(tmp_path / "b6")


def _save_b7_affordance_budget_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b6_checkpoint = _save_b6_fused_risk_recurrent_source(tmp_path)
    b7_config = build_b7_affordance_budget_config(
        B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
        source_checkpoint=b6_checkpoint,
    )
    b7_source = SpiderBrain(seed=108, module_dropout=0.0, config=b7_config)
    return b7_source.save(tmp_path / "b7")


def _save_b8_spatial_affordance_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b7_checkpoint = _save_b7_affordance_budget_source(tmp_path)
    b8_config = build_b8_spatial_affordance_config(
        B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
        source_checkpoint=b7_checkpoint,
    )
    b8_source = SpiderBrain(seed=109, module_dropout=0.0, config=b8_config)
    return b8_source.save(tmp_path / "b8")


def _save_b9_waypoint_planner_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b8_checkpoint = _save_b8_spatial_affordance_source(tmp_path)
    b9_config = build_b9_waypoint_planner_config(
        B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
        source_checkpoint=b8_checkpoint,
    )
    b9_source = SpiderBrain(seed=110, module_dropout=0.0, config=b9_config)
    return b9_source.save(tmp_path / "b9")


def _save_b10_prospective_replay_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b9_checkpoint = _save_b9_waypoint_planner_source(tmp_path)
    b10_config = build_b10_prospective_replay_config(
        B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
        source_checkpoint=b9_checkpoint,
    )
    b10_source = SpiderBrain(seed=111, module_dropout=0.0, config=b10_config)
    return b10_source.save(tmp_path / "b10")


def _save_b11_confidence_arbiter_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b10_checkpoint = _save_b10_prospective_replay_source(tmp_path)
    b11_config = build_b11_confidence_arbiter_config(
        B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
        source_checkpoint=b10_checkpoint,
    )
    b11_source = SpiderBrain(seed=112, module_dropout=0.0, config=b11_config)
    return b11_source.save(tmp_path / "b11")


def _save_b12_predictive_attention_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b11_checkpoint = _save_b11_confidence_arbiter_source(tmp_path)
    b12_config = build_b12_predictive_attention_config(
        B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
        source_checkpoint=b11_checkpoint,
    )
    b12_source = SpiderBrain(seed=113, module_dropout=0.0, config=b12_config)
    return b12_source.save(tmp_path / "b12")


def _save_b13_local_affordance_search_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b12_checkpoint = _save_b12_predictive_attention_source(tmp_path)
    b13_config = build_b13_local_affordance_search_config(
        B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
        source_checkpoint=b12_checkpoint,
    )
    b13_source = SpiderBrain(seed=114, module_dropout=0.0, config=b13_config)
    return b13_source.save(tmp_path / "b13")


def _save_b14_affordance_uncertainty_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b13_checkpoint = _save_b13_local_affordance_search_source(tmp_path)
    b14_config = build_b14_affordance_uncertainty_config(
        B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
        source_checkpoint=b13_checkpoint,
    )
    b14_source = SpiderBrain(seed=115, module_dropout=0.0, config=b14_config)
    return b14_source.save(tmp_path / "b14")


def _save_b15_option_critic_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b14_checkpoint = _save_b14_affordance_uncertainty_source(tmp_path)
    b15_config = build_b15_option_critic_config(
        B15_OPTION_CRITIC_H48_POLICY_NAME,
        source_checkpoint=b14_checkpoint,
    )
    b15_source = SpiderBrain(seed=116, module_dropout=0.0, config=b15_config)
    return b15_source.save(tmp_path / "b15")


def _save_b16_option_ensemble_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b15_checkpoint = _save_b15_option_critic_source(tmp_path)
    b16_config = build_b16_option_ensemble_config(
        B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
        source_checkpoint=b15_checkpoint,
    )
    b16_source = SpiderBrain(seed=117, module_dropout=0.0, config=b16_config)
    return b16_source.save(tmp_path / "b16")


def _save_b17_neuromodulated_ensemble_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b16_checkpoint = _save_b16_option_ensemble_source(tmp_path)
    b17_config = build_b17_neuromodulated_ensemble_config(
        B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
        source_checkpoint=b16_checkpoint,
    )
    b17_source = SpiderBrain(seed=118, module_dropout=0.0, config=b17_config)
    return b17_source.save(tmp_path / "b17")


def _save_b18_eligibility_trace_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b17_checkpoint = _save_b17_neuromodulated_ensemble_source(tmp_path)
    b18_config = build_b18_eligibility_trace_config(
        B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
        source_checkpoint=b17_checkpoint,
    )
    b18_source = SpiderBrain(seed=119, module_dropout=0.0, config=b18_config)
    return b18_source.save(tmp_path / "b18")


def _save_b19_episodic_meta_memory_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b18_checkpoint = _save_b18_eligibility_trace_source(tmp_path)
    b19_config = build_b19_episodic_meta_memory_config(
        B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
        source_checkpoint=b18_checkpoint,
    )
    b19_source = SpiderBrain(seed=120, module_dropout=0.0, config=b19_config)
    return b19_source.save(tmp_path / "b19")


def _save_b20_working_memory_gate_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b19_checkpoint = _save_b19_episodic_meta_memory_source(tmp_path)
    b20_config = build_b20_working_memory_gate_config(
        B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
        source_checkpoint=b19_checkpoint,
    )
    b20_source = SpiderBrain(seed=121, module_dropout=0.0, config=b20_config)
    return b20_source.save(tmp_path / "b20")


def _save_b21_hippocampal_replay_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b20_checkpoint = _save_b20_working_memory_gate_source(tmp_path)
    b21_config = build_b21_hippocampal_replay_config(
        B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
        source_checkpoint=b20_checkpoint,
    )
    b21_source = SpiderBrain(seed=122, module_dropout=0.0, config=b21_config)
    return b21_source.save(tmp_path / "b21")


def _save_b22_prospective_replay_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b21_checkpoint = _save_b21_hippocampal_replay_source(tmp_path)
    b22_config = build_b22_prospective_replay_config(
        B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
        source_checkpoint=b21_checkpoint,
    )
    b22_source = SpiderBrain(seed=123, module_dropout=0.0, config=b22_config)
    return b22_source.save(tmp_path / "b22")


def _save_b23_conflict_monitor_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b22_checkpoint = _save_b22_prospective_replay_source(tmp_path)
    b23_config = build_b23_conflict_monitor_config(
        B23_CONFLICT_MONITOR_H48_POLICY_NAME,
        source_checkpoint=b22_checkpoint,
    )
    b23_source = SpiderBrain(seed=124, module_dropout=0.0, config=b23_config)
    return b23_source.save(tmp_path / "b23")


def _save_b24_precision_conflict_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b23_checkpoint = _save_b23_conflict_monitor_source(tmp_path)
    b24_config = build_b24_precision_conflict_config(
        B24_PRECISION_CONFLICT_H48_POLICY_NAME,
        source_checkpoint=b23_checkpoint,
    )
    b24_source = SpiderBrain(seed=125, module_dropout=0.0, config=b24_config)
    return b24_source.save(tmp_path / "b24")


def _save_b25_metacognitive_confidence_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b24_checkpoint = _save_b24_precision_conflict_source(tmp_path)
    b25_config = build_b25_metacognitive_confidence_config(
        B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
        source_checkpoint=b24_checkpoint,
    )
    b25_source = SpiderBrain(seed=126, module_dropout=0.0, config=b25_config)
    return b25_source.save(tmp_path / "b25")


def _save_b26_allostatic_prediction_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b25_checkpoint = _save_b25_metacognitive_confidence_source(tmp_path)
    b26_config = build_b26_allostatic_prediction_config(
        B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
        source_checkpoint=b25_checkpoint,
    )
    b26_source = SpiderBrain(seed=127, module_dropout=0.0, config=b26_config)
    return b26_source.save(tmp_path / "b26")


def _save_b27_arousal_gain_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b26_checkpoint = _save_b26_allostatic_prediction_source(tmp_path)
    b27_config = build_b27_arousal_gain_config(
        B27_AROUSAL_GAIN_H48_POLICY_NAME,
        source_checkpoint=b26_checkpoint,
    )
    b27_source = SpiderBrain(seed=128, module_dropout=0.0, config=b27_config)
    return b27_source.save(tmp_path / "b27")


def _save_b28_interoceptive_attention_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b27_checkpoint = _save_b27_arousal_gain_source(tmp_path)
    b28_config = build_b28_interoceptive_attention_config(
        B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
        source_checkpoint=b27_checkpoint,
    )
    b28_source = SpiderBrain(seed=129, module_dropout=0.0, config=b28_config)
    return b28_source.save(tmp_path / "b28")


def _save_b29_salience_competition_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b28_checkpoint = _save_b28_interoceptive_attention_source(tmp_path)
    b29_config = build_b29_salience_competition_config(
        B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
        source_checkpoint=b28_checkpoint,
    )
    b29_source = SpiderBrain(seed=130, module_dropout=0.0, config=b29_config)
    return b29_source.save(tmp_path / "b29")


def _save_b30_basal_ganglia_gate_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b29_checkpoint = _save_b29_salience_competition_source(tmp_path)
    b30_config = build_b30_basal_ganglia_gate_config(
        B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
        source_checkpoint=b29_checkpoint,
    )
    b30_source = SpiderBrain(seed=131, module_dropout=0.0, config=b30_config)
    return b30_source.save(tmp_path / "b30")


def _save_b31_dopamine_prediction_error_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b30_checkpoint = _save_b30_basal_ganglia_gate_source(tmp_path)
    b31_config = build_b31_dopamine_prediction_error_config(
        B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
        source_checkpoint=b30_checkpoint,
    )
    b31_source = SpiderBrain(seed=132, module_dropout=0.0, config=b31_config)
    return b31_source.save(tmp_path / "b31")


def _save_b32_actor_critic_value_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b31_checkpoint = _save_b31_dopamine_prediction_error_source(tmp_path)
    b32_config = build_b32_actor_critic_value_config(
        B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
        source_checkpoint=b31_checkpoint,
    )
    b32_source = SpiderBrain(seed=133, module_dropout=0.0, config=b32_config)
    return b32_source.save(tmp_path / "b32")


def _save_b33_td_error_decomposition_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b32_checkpoint = _save_b32_actor_critic_value_source(tmp_path)
    b33_config = build_b33_td_error_decomposition_config(
        B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
        source_checkpoint=b32_checkpoint,
    )
    b33_source = SpiderBrain(seed=134, module_dropout=0.0, config=b33_config)
    return b33_source.save(tmp_path / "b33")


def _save_b34_eligibility_credit_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b33_checkpoint = _save_b33_td_error_decomposition_source(tmp_path)
    b34_config = build_b34_eligibility_credit_config(
        B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
        source_checkpoint=b33_checkpoint,
    )
    b34_source = SpiderBrain(seed=135, module_dropout=0.0, config=b34_config)
    return b34_source.save(tmp_path / "b34")


def _save_b35_forward_model_value_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b34_checkpoint = _save_b34_eligibility_credit_source(tmp_path)
    b35_config = build_b35_forward_model_value_config(
        B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
        source_checkpoint=b34_checkpoint,
    )
    b35_source = SpiderBrain(seed=136, module_dropout=0.0, config=b35_config)
    return b35_source.save(tmp_path / "b35")


def _save_b36_latent_belief_state_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b35_checkpoint = _save_b35_forward_model_value_source(tmp_path)
    b36_config = build_b36_latent_belief_state_config(
        B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
        source_checkpoint=b35_checkpoint,
    )
    b36_source = SpiderBrain(seed=137, module_dropout=0.0, config=b36_config)
    return b36_source.save(tmp_path / "b36")


def _save_b37_state_factor_gate_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b36_checkpoint = _save_b36_latent_belief_state_source(tmp_path)
    b37_config = build_b37_state_factor_gate_config(
        B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
        source_checkpoint=b36_checkpoint,
    )
    b37_source = SpiderBrain(seed=138, module_dropout=0.0, config=b37_config)
    return b37_source.save(tmp_path / "b37")


def _save_b38_factor_attention_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b37_checkpoint = _save_b37_state_factor_gate_source(tmp_path)
    b38_config = build_b38_factor_attention_config(
        B38_FACTOR_ATTENTION_H48_POLICY_NAME,
        source_checkpoint=b37_checkpoint,
    )
    b38_source = SpiderBrain(seed=139, module_dropout=0.0, config=b38_config)
    return b38_source.save(tmp_path / "b38")


def _save_b39_attention_binding_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b38_checkpoint = _save_b38_factor_attention_source(tmp_path)
    b39_config = build_b39_attention_binding_config(
        B39_ATTENTION_BINDING_H48_POLICY_NAME,
        source_checkpoint=b38_checkpoint,
    )
    b39_source = SpiderBrain(seed=140, module_dropout=0.0, config=b39_config)
    return b39_source.save(tmp_path / "b39")


def _save_b40_global_workspace_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b39_checkpoint = _save_b39_attention_binding_source(tmp_path)
    b40_config = build_b40_global_workspace_config(
        B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
        source_checkpoint=b39_checkpoint,
    )
    b40_source = SpiderBrain(seed=141, module_dropout=0.0, config=b40_config)
    return b40_source.save(tmp_path / "b40")


def _save_b41_executive_workspace_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b40_checkpoint = _save_b40_global_workspace_source(tmp_path)
    b41_config = build_b41_executive_workspace_config(
        B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
        source_checkpoint=b40_checkpoint,
    )
    b41_source = SpiderBrain(seed=142, module_dropout=0.0, config=b41_config)
    return b41_source.save(tmp_path / "b41")


def _save_b42_error_monitor_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b41_checkpoint = _save_b41_executive_workspace_source(tmp_path)
    b42_config = build_b42_error_monitor_config(
        B42_ERROR_MONITOR_H48_POLICY_NAME,
        source_checkpoint=b41_checkpoint,
    )
    b42_source = SpiderBrain(seed=143, module_dropout=0.0, config=b42_config)
    return b42_source.save(tmp_path / "b42")


def _save_b43_adaptive_precision_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b42_checkpoint = _save_b42_error_monitor_source(tmp_path)
    b43_config = build_b43_adaptive_precision_config(
        B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
        source_checkpoint=b42_checkpoint,
    )
    b43_source = SpiderBrain(seed=144, module_dropout=0.0, config=b43_config)
    return b43_source.save(tmp_path / "b43")


def _save_b44_thalamic_relay_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b43_checkpoint = _save_b43_adaptive_precision_source(tmp_path)
    b44_config = build_b44_thalamic_relay_config(
        B44_THALAMIC_RELAY_H48_POLICY_NAME,
        source_checkpoint=b43_checkpoint,
    )
    b44_source = SpiderBrain(seed=145, module_dropout=0.0, config=b44_config)
    return b44_source.save(tmp_path / "b44")


def _save_b45_reticular_inhibition_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b44_checkpoint = _save_b44_thalamic_relay_source(tmp_path)
    b45_config = build_b45_reticular_inhibition_config(
        B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
        source_checkpoint=b44_checkpoint,
    )
    b45_source = SpiderBrain(seed=146, module_dropout=0.0, config=b45_config)
    return b45_source.save(tmp_path / "b45")


def _save_b46_corticothalamic_feedback_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b45_checkpoint = _save_b45_reticular_inhibition_source(tmp_path)
    b46_config = build_b46_corticothalamic_feedback_config(
        B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
        source_checkpoint=b45_checkpoint,
    )
    b46_source = SpiderBrain(seed=147, module_dropout=0.0, config=b46_config)
    return b46_source.save(tmp_path / "b46")


def _save_b47_oscillatory_synchrony_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b46_checkpoint = _save_b46_corticothalamic_feedback_source(tmp_path)
    b47_config = build_b47_oscillatory_synchrony_config(
        B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
        source_checkpoint=b46_checkpoint,
    )
    b47_source = SpiderBrain(seed=148, module_dropout=0.0, config=b47_config)
    return b47_source.save(tmp_path / "b47")


def _save_b48_cerebellar_timing_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b47_checkpoint = _save_b47_oscillatory_synchrony_source(tmp_path)
    b48_config = build_b48_cerebellar_timing_config(
        B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
        source_checkpoint=b47_checkpoint,
    )
    b48_source = SpiderBrain(seed=149, module_dropout=0.0, config=b48_config)
    return b48_source.save(tmp_path / "b48")


def _save_b49_striatal_action_gate_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b48_checkpoint = _save_b48_cerebellar_timing_source(tmp_path)
    b49_config = build_b49_striatal_action_gate_config(
        B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
        source_checkpoint=b48_checkpoint,
    )
    b49_source = SpiderBrain(seed=150, module_dropout=0.0, config=b49_config)
    return b49_source.save(tmp_path / "b49")


def _save_b50_habit_chunking_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b49_checkpoint = _save_b49_striatal_action_gate_source(tmp_path)
    b50_config = build_b50_habit_chunking_config(
        B50_HABIT_CHUNKING_H48_POLICY_NAME,
        source_checkpoint=b49_checkpoint,
    )
    b50_source = SpiderBrain(seed=151, module_dropout=0.0, config=b50_config)
    return b50_source.save(tmp_path / "b50")


def _save_b51_dopaminergic_habit_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b50_checkpoint = _save_b50_habit_chunking_source(tmp_path)
    b51_config = build_b51_dopaminergic_habit_modulation_config(
        B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
        source_checkpoint=b50_checkpoint,
    )
    b51_source = SpiderBrain(seed=152, module_dropout=0.0, config=b51_config)
    return b51_source.save(tmp_path / "b51")


def _save_b52_cholinergic_precision_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b51_checkpoint = _save_b51_dopaminergic_habit_source(tmp_path)
    b52_config = build_b52_cholinergic_precision_gate_config(
        B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
        source_checkpoint=b51_checkpoint,
    )
    b52_source = SpiderBrain(seed=153, module_dropout=0.0, config=b52_config)
    return b52_source.save(tmp_path / "b52")


def _save_b53_noradrenergic_arousal_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b52_checkpoint = _save_b52_cholinergic_precision_source(tmp_path)
    b53_config = build_b53_noradrenergic_arousal_gain_config(
        B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
        source_checkpoint=b52_checkpoint,
    )
    b53_source = SpiderBrain(seed=154, module_dropout=0.0, config=b53_config)
    return b53_source.save(tmp_path / "b53")


def _save_b54_serotonergic_patience_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b53_checkpoint = _save_b53_noradrenergic_arousal_source(tmp_path)
    b54_config = build_b54_serotonergic_patience_gate_config(
        B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
        source_checkpoint=b53_checkpoint,
    )
    b54_source = SpiderBrain(seed=155, module_dropout=0.0, config=b54_config)
    return b54_source.save(tmp_path / "b54")


def _save_b55_hypothalamic_drive_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b54_checkpoint = _save_b54_serotonergic_patience_source(tmp_path)
    b55_config = build_b55_hypothalamic_drive_coupling_config(
        B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
        source_checkpoint=b54_checkpoint,
    )
    b55_source = SpiderBrain(seed=156, module_dropout=0.0, config=b55_config)
    return b55_source.save(tmp_path / "b55")


def _save_b56_hpa_stress_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b55_checkpoint = _save_b55_hypothalamic_drive_source(tmp_path)
    b56_config = build_b56_hpa_stress_axis_config(
        B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
        source_checkpoint=b55_checkpoint,
    )
    b56_source = SpiderBrain(seed=157, module_dropout=0.0, config=b56_config)
    return b56_source.save(tmp_path / "b56")


def _save_b57_insular_interoceptive_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b56_checkpoint = _save_b56_hpa_stress_source(tmp_path)
    b57_config = build_b57_insular_interoceptive_awareness_config(
        B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
        source_checkpoint=b56_checkpoint,
    )
    b57_source = SpiderBrain(seed=158, module_dropout=0.0, config=b57_config)
    return b57_source.save(tmp_path / "b57")


def _save_b58_acc_conflict_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b57_checkpoint = _save_b57_insular_interoceptive_source(tmp_path)
    b58_config = build_b58_acc_conflict_monitor_config(
        B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
        source_checkpoint=b57_checkpoint,
    )
    b58_source = SpiderBrain(seed=159, module_dropout=0.0, config=b58_config)
    return b58_source.save(tmp_path / "b58")


def _save_b59_prefrontal_goal_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b58_checkpoint = _save_b58_acc_conflict_source(tmp_path)
    b59_config = build_b59_prefrontal_goal_context_config(
        B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
        source_checkpoint=b58_checkpoint,
    )
    b59_source = SpiderBrain(seed=160, module_dropout=0.0, config=b59_config)
    return b59_source.save(tmp_path / "b59")


def _save_b60_orbitofrontal_value_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b59_checkpoint = _save_b59_prefrontal_goal_source(tmp_path)
    b60_config = build_b60_orbitofrontal_outcome_value_config(
        B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
        source_checkpoint=b59_checkpoint,
    )
    b60_source = SpiderBrain(seed=161, module_dropout=0.0, config=b60_config)
    return b60_source.save(tmp_path / "b60")


def _save_b61_amygdala_safety_source(tmpdir: str | Path) -> Path:
    tmp_path = Path(tmpdir)
    b60_checkpoint = _save_b60_orbitofrontal_value_source(tmp_path)
    b61_config = build_b61_amygdala_safety_value_config(
        B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
        source_checkpoint=b60_checkpoint,
    )
    b61_source = SpiderBrain(seed=162, module_dropout=0.0, config=b61_config)
    return b61_source.save(tmp_path / "b61")


__all__ = [name for name in globals() if not name.startswith("__")]
