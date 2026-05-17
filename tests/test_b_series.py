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


class BSeriesActionSpaceTest(unittest.TestCase):
    def test_current_world_still_exposes_only_nine_primitive_actions(self) -> None:
        self.assertEqual(len(ACTIONS), 9)
        self.assertEqual(tuple(ACTIONS), tuple(ACTION_TO_INDEX.keys()))
        for semantic_action in ("MOVE_TO_FOOD", "MOVE_TO_SHELTER", "EXPLORE", "EAT", "SLEEP"):
            self.assertNotIn(semantic_action, ACTIONS)

    def test_legacy_harness_exposes_exact_six_semantic_actions(self) -> None:
        self.assertEqual(tuple(LEGACY_B0_ACTIONS), tuple(B_SEMANTIC_ACTIONS))
        self.assertEqual(len(LEGACY_B0_ACTIONS), 6)

    def test_diagnostic_catalog_registers_b0_variants(self) -> None:
        configs = resolve_ablation_configs(
            ["b0_legacy_semantic_policy", "b0_current_bridge_policy"]
        )
        self.assertEqual(configs[0].architecture, "b_series")
        self.assertEqual(configs[0].b_mode, "legacy_semantic")
        self.assertEqual(configs[1].architecture, "b_series")
        self.assertEqual(configs[1].b_mode, "current_bridge")

    def test_diagnostic_catalog_registers_b1_evolution_variants(self) -> None:
        h48, h64, threat_guard = resolve_ablation_configs(
            [
                B1_CAPACITY_H48_POLICY_NAME,
                B1_CAPACITY_H64_POLICY_NAME,
                B1_THREAT_GUARD_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 1)
        self.assertEqual(h48.b_parent_level, 0)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(
            h48.b_transfer_source_checkpoint,
            B0_CURRENT_BRIDGE_DEFAULT_CHECKPOINT,
        )
        self.assertFalse(h48.b_transfer_allow_low_coverage)
        self.assertEqual(h64.b_hidden_dim, 64)
        self.assertEqual(threat_guard.b_hidden_dim, 48)
        self.assertEqual(
            threat_guard.b_transfer_source_checkpoint,
            h48.b_transfer_source_checkpoint,
        )
        self.assertFalse(threat_guard.enable_reflexes)

    def test_diagnostic_catalog_registers_b2_temporal_threat_variants(self) -> None:
        h48, h56, h64 = resolve_ablation_configs(
            [
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                B2_TEMPORAL_THREAT_H56_POLICY_NAME,
                B2_TEMPORAL_THREAT_H64_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 2)
        self.assertEqual(h48.b_parent_level, 1)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h64.b_hidden_dim, 64)
        self.assertEqual(
            h48.b_transfer_source_checkpoint,
            B1_THREAT_GUARD_DEFAULT_CHECKPOINT,
        )
        self.assertFalse(h48.b_transfer_allow_low_coverage)

    def test_diagnostic_catalog_registers_b3_contact_memory_variants(self) -> None:
        h48, strict_h48, h56, recurrent = resolve_ablation_configs(
            [
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME,
                B3_CONTACT_MEMORY_H56_POLICY_NAME,
                B3_RECURRENT_GUARD_H48_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 3)
        self.assertEqual(h48.b_parent_level, 2)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(strict_h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(recurrent.b_hidden_dim, 48)
        self.assertEqual(recurrent.b_parent_level, 2)
        self.assertEqual(
            h48.b_transfer_source_checkpoint,
            B2_TEMPORAL_THREAT_DEFAULT_CHECKPOINT,
        )

    def test_diagnostic_catalog_registers_b4_recovery_variants(self) -> None:
        h48, exit_h48, h56, genetic = resolve_ablation_configs(
            [
                B4_RECOVERY_BALANCE_H48_POLICY_NAME,
                B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME,
                B4_RECOVERY_BALANCE_H56_POLICY_NAME,
                B4_GENETIC_RECOVERY_H48_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 4)
        self.assertEqual(h48.b_parent_level, 3)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(exit_h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(genetic.b_hidden_dim, 48)
        self.assertEqual(h48.b_controller_profile, "recovery_balance")
        self.assertEqual(exit_h48.b_controller_profile, "predator_exit_memory")
        self.assertEqual(genetic.b_controller_profile, "genetic_recovery")
        tuned = replace(
            h48,
            b_controller_params={"sleep_hunger_max": 0.70},
        )
        roundtrip = BrainAblationConfig.from_summary(tuned.to_summary())
        self.assertEqual(roundtrip.b_controller_profile, "recovery_balance")
        self.assertEqual(roundtrip.b_controller_params["sleep_hunger_max"], 0.70)
        self.assertFalse(h48.b_transfer_allow_low_coverage)

    def test_diagnostic_catalog_registers_b5_homeostatic_variants(self) -> None:
        h48, circadian, h56, genetic = resolve_ablation_configs(
            [
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME,
                B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME,
                B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
            ]
        )

        self.assertEqual(h48.architecture, "b_series")
        self.assertEqual(h48.b_level, 5)
        self.assertEqual(h48.b_parent_level, 4)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(circadian.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(genetic.b_hidden_dim, 48)
        self.assertEqual(h48.b_controller_profile, "homeostatic_arbiter")
        self.assertEqual(circadian.b_controller_profile, "circadian_recovery")
        self.assertEqual(genetic.b_controller_profile, "genetic_homeostasis")
        self.assertFalse(h48.b_transfer_allow_low_coverage)

    def test_diagnostic_catalog_registers_b6_risk_and_recurrent_variants(self) -> None:
        configs = resolve_ablation_configs(
            [
                B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
                B6_CORRIDOR_SURVIVAL_GUARD_H48_POLICY_NAME,
                B6_THREAT_PRIORITY_MEMORY_H48_POLICY_NAME,
                B6_RISK_CORRIDOR_H56_POLICY_NAME,
                B6_GENETIC_RISK_CORRIDOR_H48_POLICY_NAME,
                B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
                B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME,
                B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME,
                B6_RECURRENT_CONTEXT_H56_POLICY_NAME,
                B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME,
                B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
            ]
        )

        for config in configs:
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 6)
            self.assertEqual(config.b_parent_level, 5)
            self.assertEqual(
                config.b_transfer_source_checkpoint,
                B5_GENETIC_HOMEOSTASIS_DEFAULT_CHECKPOINT,
            )
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b6_family", config.b_controller_params)
        self.assertEqual(configs[0].b_hidden_dim, 48)
        self.assertEqual(configs[3].b_hidden_dim, 56)
        self.assertEqual(configs[8].b_hidden_dim, 56)
        self.assertEqual(configs[-1].b_controller_profile, "fused_risk_recurrent")

    def test_diagnostic_catalog_registers_b7_affordance_budget_variants(self) -> None:
        h48, energy, recurrent, h56, genetic = resolve_ablation_configs(
            [
                B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
                B7_ENERGY_BUDGET_CORRIDOR_H48_POLICY_NAME,
                B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME,
                B7_AFFORDANCE_BUDGET_H56_POLICY_NAME,
                B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME,
            ]
        )

        for config in (h48, energy, recurrent, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 7)
            self.assertEqual(config.b_parent_level, 6)
            self.assertEqual(
                config.b_transfer_source_checkpoint,
                B6_FUSED_RISK_RECURRENT_DEFAULT_CHECKPOINT,
            )
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b7_budget_step_cost", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "affordance_budget")
        self.assertEqual(energy.b_controller_profile, "energy_budget_corridor")
        self.assertEqual(recurrent.b_controller_profile, "recurrent_affordance")
        self.assertEqual(genetic.b_controller_profile, "genetic_affordance_budget")

    def test_diagnostic_catalog_registers_b8_spatial_affordance_variants(self) -> None:
        h48, return_vector, place_memory, h56, genetic = resolve_ablation_configs(
            [
                B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
                B8_RETURN_VECTOR_H48_POLICY_NAME,
                B8_CORRIDOR_PLACE_MEMORY_H48_POLICY_NAME,
                B8_SPATIAL_AFFORDANCE_MAP_H56_POLICY_NAME,
                B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, return_vector, place_memory, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 8)
            self.assertEqual(config.b_parent_level, 7)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b8_place_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "spatial_affordance_map")
        self.assertEqual(return_vector.b_controller_profile, "return_vector")
        self.assertEqual(place_memory.b_controller_profile, "corridor_place_memory")
        self.assertEqual(genetic.b_controller_profile, "genetic_spatial_affordance")

    def test_diagnostic_catalog_registers_b9_waypoint_variants(self) -> None:
        h48, path, route, h56, genetic = resolve_ablation_configs(
            [
                B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
                B9_PATH_INTEGRATION_H48_POLICY_NAME,
                B9_ROUTE_MEMORY_H48_POLICY_NAME,
                B9_WAYPOINT_PLANNER_H56_POLICY_NAME,
                B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME,
            ]
        )

        for config in (h48, path, route, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 9)
            self.assertEqual(config.b_parent_level, 8)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b9_route_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "waypoint_planner")
        self.assertEqual(path.b_controller_profile, "path_integration")
        self.assertEqual(route.b_controller_profile, "route_memory")
        self.assertEqual(genetic.b_controller_profile, "genetic_waypoint_planner")

    def test_diagnostic_catalog_registers_b10_replay_variants(self) -> None:
        h48, value_route, replay, h56, genetic = resolve_ablation_configs(
            [
                B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
                B10_VALUE_ROUTE_EVALUATOR_H48_POLICY_NAME,
                B10_REPLAY_PLANNER_H48_POLICY_NAME,
                B10_PROSPECTIVE_REPLAY_H56_POLICY_NAME,
                B10_GENETIC_REPLAY_PLANNER_H48_POLICY_NAME,
            ]
        )

        for config in (h48, value_route, replay, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 10)
            self.assertEqual(config.b_parent_level, 9)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b10_replay_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "prospective_replay")
        self.assertEqual(value_route.b_controller_profile, "value_route_evaluator")
        self.assertEqual(replay.b_controller_profile, "replay_planner")
        self.assertEqual(genetic.b_controller_profile, "genetic_replay_planner")

    def test_diagnostic_catalog_registers_b11_confidence_variants(self) -> None:
        h48, uncertainty, neuromod, h56, genetic = resolve_ablation_configs(
            [
                B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
                B11_UNCERTAINTY_GATE_H48_POLICY_NAME,
                B11_NEUROMODULATED_REPLAY_H48_POLICY_NAME,
                B11_CONFIDENCE_ARBITER_H56_POLICY_NAME,
                B11_GENETIC_CONFIDENCE_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, uncertainty, neuromod, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 11)
            self.assertEqual(config.b_parent_level, 10)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b11_confidence_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "confidence_arbiter")
        self.assertEqual(uncertainty.b_controller_profile, "uncertainty_gate")
        self.assertEqual(neuromod.b_controller_profile, "neuromodulated_replay")
        self.assertEqual(genetic.b_controller_profile, "genetic_confidence_gate")

    def test_diagnostic_catalog_registers_b12_attention_variants(self) -> None:
        h48, active, affordance, h56, genetic = resolve_ablation_configs(
            [
                B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
                B12_ACTIVE_INFERENCE_GATE_H48_POLICY_NAME,
                B12_AFFORDANCE_ATTENTION_H48_POLICY_NAME,
                B12_PREDICTIVE_ATTENTION_H56_POLICY_NAME,
                B12_GENETIC_ATTENTION_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, active, affordance, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 12)
            self.assertEqual(config.b_parent_level, 11)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b12_attention_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "predictive_attention")
        self.assertEqual(active.b_controller_profile, "active_inference_gate")
        self.assertEqual(affordance.b_controller_profile, "affordance_attention")
        self.assertEqual(genetic.b_controller_profile, "genetic_attention_gate")

    def test_diagnostic_catalog_registers_b13_local_search_variants(self) -> None:
        h48, counterfactual, sampler, h56, genetic = resolve_ablation_configs(
            [
                B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
                B13_COUNTERFACTUAL_ROUTE_H48_POLICY_NAME,
                B13_AFFORDANCE_SAMPLER_H48_POLICY_NAME,
                B13_LOCAL_AFFORDANCE_SEARCH_H56_POLICY_NAME,
                B13_GENETIC_LOCAL_SEARCH_H48_POLICY_NAME,
            ]
        )

        for config in (h48, counterfactual, sampler, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 13)
            self.assertEqual(config.b_parent_level, 12)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b13_search_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "local_affordance_search")
        self.assertEqual(counterfactual.b_controller_profile, "counterfactual_route")
        self.assertEqual(sampler.b_controller_profile, "affordance_sampler")
        self.assertEqual(genetic.b_controller_profile, "genetic_local_search")

    def test_diagnostic_catalog_registers_b14_uncertainty_variants(self) -> None:
        h48, risk, confidence, h56, genetic = resolve_ablation_configs(
            [
                B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
                B14_RISK_CALIBRATED_SEARCH_H48_POLICY_NAME,
                B14_CONFIDENCE_WEIGHTED_ROUTE_H48_POLICY_NAME,
                B14_AFFORDANCE_UNCERTAINTY_H56_POLICY_NAME,
                B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME,
            ]
        )

        for config in (h48, risk, confidence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 14)
            self.assertEqual(config.b_parent_level, 13)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b14_uncertainty_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "affordance_uncertainty")
        self.assertEqual(risk.b_controller_profile, "risk_calibrated_search")
        self.assertEqual(confidence.b_controller_profile, "confidence_weighted_route")
        self.assertEqual(genetic.b_controller_profile, "genetic_uncertainty_search")

    def test_diagnostic_catalog_registers_b15_option_variants(self) -> None:
        h48, persistence, value_gated, h56, genetic = resolve_ablation_configs(
            [
                B15_OPTION_CRITIC_H48_POLICY_NAME,
                B15_PERSISTENCE_GATE_H48_POLICY_NAME,
                B15_VALUE_GATED_OPTION_H48_POLICY_NAME,
                B15_OPTION_CRITIC_H56_POLICY_NAME,
                B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME,
            ]
        )

        for config in (h48, persistence, value_gated, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 15)
            self.assertEqual(config.b_parent_level, 14)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b15_option_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "option_critic")
        self.assertEqual(persistence.b_controller_profile, "persistence_gate")
        self.assertEqual(value_gated.b_controller_profile, "value_gated_option")
        self.assertEqual(genetic.b_controller_profile, "genetic_option_critic")

    def test_diagnostic_catalog_registers_b16_ensemble_variants(self) -> None:
        h48, competing, voter, h56, genetic = resolve_ablation_configs(
            [
                B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
                B16_COMPETING_OPTIONS_H48_POLICY_NAME,
                B16_ACTION_SET_VOTER_H48_POLICY_NAME,
                B16_OPTION_ENSEMBLE_H56_POLICY_NAME,
                B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, competing, voter, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 16)
            self.assertEqual(config.b_parent_level, 15)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b16_ensemble_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "option_ensemble")
        self.assertEqual(competing.b_controller_profile, "competing_options")
        self.assertEqual(voter.b_controller_profile, "action_set_voter")
        self.assertEqual(genetic.b_controller_profile, "genetic_option_ensemble")

    def test_diagnostic_catalog_registers_b17_neuromodulated_variants(self) -> None:
        h48, arousal, homeostasis, h56, genetic = resolve_ablation_configs(
            [
                B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
                B17_AROUSAL_GATED_OPTIONS_H48_POLICY_NAME,
                B17_HOMEOSTATIC_MODULATOR_H48_POLICY_NAME,
                B17_NEUROMODULATED_ENSEMBLE_H56_POLICY_NAME,
                B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, arousal, homeostasis, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 17)
            self.assertEqual(config.b_parent_level, 16)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b17_arousal_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "neuromodulated_ensemble")
        self.assertEqual(arousal.b_controller_profile, "arousal_gated_options")
        self.assertEqual(homeostasis.b_controller_profile, "homeostatic_modulator")
        self.assertEqual(genetic.b_controller_profile, "genetic_neuromodulated_ensemble")

    def test_diagnostic_catalog_registers_b18_eligibility_variants(self) -> None:
        h48, metastable, synaptic, h56, genetic = resolve_ablation_configs(
            [
                B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
                B18_METASTABLE_AROUSAL_H48_POLICY_NAME,
                B18_SYNAPTIC_TRACE_MODULATOR_H48_POLICY_NAME,
                B18_ELIGIBILITY_TRACE_H56_POLICY_NAME,
                B18_GENETIC_ELIGIBILITY_TRACE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, metastable, synaptic, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 18)
            self.assertEqual(config.b_parent_level, 17)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b18_trace_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "eligibility_trace")
        self.assertEqual(metastable.b_controller_profile, "metastable_arousal")
        self.assertEqual(synaptic.b_controller_profile, "synaptic_trace_modulator")
        self.assertEqual(genetic.b_controller_profile, "genetic_eligibility_trace")

    def test_diagnostic_catalog_registers_b19_meta_memory_variants(self) -> None:
        h48, stability, suppression, h56, genetic = resolve_ablation_configs(
            [
                B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
                B19_STABILITY_MEMORY_H48_POLICY_NAME,
                B19_SWITCH_SUPPRESSION_H48_POLICY_NAME,
                B19_EPISODIC_META_MEMORY_H56_POLICY_NAME,
                B19_GENETIC_META_MEMORY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, stability, suppression, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 19)
            self.assertEqual(config.b_parent_level, 18)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b19_memory_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "episodic_meta_memory")
        self.assertEqual(stability.b_controller_profile, "stability_memory")
        self.assertEqual(suppression.b_controller_profile, "switch_suppression")
        self.assertEqual(genetic.b_controller_profile, "genetic_meta_memory")

    def test_diagnostic_catalog_registers_b20_working_memory_variants(self) -> None:
        h48, context, stability, h56, genetic = resolve_ablation_configs(
            [
                B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
                B20_CONTEXT_BINDING_H48_POLICY_NAME,
                B20_STABILITY_BUFFER_H48_POLICY_NAME,
                B20_WORKING_MEMORY_GATE_H56_POLICY_NAME,
                B20_GENETIC_WORKING_MEMORY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, context, stability, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 20)
            self.assertEqual(config.b_parent_level, 19)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b20_buffer_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "working_memory_gate")
        self.assertEqual(context.b_controller_profile, "context_binding")
        self.assertEqual(stability.b_controller_profile, "stability_buffer")
        self.assertEqual(genetic.b_controller_profile, "genetic_working_memory")

    def test_diagnostic_catalog_registers_b21_replay_variants(self) -> None:
        h48, sequence, route, h56, genetic = resolve_ablation_configs(
            [
                B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
                B21_SEQUENCE_BINDING_H48_POLICY_NAME,
                B21_ROUTE_REHEARSAL_H48_POLICY_NAME,
                B21_HIPPOCAMPAL_REPLAY_H56_POLICY_NAME,
                B21_GENETIC_REPLAY_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, sequence, route, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 21)
            self.assertEqual(config.b_parent_level, 20)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b21_replay_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "hippocampal_replay")
        self.assertEqual(sequence.b_controller_profile, "sequence_binding")
        self.assertEqual(route.b_controller_profile, "route_rehearsal")
        self.assertEqual(genetic.b_controller_profile, "genetic_replay_gate")

    def test_diagnostic_catalog_registers_b22_prospective_variants(self) -> None:
        h48, forward, route, h56, genetic = resolve_ablation_configs(
            [
                B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
                B22_FORWARD_MODEL_GATE_H48_POLICY_NAME,
                B22_ROUTE_VIABILITY_SIM_H48_POLICY_NAME,
                B22_PROSPECTIVE_MAP_REPLAY_H56_POLICY_NAME,
                B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, forward, route, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 22)
            self.assertEqual(config.b_parent_level, 21)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b22_sim_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "prospective_map_replay")
        self.assertEqual(forward.b_controller_profile, "forward_model_gate")
        self.assertEqual(route.b_controller_profile, "route_viability_sim")
        self.assertEqual(genetic.b_controller_profile, "genetic_prospective_replay")

    def test_diagnostic_catalog_registers_b23_conflict_variants(self) -> None:
        h48, error, abort, h56, genetic = resolve_ablation_configs(
            [
                B23_CONFLICT_MONITOR_H48_POLICY_NAME,
                B23_ERROR_GATED_REPLAY_H48_POLICY_NAME,
                B23_ABORT_CONFLICT_ARBITER_H48_POLICY_NAME,
                B23_CONFLICT_MONITOR_H56_POLICY_NAME,
                B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME,
            ]
        )

        for config in (h48, error, abort, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 23)
            self.assertEqual(config.b_parent_level, 22)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b23_conflict_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "conflict_monitor")
        self.assertEqual(error.b_controller_profile, "error_gated_replay")
        self.assertEqual(abort.b_controller_profile, "abort_conflict_arbiter")
        self.assertEqual(genetic.b_controller_profile, "genetic_conflict_monitor")

    def test_diagnostic_catalog_registers_b24_precision_variants(self) -> None:
        h48, precision, abort, h56, genetic = resolve_ablation_configs(
            [
                B24_PRECISION_CONFLICT_H48_POLICY_NAME,
                B24_PREDICTION_PRECISION_GATE_H48_POLICY_NAME,
                B24_RELIABILITY_ABORT_H48_POLICY_NAME,
                B24_PRECISION_CONFLICT_H56_POLICY_NAME,
                B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME,
            ]
        )

        for config in (h48, precision, abort, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 24)
            self.assertEqual(config.b_parent_level, 23)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b24_precision_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "precision_conflict")
        self.assertEqual(precision.b_controller_profile, "prediction_precision_gate")
        self.assertEqual(abort.b_controller_profile, "reliability_abort")
        self.assertEqual(genetic.b_controller_profile, "genetic_precision_conflict")

    def test_diagnostic_catalog_registers_b25_metacognitive_variants(self) -> None:
        h48, calibration, integrator, h56, genetic = resolve_ablation_configs(
            [
                B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
                B25_CONFIDENCE_CALIBRATION_H48_POLICY_NAME,
                B25_UNCERTAINTY_INTEGRATOR_H48_POLICY_NAME,
                B25_METACOGNITIVE_CONFIDENCE_H56_POLICY_NAME,
                B25_GENETIC_METACOGNITION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, calibration, integrator, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 25)
            self.assertEqual(config.b_parent_level, 24)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b25_confidence_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "metacognitive_confidence")
        self.assertEqual(calibration.b_controller_profile, "confidence_calibration")
        self.assertEqual(integrator.b_controller_profile, "uncertainty_integrator")
        self.assertEqual(genetic.b_controller_profile, "genetic_metacognition")

    def test_diagnostic_catalog_registers_b26_allostatic_variants(self) -> None:
        h48, drift, suppression, h56, genetic = resolve_ablation_configs(
            [
                B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
                B26_SETPOINT_DRIFT_H48_POLICY_NAME,
                B26_ERROR_SUPPRESSION_H48_POLICY_NAME,
                B26_ALLOSTATIC_PREDICTION_H56_POLICY_NAME,
                B26_GENETIC_ALLOSTASIS_H48_POLICY_NAME,
            ]
        )

        for config in (h48, drift, suppression, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 26)
            self.assertEqual(config.b_parent_level, 25)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b26_error_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "allostatic_prediction")
        self.assertEqual(drift.b_controller_profile, "setpoint_drift")
        self.assertEqual(suppression.b_controller_profile, "error_suppression")
        self.assertEqual(genetic.b_controller_profile, "genetic_allostasis")

    def test_diagnostic_catalog_registers_b27_arousal_variants(self) -> None:
        h48, stress, energy, h56, genetic = resolve_ablation_configs(
            [
                B27_AROUSAL_GAIN_H48_POLICY_NAME,
                B27_STRESS_MODULATION_H48_POLICY_NAME,
                B27_ENERGY_AROUSAL_H48_POLICY_NAME,
                B27_AROUSAL_GAIN_H56_POLICY_NAME,
                B27_GENETIC_AROUSAL_H48_POLICY_NAME,
            ]
        )

        for config in (h48, stress, energy, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 27)
            self.assertEqual(config.b_parent_level, 26)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b27_arousal_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "arousal_gain")
        self.assertEqual(stress.b_controller_profile, "stress_modulation")
        self.assertEqual(energy.b_controller_profile, "energy_arousal")
        self.assertEqual(genetic.b_controller_profile, "genetic_arousal")

    def test_diagnostic_catalog_registers_b28_attention_variants(self) -> None:
        h48, threat, homeostatic, h56, genetic = resolve_ablation_configs(
            [
                B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                B28_THREAT_FOCUS_ATTENTION_H48_POLICY_NAME,
                B28_HOMEOSTATIC_ATTENTION_H48_POLICY_NAME,
                B28_INTEROCEPTIVE_ATTENTION_H56_POLICY_NAME,
                B28_GENETIC_ATTENTION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, threat, homeostatic, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 28)
            self.assertEqual(config.b_parent_level, 27)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b28_attention_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "interoceptive_attention")
        self.assertEqual(threat.b_controller_profile, "threat_focus_attention")
        self.assertEqual(homeostatic.b_controller_profile, "homeostatic_attention")
        self.assertEqual(genetic.b_controller_profile, "genetic_attention")

    def test_diagnostic_catalog_registers_b29_salience_variants(self) -> None:
        h48, threat, homeostatic, h56, genetic = resolve_ablation_configs(
            [
                B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
                B29_THREAT_SALIENCE_GATE_H48_POLICY_NAME,
                B29_HOMEOSTATIC_SALIENCE_GATE_H48_POLICY_NAME,
                B29_SALIENCE_COMPETITION_H56_POLICY_NAME,
                B29_GENETIC_SALIENCE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, threat, homeostatic, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 29)
            self.assertEqual(config.b_parent_level, 28)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b29_salience_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "salience_competition")
        self.assertEqual(threat.b_controller_profile, "threat_salience_gate")
        self.assertEqual(homeostatic.b_controller_profile, "homeostatic_salience_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_salience")

    def test_diagnostic_catalog_registers_b30_gate_variants(self) -> None:
        h48, balance, inhibition, h56, genetic = resolve_ablation_configs(
            [
                B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
                B30_GO_NOGO_BALANCE_H48_POLICY_NAME,
                B30_THREAT_INHIBITION_GATE_H48_POLICY_NAME,
                B30_BASAL_GANGLIA_GATE_H56_POLICY_NAME,
                B30_GENETIC_ACTION_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, balance, inhibition, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 30)
            self.assertEqual(config.b_parent_level, 29)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b30_gate_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "basal_ganglia_gate")
        self.assertEqual(balance.b_controller_profile, "go_nogo_balance")
        self.assertEqual(inhibition.b_controller_profile, "threat_inhibition_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_action_gate")

    def test_diagnostic_catalog_registers_b31_dopamine_variants(self) -> None:
        h48, tonic, phasic, h56, genetic = resolve_ablation_configs(
            [
                B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
                B31_TONIC_DOPAMINE_GATE_H48_POLICY_NAME,
                B31_PHASIC_DOPAMINE_GATE_H48_POLICY_NAME,
                B31_DOPAMINE_PREDICTION_ERROR_H56_POLICY_NAME,
                B31_GENETIC_DOPAMINE_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, tonic, phasic, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 31)
            self.assertEqual(config.b_parent_level, 30)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b31_dopamine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "dopamine_prediction_error")
        self.assertEqual(tonic.b_controller_profile, "tonic_dopamine_gate")
        self.assertEqual(phasic.b_controller_profile, "phasic_dopamine_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_dopamine_gate")

    def test_diagnostic_catalog_registers_b32_actor_critic_variants(self) -> None:
        h48, advantage, stability, h56, genetic = resolve_ablation_configs(
            [
                B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
                B32_ADVANTAGE_VALUE_GATE_H48_POLICY_NAME,
                B32_CRITIC_STABILITY_H48_POLICY_NAME,
                B32_ACTOR_CRITIC_VALUE_H56_POLICY_NAME,
                B32_GENETIC_ACTOR_CRITIC_H48_POLICY_NAME,
            ]
        )

        for config in (h48, advantage, stability, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 32)
            self.assertEqual(config.b_parent_level, 31)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b32_value_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "actor_critic_value")
        self.assertEqual(advantage.b_controller_profile, "advantage_value_gate")
        self.assertEqual(stability.b_controller_profile, "critic_stability")
        self.assertEqual(genetic.b_controller_profile, "genetic_actor_critic")

    def test_diagnostic_catalog_registers_b33_td_error_variants(self) -> None:
        h48, bootstrap, reward_trace, h56, genetic = resolve_ablation_configs(
            [
                B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
                B33_BOOTSTRAPPED_VALUE_GATE_H48_POLICY_NAME,
                B33_REWARD_TRACE_CRITIC_H48_POLICY_NAME,
                B33_TD_ERROR_DECOMPOSITION_H56_POLICY_NAME,
                B33_GENETIC_TD_VALUE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, bootstrap, reward_trace, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 33)
            self.assertEqual(config.b_parent_level, 32)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b33_td_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "td_error_decomposition")
        self.assertEqual(bootstrap.b_controller_profile, "bootstrapped_value_gate")
        self.assertEqual(reward_trace.b_controller_profile, "reward_trace_critic")
        self.assertEqual(genetic.b_controller_profile, "genetic_td_value")

    def test_diagnostic_catalog_registers_b34_eligibility_variants(self) -> None:
        h48, delayed, tagging, h56, genetic = resolve_ablation_configs(
            [
                B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
                B34_DELAYED_CREDIT_GATE_H48_POLICY_NAME,
                B34_SYNAPTIC_TAGGING_H48_POLICY_NAME,
                B34_ELIGIBILITY_CREDIT_H56_POLICY_NAME,
                B34_GENETIC_ELIGIBILITY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, delayed, tagging, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 34)
            self.assertEqual(config.b_parent_level, 33)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b34_eligibility_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "eligibility_credit")
        self.assertEqual(delayed.b_controller_profile, "delayed_credit_gate")
        self.assertEqual(tagging.b_controller_profile, "synaptic_tagging")
        self.assertEqual(genetic.b_controller_profile, "genetic_eligibility")

    def test_diagnostic_catalog_registers_b35_forward_model_variants(self) -> None:
        h48, transition, confidence, h56, genetic = resolve_ablation_configs(
            [
                B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
                B35_TRANSITION_ERROR_GATE_H48_POLICY_NAME,
                B35_MODEL_CONFIDENCE_H48_POLICY_NAME,
                B35_FORWARD_MODEL_VALUE_H56_POLICY_NAME,
                B35_GENETIC_FORWARD_MODEL_H48_POLICY_NAME,
            ]
        )

        for config in (h48, transition, confidence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 35)
            self.assertEqual(config.b_parent_level, 34)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b35_model_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "forward_model_value")
        self.assertEqual(transition.b_controller_profile, "transition_error_gate")
        self.assertEqual(confidence.b_controller_profile, "model_confidence")
        self.assertEqual(genetic.b_controller_profile, "genetic_forward_model")

    def test_diagnostic_catalog_registers_b36_belief_state_variants(self) -> None:
        h48, belief_error, context, h56, genetic = resolve_ablation_configs(
            [
                B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
                B36_BELIEF_ERROR_GATE_H48_POLICY_NAME,
                B36_CONTEXT_INFERENCE_H48_POLICY_NAME,
                B36_LATENT_BELIEF_STATE_H56_POLICY_NAME,
                B36_GENETIC_BELIEF_STATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, belief_error, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 36)
            self.assertEqual(config.b_parent_level, 35)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b36_belief_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "latent_belief_state")
        self.assertEqual(belief_error.b_controller_profile, "belief_error_gate")
        self.assertEqual(context.b_controller_profile, "context_inference")
        self.assertEqual(genetic.b_controller_profile, "genetic_belief_state")

    def test_diagnostic_catalog_registers_b37_state_factor_variants(self) -> None:
        h48, intero_extero, confidence, h56, genetic = resolve_ablation_configs(
            [
                B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
                B37_INTERO_EXTERO_FACTOR_H48_POLICY_NAME,
                B37_FACTOR_CONFIDENCE_H48_POLICY_NAME,
                B37_STATE_FACTOR_GATE_H56_POLICY_NAME,
                B37_GENETIC_STATE_FACTOR_H48_POLICY_NAME,
            ]
        )

        for config in (h48, intero_extero, confidence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 37)
            self.assertEqual(config.b_parent_level, 36)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b37_factor_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "state_factor_gate")
        self.assertEqual(intero_extero.b_controller_profile, "intero_extero_factor")
        self.assertEqual(confidence.b_controller_profile, "factor_confidence")
        self.assertEqual(genetic.b_controller_profile, "genetic_state_factor")

    def test_diagnostic_catalog_registers_b38_factor_attention_variants(self) -> None:
        h48, interoceptive, confidence, h56, genetic = resolve_ablation_configs(
            [
                B38_FACTOR_ATTENTION_H48_POLICY_NAME,
                B38_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                B38_CONFIDENCE_ATTENTION_H48_POLICY_NAME,
                B38_FACTOR_ATTENTION_H56_POLICY_NAME,
                B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, interoceptive, confidence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 38)
            self.assertEqual(config.b_parent_level, 37)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b38_attention_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "factor_attention")
        self.assertEqual(interoceptive.b_controller_profile, "interoceptive_attention")
        self.assertEqual(confidence.b_controller_profile, "confidence_attention")
        self.assertEqual(genetic.b_controller_profile, "genetic_factor_attention")

    def test_diagnostic_catalog_registers_b39_attention_binding_variants(self) -> None:
        h48, cross_factor, context, h56, genetic = resolve_ablation_configs(
            [
                B39_ATTENTION_BINDING_H48_POLICY_NAME,
                B39_CROSS_FACTOR_BINDING_H48_POLICY_NAME,
                B39_CONTEXT_BINDING_ATTENTION_H48_POLICY_NAME,
                B39_ATTENTION_BINDING_H56_POLICY_NAME,
                B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME,
            ]
        )

        for config in (h48, cross_factor, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 39)
            self.assertEqual(config.b_parent_level, 38)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b39_binding_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "attention_binding")
        self.assertEqual(cross_factor.b_controller_profile, "cross_factor_binding")
        self.assertEqual(context.b_controller_profile, "context_binding_attention")
        self.assertEqual(genetic.b_controller_profile, "genetic_attention_binding")

    def test_diagnostic_catalog_registers_b40_global_workspace_variants(self) -> None:
        h48, sensory, context, h56, genetic = resolve_ablation_configs(
            [
                B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
                B40_SENSORY_WORKSPACE_H48_POLICY_NAME,
                B40_CONTEXT_WORKSPACE_H48_POLICY_NAME,
                B40_GLOBAL_WORKSPACE_H56_POLICY_NAME,
                B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, sensory, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 40)
            self.assertEqual(config.b_parent_level, 39)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b40_workspace_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "global_workspace")
        self.assertEqual(sensory.b_controller_profile, "sensory_workspace")
        self.assertEqual(context.b_controller_profile, "context_workspace")
        self.assertEqual(genetic.b_controller_profile, "genetic_global_workspace")

    def test_diagnostic_catalog_registers_b41_executive_workspace_variants(self) -> None:
        h48, inhibitory, selector, h56, genetic = resolve_ablation_configs(
            [
                B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
                B41_INHIBITORY_CONTROL_H48_POLICY_NAME,
                B41_GOAL_CONTEXT_SELECTOR_H48_POLICY_NAME,
                B41_EXECUTIVE_WORKSPACE_H56_POLICY_NAME,
                B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, inhibitory, selector, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 41)
            self.assertEqual(config.b_parent_level, 40)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b41_executive_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "executive_workspace")
        self.assertEqual(inhibitory.b_controller_profile, "inhibitory_control")
        self.assertEqual(selector.b_controller_profile, "goal_context_selector")
        self.assertEqual(genetic.b_controller_profile, "genetic_executive_workspace")

    def test_diagnostic_catalog_registers_b42_error_monitor_variants(self) -> None:
        h48, conflict, performance, h56, genetic = resolve_ablation_configs(
            [
                B42_ERROR_MONITOR_H48_POLICY_NAME,
                B42_CONFLICT_MONITOR_H48_POLICY_NAME,
                B42_PERFORMANCE_MONITOR_H48_POLICY_NAME,
                B42_ERROR_MONITOR_H56_POLICY_NAME,
                B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME,
            ]
        )

        for config in (h48, conflict, performance, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 42)
            self.assertEqual(config.b_parent_level, 41)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b42_monitor_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "error_monitor")
        self.assertEqual(conflict.b_controller_profile, "conflict_monitor")
        self.assertEqual(performance.b_controller_profile, "performance_monitor")
        self.assertEqual(genetic.b_controller_profile, "genetic_error_monitor")

    def test_diagnostic_catalog_registers_b43_adaptive_precision_variants(self) -> None:
        h48, arousal, threshold, h56, genetic = resolve_ablation_configs(
            [
                B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
                B43_AROUSAL_PRECISION_H48_POLICY_NAME,
                B43_THRESHOLD_ADAPTATION_H48_POLICY_NAME,
                B43_ADAPTIVE_PRECISION_H56_POLICY_NAME,
                B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, arousal, threshold, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 43)
            self.assertEqual(config.b_parent_level, 42)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b43_precision_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "adaptive_precision")
        self.assertEqual(arousal.b_controller_profile, "arousal_precision")
        self.assertEqual(threshold.b_controller_profile, "threshold_adaptation")
        self.assertEqual(genetic.b_controller_profile, "genetic_adaptive_precision")

    def test_diagnostic_catalog_registers_b44_thalamic_relay_variants(self) -> None:
        h48, sensory, context, h56, genetic = resolve_ablation_configs(
            [
                B44_THALAMIC_RELAY_H48_POLICY_NAME,
                B44_SENSORY_RELAY_H48_POLICY_NAME,
                B44_CONTEXT_RELAY_H48_POLICY_NAME,
                B44_THALAMIC_RELAY_H56_POLICY_NAME,
                B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, sensory, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 44)
            self.assertEqual(config.b_parent_level, 43)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b44_relay_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "thalamic_relay")
        self.assertEqual(sensory.b_controller_profile, "sensory_relay")
        self.assertEqual(context.b_controller_profile, "context_relay")
        self.assertEqual(genetic.b_controller_profile, "genetic_thalamic_relay")

    def test_diagnostic_catalog_registers_b45_reticular_inhibition_variants(self) -> None:
        h48, sensory, context, h56, genetic = resolve_ablation_configs(
            [
                B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
                B45_SENSORY_INHIBITION_H48_POLICY_NAME,
                B45_CONTEXT_INHIBITION_H48_POLICY_NAME,
                B45_RETICULAR_INHIBITION_H56_POLICY_NAME,
                B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, sensory, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 45)
            self.assertEqual(config.b_parent_level, 44)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b45_inhibition_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "reticular_inhibition")
        self.assertEqual(sensory.b_controller_profile, "sensory_inhibition")
        self.assertEqual(context.b_controller_profile, "context_inhibition")
        self.assertEqual(genetic.b_controller_profile, "genetic_reticular_inhibition")

    def test_diagnostic_catalog_registers_b46_corticothalamic_feedback_variants(self) -> None:
        h48, feedback, context, h56, genetic = resolve_ablation_configs(
            [
                B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
                B46_FEEDBACK_GAIN_H48_POLICY_NAME,
                B46_CONTEXT_FEEDBACK_H48_POLICY_NAME,
                B46_CORTICOTHALAMIC_FEEDBACK_H56_POLICY_NAME,
                B46_GENETIC_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
            ]
        )

        for config in (h48, feedback, context, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 46)
            self.assertEqual(config.b_parent_level, 45)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b46_feedback_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "corticothalamic_feedback")
        self.assertEqual(feedback.b_controller_profile, "feedback_gain")
        self.assertEqual(context.b_controller_profile, "context_feedback")
        self.assertEqual(genetic.b_controller_profile, "genetic_corticothalamic_feedback")

    def test_diagnostic_catalog_registers_b47_oscillatory_synchrony_variants(self) -> None:
        h48, phase, coherence, h56, genetic = resolve_ablation_configs(
            [
                B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
                B47_PHASE_LOCKING_H48_POLICY_NAME,
                B47_COHERENCE_GATE_H48_POLICY_NAME,
                B47_OSCILLATORY_SYNCHRONY_H56_POLICY_NAME,
                B47_GENETIC_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, phase, coherence, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 47)
            self.assertEqual(config.b_parent_level, 46)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b47_phase_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "oscillatory_synchrony")
        self.assertEqual(phase.b_controller_profile, "phase_locking")
        self.assertEqual(coherence.b_controller_profile, "coherence_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_oscillatory_synchrony")

    def test_diagnostic_catalog_registers_b48_cerebellar_timing_variants(self) -> None:
        h48, correction, predictive, h56, genetic = resolve_ablation_configs(
            [
                B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
                B48_TIMING_ERROR_CORRECTION_H48_POLICY_NAME,
                B48_PREDICTIVE_TIMING_H48_POLICY_NAME,
                B48_CEREBELLAR_TIMING_H56_POLICY_NAME,
                B48_GENETIC_CEREBELLAR_TIMING_H48_POLICY_NAME,
            ]
        )

        for config in (h48, correction, predictive, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 48)
            self.assertEqual(config.b_parent_level, 47)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b48_timing_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "cerebellar_timing")
        self.assertEqual(correction.b_controller_profile, "timing_error_correction")
        self.assertEqual(predictive.b_controller_profile, "predictive_timing")
        self.assertEqual(genetic.b_controller_profile, "genetic_cerebellar_timing")

    def test_diagnostic_catalog_registers_b49_striatal_action_gate_variants(self) -> None:
        h48, direct, indirect, h56, genetic = resolve_ablation_configs(
            [
                B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
                B49_DIRECT_PATH_FACILITATION_H48_POLICY_NAME,
                B49_INDIRECT_PATH_SUPPRESSION_H48_POLICY_NAME,
                B49_STRIATAL_ACTION_GATE_H56_POLICY_NAME,
                B49_GENETIC_STRIATAL_GATE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, direct, indirect, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 49)
            self.assertEqual(config.b_parent_level, 48)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b49_gate_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "striatal_action_gate")
        self.assertEqual(direct.b_controller_profile, "direct_path_facilitation")
        self.assertEqual(indirect.b_controller_profile, "indirect_path_suppression")
        self.assertEqual(genetic.b_controller_profile, "genetic_striatal_gate")

    def test_diagnostic_catalog_registers_b50_habit_chunking_variants(self) -> None:
        h48, value, stability, h56, genetic = resolve_ablation_configs(
            [
                B50_HABIT_CHUNKING_H48_POLICY_NAME,
                B50_ACTION_CHUNK_VALUE_H48_POLICY_NAME,
                B50_HABIT_STABILITY_H48_POLICY_NAME,
                B50_HABIT_CHUNKING_H56_POLICY_NAME,
                B50_GENETIC_HABIT_CHUNKING_H48_POLICY_NAME,
            ]
        )

        for config in (h48, value, stability, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 50)
            self.assertEqual(config.b_parent_level, 49)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b50_habit_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "habit_chunking")
        self.assertEqual(value.b_controller_profile, "action_chunk_value")
        self.assertEqual(stability.b_controller_profile, "habit_stability")
        self.assertEqual(genetic.b_controller_profile, "genetic_habit_chunking")

    def test_diagnostic_catalog_registers_b51_dopaminergic_habit_variants(self) -> None:
        h48, reward, novelty, h56, genetic = resolve_ablation_configs(
            [
                B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
                B51_REWARD_PREDICTION_GAIN_H48_POLICY_NAME,
                B51_NOVELTY_MODULATED_HABIT_H48_POLICY_NAME,
                B51_DOPAMINERGIC_HABIT_MODULATION_H56_POLICY_NAME,
                B51_GENETIC_DOPAMINE_HABIT_H48_POLICY_NAME,
            ]
        )

        for config in (h48, reward, novelty, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 51)
            self.assertEqual(config.b_parent_level, 50)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b51_dopamine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "dopaminergic_habit_modulation")
        self.assertEqual(reward.b_controller_profile, "reward_prediction_gain")
        self.assertEqual(novelty.b_controller_profile, "novelty_modulated_habit")
        self.assertEqual(genetic.b_controller_profile, "genetic_dopamine_habit")

    def test_diagnostic_catalog_registers_b52_cholinergic_precision_variants(self) -> None:
        h48, attention, uncertainty, h56, genetic = resolve_ablation_configs(
            [
                B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
                B52_ATTENTION_GAIN_H48_POLICY_NAME,
                B52_UNCERTAINTY_RELEASE_H48_POLICY_NAME,
                B52_CHOLINERGIC_PRECISION_GATE_H56_POLICY_NAME,
                B52_GENETIC_CHOLINERGIC_PRECISION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, attention, uncertainty, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 52)
            self.assertEqual(config.b_parent_level, 51)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b52_acetylcholine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "cholinergic_precision_gate")
        self.assertEqual(attention.b_controller_profile, "attention_gain")
        self.assertEqual(uncertainty.b_controller_profile, "uncertainty_release")
        self.assertEqual(genetic.b_controller_profile, "genetic_cholinergic_precision")

    def test_diagnostic_catalog_registers_b53_noradrenergic_arousal_variants(self) -> None:
        h48, surprise, stress, h56, genetic = resolve_ablation_configs(
            [
                B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
                B53_SURPRISE_GAIN_H48_POLICY_NAME,
                B53_STRESS_PRECISION_H48_POLICY_NAME,
                B53_NORADRENERGIC_AROUSAL_GAIN_H56_POLICY_NAME,
                B53_GENETIC_AROUSAL_PRECISION_H48_POLICY_NAME,
            ]
        )

        for config in (h48, surprise, stress, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 53)
            self.assertEqual(config.b_parent_level, 52)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b53_norepinephrine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "noradrenergic_arousal_gain")
        self.assertEqual(surprise.b_controller_profile, "surprise_gain")
        self.assertEqual(stress.b_controller_profile, "stress_precision")
        self.assertEqual(genetic.b_controller_profile, "genetic_arousal_precision")

    def test_diagnostic_catalog_registers_b54_serotonergic_patience_variants(self) -> None:
        h48, suppression, balance, h56, genetic = resolve_ablation_configs(
            [
                B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
                B54_IMPULSE_SUPPRESSION_H48_POLICY_NAME,
                B54_PATIENCE_BALANCE_H48_POLICY_NAME,
                B54_SEROTONERGIC_PATIENCE_GATE_H56_POLICY_NAME,
                B54_GENETIC_SEROTONIN_PATIENCE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, suppression, balance, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 54)
            self.assertEqual(config.b_parent_level, 53)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b54_serotonin_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "serotonergic_patience_gate")
        self.assertEqual(suppression.b_controller_profile, "impulse_suppression")
        self.assertEqual(balance.b_controller_profile, "patience_balance")
        self.assertEqual(genetic.b_controller_profile, "genetic_serotonin_patience")

    def test_diagnostic_catalog_registers_b55_hypothalamic_drive_variants(self) -> None:
        h48, recovery, arbiter, h56, genetic = resolve_ablation_configs(
            [
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
                B55_SATIETY_RECOVERY_BALANCE_H48_POLICY_NAME,
                B55_SLEEP_HUNGER_ARBITER_H48_POLICY_NAME,
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H56_POLICY_NAME,
                B55_GENETIC_HYPOTHALAMIC_DRIVE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, recovery, arbiter, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 55)
            self.assertEqual(config.b_parent_level, 54)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b55_drive_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "hypothalamic_drive_coupling")
        self.assertEqual(recovery.b_controller_profile, "satiety_recovery_balance")
        self.assertEqual(arbiter.b_controller_profile, "sleep_hunger_arbiter")
        self.assertEqual(genetic.b_controller_profile, "genetic_hypothalamic_drive")

    def test_diagnostic_catalog_registers_b56_hpa_stress_variants(self) -> None:
        h48, recovery, stress, h56, genetic = resolve_ablation_configs(
            [
                B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
                B56_CORTISOL_RECOVERY_BALANCE_H48_POLICY_NAME,
                B56_STRESS_LOAD_GATE_H48_POLICY_NAME,
                B56_HPA_STRESS_AXIS_H56_POLICY_NAME,
                B56_GENETIC_HPA_STRESS_H48_POLICY_NAME,
            ]
        )

        for config in (h48, recovery, stress, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 56)
            self.assertEqual(config.b_parent_level, 55)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b56_endocrine_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "hpa_stress_axis")
        self.assertEqual(recovery.b_controller_profile, "cortisol_recovery_balance")
        self.assertEqual(stress.b_controller_profile, "stress_load_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_hpa_stress")

    def test_diagnostic_catalog_registers_b57_insular_variants(self) -> None:
        h48, salience, awareness, h56, genetic = resolve_ablation_configs(
            [
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
                B57_VISCERAL_SALIENCE_GATE_H48_POLICY_NAME,
                B57_STRESS_DRIVE_AWARENESS_H48_POLICY_NAME,
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H56_POLICY_NAME,
                B57_GENETIC_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
            ]
        )

        for config in (h48, salience, awareness, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 57)
            self.assertEqual(config.b_parent_level, 56)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b57_awareness_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(
            h48.b_controller_profile,
            "insular_interoceptive_awareness",
        )
        self.assertEqual(salience.b_controller_profile, "visceral_salience_gate")
        self.assertEqual(awareness.b_controller_profile, "stress_drive_awareness")
        self.assertEqual(genetic.b_controller_profile, "genetic_interoceptive_awareness")

    def test_diagnostic_catalog_registers_b58_acc_conflict_variants(self) -> None:
        h48, error, balance, h56, genetic = resolve_ablation_configs(
            [
                B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
                B58_ERROR_SALIENCE_GATE_H48_POLICY_NAME,
                B58_CONFLICT_RESOLUTION_BALANCE_H48_POLICY_NAME,
                B58_ACC_CONFLICT_MONITOR_H56_POLICY_NAME,
                B58_GENETIC_ACC_CONFLICT_H48_POLICY_NAME,
            ]
        )

        for config in (h48, error, balance, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 58)
            self.assertEqual(config.b_parent_level, 57)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b58_conflict_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "acc_conflict_monitor")
        self.assertEqual(error.b_controller_profile, "error_salience_gate")
        self.assertEqual(balance.b_controller_profile, "conflict_resolution_balance")
        self.assertEqual(genetic.b_controller_profile, "genetic_acc_conflict")

    def test_diagnostic_catalog_registers_b59_prefrontal_variants(self) -> None:
        h48, working, task, h56, genetic = resolve_ablation_configs(
            [
                B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
                B59_WORKING_SET_STABILITY_H48_POLICY_NAME,
                B59_EXECUTIVE_TASK_SET_H48_POLICY_NAME,
                B59_PREFRONTAL_GOAL_CONTEXT_H56_POLICY_NAME,
                B59_GENETIC_PREFRONTAL_CONTROL_H48_POLICY_NAME,
            ]
        )

        for config in (h48, working, task, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 59)
            self.assertEqual(config.b_parent_level, 58)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b59_context_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "prefrontal_goal_context")
        self.assertEqual(working.b_controller_profile, "working_set_stability")
        self.assertEqual(task.b_controller_profile, "executive_task_set")
        self.assertEqual(genetic.b_controller_profile, "genetic_prefrontal_control")

    def test_diagnostic_catalog_registers_b60_orbitofrontal_variants(self) -> None:
        h48, reversal, prediction, h56, genetic = resolve_ablation_configs(
            [
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
                B60_REVERSAL_VALUE_GATE_H48_POLICY_NAME,
                B60_GOAL_OUTCOME_PREDICTION_H48_POLICY_NAME,
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H56_POLICY_NAME,
                B60_GENETIC_ORBITOFRONTAL_VALUE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, reversal, prediction, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 60)
            self.assertEqual(config.b_parent_level, 59)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b60_value_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "orbitofrontal_outcome_value")
        self.assertEqual(reversal.b_controller_profile, "reversal_value_gate")
        self.assertEqual(prediction.b_controller_profile, "goal_outcome_prediction")
        self.assertEqual(genetic.b_controller_profile, "genetic_orbitofrontal_value")

    def test_diagnostic_catalog_registers_b61_amygdala_variants(self) -> None:
        h48, threat, safety, h56, genetic = resolve_ablation_configs(
            [
                B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
                B61_THREAT_VALUE_TAG_H48_POLICY_NAME,
                B61_SAFETY_PREDICTION_GATE_H48_POLICY_NAME,
                B61_AMYGDALA_SAFETY_VALUE_H56_POLICY_NAME,
                B61_GENETIC_AMYGDALA_SAFETY_H48_POLICY_NAME,
            ]
        )

        for config in (h48, threat, safety, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 61)
            self.assertEqual(config.b_parent_level, 60)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b61_affect_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "amygdala_safety_value")
        self.assertEqual(threat.b_controller_profile, "threat_value_tag")
        self.assertEqual(safety.b_controller_profile, "safety_prediction_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_amygdala_safety")

    def test_diagnostic_catalog_registers_b62_defensive_mode_variants(self) -> None:
        h48, balance, shelter, h56, genetic = resolve_ablation_configs(
            [
                B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
                B62_FREEZE_FLEE_BALANCE_H48_POLICY_NAME,
                B62_SHELTER_DEFENSE_GATE_H48_POLICY_NAME,
                B62_DEFENSIVE_MODE_SELECTOR_H56_POLICY_NAME,
                B62_GENETIC_DEFENSIVE_MODE_H48_POLICY_NAME,
            ]
        )

        for config in (h48, balance, shelter, h56, genetic):
            self.assertEqual(config.architecture, "b_series")
            self.assertEqual(config.b_level, 62)
            self.assertEqual(config.b_parent_level, 61)
            self.assertFalse(config.b_transfer_allow_low_coverage)
            self.assertIn("b62_defense_decay", config.b_controller_params)
        self.assertEqual(h48.b_hidden_dim, 48)
        self.assertEqual(h56.b_hidden_dim, 56)
        self.assertEqual(h48.b_controller_profile, "defensive_mode_selector")
        self.assertEqual(balance.b_controller_profile, "freeze_flee_balance")
        self.assertEqual(shelter.b_controller_profile, "shelter_defense_gate")
        self.assertEqual(genetic.b_controller_profile, "genetic_defensive_mode")

    def test_legacy_variant_is_not_routed_through_current_world_brain(self) -> None:
        config = _b0_config("b0_legacy_semantic_policy")
        with self.assertRaisesRegex(ValueError, "LegacyB0Simulation"):
            SpiderBrain(seed=1, module_dropout=0.0, config=config)


class BSeriesBridgeTest(unittest.TestCase):
    def test_move_to_food_selects_primitive_food_progress(self) -> None:
        decision = bridge_b_semantic_action("MOVE_TO_FOOD", _bridge_observation())
        self.assertEqual(decision.primitive_action, "MOVE_RIGHT")
        self.assertEqual(decision.reason, "food_progress")
        self.assertAlmostEqual(decision.food_delta_used, 1.0)

    def test_move_to_food_inside_shelter_first_favors_exit_geodesic(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        geodesics = meta["local_geodesic_consequences"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(geodesics, dict)
        assert isinstance(transitions, dict)
        geodesics["MOVE_UP"] = {
            "exit_geodesic_delta": 1.0,
            "deep_geodesic_delta": -1.0,
            "next_on_exit_target": False,
            "next_on_deep_target": False,
        }
        transitions["MOVE_UP"] = {
            "food_dist_delta": -1.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 1.0,
            "next_cell_has_food": False,
        }
        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)
        self.assertEqual(decision.primitive_action, "MOVE_UP")
        self.assertEqual(decision.reason, "food_exit_to_outside")

    def test_move_to_food_uses_memory_vector_after_shelter_exit(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["shelter_role"] = "outside"
        meta["memory_vectors"] = {
            "food": {"dx": 0.60, "dy": 0.10, "age": 0.0, "ttl": 30},
        }
        affordances = meta["local_affordances"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(affordances, dict)
        assert isinstance(transitions, dict)
        affordances["MOVE_DOWN"] = {"blocked": False, "next_role": "entrance"}
        transitions["MOVE_DOWN"] = {
            "food_dist_delta": 1.0,
            "shelter_dist_delta": 1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }
        transitions["MOVE_RIGHT"] = {
            "food_dist_delta": 0.0,
            "shelter_dist_delta": -1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }

        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)

        self.assertEqual(decision.primitive_action, "MOVE_RIGHT")
        self.assertEqual(decision.reason, "food_memory_vector")

    def test_move_to_food_does_not_reenter_shelter_without_food(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["shelter_role"] = "outside"
        affordances = meta["local_affordances"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(affordances, dict)
        assert isinstance(transitions, dict)
        affordances["MOVE_DOWN"] = {"blocked": False, "next_role": "entrance"}
        transitions["MOVE_DOWN"] = {
            "food_dist_delta": 1.0,
            "shelter_dist_delta": 1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }
        transitions["MOVE_RIGHT"] = {
            "food_dist_delta": 0.25,
            "shelter_dist_delta": -1.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }
        transitions["MOVE_LEFT"] = {
            "food_dist_delta": 0.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 0.0,
            "next_cell_has_food": False,
        }

        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)

        self.assertEqual(decision.primitive_action, "MOVE_RIGHT")

    def test_move_to_shelter_selects_primitive_shelter_progress(self) -> None:
        decision = bridge_b_semantic_action("MOVE_TO_SHELTER", _bridge_observation())
        self.assertEqual(decision.primitive_action, "MOVE_LEFT")
        self.assertEqual(decision.reason, "shelter_progress")
        self.assertAlmostEqual(decision.shelter_delta_used, 1.5)

    def test_move_to_shelter_holds_when_already_deep(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        decision = bridge_b_semantic_action("MOVE_TO_SHELTER", observation)
        self.assertEqual(decision.primitive_action, "STAY")
        self.assertEqual(decision.reason, "already_deep_shelter")

    def test_blocked_winning_move_is_masked_without_semantic_rewrite(self) -> None:
        observation = _bridge_observation()
        meta = observation["meta"]
        assert isinstance(meta, dict)
        affordances = meta["local_affordances"]
        assert isinstance(affordances, dict)
        affordances["MOVE_RIGHT"] = {"blocked": True}
        decision = bridge_b_semantic_action("MOVE_TO_FOOD", observation)
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.primitive_action, "MOVE_LEFT")
        self.assertTrue(decision.blocked_mask["MOVE_RIGHT"])
        self.assertEqual(decision.external_override_count, 0)

    def test_non_movement_semantic_actions_map_to_stay(self) -> None:
        for semantic_action in ("STAY", "EAT", "SLEEP"):
            with self.subTest(semantic_action=semantic_action):
                decision = bridge_b_semantic_action(
                    semantic_action,
                    _bridge_observation(),
                )
                self.assertEqual(decision.primitive_action, "STAY")

    def test_explore_selects_unblocked_primitive_move(self) -> None:
        decision = bridge_b_semantic_action("EXPLORE", _bridge_observation())
        self.assertIn(decision.primitive_action, {"MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"})


class BSeriesRuntimeTest(unittest.TestCase):
    def test_b0_current_bridge_never_emits_semantic_action_to_world(self) -> None:
        brain = SpiderBrain(seed=11, module_dropout=0.0, config=_b0_config())
        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["SLEEP"]
        ] = 10.0

        decision = brain.act_inference(
            _brain_observation(
                hunger={
                    "hunger": 0.82,
                    "food_visible": 1.0,
                    "food_certainty": 1.0,
                },
                sleep={"health": 1.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.learned_semantic_action, "SLEEP")
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.b_effective_level, B_CURRENT_BRIDGE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B_CURRENT_BRIDGE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.semantic_override_count, 1)
        self.assertEqual(decision.bridge_primitive_action, "MOVE_RIGHT")
        self.assertEqual(ACTIONS[decision.action_idx], "MOVE_RIGHT")
        self.assertNotIn(decision.semantic_action, ACTIONS)
        self.assertEqual(decision.external_override_count, 0)

    def test_b0_current_rest_phase_maps_to_stay_without_final_bias(self) -> None:
        brain = SpiderBrain(seed=12, module_dropout=0.0, config=_b0_config())
        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["MOVE_TO_FOOD"]
        ] = 10.0

        meta = _bridge_observation()["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        geodesics = meta["local_geodesic_consequences"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(geodesics, dict)
        assert isinstance(transitions, dict)
        geodesics["MOVE_UP"] = {
            "exit_geodesic_delta": 1.0,
            "deep_geodesic_delta": -1.0,
            "next_on_exit_target": False,
            "next_on_deep_target": False,
        }
        geodesics["MOVE_LEFT"] = {
            "exit_geodesic_delta": 0.0,
            "deep_geodesic_delta": 0.0,
            "next_on_exit_target": False,
            "next_on_deep_target": True,
        }
        transitions["MOVE_UP"] = {
            "food_dist_delta": -1.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 1.0,
            "next_cell_has_food": False,
        }
        decision = brain.act_inference(
            _brain_observation(
                meta,
                sleep={
                    "fatigue": 0.90,
                    "hunger": 0.12,
                    "on_shelter": 1.0,
                    "night": 1.0,
                    "health": 1.0,
                    "sleep_debt": 0.90,
                    "shelter_role_level": 1.0,
                },
            ),
            sample=False,
        )

        self.assertEqual(decision.learned_semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.semantic_action, "SLEEP")
        self.assertEqual(decision.bridge_primitive_action, "STAY")
        self.assertEqual(ACTIONS[decision.action_idx], "STAY")
        self.assertEqual(decision.semantic_override_count, 1)

    def test_b0_current_critical_hunger_can_leave_deep_shelter_under_residual_smell(self) -> None:
        brain = SpiderBrain(seed=13, module_dropout=0.0, config=_b0_config())
        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[B_SEMANTIC_ACTION_TO_INDEX["STAY"]] = 10.0

        meta = _bridge_observation()["meta"]
        assert isinstance(meta, dict)
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0
        geodesics = meta["local_geodesic_consequences"]
        transitions = meta["local_transition_consequences"]
        assert isinstance(geodesics, dict)
        assert isinstance(transitions, dict)
        geodesics["MOVE_UP"] = {
            "exit_geodesic_delta": 1.0,
            "deep_geodesic_delta": -1.0,
            "next_on_exit_target": False,
            "next_on_deep_target": False,
        }
        geodesics["MOVE_LEFT"] = {
            "exit_geodesic_delta": 0.0,
            "deep_geodesic_delta": 0.0,
            "next_on_exit_target": False,
            "next_on_deep_target": True,
        }
        transitions["MOVE_UP"] = {
            "food_dist_delta": -1.0,
            "shelter_dist_delta": 0.0,
            "predator_dist_delta": 1.0,
            "next_cell_has_food": False,
        }
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.92},
                sleep={
                    "fatigue": 0.12,
                    "hunger": 0.92,
                    "on_shelter": 1.0,
                    "health": 1.0,
                    "shelter_role_level": 1.0,
                },
                threat={"predator_smell_strength": 0.70},
            ),
            sample=False,
        )

        self.assertEqual(decision.learned_semantic_action, "STAY")
        self.assertEqual(decision.semantic_action, "MOVE_TO_FOOD")
        self.assertEqual(decision.bridge_primitive_action, "MOVE_UP")
        self.assertEqual(ACTIONS[decision.action_idx], "MOVE_UP")

    def test_b1_uses_network_policy_without_b0_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SpiderBrain(seed=14, module_dropout=0.0, config=_b0_config())
            checkpoint = source.save(Path(tmpdir) / "b0")
            config = build_b1_capacity_config(
                B1_CAPACITY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=15, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["SLEEP"]
        ] = 10.0

        decision = brain.act_inference(
            _brain_observation(
                hunger={
                    "hunger": 0.82,
                    "food_visible": 1.0,
                    "food_certainty": 1.0,
                },
                sleep={"health": 1.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, "B1")
        self.assertEqual(decision.learned_semantic_action, "SLEEP")
        self.assertEqual(decision.semantic_action, "SLEEP")
        self.assertEqual(decision.semantic_action_source, "network_policy")
        self.assertEqual(decision.semantic_override_count, 0)
        self.assertEqual(decision.bridge_primitive_action, "STAY")
        self.assertEqual(ACTIONS[decision.action_idx], "STAY")

    def test_b1_threat_guard_uses_transfer_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = SpiderBrain(seed=16, module_dropout=0.0, config=_b0_config())
            checkpoint = source.save(Path(tmpdir) / "b0")
            config = build_b1_capacity_config(
                B1_THREAT_GUARD_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=17, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["STAY"]
        ] = 10.0

        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.45},
                sleep={"health": 1.0, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.70},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B1_THREAT_GUARD_EFFECTIVE_LEVEL)
        self.assertEqual(decision.learned_semantic_action, "STAY")
        self.assertEqual(decision.semantic_action, "MOVE_TO_SHELTER")
        self.assertEqual(
            decision.semantic_action_source,
            B1_THREAT_GUARD_SELECTION_SOURCE,
        )
        self.assertEqual(decision.semantic_override_count, 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b3_recurrent_guard_uses_transfer_memory_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            config = build_b3_contact_memory_config(
                B3_RECURRENT_GUARD_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=23, module_dropout=0.0, config=config)

        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.90},
                sleep={"health": 0.70, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.6},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B3_RECURRENT_GUARD_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B3_RECURRENT_GUARD_SELECTION_SOURCE,
        )
        self.assertIn(decision.bridge_primitive_action, ACTIONS)
        self.assertEqual(decision.b3_controller_profile, "recurrent_guard_strict_until_60")

    def test_b4_recovery_balance_uses_transfer_memory_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b3_recurrent_guard_source(tmpdir)
            config = build_b4_recovery_balance_config(
                B4_RECOVERY_BALANCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=24, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["MOVE_TO_FOOD"]
        ] = 10.0
        brain.set_direct_policy_event_clock(20)
        brain.act_inference(
            _brain_observation(hunger={"hunger": 0.90}),
            sample=False,
        )
        brain.set_direct_policy_event_clock(21)
        meta = dict(_bridge_observation()["meta"])
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0

        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.50},
                sleep={
                    "health": 0.35,
                    "fatigue": 0.70,
                    "sleep_debt": 0.70,
                    "on_shelter": 1.0,
                    "shelter_role_level": 1.0,
                },
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B4_RECOVERY_BALANCE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B4_RECOVERY_BALANCE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.semantic_action, "SLEEP")
        self.assertEqual(decision.b4_controller_profile, "recovery_balance")
        self.assertGreaterEqual(float(decision.b4_recovery_pressure), 0.60)
        self.assertTrue(decision.b4_sleep_hold)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b4_genetic_recovery_records_genetic_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b3_recurrent_guard_source(tmpdir)
            config = build_b4_recovery_balance_config(
                B4_GENETIC_RECOVERY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
                controller_profile="genetic_recovery",
                controller_params={"ga_generation": 2, "ga_candidate": 3},
            )
            brain = SpiderBrain(seed=25, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(20)
        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.50},
                sleep={"health": 0.35, "on_shelter": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.semantic_action_source,
            B4_GENETIC_RECOVERY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b4_genetic_generation, 2)
        self.assertEqual(decision.b4_genetic_candidate, 3)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b5_homeostatic_arbiter_uses_transfer_memory_and_primitive_bridge(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            config = build_b5_homeostatic_arbiter_config(
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=26, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["MOVE_TO_FOOD"]
        ] = 10.0
        brain.set_direct_policy_event_clock(69)
        brain.act_inference(
            _brain_observation(hunger={"hunger": 0.90}),
            sample=False,
        )
        brain.set_direct_policy_event_clock(70)
        meta = dict(_bridge_observation()["meta"])
        meta["on_shelter"] = True
        meta["shelter_role"] = "deep"
        meta["shelter_role_level"] = 1.0

        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.30},
                sleep={
                    "health": 0.40,
                    "fatigue": 0.88,
                    "sleep_debt": 0.86,
                    "on_shelter": 1.0,
                },
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B5_HOMEOSTATIC_ARBITER_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B5_HOMEOSTATIC_ARBITER_SELECTION_SOURCE,
        )
        self.assertEqual(decision.semantic_action, "SLEEP")
        self.assertEqual(decision.b5_controller_profile, "homeostatic_arbiter")
        self.assertGreaterEqual(float(decision.b5_sleep_pressure), 0.80)
        self.assertGreaterEqual(int(decision.b5_sleep_bout_lock), 1)
        self.assertEqual(decision.b5_homeostatic_decision, "sleep_bout_hold")
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b5_genetic_homeostasis_records_genetic_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            config = build_b5_homeostatic_arbiter_config(
                B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
                controller_profile="genetic_homeostasis",
                controller_params={"ga_generation": 3, "ga_candidate": 4},
            )
            brain = SpiderBrain(seed=27, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(0)
        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.92},
                sleep={"health": 0.80, "fatigue": 0.20, "sleep_debt": 0.20},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.semantic_action_source,
            B5_GENETIC_HOMEOSTASIS_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b5_genetic_generation, 3)
        self.assertEqual(decision.b5_genetic_candidate, 4)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b5_homeostatic_locks_reset_on_episode_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            config = build_b5_homeostatic_arbiter_config(
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=28, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(10)
        first = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.92},
                sleep={"health": 0.85, "fatigue": 0.20, "sleep_debt": 0.20},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )
        brain.set_direct_policy_event_clock(0)
        second = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.20},
                sleep={"health": 0.85, "fatigue": 0.20, "sleep_debt": 0.20},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertGreaterEqual(int(first.b5_forage_commitment_lock), 1)
        self.assertEqual(int(second.b5_forage_commitment_lock), 0)

    def test_b6_risk_corridor_uses_b5_transfer_and_action_center_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            config = build_b6_risk_corridor_config(
                B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=29, module_dropout=0.0, config=config)

        bus = MessageBus()
        bus.set_tick(20)
        brain.set_direct_policy_event_clock(20)
        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.85},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_visible": 1.0, "predator_certainty": 1.0},
            ),
            bus=bus,
            sample=False,
        )
        action_center_messages = [
            message for message in bus.topic_messages("action.selection")
            if message.sender == "action_center"
        ]

        self.assertEqual(decision.b_effective_level, B6_RISK_CORRIDOR_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B6_RISK_CORRIDOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b6_controller_family, "risk_corridor")
        self.assertEqual(decision.semantic_action, "MOVE_TO_SHELTER")
        self.assertGreater(float(decision.b6_risk_pressure), 0.9)
        self.assertEqual(float(decision.b6_threat_priority), 1.0)
        self.assertEqual(float(decision.b6_forage_suppressed), 1.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)
        self.assertEqual(len(action_center_messages), 1)
        self.assertEqual(
            action_center_messages[0].payload["winning_valence"],
            "threat",
        )
        self.assertLess(
            action_center_messages[0].payload["module_gates"]["hunger_center"],
            0.5,
        )

    def test_b6_recurrent_memory_resets_locks_on_episode_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            config = build_b6_risk_corridor_config(
                B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=30, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(12)
        first = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.85},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_visible": 1.0, "predator_certainty": 1.0},
            ),
            sample=False,
        )
        brain.set_direct_policy_event_clock(0)
        second = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.30},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_visible": 0.0, "predator_certainty": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            first.semantic_action_source,
            B6_RECURRENT_MEMORY_SELECTION_SOURCE,
        )
        self.assertEqual(first.b_effective_level, B6_RECURRENT_MEMORY_EFFECTIVE_LEVEL)
        self.assertGreaterEqual(int(first.b6_return_lock), 1)
        self.assertEqual(int(second.b6_return_lock), 0)

    def test_b6_fused_controller_records_fused_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            config = build_b6_risk_corridor_config(
                B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
                controller_profile="fused_risk_recurrent",
                controller_params={"ga_generation": 1, "ga_candidate": 2},
            )
            brain = SpiderBrain(seed=31, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.50},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_visible": 1.0, "predator_certainty": 1.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B6_FUSED_RISK_RECURRENT_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b6_controller_family, "fused_risk_recurrent")
        self.assertEqual(decision.b6_genetic_generation, 1)
        self.assertEqual(decision.b6_genetic_candidate, 2)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b7_affordance_budget_uses_b6_transfer_and_records_viability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b6_fused_risk_recurrent_source(tmpdir)
            config = build_b7_affordance_budget_config(
                B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=32, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 10.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.30, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B7_AFFORDANCE_BUDGET_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B7_AFFORDANCE_BUDGET_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b7_controller_profile, "affordance_budget")
        self.assertEqual(decision.b7_decision, "abort_return_unviable")
        self.assertTrue(decision.b7_abort_return)
        self.assertIsNotNone(decision.b7_budget_margin)
        self.assertEqual(decision.b7_food_steps_estimate, 10.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b7_affordance_locks_reset_on_episode_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b6_fused_risk_recurrent_source(tmpdir)
            config = build_b7_affordance_budget_config(
                B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=33, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 5.0
        meta["shelter_dist"] = 1.0
        brain.set_direct_policy_event_clock(12)
        first = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.90, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )
        brain.set_direct_policy_event_clock(0)
        second = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.20},
                sleep={"health": 0.90, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(first.b7_decision, "continue_viable")
        self.assertGreaterEqual(int(first.b7_commitment_lock), 1)
        self.assertEqual(int(second.b7_commitment_lock), 0)

    def test_b8_spatial_affordance_uses_b7_transfer_and_records_map(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b7_affordance_budget_source(tmpdir)
            config = build_b8_spatial_affordance_config(
                B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=34, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B8_SPATIAL_AFFORDANCE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B8_SPATIAL_AFFORDANCE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b8_controller_profile, "spatial_affordance_map")
        self.assertEqual(decision.b8_decision, "corridor_continue_mapped")
        self.assertEqual(decision.b8_spatial_map_state, "food_vector_available")
        self.assertIsNotNone(decision.b8_local_affordance_score)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b9_waypoint_planner_uses_b8_transfer_and_records_route(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b8_spatial_affordance_source(tmpdir)
            config = build_b9_waypoint_planner_config(
                B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=35, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B9_WAYPOINT_PLANNER_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B9_WAYPOINT_PLANNER_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b9_controller_profile, "waypoint_planner")
        self.assertEqual(decision.b9_decision, "commit_food_waypoint")
        self.assertEqual(decision.b9_route_state, "food_waypoint_locked")
        self.assertGreaterEqual(int(decision.b9_waypoint_lock), 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b10_prospective_replay_uses_b9_transfer_and_records_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b9_waypoint_planner_source(tmpdir)
            config = build_b10_prospective_replay_config(
                B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=36, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B10_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B10_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b10_controller_profile, "prospective_replay")
        self.assertEqual(decision.b10_decision, "commit_replayed_route")
        self.assertEqual(decision.b10_replay_state, "prospective_food_plan")
        self.assertGreaterEqual(int(decision.b10_plan_commitment), 1)
        self.assertGreater(float(decision.b10_prospective_value), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b11_confidence_arbiter_uses_b10_transfer_and_records_confidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b10_prospective_replay_source(tmpdir)
            config = build_b11_confidence_arbiter_config(
                B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=37, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B11_CONFIDENCE_ARBITER_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B11_CONFIDENCE_ARBITER_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b11_controller_profile, "confidence_arbiter")
        self.assertEqual(decision.b11_decision, "commit_confident_plan")
        self.assertEqual(decision.b11_confidence_state, "high_confidence_plan")
        self.assertGreaterEqual(int(decision.b11_confidence_lock), 1)
        self.assertGreater(float(decision.b11_neuromod_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b12_predictive_attention_uses_b11_transfer_and_records_attention(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b11_confidence_arbiter_source(tmpdir)
            config = build_b12_predictive_attention_config(
                B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=38, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(15)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B12_PREDICTIVE_ATTENTION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B12_PREDICTIVE_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b12_controller_profile, "predictive_attention")
        self.assertEqual(decision.b12_decision, "commit_attended_affordance")
        self.assertEqual(decision.b12_attention_state, "attended_food_affordance")
        self.assertGreaterEqual(int(decision.b12_search_lock), 1)
        self.assertGreater(float(decision.b12_attention_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b13_local_search_uses_b12_transfer_and_records_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b12_predictive_attention_source(tmpdir)
            config = build_b13_local_affordance_search_config(
                B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=39, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(16)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B13_LOCAL_SEARCH_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B13_LOCAL_SEARCH_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b13_controller_profile, "local_affordance_search")
        self.assertEqual(decision.b13_decision, "commit_local_affordance_search")
        self.assertEqual(decision.b13_search_state, "local_route_viable")
        self.assertGreaterEqual(int(decision.b13_search_lock), 1)
        self.assertGreater(float(decision.b13_local_route_score), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b14_uncertainty_uses_b13_transfer_and_records_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b13_local_affordance_search_source(tmpdir)
            config = build_b14_affordance_uncertainty_config(
                B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=40, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(17)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B14_AFFORDANCE_UNCERTAINTY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B14_AFFORDANCE_UNCERTAINTY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b14_controller_profile, "affordance_uncertainty")
        self.assertEqual(decision.b14_decision, "commit_confident_affordance")
        self.assertEqual(decision.b14_uncertainty_state, "confidence_calibrated_route")
        self.assertGreaterEqual(int(decision.b14_commitment_lock), 1)
        self.assertGreater(float(decision.b14_affordance_confidence), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b15_option_critic_uses_b14_transfer_and_records_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b14_affordance_uncertainty_source(tmpdir)
            config = build_b15_option_critic_config(
                B15_OPTION_CRITIC_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=41, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(18)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B15_OPTION_CRITIC_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B15_OPTION_CRITIC_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b15_controller_profile, "option_critic")
        self.assertEqual(decision.b15_decision, "persist_food_option")
        self.assertEqual(decision.b15_option_state, "option_persist_food_route")
        self.assertGreaterEqual(int(decision.b15_option_lock), 1)
        self.assertGreater(float(decision.b15_option_value), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b16_option_ensemble_uses_b15_transfer_and_records_votes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b15_option_critic_source(tmpdir)
            config = build_b16_option_ensemble_config(
                B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=42, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(18)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B16_OPTION_ENSEMBLE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B16_OPTION_ENSEMBLE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b16_controller_profile, "option_ensemble")
        self.assertEqual(decision.b16_decision, "ensemble_continue_option")
        self.assertEqual(decision.b16_ensemble_state, "ensemble_continue_consensus")
        self.assertGreaterEqual(int(decision.b16_ensemble_lock), 1)
        self.assertGreater(float(decision.b16_continue_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b17_neuromodulated_ensemble_uses_b16_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b16_option_ensemble_source(tmpdir)
            config = build_b17_neuromodulated_ensemble_config(
                B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=43, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(18)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B17_NEUROMODULATED_ENSEMBLE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B17_NEUROMODULATED_ENSEMBLE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b17_controller_profile, "neuromodulated_ensemble")
        self.assertEqual(decision.b17_decision, "neuromodulated_continue")
        self.assertEqual(decision.b17_modulator_state, "modulated_continue")
        self.assertGreaterEqual(int(decision.b17_modulation_lock), 1)
        self.assertGreater(float(decision.b17_arousal_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b18_eligibility_trace_uses_b17_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b17_neuromodulated_ensemble_source(tmpdir)
            config = build_b18_eligibility_trace_config(
                B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=44, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(18)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B18_ELIGIBILITY_TRACE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B18_ELIGIBILITY_TRACE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b18_controller_profile, "eligibility_trace")
        self.assertEqual(decision.b18_decision, "eligibility_stabilize_option")
        self.assertEqual(decision.b18_trace_state, "trace_stabilizes_option")
        self.assertGreaterEqual(int(decision.b18_trace_lock), 1)
        self.assertGreater(float(decision.b18_eligibility_trace), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b19_episodic_meta_memory_uses_b18_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b18_eligibility_trace_source(tmpdir)
            config = build_b19_episodic_meta_memory_config(
                B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=45, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(19)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B19_EPISODIC_META_MEMORY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B19_EPISODIC_META_MEMORY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b19_controller_profile, "episodic_meta_memory")
        self.assertEqual(decision.b19_decision, "episodic_consolidate_option")
        self.assertEqual(decision.b19_memory_state, "memory_consolidates_option")
        self.assertGreaterEqual(int(decision.b19_memory_lock), 1)
        self.assertGreater(float(decision.b19_episode_memory), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b20_working_memory_gate_uses_b19_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b19_episodic_meta_memory_source(tmpdir)
            config = build_b20_working_memory_gate_config(
                B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=46, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(20)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B20_WORKING_MEMORY_GATE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B20_WORKING_MEMORY_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b20_controller_profile, "working_memory_gate")
        self.assertEqual(decision.b20_decision, "working_memory_gate_continue")
        self.assertEqual(decision.b20_buffer_state, "working_memory_holds_context")
        self.assertGreaterEqual(int(decision.b20_buffer_lock), 1)
        self.assertGreater(float(decision.b20_working_buffer), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b21_hippocampal_replay_uses_b20_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b20_working_memory_gate_source(tmpdir)
            config = build_b21_hippocampal_replay_config(
                B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=47, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(21)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B21_HIPPOCAMPAL_REPLAY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B21_HIPPOCAMPAL_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b21_controller_profile, "hippocampal_replay")
        self.assertEqual(decision.b21_decision, "hippocampal_replay_continue")
        self.assertEqual(decision.b21_replay_state, "replay_rehearses_route")
        self.assertGreaterEqual(int(decision.b21_replay_lock), 1)
        self.assertGreater(float(decision.b21_sequence_memory), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b22_prospective_replay_uses_b21_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b21_hippocampal_replay_source(tmpdir)
            config = build_b22_prospective_replay_config(
                B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=48, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(22)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B22_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B22_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b22_controller_profile, "prospective_map_replay")
        self.assertEqual(decision.b22_decision, "prospective_replay_continue")
        self.assertEqual(decision.b22_sim_state, "prospective_sim_commits_route")
        self.assertGreaterEqual(int(decision.b22_sim_lock), 1)
        self.assertGreater(float(decision.b22_prospective_sim), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b23_conflict_monitor_uses_b22_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b22_prospective_replay_source(tmpdir)
            config = build_b23_conflict_monitor_config(
                B23_CONFLICT_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=49, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(23)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B23_CONFLICT_MONITOR_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B23_CONFLICT_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b23_controller_profile, "conflict_monitor")
        self.assertEqual(decision.b23_decision, "conflict_monitor_continue")
        self.assertEqual(decision.b23_conflict_state, "conflict_monitor_stabilizes_route")
        self.assertGreaterEqual(int(decision.b23_monitor_lock), 1)
        self.assertGreater(float(decision.b23_stability_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b24_precision_conflict_uses_b23_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b23_conflict_monitor_source(tmpdir)
            config = build_b24_precision_conflict_config(
                B24_PRECISION_CONFLICT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=50, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(24)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B24_PRECISION_CONFLICT_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B24_PRECISION_CONFLICT_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b24_controller_profile, "precision_conflict")
        self.assertEqual(decision.b24_decision, "precision_conflict_continue")
        self.assertEqual(decision.b24_precision_state, "precision_conflict_stabilizes_route")
        self.assertGreaterEqual(int(decision.b24_precision_lock), 1)
        self.assertGreater(float(decision.b24_precision_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b25_metacognitive_confidence_uses_b24_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b24_precision_conflict_source(tmpdir)
            config = build_b25_metacognitive_confidence_config(
                B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=51, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(25)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B25_METACOGNITIVE_CONFIDENCE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B25_METACOGNITIVE_CONFIDENCE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b25_controller_profile, "metacognitive_confidence")
        self.assertIn(
            decision.b25_decision,
            {"metacognitive_confidence_continue", "continue_meta_lock"},
        )
        self.assertNotEqual(decision.b25_metacognitive_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b25_meta_lock), 1)
        self.assertGreater(float(decision.b25_confidence_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b26_allostatic_prediction_uses_b25_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b25_metacognitive_confidence_source(tmpdir)
            config = build_b26_allostatic_prediction_config(
                B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=52, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(26)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B26_ALLOSTATIC_PREDICTION_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B26_ALLOSTATIC_PREDICTION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b26_controller_profile, "allostatic_prediction")
        self.assertIn(
            decision.b26_decision,
            {"allostatic_prediction_continue", "continue_allostatic_lock"},
        )
        self.assertNotEqual(decision.b26_allostatic_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b26_stability_lock), 1)
        self.assertGreater(float(decision.b26_control_vote), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b27_arousal_gain_uses_b26_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b26_allostatic_prediction_source(tmpdir)
            config = build_b27_arousal_gain_config(
                B27_AROUSAL_GAIN_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=53, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(27)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B27_AROUSAL_GAIN_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B27_AROUSAL_GAIN_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b27_controller_profile, "arousal_gain")
        self.assertIn(
            decision.b27_decision,
            {"arousal_gain_continue", "continue_arousal_lock"},
        )
        self.assertNotEqual(decision.b27_arousal_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b27_arousal_lock), 1)
        self.assertGreater(float(decision.b27_gain_modulation), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b28_interoceptive_attention_uses_b27_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b27_arousal_gain_source(tmpdir)
            config = build_b28_interoceptive_attention_config(
                B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=54, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(28)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B28_INTEROCEPTIVE_ATTENTION_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B28_INTEROCEPTIVE_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b28_controller_profile, "interoceptive_attention")
        self.assertIn(
            decision.b28_decision,
            {"interoceptive_attention_continue", "continue_attention_lock"},
        )
        self.assertNotEqual(decision.b28_attention_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b28_attention_lock), 1)
        self.assertGreater(float(decision.b28_attention_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b29_salience_competition_uses_b28_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b28_interoceptive_attention_source(tmpdir)
            config = build_b29_salience_competition_config(
                B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=55, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(29)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B29_SALIENCE_COMPETITION_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B29_SALIENCE_COMPETITION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b29_controller_profile, "salience_competition")
        self.assertIn(
            decision.b29_decision,
            {"salience_competition_continue", "continue_salience_lock"},
        )
        self.assertNotEqual(decision.b29_salience_state, "non_corridor")
        self.assertIn(decision.b29_winner_channel, {"corridor", "homeostasis", "threat"})
        self.assertGreaterEqual(int(decision.b29_salience_lock), 1)
        self.assertGreater(float(decision.b29_corridor_salience), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b30_basal_ganglia_gate_uses_b29_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b29_salience_competition_source(tmpdir)
            config = build_b30_basal_ganglia_gate_config(
                B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=56, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(30)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B30_BASAL_GANGLIA_GATE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B30_BASAL_GANGLIA_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b30_controller_profile, "basal_ganglia_gate")
        self.assertIn(
            decision.b30_decision,
            {"basal_gate_go", "continue_basal_gate_lock"},
        )
        self.assertNotEqual(decision.b30_gate_state, "non_corridor")
        self.assertIn(decision.b30_action_gate, {"go", "no_go"})
        self.assertGreaterEqual(int(decision.b30_gate_lock), 1)
        self.assertGreater(float(decision.b30_go_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b31_dopamine_prediction_error_uses_b30_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b30_basal_ganglia_gate_source(tmpdir)
            config = build_b31_dopamine_prediction_error_config(
                B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=57, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(31)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(
            decision.b_effective_level,
            B31_DOPAMINE_PREDICTION_ERROR_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B31_DOPAMINE_PREDICTION_ERROR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b31_controller_profile, "dopamine_prediction_error")
        self.assertIn(
            decision.b31_decision,
            {"dopamine_gate_go", "continue_dopamine_lock"},
        )
        self.assertNotEqual(decision.b31_dopamine_state, "non_corridor")
        self.assertGreaterEqual(int(decision.b31_dopamine_lock), 1)
        self.assertGreater(float(decision.b31_gate_bias), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b32_actor_critic_value_uses_b31_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b31_dopamine_prediction_error_source(tmpdir)
            config = build_b32_actor_critic_value_config(
                B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=58, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(32)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B32_ACTOR_CRITIC_VALUE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B32_ACTOR_CRITIC_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b32_controller_profile, "actor_critic_value")
        self.assertIn(
            decision.b32_decision,
            {"actor_critic_commit", "continue_value_lock"},
        )
        self.assertGreaterEqual(int(decision.b32_value_lock), 1)
        self.assertGreater(float(decision.b32_actor_advantage), 0.0)
        self.assertGreater(float(decision.b32_policy_bias), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b33_td_error_decomposition_uses_b32_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b32_actor_critic_value_source(tmpdir)
            config = build_b33_td_error_decomposition_config(
                B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=59, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(33)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B33_TD_ERROR_DECOMPOSITION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B33_TD_ERROR_DECOMPOSITION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b33_controller_profile, "td_error_decomposition")
        self.assertIn(decision.b33_decision, {"td_error_commit", "continue_td_lock"})
        self.assertGreaterEqual(int(decision.b33_td_lock), 1)
        self.assertGreater(float(decision.b33_td_error), 0.0)
        self.assertGreater(float(decision.b33_actor_update), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b34_eligibility_credit_uses_b33_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b33_td_error_decomposition_source(tmpdir)
            config = build_b34_eligibility_credit_config(
                B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=60, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(34)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B34_ELIGIBILITY_CREDIT_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B34_ELIGIBILITY_CREDIT_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b34_controller_profile, "eligibility_credit")
        self.assertIn(
            decision.b34_decision,
            {"eligibility_credit_commit", "continue_eligibility_lock"},
        )
        self.assertGreaterEqual(int(decision.b34_credit_lock), 1)
        self.assertGreater(float(decision.b34_eligibility_trace), 0.0)
        self.assertGreater(float(decision.b34_credit_assignment), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b35_forward_model_uses_b34_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b34_eligibility_credit_source(tmpdir)
            config = build_b35_forward_model_value_config(
                B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=61, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(35)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B35_FORWARD_MODEL_VALUE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B35_FORWARD_MODEL_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b35_controller_profile, "forward_model_value")
        self.assertIn(
            decision.b35_decision,
            {"forward_model_commit", "continue_model_lock"},
        )
        self.assertGreaterEqual(int(decision.b35_model_lock), 1)
        self.assertGreater(float(decision.b35_forward_value), 0.0)
        self.assertGreater(float(decision.b35_prediction_memory), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b36_latent_belief_state_uses_b35_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b35_forward_model_value_source(tmpdir)
            config = build_b36_latent_belief_state_config(
                B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=62, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        brain.set_direct_policy_event_clock(36)
        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.90},
                sleep={"health": 0.80, "on_shelter": 0.0},
                threat={"predator_smell_strength": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B36_LATENT_BELIEF_STATE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B36_LATENT_BELIEF_STATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b36_controller_profile, "latent_belief_state")
        self.assertIn(
            decision.b36_decision,
            {"latent_belief_commit", "continue_belief_lock"},
        )
        self.assertGreaterEqual(int(decision.b36_belief_lock), 1)
        self.assertGreater(float(decision.b36_latent_state), 0.0)
        self.assertGreater(float(decision.b36_belief_error), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b37_state_factor_gate_uses_b36_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b36_latent_belief_state_source(tmpdir)
            config = build_b37_state_factor_gate_config(
                B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=63, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(37, 40):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B37_STATE_FACTOR_GATE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B37_STATE_FACTOR_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b37_controller_profile, "state_factor_gate")
        self.assertIn(
            decision.b37_decision,
            {"state_factor_commit", "continue_factor_lock"},
        )
        self.assertGreaterEqual(int(decision.b37_factor_lock), 1)
        self.assertGreater(abs(float(decision.b37_external_state_factor)), 0.0)
        self.assertGreater(float(decision.b37_internal_state_factor), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b38_factor_attention_uses_b37_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b37_state_factor_gate_source(tmpdir)
            config = build_b38_factor_attention_config(
                B38_FACTOR_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=64, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(38, 42):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B38_FACTOR_ATTENTION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B38_FACTOR_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b38_controller_profile, "factor_attention")
        self.assertIn(
            decision.b38_decision,
            {"factor_attention_commit", "continue_attention_lock"},
        )
        self.assertGreaterEqual(int(decision.b38_attention_lock), 1)
        self.assertGreater(float(decision.b38_internal_attention), 0.0)
        self.assertGreater(float(decision.b38_attention_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b39_attention_binding_uses_b38_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b38_factor_attention_source(tmpdir)
            config = build_b39_attention_binding_config(
                B39_ATTENTION_BINDING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=65, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(39, 43):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B39_ATTENTION_BINDING_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B39_ATTENTION_BINDING_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b39_controller_profile, "attention_binding")
        self.assertIn(
            decision.b39_decision,
            {"attention_binding_commit", "continue_binding_lock"},
        )
        self.assertGreaterEqual(int(decision.b39_binding_lock), 1)
        self.assertGreater(float(decision.b39_binding_strength), 0.0)
        self.assertGreater(float(decision.b39_binding_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b40_global_workspace_uses_b39_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b39_attention_binding_source(tmpdir)
            config = build_b40_global_workspace_config(
                B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=66, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(40, 45):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B40_GLOBAL_WORKSPACE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B40_GLOBAL_WORKSPACE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b40_controller_profile, "global_workspace")
        self.assertIn(
            decision.b40_decision,
            {"global_workspace_commit", "continue_workspace_lock"},
        )
        self.assertGreaterEqual(int(decision.b40_workspace_lock), 1)
        self.assertGreater(float(decision.b40_workspace_activation), 0.0)
        self.assertGreater(float(decision.b40_broadcast_gain), 0.0)
        self.assertGreater(float(decision.b40_context_availability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b41_executive_workspace_uses_b40_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b40_global_workspace_source(tmpdir)
            config = build_b41_executive_workspace_config(
                B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=67, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(41, 46):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B41_EXECUTIVE_WORKSPACE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B41_EXECUTIVE_WORKSPACE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b41_controller_profile, "executive_workspace")
        self.assertIn(
            decision.b41_decision,
            {"executive_workspace_select", "continue_executive_lock"},
        )
        self.assertGreaterEqual(int(decision.b41_executive_lock), 1)
        self.assertGreater(float(decision.b41_executive_selection), 0.0)
        self.assertGreater(float(decision.b41_goal_context), 0.0)
        self.assertGreater(float(decision.b41_executive_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b42_error_monitor_uses_b41_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b41_executive_workspace_source(tmpdir)
            config = build_b42_error_monitor_config(
                B42_ERROR_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=68, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(42, 47):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B42_ERROR_MONITOR_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B42_ERROR_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b42_controller_profile, "error_monitor")
        self.assertIn(
            decision.b42_decision,
            {"error_monitor_commit", "continue_monitor_lock"},
        )
        self.assertGreaterEqual(int(decision.b42_monitor_lock), 1)
        self.assertGreaterEqual(float(decision.b42_error_signal), 0.0)
        self.assertGreaterEqual(float(decision.b42_conflict_signal), 0.0)
        self.assertGreater(float(decision.b42_performance_context), 0.0)
        self.assertGreater(float(decision.b42_monitor_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b43_adaptive_precision_uses_b42_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b42_error_monitor_source(tmpdir)
            config = build_b43_adaptive_precision_config(
                B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=69, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(43, 48):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B43_ADAPTIVE_PRECISION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B43_ADAPTIVE_PRECISION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b43_controller_profile, "adaptive_precision")
        self.assertIn(
            decision.b43_decision,
            {"adaptive_precision_commit", "continue_precision_lock"},
        )
        self.assertGreaterEqual(int(decision.b43_precision_lock), 1)
        self.assertGreater(float(decision.b43_precision_signal), 0.0)
        self.assertGreater(float(decision.b43_adaptive_threshold), 0.0)
        self.assertGreaterEqual(float(decision.b43_arousal_context), 0.0)
        self.assertGreater(float(decision.b43_control_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b44_thalamic_relay_uses_b43_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b43_adaptive_precision_source(tmpdir)
            config = build_b44_thalamic_relay_config(
                B44_THALAMIC_RELAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=70, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(44, 49):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B44_THALAMIC_RELAY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B44_THALAMIC_RELAY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b44_controller_profile, "thalamic_relay")
        self.assertIn(
            decision.b44_decision,
            {"thalamic_relay_commit", "continue_relay_lock"},
        )
        self.assertGreaterEqual(int(decision.b44_relay_lock), 1)
        self.assertGreater(float(decision.b44_relay_gate), 0.0)
        self.assertGreater(float(decision.b44_sensory_precision), 0.0)
        self.assertGreater(float(decision.b44_context_relay), 0.0)
        self.assertGreater(float(decision.b44_gate_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b45_reticular_inhibition_uses_b44_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b44_thalamic_relay_source(tmpdir)
            config = build_b45_reticular_inhibition_config(
                B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=71, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(45, 50):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B45_RETICULAR_INHIBITION_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B45_RETICULAR_INHIBITION_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b45_controller_profile, "reticular_inhibition")
        self.assertIn(
            decision.b45_decision,
            {"reticular_inhibition_commit", "continue_inhibition_lock"},
        )
        self.assertGreaterEqual(int(decision.b45_inhibition_lock), 1)
        self.assertGreater(float(decision.b45_inhibitory_gate), 0.0)
        self.assertGreater(float(decision.b45_sensory_filter), 0.0)
        self.assertGreater(float(decision.b45_context_suppression), 0.0)
        self.assertGreater(float(decision.b45_loop_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b46_corticothalamic_feedback_uses_b45_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b45_reticular_inhibition_source(tmpdir)
            config = build_b46_corticothalamic_feedback_config(
                B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=72, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(46, 51):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B46_CORTICOTHALAMIC_FEEDBACK_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B46_CORTICOTHALAMIC_FEEDBACK_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b46_controller_profile, "corticothalamic_feedback")
        self.assertIn(
            decision.b46_decision,
            {"corticothalamic_feedback_commit", "continue_feedback_lock"},
        )
        self.assertGreaterEqual(int(decision.b46_feedback_lock), 1)
        self.assertGreater(float(decision.b46_feedback_gain), 0.0)
        self.assertGreater(float(decision.b46_topdown_context), 0.0)
        self.assertGreater(float(decision.b46_prediction_match), 0.0)
        self.assertGreater(float(decision.b46_feedback_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b47_oscillatory_synchrony_uses_b46_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b46_corticothalamic_feedback_source(tmpdir)
            config = build_b47_oscillatory_synchrony_config(
                B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=73, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(47, 53):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B47_OSCILLATORY_SYNCHRONY_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B47_OSCILLATORY_SYNCHRONY_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b47_controller_profile, "oscillatory_synchrony")
        self.assertIn(
            decision.b47_decision,
            {"oscillatory_synchrony_commit", "continue_phase_lock"},
        )
        self.assertGreaterEqual(int(decision.b47_phase_lock), 1)
        self.assertGreater(float(decision.b47_phase_alignment), 0.0)
        self.assertGreater(float(decision.b47_synchrony_gain), 0.0)
        self.assertGreater(float(decision.b47_cross_loop_coherence), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b48_cerebellar_timing_uses_b47_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b47_oscillatory_synchrony_source(tmpdir)
            config = build_b48_cerebellar_timing_config(
                B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=74, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(48, 54):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B48_CEREBELLAR_TIMING_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B48_CEREBELLAR_TIMING_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b48_controller_profile, "cerebellar_timing")
        self.assertIn(
            decision.b48_decision,
            {"cerebellar_timing_commit", "continue_calibration_lock"},
        )
        self.assertGreaterEqual(int(decision.b48_calibration_lock), 1)
        self.assertGreater(float(decision.b48_timing_error), 0.0)
        self.assertGreater(float(decision.b48_predictive_timing), 0.0)
        self.assertGreater(float(decision.b48_corrective_gain), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b49_striatal_action_gate_uses_b48_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b48_cerebellar_timing_source(tmpdir)
            config = build_b49_striatal_action_gate_config(
                B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=75, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(49, 55):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B49_STRIATAL_ACTION_GATE_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B49_STRIATAL_ACTION_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b49_controller_profile, "striatal_action_gate")
        self.assertIn(
            decision.b49_decision,
            {"striatal_gate_commit", "continue_selection_lock"},
        )
        self.assertGreaterEqual(int(decision.b49_selection_lock), 1)
        self.assertGreater(float(decision.b49_go_signal), 0.0)
        self.assertGreaterEqual(float(decision.b49_no_go_signal), 0.0)
        self.assertGreater(float(decision.b49_action_gate_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b50_habit_chunking_uses_b49_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b49_striatal_action_gate_source(tmpdir)
            config = build_b50_habit_chunking_config(
                B50_HABIT_CHUNKING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=76, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(50, 56):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B50_HABIT_CHUNKING_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B50_HABIT_CHUNKING_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b50_controller_profile, "habit_chunking")
        self.assertIn(
            decision.b50_decision,
            {"habit_chunk_commit", "continue_habit_chunk"},
        )
        self.assertGreaterEqual(int(decision.b50_chunk_lock), 1)
        self.assertGreater(float(decision.b50_habit_strength), 0.0)
        self.assertGreater(float(decision.b50_chunk_value), 0.0)
        self.assertGreater(float(decision.b50_habit_stability), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b51_dopaminergic_habit_uses_b50_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b50_habit_chunking_source(tmpdir)
            config = build_b51_dopaminergic_habit_modulation_config(
                B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=77, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(51, 57):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B51_DOPAMINERGIC_HABIT_MODULATION_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B51_DOPAMINERGIC_HABIT_MODULATION_SELECTION_SOURCE,
        )
        self.assertEqual(
            decision.b51_controller_profile,
            "dopaminergic_habit_modulation",
        )
        self.assertIn(
            decision.b51_decision,
            {"dopamine_habit_commit", "continue_dopamine_modulation"},
        )
        self.assertGreaterEqual(int(decision.b51_modulation_lock), 1)
        self.assertGreater(float(decision.b51_prediction_error), 0.0)
        self.assertGreater(float(decision.b51_dopamine_gain), 0.0)
        self.assertGreater(float(decision.b51_habit_modulation), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b52_cholinergic_precision_uses_b51_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b51_dopaminergic_habit_source(tmpdir)
            config = build_b52_cholinergic_precision_gate_config(
                B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=78, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(52, 58):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B52_CHOLINERGIC_PRECISION_GATE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B52_CHOLINERGIC_PRECISION_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b52_controller_profile, "cholinergic_precision_gate")
        self.assertIn(
            decision.b52_decision,
            {"cholinergic_precision_commit", "continue_precision_attention"},
        )
        self.assertGreaterEqual(int(decision.b52_attention_lock), 1)
        self.assertGreater(float(decision.b52_acetylcholine_level), 0.0)
        self.assertGreater(float(decision.b52_precision_gain), 0.0)
        self.assertGreater(float(decision.b52_uncertainty_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b53_noradrenergic_arousal_uses_b52_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b52_cholinergic_precision_source(tmpdir)
            config = build_b53_noradrenergic_arousal_gain_config(
                B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=79, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(53, 59):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B53_NORADRENERGIC_AROUSAL_GAIN_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B53_NORADRENERGIC_AROUSAL_GAIN_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b53_controller_profile, "noradrenergic_arousal_gain")
        self.assertIn(
            decision.b53_decision,
            {"noradrenergic_arousal_commit", "continue_arousal_gain"},
        )
        self.assertGreaterEqual(int(decision.b53_gain_lock), 1)
        self.assertGreater(float(decision.b53_norepinephrine_level), 0.0)
        self.assertGreater(float(decision.b53_arousal_gain), 0.0)
        self.assertGreater(float(decision.b53_surprise_signal), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b54_serotonergic_patience_uses_b53_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b53_noradrenergic_arousal_source(tmpdir)
            config = build_b54_serotonergic_patience_gate_config(
                B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=80, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(54, 61):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B54_SEROTONERGIC_PATIENCE_GATE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B54_SEROTONERGIC_PATIENCE_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b54_controller_profile, "serotonergic_patience_gate")
        self.assertIn(
            decision.b54_decision,
            {"serotonergic_patience_commit", "continue_patience_lock"},
        )
        self.assertGreaterEqual(int(decision.b54_patience_lock), 1)
        self.assertGreater(float(decision.b54_serotonin_level), 0.0)
        self.assertGreater(float(decision.b54_patience_signal), 0.0)
        self.assertGreaterEqual(float(decision.b54_impulse_suppression), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b55_hypothalamic_drive_uses_b54_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b54_serotonergic_patience_source(tmpdir)
            config = build_b55_hypothalamic_drive_coupling_config(
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=81, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(55, 63):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B55_HYPOTHALAMIC_DRIVE_COUPLING_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B55_HYPOTHALAMIC_DRIVE_COUPLING_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b55_controller_profile, "hypothalamic_drive_coupling")
        self.assertIn(
            decision.b55_decision,
            {"hypothalamic_drive_commit", "continue_drive_lock"},
        )
        self.assertGreaterEqual(int(decision.b55_drive_lock), 1)
        self.assertGreater(float(decision.b55_hypothalamic_drive), 0.0)
        self.assertGreater(float(decision.b55_satiety_signal), 0.0)
        self.assertGreater(float(decision.b55_recovery_bias), 0.0)
        self.assertNotEqual(float(decision.b55_drive_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b56_hpa_stress_axis_uses_b55_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b55_hypothalamic_drive_source(tmpdir)
            config = build_b56_hpa_stress_axis_config(
                B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=82, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(56, 65):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B56_HPA_STRESS_AXIS_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B56_HPA_STRESS_AXIS_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b56_controller_profile, "hpa_stress_axis")
        self.assertIn(
            decision.b56_decision,
            {"hpa_stress_axis_commit", "continue_stress_axis_lock"},
        )
        self.assertGreaterEqual(int(decision.b56_stress_lock), 1)
        self.assertGreater(float(decision.b56_cortisol_level), 0.0)
        self.assertGreater(float(decision.b56_stress_load), 0.0)
        self.assertGreater(float(decision.b56_recovery_signal), 0.0)
        self.assertNotEqual(float(decision.b56_endocrine_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b57_insular_interoception_uses_b56_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b56_hpa_stress_source(tmpdir)
            config = build_b57_insular_interoceptive_awareness_config(
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=83, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(57, 67):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_SELECTION_SOURCE,
        )
        self.assertEqual(
            decision.b57_controller_profile,
            "insular_interoceptive_awareness",
        )
        self.assertIn(
            decision.b57_decision,
            {"interoceptive_awareness_commit", "continue_awareness_lock"},
        )
        self.assertGreaterEqual(int(decision.b57_awareness_lock), 1)
        self.assertGreater(float(decision.b57_interoceptive_awareness), 0.0)
        self.assertGreater(float(decision.b57_visceral_salience), 0.0)
        self.assertGreater(float(decision.b57_body_state_confidence), 0.0)
        self.assertNotEqual(float(decision.b57_awareness_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b58_acc_conflict_monitor_uses_b57_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b57_insular_interoceptive_source(tmpdir)
            config = build_b58_acc_conflict_monitor_config(
                B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=84, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(58, 69):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(decision.b_effective_level, B58_ACC_CONFLICT_MONITOR_EFFECTIVE_LEVEL)
        self.assertEqual(
            decision.semantic_action_source,
            B58_ACC_CONFLICT_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b58_controller_profile, "acc_conflict_monitor")
        self.assertIn(
            decision.b58_decision,
            {"acc_conflict_commit", "continue_conflict_lock"},
        )
        self.assertGreaterEqual(int(decision.b58_conflict_lock), 1)
        self.assertGreater(float(decision.b58_conflict_signal), 0.0)
        self.assertGreater(float(decision.b58_error_likelihood), 0.0)
        self.assertGreater(float(decision.b58_control_allocation), 0.0)
        self.assertNotEqual(float(decision.b58_resolution_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b59_prefrontal_goal_context_uses_b58_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b58_acc_conflict_source(tmpdir)
            config = build_b59_prefrontal_goal_context_config(
                B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=85, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(59, 71):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B59_PREFRONTAL_GOAL_CONTEXT_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B59_PREFRONTAL_GOAL_CONTEXT_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b59_controller_profile, "prefrontal_goal_context")
        self.assertIn(
            decision.b59_decision,
            {"prefrontal_goal_commit", "continue_executive_lock"},
        )
        self.assertGreaterEqual(int(decision.b59_executive_lock), 1)
        self.assertGreater(float(decision.b59_goal_context), 0.0)
        self.assertGreater(float(decision.b59_working_set_stability), 0.0)
        self.assertGreater(float(decision.b59_task_set_confidence), 0.0)
        self.assertNotEqual(float(decision.b59_executive_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b60_orbitofrontal_value_uses_b59_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b59_prefrontal_goal_source(tmpdir)
            config = build_b60_orbitofrontal_outcome_value_config(
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=86, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(60, 73):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B60_ORBITOFRONTAL_OUTCOME_VALUE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B60_ORBITOFRONTAL_OUTCOME_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b60_controller_profile, "orbitofrontal_outcome_value")
        self.assertIn(
            decision.b60_decision,
            {"orbitofrontal_value_commit", "continue_value_lock"},
        )
        self.assertGreaterEqual(int(decision.b60_value_lock), 1)
        self.assertNotEqual(float(decision.b60_outcome_value), 0.0)
        self.assertGreaterEqual(float(decision.b60_reversal_signal), 0.0)
        self.assertGreater(float(decision.b60_goal_value_confidence), 0.0)
        self.assertNotEqual(float(decision.b60_value_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b61_amygdala_safety_uses_b60_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b60_orbitofrontal_value_source(tmpdir)
            config = build_b61_amygdala_safety_value_config(
                B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=87, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 8.0
        meta["shelter_dist"] = 2.0
        meta["shelter_role"] = "outside"
        for tick in range(61, 75):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.90},
                    sleep={"health": 0.80, "sleep_debt": 0.10, "on_shelter": 0.0},
                    threat={"predator_smell_strength": 0.0},
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B61_AMYGDALA_SAFETY_VALUE_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B61_AMYGDALA_SAFETY_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b61_controller_profile, "amygdala_safety_value")
        self.assertIn(
            decision.b61_decision,
            {"amygdala_safety_commit", "continue_safety_lock"},
        )
        self.assertGreaterEqual(int(decision.b61_safety_lock), 1)
        self.assertGreater(float(decision.b61_safety_value), 0.0)
        self.assertGreaterEqual(float(decision.b61_threat_value), 0.0)
        self.assertGreater(float(decision.b61_safety_confidence), 0.0)
        self.assertNotEqual(float(decision.b61_affective_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b62_defensive_mode_uses_b61_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b61_amygdala_safety_source(tmpdir)
            config = build_b62_defensive_mode_selector_config(
                B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=88, module_dropout=0.0, config=config)

        meta = dict(_bridge_observation()["meta"])
        meta["map_template"] = "corridor_escape"
        meta["food_dist"] = 9.0
        meta["shelter_dist"] = 1.0
        meta["shelter_role"] = "outside"
        for tick in range(62, 76):
            brain.set_direct_policy_event_clock(tick)
            decision = brain.act_inference(
                _brain_observation(
                    meta,
                    hunger={"hunger": 0.82},
                    sleep={"health": 0.60, "sleep_debt": 0.25, "on_shelter": 0.0},
                    threat={
                        "predator_smell_strength": 0.30,
                        "predator_motion_salience": 0.20,
                    },
                ),
                sample=False,
            )

        self.assertEqual(
            decision.b_effective_level,
            B62_DEFENSIVE_MODE_SELECTOR_EFFECTIVE_LEVEL,
        )
        self.assertEqual(
            decision.semantic_action_source,
            B62_DEFENSIVE_MODE_SELECTOR_SELECTION_SOURCE,
        )
        self.assertEqual(decision.b62_controller_profile, "defensive_mode_selector")
        self.assertIn(
            decision.b62_decision,
            {"defensive_flee_to_shelter", "continue_defense_lock"},
        )
        self.assertIn(decision.b62_defensive_mode, {"flee_to_shelter", "continue_defense_lock"})
        self.assertGreaterEqual(int(decision.b62_defense_lock), 1)
        self.assertGreater(float(decision.b62_flee_pressure), 0.0)
        self.assertGreater(float(decision.b62_shelter_bias), 0.0)
        self.assertNotEqual(float(decision.b62_defense_balance), 0.0)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b2_temporal_threat_uses_transfer_memory_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b1_threat_guard_source(tmpdir)
            config = build_b2_temporal_threat_config(
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=18, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["STAY"]
        ] = 10.0
        meta = dict(_bridge_observation()["meta"])
        meta["on_shelter"] = False
        meta["shelter_role"] = "outside"
        meta["memory_vectors"] = {
            "predator": {"dx": 0.4, "dy": 0.0, "age": 0.1, "ttl": 10}
        }

        decision = brain.act_inference(
            _brain_observation(
                meta,
                hunger={"hunger": 0.45},
                sleep={"health": 1.0, "on_shelter": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL)
        self.assertEqual(decision.learned_semantic_action, "STAY")
        self.assertEqual(decision.semantic_action, "MOVE_TO_SHELTER")
        self.assertEqual(
            decision.semantic_action_source,
            B2_TEMPORAL_THREAT_SELECTION_SOURCE,
        )
        self.assertGreaterEqual(float(decision.b_temporal_threat_pressure), 0.70)
        self.assertEqual(decision.semantic_override_count, 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b3_contact_memory_uses_transfer_memory_and_primitive_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            config = build_b3_contact_memory_config(
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=19, module_dropout=0.0, config=config)

        assert brain.b_series_policy is not None
        brain.b_series_policy.b2_policy[:] = -10.0
        brain.b_series_policy.b2_policy[
            B_SEMANTIC_ACTION_TO_INDEX["STAY"]
        ] = 10.0
        brain.set_direct_policy_event_clock(5)

        decision = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.45},
                sleep={"health": 0.70, "on_shelter": 0.0},
                threat={
                    "recent_contact": 1.0,
                    "recent_pain": 0.5,
                    "predator_smell_strength": 0.6,
                },
            ),
            sample=False,
        )

        self.assertEqual(decision.b_effective_level, B3_CONTACT_MEMORY_EFFECTIVE_LEVEL)
        self.assertEqual(decision.learned_semantic_action, "STAY")
        self.assertEqual(decision.semantic_action, "MOVE_TO_SHELTER")
        self.assertEqual(
            decision.semantic_action_source,
            B3_CONTACT_MEMORY_SELECTION_SOURCE,
        )
        self.assertGreater(int(decision.b3_contact_cooldown), 0)
        self.assertEqual(decision.b3_controller_profile, "standard")
        self.assertEqual(decision.semantic_override_count, 1)
        self.assertIn(decision.bridge_primitive_action, ACTIONS)

    def test_b3_contact_memory_cooldowns_reset_on_episode_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            config = build_b3_contact_memory_config(
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            brain = SpiderBrain(seed=20, module_dropout=0.0, config=config)

        brain.set_direct_policy_event_clock(8)
        first = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.45},
                sleep={"health": 0.70, "on_shelter": 0.0},
                threat={"recent_contact": 1.0},
            ),
            sample=False,
        )
        self.assertGreater(int(first.b3_contact_cooldown), 0)

        brain.set_direct_policy_event_clock(0)
        second = brain.act_inference(
            _brain_observation(
                hunger={"hunger": 0.45, "on_food": 0.0},
                sleep={"health": 1.0, "on_shelter": 0.0},
                threat={"recent_contact": 0.0, "recent_pain": 0.0},
            ),
            sample=False,
        )

        self.assertEqual(int(second.b3_contact_cooldown), 0)
        self.assertEqual(int(second.b3_post_food_cooldown), 0)


class BSeriesCheckpointTest(unittest.TestCase):
    def test_checkpoint_save_load_preserves_b_series_metadata_and_weights(self) -> None:
        config = _b0_config()
        source = SpiderBrain(seed=21, module_dropout=0.0, config=config)
        target = SpiderBrain(seed=22, module_dropout=0.0, config=config)
        assert source.b_series_policy is not None
        assert target.b_series_policy is not None
        source.b_series_policy.b2_policy[:] = np.arange(
            len(B_SEMANTIC_ACTIONS),
            dtype=float,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            loaded = target.load(checkpoint)
            metadata = json.loads((checkpoint / "metadata.json").read_text())

        self.assertIn("b_series_policy", loaded)
        self.assertEqual(
            metadata["modules"]["b_series_policy"]["semantic_actions"],
            list(B_SEMANTIC_ACTIONS),
        )
        self.assertEqual(metadata["modules"]["b_series_policy"]["b_level"], 0)
        self.assertEqual(metadata["modules"]["b_series_policy"]["b_mode"], "current_bridge")
        np.testing.assert_allclose(
            source.b_series_policy.b2_policy,
            target.b_series_policy.b2_policy,
        )

    def test_b1_partial_transfer_reports_coverage(self) -> None:
        source_config = _b0_config()
        source = SpiderBrain(seed=31, module_dropout=0.0, config=source_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            b1_config = BrainAblationConfig(
                name="b1_transfer_smoke",
                architecture="b_series",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                use_learned_arbitration=False,
                enable_food_direction_bias=False,
                warm_start_scale=0.0,
                credit_strategy="broadcast",
                disabled_modules=(),
                reflex_scale=0.0,
                module_reflex_scales={},
                b_level=1,
                b_mode="current_bridge",
                b_parent_level=0,
                b_hidden_dim=40,
                b_transfer_source_checkpoint=str(checkpoint),
            )
            target = SpiderBrain(seed=32, module_dropout=0.0, config=b1_config)

        report = target.b_series_transfer_report
        self.assertIsNotNone(report)
        assert report is not None
        self.assertGreaterEqual(float(report["coverage"]), 0.50)
        self.assertEqual(report["target_b_level"], 1)
        self.assertEqual(report["parent_level"], 0)
        self.assertIn("W1", report["partially_loaded_keys"])

    def test_b1_requires_b0_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B1_CAPACITY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b0-current",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=33, module_dropout=0.0, config=config)

    def test_b1_h48_transfer_report_records_source_and_coverage(self) -> None:
        source_config = _b0_config()
        source = SpiderBrain(seed=34, module_dropout=0.0, config=source_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            config = build_b1_capacity_config(
                B1_CAPACITY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            target = SpiderBrain(seed=35, module_dropout=0.0, config=config)

        report = target.b_series_transfer_report
        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report["source_checkpoint"], str(checkpoint))
        self.assertEqual(report["target_b_level"], 1)
        self.assertEqual(report["parent_level"], 0)
        self.assertGreaterEqual(float(report["coverage"]), 0.50)
        self.assertLess(float(report["coverage"]), 1.0)
        self.assertFalse(report["allow_low_coverage"])

    def test_b2_requires_b1_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B2_TEMPORAL_THREAT_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b1-threat-guard",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=38, module_dropout=0.0, config=config)

    def test_b3_requires_b2_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B3_CONTACT_MEMORY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b2-temporal-threat",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=41, module_dropout=0.0, config=config)

    def test_b4_requires_b3_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B4_RECOVERY_BALANCE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b3-recurrent-guard",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=45, module_dropout=0.0, config=config)

    def test_b5_requires_b4_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b4-genetic-recovery",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=46, module_dropout=0.0, config=config)

    def test_b6_requires_b5_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b5-genetic-homeostasis",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=47, module_dropout=0.0, config=config)

    def test_b7_requires_b6_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B7_AFFORDANCE_BUDGET_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b6-fused-risk-recurrent",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=48, module_dropout=0.0, config=config)

    def test_b8_requires_b7_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b7-affordance-budget",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=49, module_dropout=0.0, config=config)

    def test_b9_requires_b8_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B9_WAYPOINT_PLANNER_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b8-spatial-affordance",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=50, module_dropout=0.0, config=config)

    def test_b10_requires_b9_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b9-waypoint-planner",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=51, module_dropout=0.0, config=config)

    def test_b11_requires_b10_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B11_CONFIDENCE_ARBITER_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b10-prospective-replay",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=52, module_dropout=0.0, config=config)

    def test_b12_requires_b11_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b11-confidence-arbiter",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=53, module_dropout=0.0, config=config)

    def test_b13_requires_b12_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b12-predictive-attention",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=54, module_dropout=0.0, config=config)

    def test_b14_requires_b13_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b13-local-search",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=55, module_dropout=0.0, config=config)

    def test_b15_requires_b14_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B15_OPTION_CRITIC_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b14-affordance-uncertainty",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=56, module_dropout=0.0, config=config)

    def test_b16_requires_b15_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B16_OPTION_ENSEMBLE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b15-option-critic",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=57, module_dropout=0.0, config=config)

    def test_b17_requires_b16_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b16-option-ensemble",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=58, module_dropout=0.0, config=config)

    def test_b18_requires_b17_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B18_ELIGIBILITY_TRACE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b17-neuromodulated",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=59, module_dropout=0.0, config=config)

    def test_b19_requires_b18_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B19_EPISODIC_META_MEMORY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b18-eligibility",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=60, module_dropout=0.0, config=config)

    def test_b20_requires_b19_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B20_WORKING_MEMORY_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b19-meta-memory",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=61, module_dropout=0.0, config=config)

    def test_b21_requires_b20_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b20-working-memory",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=62, module_dropout=0.0, config=config)

    def test_b22_requires_b21_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b21-replay",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=63, module_dropout=0.0, config=config)

    def test_b23_requires_b22_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B23_CONFLICT_MONITOR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b22-prospective",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=64, module_dropout=0.0, config=config)

    def test_b24_requires_b23_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B24_PRECISION_CONFLICT_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b23-conflict",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=65, module_dropout=0.0, config=config)

    def test_b25_requires_b24_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b24-precision",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=66, module_dropout=0.0, config=config)

    def test_b26_requires_b25_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b25-metacog",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=67, module_dropout=0.0, config=config)

    def test_b27_requires_b26_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B27_AROUSAL_GAIN_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b26-allostasis",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=68, module_dropout=0.0, config=config)

    def test_b28_requires_b27_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b27-arousal",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=69, module_dropout=0.0, config=config)

    def test_b29_requires_b28_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B29_SALIENCE_COMPETITION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b28-attention",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=70, module_dropout=0.0, config=config)

    def test_b30_requires_b29_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b29-salience",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=71, module_dropout=0.0, config=config)

    def test_b31_requires_b30_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b30-gate",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=72, module_dropout=0.0, config=config)

    def test_b32_requires_b31_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b31-dopamine",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=73, module_dropout=0.0, config=config)

    def test_b33_requires_b32_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b32-value",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=74, module_dropout=0.0, config=config)

    def test_b34_requires_b33_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b33-td",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=75, module_dropout=0.0, config=config)

    def test_b35_requires_b34_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b34-eligibility",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=76, module_dropout=0.0, config=config)

    def test_b36_requires_b35_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B36_LATENT_BELIEF_STATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b35-forward-model",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=77, module_dropout=0.0, config=config)

    def test_b37_requires_b36_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B37_STATE_FACTOR_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b36-belief-state",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=78, module_dropout=0.0, config=config)

    def test_b38_requires_b37_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B38_FACTOR_ATTENTION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b37-state-factor",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=79, module_dropout=0.0, config=config)

    def test_b39_requires_b38_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B39_ATTENTION_BINDING_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b38-factor-attention",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=80, module_dropout=0.0, config=config)

    def test_b40_requires_b39_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B40_GLOBAL_WORKSPACE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b39-attention-binding",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=81, module_dropout=0.0, config=config)

    def test_b41_requires_b40_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b40-global-workspace",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=82, module_dropout=0.0, config=config)

    def test_b42_requires_b41_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B42_ERROR_MONITOR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b41-executive-workspace",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=83, module_dropout=0.0, config=config)

    def test_b43_requires_b42_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B43_ADAPTIVE_PRECISION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b42-error-monitor",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=84, module_dropout=0.0, config=config)

    def test_b44_requires_b43_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B44_THALAMIC_RELAY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b43-adaptive-precision",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=85, module_dropout=0.0, config=config)

    def test_b45_requires_b44_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B45_RETICULAR_INHIBITION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b44-thalamic-relay",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=86, module_dropout=0.0, config=config)

    def test_b46_requires_b45_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b45-reticular-inhibition",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=87, module_dropout=0.0, config=config)

    def test_b47_requires_b46_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b46-corticothalamic-feedback",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=88, module_dropout=0.0, config=config)

    def test_b48_requires_b47_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B48_CEREBELLAR_TIMING_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b47-oscillatory-synchrony",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=89, module_dropout=0.0, config=config)

    def test_b49_requires_b48_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b48-cerebellar-timing",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=90, module_dropout=0.0, config=config)

    def test_b50_requires_b49_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B50_HABIT_CHUNKING_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b49-striatal-gate",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=91, module_dropout=0.0, config=config)

    def test_b51_requires_b50_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b50-habit-chunking",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=92, module_dropout=0.0, config=config)

    def test_b52_requires_b51_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b51-dopaminergic-habit",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=93, module_dropout=0.0, config=config)

    def test_b53_requires_b52_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b52-cholinergic-precision",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=94, module_dropout=0.0, config=config)

    def test_b54_requires_b53_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b53-noradrenergic-arousal",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=95, module_dropout=0.0, config=config)

    def test_b55_requires_b54_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b54-serotonergic-patience",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=96, module_dropout=0.0, config=config)

    def test_b56_requires_b55_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B56_HPA_STRESS_AXIS_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b55-hypothalamic-drive",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=97, module_dropout=0.0, config=config)

    def test_b57_requires_b56_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b56-hpa-stress-axis",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=98, module_dropout=0.0, config=config)

    def test_b58_requires_b57_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b57-insular-interoception",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=99, module_dropout=0.0, config=config)

    def test_b59_requires_b58_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b58-acc-conflict",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=100, module_dropout=0.0, config=config)

    def test_b60_requires_b59_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b59-prefrontal-goal",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=101, module_dropout=0.0, config=config)

    def test_b61_requires_b60_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b60-orbitofrontal-value",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=102, module_dropout=0.0, config=config)

    def test_b62_requires_b61_checkpoint_source(self) -> None:
        config = replace(
            resolve_ablation_configs([B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME])[0],
            b_transfer_source_checkpoint="/tmp/does-not-exist-b61-amygdala-safety",
        )

        with self.assertRaises(FileNotFoundError):
            SpiderBrain(seed=103, module_dropout=0.0, config=config)

    def test_b2_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b1_threat_guard_source(tmpdir)
            variants = (
                (B2_TEMPORAL_THREAT_H48_POLICY_NAME, 1.0),
                (B2_TEMPORAL_THREAT_H56_POLICY_NAME, 0.85),
                (B2_TEMPORAL_THREAT_H64_POLICY_NAME, 0.75),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b2_temporal_threat_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=39 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 2)
                self.assertEqual(report["parent_level"], 1)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b3_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            variants = (
                (B3_CONTACT_MEMORY_H48_POLICY_NAME, 1.0),
                (B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME, 1.0),
                (B3_CONTACT_MEMORY_H56_POLICY_NAME, 0.85),
                (B3_RECURRENT_GUARD_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b3_contact_memory_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=42 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 3)
                self.assertEqual(report["parent_level"], 2)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b4_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b3_recurrent_guard_source(tmpdir)
            variants = (
                (B4_RECOVERY_BALANCE_H48_POLICY_NAME, 1.0),
                (B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME, 1.0),
                (B4_RECOVERY_BALANCE_H56_POLICY_NAME, 0.85),
                (B4_GENETIC_RECOVERY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b4_recovery_balance_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=52 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 4)
                self.assertEqual(report["parent_level"], 3)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b5_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            variants = (
                (B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME, 1.0),
                (B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME, 1.0),
                (B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME, 0.85),
                (B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b5_homeostatic_arbiter_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=56 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 5)
                self.assertEqual(report["parent_level"], 4)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b6_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            variants = (
                (B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME, 1.0),
                (B6_RECURRENT_CONTEXT_H48_POLICY_NAME, 1.0),
                (B6_RISK_CORRIDOR_H56_POLICY_NAME, 0.85),
                (B6_RECURRENT_CONTEXT_H56_POLICY_NAME, 0.85),
                (B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b6_risk_corridor_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=60 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 6)
                self.assertEqual(report["parent_level"], 5)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b7_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b6_fused_risk_recurrent_source(tmpdir)
            variants = (
                (B7_AFFORDANCE_BUDGET_H48_POLICY_NAME, 1.0),
                (B7_ENERGY_BUDGET_CORRIDOR_H48_POLICY_NAME, 1.0),
                (B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME, 1.0),
                (B7_AFFORDANCE_BUDGET_H56_POLICY_NAME, 0.85),
                (B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b7_affordance_budget_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=66 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 7)
                self.assertEqual(report["parent_level"], 6)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b8_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b7_affordance_budget_source(tmpdir)
            variants = (
                (B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME, 1.0),
                (B8_RETURN_VECTOR_H48_POLICY_NAME, 1.0),
                (B8_CORRIDOR_PLACE_MEMORY_H48_POLICY_NAME, 1.0),
                (B8_SPATIAL_AFFORDANCE_MAP_H56_POLICY_NAME, 0.85),
                (B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b8_spatial_affordance_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=72 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 8)
                self.assertEqual(report["parent_level"], 7)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b9_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b8_spatial_affordance_source(tmpdir)
            variants = (
                (B9_WAYPOINT_PLANNER_H48_POLICY_NAME, 1.0),
                (B9_PATH_INTEGRATION_H48_POLICY_NAME, 1.0),
                (B9_ROUTE_MEMORY_H48_POLICY_NAME, 1.0),
                (B9_WAYPOINT_PLANNER_H56_POLICY_NAME, 0.85),
                (B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b9_waypoint_planner_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=78 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 9)
                self.assertEqual(report["parent_level"], 8)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b10_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b9_waypoint_planner_source(tmpdir)
            variants = (
                (B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME, 1.0),
                (B10_VALUE_ROUTE_EVALUATOR_H48_POLICY_NAME, 1.0),
                (B10_REPLAY_PLANNER_H48_POLICY_NAME, 1.0),
                (B10_PROSPECTIVE_REPLAY_H56_POLICY_NAME, 0.85),
                (B10_GENETIC_REPLAY_PLANNER_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b10_prospective_replay_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=84 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 10)
                self.assertEqual(report["parent_level"], 9)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b11_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b10_prospective_replay_source(tmpdir)
            variants = (
                (B11_CONFIDENCE_ARBITER_H48_POLICY_NAME, 1.0),
                (B11_UNCERTAINTY_GATE_H48_POLICY_NAME, 1.0),
                (B11_NEUROMODULATED_REPLAY_H48_POLICY_NAME, 1.0),
                (B11_CONFIDENCE_ARBITER_H56_POLICY_NAME, 0.85),
                (B11_GENETIC_CONFIDENCE_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b11_confidence_arbiter_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=90 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 11)
                self.assertEqual(report["parent_level"], 10)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b12_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b11_confidence_arbiter_source(tmpdir)
            variants = (
                (B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B12_ACTIVE_INFERENCE_GATE_H48_POLICY_NAME, 1.0),
                (B12_AFFORDANCE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B12_PREDICTIVE_ATTENTION_H56_POLICY_NAME, 0.85),
                (B12_GENETIC_ATTENTION_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b12_predictive_attention_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=96 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 12)
                self.assertEqual(report["parent_level"], 11)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b13_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b12_predictive_attention_source(tmpdir)
            variants = (
                (B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME, 1.0),
                (B13_COUNTERFACTUAL_ROUTE_H48_POLICY_NAME, 1.0),
                (B13_AFFORDANCE_SAMPLER_H48_POLICY_NAME, 1.0),
                (B13_LOCAL_AFFORDANCE_SEARCH_H56_POLICY_NAME, 0.85),
                (B13_GENETIC_LOCAL_SEARCH_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b13_local_affordance_search_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=101 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 13)
                self.assertEqual(report["parent_level"], 12)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b14_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b13_local_affordance_search_source(tmpdir)
            variants = (
                (B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME, 1.0),
                (B14_RISK_CALIBRATED_SEARCH_H48_POLICY_NAME, 1.0),
                (B14_CONFIDENCE_WEIGHTED_ROUTE_H48_POLICY_NAME, 1.0),
                (B14_AFFORDANCE_UNCERTAINTY_H56_POLICY_NAME, 0.85),
                (B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b14_affordance_uncertainty_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=106 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 14)
                self.assertEqual(report["parent_level"], 13)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b15_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b14_affordance_uncertainty_source(tmpdir)
            variants = (
                (B15_OPTION_CRITIC_H48_POLICY_NAME, 1.0),
                (B15_PERSISTENCE_GATE_H48_POLICY_NAME, 1.0),
                (B15_VALUE_GATED_OPTION_H48_POLICY_NAME, 1.0),
                (B15_OPTION_CRITIC_H56_POLICY_NAME, 0.85),
                (B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b15_option_critic_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=111 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 15)
                self.assertEqual(report["parent_level"], 14)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b16_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b15_option_critic_source(tmpdir)
            variants = (
                (B16_OPTION_ENSEMBLE_H48_POLICY_NAME, 1.0),
                (B16_COMPETING_OPTIONS_H48_POLICY_NAME, 1.0),
                (B16_ACTION_SET_VOTER_H48_POLICY_NAME, 1.0),
                (B16_OPTION_ENSEMBLE_H56_POLICY_NAME, 0.85),
                (B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b16_option_ensemble_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=117 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 16)
                self.assertEqual(report["parent_level"], 15)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b17_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b16_option_ensemble_source(tmpdir)
            variants = (
                (B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME, 1.0),
                (B17_AROUSAL_GATED_OPTIONS_H48_POLICY_NAME, 1.0),
                (B17_HOMEOSTATIC_MODULATOR_H48_POLICY_NAME, 1.0),
                (B17_NEUROMODULATED_ENSEMBLE_H56_POLICY_NAME, 0.85),
                (B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b17_neuromodulated_ensemble_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=122 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 17)
                self.assertEqual(report["parent_level"], 16)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b18_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b17_neuromodulated_ensemble_source(tmpdir)
            variants = (
                (B18_ELIGIBILITY_TRACE_H48_POLICY_NAME, 1.0),
                (B18_METASTABLE_AROUSAL_H48_POLICY_NAME, 1.0),
                (B18_SYNAPTIC_TRACE_MODULATOR_H48_POLICY_NAME, 1.0),
                (B18_ELIGIBILITY_TRACE_H56_POLICY_NAME, 0.85),
                (B18_GENETIC_ELIGIBILITY_TRACE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b18_eligibility_trace_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=127 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 18)
                self.assertEqual(report["parent_level"], 17)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b19_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b18_eligibility_trace_source(tmpdir)
            variants = (
                (B19_EPISODIC_META_MEMORY_H48_POLICY_NAME, 1.0),
                (B19_STABILITY_MEMORY_H48_POLICY_NAME, 1.0),
                (B19_SWITCH_SUPPRESSION_H48_POLICY_NAME, 1.0),
                (B19_EPISODIC_META_MEMORY_H56_POLICY_NAME, 0.85),
                (B19_GENETIC_META_MEMORY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b19_episodic_meta_memory_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=132 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 19)
                self.assertEqual(report["parent_level"], 18)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b20_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b19_episodic_meta_memory_source(tmpdir)
            variants = (
                (B20_WORKING_MEMORY_GATE_H48_POLICY_NAME, 1.0),
                (B20_CONTEXT_BINDING_H48_POLICY_NAME, 1.0),
                (B20_STABILITY_BUFFER_H48_POLICY_NAME, 1.0),
                (B20_WORKING_MEMORY_GATE_H56_POLICY_NAME, 0.85),
                (B20_GENETIC_WORKING_MEMORY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b20_working_memory_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=137 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 20)
                self.assertEqual(report["parent_level"], 19)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b21_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b20_working_memory_gate_source(tmpdir)
            variants = (
                (B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME, 1.0),
                (B21_SEQUENCE_BINDING_H48_POLICY_NAME, 1.0),
                (B21_ROUTE_REHEARSAL_H48_POLICY_NAME, 1.0),
                (B21_HIPPOCAMPAL_REPLAY_H56_POLICY_NAME, 0.85),
                (B21_GENETIC_REPLAY_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b21_hippocampal_replay_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=142 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 21)
                self.assertEqual(report["parent_level"], 20)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b22_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b21_hippocampal_replay_source(tmpdir)
            variants = (
                (B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME, 1.0),
                (B22_FORWARD_MODEL_GATE_H48_POLICY_NAME, 1.0),
                (B22_ROUTE_VIABILITY_SIM_H48_POLICY_NAME, 1.0),
                (B22_PROSPECTIVE_MAP_REPLAY_H56_POLICY_NAME, 0.85),
                (B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b22_prospective_replay_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=147 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 22)
                self.assertEqual(report["parent_level"], 21)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b23_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b22_prospective_replay_source(tmpdir)
            variants = (
                (B23_CONFLICT_MONITOR_H48_POLICY_NAME, 1.0),
                (B23_ERROR_GATED_REPLAY_H48_POLICY_NAME, 1.0),
                (B23_ABORT_CONFLICT_ARBITER_H48_POLICY_NAME, 1.0),
                (B23_CONFLICT_MONITOR_H56_POLICY_NAME, 0.85),
                (B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b23_conflict_monitor_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=152 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 23)
                self.assertEqual(report["parent_level"], 22)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b24_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b23_conflict_monitor_source(tmpdir)
            variants = (
                (B24_PRECISION_CONFLICT_H48_POLICY_NAME, 1.0),
                (B24_PREDICTION_PRECISION_GATE_H48_POLICY_NAME, 1.0),
                (B24_RELIABILITY_ABORT_H48_POLICY_NAME, 1.0),
                (B24_PRECISION_CONFLICT_H56_POLICY_NAME, 0.85),
                (B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b24_precision_conflict_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=157 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 24)
                self.assertEqual(report["parent_level"], 23)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b25_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b24_precision_conflict_source(tmpdir)
            variants = (
                (B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME, 1.0),
                (B25_CONFIDENCE_CALIBRATION_H48_POLICY_NAME, 1.0),
                (B25_UNCERTAINTY_INTEGRATOR_H48_POLICY_NAME, 1.0),
                (B25_METACOGNITIVE_CONFIDENCE_H56_POLICY_NAME, 0.85),
                (B25_GENETIC_METACOGNITION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b25_metacognitive_confidence_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=162 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 25)
                self.assertEqual(report["parent_level"], 24)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b26_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b25_metacognitive_confidence_source(tmpdir)
            variants = (
                (B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME, 1.0),
                (B26_SETPOINT_DRIFT_H48_POLICY_NAME, 1.0),
                (B26_ERROR_SUPPRESSION_H48_POLICY_NAME, 1.0),
                (B26_ALLOSTATIC_PREDICTION_H56_POLICY_NAME, 0.85),
                (B26_GENETIC_ALLOSTASIS_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b26_allostatic_prediction_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=167 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 26)
                self.assertEqual(report["parent_level"], 25)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b27_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b26_allostatic_prediction_source(tmpdir)
            variants = (
                (B27_AROUSAL_GAIN_H48_POLICY_NAME, 1.0),
                (B27_STRESS_MODULATION_H48_POLICY_NAME, 1.0),
                (B27_ENERGY_AROUSAL_H48_POLICY_NAME, 1.0),
                (B27_AROUSAL_GAIN_H56_POLICY_NAME, 0.85),
                (B27_GENETIC_AROUSAL_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b27_arousal_gain_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=172 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 27)
                self.assertEqual(report["parent_level"], 26)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b28_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b27_arousal_gain_source(tmpdir)
            variants = (
                (B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B28_THREAT_FOCUS_ATTENTION_H48_POLICY_NAME, 1.0),
                (B28_HOMEOSTATIC_ATTENTION_H48_POLICY_NAME, 1.0),
                (B28_INTEROCEPTIVE_ATTENTION_H56_POLICY_NAME, 0.85),
                (B28_GENETIC_ATTENTION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b28_interoceptive_attention_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=177 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 28)
                self.assertEqual(report["parent_level"], 27)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b29_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b28_interoceptive_attention_source(tmpdir)
            variants = (
                (B29_SALIENCE_COMPETITION_H48_POLICY_NAME, 1.0),
                (B29_THREAT_SALIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B29_HOMEOSTATIC_SALIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B29_SALIENCE_COMPETITION_H56_POLICY_NAME, 0.85),
                (B29_GENETIC_SALIENCE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b29_salience_competition_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=182 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 29)
                self.assertEqual(report["parent_level"], 28)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b30_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b29_salience_competition_source(tmpdir)
            variants = (
                (B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME, 1.0),
                (B30_GO_NOGO_BALANCE_H48_POLICY_NAME, 1.0),
                (B30_THREAT_INHIBITION_GATE_H48_POLICY_NAME, 1.0),
                (B30_BASAL_GANGLIA_GATE_H56_POLICY_NAME, 0.85),
                (B30_GENETIC_ACTION_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b30_basal_ganglia_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=187 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 30)
                self.assertEqual(report["parent_level"], 29)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b31_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b30_basal_ganglia_gate_source(tmpdir)
            variants = (
                (B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME, 1.0),
                (B31_TONIC_DOPAMINE_GATE_H48_POLICY_NAME, 1.0),
                (B31_PHASIC_DOPAMINE_GATE_H48_POLICY_NAME, 1.0),
                (B31_DOPAMINE_PREDICTION_ERROR_H56_POLICY_NAME, 0.85),
                (B31_GENETIC_DOPAMINE_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b31_dopamine_prediction_error_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=192 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 31)
                self.assertEqual(report["parent_level"], 30)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b32_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b31_dopamine_prediction_error_source(tmpdir)
            variants = (
                (B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME, 1.0),
                (B32_ADVANTAGE_VALUE_GATE_H48_POLICY_NAME, 1.0),
                (B32_CRITIC_STABILITY_H48_POLICY_NAME, 1.0),
                (B32_ACTOR_CRITIC_VALUE_H56_POLICY_NAME, 0.85),
                (B32_GENETIC_ACTOR_CRITIC_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b32_actor_critic_value_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=197 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 32)
                self.assertEqual(report["parent_level"], 31)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b33_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b32_actor_critic_value_source(tmpdir)
            variants = (
                (B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME, 1.0),
                (B33_BOOTSTRAPPED_VALUE_GATE_H48_POLICY_NAME, 1.0),
                (B33_REWARD_TRACE_CRITIC_H48_POLICY_NAME, 1.0),
                (B33_TD_ERROR_DECOMPOSITION_H56_POLICY_NAME, 0.85),
                (B33_GENETIC_TD_VALUE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b33_td_error_decomposition_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=202 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 33)
                self.assertEqual(report["parent_level"], 32)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b34_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b33_td_error_decomposition_source(tmpdir)
            variants = (
                (B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME, 1.0),
                (B34_DELAYED_CREDIT_GATE_H48_POLICY_NAME, 1.0),
                (B34_SYNAPTIC_TAGGING_H48_POLICY_NAME, 1.0),
                (B34_ELIGIBILITY_CREDIT_H56_POLICY_NAME, 0.85),
                (B34_GENETIC_ELIGIBILITY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b34_eligibility_credit_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=207 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 34)
                self.assertEqual(report["parent_level"], 33)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b35_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b34_eligibility_credit_source(tmpdir)
            variants = (
                (B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME, 1.0),
                (B35_TRANSITION_ERROR_GATE_H48_POLICY_NAME, 1.0),
                (B35_MODEL_CONFIDENCE_H48_POLICY_NAME, 1.0),
                (B35_FORWARD_MODEL_VALUE_H56_POLICY_NAME, 0.85),
                (B35_GENETIC_FORWARD_MODEL_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b35_forward_model_value_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=212 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 35)
                self.assertEqual(report["parent_level"], 34)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b36_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b35_forward_model_value_source(tmpdir)
            variants = (
                (B36_LATENT_BELIEF_STATE_H48_POLICY_NAME, 1.0),
                (B36_BELIEF_ERROR_GATE_H48_POLICY_NAME, 1.0),
                (B36_CONTEXT_INFERENCE_H48_POLICY_NAME, 1.0),
                (B36_LATENT_BELIEF_STATE_H56_POLICY_NAME, 0.85),
                (B36_GENETIC_BELIEF_STATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b36_latent_belief_state_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=217 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 36)
                self.assertEqual(report["parent_level"], 35)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b37_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b36_latent_belief_state_source(tmpdir)
            variants = (
                (B37_STATE_FACTOR_GATE_H48_POLICY_NAME, 1.0),
                (B37_INTERO_EXTERO_FACTOR_H48_POLICY_NAME, 1.0),
                (B37_FACTOR_CONFIDENCE_H48_POLICY_NAME, 1.0),
                (B37_STATE_FACTOR_GATE_H56_POLICY_NAME, 0.85),
                (B37_GENETIC_STATE_FACTOR_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b37_state_factor_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=222 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 37)
                self.assertEqual(report["parent_level"], 36)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b38_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b37_state_factor_gate_source(tmpdir)
            variants = (
                (B38_FACTOR_ATTENTION_H48_POLICY_NAME, 1.0),
                (B38_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B38_CONFIDENCE_ATTENTION_H48_POLICY_NAME, 1.0),
                (B38_FACTOR_ATTENTION_H56_POLICY_NAME, 0.85),
                (B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b38_factor_attention_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=227 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 38)
                self.assertEqual(report["parent_level"], 37)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b39_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b38_factor_attention_source(tmpdir)
            variants = (
                (B39_ATTENTION_BINDING_H48_POLICY_NAME, 1.0),
                (B39_CROSS_FACTOR_BINDING_H48_POLICY_NAME, 1.0),
                (B39_CONTEXT_BINDING_ATTENTION_H48_POLICY_NAME, 1.0),
                (B39_ATTENTION_BINDING_H56_POLICY_NAME, 0.85),
                (B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b39_attention_binding_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=232 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 39)
                self.assertEqual(report["parent_level"], 38)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b40_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b39_attention_binding_source(tmpdir)
            variants = (
                (B40_GLOBAL_WORKSPACE_H48_POLICY_NAME, 1.0),
                (B40_SENSORY_WORKSPACE_H48_POLICY_NAME, 1.0),
                (B40_CONTEXT_WORKSPACE_H48_POLICY_NAME, 1.0),
                (B40_GLOBAL_WORKSPACE_H56_POLICY_NAME, 0.85),
                (B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b40_global_workspace_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=237 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 40)
                self.assertEqual(report["parent_level"], 39)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b41_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b40_global_workspace_source(tmpdir)
            variants = (
                (B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME, 1.0),
                (B41_INHIBITORY_CONTROL_H48_POLICY_NAME, 1.0),
                (B41_GOAL_CONTEXT_SELECTOR_H48_POLICY_NAME, 1.0),
                (B41_EXECUTIVE_WORKSPACE_H56_POLICY_NAME, 0.85),
                (B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b41_executive_workspace_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=242 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 41)
                self.assertEqual(report["parent_level"], 40)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b42_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b41_executive_workspace_source(tmpdir)
            variants = (
                (B42_ERROR_MONITOR_H48_POLICY_NAME, 1.0),
                (B42_CONFLICT_MONITOR_H48_POLICY_NAME, 1.0),
                (B42_PERFORMANCE_MONITOR_H48_POLICY_NAME, 1.0),
                (B42_ERROR_MONITOR_H56_POLICY_NAME, 0.85),
                (B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b42_error_monitor_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=247 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 42)
                self.assertEqual(report["parent_level"], 41)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b43_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b42_error_monitor_source(tmpdir)
            variants = (
                (B43_ADAPTIVE_PRECISION_H48_POLICY_NAME, 1.0),
                (B43_AROUSAL_PRECISION_H48_POLICY_NAME, 1.0),
                (B43_THRESHOLD_ADAPTATION_H48_POLICY_NAME, 1.0),
                (B43_ADAPTIVE_PRECISION_H56_POLICY_NAME, 0.85),
                (B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b43_adaptive_precision_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=252 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 43)
                self.assertEqual(report["parent_level"], 42)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b44_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b43_adaptive_precision_source(tmpdir)
            variants = (
                (B44_THALAMIC_RELAY_H48_POLICY_NAME, 1.0),
                (B44_SENSORY_RELAY_H48_POLICY_NAME, 1.0),
                (B44_CONTEXT_RELAY_H48_POLICY_NAME, 1.0),
                (B44_THALAMIC_RELAY_H56_POLICY_NAME, 0.85),
                (B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b44_thalamic_relay_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=257 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 44)
                self.assertEqual(report["parent_level"], 43)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b45_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b44_thalamic_relay_source(tmpdir)
            variants = (
                (B45_RETICULAR_INHIBITION_H48_POLICY_NAME, 1.0),
                (B45_SENSORY_INHIBITION_H48_POLICY_NAME, 1.0),
                (B45_CONTEXT_INHIBITION_H48_POLICY_NAME, 1.0),
                (B45_RETICULAR_INHIBITION_H56_POLICY_NAME, 0.85),
                (B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b45_reticular_inhibition_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=262 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 45)
                self.assertEqual(report["parent_level"], 44)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b46_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b45_reticular_inhibition_source(tmpdir)
            variants = (
                (B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME, 1.0),
                (B46_FEEDBACK_GAIN_H48_POLICY_NAME, 1.0),
                (B46_CONTEXT_FEEDBACK_H48_POLICY_NAME, 1.0),
                (B46_CORTICOTHALAMIC_FEEDBACK_H56_POLICY_NAME, 0.85),
                (B46_GENETIC_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b46_corticothalamic_feedback_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=267 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 46)
                self.assertEqual(report["parent_level"], 45)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b47_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b46_corticothalamic_feedback_source(tmpdir)
            variants = (
                (B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME, 1.0),
                (B47_PHASE_LOCKING_H48_POLICY_NAME, 1.0),
                (B47_COHERENCE_GATE_H48_POLICY_NAME, 1.0),
                (B47_OSCILLATORY_SYNCHRONY_H56_POLICY_NAME, 0.85),
                (B47_GENETIC_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b47_oscillatory_synchrony_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=272 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 47)
                self.assertEqual(report["parent_level"], 46)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b48_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b47_oscillatory_synchrony_source(tmpdir)
            variants = (
                (B48_CEREBELLAR_TIMING_H48_POLICY_NAME, 1.0),
                (B48_TIMING_ERROR_CORRECTION_H48_POLICY_NAME, 1.0),
                (B48_PREDICTIVE_TIMING_H48_POLICY_NAME, 1.0),
                (B48_CEREBELLAR_TIMING_H56_POLICY_NAME, 0.85),
                (B48_GENETIC_CEREBELLAR_TIMING_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b48_cerebellar_timing_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=277 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 48)
                self.assertEqual(report["parent_level"], 47)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b49_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b48_cerebellar_timing_source(tmpdir)
            variants = (
                (B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME, 1.0),
                (B49_DIRECT_PATH_FACILITATION_H48_POLICY_NAME, 1.0),
                (B49_INDIRECT_PATH_SUPPRESSION_H48_POLICY_NAME, 1.0),
                (B49_STRIATAL_ACTION_GATE_H56_POLICY_NAME, 0.85),
                (B49_GENETIC_STRIATAL_GATE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b49_striatal_action_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=282 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 49)
                self.assertEqual(report["parent_level"], 48)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b50_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b49_striatal_action_gate_source(tmpdir)
            variants = (
                (B50_HABIT_CHUNKING_H48_POLICY_NAME, 1.0),
                (B50_ACTION_CHUNK_VALUE_H48_POLICY_NAME, 1.0),
                (B50_HABIT_STABILITY_H48_POLICY_NAME, 1.0),
                (B50_HABIT_CHUNKING_H56_POLICY_NAME, 0.85),
                (B50_GENETIC_HABIT_CHUNKING_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b50_habit_chunking_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=287 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 50)
                self.assertEqual(report["parent_level"], 49)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b51_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b50_habit_chunking_source(tmpdir)
            variants = (
                (B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME, 1.0),
                (B51_REWARD_PREDICTION_GAIN_H48_POLICY_NAME, 1.0),
                (B51_NOVELTY_MODULATED_HABIT_H48_POLICY_NAME, 1.0),
                (B51_DOPAMINERGIC_HABIT_MODULATION_H56_POLICY_NAME, 0.85),
                (B51_GENETIC_DOPAMINE_HABIT_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b51_dopaminergic_habit_modulation_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=292 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 51)
                self.assertEqual(report["parent_level"], 50)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b52_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b51_dopaminergic_habit_source(tmpdir)
            variants = (
                (B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME, 1.0),
                (B52_ATTENTION_GAIN_H48_POLICY_NAME, 1.0),
                (B52_UNCERTAINTY_RELEASE_H48_POLICY_NAME, 1.0),
                (B52_CHOLINERGIC_PRECISION_GATE_H56_POLICY_NAME, 0.85),
                (B52_GENETIC_CHOLINERGIC_PRECISION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b52_cholinergic_precision_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=297 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 52)
                self.assertEqual(report["parent_level"], 51)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b53_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b52_cholinergic_precision_source(tmpdir)
            variants = (
                (B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME, 1.0),
                (B53_SURPRISE_GAIN_H48_POLICY_NAME, 1.0),
                (B53_STRESS_PRECISION_H48_POLICY_NAME, 1.0),
                (B53_NORADRENERGIC_AROUSAL_GAIN_H56_POLICY_NAME, 0.85),
                (B53_GENETIC_AROUSAL_PRECISION_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b53_noradrenergic_arousal_gain_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=302 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 53)
                self.assertEqual(report["parent_level"], 52)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b54_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b53_noradrenergic_arousal_source(tmpdir)
            variants = (
                (B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B54_IMPULSE_SUPPRESSION_H48_POLICY_NAME, 1.0),
                (B54_PATIENCE_BALANCE_H48_POLICY_NAME, 1.0),
                (B54_SEROTONERGIC_PATIENCE_GATE_H56_POLICY_NAME, 0.85),
                (B54_GENETIC_SEROTONIN_PATIENCE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b54_serotonergic_patience_gate_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=307 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 54)
                self.assertEqual(report["parent_level"], 53)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b55_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b54_serotonergic_patience_source(tmpdir)
            variants = (
                (B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME, 1.0),
                (B55_SATIETY_RECOVERY_BALANCE_H48_POLICY_NAME, 1.0),
                (B55_SLEEP_HUNGER_ARBITER_H48_POLICY_NAME, 1.0),
                (B55_HYPOTHALAMIC_DRIVE_COUPLING_H56_POLICY_NAME, 0.85),
                (B55_GENETIC_HYPOTHALAMIC_DRIVE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b55_hypothalamic_drive_coupling_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=312 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 55)
                self.assertEqual(report["parent_level"], 54)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b56_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b55_hypothalamic_drive_source(tmpdir)
            variants = (
                (B56_HPA_STRESS_AXIS_H48_POLICY_NAME, 1.0),
                (B56_CORTISOL_RECOVERY_BALANCE_H48_POLICY_NAME, 1.0),
                (B56_STRESS_LOAD_GATE_H48_POLICY_NAME, 1.0),
                (B56_HPA_STRESS_AXIS_H56_POLICY_NAME, 0.85),
                (B56_GENETIC_HPA_STRESS_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b56_hpa_stress_axis_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=317 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 56)
                self.assertEqual(report["parent_level"], 55)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b57_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b56_hpa_stress_source(tmpdir)
            variants = (
                (B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME, 1.0),
                (B57_VISCERAL_SALIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B57_STRESS_DRIVE_AWARENESS_H48_POLICY_NAME, 1.0),
                (B57_INSULAR_INTEROCEPTIVE_AWARENESS_H56_POLICY_NAME, 0.85),
                (B57_GENETIC_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b57_insular_interoceptive_awareness_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=322 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 57)
                self.assertEqual(report["parent_level"], 56)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b58_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b57_insular_interoceptive_source(tmpdir)
            variants = (
                (B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME, 1.0),
                (B58_ERROR_SALIENCE_GATE_H48_POLICY_NAME, 1.0),
                (B58_CONFLICT_RESOLUTION_BALANCE_H48_POLICY_NAME, 1.0),
                (B58_ACC_CONFLICT_MONITOR_H56_POLICY_NAME, 0.85),
                (B58_GENETIC_ACC_CONFLICT_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b58_acc_conflict_monitor_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=327 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 58)
                self.assertEqual(report["parent_level"], 57)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b59_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b58_acc_conflict_source(tmpdir)
            variants = (
                (B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME, 1.0),
                (B59_WORKING_SET_STABILITY_H48_POLICY_NAME, 1.0),
                (B59_EXECUTIVE_TASK_SET_H48_POLICY_NAME, 1.0),
                (B59_PREFRONTAL_GOAL_CONTEXT_H56_POLICY_NAME, 0.85),
                (B59_GENETIC_PREFRONTAL_CONTROL_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b59_prefrontal_goal_context_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=332 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 59)
                self.assertEqual(report["parent_level"], 58)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b60_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b59_prefrontal_goal_source(tmpdir)
            variants = (
                (B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME, 1.0),
                (B60_REVERSAL_VALUE_GATE_H48_POLICY_NAME, 1.0),
                (B60_GOAL_OUTCOME_PREDICTION_H48_POLICY_NAME, 1.0),
                (B60_ORBITOFRONTAL_OUTCOME_VALUE_H56_POLICY_NAME, 0.85),
                (B60_GENETIC_ORBITOFRONTAL_VALUE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b60_orbitofrontal_outcome_value_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=337 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 60)
                self.assertEqual(report["parent_level"], 59)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b61_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b60_orbitofrontal_value_source(tmpdir)
            variants = (
                (B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME, 1.0),
                (B61_THREAT_VALUE_TAG_H48_POLICY_NAME, 1.0),
                (B61_SAFETY_PREDICTION_GATE_H48_POLICY_NAME, 1.0),
                (B61_AMYGDALA_SAFETY_VALUE_H56_POLICY_NAME, 0.85),
                (B61_GENETIC_AMYGDALA_SAFETY_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b61_amygdala_safety_value_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=342 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 61)
                self.assertEqual(report["parent_level"], 60)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b62_transfer_reports_source_parent_and_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b61_amygdala_safety_source(tmpdir)
            variants = (
                (B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME, 1.0),
                (B62_FREEZE_FLEE_BALANCE_H48_POLICY_NAME, 1.0),
                (B62_SHELTER_DEFENSE_GATE_H48_POLICY_NAME, 1.0),
                (B62_DEFENSIVE_MODE_SELECTOR_H56_POLICY_NAME, 0.85),
                (B62_GENETIC_DEFENSIVE_MODE_H48_POLICY_NAME, 1.0),
            )
            for index, (variant_name, min_coverage) in enumerate(variants):
                config = build_b62_defensive_mode_selector_config(
                    variant_name,
                    source_checkpoint=checkpoint,
                )
                target = SpiderBrain(
                    seed=352 + index,
                    module_dropout=0.0,
                    config=config,
                )
                report = target.b_series_transfer_report
                self.assertIsNotNone(report)
                assert report is not None
                self.assertEqual(report["source_checkpoint"], str(checkpoint))
                self.assertEqual(report["target_b_level"], 62)
                self.assertEqual(report["parent_level"], 61)
                self.assertGreaterEqual(float(report["coverage"]), min_coverage)
                self.assertFalse(report["allow_low_coverage"])

    def test_b1_trace_fields_and_primitive_contract(self) -> None:
        source_config = _b0_config()
        source = SpiderBrain(seed=36, module_dropout=0.0, config=source_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = source.save(Path(tmpdir) / "b0")
            config = build_b1_capacity_config(
                B1_CAPACITY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=37,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_effective_level",
            "b_mode",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "semantic_action_reason",
            "semantic_override_count",
            "bridge_primitive_action",
            "bridge_reason",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 1)
        self.assertEqual(first["b_effective_level"], "B1")
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(first["semantic_action_source"], "network_policy")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b2_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b1_threat_guard_source(tmpdir)
            config = build_b2_temporal_threat_config(
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=40,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b_temporal_threat_pressure",
            "b_predator_memory_pressure",
            "b_predator_trace_pressure",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 2)
        self.assertEqual(first["b_parent_level"], 1)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B2_TEMPORAL_THREAT_SELECTION_SOURCE,
        )
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b3_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b2_temporal_threat_source(tmpdir)
            config = build_b3_contact_memory_config(
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=44,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b_temporal_threat_pressure",
            "b_predator_memory_pressure",
            "b_predator_trace_pressure",
            "b3_contact_cooldown",
            "b3_post_food_cooldown",
            "b3_hunger_drop",
            "b3_controller_profile",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 3)
        self.assertEqual(first["b_parent_level"], 2)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B3_CONTACT_MEMORY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b3_controller_profile"], "standard")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b4_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b3_recurrent_guard_source(tmpdir)
            config = build_b4_recovery_balance_config(
                B4_RECOVERY_BALANCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=46,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b_temporal_threat_pressure",
            "b_predator_memory_pressure",
            "b_predator_trace_pressure",
            "b3_contact_cooldown",
            "b3_post_food_cooldown",
            "b3_hunger_drop",
            "b3_controller_profile",
            "b4_controller_profile",
            "b4_recovery_pressure",
            "b4_sleep_hold",
            "b4_exit_blocked",
            "b4_hunger_release",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 4)
        self.assertEqual(first["b_parent_level"], 3)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B4_RECOVERY_BALANCE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b4_controller_profile"], "recovery_balance")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b5_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b4_genetic_recovery_source(tmpdir)
            config = build_b5_homeostatic_arbiter_config(
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=47,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b4_controller_profile",
            "b4_recovery_pressure",
            "b5_controller_profile",
            "b5_hunger_urgency",
            "b5_sleep_pressure",
            "b5_recovery_debt",
            "b5_threat_gate",
            "b5_sleep_bout_lock",
            "b5_forage_commitment_lock",
            "b5_homeostatic_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 5)
        self.assertEqual(first["b_parent_level"], 4)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B5_HOMEOSTATIC_ARBITER_SELECTION_SOURCE,
        )
        self.assertEqual(first["b5_controller_profile"], "homeostatic_arbiter")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b6_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b5_genetic_homeostasis_source(tmpdir)
            config = build_b6_risk_corridor_config(
                B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=48,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b5_controller_profile",
            "b6_controller_family",
            "b6_controller_profile",
            "b6_risk_pressure",
            "b6_threat_priority",
            "b6_forage_suppressed",
            "b6_corridor_commitment",
            "b6_corridor_progress_memory",
            "b6_recurrent_state",
            "b6_return_lock",
            "b6_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 6)
        self.assertEqual(first["b_parent_level"], 5)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B6_RISK_CORRIDOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b6_controller_family"], "risk_corridor")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b7_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b6_fused_risk_recurrent_source(tmpdir)
            config = build_b7_affordance_budget_config(
                B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=49,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b6_controller_family",
            "b7_controller_profile",
            "b7_affordance_state",
            "b7_energy_budget",
            "b7_budget_margin",
            "b7_food_steps_estimate",
            "b7_return_steps_estimate",
            "b7_corridor_viability",
            "b7_abort_return",
            "b7_commitment_lock",
            "b7_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 7)
        self.assertEqual(first["b_parent_level"], 6)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B7_AFFORDANCE_BUDGET_SELECTION_SOURCE,
        )
        self.assertEqual(first["b7_controller_profile"], "affordance_budget")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b8_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b7_affordance_budget_source(tmpdir)
            config = build_b8_spatial_affordance_config(
                B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=50,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b7_controller_profile",
            "b8_controller_profile",
            "b8_spatial_map_state",
            "b8_local_affordance_score",
            "b8_return_vector_strength",
            "b8_corridor_dead_end_risk",
            "b8_abort_executed",
            "b8_place_memory",
            "b8_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 8)
        self.assertEqual(first["b_parent_level"], 7)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B8_SPATIAL_AFFORDANCE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b8_controller_profile"], "spatial_affordance_map")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b9_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b8_spatial_affordance_source(tmpdir)
            config = build_b9_waypoint_planner_config(
                B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=51,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b8_controller_profile",
            "b9_controller_profile",
            "b9_route_state",
            "b9_route_confidence",
            "b9_waypoint_lock",
            "b9_path_integrator",
            "b9_replan_signal",
            "b9_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 9)
        self.assertEqual(first["b_parent_level"], 8)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B9_WAYPOINT_PLANNER_SELECTION_SOURCE,
        )
        self.assertEqual(first["b9_controller_profile"], "waypoint_planner")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b10_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b9_waypoint_planner_source(tmpdir)
            config = build_b10_prospective_replay_config(
                B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=52,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b9_controller_profile",
            "b10_controller_profile",
            "b10_replay_state",
            "b10_prospective_value",
            "b10_rollout_depth",
            "b10_replay_memory",
            "b10_plan_commitment",
            "b10_abort_signal",
            "b10_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 10)
        self.assertEqual(first["b_parent_level"], 9)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B10_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b10_controller_profile"], "prospective_replay")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b11_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b10_prospective_replay_source(tmpdir)
            config = build_b11_confidence_arbiter_config(
                B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=53,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b10_controller_profile",
            "b11_controller_profile",
            "b11_confidence_state",
            "b11_plan_confidence",
            "b11_uncertainty",
            "b11_neuromod_signal",
            "b11_confidence_lock",
            "b11_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 11)
        self.assertEqual(first["b_parent_level"], 10)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B11_CONFIDENCE_ARBITER_SELECTION_SOURCE,
        )
        self.assertEqual(first["b11_controller_profile"], "confidence_arbiter")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b12_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b11_confidence_arbiter_source(tmpdir)
            config = build_b12_predictive_attention_config(
                B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=54,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b11_controller_profile",
            "b12_controller_profile",
            "b12_attention_state",
            "b12_prediction_error",
            "b12_attention_gain",
            "b12_expected_progress",
            "b12_search_lock",
            "b12_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 12)
        self.assertEqual(first["b_parent_level"], 11)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B12_PREDICTIVE_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b12_controller_profile"], "predictive_attention")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b13_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b12_predictive_attention_source(tmpdir)
            config = build_b13_local_affordance_search_config(
                B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=55,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b12_controller_profile",
            "b13_controller_profile",
            "b13_search_state",
            "b13_local_route_score",
            "b13_affordance_samples",
            "b13_search_memory",
            "b13_dead_end_score",
            "b13_search_lock",
            "b13_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 13)
        self.assertEqual(first["b_parent_level"], 12)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B13_LOCAL_SEARCH_SELECTION_SOURCE,
        )
        self.assertEqual(first["b13_controller_profile"], "local_affordance_search")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b14_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b13_local_affordance_search_source(tmpdir)
            config = build_b14_affordance_uncertainty_config(
                B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=56,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b13_controller_profile",
            "b14_controller_profile",
            "b14_uncertainty_state",
            "b14_affordance_confidence",
            "b14_uncertainty",
            "b14_risk_adjusted_score",
            "b14_commitment_lock",
            "b14_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 14)
        self.assertEqual(first["b_parent_level"], 13)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B14_AFFORDANCE_UNCERTAINTY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b14_controller_profile"], "affordance_uncertainty")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b15_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b14_affordance_uncertainty_source(tmpdir)
            config = build_b15_option_critic_config(
                B15_OPTION_CRITIC_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=57,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b14_controller_profile",
            "b15_controller_profile",
            "b15_option_state",
            "b15_option_value",
            "b15_termination_pressure",
            "b15_persistence_score",
            "b15_option_lock",
            "b15_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 15)
        self.assertEqual(first["b_parent_level"], 14)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B15_OPTION_CRITIC_SELECTION_SOURCE,
        )
        self.assertEqual(first["b15_controller_profile"], "option_critic")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b16_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b15_option_critic_source(tmpdir)
            config = build_b16_option_ensemble_config(
                B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=58,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b15_controller_profile",
            "b16_controller_profile",
            "b16_ensemble_state",
            "b16_continue_vote",
            "b16_return_vote",
            "b16_option_votes",
            "b16_consensus_score",
            "b16_conflict_score",
            "b16_ensemble_lock",
            "b16_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 16)
        self.assertEqual(first["b_parent_level"], 15)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B16_OPTION_ENSEMBLE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b16_controller_profile"], "option_ensemble")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b17_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b16_option_ensemble_source(tmpdir)
            config = build_b17_neuromodulated_ensemble_config(
                B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=59,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b16_controller_profile",
            "b17_controller_profile",
            "b17_modulator_state",
            "b17_arousal_signal",
            "b17_homeostatic_gain",
            "b17_option_gain",
            "b17_conflict_release",
            "b17_modulation_lock",
            "b17_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 17)
        self.assertEqual(first["b_parent_level"], 16)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B17_NEUROMODULATED_ENSEMBLE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b17_controller_profile"], "neuromodulated_ensemble")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b18_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b17_neuromodulated_ensemble_source(tmpdir)
            config = build_b18_eligibility_trace_config(
                B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=60,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b17_controller_profile",
            "b18_controller_profile",
            "b18_trace_state",
            "b18_eligibility_trace",
            "b18_reward_prediction_proxy",
            "b18_stability_bias",
            "b18_switch_pressure",
            "b18_trace_lock",
            "b18_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 18)
        self.assertEqual(first["b_parent_level"], 17)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B18_ELIGIBILITY_TRACE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b18_controller_profile"], "eligibility_trace")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b19_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b18_eligibility_trace_source(tmpdir)
            config = build_b19_episodic_meta_memory_config(
                B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=61,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b18_controller_profile",
            "b19_controller_profile",
            "b19_memory_state",
            "b19_episode_memory",
            "b19_consolidation_score",
            "b19_stability_vote",
            "b19_switch_suppression",
            "b19_memory_lock",
            "b19_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 19)
        self.assertEqual(first["b_parent_level"], 18)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B19_EPISODIC_META_MEMORY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b19_controller_profile"], "episodic_meta_memory")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b20_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b19_episodic_meta_memory_source(tmpdir)
            config = build_b20_working_memory_gate_config(
                B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=62,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b19_controller_profile",
            "b20_controller_profile",
            "b20_buffer_state",
            "b20_working_buffer",
            "b20_context_binding",
            "b20_gate_vote",
            "b20_release_vote",
            "b20_buffer_lock",
            "b20_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 20)
        self.assertEqual(first["b_parent_level"], 19)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B20_WORKING_MEMORY_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b20_controller_profile"], "working_memory_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b21_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b20_working_memory_gate_source(tmpdir)
            config = build_b21_hippocampal_replay_config(
                B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=63,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b20_controller_profile",
            "b21_controller_profile",
            "b21_replay_state",
            "b21_sequence_memory",
            "b21_replay_score",
            "b21_route_commitment",
            "b21_abort_prediction",
            "b21_replay_lock",
            "b21_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 21)
        self.assertEqual(first["b_parent_level"], 20)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B21_HIPPOCAMPAL_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b21_controller_profile"], "hippocampal_replay")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b22_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b21_hippocampal_replay_source(tmpdir)
            config = build_b22_prospective_replay_config(
                B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=64,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b21_controller_profile",
            "b22_controller_profile",
            "b22_sim_state",
            "b22_prospective_sim",
            "b22_forward_model_score",
            "b22_viability_projection",
            "b22_abort_projection",
            "b22_sim_lock",
            "b22_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 22)
        self.assertEqual(first["b_parent_level"], 21)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B22_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b22_controller_profile"], "prospective_map_replay")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b23_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b22_prospective_replay_source(tmpdir)
            config = build_b23_conflict_monitor_config(
                B23_CONFLICT_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=65,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b22_controller_profile",
            "b23_controller_profile",
            "b23_conflict_state",
            "b23_prediction_error",
            "b23_conflict_memory",
            "b23_stability_vote",
            "b23_abort_bias",
            "b23_monitor_lock",
            "b23_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 23)
        self.assertEqual(first["b_parent_level"], 22)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B23_CONFLICT_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b23_controller_profile"], "conflict_monitor")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b24_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b23_conflict_monitor_source(tmpdir)
            config = build_b24_precision_conflict_config(
                B24_PRECISION_CONFLICT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=66,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b23_controller_profile",
            "b24_controller_profile",
            "b24_precision_state",
            "b24_precision_memory",
            "b24_precision_vote",
            "b24_uncertainty_pressure",
            "b24_abort_precision",
            "b24_precision_lock",
            "b24_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 24)
        self.assertEqual(first["b_parent_level"], 23)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B24_PRECISION_CONFLICT_SELECTION_SOURCE,
        )
        self.assertEqual(first["b24_controller_profile"], "precision_conflict")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b25_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b24_precision_conflict_source(tmpdir)
            config = build_b25_metacognitive_confidence_config(
                B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=67,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b24_controller_profile",
            "b25_controller_profile",
            "b25_metacognitive_state",
            "b25_confidence_memory",
            "b25_confidence_vote",
            "b25_doubt_pressure",
            "b25_control_gain",
            "b25_meta_lock",
            "b25_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 25)
        self.assertEqual(first["b_parent_level"], 24)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B25_METACOGNITIVE_CONFIDENCE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b25_controller_profile"], "metacognitive_confidence")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b26_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b25_metacognitive_confidence_source(tmpdir)
            config = build_b26_allostatic_prediction_config(
                B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=68,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b25_controller_profile",
            "b26_controller_profile",
            "b26_allostatic_state",
            "b26_prediction_error",
            "b26_setpoint_pressure",
            "b26_control_vote",
            "b26_stability_lock",
            "b26_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 26)
        self.assertEqual(first["b_parent_level"], 25)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B26_ALLOSTATIC_PREDICTION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b26_controller_profile"], "allostatic_prediction")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b27_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b26_allostatic_prediction_source(tmpdir)
            config = build_b27_arousal_gain_config(
                B27_AROUSAL_GAIN_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=69,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b26_controller_profile",
            "b27_controller_profile",
            "b27_arousal_state",
            "b27_arousal_level",
            "b27_gain_modulation",
            "b27_stress_pressure",
            "b27_arousal_lock",
            "b27_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 27)
        self.assertEqual(first["b_parent_level"], 26)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B27_AROUSAL_GAIN_SELECTION_SOURCE,
        )
        self.assertEqual(first["b27_controller_profile"], "arousal_gain")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b28_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b27_arousal_gain_source(tmpdir)
            config = build_b28_interoceptive_attention_config(
                B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=70,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b27_controller_profile",
            "b28_controller_profile",
            "b28_attention_state",
            "b28_interoceptive_focus",
            "b28_attention_gain",
            "b28_distractor_pressure",
            "b28_attention_lock",
            "b28_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 28)
        self.assertEqual(first["b_parent_level"], 27)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B28_INTEROCEPTIVE_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b28_controller_profile"], "interoceptive_attention")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b29_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b28_interoceptive_attention_source(tmpdir)
            config = build_b29_salience_competition_config(
                B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=71,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b28_controller_profile",
            "b29_controller_profile",
            "b29_salience_state",
            "b29_threat_salience",
            "b29_homeostatic_salience",
            "b29_corridor_salience",
            "b29_winner_channel",
            "b29_salience_lock",
            "b29_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 29)
        self.assertEqual(first["b_parent_level"], 28)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B29_SALIENCE_COMPETITION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b29_controller_profile"], "salience_competition")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b30_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b29_salience_competition_source(tmpdir)
            config = build_b30_basal_ganglia_gate_config(
                B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=72,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b29_controller_profile",
            "b30_controller_profile",
            "b30_gate_state",
            "b30_go_signal",
            "b30_no_go_signal",
            "b30_action_gate",
            "b30_gate_lock",
            "b30_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 30)
        self.assertEqual(first["b_parent_level"], 29)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B30_BASAL_GANGLIA_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b30_controller_profile"], "basal_ganglia_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b31_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b30_basal_ganglia_gate_source(tmpdir)
            config = build_b31_dopamine_prediction_error_config(
                B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=73,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b30_controller_profile",
            "b31_controller_profile",
            "b31_dopamine_state",
            "b31_reward_prediction_error",
            "b31_tonic_dopamine",
            "b31_phasic_dopamine",
            "b31_gate_bias",
            "b31_dopamine_lock",
            "b31_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 31)
        self.assertEqual(first["b_parent_level"], 30)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B31_DOPAMINE_PREDICTION_ERROR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b31_controller_profile"], "dopamine_prediction_error")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b32_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b31_dopamine_prediction_error_source(tmpdir)
            config = build_b32_actor_critic_value_config(
                B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=74,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b31_controller_profile",
            "b32_controller_profile",
            "b32_critic_value",
            "b32_actor_advantage",
            "b32_value_error",
            "b32_policy_bias",
            "b32_value_lock",
            "b32_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 32)
        self.assertEqual(first["b_parent_level"], 31)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B32_ACTOR_CRITIC_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b32_controller_profile"], "actor_critic_value")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b33_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b32_actor_critic_value_source(tmpdir)
            config = build_b33_td_error_decomposition_config(
                B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=75,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b32_controller_profile",
            "b33_controller_profile",
            "b33_td_error",
            "b33_bootstrap_value",
            "b33_reward_trace",
            "b33_actor_update",
            "b33_td_lock",
            "b33_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 33)
        self.assertEqual(first["b_parent_level"], 32)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B33_TD_ERROR_DECOMPOSITION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b33_controller_profile"], "td_error_decomposition")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b34_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b33_td_error_decomposition_source(tmpdir)
            config = build_b34_eligibility_credit_config(
                B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=76,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b33_controller_profile",
            "b34_controller_profile",
            "b34_eligibility_trace",
            "b34_credit_assignment",
            "b34_synaptic_tag",
            "b34_decay_memory",
            "b34_credit_lock",
            "b34_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 34)
        self.assertEqual(first["b_parent_level"], 33)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B34_ELIGIBILITY_CREDIT_SELECTION_SOURCE,
        )
        self.assertEqual(first["b34_controller_profile"], "eligibility_credit")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b35_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b34_eligibility_credit_source(tmpdir)
            config = build_b35_forward_model_value_config(
                B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=77,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b34_controller_profile",
            "b35_controller_profile",
            "b35_forward_value",
            "b35_transition_error",
            "b35_model_confidence",
            "b35_prediction_memory",
            "b35_model_lock",
            "b35_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 35)
        self.assertEqual(first["b_parent_level"], 34)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B35_FORWARD_MODEL_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b35_controller_profile"], "forward_model_value")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b36_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b35_forward_model_value_source(tmpdir)
            config = build_b36_latent_belief_state_config(
                B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=78,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b35_controller_profile",
            "b36_controller_profile",
            "b36_latent_state",
            "b36_belief_error",
            "b36_state_confidence",
            "b36_context_memory",
            "b36_belief_lock",
            "b36_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 36)
        self.assertEqual(first["b_parent_level"], 35)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B36_LATENT_BELIEF_STATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b36_controller_profile"], "latent_belief_state")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b37_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b36_latent_belief_state_source(tmpdir)
            config = build_b37_state_factor_gate_config(
                B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=79,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b36_controller_profile",
            "b37_controller_profile",
            "b37_external_state_factor",
            "b37_internal_state_factor",
            "b37_factor_alignment",
            "b37_factor_confidence",
            "b37_factor_lock",
            "b37_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 37)
        self.assertEqual(first["b_parent_level"], 36)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B37_STATE_FACTOR_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b37_controller_profile"], "state_factor_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b38_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b37_state_factor_gate_source(tmpdir)
            config = build_b38_factor_attention_config(
                B38_FACTOR_ATTENTION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=80,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b37_controller_profile",
            "b38_controller_profile",
            "b38_external_attention",
            "b38_internal_attention",
            "b38_attention_balance",
            "b38_attention_gain",
            "b38_attention_lock",
            "b38_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 38)
        self.assertEqual(first["b_parent_level"], 37)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B38_FACTOR_ATTENTION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b38_controller_profile"], "factor_attention")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b39_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b38_factor_attention_source(tmpdir)
            config = build_b39_attention_binding_config(
                B39_ATTENTION_BINDING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=81,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b38_controller_profile",
            "b39_controller_profile",
            "b39_binding_strength",
            "b39_cross_factor_coherence",
            "b39_bound_context",
            "b39_binding_gain",
            "b39_binding_lock",
            "b39_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 39)
        self.assertEqual(first["b_parent_level"], 38)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B39_ATTENTION_BINDING_SELECTION_SOURCE,
        )
        self.assertEqual(first["b39_controller_profile"], "attention_binding")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b40_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b39_attention_binding_source(tmpdir)
            config = build_b40_global_workspace_config(
                B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=82,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b39_controller_profile",
            "b40_controller_profile",
            "b40_workspace_activation",
            "b40_broadcast_gain",
            "b40_context_availability",
            "b40_workspace_stability",
            "b40_workspace_lock",
            "b40_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 40)
        self.assertEqual(first["b_parent_level"], 39)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B40_GLOBAL_WORKSPACE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b40_controller_profile"], "global_workspace")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b41_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b40_global_workspace_source(tmpdir)
            config = build_b41_executive_workspace_config(
                B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=83,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b40_controller_profile",
            "b41_controller_profile",
            "b41_executive_selection",
            "b41_inhibitory_pressure",
            "b41_goal_context",
            "b41_executive_stability",
            "b41_executive_lock",
            "b41_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 41)
        self.assertEqual(first["b_parent_level"], 40)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B41_EXECUTIVE_WORKSPACE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b41_controller_profile"], "executive_workspace")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b42_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b41_executive_workspace_source(tmpdir)
            config = build_b42_error_monitor_config(
                B42_ERROR_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=84,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b41_controller_profile",
            "b42_controller_profile",
            "b42_error_signal",
            "b42_conflict_signal",
            "b42_performance_context",
            "b42_monitor_stability",
            "b42_monitor_lock",
            "b42_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 42)
        self.assertEqual(first["b_parent_level"], 41)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B42_ERROR_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b42_controller_profile"], "error_monitor")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b43_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b42_error_monitor_source(tmpdir)
            config = build_b43_adaptive_precision_config(
                B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=85,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b42_controller_profile",
            "b43_controller_profile",
            "b43_precision_signal",
            "b43_adaptive_threshold",
            "b43_arousal_context",
            "b43_control_stability",
            "b43_precision_lock",
            "b43_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 43)
        self.assertEqual(first["b_parent_level"], 42)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B43_ADAPTIVE_PRECISION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b43_controller_profile"], "adaptive_precision")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b44_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b43_adaptive_precision_source(tmpdir)
            config = build_b44_thalamic_relay_config(
                B44_THALAMIC_RELAY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=86,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b43_controller_profile",
            "b44_controller_profile",
            "b44_relay_gate",
            "b44_sensory_precision",
            "b44_context_relay",
            "b44_gate_stability",
            "b44_relay_lock",
            "b44_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 44)
        self.assertEqual(first["b_parent_level"], 43)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B44_THALAMIC_RELAY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b44_controller_profile"], "thalamic_relay")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b45_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b44_thalamic_relay_source(tmpdir)
            config = build_b45_reticular_inhibition_config(
                B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=87,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b44_controller_profile",
            "b45_controller_profile",
            "b45_inhibitory_gate",
            "b45_sensory_filter",
            "b45_context_suppression",
            "b45_loop_stability",
            "b45_inhibition_lock",
            "b45_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 45)
        self.assertEqual(first["b_parent_level"], 44)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B45_RETICULAR_INHIBITION_SELECTION_SOURCE,
        )
        self.assertEqual(first["b45_controller_profile"], "reticular_inhibition")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b46_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b45_reticular_inhibition_source(tmpdir)
            config = build_b46_corticothalamic_feedback_config(
                B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=88,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b45_controller_profile",
            "b46_controller_profile",
            "b46_feedback_gain",
            "b46_topdown_context",
            "b46_prediction_match",
            "b46_feedback_stability",
            "b46_feedback_lock",
            "b46_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 46)
        self.assertEqual(first["b_parent_level"], 45)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B46_CORTICOTHALAMIC_FEEDBACK_SELECTION_SOURCE,
        )
        self.assertEqual(first["b46_controller_profile"], "corticothalamic_feedback")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b47_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b46_corticothalamic_feedback_source(tmpdir)
            config = build_b47_oscillatory_synchrony_config(
                B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=89,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b46_controller_profile",
            "b47_controller_profile",
            "b47_phase_alignment",
            "b47_synchrony_gain",
            "b47_cross_loop_coherence",
            "b47_phase_lock",
            "b47_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 47)
        self.assertEqual(first["b_parent_level"], 46)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B47_OSCILLATORY_SYNCHRONY_SELECTION_SOURCE,
        )
        self.assertEqual(first["b47_controller_profile"], "oscillatory_synchrony")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b48_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b47_oscillatory_synchrony_source(tmpdir)
            config = build_b48_cerebellar_timing_config(
                B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=90,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b47_controller_profile",
            "b48_controller_profile",
            "b48_timing_error",
            "b48_predictive_timing",
            "b48_corrective_gain",
            "b48_calibration_lock",
            "b48_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 48)
        self.assertEqual(first["b_parent_level"], 47)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B48_CEREBELLAR_TIMING_SELECTION_SOURCE,
        )
        self.assertEqual(first["b48_controller_profile"], "cerebellar_timing")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b49_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b48_cerebellar_timing_source(tmpdir)
            config = build_b49_striatal_action_gate_config(
                B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=91,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b48_controller_profile",
            "b49_controller_profile",
            "b49_go_signal",
            "b49_no_go_signal",
            "b49_action_gate_balance",
            "b49_selection_lock",
            "b49_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 49)
        self.assertEqual(first["b_parent_level"], 48)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B49_STRIATAL_ACTION_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b49_controller_profile"], "striatal_action_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b50_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b49_striatal_action_gate_source(tmpdir)
            config = build_b50_habit_chunking_config(
                B50_HABIT_CHUNKING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=92,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b49_controller_profile",
            "b50_controller_profile",
            "b50_habit_strength",
            "b50_chunk_value",
            "b50_habit_stability",
            "b50_chunk_lock",
            "b50_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 50)
        self.assertEqual(first["b_parent_level"], 49)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B50_HABIT_CHUNKING_SELECTION_SOURCE,
        )
        self.assertEqual(first["b50_controller_profile"], "habit_chunking")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b51_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b50_habit_chunking_source(tmpdir)
            config = build_b51_dopaminergic_habit_modulation_config(
                B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=93,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b50_controller_profile",
            "b51_controller_profile",
            "b51_prediction_error",
            "b51_dopamine_gain",
            "b51_habit_modulation",
            "b51_modulation_lock",
            "b51_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 51)
        self.assertEqual(first["b_parent_level"], 50)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B51_DOPAMINERGIC_HABIT_MODULATION_SELECTION_SOURCE,
        )
        self.assertEqual(
            first["b51_controller_profile"],
            "dopaminergic_habit_modulation",
        )
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b52_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b51_dopaminergic_habit_source(tmpdir)
            config = build_b52_cholinergic_precision_gate_config(
                B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=94,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b51_controller_profile",
            "b52_controller_profile",
            "b52_acetylcholine_level",
            "b52_precision_gain",
            "b52_uncertainty_signal",
            "b52_attention_lock",
            "b52_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 52)
        self.assertEqual(first["b_parent_level"], 51)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B52_CHOLINERGIC_PRECISION_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b52_controller_profile"], "cholinergic_precision_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b53_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b52_cholinergic_precision_source(tmpdir)
            config = build_b53_noradrenergic_arousal_gain_config(
                B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=95,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b52_controller_profile",
            "b53_controller_profile",
            "b53_norepinephrine_level",
            "b53_arousal_gain",
            "b53_surprise_signal",
            "b53_gain_lock",
            "b53_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 53)
        self.assertEqual(first["b_parent_level"], 52)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B53_NORADRENERGIC_AROUSAL_GAIN_SELECTION_SOURCE,
        )
        self.assertEqual(first["b53_controller_profile"], "noradrenergic_arousal_gain")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b54_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b53_noradrenergic_arousal_source(tmpdir)
            config = build_b54_serotonergic_patience_gate_config(
                B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=96,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b53_controller_profile",
            "b54_controller_profile",
            "b54_serotonin_level",
            "b54_patience_signal",
            "b54_impulse_suppression",
            "b54_patience_lock",
            "b54_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 54)
        self.assertEqual(first["b_parent_level"], 53)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B54_SEROTONERGIC_PATIENCE_GATE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b54_controller_profile"], "serotonergic_patience_gate")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b55_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b54_serotonergic_patience_source(tmpdir)
            config = build_b55_hypothalamic_drive_coupling_config(
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=97,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b54_controller_profile",
            "b55_controller_profile",
            "b55_hypothalamic_drive",
            "b55_satiety_signal",
            "b55_recovery_bias",
            "b55_drive_balance",
            "b55_drive_lock",
            "b55_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 55)
        self.assertEqual(first["b_parent_level"], 54)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B55_HYPOTHALAMIC_DRIVE_COUPLING_SELECTION_SOURCE,
        )
        self.assertEqual(first["b55_controller_profile"], "hypothalamic_drive_coupling")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b56_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b55_hypothalamic_drive_source(tmpdir)
            config = build_b56_hpa_stress_axis_config(
                B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=98,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b55_controller_profile",
            "b56_controller_profile",
            "b56_cortisol_level",
            "b56_stress_load",
            "b56_recovery_signal",
            "b56_endocrine_balance",
            "b56_stress_lock",
            "b56_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 56)
        self.assertEqual(first["b_parent_level"], 55)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B56_HPA_STRESS_AXIS_SELECTION_SOURCE,
        )
        self.assertEqual(first["b56_controller_profile"], "hpa_stress_axis")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b57_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b56_hpa_stress_source(tmpdir)
            config = build_b57_insular_interoceptive_awareness_config(
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=99,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b56_controller_profile",
            "b57_controller_profile",
            "b57_interoceptive_awareness",
            "b57_visceral_salience",
            "b57_body_state_confidence",
            "b57_awareness_balance",
            "b57_awareness_lock",
            "b57_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 57)
        self.assertEqual(first["b_parent_level"], 56)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_SELECTION_SOURCE,
        )
        self.assertEqual(
            first["b57_controller_profile"],
            "insular_interoceptive_awareness",
        )
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b58_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b57_insular_interoceptive_source(tmpdir)
            config = build_b58_acc_conflict_monitor_config(
                B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=100,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b57_controller_profile",
            "b58_controller_profile",
            "b58_conflict_signal",
            "b58_error_likelihood",
            "b58_control_allocation",
            "b58_resolution_balance",
            "b58_conflict_lock",
            "b58_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 58)
        self.assertEqual(first["b_parent_level"], 57)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B58_ACC_CONFLICT_MONITOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b58_controller_profile"], "acc_conflict_monitor")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b59_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b58_acc_conflict_source(tmpdir)
            config = build_b59_prefrontal_goal_context_config(
                B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=101,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b58_controller_profile",
            "b59_controller_profile",
            "b59_goal_context",
            "b59_working_set_stability",
            "b59_task_set_confidence",
            "b59_executive_balance",
            "b59_executive_lock",
            "b59_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 59)
        self.assertEqual(first["b_parent_level"], 58)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B59_PREFRONTAL_GOAL_CONTEXT_SELECTION_SOURCE,
        )
        self.assertEqual(first["b59_controller_profile"], "prefrontal_goal_context")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b60_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b59_prefrontal_goal_source(tmpdir)
            config = build_b60_orbitofrontal_outcome_value_config(
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=102,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b59_controller_profile",
            "b60_controller_profile",
            "b60_outcome_value",
            "b60_reversal_signal",
            "b60_goal_value_confidence",
            "b60_value_balance",
            "b60_value_lock",
            "b60_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 60)
        self.assertEqual(first["b_parent_level"], 59)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B60_ORBITOFRONTAL_OUTCOME_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b60_controller_profile"], "orbitofrontal_outcome_value")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b61_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b60_orbitofrontal_value_source(tmpdir)
            config = build_b61_amygdala_safety_value_config(
                B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=103,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b60_controller_profile",
            "b61_controller_profile",
            "b61_safety_value",
            "b61_threat_value",
            "b61_safety_confidence",
            "b61_affective_balance",
            "b61_safety_lock",
            "b61_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 61)
        self.assertEqual(first["b_parent_level"], 60)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B61_AMYGDALA_SAFETY_VALUE_SELECTION_SOURCE,
        )
        self.assertEqual(first["b61_controller_profile"], "amygdala_safety_value")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)

    def test_b62_trace_fields_and_primitive_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = _save_b61_amygdala_safety_source(tmpdir)
            config = build_b62_defensive_mode_selector_config(
                B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
                source_checkpoint=checkpoint,
            )
            sim = SpiderSimulation(
                seed=104,
                max_steps=3,
                module_dropout=0.0,
                brain_config=config,
            )
            _, trace = sim.run_episode(
                0,
                training=False,
                sample=False,
                capture_trace=True,
            )

        self.assertGreater(len(trace), 0)
        first = trace[0]
        for field in (
            "b_level",
            "b_parent_level",
            "b_transfer_source_checkpoint",
            "b_transfer_coverage",
            "b61_controller_profile",
            "b62_controller_profile",
            "b62_defensive_mode",
            "b62_freeze_pressure",
            "b62_flee_pressure",
            "b62_shelter_bias",
            "b62_defense_balance",
            "b62_defense_lock",
            "b62_decision",
            "semantic_action",
            "learned_semantic_action",
            "semantic_action_source",
            "bridge_primitive_action",
        ):
            self.assertIn(field, first)
        self.assertEqual(first["b_level"], 62)
        self.assertEqual(first["b_parent_level"], 61)
        self.assertEqual(first["b_transfer_source_checkpoint"], str(checkpoint))
        self.assertGreaterEqual(float(first["b_transfer_coverage"]), 0.50)
        self.assertEqual(
            first["semantic_action_source"],
            B62_DEFENSIVE_MODE_SELECTOR_SELECTION_SOURCE,
        )
        self.assertEqual(first["b62_controller_profile"], "defensive_mode_selector")
        ok, violations = trace_uses_only_primitive_actions(trace)
        self.assertTrue(ok, msg=violations)


class BSeriesEvolutionGateTest(unittest.TestCase):
    def _b4_result(
        self,
        episode: int,
        *,
        steps: int,
        alive: bool,
        food: int,
        sleep: int,
        shelter: int,
        contacts: int,
    ) -> dict[str, object]:
        return {
            "evaluation_episode": episode,
            "stats": SimpleNamespace(
                scenario="continuous_survival_canonical",
                steps=steps,
                alive=alive,
                food_eaten=food,
                sleep_events=sleep,
                shelter_entries=shelter,
                predator_contacts=contacts,
                final_health=1.0 if alive else 0.0,
                total_reward=0.0,
            ),
            "trace": [
                {
                    "tick": 0,
                    "action": "STAY",
                    "intended_action": "STAY",
                    "executed_action": "STAY",
                    "bridge_primitive_action": "STAY",
                }
            ],
        }

    def test_b4_canonical_gate_accepts_retention_without_forcing_improvement(
        self,
    ) -> None:
        results = [
            self._b4_result(
                0,
                steps=300,
                alive=True,
                food=14,
                sleep=54,
                shelter=10,
                contacts=2,
            ),
            self._b4_result(
                1,
                steps=300,
                alive=True,
                food=15,
                sleep=46,
                shelter=11,
                contacts=4,
            ),
            self._b4_result(
                2,
                steps=214,
                alive=False,
                food=8,
                sleep=37,
                shelter=8,
                contacts=1,
            ),
            self._b4_result(
                3,
                steps=244,
                alive=False,
                food=9,
                sleep=40,
                shelter=12,
                contacts=3,
            ),
            self._b4_result(
                4,
                steps=160,
                alive=False,
                food=6,
                sleep=18,
                shelter=3,
                contacts=2,
            ),
            self._b4_result(
                5,
                steps=160,
                alive=False,
                food=6,
                sleep=18,
                shelter=3,
                contacts=2,
            ),
            self._b4_result(
                6,
                steps=215,
                alive=False,
                food=8,
                sleep=37,
                shelter=7,
                contacts=1,
            ),
            self._b4_result(
                7,
                steps=300,
                alive=True,
                food=12,
                sleep=38,
                shelter=5,
                contacts=4,
            ),
            self._b4_result(
                8,
                steps=49,
                alive=False,
                food=2,
                sleep=0,
                shelter=2,
                contacts=1,
            ),
            self._b4_result(
                9,
                steps=300,
                alive=True,
                food=13,
                sleep=54,
                shelter=11,
                contacts=4,
            ),
        ]

        gate = b4_canonical_multi_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        aggregate = gate["aggregate"]
        self.assertEqual(aggregate["completed_horizons"], 4)
        self.assertEqual(aggregate["min_steps"], 49)
        self.assertEqual(aggregate["total_predator_contacts"], 24)
        self.assertEqual(aggregate["food_cycle_episodes"], 9)
        self.assertEqual(aggregate["sleep_cycle_episodes"], 9)
        self.assertEqual(aggregate["shelter_cycle_episodes"], 10)

    def test_b4_canonical_gate_rejects_b3_anchor_regression(self) -> None:
        results = [
            self._b4_result(
                0,
                steps=168,
                alive=False,
                food=8,
                sleep=21,
                shelter=2,
                contacts=0,
            ),
            self._b4_result(
                1,
                steps=300,
                alive=True,
                food=14,
                sleep=93,
                shelter=7,
                contacts=3,
            ),
            self._b4_result(
                2,
                steps=300,
                alive=True,
                food=8,
                sleep=37,
                shelter=8,
                contacts=1,
            ),
            self._b4_result(
                3,
                steps=300,
                alive=True,
                food=9,
                sleep=40,
                shelter=12,
                contacts=3,
            ),
            self._b4_result(
                4,
                steps=160,
                alive=False,
                food=6,
                sleep=18,
                shelter=3,
                contacts=2,
            ),
            self._b4_result(
                5,
                steps=160,
                alive=False,
                food=6,
                sleep=18,
                shelter=3,
                contacts=2,
            ),
            self._b4_result(
                6,
                steps=215,
                alive=False,
                food=8,
                sleep=37,
                shelter=7,
                contacts=1,
            ),
            self._b4_result(
                7,
                steps=300,
                alive=True,
                food=12,
                sleep=38,
                shelter=5,
                contacts=4,
            ),
            self._b4_result(
                8,
                steps=49,
                alive=False,
                food=2,
                sleep=0,
                shelter=2,
                contacts=1,
            ),
            self._b4_result(
                9,
                steps=300,
                alive=True,
                food=13,
                sleep=54,
                shelter=11,
                contacts=4,
            ),
        ]

        gate = b4_canonical_multi_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn("canonical_ep0:b3_anchor_completed_horizon", gate["failures"])

    def _b5_probe_result(
        self,
        episode: int,
        *,
        scenario: str,
        steps: int,
        alive: bool,
        food: int = 0,
        sleep: int = 0,
        contacts: int = 0,
        final_hunger: float = 0.50,
        final_sleep_debt: float = 0.04,
        food_distance_delta: float = 0.0,
    ) -> dict[str, object]:
        return {
            "evaluation_episode": episode,
            "stats": SimpleNamespace(
                episode=episode,
                seed=7,
                scenario=scenario,
                steps=steps,
                alive=alive,
                food_eaten=food,
                sleep_events=sleep,
                shelter_entries=0,
                predator_contacts=contacts,
                final_health=1.0 if alive else 0.0,
                final_hunger=final_hunger,
                final_sleep_debt=final_sleep_debt,
                food_distance_delta=food_distance_delta,
                total_reward=0.0,
            ),
            "trace": [
                {
                    "tick": 0,
                    "action": "STAY",
                    "intended_action": "STAY",
                    "executed_action": "STAY",
                    "bridge_primitive_action": "STAY",
                }
            ],
        }

    def _b6_probe_result(
        self,
        episode: int,
        *,
        scenario: str,
        steps: int,
        alive: bool,
        contacts: int = 0,
        food_distance_delta: float = 0.0,
        action_selection_payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        messages = []
        if action_selection_payload is not None:
            messages.append(
                {
                    "sender": "action_center",
                    "topic": "action.selection",
                    "payload": action_selection_payload,
                }
            )
        return {
            "evaluation_episode": episode,
            "stats": SimpleNamespace(
                episode=episode,
                seed=7,
                scenario=scenario,
                steps=steps,
                alive=alive,
                food_eaten=1 if scenario == "food_vs_predator_conflict" else 0,
                sleep_events=0,
                shelter_entries=0,
                predator_contacts=contacts,
                predator_mode_transitions=0,
                final_health=1.0 if alive else 0.0,
                final_hunger=0.50,
                final_sleep_debt=0.04,
                food_distance_delta=food_distance_delta,
                total_reward=0.0,
            ),
            "trace": [
                {
                    "tick": 0,
                    "action": "STAY",
                    "intended_action": "STAY",
                    "executed_action": "STAY",
                    "bridge_primitive_action": "STAY",
                    "state": {
                        "alive": alive,
                        "shelter_role": "outside",
                        "x": 4,
                        "y": 4,
                    },
                    "messages": messages,
                }
            ],
        }

    def _b7_corridor_result(
        self,
        episode: int,
        *,
        steps: int,
        alive: bool,
        contacts: int = 0,
        food_distance_delta: float = 0.0,
        final_health: float = 0.0,
        food: int = 0,
        decision: str | None = None,
    ) -> dict[str, object]:
        trace_item = {
            "tick": 0,
            "action": "STAY",
            "intended_action": "STAY",
            "executed_action": "STAY",
            "bridge_primitive_action": "STAY",
            "state": {
                "alive": alive,
                "shelter_role": "outside",
                "x": 4,
                "y": 4,
            },
        }
        if decision is not None:
            trace_item["b7_decision"] = decision
        return {
            "evaluation_episode": episode,
            "stats": SimpleNamespace(
                episode=episode,
                seed=7,
                scenario="corridor_gauntlet",
                steps=steps,
                alive=alive,
                food_eaten=food,
                sleep_events=0,
                shelter_entries=0,
                predator_contacts=contacts,
                predator_mode_transitions=0,
                final_health=final_health,
                final_hunger=0.50,
                final_sleep_debt=0.04,
                food_distance_delta=food_distance_delta,
                total_reward=0.0,
            ),
            "trace": [trace_item],
        }

    def _b8_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "corridor_continue_mapped",
        map_state: str | None = "food_vector_available",
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b7_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b7_decision"] = "continue_viable"
            item["b8_decision"] = decision
        if map_state is not None:
            item["b8_spatial_map_state"] = map_state
        return result

    def _b9_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "commit_food_waypoint",
        route_state: str | None = "food_waypoint_locked",
        waypoint_lock: int | None = 6,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b8_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b9_decision"] = decision
        if route_state is not None:
            item["b9_route_state"] = route_state
        if waypoint_lock is not None:
            item["b9_waypoint_lock"] = waypoint_lock
        return result

    def _b10_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "commit_replayed_route",
        replay_state: str | None = "prospective_food_plan",
        plan_commitment: int | None = 5,
        prospective_value: float | None = 0.60,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b9_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b10_decision"] = decision
        if replay_state is not None:
            item["b10_replay_state"] = replay_state
        if plan_commitment is not None:
            item["b10_plan_commitment"] = plan_commitment
        if prospective_value is not None:
            item["b10_prospective_value"] = prospective_value
        return result

    def _b11_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "commit_confident_plan",
        confidence_state: str | None = "high_confidence_plan",
        confidence_lock: int | None = 5,
        neuromod_signal: float | None = 0.60,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b10_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b11_decision"] = decision
        if confidence_state is not None:
            item["b11_confidence_state"] = confidence_state
        if confidence_lock is not None:
            item["b11_confidence_lock"] = confidence_lock
        if neuromod_signal is not None:
            item["b11_neuromod_signal"] = neuromod_signal
        return result

    def _b12_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "commit_attended_affordance",
        attention_state: str | None = "attended_food_affordance",
        search_lock: int | None = 5,
        attention_gain: float | None = 0.60,
        prediction_error: float | None = 0.10,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b11_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b12_decision"] = decision
        if attention_state is not None:
            item["b12_attention_state"] = attention_state
        if search_lock is not None:
            item["b12_search_lock"] = search_lock
        if attention_gain is not None:
            item["b12_attention_gain"] = attention_gain
        if prediction_error is not None:
            item["b12_prediction_error"] = prediction_error
        return result

    def _b13_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "commit_local_affordance_search",
        search_state: str | None = "local_route_viable",
        search_lock: int | None = 5,
        route_score: float | None = 0.60,
        affordance_samples: float | None = 0.45,
        dead_end_score: float | None = 0.10,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b12_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b13_decision"] = decision
        if search_state is not None:
            item["b13_search_state"] = search_state
        if search_lock is not None:
            item["b13_search_lock"] = search_lock
        if route_score is not None:
            item["b13_local_route_score"] = route_score
        if affordance_samples is not None:
            item["b13_affordance_samples"] = affordance_samples
        if dead_end_score is not None:
            item["b13_dead_end_score"] = dead_end_score
        return result

    def _b14_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "commit_confident_affordance",
        uncertainty_state: str | None = "confidence_calibrated_route",
        commitment_lock: int | None = 4,
        confidence: float | None = 0.65,
        uncertainty: float | None = 0.20,
        risk_adjusted_score: float | None = 0.55,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b13_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b14_decision"] = decision
        if uncertainty_state is not None:
            item["b14_uncertainty_state"] = uncertainty_state
        if commitment_lock is not None:
            item["b14_commitment_lock"] = commitment_lock
        if confidence is not None:
            item["b14_affordance_confidence"] = confidence
        if uncertainty is not None:
            item["b14_uncertainty"] = uncertainty
        if risk_adjusted_score is not None:
            item["b14_risk_adjusted_score"] = risk_adjusted_score
        return result

    def _b15_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "persist_food_option",
        option_state: str | None = "option_persist_food_route",
        option_lock: int | None = 5,
        option_value: float | None = 0.65,
        termination_pressure: float | None = 0.12,
        persistence_score: float | None = 0.70,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b14_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b15_decision"] = decision
        if option_state is not None:
            item["b15_option_state"] = option_state
        if option_lock is not None:
            item["b15_option_lock"] = option_lock
        if option_value is not None:
            item["b15_option_value"] = option_value
        if termination_pressure is not None:
            item["b15_termination_pressure"] = termination_pressure
        if persistence_score is not None:
            item["b15_persistence_score"] = persistence_score
        return result

    def _b16_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "ensemble_continue_option",
        ensemble_state: str | None = "ensemble_continue_consensus",
        ensemble_lock: int | None = 5,
        continue_vote: float | None = 0.70,
        return_vote: float | None = 0.20,
        consensus_score: float | None = 0.65,
        conflict_score: float | None = 0.50,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b15_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b16_decision"] = decision
        if ensemble_state is not None:
            item["b16_ensemble_state"] = ensemble_state
        if ensemble_lock is not None:
            item["b16_ensemble_lock"] = ensemble_lock
        if continue_vote is not None:
            item["b16_continue_vote"] = continue_vote
        if return_vote is not None:
            item["b16_return_vote"] = return_vote
        if consensus_score is not None:
            item["b16_consensus_score"] = consensus_score
        if conflict_score is not None:
            item["b16_conflict_score"] = conflict_score
        return result

    def _b17_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "neuromodulated_continue",
        modulator_state: str | None = "modulated_continue",
        modulation_lock: int | None = 5,
        arousal_signal: float | None = 0.62,
        homeostatic_gain: float | None = 0.55,
        option_gain: float | None = 0.70,
        conflict_release: float | None = 0.20,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b16_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b17_decision"] = decision
        if modulator_state is not None:
            item["b17_modulator_state"] = modulator_state
        if modulation_lock is not None:
            item["b17_modulation_lock"] = modulation_lock
        if arousal_signal is not None:
            item["b17_arousal_signal"] = arousal_signal
        if homeostatic_gain is not None:
            item["b17_homeostatic_gain"] = homeostatic_gain
        if option_gain is not None:
            item["b17_option_gain"] = option_gain
        if conflict_release is not None:
            item["b17_conflict_release"] = conflict_release
        return result

    def _b18_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "eligibility_stabilize_option",
        trace_state: str | None = "trace_stabilizes_option",
        trace_lock: int | None = 5,
        eligibility_trace: float | None = 0.62,
        prediction_proxy: float | None = 0.58,
        stability_bias: float | None = 0.70,
        switch_pressure: float | None = 0.20,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b17_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b18_decision"] = decision
        if trace_state is not None:
            item["b18_trace_state"] = trace_state
        if trace_lock is not None:
            item["b18_trace_lock"] = trace_lock
        if eligibility_trace is not None:
            item["b18_eligibility_trace"] = eligibility_trace
        if prediction_proxy is not None:
            item["b18_reward_prediction_proxy"] = prediction_proxy
        if stability_bias is not None:
            item["b18_stability_bias"] = stability_bias
        if switch_pressure is not None:
            item["b18_switch_pressure"] = switch_pressure
        return result

    def _b19_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "episodic_consolidate_option",
        memory_state: str | None = "memory_consolidates_option",
        memory_lock: int | None = 5,
        episode_memory: float | None = 0.62,
        consolidation_score: float | None = 0.58,
        stability_vote: float | None = 0.70,
        switch_suppression: float | None = 0.20,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b18_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b19_decision"] = decision
        if memory_state is not None:
            item["b19_memory_state"] = memory_state
        if memory_lock is not None:
            item["b19_memory_lock"] = memory_lock
        if episode_memory is not None:
            item["b19_episode_memory"] = episode_memory
        if consolidation_score is not None:
            item["b19_consolidation_score"] = consolidation_score
        if stability_vote is not None:
            item["b19_stability_vote"] = stability_vote
        if switch_suppression is not None:
            item["b19_switch_suppression"] = switch_suppression
        return result

    def _b20_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "working_memory_gate_continue",
        buffer_state: str | None = "working_memory_holds_context",
        buffer_lock: int | None = 5,
        working_buffer: float | None = 0.62,
        context_binding: float | None = 0.58,
        gate_vote: float | None = 0.70,
        release_vote: float | None = 0.20,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b19_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b20_decision"] = decision
        if buffer_state is not None:
            item["b20_buffer_state"] = buffer_state
        if buffer_lock is not None:
            item["b20_buffer_lock"] = buffer_lock
        if working_buffer is not None:
            item["b20_working_buffer"] = working_buffer
        if context_binding is not None:
            item["b20_context_binding"] = context_binding
        if gate_vote is not None:
            item["b20_gate_vote"] = gate_vote
        if release_vote is not None:
            item["b20_release_vote"] = release_vote
        return result

    def _b21_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "hippocampal_replay_continue",
        replay_state: str | None = "replay_rehearses_route",
        replay_lock: int | None = 5,
        sequence_memory: float | None = 0.62,
        replay_score: float | None = 0.58,
        route_commitment: float | None = 0.70,
        abort_prediction: float | None = 0.20,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b20_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b21_decision"] = decision
        if replay_state is not None:
            item["b21_replay_state"] = replay_state
        if replay_lock is not None:
            item["b21_replay_lock"] = replay_lock
        if sequence_memory is not None:
            item["b21_sequence_memory"] = sequence_memory
        if replay_score is not None:
            item["b21_replay_score"] = replay_score
        if route_commitment is not None:
            item["b21_route_commitment"] = route_commitment
        if abort_prediction is not None:
            item["b21_abort_prediction"] = abort_prediction
        return result

    def _b22_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "prospective_replay_continue",
        sim_state: str | None = "prospective_sim_commits_route",
        sim_lock: int | None = 5,
        prospective_sim: float | None = 0.62,
        forward_model_score: float | None = 0.58,
        viability_projection: float | None = 0.70,
        abort_projection: float | None = 0.20,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b21_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b22_decision"] = decision
        if sim_state is not None:
            item["b22_sim_state"] = sim_state
        if sim_lock is not None:
            item["b22_sim_lock"] = sim_lock
        if prospective_sim is not None:
            item["b22_prospective_sim"] = prospective_sim
        if forward_model_score is not None:
            item["b22_forward_model_score"] = forward_model_score
        if viability_projection is not None:
            item["b22_viability_projection"] = viability_projection
        if abort_projection is not None:
            item["b22_abort_projection"] = abort_projection
        return result

    def _b23_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "conflict_monitor_continue",
        conflict_state: str | None = "conflict_monitor_stabilizes_route",
        monitor_lock: int | None = 5,
        prediction_error: float | None = 0.18,
        conflict_memory: float | None = 0.20,
        stability_vote: float | None = 0.62,
        abort_bias: float | None = 0.16,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b22_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b23_decision"] = decision
        if conflict_state is not None:
            item["b23_conflict_state"] = conflict_state
        if monitor_lock is not None:
            item["b23_monitor_lock"] = monitor_lock
        if prediction_error is not None:
            item["b23_prediction_error"] = prediction_error
        if conflict_memory is not None:
            item["b23_conflict_memory"] = conflict_memory
        if stability_vote is not None:
            item["b23_stability_vote"] = stability_vote
        if abort_bias is not None:
            item["b23_abort_bias"] = abort_bias
        return result

    def _b24_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "precision_conflict_continue",
        precision_state: str | None = "precision_conflict_stabilizes_route",
        precision_lock: int | None = 5,
        precision_memory: float | None = 0.58,
        precision_vote: float | None = 0.60,
        uncertainty_pressure: float | None = 0.18,
        abort_precision: float | None = 0.16,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b23_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b24_decision"] = decision
        if precision_state is not None:
            item["b24_precision_state"] = precision_state
        if precision_lock is not None:
            item["b24_precision_lock"] = precision_lock
        if precision_memory is not None:
            item["b24_precision_memory"] = precision_memory
        if precision_vote is not None:
            item["b24_precision_vote"] = precision_vote
        if uncertainty_pressure is not None:
            item["b24_uncertainty_pressure"] = uncertainty_pressure
        if abort_precision is not None:
            item["b24_abort_precision"] = abort_precision
        return result

    def _b25_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "metacognitive_confidence_continue",
        metacognitive_state: str | None = "metacognition_confirms_route",
        meta_lock: int | None = 5,
        confidence_memory: float | None = 0.54,
        confidence_vote: float | None = 0.56,
        doubt_pressure: float | None = 0.12,
        control_gain: float | None = 0.40,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b24_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b25_decision"] = decision
        if metacognitive_state is not None:
            item["b25_metacognitive_state"] = metacognitive_state
        if meta_lock is not None:
            item["b25_meta_lock"] = meta_lock
        if confidence_memory is not None:
            item["b25_confidence_memory"] = confidence_memory
        if confidence_vote is not None:
            item["b25_confidence_vote"] = confidence_vote
        if doubt_pressure is not None:
            item["b25_doubt_pressure"] = doubt_pressure
        if control_gain is not None:
            item["b25_control_gain"] = control_gain
        return result

    def _b26_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "allostatic_prediction_continue",
        allostatic_state: str | None = "allostasis_confirms_route",
        stability_lock: int | None = 5,
        prediction_error: float | None = 0.18,
        setpoint_pressure: float | None = 0.34,
        control_vote: float | None = 0.42,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b25_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b26_decision"] = decision
        if allostatic_state is not None:
            item["b26_allostatic_state"] = allostatic_state
        if stability_lock is not None:
            item["b26_stability_lock"] = stability_lock
        if prediction_error is not None:
            item["b26_prediction_error"] = prediction_error
        if setpoint_pressure is not None:
            item["b26_setpoint_pressure"] = setpoint_pressure
        if control_vote is not None:
            item["b26_control_vote"] = control_vote
        return result

    def _b27_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "arousal_gain_continue",
        arousal_state: str | None = "arousal_gain_stabilizes_route",
        arousal_lock: int | None = 5,
        arousal_level: float | None = 0.42,
        gain_modulation: float | None = 0.40,
        stress_pressure: float | None = 0.14,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b26_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b27_decision"] = decision
        if arousal_state is not None:
            item["b27_arousal_state"] = arousal_state
        if arousal_lock is not None:
            item["b27_arousal_lock"] = arousal_lock
        if arousal_level is not None:
            item["b27_arousal_level"] = arousal_level
        if gain_modulation is not None:
            item["b27_gain_modulation"] = gain_modulation
        if stress_pressure is not None:
            item["b27_stress_pressure"] = stress_pressure
        return result

    def _b28_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "interoceptive_attention_continue",
        attention_state: str | None = "interoceptive_attention_stabilizes_route",
        attention_lock: int | None = 5,
        interoceptive_focus: float | None = 0.42,
        attention_gain: float | None = 0.40,
        distractor_pressure: float | None = 0.14,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b27_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b28_decision"] = decision
        if attention_state is not None:
            item["b28_attention_state"] = attention_state
        if attention_lock is not None:
            item["b28_attention_lock"] = attention_lock
        if interoceptive_focus is not None:
            item["b28_interoceptive_focus"] = interoceptive_focus
        if attention_gain is not None:
            item["b28_attention_gain"] = attention_gain
        if distractor_pressure is not None:
            item["b28_distractor_pressure"] = distractor_pressure
        return result

    def _b29_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "salience_competition_continue",
        salience_state: str | None = "corridor_salience_wins",
        salience_lock: int | None = 5,
        threat_salience: float | None = 0.12,
        homeostatic_salience: float | None = 0.48,
        corridor_salience: float | None = 0.42,
        winner_channel: str | None = "corridor",
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b28_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b29_decision"] = decision
        if salience_state is not None:
            item["b29_salience_state"] = salience_state
        if salience_lock is not None:
            item["b29_salience_lock"] = salience_lock
        if threat_salience is not None:
            item["b29_threat_salience"] = threat_salience
        if homeostatic_salience is not None:
            item["b29_homeostatic_salience"] = homeostatic_salience
        if corridor_salience is not None:
            item["b29_corridor_salience"] = corridor_salience
        if winner_channel is not None:
            item["b29_winner_channel"] = winner_channel
        return result

    def _b30_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "basal_gate_go",
        gate_state: str | None = "basal_go_gate_opens",
        gate_lock: int | None = 5,
        go_signal: float | None = 0.42,
        no_go_signal: float | None = 0.12,
        action_gate: str | None = "go",
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b29_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b30_decision"] = decision
        if gate_state is not None:
            item["b30_gate_state"] = gate_state
        if gate_lock is not None:
            item["b30_gate_lock"] = gate_lock
        if go_signal is not None:
            item["b30_go_signal"] = go_signal
        if no_go_signal is not None:
            item["b30_no_go_signal"] = no_go_signal
        if action_gate is not None:
            item["b30_action_gate"] = action_gate
        return result

    def _b31_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "dopamine_gate_go",
        dopamine_state: str | None = "dopamine_go_bias_stabilizes",
        dopamine_lock: int | None = 5,
        reward_prediction_error: float | None = 0.18,
        tonic_dopamine: float | None = 0.32,
        phasic_dopamine: float | None = 0.24,
        gate_bias: float | None = 0.42,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b30_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b31_decision"] = decision
        if dopamine_state is not None:
            item["b31_dopamine_state"] = dopamine_state
        if dopamine_lock is not None:
            item["b31_dopamine_lock"] = dopamine_lock
        if reward_prediction_error is not None:
            item["b31_reward_prediction_error"] = reward_prediction_error
        if tonic_dopamine is not None:
            item["b31_tonic_dopamine"] = tonic_dopamine
        if phasic_dopamine is not None:
            item["b31_phasic_dopamine"] = phasic_dopamine
        if gate_bias is not None:
            item["b31_gate_bias"] = gate_bias
        return result

    def _b32_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "actor_critic_commit",
        critic_value: float | None = 0.36,
        actor_advantage: float | None = 0.28,
        value_error: float | None = 0.16,
        policy_bias: float | None = 0.44,
        value_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b31_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b32_decision"] = decision
        if critic_value is not None:
            item["b32_critic_value"] = critic_value
        if actor_advantage is not None:
            item["b32_actor_advantage"] = actor_advantage
        if value_error is not None:
            item["b32_value_error"] = value_error
        if policy_bias is not None:
            item["b32_policy_bias"] = policy_bias
        if value_lock is not None:
            item["b32_value_lock"] = value_lock
        return result

    def _b33_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "td_error_commit",
        td_error: float | None = 0.24,
        bootstrap_value: float | None = 0.34,
        reward_trace: float | None = 0.22,
        actor_update: float | None = 0.38,
        td_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b32_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b33_decision"] = decision
        if td_error is not None:
            item["b33_td_error"] = td_error
        if bootstrap_value is not None:
            item["b33_bootstrap_value"] = bootstrap_value
        if reward_trace is not None:
            item["b33_reward_trace"] = reward_trace
        if actor_update is not None:
            item["b33_actor_update"] = actor_update
        if td_lock is not None:
            item["b33_td_lock"] = td_lock
        return result

    def _b34_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "eligibility_credit_commit",
        eligibility_trace: float | None = 0.28,
        credit_assignment: float | None = 0.24,
        synaptic_tag: float | None = 0.18,
        decay_memory: float | None = 0.12,
        credit_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b33_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b34_decision"] = decision
        if eligibility_trace is not None:
            item["b34_eligibility_trace"] = eligibility_trace
        if credit_assignment is not None:
            item["b34_credit_assignment"] = credit_assignment
        if synaptic_tag is not None:
            item["b34_synaptic_tag"] = synaptic_tag
        if decay_memory is not None:
            item["b34_decay_memory"] = decay_memory
        if credit_lock is not None:
            item["b34_credit_lock"] = credit_lock
        return result

    def _b35_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "forward_model_commit",
        forward_value: float | None = 0.30,
        transition_error: float | None = 0.20,
        model_confidence: float | None = 0.18,
        prediction_memory: float | None = 0.24,
        model_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b34_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b35_decision"] = decision
        if forward_value is not None:
            item["b35_forward_value"] = forward_value
        if transition_error is not None:
            item["b35_transition_error"] = transition_error
        if model_confidence is not None:
            item["b35_model_confidence"] = model_confidence
        if prediction_memory is not None:
            item["b35_prediction_memory"] = prediction_memory
        if model_lock is not None:
            item["b35_model_lock"] = model_lock
        return result

    def _b36_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "latent_belief_commit",
        latent_state: float | None = 0.28,
        belief_error: float | None = 0.20,
        state_confidence: float | None = 0.18,
        context_memory: float | None = 0.16,
        belief_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b35_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b36_decision"] = decision
        if latent_state is not None:
            item["b36_latent_state"] = latent_state
        if belief_error is not None:
            item["b36_belief_error"] = belief_error
        if state_confidence is not None:
            item["b36_state_confidence"] = state_confidence
        if context_memory is not None:
            item["b36_context_memory"] = context_memory
        if belief_lock is not None:
            item["b36_belief_lock"] = belief_lock
        return result

    def _b37_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "state_factor_commit",
        external_factor: float | None = 0.24,
        internal_factor: float | None = 0.22,
        factor_alignment: float | None = 0.20,
        factor_confidence: float | None = 0.18,
        factor_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b36_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b37_decision"] = decision
        if external_factor is not None:
            item["b37_external_state_factor"] = external_factor
        if internal_factor is not None:
            item["b37_internal_state_factor"] = internal_factor
        if factor_alignment is not None:
            item["b37_factor_alignment"] = factor_alignment
        if factor_confidence is not None:
            item["b37_factor_confidence"] = factor_confidence
        if factor_lock is not None:
            item["b37_factor_lock"] = factor_lock
        return result

    def _b38_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "factor_attention_commit",
        external_attention: float | None = 0.24,
        internal_attention: float | None = 0.22,
        attention_balance: float | None = 0.20,
        attention_gain: float | None = 0.18,
        attention_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b37_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b38_decision"] = decision
        if external_attention is not None:
            item["b38_external_attention"] = external_attention
        if internal_attention is not None:
            item["b38_internal_attention"] = internal_attention
        if attention_balance is not None:
            item["b38_attention_balance"] = attention_balance
        if attention_gain is not None:
            item["b38_attention_gain"] = attention_gain
        if attention_lock is not None:
            item["b38_attention_lock"] = attention_lock
        return result

    def _b39_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "attention_binding_commit",
        binding_strength: float | None = 0.24,
        cross_factor_coherence: float | None = 0.22,
        bound_context: float | None = 0.20,
        binding_gain: float | None = 0.18,
        binding_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b38_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b39_decision"] = decision
        if binding_strength is not None:
            item["b39_binding_strength"] = binding_strength
        if cross_factor_coherence is not None:
            item["b39_cross_factor_coherence"] = cross_factor_coherence
        if bound_context is not None:
            item["b39_bound_context"] = bound_context
        if binding_gain is not None:
            item["b39_binding_gain"] = binding_gain
        if binding_lock is not None:
            item["b39_binding_lock"] = binding_lock
        return result

    def _b40_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "global_workspace_commit",
        workspace_activation: float | None = 0.24,
        broadcast_gain: float | None = 0.22,
        context_availability: float | None = 0.20,
        workspace_stability: float | None = 0.18,
        workspace_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b39_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b40_decision"] = decision
        if workspace_activation is not None:
            item["b40_workspace_activation"] = workspace_activation
        if broadcast_gain is not None:
            item["b40_broadcast_gain"] = broadcast_gain
        if context_availability is not None:
            item["b40_context_availability"] = context_availability
        if workspace_stability is not None:
            item["b40_workspace_stability"] = workspace_stability
        if workspace_lock is not None:
            item["b40_workspace_lock"] = workspace_lock
        return result

    def _b41_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "executive_workspace_select",
        executive_selection: float | None = 0.24,
        inhibitory_pressure: float | None = 0.02,
        goal_context: float | None = 0.20,
        executive_stability: float | None = 0.18,
        executive_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b40_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b41_decision"] = decision
        if executive_selection is not None:
            item["b41_executive_selection"] = executive_selection
        if inhibitory_pressure is not None:
            item["b41_inhibitory_pressure"] = inhibitory_pressure
        if goal_context is not None:
            item["b41_goal_context"] = goal_context
        if executive_stability is not None:
            item["b41_executive_stability"] = executive_stability
        if executive_lock is not None:
            item["b41_executive_lock"] = executive_lock
        return result

    def _b42_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "error_monitor_commit",
        error_signal: float | None = 0.04,
        conflict_signal: float | None = 0.02,
        performance_context: float | None = 0.22,
        monitor_stability: float | None = 0.18,
        monitor_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b41_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b42_decision"] = decision
        if error_signal is not None:
            item["b42_error_signal"] = error_signal
        if conflict_signal is not None:
            item["b42_conflict_signal"] = conflict_signal
        if performance_context is not None:
            item["b42_performance_context"] = performance_context
        if monitor_stability is not None:
            item["b42_monitor_stability"] = monitor_stability
        if monitor_lock is not None:
            item["b42_monitor_lock"] = monitor_lock
        return result

    def _b43_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "adaptive_precision_commit",
        precision_signal: float | None = 0.19,
        adaptive_threshold: float | None = 0.12,
        arousal_context: float | None = 0.04,
        control_stability: float | None = 0.21,
        precision_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b42_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b43_decision"] = decision
        if precision_signal is not None:
            item["b43_precision_signal"] = precision_signal
        if adaptive_threshold is not None:
            item["b43_adaptive_threshold"] = adaptive_threshold
        if arousal_context is not None:
            item["b43_arousal_context"] = arousal_context
        if control_stability is not None:
            item["b43_control_stability"] = control_stability
        if precision_lock is not None:
            item["b43_precision_lock"] = precision_lock
        return result

    def _b44_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "thalamic_relay_commit",
        relay_gate: float | None = 0.18,
        sensory_precision: float | None = 0.15,
        context_relay: float | None = 0.16,
        gate_stability: float | None = 0.20,
        relay_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b43_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b44_decision"] = decision
        if relay_gate is not None:
            item["b44_relay_gate"] = relay_gate
        if sensory_precision is not None:
            item["b44_sensory_precision"] = sensory_precision
        if context_relay is not None:
            item["b44_context_relay"] = context_relay
        if gate_stability is not None:
            item["b44_gate_stability"] = gate_stability
        if relay_lock is not None:
            item["b44_relay_lock"] = relay_lock
        return result

    def _b45_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "reticular_inhibition_commit",
        inhibitory_gate: float | None = 0.18,
        sensory_filter: float | None = 0.15,
        context_suppression: float | None = 0.16,
        loop_stability: float | None = 0.20,
        inhibition_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b44_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b45_decision"] = decision
        if inhibitory_gate is not None:
            item["b45_inhibitory_gate"] = inhibitory_gate
        if sensory_filter is not None:
            item["b45_sensory_filter"] = sensory_filter
        if context_suppression is not None:
            item["b45_context_suppression"] = context_suppression
        if loop_stability is not None:
            item["b45_loop_stability"] = loop_stability
        if inhibition_lock is not None:
            item["b45_inhibition_lock"] = inhibition_lock
        return result

    def _b46_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "corticothalamic_feedback_commit",
        feedback_gain: float | None = 0.22,
        topdown_context: float | None = 0.18,
        prediction_match: float | None = 0.20,
        feedback_stability: float | None = 0.21,
        feedback_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b45_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b46_decision"] = decision
        if feedback_gain is not None:
            item["b46_feedback_gain"] = feedback_gain
        if topdown_context is not None:
            item["b46_topdown_context"] = topdown_context
        if prediction_match is not None:
            item["b46_prediction_match"] = prediction_match
        if feedback_stability is not None:
            item["b46_feedback_stability"] = feedback_stability
        if feedback_lock is not None:
            item["b46_feedback_lock"] = feedback_lock
        return result

    def _b47_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "oscillatory_synchrony_commit",
        phase_alignment: float | None = 0.21,
        synchrony_gain: float | None = 0.22,
        cross_loop_coherence: float | None = 0.18,
        phase_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b46_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b47_decision"] = decision
        if phase_alignment is not None:
            item["b47_phase_alignment"] = phase_alignment
        if synchrony_gain is not None:
            item["b47_synchrony_gain"] = synchrony_gain
        if cross_loop_coherence is not None:
            item["b47_cross_loop_coherence"] = cross_loop_coherence
        if phase_lock is not None:
            item["b47_phase_lock"] = phase_lock
        return result

    def _b48_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "cerebellar_timing_commit",
        timing_error: float | None = 0.20,
        predictive_timing: float | None = 0.22,
        corrective_gain: float | None = 0.19,
        calibration_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b47_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b48_decision"] = decision
        if timing_error is not None:
            item["b48_timing_error"] = timing_error
        if predictive_timing is not None:
            item["b48_predictive_timing"] = predictive_timing
        if corrective_gain is not None:
            item["b48_corrective_gain"] = corrective_gain
        if calibration_lock is not None:
            item["b48_calibration_lock"] = calibration_lock
        return result

    def _b49_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "striatal_gate_commit",
        go_signal: float | None = 0.23,
        no_go_signal: float | None = 0.04,
        action_gate_balance: float | None = 0.21,
        selection_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b48_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b49_decision"] = decision
        if go_signal is not None:
            item["b49_go_signal"] = go_signal
        if no_go_signal is not None:
            item["b49_no_go_signal"] = no_go_signal
        if action_gate_balance is not None:
            item["b49_action_gate_balance"] = action_gate_balance
        if selection_lock is not None:
            item["b49_selection_lock"] = selection_lock
        return result

    def _b50_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "habit_chunk_commit",
        habit_strength: float | None = 0.23,
        chunk_value: float | None = 0.21,
        habit_stability: float | None = 0.20,
        chunk_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b49_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b50_decision"] = decision
        if habit_strength is not None:
            item["b50_habit_strength"] = habit_strength
        if chunk_value is not None:
            item["b50_chunk_value"] = chunk_value
        if habit_stability is not None:
            item["b50_habit_stability"] = habit_stability
        if chunk_lock is not None:
            item["b50_chunk_lock"] = chunk_lock
        return result

    def _b51_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "dopamine_habit_commit",
        prediction_error: float | None = 0.20,
        dopamine_gain: float | None = 0.22,
        habit_modulation: float | None = 0.21,
        modulation_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b50_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b51_decision"] = decision
        if prediction_error is not None:
            item["b51_prediction_error"] = prediction_error
        if dopamine_gain is not None:
            item["b51_dopamine_gain"] = dopamine_gain
        if habit_modulation is not None:
            item["b51_habit_modulation"] = habit_modulation
        if modulation_lock is not None:
            item["b51_modulation_lock"] = modulation_lock
        return result

    def _b52_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "cholinergic_precision_commit",
        acetylcholine_level: float | None = 0.20,
        precision_gain: float | None = 0.22,
        uncertainty_signal: float | None = 0.18,
        attention_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b51_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b52_decision"] = decision
        if acetylcholine_level is not None:
            item["b52_acetylcholine_level"] = acetylcholine_level
        if precision_gain is not None:
            item["b52_precision_gain"] = precision_gain
        if uncertainty_signal is not None:
            item["b52_uncertainty_signal"] = uncertainty_signal
        if attention_lock is not None:
            item["b52_attention_lock"] = attention_lock
        return result

    def _b53_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "noradrenergic_arousal_commit",
        norepinephrine_level: float | None = 0.20,
        arousal_gain: float | None = 0.22,
        surprise_signal: float | None = 0.18,
        gain_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b52_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b53_decision"] = decision
        if norepinephrine_level is not None:
            item["b53_norepinephrine_level"] = norepinephrine_level
        if arousal_gain is not None:
            item["b53_arousal_gain"] = arousal_gain
        if surprise_signal is not None:
            item["b53_surprise_signal"] = surprise_signal
        if gain_lock is not None:
            item["b53_gain_lock"] = gain_lock
        return result

    def _b54_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "serotonergic_patience_commit",
        serotonin_level: float | None = 0.20,
        patience_signal: float | None = 0.22,
        impulse_suppression: float | None = 0.18,
        patience_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b53_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b54_decision"] = decision
        if serotonin_level is not None:
            item["b54_serotonin_level"] = serotonin_level
        if patience_signal is not None:
            item["b54_patience_signal"] = patience_signal
        if impulse_suppression is not None:
            item["b54_impulse_suppression"] = impulse_suppression
        if patience_lock is not None:
            item["b54_patience_lock"] = patience_lock
        return result

    def _b55_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "hypothalamic_drive_commit",
        hypothalamic_drive: float | None = 0.24,
        satiety_signal: float | None = 0.16,
        recovery_bias: float | None = 0.14,
        drive_balance: float | None = 0.20,
        drive_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b54_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b55_decision"] = decision
        if hypothalamic_drive is not None:
            item["b55_hypothalamic_drive"] = hypothalamic_drive
        if satiety_signal is not None:
            item["b55_satiety_signal"] = satiety_signal
        if recovery_bias is not None:
            item["b55_recovery_bias"] = recovery_bias
        if drive_balance is not None:
            item["b55_drive_balance"] = drive_balance
        if drive_lock is not None:
            item["b55_drive_lock"] = drive_lock
        return result

    def _b56_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "hpa_stress_axis_commit",
        cortisol_level: float | None = 0.18,
        stress_load: float | None = 0.16,
        recovery_signal: float | None = 0.14,
        endocrine_balance: float | None = 0.20,
        stress_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b55_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b56_decision"] = decision
        if cortisol_level is not None:
            item["b56_cortisol_level"] = cortisol_level
        if stress_load is not None:
            item["b56_stress_load"] = stress_load
        if recovery_signal is not None:
            item["b56_recovery_signal"] = recovery_signal
        if endocrine_balance is not None:
            item["b56_endocrine_balance"] = endocrine_balance
        if stress_lock is not None:
            item["b56_stress_lock"] = stress_lock
        return result

    def _b57_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "interoceptive_awareness_commit",
        interoceptive_awareness: float | None = 0.22,
        visceral_salience: float | None = 0.20,
        body_state_confidence: float | None = 0.18,
        awareness_balance: float | None = 0.21,
        awareness_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b56_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b57_decision"] = decision
        if interoceptive_awareness is not None:
            item["b57_interoceptive_awareness"] = interoceptive_awareness
        if visceral_salience is not None:
            item["b57_visceral_salience"] = visceral_salience
        if body_state_confidence is not None:
            item["b57_body_state_confidence"] = body_state_confidence
        if awareness_balance is not None:
            item["b57_awareness_balance"] = awareness_balance
        if awareness_lock is not None:
            item["b57_awareness_lock"] = awareness_lock
        return result

    def _b58_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "acc_conflict_commit",
        conflict_signal: float | None = 0.22,
        error_likelihood: float | None = 0.18,
        control_allocation: float | None = 0.20,
        resolution_balance: float | None = 0.21,
        conflict_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b57_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b58_decision"] = decision
        if conflict_signal is not None:
            item["b58_conflict_signal"] = conflict_signal
        if error_likelihood is not None:
            item["b58_error_likelihood"] = error_likelihood
        if control_allocation is not None:
            item["b58_control_allocation"] = control_allocation
        if resolution_balance is not None:
            item["b58_resolution_balance"] = resolution_balance
        if conflict_lock is not None:
            item["b58_conflict_lock"] = conflict_lock
        return result

    def _b59_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "prefrontal_goal_commit",
        goal_context: float | None = 0.22,
        working_set_stability: float | None = 0.20,
        task_set_confidence: float | None = 0.18,
        executive_balance: float | None = 0.21,
        executive_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b58_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b59_decision"] = decision
        if goal_context is not None:
            item["b59_goal_context"] = goal_context
        if working_set_stability is not None:
            item["b59_working_set_stability"] = working_set_stability
        if task_set_confidence is not None:
            item["b59_task_set_confidence"] = task_set_confidence
        if executive_balance is not None:
            item["b59_executive_balance"] = executive_balance
        if executive_lock is not None:
            item["b59_executive_lock"] = executive_lock
        return result

    def _b60_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "orbitofrontal_value_commit",
        outcome_value: float | None = 0.22,
        reversal_signal: float | None = 0.18,
        goal_value_confidence: float | None = 0.20,
        value_balance: float | None = 0.21,
        value_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b59_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b60_decision"] = decision
        if outcome_value is not None:
            item["b60_outcome_value"] = outcome_value
        if reversal_signal is not None:
            item["b60_reversal_signal"] = reversal_signal
        if goal_value_confidence is not None:
            item["b60_goal_value_confidence"] = goal_value_confidence
        if value_balance is not None:
            item["b60_value_balance"] = value_balance
        if value_lock is not None:
            item["b60_value_lock"] = value_lock
        return result

    def _b61_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "amygdala_safety_commit",
        safety_value: float | None = 0.22,
        threat_value: float | None = 0.0,
        safety_confidence: float | None = 0.20,
        affective_balance: float | None = 0.21,
        safety_lock: int | None = 5,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b60_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b61_decision"] = decision
        if safety_value is not None:
            item["b61_safety_value"] = safety_value
        if threat_value is not None:
            item["b61_threat_value"] = threat_value
        if safety_confidence is not None:
            item["b61_safety_confidence"] = safety_confidence
        if affective_balance is not None:
            item["b61_affective_balance"] = affective_balance
        if safety_lock is not None:
            item["b61_safety_lock"] = safety_lock
        return result

    def _b62_corridor_result(
        self,
        episode: int,
        *,
        decision: str | None = "defensive_safe_advance",
        defensive_mode: str | None = "safe_advance",
        freeze_pressure: float | None = 0.05,
        flee_pressure: float | None = 0.08,
        shelter_bias: float | None = 0.12,
        defense_balance: float | None = -0.04,
        defense_lock: int | None = 0,
        **kwargs: object,
    ) -> dict[str, object]:
        result = self._b61_corridor_result(episode, **kwargs)
        item = result["trace"][0]
        if decision is not None:
            item["b62_decision"] = decision
        if defensive_mode is not None:
            item["b62_defensive_mode"] = defensive_mode
        if freeze_pressure is not None:
            item["b62_freeze_pressure"] = freeze_pressure
        if flee_pressure is not None:
            item["b62_flee_pressure"] = flee_pressure
        if shelter_bias is not None:
            item["b62_shelter_bias"] = shelter_bias
        if defense_balance is not None:
            item["b62_defense_balance"] = defense_balance
        if defense_lock is not None:
            item["b62_defense_lock"] = defense_lock
        return result

    def test_b5_food_deprivation_gate_accepts_raw_probe_progress(self) -> None:
        results = [
            self._b5_probe_result(
                0,
                scenario="food_deprivation",
                steps=22,
                alive=True,
                food=1,
                final_hunger=0.88,
                food_distance_delta=0.0,
            ),
            self._b5_probe_result(
                1,
                scenario="food_deprivation",
                steps=22,
                alive=True,
                food=2,
                final_hunger=0.56,
                food_distance_delta=0.0,
            ),
            self._b5_probe_result(
                2,
                scenario="food_deprivation",
                steps=22,
                alive=True,
                food=1,
                final_hunger=0.89,
                food_distance_delta=3.0,
            ),
        ]

        gate = b5_food_deprivation_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["progress_episodes"], 2)

    def test_b5_sleep_conflict_gate_rejects_missing_recovery_movement(self) -> None:
        results = [
            self._b5_probe_result(
                episode,
                scenario="sleep_vs_exploration_conflict",
                steps=18,
                alive=True,
                sleep=8,
                final_sleep_debt=0.04,
            )
            for episode in range(3)
        ]

        gate = b5_sleep_conflict_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "sleep_conflict_aggregate:post_recovery_movement_episodes",
            gate["failures"],
        )

    def test_b6_food_predator_gate_accepts_partial_threat_progress(self) -> None:
        payload = {
            "winning_valence": "threat",
            "evidence": {
                "threat": {
                    "predator_visible": 1.0,
                    "predator_proximity": 0.5,
                    "predator_certainty": 0.5,
                }
            },
            "module_gates": {"hunger_center": 0.2},
        }
        results = [
            self._b6_probe_result(
                episode,
                scenario="food_vs_predator_conflict",
                steps=16,
                alive=True,
                action_selection_payload=payload if episode < 2 else None,
            )
            for episode in range(3)
        ]

        gate = b6_food_predator_conflict_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["threat_exposure_episodes"], 2)
        self.assertEqual(
            gate["aggregate"]["threat_priority_or_suppression_episodes"],
            2,
        )

    def test_b6_food_predator_gate_rejects_absent_threat_progress(self) -> None:
        results = [
            self._b6_probe_result(
                episode,
                scenario="food_vs_predator_conflict",
                steps=16,
                alive=True,
            )
            for episode in range(3)
        ]

        gate = b6_food_predator_conflict_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "food_predator_aggregate:threat_exposure_episodes",
            gate["failures"],
        )

    def test_b6_corridor_gate_accepts_partial_survival_progress(self) -> None:
        results = [
            self._b6_probe_result(
                0,
                scenario="corridor_gauntlet",
                steps=16,
                alive=False,
                food_distance_delta=12.0,
            ),
            self._b6_probe_result(
                1,
                scenario="corridor_gauntlet",
                steps=17,
                alive=False,
                food_distance_delta=10.0,
            ),
            self._b6_probe_result(
                2,
                scenario="corridor_gauntlet",
                steps=14,
                alive=False,
                food_distance_delta=0.0,
            ),
        ]

        gate = b6_corridor_progress_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["progress_episodes"], 2)
        self.assertEqual(gate["aggregate"]["survival_progress_episodes"], 2)

    def test_b6_corridor_gate_rejects_b5_baseline_steps(self) -> None:
        results = [
            self._b6_probe_result(
                episode,
                scenario="corridor_gauntlet",
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b6_corridor_progress_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_aggregate:survival_progress_episodes",
            gate["failures"],
        )

    def test_b6_corridor_gate_accepts_food_progress_over_b5(self) -> None:
        results = [
            self._b6_probe_result(
                episode,
                scenario="corridor_gauntlet",
                steps=14,
                alive=False,
                food_distance_delta=14.0,
            )
            for episode in range(3)
        ]

        gate = b6_corridor_progress_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["survival_progress_episodes"], 3)
        checks = gate["episode_results"][0]["gate"]["checks"]
        self.assertTrue(checks["food_progress_over_b5"])

    def test_b7_corridor_gate_accepts_explicit_viability_progress(self) -> None:
        results = [
            self._b7_corridor_result(
                0,
                steps=14,
                alive=False,
                food_distance_delta=13.0,
                decision="abort_return_unviable",
            ),
            self._b7_corridor_result(
                1,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision="continue_viable",
            ),
            self._b7_corridor_result(
                2,
                steps=14,
                alive=False,
                food_distance_delta=0.0,
            ),
        ]

        gate = b7_corridor_viability_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["food_progress_episodes"], 2)
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 2)
        self.assertEqual(gate["aggregate"]["improvement_episodes"], 2)

    def test_b7_corridor_gate_rejects_clone_without_explicit_decision(self) -> None:
        results = [
            self._b7_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b7_corridor_viability_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_aggregate:explicit_decision_episodes",
            gate["failures"],
        )

    def test_b7_corridor_gate_rejects_contacts_or_absent_improvement(self) -> None:
        contact_gate = b7_corridor_viability_gate_result(
            [
                self._b7_corridor_result(
                    0,
                    steps=15,
                    alive=False,
                    contacts=1,
                    food_distance_delta=13.0,
                    decision="continue_viable",
                ),
                self._b7_corridor_result(
                    1,
                    steps=15,
                    alive=False,
                    food_distance_delta=13.0,
                    decision="continue_viable",
                ),
                self._b7_corridor_result(
                    2,
                    steps=15,
                    alive=False,
                    food_distance_delta=13.0,
                    decision="continue_viable",
                ),
            ]
        )
        self.assertFalse(contact_gate["passed"])
        self.assertIn("corridor_ep0:predator_contacts", contact_gate["failures"])

        no_improvement_gate = b7_corridor_viability_gate_result(
            [
                self._b7_corridor_result(
                    episode,
                    steps=14,
                    alive=False,
                    food_distance_delta=13.0,
                    decision="continue_viable",
                )
                for episode in range(3)
            ]
        )
        self.assertFalse(no_improvement_gate["passed"])
        self.assertIn(
            "corridor_aggregate:improvement_episodes",
            no_improvement_gate["failures"],
        )

    def test_b8_corridor_gate_accepts_spatial_map_progress(self) -> None:
        results = [
            self._b8_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b8_spatial_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["spatial_map_episodes"], 3)
        self.assertEqual(gate["aggregate"]["mapped_progress_episodes"], 3)

    def test_b8_corridor_gate_rejects_b7_clone_without_map(self) -> None:
        results = [
            self._b8_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                map_state=None,
            )
            for episode in range(3)
        ]

        gate = b8_spatial_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b8_aggregate:explicit_b8_decision_episodes",
            gate["failures"],
        )

    def test_b9_corridor_gate_accepts_waypoint_route_progress(self) -> None:
        results = [
            self._b9_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b9_waypoint_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["route_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["locked_waypoint_episodes"], 3)

    def test_b9_corridor_gate_rejects_b8_clone_without_waypoint(self) -> None:
        results = [
            self._b9_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                route_state=None,
                waypoint_lock=None,
            )
            for episode in range(3)
        ]

        gate = b9_waypoint_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b9_aggregate:explicit_b9_decision_episodes",
            gate["failures"],
        )

    def test_b10_corridor_gate_accepts_prospective_replay_progress(self) -> None:
        results = [
            self._b10_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b10_prospective_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["replay_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["committed_plan_episodes"], 3)
        self.assertEqual(gate["aggregate"]["value_signal_episodes"], 3)

    def test_b10_corridor_gate_rejects_b9_clone_without_replay(self) -> None:
        results = [
            self._b10_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                replay_state=None,
                plan_commitment=None,
                prospective_value=None,
            )
            for episode in range(3)
        ]

        gate = b10_prospective_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b10_aggregate:explicit_b10_decision_episodes",
            gate["failures"],
        )

    def test_b11_corridor_gate_accepts_confidence_progress(self) -> None:
        results = [
            self._b11_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b11_confidence_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["confidence_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["confidence_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["neuromod_signal_episodes"], 3)

    def test_b11_corridor_gate_rejects_b10_clone_without_confidence(self) -> None:
        results = [
            self._b11_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                confidence_state=None,
                confidence_lock=None,
                neuromod_signal=None,
            )
            for episode in range(3)
        ]

        gate = b11_confidence_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b11_aggregate:explicit_b11_decision_episodes",
            gate["failures"],
        )

    def test_b12_corridor_gate_accepts_attention_progress(self) -> None:
        results = [
            self._b12_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b12_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["prediction_signal_episodes"], 3)

    def test_b12_corridor_gate_rejects_b11_clone_without_attention(self) -> None:
        results = [
            self._b12_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                attention_state=None,
                search_lock=None,
                attention_gain=None,
                prediction_error=None,
            )
            for episode in range(3)
        ]

        gate = b12_attention_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b12_aggregate:explicit_b12_decision_episodes",
            gate["failures"],
        )

    def test_b13_corridor_gate_accepts_local_search_progress(self) -> None:
        results = [
            self._b13_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b13_local_search_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["search_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["search_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["local_search_signal_episodes"], 3)

    def test_b13_corridor_gate_rejects_b12_clone_without_local_search(self) -> None:
        results = [
            self._b13_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                search_state=None,
                search_lock=None,
                route_score=None,
                affordance_samples=None,
                dead_end_score=None,
            )
            for episode in range(3)
        ]

        gate = b13_local_search_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b13_aggregate:explicit_b13_decision_episodes",
            gate["failures"],
        )

    def test_b14_corridor_gate_accepts_uncertainty_progress(self) -> None:
        results = [
            self._b14_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b14_uncertainty_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["uncertainty_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["confidence_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["commitment_lock_episodes"], 3)

    def test_b14_corridor_gate_rejects_b13_clone_without_uncertainty(self) -> None:
        results = [
            self._b14_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                uncertainty_state=None,
                commitment_lock=None,
                confidence=None,
                uncertainty=None,
                risk_adjusted_score=None,
            )
            for episode in range(3)
        ]

        gate = b14_uncertainty_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b14_aggregate:explicit_b14_decision_episodes",
            gate["failures"],
        )

    def test_b15_corridor_gate_accepts_option_progress(self) -> None:
        results = [
            self._b15_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b15_option_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["option_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["option_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["option_value_signal_episodes"], 3)

    def test_b15_corridor_gate_rejects_b14_clone_without_option(self) -> None:
        results = [
            self._b15_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                option_state=None,
                option_lock=None,
                option_value=None,
                termination_pressure=None,
                persistence_score=None,
            )
            for episode in range(3)
        ]

        gate = b15_option_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b15_aggregate:explicit_b15_decision_episodes",
            gate["failures"],
        )

    def test_b16_corridor_gate_accepts_ensemble_progress(self) -> None:
        results = [
            self._b16_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b16_option_ensemble_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["ensemble_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["ensemble_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["ensemble_signal_episodes"], 3)

    def test_b16_corridor_gate_rejects_b15_clone_without_ensemble(self) -> None:
        results = [
            self._b16_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                ensemble_state=None,
                ensemble_lock=None,
                continue_vote=None,
                return_vote=None,
                consensus_score=None,
                conflict_score=None,
            )
            for episode in range(3)
        ]

        gate = b16_option_ensemble_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b16_aggregate:explicit_b16_decision_episodes",
            gate["failures"],
        )

    def test_b17_corridor_gate_accepts_neuromodulated_progress(self) -> None:
        results = [
            self._b17_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b17_neuromodulated_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["modulator_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["modulation_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["modulation_signal_episodes"], 3)

    def test_b17_corridor_gate_rejects_b16_clone_without_modulator(self) -> None:
        results = [
            self._b17_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                modulator_state=None,
                modulation_lock=None,
                arousal_signal=None,
                homeostatic_gain=None,
                option_gain=None,
                conflict_release=None,
            )
            for episode in range(3)
        ]

        gate = b17_neuromodulated_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b17_aggregate:explicit_b17_decision_episodes",
            gate["failures"],
        )

    def test_b18_corridor_gate_accepts_eligibility_trace_progress(self) -> None:
        results = [
            self._b18_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b18_eligibility_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["trace_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["trace_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["trace_signal_episodes"], 3)

    def test_b18_corridor_gate_rejects_b17_clone_without_trace(self) -> None:
        results = [
            self._b18_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                trace_state=None,
                trace_lock=None,
                eligibility_trace=None,
                prediction_proxy=None,
                stability_bias=None,
                switch_pressure=None,
            )
            for episode in range(3)
        ]

        gate = b18_eligibility_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b18_aggregate:explicit_b18_decision_episodes",
            gate["failures"],
        )

    def test_b19_corridor_gate_accepts_episodic_memory_progress(self) -> None:
        results = [
            self._b19_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
            )
            for episode in range(3)
        ]

        gate = b19_episodic_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["memory_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["memory_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["memory_signal_episodes"], 3)

    def test_b19_corridor_gate_rejects_b18_clone_without_memory(self) -> None:
        results = [
            self._b19_corridor_result(
                episode,
                steps=15,
                alive=False,
                food_distance_delta=13.0,
                decision=None,
                memory_state=None,
                memory_lock=None,
                episode_memory=None,
                consolidation_score=None,
                stability_vote=None,
                switch_suppression=None,
            )
            for episode in range(3)
        ]

        gate = b19_episodic_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b19_aggregate:explicit_b19_decision_episodes",
            gate["failures"],
        )

    def test_b19_corridor_gate_keeps_b18_base_as_diagnostic(self) -> None:
        results = [
            self._b19_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b19_episodic_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertFalse(gate["aggregate"]["base_b18_corridor_diagnostic"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)

    def test_b20_corridor_gate_accepts_working_memory_progress(self) -> None:
        results = [
            self._b20_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b20_working_memory_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["buffer_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["buffer_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["buffer_signal_episodes"], 3)

    def test_b20_corridor_gate_rejects_b19_clone_without_buffer(self) -> None:
        results = [
            self._b20_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                buffer_state=None,
                buffer_lock=None,
                working_buffer=None,
                context_binding=None,
                gate_vote=None,
                release_vote=None,
            )
            for episode in range(3)
        ]

        gate = b20_working_memory_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b20_aggregate:explicit_b20_decision_episodes",
            gate["failures"],
        )

    def test_b20_corridor_gate_keeps_b19_base_as_diagnostic(self) -> None:
        results = [
            self._b20_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b20_working_memory_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b19_corridor_diagnostic", gate["aggregate"])

    def test_b21_corridor_gate_accepts_hippocampal_replay_progress(self) -> None:
        results = [
            self._b21_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b21_hippocampal_replay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["replay_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["replay_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["replay_signal_episodes"], 3)

    def test_b21_corridor_gate_rejects_b20_clone_without_replay(self) -> None:
        results = [
            self._b21_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                replay_state=None,
                replay_lock=None,
                sequence_memory=None,
                replay_score=None,
                route_commitment=None,
                abort_prediction=None,
            )
            for episode in range(3)
        ]

        gate = b21_hippocampal_replay_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b21_aggregate:explicit_b21_decision_episodes",
            gate["failures"],
        )

    def test_b21_corridor_gate_keeps_b20_base_as_diagnostic(self) -> None:
        results = [
            self._b21_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b21_hippocampal_replay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b20_corridor_diagnostic", gate["aggregate"])

    def test_b22_corridor_gate_accepts_prospective_replay_progress(self) -> None:
        results = [
            self._b22_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b22_prospective_replay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sim_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sim_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sim_signal_episodes"], 3)

    def test_b22_corridor_gate_rejects_b21_clone_without_simulation(self) -> None:
        results = [
            self._b22_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                sim_state=None,
                sim_lock=None,
                prospective_sim=None,
                forward_model_score=None,
                viability_projection=None,
                abort_projection=None,
            )
            for episode in range(3)
        ]

        gate = b22_prospective_replay_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b22_aggregate:explicit_b22_decision_episodes",
            gate["failures"],
        )

    def test_b22_corridor_gate_keeps_b21_base_as_diagnostic(self) -> None:
        results = [
            self._b22_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b22_prospective_replay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b21_corridor_diagnostic", gate["aggregate"])

    def test_b23_corridor_gate_accepts_conflict_monitor_progress(self) -> None:
        results = [
            self._b23_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b23_conflict_monitor_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["monitor_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_signal_episodes"], 3)

    def test_b23_corridor_gate_rejects_b22_clone_without_monitor(self) -> None:
        results = [
            self._b23_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                conflict_state=None,
                monitor_lock=None,
                prediction_error=None,
                conflict_memory=None,
                stability_vote=None,
                abort_bias=None,
            )
            for episode in range(3)
        ]

        gate = b23_conflict_monitor_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b23_aggregate:explicit_b23_decision_episodes",
            gate["failures"],
        )

    def test_b23_corridor_gate_keeps_b22_base_as_diagnostic(self) -> None:
        results = [
            self._b23_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b23_conflict_monitor_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b22_corridor_diagnostic", gate["aggregate"])

    def test_b24_corridor_gate_accepts_precision_conflict_progress(self) -> None:
        results = [
            self._b24_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b24_precision_conflict_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_signal_episodes"], 3)

    def test_b24_corridor_gate_rejects_b23_clone_without_precision(self) -> None:
        results = [
            self._b24_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                precision_state=None,
                precision_lock=None,
                precision_memory=None,
                precision_vote=None,
                uncertainty_pressure=None,
                abort_precision=None,
            )
            for episode in range(3)
        ]

        gate = b24_precision_conflict_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b24_aggregate:explicit_b24_decision_episodes",
            gate["failures"],
        )

    def test_b24_corridor_gate_keeps_b23_base_as_diagnostic(self) -> None:
        results = [
            self._b24_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b24_precision_conflict_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b23_corridor_diagnostic", gate["aggregate"])

    def test_b25_corridor_gate_accepts_metacognitive_progress(self) -> None:
        results = [
            self._b25_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b25_metacognitive_confidence_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["metacognitive_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["meta_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["metacognitive_signal_episodes"], 3)

    def test_b25_corridor_gate_rejects_b24_clone_without_metacognition(self) -> None:
        results = [
            self._b25_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                metacognitive_state=None,
                meta_lock=None,
                confidence_memory=None,
                confidence_vote=None,
                doubt_pressure=None,
                control_gain=None,
            )
            for episode in range(3)
        ]

        gate = b25_metacognitive_confidence_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b25_aggregate:explicit_b25_decision_episodes",
            gate["failures"],
        )

    def test_b25_corridor_gate_keeps_b24_base_as_diagnostic(self) -> None:
        results = [
            self._b25_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b25_metacognitive_confidence_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b24_corridor_diagnostic", gate["aggregate"])

    def test_b26_corridor_gate_accepts_allostatic_progress(self) -> None:
        results = [
            self._b26_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b26_allostatic_prediction_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["allostatic_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stability_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["allostatic_signal_episodes"], 3)

    def test_b26_corridor_gate_rejects_b25_clone_without_allostasis(self) -> None:
        results = [
            self._b26_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                allostatic_state=None,
                stability_lock=None,
                prediction_error=None,
                setpoint_pressure=None,
                control_vote=None,
            )
            for episode in range(3)
        ]

        gate = b26_allostatic_prediction_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b26_aggregate:explicit_b26_decision_episodes",
            gate["failures"],
        )

    def test_b26_corridor_gate_keeps_b25_base_as_diagnostic(self) -> None:
        results = [
            self._b26_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b26_allostatic_prediction_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b25_corridor_diagnostic", gate["aggregate"])

    def test_b27_corridor_gate_accepts_arousal_progress(self) -> None:
        results = [
            self._b27_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b27_arousal_gain_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_signal_episodes"], 3)

    def test_b27_corridor_gate_rejects_b26_clone_without_arousal(self) -> None:
        results = [
            self._b27_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                arousal_state=None,
                arousal_lock=None,
                arousal_level=None,
                gain_modulation=None,
                stress_pressure=None,
            )
            for episode in range(3)
        ]

        gate = b27_arousal_gain_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b27_aggregate:explicit_b27_decision_episodes",
            gate["failures"],
        )

    def test_b27_corridor_gate_keeps_b26_base_as_diagnostic(self) -> None:
        results = [
            self._b27_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b27_arousal_gain_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b26_corridor_diagnostic", gate["aggregate"])

    def test_b28_corridor_gate_accepts_attention_progress(self) -> None:
        results = [
            self._b28_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b28_interoceptive_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_signal_episodes"], 3)

    def test_b28_corridor_gate_rejects_b27_clone_without_attention(self) -> None:
        results = [
            self._b28_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                attention_state=None,
                attention_lock=None,
                interoceptive_focus=None,
                attention_gain=None,
                distractor_pressure=None,
            )
            for episode in range(3)
        ]

        gate = b28_interoceptive_attention_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b28_aggregate:explicit_b28_decision_episodes",
            gate["failures"],
        )

    def test_b28_corridor_gate_keeps_b27_base_as_diagnostic(self) -> None:
        results = [
            self._b28_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b28_interoceptive_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b27_corridor_diagnostic", gate["aggregate"])

    def test_b29_corridor_gate_accepts_salience_progress(self) -> None:
        results = [
            self._b29_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b29_salience_competition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["salience_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["salience_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["salience_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["winner_channel_episodes"], 3)

    def test_b29_corridor_gate_rejects_b28_clone_without_salience(self) -> None:
        results = [
            self._b29_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                salience_state=None,
                salience_lock=None,
                threat_salience=None,
                homeostatic_salience=None,
                corridor_salience=None,
                winner_channel=None,
            )
            for episode in range(3)
        ]

        gate = b29_salience_competition_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b29_aggregate:explicit_b29_decision_episodes",
            gate["failures"],
        )

    def test_b29_corridor_gate_keeps_b28_base_as_diagnostic(self) -> None:
        results = [
            self._b29_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b29_salience_competition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b28_corridor_diagnostic", gate["aggregate"])

    def test_b30_corridor_gate_accepts_basal_gate_progress(self) -> None:
        results = [
            self._b30_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b30_basal_ganglia_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["action_gate_episodes"], 3)

    def test_b30_corridor_gate_rejects_b29_clone_without_gate(self) -> None:
        results = [
            self._b30_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                gate_state=None,
                gate_lock=None,
                go_signal=None,
                no_go_signal=None,
                action_gate=None,
            )
            for episode in range(3)
        ]

        gate = b30_basal_ganglia_gate_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b30_aggregate:explicit_b30_decision_episodes",
            gate["failures"],
        )

    def test_b30_corridor_gate_keeps_b29_base_as_diagnostic(self) -> None:
        results = [
            self._b30_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b30_basal_ganglia_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b29_corridor_diagnostic", gate["aggregate"])

    def test_b31_corridor_gate_accepts_dopamine_progress(self) -> None:
        results = [
            self._b31_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b31_dopamine_prediction_error_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["dopamine_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["dopamine_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["dopamine_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_bias_episodes"], 3)

    def test_b31_corridor_gate_rejects_b30_clone_without_dopamine(self) -> None:
        results = [
            self._b31_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                dopamine_state=None,
                dopamine_lock=None,
                reward_prediction_error=None,
                tonic_dopamine=None,
                phasic_dopamine=None,
                gate_bias=None,
            )
            for episode in range(3)
        ]

        gate = b31_dopamine_prediction_error_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b31_aggregate:explicit_b31_decision_episodes",
            gate["failures"],
        )

    def test_b31_corridor_gate_keeps_b30_base_as_diagnostic(self) -> None:
        results = [
            self._b31_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b31_dopamine_prediction_error_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b30_corridor_diagnostic", gate["aggregate"])

    def test_b32_corridor_gate_accepts_actor_critic_progress(self) -> None:
        results = [
            self._b32_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b32_actor_critic_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["critic_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["actor_advantage_episodes"], 3)
        self.assertEqual(gate["aggregate"]["value_lock_episodes"], 3)
        self.assertEqual(gate["aggregate"]["policy_bias_episodes"], 3)

    def test_b32_corridor_gate_rejects_b31_clone_without_value(self) -> None:
        results = [
            self._b32_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                critic_value=None,
                actor_advantage=None,
                value_error=None,
                policy_bias=None,
                value_lock=None,
            )
            for episode in range(3)
        ]

        gate = b32_actor_critic_value_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b32_aggregate:explicit_b32_decision_episodes",
            gate["failures"],
        )

    def test_b32_corridor_gate_keeps_b31_base_as_diagnostic(self) -> None:
        results = [
            self._b32_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b32_actor_critic_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b31_corridor_diagnostic", gate["aggregate"])

    def test_b33_corridor_gate_accepts_td_error_progress(self) -> None:
        results = [
            self._b33_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b33_td_error_decomposition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["td_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["bootstrap_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["reward_trace_episodes"], 3)
        self.assertEqual(gate["aggregate"]["td_lock_episodes"], 3)

    def test_b33_corridor_gate_rejects_b32_clone_without_td_error(self) -> None:
        results = [
            self._b33_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                td_error=None,
                bootstrap_value=None,
                reward_trace=None,
                actor_update=None,
                td_lock=None,
            )
            for episode in range(3)
        ]

        gate = b33_td_error_decomposition_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b33_aggregate:explicit_b33_decision_episodes",
            gate["failures"],
        )

    def test_b33_corridor_gate_keeps_b32_base_as_diagnostic(self) -> None:
        results = [
            self._b33_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b33_td_error_decomposition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b32_corridor_diagnostic", gate["aggregate"])

    def test_b34_corridor_gate_accepts_eligibility_credit_progress(self) -> None:
        results = [
            self._b34_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b34_eligibility_credit_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["eligibility_trace_episodes"], 3)
        self.assertEqual(gate["aggregate"]["credit_assignment_episodes"], 3)
        self.assertEqual(gate["aggregate"]["synaptic_tag_episodes"], 3)
        self.assertEqual(gate["aggregate"]["credit_lock_episodes"], 3)

    def test_b34_corridor_gate_rejects_b33_clone_without_eligibility(self) -> None:
        results = [
            self._b34_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                eligibility_trace=None,
                credit_assignment=None,
                synaptic_tag=None,
                decay_memory=None,
                credit_lock=None,
            )
            for episode in range(3)
        ]

        gate = b34_eligibility_credit_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b34_aggregate:explicit_b34_decision_episodes",
            gate["failures"],
        )

    def test_b34_corridor_gate_keeps_b33_base_as_diagnostic(self) -> None:
        results = [
            self._b34_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b34_eligibility_credit_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b33_corridor_diagnostic", gate["aggregate"])

    def test_b35_corridor_gate_accepts_forward_model_progress(self) -> None:
        results = [
            self._b35_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b35_forward_model_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["forward_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["transition_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["model_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["prediction_memory_episodes"], 3)
        self.assertEqual(gate["aggregate"]["model_lock_episodes"], 3)

    def test_b35_corridor_gate_rejects_b34_clone_without_model(self) -> None:
        results = [
            self._b35_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                forward_value=None,
                transition_error=None,
                model_confidence=None,
                prediction_memory=None,
                model_lock=None,
            )
            for episode in range(3)
        ]

        gate = b35_forward_model_value_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b35_aggregate:explicit_b35_decision_episodes",
            gate["failures"],
        )

    def test_b35_corridor_gate_keeps_b34_base_as_diagnostic(self) -> None:
        results = [
            self._b35_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b35_forward_model_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b34_corridor_diagnostic", gate["aggregate"])

    def test_b36_corridor_gate_accepts_latent_belief_progress(self) -> None:
        results = [
            self._b36_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b36_latent_belief_state_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["latent_state_episodes"], 3)
        self.assertEqual(gate["aggregate"]["belief_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["state_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["context_memory_episodes"], 3)
        self.assertEqual(gate["aggregate"]["belief_lock_episodes"], 3)

    def test_b36_corridor_gate_rejects_b35_clone_without_belief(self) -> None:
        results = [
            self._b36_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                latent_state=None,
                belief_error=None,
                state_confidence=None,
                context_memory=None,
                belief_lock=None,
            )
            for episode in range(3)
        ]

        gate = b36_latent_belief_state_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b36_aggregate:explicit_b36_decision_episodes",
            gate["failures"],
        )

    def test_b36_corridor_gate_keeps_b35_base_as_diagnostic(self) -> None:
        results = [
            self._b36_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b36_latent_belief_state_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b35_corridor_diagnostic", gate["aggregate"])

    def test_b37_corridor_gate_accepts_state_factor_progress(self) -> None:
        results = [
            self._b37_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b37_state_factor_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["external_factor_episodes"], 3)
        self.assertEqual(gate["aggregate"]["internal_factor_episodes"], 3)
        self.assertEqual(gate["aggregate"]["factor_alignment_episodes"], 3)
        self.assertEqual(gate["aggregate"]["factor_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["factor_lock_episodes"], 3)

    def test_b37_corridor_gate_rejects_b36_clone_without_factorization(self) -> None:
        results = [
            self._b37_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                external_factor=None,
                internal_factor=None,
                factor_alignment=None,
                factor_confidence=None,
                factor_lock=None,
            )
            for episode in range(3)
        ]

        gate = b37_state_factor_gate_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b37_aggregate:explicit_b37_decision_episodes",
            gate["failures"],
        )

    def test_b37_corridor_gate_keeps_b36_base_as_diagnostic(self) -> None:
        results = [
            self._b37_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b37_state_factor_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b36_corridor_diagnostic", gate["aggregate"])

    def test_b38_corridor_gate_accepts_factor_attention_progress(self) -> None:
        results = [
            self._b38_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b38_factor_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["external_attention_episodes"], 3)
        self.assertEqual(gate["aggregate"]["internal_attention_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_lock_episodes"], 3)

    def test_b38_corridor_gate_rejects_b37_clone_without_attention(self) -> None:
        results = [
            self._b38_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                external_attention=None,
                internal_attention=None,
                attention_balance=None,
                attention_gain=None,
                attention_lock=None,
            )
            for episode in range(3)
        ]

        gate = b38_factor_attention_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b38_aggregate:explicit_b38_decision_episodes",
            gate["failures"],
        )

    def test_b38_corridor_gate_keeps_b37_base_as_diagnostic(self) -> None:
        results = [
            self._b38_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b38_factor_attention_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b37_corridor_diagnostic", gate["aggregate"])

    def test_b39_corridor_gate_accepts_attention_binding_progress(self) -> None:
        results = [
            self._b39_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b39_attention_binding_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["binding_strength_episodes"], 3)
        self.assertEqual(gate["aggregate"]["cross_factor_coherence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["bound_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["binding_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["binding_lock_episodes"], 3)

    def test_b39_corridor_gate_rejects_b38_clone_without_binding(self) -> None:
        results = [
            self._b39_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                binding_strength=None,
                cross_factor_coherence=None,
                bound_context=None,
                binding_gain=None,
                binding_lock=None,
            )
            for episode in range(3)
        ]

        gate = b39_attention_binding_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b39_aggregate:explicit_b39_decision_episodes",
            gate["failures"],
        )

    def test_b39_corridor_gate_keeps_b38_base_as_diagnostic(self) -> None:
        results = [
            self._b39_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b39_attention_binding_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b38_corridor_diagnostic", gate["aggregate"])

    def test_b40_corridor_gate_accepts_global_workspace_progress(self) -> None:
        results = [
            self._b40_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b40_global_workspace_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["workspace_activation_episodes"], 3)
        self.assertEqual(gate["aggregate"]["broadcast_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["context_availability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["workspace_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["workspace_lock_episodes"], 3)

    def test_b40_corridor_gate_rejects_b39_clone_without_workspace(self) -> None:
        results = [
            self._b40_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                workspace_activation=None,
                broadcast_gain=None,
                context_availability=None,
                workspace_stability=None,
                workspace_lock=None,
            )
            for episode in range(3)
        ]

        gate = b40_global_workspace_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b40_aggregate:explicit_b40_decision_episodes",
            gate["failures"],
        )

    def test_b40_corridor_gate_keeps_b39_base_as_diagnostic(self) -> None:
        results = [
            self._b40_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b40_global_workspace_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b39_corridor_diagnostic", gate["aggregate"])

    def test_b41_corridor_gate_accepts_executive_workspace_progress(self) -> None:
        results = [
            self._b41_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b41_executive_workspace_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_selection_episodes"], 3)
        self.assertEqual(gate["aggregate"]["inhibitory_pressure_episodes"], 3)
        self.assertEqual(gate["aggregate"]["goal_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_lock_episodes"], 3)

    def test_b41_corridor_gate_rejects_b40_clone_without_executive_control(self) -> None:
        results = [
            self._b41_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                executive_selection=None,
                inhibitory_pressure=None,
                goal_context=None,
                executive_stability=None,
                executive_lock=None,
            )
            for episode in range(3)
        ]

        gate = b41_executive_workspace_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b41_aggregate:explicit_b41_decision_episodes",
            gate["failures"],
        )

    def test_b41_corridor_gate_keeps_b40_base_as_diagnostic(self) -> None:
        results = [
            self._b41_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b41_executive_workspace_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b40_corridor_diagnostic", gate["aggregate"])

    def test_b42_corridor_gate_accepts_error_monitor_progress(self) -> None:
        results = [
            self._b42_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b42_error_monitor_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["error_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["performance_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["monitor_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["monitor_lock_episodes"], 3)

    def test_b42_corridor_gate_rejects_b41_clone_without_error_monitor(self) -> None:
        results = [
            self._b42_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                error_signal=None,
                conflict_signal=None,
                performance_context=None,
                monitor_stability=None,
                monitor_lock=None,
            )
            for episode in range(3)
        ]

        gate = b42_error_monitor_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b42_aggregate:explicit_b42_decision_episodes",
            gate["failures"],
        )

    def test_b42_corridor_gate_keeps_b41_base_as_diagnostic(self) -> None:
        results = [
            self._b42_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b42_error_monitor_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b41_corridor_diagnostic", gate["aggregate"])

    def test_b43_corridor_gate_accepts_adaptive_precision_progress(self) -> None:
        results = [
            self._b43_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b43_adaptive_precision_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["adaptive_threshold_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["control_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_lock_episodes"], 3)

    def test_b43_corridor_gate_rejects_b42_clone_without_precision(self) -> None:
        results = [
            self._b43_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                precision_signal=None,
                adaptive_threshold=None,
                arousal_context=None,
                control_stability=None,
                precision_lock=None,
            )
            for episode in range(3)
        ]

        gate = b43_adaptive_precision_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b43_aggregate:explicit_b43_decision_episodes",
            gate["failures"],
        )

    def test_b43_corridor_gate_keeps_b42_base_as_diagnostic(self) -> None:
        results = [
            self._b43_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b43_adaptive_precision_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b42_corridor_diagnostic", gate["aggregate"])

    def test_b44_corridor_gate_accepts_thalamic_relay_progress(self) -> None:
        results = [
            self._b44_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b44_thalamic_relay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["relay_gate_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sensory_precision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["context_relay_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gate_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["relay_lock_episodes"], 3)

    def test_b44_corridor_gate_rejects_b43_clone_without_relay(self) -> None:
        results = [
            self._b44_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                relay_gate=None,
                sensory_precision=None,
                context_relay=None,
                gate_stability=None,
                relay_lock=None,
            )
            for episode in range(3)
        ]

        gate = b44_thalamic_relay_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b44_aggregate:explicit_b44_decision_episodes",
            gate["failures"],
        )

    def test_b44_corridor_gate_keeps_b43_base_as_diagnostic(self) -> None:
        results = [
            self._b44_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b44_thalamic_relay_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b43_corridor_diagnostic", gate["aggregate"])

    def test_b45_corridor_gate_accepts_reticular_inhibition_progress(self) -> None:
        results = [
            self._b45_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b45_reticular_inhibition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["inhibitory_gate_episodes"], 3)
        self.assertEqual(gate["aggregate"]["sensory_filter_episodes"], 3)
        self.assertEqual(gate["aggregate"]["context_suppression_episodes"], 3)
        self.assertEqual(gate["aggregate"]["loop_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["inhibition_lock_episodes"], 3)

    def test_b45_corridor_gate_rejects_b44_clone_without_inhibition(self) -> None:
        results = [
            self._b45_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                inhibitory_gate=None,
                sensory_filter=None,
                context_suppression=None,
                loop_stability=None,
                inhibition_lock=None,
            )
            for episode in range(3)
        ]

        gate = b45_reticular_inhibition_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b45_aggregate:explicit_b45_decision_episodes",
            gate["failures"],
        )

    def test_b45_corridor_gate_keeps_b44_base_as_diagnostic(self) -> None:
        results = [
            self._b45_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b45_reticular_inhibition_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b44_corridor_diagnostic", gate["aggregate"])

    def test_b46_corridor_gate_accepts_feedback_progress(self) -> None:
        results = [
            self._b46_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b46_corticothalamic_feedback_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["feedback_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["topdown_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["prediction_match_episodes"], 3)
        self.assertEqual(gate["aggregate"]["feedback_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["feedback_lock_episodes"], 3)

    def test_b46_corridor_gate_rejects_b45_clone_without_feedback(self) -> None:
        results = [
            self._b46_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                feedback_gain=None,
                topdown_context=None,
                prediction_match=None,
                feedback_stability=None,
                feedback_lock=None,
            )
            for episode in range(3)
        ]

        gate = b46_corticothalamic_feedback_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b46_aggregate:explicit_b46_decision_episodes",
            gate["failures"],
        )

    def test_b46_corridor_gate_keeps_b45_base_as_diagnostic(self) -> None:
        results = [
            self._b46_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b46_corticothalamic_feedback_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b45_corridor_diagnostic", gate["aggregate"])

    def test_b47_corridor_gate_accepts_oscillatory_synchrony_progress(self) -> None:
        results = [
            self._b47_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b47_oscillatory_synchrony_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["phase_alignment_episodes"], 3)
        self.assertEqual(gate["aggregate"]["synchrony_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["cross_loop_coherence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["phase_lock_episodes"], 3)

    def test_b47_corridor_gate_rejects_b46_clone_without_synchrony(self) -> None:
        results = [
            self._b47_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                phase_alignment=None,
                synchrony_gain=None,
                cross_loop_coherence=None,
                phase_lock=None,
            )
            for episode in range(3)
        ]

        gate = b47_oscillatory_synchrony_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b47_aggregate:explicit_b47_decision_episodes",
            gate["failures"],
        )

    def test_b47_corridor_gate_keeps_b46_base_as_diagnostic(self) -> None:
        results = [
            self._b47_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b47_oscillatory_synchrony_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b46_corridor_diagnostic", gate["aggregate"])

    def test_b48_corridor_gate_accepts_cerebellar_timing_progress(self) -> None:
        results = [
            self._b48_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b48_cerebellar_timing_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["timing_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["predictive_timing_episodes"], 3)
        self.assertEqual(gate["aggregate"]["corrective_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["calibration_lock_episodes"], 3)

    def test_b48_corridor_gate_rejects_b47_clone_without_timing(self) -> None:
        results = [
            self._b48_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                timing_error=None,
                predictive_timing=None,
                corrective_gain=None,
                calibration_lock=None,
            )
            for episode in range(3)
        ]

        gate = b48_cerebellar_timing_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b48_aggregate:explicit_b48_decision_episodes",
            gate["failures"],
        )

    def test_b48_corridor_gate_keeps_b47_base_as_diagnostic(self) -> None:
        results = [
            self._b48_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b48_cerebellar_timing_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b47_corridor_diagnostic", gate["aggregate"])

    def test_b49_corridor_gate_accepts_striatal_gate_progress(self) -> None:
        results = [
            self._b49_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b49_striatal_action_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["go_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["no_go_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["action_gate_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["selection_lock_episodes"], 3)

    def test_b49_corridor_gate_rejects_b48_clone_without_striatal_gate(self) -> None:
        results = [
            self._b49_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                go_signal=None,
                no_go_signal=None,
                action_gate_balance=None,
                selection_lock=None,
            )
            for episode in range(3)
        ]

        gate = b49_striatal_action_gate_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b49_aggregate:explicit_b49_decision_episodes",
            gate["failures"],
        )

    def test_b49_corridor_gate_keeps_b48_base_as_diagnostic(self) -> None:
        results = [
            self._b49_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b49_striatal_action_gate_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b48_corridor_diagnostic", gate["aggregate"])

    def test_b50_corridor_gate_accepts_habit_chunking_progress(self) -> None:
        results = [
            self._b50_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b50_habit_chunking_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["habit_strength_episodes"], 3)
        self.assertEqual(gate["aggregate"]["chunk_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["habit_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["chunk_lock_episodes"], 3)

    def test_b50_corridor_gate_rejects_b49_clone_without_habit_chunking(self) -> None:
        results = [
            self._b50_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                habit_strength=None,
                chunk_value=None,
                habit_stability=None,
                chunk_lock=None,
            )
            for episode in range(3)
        ]

        gate = b50_habit_chunking_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b50_aggregate:explicit_b50_decision_episodes",
            gate["failures"],
        )

    def test_b50_corridor_gate_keeps_b49_base_as_diagnostic(self) -> None:
        results = [
            self._b50_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b50_habit_chunking_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b49_corridor_diagnostic", gate["aggregate"])

    def test_b51_corridor_gate_accepts_dopaminergic_habit_progress(self) -> None:
        results = [
            self._b51_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b51_dopaminergic_habit_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["prediction_error_episodes"], 3)
        self.assertEqual(gate["aggregate"]["dopamine_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["habit_modulation_episodes"], 3)
        self.assertEqual(gate["aggregate"]["modulation_lock_episodes"], 3)

    def test_b51_corridor_gate_rejects_b50_clone_without_dopamine(self) -> None:
        results = [
            self._b51_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                prediction_error=None,
                dopamine_gain=None,
                habit_modulation=None,
                modulation_lock=None,
            )
            for episode in range(3)
        ]

        gate = b51_dopaminergic_habit_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b51_aggregate:explicit_b51_decision_episodes",
            gate["failures"],
        )

    def test_b51_corridor_gate_keeps_b50_base_as_diagnostic(self) -> None:
        results = [
            self._b51_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b51_dopaminergic_habit_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b50_corridor_diagnostic", gate["aggregate"])

    def test_b52_corridor_gate_accepts_cholinergic_precision_progress(self) -> None:
        results = [
            self._b52_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b52_cholinergic_precision_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["acetylcholine_level_episodes"], 3)
        self.assertEqual(gate["aggregate"]["precision_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["uncertainty_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["attention_lock_episodes"], 3)

    def test_b52_corridor_gate_rejects_b51_clone_without_precision(self) -> None:
        results = [
            self._b52_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                acetylcholine_level=None,
                precision_gain=None,
                uncertainty_signal=None,
                attention_lock=None,
            )
            for episode in range(3)
        ]

        gate = b52_cholinergic_precision_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b52_aggregate:explicit_b52_decision_episodes",
            gate["failures"],
        )

    def test_b52_corridor_gate_keeps_b51_base_as_diagnostic(self) -> None:
        results = [
            self._b52_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b52_cholinergic_precision_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b51_corridor_diagnostic", gate["aggregate"])

    def test_b53_corridor_gate_accepts_noradrenergic_arousal_progress(self) -> None:
        results = [
            self._b53_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b53_noradrenergic_arousal_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["norepinephrine_level_episodes"], 3)
        self.assertEqual(gate["aggregate"]["arousal_gain_episodes"], 3)
        self.assertEqual(gate["aggregate"]["surprise_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["gain_lock_episodes"], 3)

    def test_b53_corridor_gate_rejects_b52_clone_without_arousal(self) -> None:
        results = [
            self._b53_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                norepinephrine_level=None,
                arousal_gain=None,
                surprise_signal=None,
                gain_lock=None,
            )
            for episode in range(3)
        ]

        gate = b53_noradrenergic_arousal_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b53_aggregate:explicit_b53_decision_episodes",
            gate["failures"],
        )

    def test_b53_corridor_gate_keeps_b52_base_as_diagnostic(self) -> None:
        results = [
            self._b53_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b53_noradrenergic_arousal_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b52_corridor_diagnostic", gate["aggregate"])

    def test_b54_corridor_gate_accepts_serotonergic_patience_progress(self) -> None:
        results = [
            self._b54_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b54_serotonergic_patience_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["serotonin_level_episodes"], 3)
        self.assertEqual(gate["aggregate"]["patience_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["impulse_suppression_episodes"], 3)
        self.assertEqual(gate["aggregate"]["patience_lock_episodes"], 3)

    def test_b54_corridor_gate_rejects_b53_clone_without_patience(self) -> None:
        results = [
            self._b54_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                serotonin_level=None,
                patience_signal=None,
                impulse_suppression=None,
                patience_lock=None,
            )
            for episode in range(3)
        ]

        gate = b54_serotonergic_patience_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b54_aggregate:explicit_b54_decision_episodes",
            gate["failures"],
        )

    def test_b54_corridor_gate_keeps_b53_base_as_diagnostic(self) -> None:
        results = [
            self._b54_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b54_serotonergic_patience_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b53_corridor_diagnostic", gate["aggregate"])

    def test_b55_corridor_gate_accepts_hypothalamic_drive_progress(self) -> None:
        results = [
            self._b55_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b55_hypothalamic_drive_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["hypothalamic_drive_episodes"], 3)
        self.assertEqual(gate["aggregate"]["satiety_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["recovery_bias_episodes"], 3)
        self.assertEqual(gate["aggregate"]["drive_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["drive_lock_episodes"], 3)

    def test_b55_corridor_gate_rejects_b54_clone_without_drive(self) -> None:
        results = [
            self._b55_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                hypothalamic_drive=None,
                satiety_signal=None,
                recovery_bias=None,
                drive_balance=None,
                drive_lock=None,
            )
            for episode in range(3)
        ]

        gate = b55_hypothalamic_drive_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b55_aggregate:explicit_b55_decision_episodes",
            gate["failures"],
        )

    def test_b55_corridor_gate_keeps_b54_base_as_diagnostic(self) -> None:
        results = [
            self._b55_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b55_hypothalamic_drive_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b54_corridor_diagnostic", gate["aggregate"])

    def test_b56_corridor_gate_accepts_hpa_stress_progress(self) -> None:
        results = [
            self._b56_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b56_hpa_stress_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["cortisol_level_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stress_load_episodes"], 3)
        self.assertEqual(gate["aggregate"]["recovery_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["endocrine_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["stress_lock_episodes"], 3)

    def test_b56_corridor_gate_rejects_b55_clone_without_hpa_stress(self) -> None:
        results = [
            self._b56_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                cortisol_level=None,
                stress_load=None,
                recovery_signal=None,
                endocrine_balance=None,
                stress_lock=None,
            )
            for episode in range(3)
        ]

        gate = b56_hpa_stress_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b56_aggregate:explicit_b56_decision_episodes",
            gate["failures"],
        )

    def test_b56_corridor_gate_keeps_b55_base_as_diagnostic(self) -> None:
        results = [
            self._b56_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b56_hpa_stress_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b55_corridor_diagnostic", gate["aggregate"])

    def test_b57_corridor_gate_accepts_interoceptive_progress(self) -> None:
        results = [
            self._b57_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b57_insular_interoceptive_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["interoceptive_awareness_episodes"], 3)
        self.assertEqual(gate["aggregate"]["visceral_salience_episodes"], 3)
        self.assertEqual(gate["aggregate"]["body_state_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["awareness_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["awareness_lock_episodes"], 3)

    def test_b57_corridor_gate_rejects_b56_clone_without_interoception(self) -> None:
        results = [
            self._b57_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                interoceptive_awareness=None,
                visceral_salience=None,
                body_state_confidence=None,
                awareness_balance=None,
                awareness_lock=None,
            )
            for episode in range(3)
        ]

        gate = b57_insular_interoceptive_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b57_aggregate:explicit_b57_decision_episodes",
            gate["failures"],
        )

    def test_b57_corridor_gate_keeps_b56_base_as_diagnostic(self) -> None:
        results = [
            self._b57_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b57_insular_interoceptive_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b56_corridor_diagnostic", gate["aggregate"])

    def test_b58_corridor_gate_accepts_acc_conflict_progress(self) -> None:
        results = [
            self._b58_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b58_acc_conflict_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["error_likelihood_episodes"], 3)
        self.assertEqual(gate["aggregate"]["control_allocation_episodes"], 3)
        self.assertEqual(gate["aggregate"]["resolution_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["conflict_lock_episodes"], 3)

    def test_b58_corridor_gate_rejects_b57_clone_without_conflict(self) -> None:
        results = [
            self._b58_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                conflict_signal=None,
                error_likelihood=None,
                control_allocation=None,
                resolution_balance=None,
                conflict_lock=None,
            )
            for episode in range(3)
        ]

        gate = b58_acc_conflict_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b58_aggregate:explicit_b58_decision_episodes",
            gate["failures"],
        )

    def test_b58_corridor_gate_keeps_b57_base_as_diagnostic(self) -> None:
        results = [
            self._b58_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b58_acc_conflict_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b57_corridor_diagnostic", gate["aggregate"])

    def test_b59_corridor_gate_accepts_prefrontal_goal_progress(self) -> None:
        results = [
            self._b59_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b59_prefrontal_goal_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["goal_context_episodes"], 3)
        self.assertEqual(gate["aggregate"]["working_set_stability_episodes"], 3)
        self.assertEqual(gate["aggregate"]["task_set_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["executive_lock_episodes"], 3)

    def test_b59_corridor_gate_rejects_b58_clone_without_prefrontal_context(self) -> None:
        results = [
            self._b59_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                goal_context=None,
                working_set_stability=None,
                task_set_confidence=None,
                executive_balance=None,
                executive_lock=None,
            )
            for episode in range(3)
        ]

        gate = b59_prefrontal_goal_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b59_aggregate:explicit_b59_decision_episodes",
            gate["failures"],
        )

    def test_b59_corridor_gate_keeps_b58_base_as_diagnostic(self) -> None:
        results = [
            self._b59_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b59_prefrontal_goal_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b58_corridor_diagnostic", gate["aggregate"])

    def test_b60_corridor_gate_accepts_orbitofrontal_value_progress(self) -> None:
        results = [
            self._b60_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b60_orbitofrontal_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["outcome_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["reversal_signal_episodes"], 3)
        self.assertEqual(gate["aggregate"]["goal_value_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["value_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["value_lock_episodes"], 3)

    def test_b60_corridor_gate_rejects_b59_clone_without_value(self) -> None:
        results = [
            self._b60_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                outcome_value=None,
                reversal_signal=None,
                goal_value_confidence=None,
                value_balance=None,
                value_lock=None,
            )
            for episode in range(3)
        ]

        gate = b60_orbitofrontal_value_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b60_aggregate:explicit_b60_decision_episodes",
            gate["failures"],
        )

    def test_b60_corridor_gate_keeps_b59_base_as_diagnostic(self) -> None:
        results = [
            self._b60_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b60_orbitofrontal_value_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b59_corridor_diagnostic", gate["aggregate"])

    def test_b61_corridor_gate_accepts_amygdala_safety_progress(self) -> None:
        results = [
            self._b61_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b61_amygdala_safety_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["safety_value_episodes"], 3)
        self.assertEqual(gate["aggregate"]["threat_channel_episodes"], 3)
        self.assertEqual(gate["aggregate"]["safety_confidence_episodes"], 3)
        self.assertEqual(gate["aggregate"]["affective_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["safety_lock_episodes"], 3)

    def test_b61_corridor_gate_rejects_b60_clone_without_safety(self) -> None:
        results = [
            self._b61_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                safety_value=None,
                threat_value=None,
                safety_confidence=None,
                affective_balance=None,
                safety_lock=None,
            )
            for episode in range(3)
        ]

        gate = b61_amygdala_safety_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b61_aggregate:explicit_b61_decision_episodes",
            gate["failures"],
        )

    def test_b61_corridor_gate_keeps_b60_base_as_diagnostic(self) -> None:
        results = [
            self._b61_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b61_amygdala_safety_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b60_corridor_diagnostic", gate["aggregate"])

    def test_b62_corridor_gate_accepts_defensive_mode_progress(self) -> None:
        results = [
            self._b62_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b62_defensive_mode_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["explicit_decision_episodes"], 3)
        self.assertEqual(gate["aggregate"]["defensive_mode_episodes"], 3)
        self.assertEqual(gate["aggregate"]["defense_pressure_episodes"], 3)
        self.assertEqual(gate["aggregate"]["shelter_bias_episodes"], 3)
        self.assertEqual(gate["aggregate"]["defense_balance_episodes"], 3)
        self.assertEqual(gate["aggregate"]["lock_or_safe_episodes"], 3)

    def test_b62_corridor_gate_rejects_b61_clone_without_defense(self) -> None:
        results = [
            self._b62_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
                decision=None,
                defensive_mode=None,
                freeze_pressure=None,
                flee_pressure=None,
                shelter_bias=None,
                defense_balance=None,
                defense_lock=None,
            )
            for episode in range(3)
        ]

        gate = b62_defensive_mode_corridor_gate_result(results)

        self.assertFalse(gate["passed"])
        self.assertIn(
            "corridor_b62_aggregate:explicit_b62_decision_episodes",
            gate["failures"],
        )

    def test_b62_corridor_gate_keeps_b61_base_as_diagnostic(self) -> None:
        results = [
            self._b62_corridor_result(
                episode,
                steps=14,
                alive=False,
                food_distance_delta=12.0,
            )
            for episode in range(3)
        ]

        gate = b62_defensive_mode_corridor_gate_result(results)

        self.assertTrue(gate["passed"], msg=gate["failures"])
        self.assertEqual(gate["aggregate"]["corridor_safety_episodes"], 3)
        self.assertIn("base_b61_corridor_diagnostic", gate["aggregate"])

    def test_b6_promotion_prefers_passing_fusion_then_best_individual(self) -> None:
        risk = {
            "variant": B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
            "controller_family": "risk_corridor",
            "status": "accepted",
            "canonical_gate": {"aggregate": {"completed_horizons": 4, "min_steps": 50, "total_predator_contacts": 20}},
            "food_predator_gate": {"aggregate": {"threat_exposure_episodes": 2, "threat_priority_or_suppression_episodes": 2}},
            "corridor_gate": {"aggregate": {"progress_episodes": 2, "survival_progress_episodes": 2}},
            "easy_gate": {"passed": True},
        }
        recurrent = {
            **risk,
            "variant": B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
            "controller_family": "recurrent_memory",
            "canonical_gate": {"aggregate": {"completed_horizons": 5, "min_steps": 60, "total_predator_contacts": 18}},
        }
        fusion = {
            **risk,
            "variant": B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
            "controller_family": "fused_risk_recurrent",
            "status": "discarded",
        }

        selected = select_b6_promotion([risk], [recurrent], fusion)
        self.assertEqual(selected["variant"], B6_RECURRENT_CONTEXT_H48_POLICY_NAME)

        fusion["status"] = "accepted"
        selected = select_b6_promotion([risk], [recurrent], fusion)
        self.assertEqual(selected["variant"], B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME)


class BSeriesLegacyHarnessTest(unittest.TestCase):
    def test_legacy_harness_runs_short_training_smoke(self) -> None:
        sim = LegacyB0Simulation(max_steps=20, seed=5)
        summary, trace = sim.train(
            2,
            evaluation_episodes=1,
            capture_evaluation_trace=True,
        )

        self.assertEqual(summary["b_level"], 0)
        self.assertEqual(summary["b_mode"], "legacy_semantic")
        self.assertEqual(summary["semantic_actions"], list(B_SEMANTIC_ACTIONS))
        self.assertEqual(summary["evaluation"]["episodes"], 1)
        self.assertGreater(len(trace), 0)


if __name__ == "__main__":
    unittest.main()
