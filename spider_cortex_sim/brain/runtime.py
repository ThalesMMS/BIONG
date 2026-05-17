from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Sequence

import numpy as np

from ..ablations import BrainAblationConfig, default_brain_config
from ..direct_policy_affordances import AFFORDANCE_SHELTER_ROLE_NAMES
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
from ..b_series import (
    B_CURRENT_BRIDGE_EFFECTIVE_LEVEL,
    B_CURRENT_BRIDGE_SELECTION_SOURCE,
    B1_THREAT_GUARD_EFFECTIVE_LEVEL,
    B1_THREAT_GUARD_POLICY_NAME,
    B1_THREAT_GUARD_SELECTION_SOURCE,
    B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL,
    B2_TEMPORAL_THREAT_H48_POLICY_NAME,
    B2_TEMPORAL_THREAT_H56_POLICY_NAME,
    B2_TEMPORAL_THREAT_H64_POLICY_NAME,
    B2_TEMPORAL_THREAT_SELECTION_SOURCE,
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
    B5_HOMEOSTATIC_ARBITER_EFFECTIVE_LEVEL,
    B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
    B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME,
    B5_HOMEOSTATIC_ARBITER_SELECTION_SOURCE,
    B6_CORRIDOR_SURVIVAL_GUARD_H48_POLICY_NAME,
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
    B_SERIES_POLICY_NAME,
    B_SEMANTIC_ACTIONS,
    B_SEMANTIC_ACTION_TO_INDEX,
    bridge_b_semantic_action,
)
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
from ..nn import (
    ArbitrationNetwork,
    MotorNetwork,
    ProposalNetwork,
    RecurrentEventAttentionTrueMonolithicNetwork,
    RecurrentOptionAffordanceTrueMonolithicNetwork,
    RecurrentOptionTrueMonolithicNetwork,
    RecurrentProposalNetwork,
    one_hot,
    softmax,
)
from ..noise import _compute_execution_difficulty_core
from ..operational_profiles import OperationalProfile, runtime_operational_profile
from ..phase import PHASE_LABELS
from ..reflexes import (
    _apply_reflex_path as apply_reflex_path,
    _direction_action as direction_action,
    _module_reflex_decision as module_reflex_decision,
)
from ..world import ACTIONS

from .types import BrainStep

TRUE_MONOLITHIC_NO_FOOD_DIRECTION_VARIANTS = {
    "true_monolithic_executive_option_guarded_policy",
}

TRUE_MONOLITHIC_DIRECTION_BIAS_LOGIT = 3.0
MODULAR_DIRECTION_BIAS_LOGIT = 3.0
TRUE_MONOLITHIC_THREAT_ESCAPE_BIAS_LOGIT = 6.0
TRUE_MONOLITHIC_THREAT_ESCAPE_SMELL_THRESHOLD = 0.43
TRUE_MONOLITHIC_SLEEP_REST_BIAS_LOGIT = 6.0


class BrainRuntimeMixin:
    @staticmethod
    def _b_series_float(mapping: MappingProxyType | Dict[str, float], key: str) -> float:
        try:
            value = float(mapping.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    def _b0_current_simple_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int]:
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        hunger = self._b_series_float(hunger_obs, "hunger")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        health = self._b_series_float(sleep_obs, "health")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        on_shelter = self._b_series_float(sleep_obs, "on_shelter") > 0.5 or bool(
            meta.get("on_shelter", False)
        )
        night = self._b_series_float(sleep_obs, "night") > 0.5 or bool(
            meta.get("night", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        shelter_role_level = max(
            self._b_series_float(sleep_obs, "shelter_role_level"),
            self._b_series_float(meta, "shelter_role_level"),
        )
        food_memory_signal = (
            1.0 - self._b_series_float(hunger_obs, "food_memory_age")
            if (
                abs(float(hunger_obs.get("food_memory_dx", 0.0)))
                + abs(float(hunger_obs.get("food_memory_dy", 0.0)))
            )
            > 0.05
            else 0.0
        )
        food_signal = max(
            self._b_series_float(hunger_obs, "food_visible"),
            self._b_series_float(hunger_obs, "food_certainty"),
            self._b_series_float(hunger_obs, "food_smell_strength"),
            self._b_series_float(hunger_obs, "food_trace_strength"),
            food_memory_signal,
        )
        acute_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
        )
        threat_pressure = max(
            acute_threat,
            self._b_series_float(threat_obs, "predator_smell_strength"),
        )

        if on_food and hunger >= 0.10:
            semantic_action = "EAT"
            reason = "b0_current_eat_on_food"
        elif on_shelter:
            rest_pressure = bool(night or fatigue >= 0.25 or sleep_debt >= 0.25)
            if health <= 0.65 and hunger < 0.55:
                if shelter_role_level < 0.75:
                    semantic_action = "MOVE_TO_SHELTER"
                    reason = "b0_current_low_health_deepen"
                elif rest_pressure:
                    semantic_action = "SLEEP"
                    reason = "b0_current_low_health_rest"
                else:
                    semantic_action = "STAY"
                    reason = "b0_current_low_health_hold"
            elif threat_pressure >= 0.55 and hunger < 0.48:
                if shelter_role_level < 0.75:
                    semantic_action = "MOVE_TO_SHELTER"
                    reason = "b0_current_threat_hold_deepen"
                elif rest_pressure:
                    semantic_action = "SLEEP"
                    reason = "b0_current_threat_hold_rest"
                else:
                    semantic_action = "STAY"
                    reason = "b0_current_threat_hold_shelter"
            elif rest_pressure and hunger < 0.55:
                if shelter_role_level < 0.75:
                    semantic_action = "MOVE_TO_SHELTER"
                    reason = "b0_current_deepen_before_rest"
                else:
                    semantic_action = "SLEEP"
                    reason = "b0_current_rest_in_shelter"
            elif hunger >= 0.50 or (food_signal >= 0.35 and not rest_pressure):
                semantic_action = "MOVE_TO_FOOD"
                reason = "b0_current_forage_from_shelter"
            else:
                semantic_action = "STAY"
                reason = "b0_current_shelter_hold"
        elif (
            (hunger < 0.40 and (night or fatigue >= 0.25 or sleep_debt >= 0.25))
            or acute_threat >= 0.85
            or (threat_pressure >= 0.55 and hunger < 0.55)
            or (health <= 0.65 and hunger < 0.55)
            or health <= 0.35
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b0_current_recover_return"
        elif hunger >= 0.45 or food_signal >= 0.15:
            semantic_action = "MOVE_TO_FOOD"
            reason = "b0_current_forage"
        elif night or fatigue >= 0.52 or sleep_debt >= 0.52:
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b0_current_rest_return"
        else:
            semantic_action = "EXPLORE"
            reason = "b0_current_explore"

        return (
            semantic_action,
            B_CURRENT_BRIDGE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
        )

    def _b1_threat_guard_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
        ) = self._b0_current_simple_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        hunger = self._b_series_float(hunger_obs, "hunger")
        on_shelter = self._b_series_float(sleep_obs, "on_shelter") > 0.5 or bool(
            meta.get("on_shelter", False)
        )
        acute_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
        )
        threat_pressure = max(
            acute_threat,
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(meta, "predator_smell_strength"),
        )
        if not on_shelter and threat_pressure >= 0.60 and hunger < 0.80:
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b1_threat_guard_return_under_threat"
        else:
            reason = f"b1_threat_guard_{reason}"
        return (
            semantic_action,
            B1_THREAT_GUARD_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
        )

    def _b2_temporal_threat_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
        ) = self._b1_threat_guard_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        def _mapping(value: object) -> dict[str, object]:
            return value if isinstance(value, dict) else {}

        def _signed_float(value: object) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return 0.0
            return numeric if np.isfinite(numeric) else 0.0

        hunger = self._b_series_float(hunger_obs, "hunger")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        health = self._b_series_float(sleep_obs, "health")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or shelter_role != "outside"
        )
        night = self._b_series_float(sleep_obs, "night") > 0.5 or bool(
            meta.get("night", False)
        )
        shelter_role_level = max(
            self._b_series_float(sleep_obs, "shelter_role_level"),
            self._b_series_float(meta, "shelter_role_level"),
        )
        rest_pressure = bool(night or fatigue >= 0.25 or sleep_debt >= 0.25)
        recent_contact_pressure = max(
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "recent_contact"),
        )
        current_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "predator_smell_strength"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
        )
        memory_vectors = _mapping(meta.get("memory_vectors"))
        predator_memory = _mapping(memory_vectors.get("predator"))
        predator_memory_pressure = (
            1.0 - self._b_series_float(predator_memory, "age")
            if abs(_signed_float(predator_memory.get("dx", 0.0)))
            + abs(_signed_float(predator_memory.get("dy", 0.0)))
            > 0.05
            else 0.0
        )
        predator_trace = _mapping(_mapping(meta.get("percept_traces")).get("predator"))
        predator_trace_pressure = max(
            self._b_series_float(predator_trace, "strength"),
            self._b_series_float(predator_trace, "freshness"),
            self._b_series_float(predator_trace, "certainty"),
        )
        temporal_threat = max(
            current_threat,
            0.85 * predator_memory_pressure,
            0.75 * predator_trace_pressure,
        )

        if on_food and hunger >= 0.10:
            semantic_action = "EAT"
            reason = "b2_temporal_threat_eat_on_food"
        elif on_shelter and health < 0.45 and hunger < 0.65:
            if shelter_role != "deep" or shelter_role_level < 0.95:
                semantic_action = "MOVE_TO_SHELTER"
                reason = "b2_temporal_threat_low_health_deepen"
            elif rest_pressure or fatigue >= 0.15 or sleep_debt >= 0.15:
                semantic_action = "SLEEP"
                reason = "b2_temporal_threat_low_health_rest"
            else:
                semantic_action = "STAY"
                reason = "b2_temporal_threat_low_health_hold"
        elif on_shelter and (
            (current_threat >= 0.70 and hunger < 0.62 and health < 0.70)
            or predator_trace_pressure >= 0.75
            or (predator_memory_pressure >= 0.85 and hunger < 0.55)
        ):
            if hunger >= 0.62 and not (
                predator_trace_pressure >= 0.75 and hunger < 0.62
            ):
                semantic_action = "MOVE_TO_FOOD"
                reason = "b2_temporal_threat_safe_hunger_release"
            elif shelter_role != "deep" or shelter_role_level < 0.95:
                semantic_action = "MOVE_TO_SHELTER"
                reason = "b2_temporal_threat_deepen_shelter"
            elif rest_pressure and hunger < 0.70:
                semantic_action = "SLEEP"
                reason = "b2_temporal_threat_rest_deep"
            else:
                semantic_action = "STAY"
                reason = "b2_temporal_threat_hold_deep"
        elif (
            on_shelter
            and semantic_action == "MOVE_TO_SHELTER"
            and hunger >= 0.70
            and current_threat < 0.90
            and predator_trace_pressure < 0.75
        ):
            semantic_action = "MOVE_TO_FOOD"
            reason = "b2_temporal_threat_shelter_role_hunger_release"
        elif (
            not on_shelter
            and hunger >= 0.74
            and health <= 0.35
            and health >= 0.08
            and current_threat < 0.90
            and recent_contact_pressure < 0.50
            and predator_trace_pressure < 0.75
        ):
            semantic_action = "MOVE_TO_FOOD"
            reason = "b2_temporal_threat_emergency_food_over_recover"
        elif not on_shelter and hunger < 0.80 and (
            predator_trace_pressure >= 0.70
            or (predator_memory_pressure >= 0.85 and hunger < 0.60)
            or (current_threat >= 0.50 and health < 0.70 and hunger < 0.75)
            or (current_threat >= 0.50 and 0.45 <= health < 0.70 and hunger < 0.90)
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b2_temporal_threat_return_from_recent_threat"
        else:
            reason = f"b2_temporal_threat_{reason}"

        trace_metrics = {
            "b_current_threat_pressure": round(float(current_threat), 6),
            "b_temporal_threat_pressure": round(float(temporal_threat), 6),
            "b_predator_memory_pressure": round(float(predator_memory_pressure), 6),
            "b_predator_trace_pressure": round(float(predator_trace_pressure), 6),
        }
        return (
            semantic_action,
            B2_TEMPORAL_THREAT_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_metrics,
        )

    def _b3_reset_contact_memory_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b3_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b3_contact_cooldown = 0
        self._b3_post_food_cooldown = 0
        self._b3_last_hunger = None
        self._b3_last_on_food = False

    def _b3_contact_memory_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float | int | str]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_metrics,
        ) = self._b2_temporal_threat_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        profile = (
            str(getattr(self, "_b3_contact_memory_profile_override"))
            if getattr(self, "_b3_contact_memory_profile_override", None) is not None
            else (
                "strict"
                if str(getattr(self.config, "name", ""))
                == B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME
                else "standard"
            )
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b3_reset_contact_memory_if_needed(tick)

        def _mapping(value: object) -> dict[str, object]:
            return value if isinstance(value, dict) else {}

        hunger = self._b_series_float(hunger_obs, "hunger")
        health = self._b_series_float(sleep_obs, "health")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or shelter_role != "outside"
        )
        shelter_role_level = max(
            self._b_series_float(sleep_obs, "shelter_role_level"),
            self._b_series_float(meta, "shelter_role_level"),
        )
        current_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "predator_smell_strength"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
        )
        recent_contact_pressure = max(
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "recent_contact"),
        )
        recent_pain = max(
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(meta, "recent_pain"),
        )
        predator_trace = _mapping(_mapping(meta.get("percept_traces")).get("predator"))
        predator_trace_pressure = max(
            self._b_series_float(predator_trace, "strength"),
            self._b_series_float(predator_trace, "freshness"),
            self._b_series_float(predator_trace, "certainty"),
        )
        temporal_threat = float(trace_metrics.get("b_temporal_threat_pressure", 0.0))
        previous_hunger = getattr(self, "_b3_last_hunger", None)
        hunger_drop = (
            max(0.0, float(previous_hunger) - float(hunger))
            if previous_hunger is not None
            else 0.0
        )
        contact_cooldown = int(getattr(self, "_b3_contact_cooldown", 0))
        post_food_cooldown = int(getattr(self, "_b3_post_food_cooldown", 0))
        if recent_contact_pressure >= 0.35 or recent_pain >= 0.30:
            contact_cooldown = max(contact_cooldown, 14 if profile == "strict" else 10)
        if bool(getattr(self, "_b3_last_on_food", False)) and not on_food:
            post_food_cooldown = max(
                post_food_cooldown,
                10 if profile == "strict" else 7,
            )
        if hunger_drop >= 0.18:
            post_food_cooldown = max(
                post_food_cooldown,
                12 if profile == "strict" else 8,
            )

        contact_active = contact_cooldown > 0 or recent_contact_pressure >= 0.25
        post_food_active = post_food_cooldown > 0 or hunger_drop >= 0.18
        rest_pressure = bool(
            self._b_series_float(sleep_obs, "night") > 0.5
            or bool(meta.get("night", False))
            or fatigue >= 0.25
            or sleep_debt >= 0.25
        )
        extreme_forage_release = (
            hunger >= (0.92 if profile == "strict" else 0.90)
            and current_threat < 0.45
            and predator_trace_pressure < 0.50
            and recent_contact_pressure < 0.20
        )

        if semantic_action == "EAT" and on_food:
            reason = "b3_contact_memory_eat_on_food"
        elif on_shelter and contact_active and not extreme_forage_release:
            if shelter_role != "deep" or shelter_role_level < 0.95:
                semantic_action = "MOVE_TO_SHELTER"
                reason = "b3_contact_memory_deepen_contact_cooldown"
            elif rest_pressure and hunger < 0.72:
                semantic_action = "SLEEP"
                reason = "b3_contact_memory_rest_contact_cooldown"
            else:
                semantic_action = "STAY"
                reason = "b3_contact_memory_hold_contact_cooldown"
        elif (
            not on_shelter
            and contact_active
            and not extreme_forage_release
            and hunger < (0.88 if profile == "strict" else 0.86)
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b3_contact_memory_return_contact_cooldown"
        elif (
            not on_shelter
            and post_food_active
            and current_threat >= (0.38 if profile == "strict" else 0.45)
            and hunger < (0.88 if profile == "strict" else 0.86)
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b3_contact_memory_return_after_food"
        elif (
            semantic_action == "MOVE_TO_FOOD"
            and not on_shelter
            and current_threat >= (0.55 if profile == "strict" else 0.62)
            and hunger < (0.86 if profile == "strict" else 0.82)
            and health >= 0.12
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b3_contact_memory_cancel_forage_under_threat"
        else:
            reason = f"b3_contact_memory_{reason}"

        trace_payload: dict[str, float | int | str] = {
            **trace_metrics,
            "b3_contact_cooldown": int(contact_cooldown),
            "b3_post_food_cooldown": int(post_food_cooldown),
            "b3_hunger_drop": round(float(hunger_drop), 6),
            "b3_controller_profile": profile,
        }
        self._b3_last_hunger = float(hunger)
        self._b3_last_on_food = bool(on_food)
        self._b3_contact_cooldown = max(0, int(contact_cooldown) - 1)
        self._b3_post_food_cooldown = max(0, int(post_food_cooldown) - 1)
        self._b3_last_tick = int(tick)
        return (
            semantic_action,
            B3_CONTACT_MEMORY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b3_recurrent_guard_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float | int | str]]:
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        if getattr(self, "_b3_recurrent_guard_last_tick", None) is None or tick <= int(
            getattr(self, "_b3_recurrent_guard_last_tick", 0)
        ):
            hunger_obs = self._bound_observation("hunger_center", observation)
            initial_hunger = self._b_series_float(hunger_obs, "hunger")
            self._b3_recurrent_guard_profile = (
                "easy_like_b2" if initial_hunger < 0.80 else "canonical_guard"
            )
        profile = str(getattr(self, "_b3_recurrent_guard_profile", "canonical_guard"))
        self._b3_recurrent_guard_last_tick = int(tick)
        if profile == "easy_like_b2":
            (
                semantic_action,
                _source,
                reason,
                _override_count,
                trace_payload,
            ) = self._b2_temporal_threat_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            trace_payload = dict(trace_payload)
            trace_payload.update(
                {
                    "b3_contact_cooldown": 0,
                    "b3_post_food_cooldown": 0,
                    "b3_hunger_drop": 0.0,
                    "b3_controller_profile": "recurrent_guard_easy_b2",
                }
            )
            return (
                semantic_action,
                B3_RECURRENT_GUARD_SELECTION_SOURCE,
                f"b3_recurrent_guard_easy_b2_{reason}",
                int(semantic_action != learned_semantic_action),
                trace_payload,
            )

        previous_override = getattr(self, "_b3_contact_memory_profile_override", None)
        self._b3_contact_memory_profile_override = "strict" if tick < 60 else "standard"
        try:
            (
                semantic_action,
                _source,
                reason,
                _override_count,
                trace_payload,
            ) = self._b3_contact_memory_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
        finally:
            self._b3_contact_memory_profile_override = previous_override
        trace_payload = dict(trace_payload)
        phase = "strict_until_60" if tick < 60 else "standard_after_60"
        trace_payload["b3_controller_profile"] = f"recurrent_guard_{phase}"
        return (
            semantic_action,
            B3_RECURRENT_GUARD_SELECTION_SOURCE,
            f"b3_recurrent_guard_{phase}_{reason}",
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b4_controller_params(self) -> dict[str, float]:
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "recovery_balance"
        )
        defaults: dict[str, float] = {
            "recovery_pressure_threshold": 0.62,
            "sleep_hunger_max": 0.72,
            "sleep_threat_max": 0.50,
            "exit_health_min": 0.26,
            "exit_threat_max": 0.55,
            "hunger_release": 0.88,
            "emergency_hunger_release": 0.94,
            "contact_hold_hunger_max": 0.86,
            "return_threat_min": 0.62,
            "return_hunger_max": 0.90,
            "deep_shelter_level": 0.95,
        }
        if profile == "predator_exit_memory":
            defaults.update(
                {
                    "recovery_pressure_threshold": 0.56,
                    "sleep_threat_max": 0.42,
                    "exit_health_min": 0.38,
                    "exit_threat_max": 0.38,
                    "hunger_release": 0.90,
                    "emergency_hunger_release": 0.96,
                    "contact_hold_hunger_max": 0.90,
                    "return_threat_min": 0.50,
                }
            )
        elif profile == "recovery_balance_h56":
            defaults.update(
                {
                    "recovery_pressure_threshold": 0.58,
                    "sleep_hunger_max": 0.74,
                    "exit_health_min": 0.30,
                    "exit_threat_max": 0.50,
                }
            )
        elif profile == "genetic_recovery":
            defaults.update(
                {
                    "recovery_pressure_threshold": 0.58,
                    "sleep_hunger_max": 0.73,
                    "sleep_threat_max": 0.55,
                    "exit_health_min": 0.32,
                    "exit_threat_max": 0.46,
                    "hunger_release": 0.88,
                    "emergency_hunger_release": 0.95,
                    "contact_hold_hunger_max": 0.88,
                    "return_threat_min": 0.54,
                    "return_hunger_max": 0.91,
                }
            )
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            defaults[str(key)] = float(value)
        return defaults

    def _b4_recovery_balance_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float | int | str | bool]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b3_recurrent_guard_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b4_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "recovery_balance"
        )
        source = (
            B4_GENETIC_RECOVERY_SELECTION_SOURCE
            if str(getattr(self.config, "name", ""))
            == B4_GENETIC_RECOVERY_H48_POLICY_NAME
            or profile == "genetic_recovery"
            else B4_RECOVERY_BALANCE_SELECTION_SOURCE
        )
        if trace_payload.get("b3_controller_profile") == "recurrent_guard_easy_b2":
            trace_payload.update(
                {
                    "b4_controller_profile": profile,
                    "b4_recovery_pressure": 0.0,
                    "b4_sleep_hold": False,
                    "b4_exit_blocked": False,
                    "b4_hunger_release": round(float(params["hunger_release"]), 6),
                }
            )
            if "ga_generation" in params:
                trace_payload["b4_genetic_generation"] = int(params["ga_generation"])
            if "ga_candidate" in params:
                trace_payload["b4_genetic_candidate"] = int(params["ga_candidate"])
            return (
                semantic_action,
                source,
                f"b4_recovery_balance_easy_passthrough_{reason}",
                int(semantic_action != learned_semantic_action),
                trace_payload,
            )
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        hunger = self._b_series_float(hunger_obs, "hunger")
        health = self._b_series_float(sleep_obs, "health")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or shelter_role != "outside"
        )
        shelter_role_level = max(
            self._b_series_float(sleep_obs, "shelter_role_level"),
            self._b_series_float(meta, "shelter_role_level"),
        )
        current_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "predator_smell_strength"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
            float(trace_payload.get("b_current_threat_pressure", 0.0) or 0.0),
        )
        temporal_threat = max(
            float(trace_payload.get("b_temporal_threat_pressure", 0.0) or 0.0),
            current_threat,
        )
        predator_memory_pressure = float(
            trace_payload.get("b_predator_memory_pressure", 0.0) or 0.0
        )
        predator_trace_pressure = float(
            trace_payload.get("b_predator_trace_pressure", 0.0) or 0.0
        )
        contact_cooldown = int(trace_payload.get("b3_contact_cooldown", 0) or 0)
        recovery_pressure = max(
            float(fatigue),
            float(sleep_debt),
            1.0 - float(health),
            min(1.0, float(contact_cooldown) / 14.0),
        )
        threat_pressure = max(
            current_threat,
            temporal_threat,
            predator_memory_pressure,
            predator_trace_pressure,
        )
        sleep_hold = False
        exit_blocked = False
        hunger_release = float(params["hunger_release"])
        emergency_release = float(params["emergency_hunger_release"])
        deep_enough = (
            shelter_role == "deep"
            or shelter_role_level >= float(params["deep_shelter_level"])
        )

        if semantic_action == "EAT" and on_food:
            reason = "b4_recovery_balance_eat_on_food"
        elif (
            on_shelter
            and recovery_pressure >= float(params["recovery_pressure_threshold"])
            and hunger < float(params["sleep_hunger_max"])
            and threat_pressure <= float(params["sleep_threat_max"])
        ):
            semantic_action = "SLEEP" if deep_enough else "MOVE_TO_SHELTER"
            sleep_hold = True
            reason = "b4_recovery_balance_sleep_recovery"
        elif (
            on_shelter
            and contact_cooldown > 0
            and hunger < float(params["contact_hold_hunger_max"])
        ):
            semantic_action = "STAY" if deep_enough else "MOVE_TO_SHELTER"
            exit_blocked = True
            reason = "b4_recovery_balance_hold_recent_contact"
        elif (
            on_shelter
            and semantic_action == "MOVE_TO_FOOD"
            and hunger < emergency_release
            and (
                health < float(params["exit_health_min"])
                or threat_pressure > float(params["exit_threat_max"])
            )
        ):
            semantic_action = "STAY" if deep_enough else "MOVE_TO_SHELTER"
            exit_blocked = True
            reason = "b4_recovery_balance_block_unsafe_exit"
        elif (
            not on_shelter
            and semantic_action == "MOVE_TO_FOOD"
            and hunger < float(params["return_hunger_max"])
            and threat_pressure >= float(params["return_threat_min"])
        ):
            semantic_action = "MOVE_TO_SHELTER"
            exit_blocked = True
            reason = "b4_recovery_balance_return_under_threat"
        elif (
            on_shelter
            and hunger >= hunger_release
            and threat_pressure <= float(params["exit_threat_max"])
        ):
            reason = f"b4_recovery_balance_release_forage_{reason}"
        else:
            reason = f"b4_recovery_balance_{reason}"

        trace_payload.update(
            {
                "b4_controller_profile": profile,
                "b4_recovery_pressure": round(float(recovery_pressure), 6),
                "b4_sleep_hold": bool(sleep_hold),
                "b4_exit_blocked": bool(exit_blocked),
                "b4_hunger_release": round(float(hunger_release), 6),
            }
        )
        if "ga_generation" in params:
            trace_payload["b4_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b4_genetic_candidate"] = int(params["ga_candidate"])
        return (
            semantic_action,
            source,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b5_controller_params(self) -> dict[str, float]:
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "homeostatic_arbiter"
        )
        defaults: dict[str, float] = {
            "hunger_release": 0.86,
            "emergency_hunger_release": 0.94,
            "forage_threat_max": 0.56,
            "forage_lock_ticks": 8.0,
            "sleep_pressure_threshold": 0.70,
            "sleep_hunger_max": 0.78,
            "sleep_threat_max": 0.58,
            "sleep_lock_ticks": 7.0,
            "exit_sleep_pressure_max": 0.58,
            "exit_recovery_debt_max": 0.66,
            "exit_threat_max": 0.62,
            "return_sleep_pressure_min": 0.84,
            "return_recovery_debt_min": 0.82,
            "return_hunger_max": 0.82,
        }
        if profile == "circadian_recovery":
            defaults.update(
                {
                    "hunger_release": 0.88,
                    "emergency_hunger_release": 0.96,
                    "forage_threat_max": 0.52,
                    "sleep_pressure_threshold": 0.58,
                    "sleep_hunger_max": 0.82,
                    "sleep_threat_max": 0.62,
                    "sleep_lock_ticks": 9.0,
                    "exit_sleep_pressure_max": 0.46,
                    "exit_recovery_debt_max": 0.56,
                }
            )
        elif profile == "homeostatic_arbiter_h56":
            defaults.update(
                {
                    "hunger_release": 0.85,
                    "forage_threat_max": 0.58,
                    "sleep_pressure_threshold": 0.66,
                    "sleep_hunger_max": 0.80,
                    "exit_sleep_pressure_max": 0.54,
                }
            )
        elif profile == "genetic_homeostasis":
            defaults.update(
                {
                    "hunger_release": 0.86,
                    "emergency_hunger_release": 0.95,
                    "forage_threat_max": 0.58,
                    "forage_lock_ticks": 9.0,
                    "sleep_pressure_threshold": 0.66,
                    "sleep_hunger_max": 0.80,
                    "sleep_threat_max": 0.60,
                    "sleep_lock_ticks": 8.0,
                    "exit_sleep_pressure_max": 0.54,
                    "exit_recovery_debt_max": 0.62,
                    "exit_threat_max": 0.60,
                }
            )
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            defaults[str(key)] = float(value)
        return defaults

    def _b5_reset_homeostatic_locks_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b5_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b5_sleep_bout_lock = 0
        self._b5_forage_commitment_lock = 0

    def _b5_homeostatic_arbiter_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float | int | str | bool]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b4_recovery_balance_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b5_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "homeostatic_arbiter"
        )
        source = (
            B5_GENETIC_HOMEOSTASIS_SELECTION_SOURCE
            if str(getattr(self.config, "name", ""))
            == B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME
            or profile == "genetic_homeostasis"
            else B5_HOMEOSTATIC_ARBITER_SELECTION_SOURCE
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b5_reset_homeostatic_locks_if_needed(tick)
        if trace_payload.get("b3_controller_profile") == "recurrent_guard_easy_b2":
            trace_payload.update(
                {
                    "b5_controller_profile": profile,
                    "b5_hunger_urgency": 0.0,
                    "b5_sleep_pressure": 0.0,
                    "b5_recovery_debt": 0.0,
                    "b5_threat_gate": 0.0,
                    "b5_sleep_bout_lock": 0,
                    "b5_forage_commitment_lock": 0,
                    "b5_homeostatic_decision": "easy_passthrough",
                }
            )
            if "ga_generation" in params:
                trace_payload["b5_genetic_generation"] = int(params["ga_generation"])
            if "ga_candidate" in params:
                trace_payload["b5_genetic_candidate"] = int(params["ga_candidate"])
            return (
                semantic_action,
                source,
                f"b5_homeostatic_easy_passthrough_{reason}",
                int(semantic_action != learned_semantic_action),
                trace_payload,
            )

        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        hunger = self._b_series_float(hunger_obs, "hunger")
        health = self._b_series_float(sleep_obs, "health")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or shelter_role != "outside"
        )
        current_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "predator_smell_strength"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
            float(trace_payload.get("b_temporal_threat_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_predator_memory_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_predator_trace_pressure", 0.0) or 0.0),
        )
        sleep_pressure = max(float(fatigue), float(sleep_debt))
        recovery_debt = max(
            1.0 - float(health),
            float(sleep_debt) * 0.90,
            float(fatigue) * 0.85,
            float(trace_payload.get("b4_recovery_pressure", 0.0) or 0.0),
        )
        threat_gate = float(current_threat)
        sleep_lock = int(getattr(self, "_b5_sleep_bout_lock", 0))
        forage_lock = int(getattr(self, "_b5_forage_commitment_lock", 0))
        decision_label = "preserve_b4"

        if (
            on_shelter
            and sleep_pressure >= float(params["sleep_pressure_threshold"])
            and hunger < float(params["sleep_hunger_max"])
            and threat_gate <= float(params["sleep_threat_max"])
        ):
            sleep_lock = max(sleep_lock, int(params["sleep_lock_ticks"]))
        if (
            hunger >= float(params["hunger_release"])
            and threat_gate <= float(params["forage_threat_max"])
            and not on_food
        ):
            forage_lock = max(forage_lock, int(params["forage_lock_ticks"]))

        if semantic_action == "EAT" and on_food:
            forage_lock = 0
            decision_label = "eat_on_food"
            reason = "b5_homeostatic_eat_on_food"
        elif (
            on_shelter
            and sleep_lock > 0
            and hunger < float(params["emergency_hunger_release"])
            and threat_gate <= float(params["sleep_threat_max"])
        ):
            semantic_action = "SLEEP"
            decision_label = "sleep_bout_hold"
            reason = "b5_homeostatic_sleep_bout_hold"
        elif (
            on_shelter
            and semantic_action == "MOVE_TO_FOOD"
            and hunger < float(params["emergency_hunger_release"])
            and (
                sleep_pressure > float(params["exit_sleep_pressure_max"])
                or recovery_debt > float(params["exit_recovery_debt_max"])
                or threat_gate > float(params["exit_threat_max"])
            )
        ):
            semantic_action = "STAY"
            decision_label = "block_premature_exit"
            reason = "b5_homeostatic_block_premature_exit"
        elif (
            not on_shelter
            and semantic_action == "MOVE_TO_FOOD"
            and hunger < float(params["return_hunger_max"])
            and (
                sleep_pressure >= float(params["return_sleep_pressure_min"])
                or recovery_debt >= float(params["return_recovery_debt_min"])
            )
            and threat_gate <= float(params["sleep_threat_max"])
        ):
            semantic_action = "MOVE_TO_SHELTER"
            decision_label = "return_for_recovery"
            reason = "b5_homeostatic_return_for_recovery"
        elif (
            forage_lock > 0
            and hunger >= float(params["hunger_release"])
            and threat_gate <= float(params["forage_threat_max"])
        ):
            semantic_action = "MOVE_TO_FOOD"
            decision_label = "forage_commitment"
            reason = "b5_homeostatic_forage_commitment"
        else:
            reason = f"b5_homeostatic_{reason}"

        trace_payload.update(
            {
                "b5_controller_profile": profile,
                "b5_hunger_urgency": round(float(hunger), 6),
                "b5_sleep_pressure": round(float(sleep_pressure), 6),
                "b5_recovery_debt": round(float(recovery_debt), 6),
                "b5_threat_gate": round(float(threat_gate), 6),
                "b5_sleep_bout_lock": int(sleep_lock),
                "b5_forage_commitment_lock": int(forage_lock),
                "b5_homeostatic_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b5_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b5_genetic_candidate"] = int(params["ga_candidate"])

        self._b5_sleep_bout_lock = max(0, int(sleep_lock) - 1)
        self._b5_forage_commitment_lock = max(0, int(forage_lock) - 1)
        self._b5_last_tick = int(tick)
        return (
            semantic_action,
            source,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b6_controller_params(self) -> dict[str, float]:
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "risk_forage_arbiter"
        )
        defaults: dict[str, float] = {
            "b6_family": 1.0,
            "b6_risk_threshold": 0.35,
            "b6_corridor_hunger": 0.86,
            "b6_corridor_lock_ticks": 10.0,
            "b6_threat_memory_ticks": 8.0,
            "b6_return_lock_ticks": 6.0,
            "b6_recurrent_decay": 0.65,
        }
        if profile == "corridor_survival_guard":
            defaults.update(
                {
                    "b6_risk_threshold": 0.40,
                    "b6_corridor_hunger": 0.82,
                    "b6_corridor_lock_ticks": 14.0,
                }
            )
        elif profile == "threat_priority_memory":
            defaults.update(
                {
                    "b6_risk_threshold": 0.22,
                    "b6_threat_memory_ticks": 10.0,
                    "b6_return_lock_ticks": 8.0,
                }
            )
        elif profile == "risk_corridor_h56":
            defaults.update(
                {
                    "b6_risk_threshold": 0.32,
                    "b6_corridor_hunger": 0.84,
                    "b6_corridor_lock_ticks": 12.0,
                }
            )
        elif profile == "genetic_risk_corridor":
            defaults.update(
                {
                    "b6_risk_threshold": 0.30,
                    "b6_corridor_hunger": 0.83,
                    "b6_corridor_lock_ticks": 14.0,
                    "b6_threat_memory_ticks": 12.0,
                }
            )
        elif profile == "recurrent_context":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_recurrent_decay": 0.70,
                    "b6_return_lock_ticks": 8.0,
                }
            )
        elif profile == "recurrent_threat_homeostasis":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_risk_threshold": 0.28,
                    "b6_threat_memory_ticks": 12.0,
                    "b6_return_lock_ticks": 10.0,
                    "b6_recurrent_decay": 0.78,
                }
            )
        elif profile == "recurrent_corridor_guard":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_corridor_hunger": 0.82,
                    "b6_corridor_lock_ticks": 16.0,
                    "b6_recurrent_decay": 0.80,
                }
            )
        elif profile == "recurrent_context_h56":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_corridor_hunger": 0.84,
                    "b6_recurrent_decay": 0.75,
                    "b6_return_lock_ticks": 8.0,
                }
            )
        elif profile == "genetic_recurrent_memory":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_risk_threshold": 0.30,
                    "b6_corridor_hunger": 0.83,
                    "b6_corridor_lock_ticks": 14.0,
                    "b6_threat_memory_ticks": 12.0,
                    "b6_return_lock_ticks": 10.0,
                    "b6_recurrent_decay": 0.78,
                }
            )
        elif profile == "fused_risk_recurrent":
            defaults.update(
                {
                    "b6_family": 3.0,
                    "b6_risk_threshold": 0.28,
                    "b6_corridor_hunger": 0.82,
                    "b6_corridor_lock_ticks": 16.0,
                    "b6_threat_memory_ticks": 12.0,
                    "b6_return_lock_ticks": 10.0,
                    "b6_recurrent_decay": 0.80,
                }
            )
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            defaults[str(key)] = float(value)
        return defaults

    def _b6_controller_family(self, profile: str, params: dict[str, float]) -> str:
        name = str(getattr(self.config, "name", ""))
        if name == B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME or profile == "fused_risk_recurrent":
            return "fused_risk_recurrent"
        if name in {
            B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
            B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME,
            B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME,
            B6_RECURRENT_CONTEXT_H56_POLICY_NAME,
            B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME,
        } or profile.startswith("recurrent") or profile == "genetic_recurrent_memory":
            return "recurrent_memory"
        if int(round(float(params.get("b6_family", 1.0)))) == 2:
            return "recurrent_memory"
        if int(round(float(params.get("b6_family", 1.0)))) == 3:
            return "fused_risk_recurrent"
        return "risk_corridor"

    def _b6_reset_recurrent_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b6_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b6_corridor_commitment = 0
        self._b6_corridor_progress_memory = 0.0
        self._b6_recurrent_threat_memory = 0.0
        self._b6_return_lock = 0
        self._b6_last_tick = int(tick)

    def _b6_local_food_progress_signal(self, meta: dict[str, object]) -> float:
        transitions = meta.get("local_transition_consequences")
        transitions = transitions if isinstance(transitions, dict) else {}
        best = 0.0
        for transition in transitions.values():
            if not isinstance(transition, dict):
                continue
            best = max(
                best,
                self._b_series_float(transition, "food_dist_delta"),
                1.0 if bool(transition.get("next_cell_has_food", False)) else 0.0,
            )
        return float(best)

    def _b6_risk_corridor_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b5_homeostatic_arbiter_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b6_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "risk_forage_arbiter"
        )
        family = self._b6_controller_family(profile, params)
        if family == "fused_risk_recurrent":
            source = B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE
        elif family == "recurrent_memory":
            source = B6_RECURRENT_MEMORY_SELECTION_SOURCE
        else:
            source = B6_RISK_CORRIDOR_SELECTION_SOURCE

        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b6_reset_recurrent_state_if_needed(tick)
        if trace_payload.get("b5_homeostatic_decision") == "easy_passthrough":
            trace_payload.update(
                {
                    "b6_controller_family": family,
                    "b6_controller_profile": profile,
                    "b6_risk_pressure": 0.0,
                    "b6_threat_priority": 0.0,
                    "b6_forage_suppressed": 0.0,
                    "b6_corridor_commitment": 0,
                    "b6_corridor_progress_memory": 0.0,
                    "b6_recurrent_state": "easy_passthrough",
                    "b6_return_lock": 0,
                    "b6_decision": "easy_passthrough",
                    "b6_emit_action_center_payload": True,
                    "b6_action_center_payload": {
                        "winning_valence": "exploration",
                        "evidence": {
                            "threat": {
                                "predator_visible": 0.0,
                                "predator_proximity": 0.0,
                                "predator_certainty": 0.0,
                            }
                        },
                        "module_gates": {"hunger_center": 1.0},
                    },
                }
            )
            if "ga_generation" in params:
                trace_payload["b6_genetic_generation"] = int(params["ga_generation"])
            if "ga_candidate" in params:
                trace_payload["b6_genetic_candidate"] = int(params["ga_candidate"])
            self._b6_last_tick = int(tick)
            return (
                semantic_action,
                source,
                f"b6_easy_passthrough_{reason}",
                int(semantic_action != learned_semantic_action),
                trace_payload,
            )

        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        hunger = self._b_series_float(hunger_obs, "hunger")
        health = self._b_series_float(sleep_obs, "health")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or shelter_role != "outside"
        )
        current_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "predator_smell_strength"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
        )
        risk_pressure = max(
            current_threat,
            float(trace_payload.get("b5_threat_gate", 0.0) or 0.0),
            float(trace_payload.get("b_temporal_threat_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_predator_memory_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_predator_trace_pressure", 0.0) or 0.0),
        )
        recent_contact = max(
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
            self._b_series_float(meta, "recent_pain"),
        )
        map_template = str(meta.get("map_template", ""))
        central_retention_map = map_template == "central_burrow"
        corridor_map = map_template == "corridor_escape"
        food_progress_signal = self._b6_local_food_progress_signal(meta)
        corridor_commitment = int(getattr(self, "_b6_corridor_commitment", 0))
        corridor_memory = float(getattr(self, "_b6_corridor_progress_memory", 0.0))
        recurrent_threat = float(getattr(self, "_b6_recurrent_threat_memory", 0.0))
        return_lock = int(getattr(self, "_b6_return_lock", 0))
        decay = float(params["b6_recurrent_decay"])
        if family in {"recurrent_memory", "fused_risk_recurrent"}:
            recurrent_threat = max(recurrent_threat * decay, risk_pressure)
            corridor_memory = max(corridor_memory * decay, food_progress_signal)
            if risk_pressure >= float(params["b6_risk_threshold"]) or recent_contact > 0.0:
                return_lock = max(return_lock, int(params["b6_return_lock_ticks"]))
        else:
            recurrent_threat = max(recurrent_threat * 0.50, risk_pressure)
            corridor_memory = max(corridor_memory * 0.50, food_progress_signal)

        threat_priority = float(
            max(risk_pressure, recurrent_threat)
            >= float(params["b6_risk_threshold"])
        )
        forage_suppressed = 0.0
        decision_label = "preserve_b5"
        emergency_hunger = float(params.get("emergency_hunger_release", 0.95))
        shelter_block_hunger = min(0.70, emergency_hunger)
        corridor_hunger = float(params["b6_corridor_hunger"])

        if on_food and hunger >= 0.10:
            semantic_action = "EAT"
            corridor_commitment = 0
            return_lock = 0
            decision_label = "eat_on_food"
            reason = "b6_eat_on_food"
        elif (
            threat_priority > 0.0
            and not on_shelter
            and hunger < shelter_block_hunger
            and not central_retention_map
            and family in {"risk_corridor", "fused_risk_recurrent"}
        ):
            semantic_action = "MOVE_TO_SHELTER"
            forage_suppressed = 1.0
            return_lock = max(return_lock, int(params["b6_return_lock_ticks"]))
            decision_label = "risk_shelter_return"
            reason = "b6_risk_corridor_return_under_threat"
        elif (
            threat_priority > 0.0
            and not on_shelter
            and hunger < shelter_block_hunger
            and not central_retention_map
            and family == "recurrent_memory"
            and recurrent_threat >= float(params["b6_risk_threshold"])
        ):
            semantic_action = "MOVE_TO_SHELTER"
            forage_suppressed = 1.0
            return_lock = max(return_lock, int(params["b6_return_lock_ticks"]))
            decision_label = "recurrent_risk_return"
            reason = "b6_recurrent_memory_return_lock"
        elif (
            return_lock > 0
            and not on_shelter
            and hunger < shelter_block_hunger
            and not central_retention_map
        ):
            semantic_action = "MOVE_TO_SHELTER"
            forage_suppressed = 1.0
            decision_label = "return_lock"
            reason = "b6_return_lock_shelter"
        elif (
            corridor_map
            and hunger >= corridor_hunger
            and risk_pressure < 0.80
            and recent_contact <= 0.0
            and not on_food
            and food_progress_signal > 0.0
        ):
            semantic_action = "MOVE_TO_FOOD"
            corridor_commitment = max(
                corridor_commitment,
                int(params["b6_corridor_lock_ticks"]),
            )
            decision_label = "corridor_commitment"
            reason = "b6_corridor_commitment_food_progress"
        elif (
            corridor_map
            and corridor_commitment > 0
            and not on_food
            and risk_pressure < 0.75
        ):
            semantic_action = "MOVE_TO_FOOD"
            decision_label = "corridor_commitment_hold"
            reason = "b6_corridor_commitment_hold"
        else:
            reason = f"b6_{family}_{reason}"

        action_center_threat = max(
            risk_pressure,
            recurrent_threat,
            0.55 if threat_priority > 0.0 else 0.0,
        )
        if threat_priority > 0.0 and hunger < emergency_hunger:
            forage_suppressed = max(float(forage_suppressed), 1.0)
        winning_valence = "threat" if threat_priority > 0.0 else (
            "hunger" if hunger >= corridor_hunger else "exploration"
        )
        hunger_gate = 0.20 if threat_priority > 0.0 else 1.0
        trace_payload.update(
            {
                "b6_controller_family": family,
                "b6_controller_profile": profile,
                "b6_risk_pressure": round(float(risk_pressure), 6),
                "b6_threat_priority": round(float(threat_priority), 6),
                "b6_forage_suppressed": round(float(forage_suppressed), 6),
                "b6_corridor_commitment": int(corridor_commitment),
                "b6_corridor_progress_memory": round(float(corridor_memory), 6),
                "b6_recurrent_state": (
                    "active"
                    if family in {"recurrent_memory", "fused_risk_recurrent"}
                    and (recurrent_threat > 0.0 or corridor_memory > 0.0)
                    else "risk_only"
                ),
                "b6_return_lock": int(return_lock),
                "b6_decision": decision_label,
                "b6_emit_action_center_payload": True,
                "b6_action_center_payload": {
                    "winning_valence": winning_valence,
                    "evidence": {
                        "threat": {
                            "predator_visible": round(
                                float(1.0 if action_center_threat >= 0.30 else 0.0),
                                6,
                            ),
                            "predator_proximity": round(
                                float(max(action_center_threat, 0.35 if threat_priority > 0.0 else 0.0)),
                                6,
                            ),
                            "predator_certainty": round(
                                float(max(action_center_threat, 0.45 if threat_priority > 0.0 else 0.0)),
                                6,
                            ),
                        }
                    },
                    "module_gates": {"hunger_center": round(float(hunger_gate), 6)},
                },
            }
        )
        if "ga_generation" in params:
            trace_payload["b6_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b6_genetic_candidate"] = int(params["ga_candidate"])

        self._b6_corridor_commitment = max(0, int(corridor_commitment) - 1)
        self._b6_corridor_progress_memory = float(corridor_memory)
        self._b6_recurrent_threat_memory = float(recurrent_threat)
        self._b6_return_lock = max(0, int(return_lock) - 1)
        self._b6_last_tick = int(tick)
        return (
            semantic_action,
            source,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b7_controller_params(self) -> dict[str, float]:
        params = self._b6_controller_params()
        defaults = {
            "b7_budget_step_cost": 0.085,
            "b7_viability_margin": -0.08,
            "b7_abort_health": 0.36,
            "b7_recover_health": 0.42,
            "b7_food_commit_distance": 13.0,
            "b7_commitment_ticks": 8.0,
            "b7_recurrent_decay": 0.72,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "affordance_budget"
        )
        if profile == "energy_budget_corridor":
            defaults.update({"b7_viability_margin": -0.02, "b7_abort_health": 0.42})
        elif profile == "recurrent_affordance":
            defaults.update({"b7_recurrent_decay": 0.84, "b7_commitment_ticks": 10.0})
        elif profile == "affordance_budget_h56":
            defaults.update({"b7_budget_step_cost": 0.080, "b7_commitment_ticks": 10.0})
        elif profile == "genetic_affordance_budget":
            defaults.update({"b7_abort_health": 0.38, "b7_recurrent_decay": 0.80})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b7_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b7_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b7_commitment_lock = 0
        self._b7_last_food_distance = None
        self._b7_progress_memory = 0.0
        self._b7_last_tick = int(tick)

    def _b7_affordance_budget_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b6_risk_corridor_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b7_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "affordance_budget"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b7_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        map_template = str(meta.get("map_template", ""))
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        hunger = self._b_series_float(hunger_obs, "hunger")
        health = self._b_series_float(sleep_obs, "health")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or str(meta.get("shelter_role", "outside")) != "outside"
        )

        def _raw_float(key: str, default: float = 0.0) -> float:
            try:
                value = float(meta.get(key, default))
            except (TypeError, ValueError):
                return float(default)
            return float(value) if np.isfinite(value) else float(default)

        food_steps = max(0.0, _raw_float("food_dist", 0.0))
        return_steps = max(0.0, _raw_float("shelter_dist", 0.0))
        threat = max(
            float(trace_payload.get("b6_risk_pressure", 0.0) or 0.0),
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
        )
        step_cost = float(params["b7_budget_step_cost"])
        energy_budget = health - min(1.0, hunger * 0.08) - min(0.25, threat * 0.12)
        budget_margin = energy_budget - (min(food_steps, return_steps + 1.0) * step_cost)

        last_food_distance = getattr(self, "_b7_last_food_distance", None)
        recent_progress = 0.0
        if last_food_distance is not None:
            recent_progress = max(0.0, float(last_food_distance) - food_steps)
        progress_memory = max(
            float(getattr(self, "_b7_progress_memory", 0.0))
            * float(params["b7_recurrent_decay"]),
            recent_progress,
            float(trace_payload.get("b6_corridor_progress_memory", 0.0) or 0.0),
        )
        commitment_lock = int(getattr(self, "_b7_commitment_lock", 0))
        corridor_map = map_template == "corridor_escape"
        decision_label = "preserve_b6"
        affordance_state = "non_corridor"
        abort_return = False
        viability = 1.0

        if on_food:
            semantic_action = "EAT"
            decision_label = "eat_on_food"
            affordance_state = "food_reached"
            commitment_lock = 0
            reason = "b7_eat_on_food"
        elif corridor_map:
            affordance_state = "corridor_open"
            explicit_budget_risk = (
                health <= float(params["b7_abort_health"])
                and food_steps <= float(params["b7_food_commit_distance"])
            )
            if on_shelter and health < float(params["b7_recover_health"]):
                semantic_action = "STAY"
                decision_label = "recover_before_crossing"
                affordance_state = "recover_in_shelter"
                reason = "b7_recover_before_crossing"
            elif explicit_budget_risk:
                abort_return = True
                decision_label = "abort_return_unviable"
                affordance_state = "budget_unviable_commitment"
                reason = "b7_abort_return_unviable"
            elif (
                budget_margin >= float(params["b7_viability_margin"])
                or semantic_action == "MOVE_TO_FOOD"
                or commitment_lock > 0
            ):
                semantic_action = "MOVE_TO_FOOD"
                commitment_lock = max(commitment_lock, int(params["b7_commitment_ticks"]))
                decision_label = "continue_viable"
                affordance_state = "corridor_commitment"
                reason = "b7_continue_viable"
            else:
                abort_return = True
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "abort_return_unviable"
                affordance_state = "budget_unviable_return"
                reason = "b7_abort_return_unviable"
            viability = float(np.clip(0.5 + budget_margin, 0.0, 1.0))

        trace_payload.update(
            {
                "b7_controller_profile": profile,
                "b7_affordance_state": affordance_state,
                "b7_energy_budget": round(float(energy_budget), 6),
                "b7_budget_margin": round(float(budget_margin), 6),
                "b7_food_steps_estimate": round(float(food_steps), 6),
                "b7_return_steps_estimate": round(float(return_steps), 6),
                "b7_corridor_viability": round(float(viability), 6),
                "b7_abort_return": bool(abort_return),
                "b7_commitment_lock": int(commitment_lock),
                "b7_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b7_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b7_genetic_candidate"] = int(params["ga_candidate"])

        self._b7_commitment_lock = max(0, int(commitment_lock) - 1)
        self._b7_last_food_distance = float(food_steps)
        self._b7_progress_memory = float(progress_memory)
        self._b7_last_tick = int(tick)
        return (
            semantic_action,
            B7_AFFORDANCE_BUDGET_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b8_controller_params(self) -> dict[str, float]:
        params = self._b7_controller_params()
        defaults = {
            "b8_place_memory_decay": 0.78,
            "b8_dead_end_risk_threshold": 0.62,
            "b8_return_vector_threshold": 0.18,
            "b8_abort_health": 0.18,
            "b8_hold_health": 0.10,
            "b8_food_progress_floor": 13.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "spatial_affordance_map"
        )
        if profile == "return_vector":
            defaults.update({"b8_return_vector_threshold": 0.10, "b8_abort_health": 0.24})
        elif profile == "corridor_place_memory":
            defaults.update({"b8_place_memory_decay": 0.88, "b8_dead_end_risk_threshold": 0.55})
        elif profile == "spatial_affordance_map_h56":
            defaults.update({"b8_place_memory_decay": 0.82, "b8_food_progress_floor": 12.5})
        elif profile == "genetic_spatial_affordance":
            defaults.update({"b8_place_memory_decay": 0.84, "b8_abort_health": 0.20})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b8_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b8_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b8_place_memory = 0.0
        self._b8_last_food_distance = None
        self._b8_last_tick = int(tick)

    def _b8_spatial_affordance_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b7_affordance_budget_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b8_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "spatial_affordance_map"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b8_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        map_template = str(meta.get("map_template", ""))
        sleep_obs = self._bound_observation("sleep_center", observation)
        health = self._b_series_float(sleep_obs, "health")

        def _raw_float(mapping: dict[str, object], key: str, default: float = 0.0) -> float:
            try:
                value = float(mapping.get(key, default))
            except (TypeError, ValueError):
                return float(default)
            return float(value) if np.isfinite(value) else float(default)

        def _unblocked_delta(action: str, field: str) -> float:
            affordances = meta.get("local_affordances")
            affordances = affordances if isinstance(affordances, dict) else {}
            transitions = meta.get("local_transition_consequences")
            transitions = transitions if isinstance(transitions, dict) else {}
            action_affordance = affordances.get(action)
            action_affordance = (
                action_affordance if isinstance(action_affordance, dict) else {}
            )
            if bool(action_affordance.get("blocked", False)):
                return -1.0
            action_transition = transitions.get(action)
            action_transition = (
                action_transition if isinstance(action_transition, dict) else {}
            )
            return _raw_float(action_transition, field, 0.0)

        food_deltas = [
            _unblocked_delta(action, "food_dist_delta")
            for action in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT")
        ]
        shelter_deltas = [
            _unblocked_delta(action, "shelter_dist_delta")
            for action in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT")
        ]
        best_food_delta = max(food_deltas) if food_deltas else 0.0
        best_shelter_delta = max(shelter_deltas) if shelter_deltas else 0.0
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        last_food_distance = getattr(self, "_b8_last_food_distance", None)
        recent_progress = (
            max(0.0, float(last_food_distance) - food_steps)
            if last_food_distance is not None
            else 0.0
        )
        place_memory = max(
            float(getattr(self, "_b8_place_memory", 0.0))
            * float(params["b8_place_memory_decay"]),
            recent_progress,
            float(trace_payload.get("b7_corridor_viability", 0.0) or 0.0) * 0.25,
        )
        threat = float(trace_payload.get("b6_risk_pressure", 0.0) or 0.0)
        local_affordance_score = best_food_delta + 0.25 * best_shelter_delta - 0.5 * threat
        return_vector_strength = max(0.0, best_shelter_delta)
        dead_end_risk = float(np.clip(1.0 - max(best_food_delta, 0.0) + threat * 0.25, 0.0, 1.0))
        corridor_map = map_template == "corridor_escape"
        b7_decision = str(trace_payload.get("b7_decision", "preserve_b7"))
        decision_label = "preserve_b7"
        spatial_map_state = "non_corridor"
        abort_executed = False

        if corridor_map:
            spatial_map_state = "corridor_place_field"
            if b7_decision == "abort_return_unviable":
                decision_label = "corridor_abort_signal"
                spatial_map_state = "return_vector_available"
                if (
                    profile == "return_vector"
                    and health <= float(params["b8_abort_health"])
                    and return_vector_strength >= float(params["b8_return_vector_threshold"])
                ):
                    semantic_action = "MOVE_TO_SHELTER"
                    abort_executed = True
                    reason = "b8_return_vector_abort"
            elif (
                dead_end_risk >= float(params["b8_dead_end_risk_threshold"])
                and health <= float(params["b8_hold_health"])
            ):
                decision_label = "corridor_hold_unviable"
                semantic_action = "STAY"
                spatial_map_state = "dead_end_hold"
                reason = "b8_spatial_hold"
            elif (
                best_food_delta > 0.0
                or food_steps <= float(params["b8_food_progress_floor"])
                or b7_decision == "continue_viable"
            ):
                decision_label = "corridor_continue_mapped"
                semantic_action = "MOVE_TO_FOOD"
                spatial_map_state = "food_vector_available"
                reason = "b8_spatial_continue"

        trace_payload.update(
            {
                "b8_controller_profile": profile,
                "b8_spatial_map_state": spatial_map_state,
                "b8_local_affordance_score": round(float(local_affordance_score), 6),
                "b8_return_vector_strength": round(float(return_vector_strength), 6),
                "b8_corridor_dead_end_risk": round(float(dead_end_risk), 6),
                "b8_abort_executed": bool(abort_executed),
                "b8_place_memory": round(float(place_memory), 6),
                "b8_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b8_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b8_genetic_candidate"] = int(params["ga_candidate"])

        self._b8_place_memory = float(place_memory)
        self._b8_last_food_distance = float(food_steps)
        self._b8_last_tick = int(tick)
        return (
            semantic_action,
            B8_SPATIAL_AFFORDANCE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b9_controller_params(self) -> dict[str, float]:
        params = self._b8_controller_params()
        defaults = {
            "b9_route_memory_decay": 0.82,
            "b9_waypoint_commit_ticks": 6.0,
            "b9_route_confidence_threshold": 0.18,
            "b9_path_integrator_gain": 0.50,
            "b9_replan_dead_end_threshold": 0.72,
            "b9_progress_floor": 13.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "waypoint_planner"
        )
        if profile == "path_integration":
            defaults.update({"b9_path_integrator_gain": 0.70, "b9_route_confidence_threshold": 0.12})
        elif profile == "route_memory":
            defaults.update({"b9_route_memory_decay": 0.90, "b9_waypoint_commit_ticks": 8.0})
        elif profile == "waypoint_planner_h56":
            defaults.update({"b9_route_memory_decay": 0.86, "b9_waypoint_commit_ticks": 7.0})
        elif profile == "genetic_waypoint_planner":
            defaults.update({"b9_route_memory_decay": 0.88, "b9_path_integrator_gain": 0.62})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b9_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b9_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b9_route_memory = 0.0
        self._b9_waypoint_lock = 0
        self._b9_last_tick = int(tick)

    def _b9_waypoint_planner_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b8_spatial_affordance_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b9_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "waypoint_planner"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b9_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        local_score = float(trace_payload.get("b8_local_affordance_score", 0.0) or 0.0)
        place_memory = float(trace_payload.get("b8_place_memory", 0.0) or 0.0)
        dead_end_risk = float(trace_payload.get("b8_corridor_dead_end_risk", 0.0) or 0.0)
        b8_decision = str(trace_payload.get("b8_decision", "preserve_b8"))
        route_memory = max(
            float(getattr(self, "_b9_route_memory", 0.0))
            * float(params["b9_route_memory_decay"]),
            place_memory,
            max(0.0, local_score) * float(params["b9_path_integrator_gain"]),
        )
        route_confidence = float(
            np.clip(route_memory + max(0.0, 16.0 - food_steps) / 16.0, 0.0, 1.0)
        )
        replan_signal = float(np.clip(dead_end_risk - route_confidence, 0.0, 1.0))
        waypoint_lock = int(getattr(self, "_b9_waypoint_lock", 0))
        route_state = "non_corridor"
        decision_label = "preserve_b8"

        if corridor_map:
            if (
                b8_decision == "corridor_continue_mapped"
                and route_confidence >= float(params["b9_route_confidence_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                waypoint_lock = max(waypoint_lock, int(params["b9_waypoint_commit_ticks"]))
                route_state = "food_waypoint_locked"
                decision_label = "commit_food_waypoint"
                reason = "b9_commit_food_waypoint"
            elif replan_signal >= float(params["b9_replan_dead_end_threshold"]):
                route_state = "replan_return_vector"
                decision_label = "replan_return"
                reason = "b9_replan_return"
            elif waypoint_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                route_state = "waypoint_lock_continues"
                decision_label = "continue_locked_waypoint"
                reason = "b9_continue_locked_waypoint"

        trace_payload.update(
            {
                "b9_controller_profile": profile,
                "b9_route_state": route_state,
                "b9_route_confidence": round(float(route_confidence), 6),
                "b9_waypoint_lock": int(waypoint_lock),
                "b9_path_integrator": round(float(route_memory), 6),
                "b9_replan_signal": round(float(replan_signal), 6),
                "b9_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b9_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b9_genetic_candidate"] = int(params["ga_candidate"])

        self._b9_route_memory = float(route_memory)
        self._b9_waypoint_lock = max(0, int(waypoint_lock) - 1)
        self._b9_last_tick = int(tick)
        return (
            semantic_action,
            B9_WAYPOINT_PLANNER_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b10_controller_params(self) -> dict[str, float]:
        params = self._b9_controller_params()
        defaults = {
            "b10_replay_memory_decay": 0.84,
            "b10_rollout_gain": 0.55,
            "b10_value_threshold": 0.20,
            "b10_replay_commit_ticks": 5.0,
            "b10_abort_threshold": 0.64,
            "b10_progress_floor": 13.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "prospective_replay"
        )
        if profile == "value_route_evaluator":
            defaults.update({"b10_rollout_gain": 0.68, "b10_value_threshold": 0.16})
        elif profile == "replay_planner":
            defaults.update({"b10_replay_memory_decay": 0.90, "b10_replay_commit_ticks": 7.0})
        elif profile == "prospective_replay_h56":
            defaults.update({"b10_replay_memory_decay": 0.87, "b10_replay_commit_ticks": 6.0})
        elif profile == "genetic_replay_planner":
            defaults.update({"b10_replay_memory_decay": 0.88, "b10_rollout_gain": 0.64})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b10_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b10_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b10_replay_memory = 0.0
        self._b10_plan_commitment = 0
        self._b10_last_tick = int(tick)

    def _b10_prospective_replay_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b9_waypoint_planner_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b10_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "prospective_replay"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b10_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b9_decision = str(trace_payload.get("b9_decision", "preserve_b9"))
        route_confidence = float(trace_payload.get("b9_route_confidence", 0.0) or 0.0)
        replan_signal = float(trace_payload.get("b9_replan_signal", 0.0) or 0.0)
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        dead_end_risk = float(trace_payload.get("b8_corridor_dead_end_risk", 0.0) or 0.0)
        replay_memory = max(
            float(getattr(self, "_b10_replay_memory", 0.0))
            * float(params["b10_replay_memory_decay"]),
            route_confidence * float(params["b10_rollout_gain"]),
        )
        rollout_depth = int(max(1, min(5, round(max(0.0, 16.0 - food_steps) / 4.0) + 1)))
        prospective_value = float(
            np.clip(
                replay_memory
                + route_confidence
                + max(0.0, 16.0 - food_steps) / 20.0
                - dead_end_risk * 0.35,
                0.0,
                1.0,
            )
        )
        abort_signal = float(np.clip(max(dead_end_risk, replan_signal) - prospective_value, 0.0, 1.0))
        plan_commitment = int(getattr(self, "_b10_plan_commitment", 0))
        replay_state = "non_corridor"
        decision_label = "preserve_b9"

        if corridor_map:
            if (
                b9_decision in {"commit_food_waypoint", "continue_locked_waypoint"}
                and prospective_value >= float(params["b10_value_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                plan_commitment = max(plan_commitment, int(params["b10_replay_commit_ticks"]))
                replay_state = "prospective_food_plan"
                decision_label = "commit_replayed_route"
                reason = "b10_commit_replayed_route"
            elif abort_signal >= float(params["b10_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                replay_state = "prospective_abort_return"
                decision_label = "abort_failed_rollout"
                reason = "b10_abort_failed_rollout"
            elif plan_commitment > 0:
                semantic_action = "MOVE_TO_FOOD"
                replay_state = "plan_commitment_continues"
                decision_label = "continue_replay_commitment"
                reason = "b10_continue_replay_commitment"

        trace_payload.update(
            {
                "b10_controller_profile": profile,
                "b10_replay_state": replay_state,
                "b10_prospective_value": round(float(prospective_value), 6),
                "b10_rollout_depth": int(rollout_depth),
                "b10_replay_memory": round(float(replay_memory), 6),
                "b10_plan_commitment": int(plan_commitment),
                "b10_abort_signal": round(float(abort_signal), 6),
                "b10_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b10_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b10_genetic_candidate"] = int(params["ga_candidate"])

        self._b10_replay_memory = float(replay_memory)
        self._b10_plan_commitment = max(0, int(plan_commitment) - 1)
        self._b10_last_tick = int(tick)
        return (
            semantic_action,
            B10_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b11_controller_params(self) -> dict[str, float]:
        params = self._b10_controller_params()
        defaults = {
            "b11_confidence_decay": 0.86,
            "b11_confidence_threshold": 0.24,
            "b11_uncertainty_threshold": 0.70,
            "b11_neuromod_gain": 0.50,
            "b11_confidence_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "confidence_arbiter"
        )
        if profile == "uncertainty_gate":
            defaults.update({"b11_uncertainty_threshold": 0.58, "b11_confidence_threshold": 0.18})
        elif profile == "neuromodulated_replay":
            defaults.update({"b11_neuromod_gain": 0.70, "b11_confidence_decay": 0.90})
        elif profile == "confidence_arbiter_h56":
            defaults.update({"b11_confidence_decay": 0.88, "b11_confidence_commit_ticks": 6.0})
        elif profile == "genetic_confidence_gate":
            defaults.update({"b11_confidence_threshold": 0.20, "b11_neuromod_gain": 0.62})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b11_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b11_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b11_confidence_memory = 0.0
        self._b11_confidence_lock = 0
        self._b11_last_tick = int(tick)

    def _b11_confidence_arbiter_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b10_prospective_replay_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b11_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "confidence_arbiter"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b11_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        prospective_value = float(trace_payload.get("b10_prospective_value", 0.0) or 0.0)
        abort_signal = float(trace_payload.get("b10_abort_signal", 0.0) or 0.0)
        route_confidence = float(trace_payload.get("b9_route_confidence", 0.0) or 0.0)
        b10_decision = str(trace_payload.get("b10_decision", "preserve_b10"))
        confidence_memory = max(
            float(getattr(self, "_b11_confidence_memory", 0.0))
            * float(params["b11_confidence_decay"]),
            prospective_value,
            route_confidence * float(params["b11_neuromod_gain"]),
        )
        uncertainty = float(np.clip(abort_signal + max(0.0, 0.50 - route_confidence), 0.0, 1.0))
        neuromod_signal = float(np.clip(confidence_memory - uncertainty, 0.0, 1.0))
        confidence_lock = int(getattr(self, "_b11_confidence_lock", 0))
        confidence_state = "non_corridor"
        decision_label = "preserve_b10"

        if corridor_map:
            if (
                b10_decision in {"commit_replayed_route", "continue_replay_commitment"}
                and neuromod_signal >= float(params["b11_confidence_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                confidence_lock = max(confidence_lock, int(params["b11_confidence_commit_ticks"]))
                confidence_state = "high_confidence_plan"
                decision_label = "commit_confident_plan"
                reason = "b11_commit_confident_plan"
            elif uncertainty >= float(params["b11_uncertainty_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                confidence_state = "high_uncertainty_return"
                decision_label = "gate_uncertain_plan"
                reason = "b11_gate_uncertain_plan"
            elif confidence_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                confidence_state = "confidence_lock_continues"
                decision_label = "continue_confidence_lock"
                reason = "b11_continue_confidence_lock"

        trace_payload.update(
            {
                "b11_controller_profile": profile,
                "b11_confidence_state": confidence_state,
                "b11_plan_confidence": round(float(confidence_memory), 6),
                "b11_uncertainty": round(float(uncertainty), 6),
                "b11_neuromod_signal": round(float(neuromod_signal), 6),
                "b11_confidence_lock": int(confidence_lock),
                "b11_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b11_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b11_genetic_candidate"] = int(params["ga_candidate"])

        self._b11_confidence_memory = float(confidence_memory)
        self._b11_confidence_lock = max(0, int(confidence_lock) - 1)
        self._b11_last_tick = int(tick)
        return (
            semantic_action,
            B11_CONFIDENCE_ARBITER_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b12_controller_params(self) -> dict[str, float]:
        params = self._b11_controller_params()
        defaults = {
            "b12_attention_decay": 0.84,
            "b12_attention_threshold": 0.18,
            "b12_prediction_error_threshold": 0.66,
            "b12_affordance_gain": 0.45,
            "b12_search_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "predictive_attention"
        )
        if profile == "active_inference_gate":
            defaults.update({"b12_prediction_error_threshold": 0.58, "b12_attention_threshold": 0.14})
        elif profile == "affordance_attention":
            defaults.update({"b12_affordance_gain": 0.66, "b12_attention_decay": 0.88})
        elif profile == "predictive_attention_h56":
            defaults.update({"b12_attention_decay": 0.87, "b12_search_commit_ticks": 6.0})
        elif profile == "genetic_attention_gate":
            defaults.update({"b12_attention_threshold": 0.16, "b12_affordance_gain": 0.58})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b12_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b12_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b12_attention_memory = 0.0
        self._b12_search_lock = 0
        self._b12_last_tick = int(tick)

    def _b12_predictive_attention_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b11_confidence_arbiter_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b12_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "predictive_attention"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b12_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b11_decision = str(trace_payload.get("b11_decision", "preserve_b11"))
        plan_confidence = float(trace_payload.get("b11_plan_confidence", 0.0) or 0.0)
        uncertainty = float(trace_payload.get("b11_uncertainty", 0.0) or 0.0)
        neuromod_signal = float(trace_payload.get("b11_neuromod_signal", 0.0) or 0.0)
        local_affordance = float(trace_payload.get("b8_local_affordance_score", 0.0) or 0.0)
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        expected_progress = float(np.clip(max(0.0, 16.0 - food_steps) / 16.0, 0.0, 1.0))
        prediction_error = float(
            np.clip(abs(plan_confidence - expected_progress) + uncertainty * 0.25, 0.0, 1.0)
        )
        attention_memory = max(
            float(getattr(self, "_b12_attention_memory", 0.0))
            * float(params["b12_attention_decay"]),
            neuromod_signal,
            max(0.0, local_affordance) * float(params["b12_affordance_gain"]),
        )
        attention_gain = float(np.clip(attention_memory + expected_progress * 0.35 - prediction_error * 0.20, 0.0, 1.0))
        search_lock = int(getattr(self, "_b12_search_lock", 0))
        attention_state = "non_corridor"
        decision_label = "preserve_b11"

        if corridor_map:
            if (
                b11_decision in {"commit_confident_plan", "continue_confidence_lock"}
                and attention_gain >= float(params["b12_attention_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                search_lock = max(search_lock, int(params["b12_search_commit_ticks"]))
                attention_state = "attended_food_affordance"
                decision_label = "commit_attended_affordance"
                reason = "b12_commit_attended_affordance"
            elif prediction_error >= float(params["b12_prediction_error_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                attention_state = "prediction_error_return_check"
                decision_label = "gate_prediction_error"
                reason = "b12_gate_prediction_error"
            elif search_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                attention_state = "attention_lock_continues"
                decision_label = "continue_attention_lock"
                reason = "b12_continue_attention_lock"

        trace_payload.update(
            {
                "b12_controller_profile": profile,
                "b12_attention_state": attention_state,
                "b12_prediction_error": round(float(prediction_error), 6),
                "b12_attention_gain": round(float(attention_gain), 6),
                "b12_expected_progress": round(float(expected_progress), 6),
                "b12_search_lock": int(search_lock),
                "b12_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b12_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b12_genetic_candidate"] = int(params["ga_candidate"])

        self._b12_attention_memory = float(attention_memory)
        self._b12_search_lock = max(0, int(search_lock) - 1)
        self._b12_last_tick = int(tick)
        return (
            semantic_action,
            B12_PREDICTIVE_ATTENTION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b13_controller_params(self) -> dict[str, float]:
        params = self._b12_controller_params()
        defaults = {
            "b13_search_memory_decay": 0.86,
            "b13_candidate_gain": 0.50,
            "b13_search_threshold": 0.20,
            "b13_dead_end_threshold": 0.68,
            "b13_local_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "local_affordance_search"
        )
        if profile == "counterfactual_route":
            defaults.update({"b13_candidate_gain": 0.64, "b13_search_threshold": 0.16})
        elif profile == "affordance_sampler":
            defaults.update({"b13_search_memory_decay": 0.90, "b13_local_commit_ticks": 7.0})
        elif profile == "local_affordance_search_h56":
            defaults.update({"b13_search_memory_decay": 0.88, "b13_local_commit_ticks": 6.0})
        elif profile == "genetic_local_search":
            defaults.update({"b13_candidate_gain": 0.58, "b13_search_threshold": 0.18})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b13_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b13_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b13_search_memory = 0.0
        self._b13_search_lock = 0
        self._b13_last_tick = int(tick)

    def _b13_local_affordance_search_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b12_predictive_attention_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b13_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "local_affordance_search"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b13_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b12_decision = str(trace_payload.get("b12_decision", "preserve_b12"))
        attention_gain = float(trace_payload.get("b12_attention_gain", 0.0) or 0.0)
        prediction_error = float(trace_payload.get("b12_prediction_error", 0.0) or 0.0)
        expected_progress = float(trace_payload.get("b12_expected_progress", 0.0) or 0.0)
        local_affordance = float(trace_payload.get("b8_local_affordance_score", 0.0) or 0.0)
        dead_end_risk = float(trace_payload.get("b8_corridor_dead_end_risk", 0.0) or 0.0)
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        food_step_bonus = float(np.clip(max(0.0, 18.0 - food_steps) / 18.0, 0.0, 1.0))
        candidate_score = float(
            np.clip(
                attention_gain * 0.45
                + expected_progress * 0.25
                + max(0.0, local_affordance) * float(params["b13_candidate_gain"])
                + food_step_bonus * 0.20,
                0.0,
                1.0,
            )
        )
        search_memory = max(
            float(getattr(self, "_b13_search_memory", 0.0))
            * float(params["b13_search_memory_decay"]),
            candidate_score,
        )
        dead_end_score = float(
            np.clip(
                prediction_error * 0.55
                + max(0.0, dead_end_risk) * 0.35
                + max(0.0, 1.0 - expected_progress) * 0.10,
                0.0,
                1.0,
            )
        )
        search_lock = int(getattr(self, "_b13_search_lock", 0))
        search_state = "non_corridor"
        decision_label = "preserve_b12"

        if corridor_map:
            if (
                b12_decision in {"commit_attended_affordance", "continue_attention_lock"}
                and search_memory >= float(params["b13_search_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                search_lock = max(search_lock, int(params["b13_local_commit_ticks"]))
                search_state = "local_route_viable"
                decision_label = "commit_local_affordance_search"
                reason = "b13_commit_local_affordance_search"
            elif (
                dead_end_score >= float(params["b13_dead_end_threshold"])
                and b12_decision == "gate_prediction_error"
            ):
                semantic_action = "MOVE_TO_SHELTER"
                search_state = "local_dead_end_return"
                decision_label = "abort_local_dead_end"
                reason = "b13_abort_local_dead_end"
            elif search_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                search_state = "local_search_lock_continues"
                decision_label = "continue_local_search_lock"
                reason = "b13_continue_local_search_lock"

        trace_payload.update(
            {
                "b13_controller_profile": profile,
                "b13_search_state": search_state,
                "b13_local_route_score": round(float(search_memory), 6),
                "b13_affordance_samples": round(float(candidate_score), 6),
                "b13_search_memory": round(float(search_memory), 6),
                "b13_dead_end_score": round(float(dead_end_score), 6),
                "b13_search_lock": int(search_lock),
                "b13_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b13_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b13_genetic_candidate"] = int(params["ga_candidate"])

        self._b13_search_memory = float(search_memory)
        self._b13_search_lock = max(0, int(search_lock) - 1)
        self._b13_last_tick = int(tick)
        return (
            semantic_action,
            B13_LOCAL_SEARCH_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b14_controller_params(self) -> dict[str, float]:
        params = self._b13_controller_params()
        defaults = {
            "b14_uncertainty_decay": 0.82,
            "b14_confidence_threshold": 0.42,
            "b14_uncertainty_threshold": 0.58,
            "b14_risk_gain": 0.40,
            "b14_commit_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "affordance_uncertainty"
        )
        if profile == "risk_calibrated_search":
            defaults.update({"b14_risk_gain": 0.56, "b14_uncertainty_threshold": 0.52})
        elif profile == "confidence_weighted_route":
            defaults.update({"b14_confidence_threshold": 0.34, "b14_commit_ticks": 5.0})
        elif profile == "affordance_uncertainty_h56":
            defaults.update({"b14_uncertainty_decay": 0.86, "b14_commit_ticks": 5.0})
        elif profile == "genetic_uncertainty_search":
            defaults.update({"b14_confidence_threshold": 0.38, "b14_risk_gain": 0.48})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b14_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b14_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b14_uncertainty_memory = 0.0
        self._b14_commitment_lock = 0
        self._b14_last_tick = int(tick)

    def _b14_affordance_uncertainty_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b13_local_affordance_search_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b14_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "affordance_uncertainty"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b14_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b13_decision = str(trace_payload.get("b13_decision", "preserve_b13"))
        route_score = float(trace_payload.get("b13_local_route_score", 0.0) or 0.0)
        affordance_samples = float(trace_payload.get("b13_affordance_samples", 0.0) or 0.0)
        dead_end_score = float(trace_payload.get("b13_dead_end_score", 0.0) or 0.0)
        attention_gain = float(trace_payload.get("b12_attention_gain", 0.0) or 0.0)
        uncertainty_signal = float(
            np.clip(
                abs(route_score - affordance_samples)
                + dead_end_score * float(params["b14_risk_gain"])
                + max(0.0, 0.25 - attention_gain),
                0.0,
                1.0,
            )
        )
        uncertainty = max(
            float(getattr(self, "_b14_uncertainty_memory", 0.0))
            * float(params["b14_uncertainty_decay"]),
            uncertainty_signal,
        )
        confidence = float(np.clip(route_score + affordance_samples * 0.35 - uncertainty * 0.40, 0.0, 1.0))
        risk_adjusted_score = float(np.clip(confidence - dead_end_score * float(params["b14_risk_gain"]), 0.0, 1.0))
        commitment_lock = int(getattr(self, "_b14_commitment_lock", 0))
        uncertainty_state = "non_corridor"
        decision_label = "preserve_b13"

        if corridor_map:
            if (
                b13_decision in {"commit_local_affordance_search", "continue_local_search_lock"}
                and confidence >= float(params["b14_confidence_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                commitment_lock = max(commitment_lock, int(params["b14_commit_ticks"]))
                uncertainty_state = "confidence_calibrated_route"
                decision_label = "commit_confident_affordance"
                reason = "b14_commit_confident_affordance"
            elif (
                uncertainty >= float(params["b14_uncertainty_threshold"])
                and dead_end_score >= 0.35
            ):
                semantic_action = "MOVE_TO_SHELTER"
                uncertainty_state = "uncertain_dead_end_return"
                decision_label = "return_on_affordance_uncertainty"
                reason = "b14_return_on_affordance_uncertainty"
            elif commitment_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                uncertainty_state = "confidence_lock_continues"
                decision_label = "continue_confidence_lock"
                reason = "b14_continue_confidence_lock"

        trace_payload.update(
            {
                "b14_controller_profile": profile,
                "b14_uncertainty_state": uncertainty_state,
                "b14_affordance_confidence": round(float(confidence), 6),
                "b14_uncertainty": round(float(uncertainty), 6),
                "b14_risk_adjusted_score": round(float(risk_adjusted_score), 6),
                "b14_commitment_lock": int(commitment_lock),
                "b14_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b14_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b14_genetic_candidate"] = int(params["ga_candidate"])

        self._b14_uncertainty_memory = float(uncertainty)
        self._b14_commitment_lock = max(0, int(commitment_lock) - 1)
        self._b14_last_tick = int(tick)
        return (
            semantic_action,
            B14_AFFORDANCE_UNCERTAINTY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b15_controller_params(self) -> dict[str, float]:
        params = self._b14_controller_params()
        defaults = {
            "b15_option_memory_decay": 0.84,
            "b15_option_value_threshold": 0.28,
            "b15_termination_threshold": 0.62,
            "b15_persistence_gain": 0.45,
            "b15_option_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "option_critic"
        )
        if profile == "persistence_gate":
            defaults.update({"b15_persistence_gain": 0.62, "b15_option_commit_ticks": 7.0})
        elif profile == "value_gated_option":
            defaults.update({"b15_option_value_threshold": 0.22, "b15_termination_threshold": 0.68})
        elif profile == "option_critic_h56":
            defaults.update({"b15_option_memory_decay": 0.88, "b15_option_commit_ticks": 6.0})
        elif profile == "genetic_option_critic":
            defaults.update({"b15_option_value_threshold": 0.24, "b15_persistence_gain": 0.56})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b15_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b15_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b15_option_memory = 0.0
        self._b15_option_lock = 0
        self._b15_last_tick = int(tick)

    def _b15_option_critic_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b14_affordance_uncertainty_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b15_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "option_critic"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b15_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b14_decision = str(trace_payload.get("b14_decision", "preserve_b14"))
        confidence = float(trace_payload.get("b14_affordance_confidence", 0.0) or 0.0)
        uncertainty = float(trace_payload.get("b14_uncertainty", 0.0) or 0.0)
        risk_adjusted_score = float(trace_payload.get("b14_risk_adjusted_score", 0.0) or 0.0)
        route_confidence = float(trace_payload.get("b9_route_confidence", 0.0) or 0.0)
        prospective_value = float(trace_payload.get("b10_prospective_value", 0.0) or 0.0)
        abort_signal = float(trace_payload.get("b10_abort_signal", 0.0) or 0.0)
        dead_end_score = float(trace_payload.get("b13_dead_end_score", 0.0) or 0.0)
        option_value_raw = float(
            np.clip(
                risk_adjusted_score * 0.50
                + confidence * 0.25
                + max(0.0, route_confidence) * 0.15
                + max(0.0, prospective_value) * 0.10,
                0.0,
                1.0,
            )
        )
        option_memory = max(
            float(getattr(self, "_b15_option_memory", 0.0))
            * float(params["b15_option_memory_decay"]),
            option_value_raw,
        )
        termination_pressure = float(
            np.clip(
                uncertainty * 0.45
                + abort_signal * 0.30
                + dead_end_score * 0.25,
                0.0,
                1.0,
            )
        )
        persistence_score = float(
            np.clip(
                option_memory
                + confidence * float(params["b15_persistence_gain"])
                - termination_pressure * 0.35,
                0.0,
                1.0,
            )
        )
        option_lock = int(getattr(self, "_b15_option_lock", 0))
        option_state = "non_corridor"
        decision_label = "preserve_b14"

        if corridor_map:
            if (
                b14_decision in {"commit_confident_affordance", "continue_confidence_lock"}
                and persistence_score >= float(params["b15_option_value_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                option_lock = max(option_lock, int(params["b15_option_commit_ticks"]))
                option_state = "option_persist_food_route"
                decision_label = "persist_food_option"
                reason = "b15_persist_food_option"
            elif termination_pressure >= float(params["b15_termination_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                option_state = "option_terminated_return"
                decision_label = "terminate_uncertain_option"
                reason = "b15_terminate_uncertain_option"
            elif option_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                option_state = "option_lock_continues"
                decision_label = "continue_option_lock"
                reason = "b15_continue_option_lock"

        trace_payload.update(
            {
                "b15_controller_profile": profile,
                "b15_option_state": option_state,
                "b15_option_value": round(float(option_memory), 6),
                "b15_termination_pressure": round(float(termination_pressure), 6),
                "b15_persistence_score": round(float(persistence_score), 6),
                "b15_option_lock": int(option_lock),
                "b15_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b15_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b15_genetic_candidate"] = int(params["ga_candidate"])

        self._b15_option_memory = float(option_memory)
        self._b15_option_lock = max(0, int(option_lock) - 1)
        self._b15_last_tick = int(tick)
        return (
            semantic_action,
            B15_OPTION_CRITIC_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b16_controller_params(self) -> dict[str, float]:
        params = self._b15_controller_params()
        defaults = {
            "b16_ensemble_decay": 0.82,
            "b16_consensus_threshold": 0.30,
            "b16_conflict_threshold": 0.56,
            "b16_vote_gain": 0.50,
            "b16_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "option_ensemble"
        )
        if profile == "competing_options":
            defaults.update({"b16_conflict_threshold": 0.48, "b16_vote_gain": 0.62})
        elif profile == "action_set_voter":
            defaults.update({"b16_consensus_threshold": 0.24, "b16_commit_ticks": 6.0})
        elif profile == "option_ensemble_h56":
            defaults.update({"b16_ensemble_decay": 0.86, "b16_commit_ticks": 6.0})
        elif profile == "genetic_option_ensemble":
            defaults.update({"b16_consensus_threshold": 0.26, "b16_vote_gain": 0.58})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b16_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b16_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b16_ensemble_memory = 0.0
        self._b16_ensemble_lock = 0
        self._b16_last_tick = int(tick)

    def _b16_option_ensemble_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b15_option_critic_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b16_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "option_ensemble"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b16_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b15_decision = str(trace_payload.get("b15_decision", "preserve_b15"))
        option_value = float(trace_payload.get("b15_option_value", 0.0) or 0.0)
        termination_pressure = float(
            trace_payload.get("b15_termination_pressure", 0.0) or 0.0
        )
        persistence_score = float(
            trace_payload.get("b15_persistence_score", 0.0) or 0.0
        )
        confidence = float(trace_payload.get("b14_affordance_confidence", 0.0) or 0.0)
        uncertainty = float(trace_payload.get("b14_uncertainty", 0.0) or 0.0)
        dead_end_score = float(trace_payload.get("b13_dead_end_score", 0.0) or 0.0)
        vote_gain = float(params["b16_vote_gain"])
        continue_vote = float(
            np.clip(
                persistence_score * 0.45
                + option_value * 0.35
                + confidence * 0.20
                + vote_gain * 0.10,
                0.0,
                1.0,
            )
        )
        return_vote = float(
            np.clip(
                termination_pressure * 0.50
                + uncertainty * 0.30
                + dead_end_score * 0.20,
                0.0,
                1.0,
            )
        )
        consensus_raw = float(
            np.clip(max(continue_vote, return_vote) - min(continue_vote, return_vote) * 0.25, 0.0, 1.0)
        )
        ensemble_memory = max(
            float(getattr(self, "_b16_ensemble_memory", 0.0))
            * float(params["b16_ensemble_decay"]),
            consensus_raw,
        )
        conflict_score = float(np.clip(abs(continue_vote - return_vote), 0.0, 1.0))
        ensemble_lock = int(getattr(self, "_b16_ensemble_lock", 0))
        ensemble_state = "non_corridor"
        decision_label = "preserve_b15"

        if corridor_map:
            if (
                b15_decision in {"persist_food_option", "continue_option_lock"}
                and continue_vote >= float(params["b16_consensus_threshold"])
                and continue_vote >= return_vote
            ):
                semantic_action = "MOVE_TO_FOOD"
                ensemble_lock = max(ensemble_lock, int(params["b16_commit_ticks"]))
                ensemble_state = "ensemble_continue_consensus"
                decision_label = "ensemble_continue_option"
                reason = "b16_ensemble_continue_option"
            elif (
                return_vote >= float(params["b16_conflict_threshold"])
                and return_vote > continue_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                ensemble_state = "ensemble_return_consensus"
                decision_label = "ensemble_return_option"
                reason = "b16_ensemble_return_option"
            elif ensemble_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                ensemble_state = "ensemble_lock_continues"
                decision_label = "continue_ensemble_lock"
                reason = "b16_continue_ensemble_lock"

        trace_payload.update(
            {
                "b16_controller_profile": profile,
                "b16_ensemble_state": ensemble_state,
                "b16_continue_vote": round(float(continue_vote), 6),
                "b16_return_vote": round(float(return_vote), 6),
                "b16_option_votes": round(float(continue_vote - return_vote), 6),
                "b16_consensus_score": round(float(ensemble_memory), 6),
                "b16_conflict_score": round(float(conflict_score), 6),
                "b16_ensemble_lock": int(ensemble_lock),
                "b16_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b16_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b16_genetic_candidate"] = int(params["ga_candidate"])

        self._b16_ensemble_memory = float(ensemble_memory)
        self._b16_ensemble_lock = max(0, int(ensemble_lock) - 1)
        self._b16_last_tick = int(tick)
        return (
            semantic_action,
            B16_OPTION_ENSEMBLE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b17_controller_params(self) -> dict[str, float]:
        params = self._b16_controller_params()
        defaults = {
            "b17_arousal_decay": 0.84,
            "b17_gain_threshold": 0.34,
            "b17_conflict_release": 0.52,
            "b17_homeostasis_gain": 0.44,
            "b17_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "neuromodulated_ensemble"
        )
        if profile == "arousal_gated_options":
            defaults.update({"b17_gain_threshold": 0.28, "b17_commit_ticks": 6.0})
        elif profile == "homeostatic_modulator":
            defaults.update({"b17_homeostasis_gain": 0.58, "b17_gain_threshold": 0.30})
        elif profile == "neuromodulated_ensemble_h56":
            defaults.update({"b17_arousal_decay": 0.88, "b17_commit_ticks": 6.0})
        elif profile == "genetic_neuromodulated_ensemble":
            defaults.update({"b17_gain_threshold": 0.30, "b17_homeostasis_gain": 0.52})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b17_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b17_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b17_arousal_memory = 0.0
        self._b17_modulation_lock = 0
        self._b17_last_tick = int(tick)

    def _b17_neuromodulated_ensemble_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b16_option_ensemble_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b17_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "neuromodulated_ensemble"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b17_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b16_decision = str(trace_payload.get("b16_decision", "preserve_b16"))
        continue_vote = float(trace_payload.get("b16_continue_vote", 0.0) or 0.0)
        return_vote = float(trace_payload.get("b16_return_vote", 0.0) or 0.0)
        consensus = float(trace_payload.get("b16_consensus_score", 0.0) or 0.0)
        conflict = float(trace_payload.get("b16_conflict_score", 0.0) or 0.0)
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        hunger = self._b_series_float(hunger_obs, "hunger")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        health = self._b_series_float(sleep_obs, "health")
        threat = max(
            float(trace_payload.get("b_current_threat_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_temporal_threat_pressure", 0.0) or 0.0),
            float(trace_payload.get("b6_risk_pressure", 0.0) or 0.0),
        )
        homeostatic_gain = float(
            np.clip(
                hunger * 0.40
                + sleep_debt * 0.25
                + max(0.0, 1.0 - health) * 0.35,
                0.0,
                1.0,
            )
        )
        arousal_raw = float(
            np.clip(
                threat * 0.35
                + conflict * 0.25
                + consensus * 0.20
                + homeostatic_gain * float(params["b17_homeostasis_gain"]),
                0.0,
                1.0,
            )
        )
        arousal_signal = max(
            float(getattr(self, "_b17_arousal_memory", 0.0))
            * float(params["b17_arousal_decay"]),
            arousal_raw,
        )
        option_gain = float(
            np.clip(continue_vote + arousal_signal * 0.25 - return_vote * 0.20, 0.0, 1.0)
        )
        conflict_release = float(
            np.clip(return_vote + conflict * 0.35 + threat * 0.20, 0.0, 1.0)
        )
        modulation_lock = int(getattr(self, "_b17_modulation_lock", 0))
        modulator_state = "non_corridor"
        decision_label = "preserve_b16"

        if corridor_map:
            if (
                b16_decision in {"ensemble_continue_option", "continue_ensemble_lock"}
                and option_gain >= float(params["b17_gain_threshold"])
                and option_gain >= conflict_release
            ):
                semantic_action = "MOVE_TO_FOOD"
                modulation_lock = max(modulation_lock, int(params["b17_commit_ticks"]))
                modulator_state = "modulated_continue"
                decision_label = "neuromodulated_continue"
                reason = "b17_neuromodulated_continue"
            elif (
                conflict_release >= float(params["b17_conflict_release"])
                and conflict_release > option_gain
            ):
                semantic_action = "MOVE_TO_SHELTER"
                modulator_state = "modulated_return"
                decision_label = "neuromodulated_return"
                reason = "b17_neuromodulated_return"
            elif modulation_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                modulator_state = "modulation_lock_continues"
                decision_label = "continue_modulation_lock"
                reason = "b17_continue_modulation_lock"

        trace_payload.update(
            {
                "b17_controller_profile": profile,
                "b17_modulator_state": modulator_state,
                "b17_arousal_signal": round(float(arousal_signal), 6),
                "b17_homeostatic_gain": round(float(homeostatic_gain), 6),
                "b17_option_gain": round(float(option_gain), 6),
                "b17_conflict_release": round(float(conflict_release), 6),
                "b17_modulation_lock": int(modulation_lock),
                "b17_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b17_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b17_genetic_candidate"] = int(params["ga_candidate"])

        self._b17_arousal_memory = float(arousal_signal)
        self._b17_modulation_lock = max(0, int(modulation_lock) - 1)
        self._b17_last_tick = int(tick)
        return (
            semantic_action,
            B17_NEUROMODULATED_ENSEMBLE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b18_controller_params(self) -> dict[str, float]:
        params = self._b17_controller_params()
        defaults = {
            "b18_trace_decay": 0.86,
            "b18_stability_threshold": 0.30,
            "b18_switch_threshold": 0.58,
            "b18_prediction_gain": 0.42,
            "b18_trace_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "eligibility_trace"
        )
        if profile == "metastable_arousal":
            defaults.update({"b18_stability_threshold": 0.25, "b18_trace_commit_ticks": 6.0})
        elif profile == "synaptic_trace_modulator":
            defaults.update({"b18_prediction_gain": 0.56, "b18_switch_threshold": 0.62})
        elif profile == "eligibility_trace_h56":
            defaults.update({"b18_trace_decay": 0.88, "b18_trace_commit_ticks": 6.0})
        elif profile == "genetic_eligibility_trace":
            defaults.update({"b18_stability_threshold": 0.27, "b18_prediction_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b18_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b18_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b18_eligibility_trace = 0.0
        self._b18_trace_lock = 0
        self._b18_last_tick = int(tick)

    def _b18_eligibility_trace_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b17_neuromodulated_ensemble_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b18_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "eligibility_trace"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b18_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b17_decision = str(trace_payload.get("b17_decision", "preserve_b17"))
        arousal_signal = float(trace_payload.get("b17_arousal_signal", 0.0) or 0.0)
        homeostatic_gain = float(trace_payload.get("b17_homeostatic_gain", 0.0) or 0.0)
        option_gain = float(trace_payload.get("b17_option_gain", 0.0) or 0.0)
        conflict_release = float(trace_payload.get("b17_conflict_release", 0.0) or 0.0)
        consensus = float(trace_payload.get("b16_consensus_score", 0.0) or 0.0)
        option_value = float(trace_payload.get("b15_option_value", 0.0) or 0.0)
        prediction_proxy = float(
            np.clip(
                option_gain * 0.35
                + consensus * 0.25
                + option_value * 0.20
                + arousal_signal * 0.20,
                0.0,
                1.0,
            )
        )
        eligibility_raw = float(
            np.clip(
                prediction_proxy * float(params["b18_prediction_gain"])
                + max(0.0, option_gain - conflict_release) * 0.35
                + homeostatic_gain * 0.20,
                0.0,
                1.0,
            )
        )
        eligibility_trace = max(
            float(getattr(self, "_b18_eligibility_trace", 0.0))
            * float(params["b18_trace_decay"]),
            eligibility_raw,
        )
        stability_bias = float(
            np.clip(eligibility_trace + option_gain * 0.30 - conflict_release * 0.20, 0.0, 1.0)
        )
        switch_pressure = float(
            np.clip(conflict_release + max(0.0, conflict_release - option_gain) * 0.35, 0.0, 1.0)
        )
        trace_lock = int(getattr(self, "_b18_trace_lock", 0))
        trace_state = "non_corridor"
        decision_label = "preserve_b17"

        if corridor_map:
            if (
                b17_decision in {"neuromodulated_continue", "continue_modulation_lock"}
                and stability_bias >= float(params["b18_stability_threshold"])
                and stability_bias >= switch_pressure
            ):
                semantic_action = "MOVE_TO_FOOD"
                trace_lock = max(trace_lock, int(params["b18_trace_commit_ticks"]))
                trace_state = "trace_stabilizes_option"
                decision_label = "eligibility_stabilize_option"
                reason = "b18_eligibility_stabilize_option"
            elif (
                switch_pressure >= float(params["b18_switch_threshold"])
                and switch_pressure > stability_bias
            ):
                semantic_action = "MOVE_TO_SHELTER"
                trace_state = "trace_releases_option"
                decision_label = "eligibility_release_option"
                reason = "b18_eligibility_release_option"
            elif trace_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                trace_state = "trace_lock_continues"
                decision_label = "continue_trace_lock"
                reason = "b18_continue_trace_lock"

        trace_payload.update(
            {
                "b18_controller_profile": profile,
                "b18_trace_state": trace_state,
                "b18_eligibility_trace": round(float(eligibility_trace), 6),
                "b18_reward_prediction_proxy": round(float(prediction_proxy), 6),
                "b18_stability_bias": round(float(stability_bias), 6),
                "b18_switch_pressure": round(float(switch_pressure), 6),
                "b18_trace_lock": int(trace_lock),
                "b18_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b18_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b18_genetic_candidate"] = int(params["ga_candidate"])

        self._b18_eligibility_trace = float(eligibility_trace)
        self._b18_trace_lock = max(0, int(trace_lock) - 1)
        self._b18_last_tick = int(tick)
        return (
            semantic_action,
            B18_ELIGIBILITY_TRACE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b19_controller_params(self) -> dict[str, float]:
        params = self._b18_controller_params()
        defaults = {
            "b19_memory_decay": 0.88,
            "b19_consolidation_threshold": 0.30,
            "b19_switch_suppression_threshold": 0.58,
            "b19_stability_gain": 0.42,
            "b19_memory_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "episodic_meta_memory"
        )
        if profile == "stability_memory":
            defaults.update({"b19_consolidation_threshold": 0.25, "b19_memory_commit_ticks": 6.0})
        elif profile == "switch_suppression":
            defaults.update(
                {"b19_switch_suppression_threshold": 0.62, "b19_stability_gain": 0.56}
            )
        elif profile == "episodic_meta_memory_h56":
            defaults.update({"b19_memory_decay": 0.90, "b19_memory_commit_ticks": 6.0})
        elif profile == "genetic_meta_memory":
            defaults.update({"b19_consolidation_threshold": 0.27, "b19_stability_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b19_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b19_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b19_episode_memory = 0.0
        self._b19_memory_lock = 0
        self._b19_last_tick = int(tick)

    def _b19_episodic_meta_memory_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b18_eligibility_trace_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b19_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "episodic_meta_memory"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b19_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b18_decision = str(trace_payload.get("b18_decision", "preserve_b18"))
        eligibility_trace = float(trace_payload.get("b18_eligibility_trace", 0.0) or 0.0)
        prediction_proxy = float(trace_payload.get("b18_reward_prediction_proxy", 0.0) or 0.0)
        stability_bias = float(trace_payload.get("b18_stability_bias", 0.0) or 0.0)
        switch_pressure = float(trace_payload.get("b18_switch_pressure", 0.0) or 0.0)
        memory_input = float(
            np.clip(
                eligibility_trace * 0.35
                + prediction_proxy * 0.30
                + stability_bias * 0.25
                + max(0.0, stability_bias - switch_pressure) * 0.10,
                0.0,
                1.0,
            )
        )
        episode_memory = max(
            float(getattr(self, "_b19_episode_memory", 0.0))
            * float(params["b19_memory_decay"]),
            memory_input,
        )
        consolidation_score = float(
            np.clip(
                episode_memory * float(params["b19_stability_gain"])
                + stability_bias * 0.35,
                0.0,
                1.0,
            )
        )
        stability_vote = float(
            np.clip(
                consolidation_score + eligibility_trace * 0.20 - switch_pressure * 0.20,
                0.0,
                1.0,
            )
        )
        switch_suppression = float(
            np.clip(switch_pressure - episode_memory * 0.25, 0.0, 1.0)
        )
        memory_lock = int(getattr(self, "_b19_memory_lock", 0))
        memory_state = "non_corridor"
        decision_label = "preserve_b18"

        if corridor_map:
            if (
                b18_decision in {"eligibility_stabilize_option", "continue_trace_lock"}
                and stability_vote >= float(params["b19_consolidation_threshold"])
                and stability_vote >= switch_suppression
            ):
                semantic_action = "MOVE_TO_FOOD"
                memory_lock = max(memory_lock, int(params["b19_memory_commit_ticks"]))
                memory_state = "memory_consolidates_option"
                decision_label = "episodic_consolidate_option"
                reason = "b19_episodic_consolidate_option"
            elif (
                switch_suppression >= float(params["b19_switch_suppression_threshold"])
                and switch_suppression > stability_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                memory_state = "memory_releases_option"
                decision_label = "episodic_release_option"
                reason = "b19_episodic_release_option"
            elif memory_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                memory_state = "memory_lock_continues"
                decision_label = "continue_memory_lock"
                reason = "b19_continue_memory_lock"

        trace_payload.update(
            {
                "b19_controller_profile": profile,
                "b19_memory_state": memory_state,
                "b19_episode_memory": round(float(episode_memory), 6),
                "b19_consolidation_score": round(float(consolidation_score), 6),
                "b19_stability_vote": round(float(stability_vote), 6),
                "b19_switch_suppression": round(float(switch_suppression), 6),
                "b19_memory_lock": int(memory_lock),
                "b19_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b19_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b19_genetic_candidate"] = int(params["ga_candidate"])

        self._b19_episode_memory = float(episode_memory)
        self._b19_memory_lock = max(0, int(memory_lock) - 1)
        self._b19_last_tick = int(tick)
        return (
            semantic_action,
            B19_EPISODIC_META_MEMORY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b20_controller_params(self) -> dict[str, float]:
        params = self._b19_controller_params()
        defaults = {
            "b20_buffer_decay": 0.86,
            "b20_gate_threshold": 0.30,
            "b20_release_threshold": 0.58,
            "b20_context_gain": 0.44,
            "b20_buffer_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "working_memory_gate"
        )
        if profile == "context_binding":
            defaults.update({"b20_gate_threshold": 0.25, "b20_buffer_commit_ticks": 6.0})
        elif profile == "stability_buffer":
            defaults.update({"b20_context_gain": 0.56, "b20_release_threshold": 0.62})
        elif profile == "working_memory_gate_h56":
            defaults.update({"b20_buffer_decay": 0.88, "b20_buffer_commit_ticks": 6.0})
        elif profile == "genetic_working_memory":
            defaults.update({"b20_gate_threshold": 0.27, "b20_context_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b20_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b20_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b20_working_buffer = 0.0
        self._b20_buffer_lock = 0
        self._b20_last_tick = int(tick)

    def _b20_working_memory_gate_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b19_episodic_meta_memory_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b20_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "working_memory_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b20_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b19_decision = str(trace_payload.get("b19_decision", "preserve_b19"))
        episode_memory = float(trace_payload.get("b19_episode_memory", 0.0) or 0.0)
        consolidation_score = float(
            trace_payload.get("b19_consolidation_score", 0.0) or 0.0
        )
        stability_vote = float(trace_payload.get("b19_stability_vote", 0.0) or 0.0)
        switch_suppression = float(
            trace_payload.get("b19_switch_suppression", 0.0) or 0.0
        )
        buffer_input = float(
            np.clip(
                episode_memory * 0.35
                + consolidation_score * 0.25
                + stability_vote * 0.25
                + max(0.0, stability_vote - switch_suppression) * 0.15,
                0.0,
                1.0,
            )
        )
        working_buffer = max(
            float(getattr(self, "_b20_working_buffer", 0.0))
            * float(params["b20_buffer_decay"]),
            buffer_input,
        )
        context_binding = float(
            np.clip(
                working_buffer * float(params["b20_context_gain"])
                + stability_vote * 0.35,
                0.0,
                1.0,
            )
        )
        gate_vote = float(
            np.clip(
                context_binding + episode_memory * 0.15 - switch_suppression * 0.15,
                0.0,
                1.0,
            )
        )
        release_vote = float(
            np.clip(switch_suppression - working_buffer * 0.20, 0.0, 1.0)
        )
        buffer_lock = int(getattr(self, "_b20_buffer_lock", 0))
        buffer_state = "non_corridor"
        decision_label = "preserve_b19"

        if corridor_map:
            if (
                b19_decision in {"episodic_consolidate_option", "continue_memory_lock"}
                and gate_vote >= float(params["b20_gate_threshold"])
                and gate_vote >= release_vote
            ):
                semantic_action = "MOVE_TO_FOOD"
                buffer_lock = max(buffer_lock, int(params["b20_buffer_commit_ticks"]))
                buffer_state = "working_memory_holds_context"
                decision_label = "working_memory_gate_continue"
                reason = "b20_working_memory_gate_continue"
            elif (
                release_vote >= float(params["b20_release_threshold"])
                and release_vote > gate_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                buffer_state = "working_memory_releases_context"
                decision_label = "working_memory_gate_release"
                reason = "b20_working_memory_gate_release"
            elif buffer_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                buffer_state = "working_memory_lock_continues"
                decision_label = "continue_working_memory_lock"
                reason = "b20_continue_working_memory_lock"

        trace_payload.update(
            {
                "b20_controller_profile": profile,
                "b20_buffer_state": buffer_state,
                "b20_working_buffer": round(float(working_buffer), 6),
                "b20_context_binding": round(float(context_binding), 6),
                "b20_gate_vote": round(float(gate_vote), 6),
                "b20_release_vote": round(float(release_vote), 6),
                "b20_buffer_lock": int(buffer_lock),
                "b20_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b20_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b20_genetic_candidate"] = int(params["ga_candidate"])

        self._b20_working_buffer = float(working_buffer)
        self._b20_buffer_lock = max(0, int(buffer_lock) - 1)
        self._b20_last_tick = int(tick)
        return (
            semantic_action,
            B20_WORKING_MEMORY_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b21_controller_params(self) -> dict[str, float]:
        params = self._b20_controller_params()
        defaults = {
            "b21_replay_decay": 0.84,
            "b21_replay_threshold": 0.30,
            "b21_abort_threshold": 0.58,
            "b21_sequence_gain": 0.46,
            "b21_replay_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "hippocampal_replay"
        )
        if profile == "sequence_binding":
            defaults.update({"b21_replay_threshold": 0.25, "b21_replay_commit_ticks": 6.0})
        elif profile == "route_rehearsal":
            defaults.update({"b21_sequence_gain": 0.58, "b21_abort_threshold": 0.62})
        elif profile == "hippocampal_replay_h56":
            defaults.update({"b21_replay_decay": 0.86, "b21_replay_commit_ticks": 6.0})
        elif profile == "genetic_replay_gate":
            defaults.update({"b21_replay_threshold": 0.27, "b21_sequence_gain": 0.52})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b21_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b21_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b21_sequence_memory = 0.0
        self._b21_replay_lock = 0
        self._b21_last_tick = int(tick)

    def _b21_hippocampal_replay_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b20_working_memory_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b21_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "hippocampal_replay"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b21_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b20_decision = str(trace_payload.get("b20_decision", "preserve_b20"))
        working_buffer = float(trace_payload.get("b20_working_buffer", 0.0) or 0.0)
        context_binding = float(trace_payload.get("b20_context_binding", 0.0) or 0.0)
        gate_vote = float(trace_payload.get("b20_gate_vote", 0.0) or 0.0)
        release_vote = float(trace_payload.get("b20_release_vote", 0.0) or 0.0)
        sequence_input = float(
            np.clip(
                working_buffer * 0.30
                + context_binding * 0.30
                + gate_vote * 0.25
                + max(0.0, gate_vote - release_vote) * 0.15,
                0.0,
                1.0,
            )
        )
        sequence_memory = max(
            float(getattr(self, "_b21_sequence_memory", 0.0))
            * float(params["b21_replay_decay"]),
            sequence_input,
        )
        replay_score = float(
            np.clip(
                sequence_memory * float(params["b21_sequence_gain"])
                + context_binding * 0.35,
                0.0,
                1.0,
            )
        )
        route_commitment = float(
            np.clip(replay_score + working_buffer * 0.20 - release_vote * 0.15, 0.0, 1.0)
        )
        abort_prediction = float(
            np.clip(release_vote - sequence_memory * 0.20, 0.0, 1.0)
        )
        replay_lock = int(getattr(self, "_b21_replay_lock", 0))
        replay_state = "non_corridor"
        decision_label = "preserve_b20"

        if corridor_map:
            if (
                b20_decision in {"working_memory_gate_continue", "continue_working_memory_lock"}
                and route_commitment >= float(params["b21_replay_threshold"])
                and route_commitment >= abort_prediction
            ):
                semantic_action = "MOVE_TO_FOOD"
                replay_lock = max(replay_lock, int(params["b21_replay_commit_ticks"]))
                replay_state = "replay_rehearses_route"
                decision_label = "hippocampal_replay_continue"
                reason = "b21_hippocampal_replay_continue"
            elif (
                abort_prediction >= float(params["b21_abort_threshold"])
                and abort_prediction > route_commitment
            ):
                semantic_action = "MOVE_TO_SHELTER"
                replay_state = "replay_predicts_abort"
                decision_label = "hippocampal_replay_abort"
                reason = "b21_hippocampal_replay_abort"
            elif replay_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                replay_state = "replay_lock_continues"
                decision_label = "continue_replay_lock"
                reason = "b21_continue_replay_lock"

        trace_payload.update(
            {
                "b21_controller_profile": profile,
                "b21_replay_state": replay_state,
                "b21_sequence_memory": round(float(sequence_memory), 6),
                "b21_replay_score": round(float(replay_score), 6),
                "b21_route_commitment": round(float(route_commitment), 6),
                "b21_abort_prediction": round(float(abort_prediction), 6),
                "b21_replay_lock": int(replay_lock),
                "b21_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b21_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b21_genetic_candidate"] = int(params["ga_candidate"])

        self._b21_sequence_memory = float(sequence_memory)
        self._b21_replay_lock = max(0, int(replay_lock) - 1)
        self._b21_last_tick = int(tick)
        return (
            semantic_action,
            B21_HIPPOCAMPAL_REPLAY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b22_controller_params(self) -> dict[str, float]:
        params = self._b21_controller_params()
        defaults = {
            "b22_sim_decay": 0.84,
            "b22_viability_threshold": 0.30,
            "b22_abort_threshold": 0.58,
            "b22_forward_gain": 0.48,
            "b22_sim_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "prospective_map_replay"
        )
        if profile == "forward_model_gate":
            defaults.update(
                {"b22_viability_threshold": 0.25, "b22_sim_commit_ticks": 6.0}
            )
        elif profile == "route_viability_sim":
            defaults.update({"b22_forward_gain": 0.60, "b22_abort_threshold": 0.62})
        elif profile == "prospective_map_replay_h56":
            defaults.update({"b22_sim_decay": 0.86, "b22_sim_commit_ticks": 6.0})
        elif profile == "genetic_prospective_replay":
            defaults.update({"b22_viability_threshold": 0.27, "b22_forward_gain": 0.54})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b22_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b22_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b22_prospective_sim = 0.0
        self._b22_sim_lock = 0
        self._b22_last_tick = int(tick)

    def _b22_prospective_replay_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b21_hippocampal_replay_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b22_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "prospective_map_replay"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b22_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b21_decision = str(trace_payload.get("b21_decision", "preserve_b21"))
        sequence_memory = float(trace_payload.get("b21_sequence_memory", 0.0) or 0.0)
        replay_score = float(trace_payload.get("b21_replay_score", 0.0) or 0.0)
        route_commitment = float(
            trace_payload.get("b21_route_commitment", 0.0) or 0.0
        )
        abort_prediction = float(
            trace_payload.get("b21_abort_prediction", 0.0) or 0.0
        )
        sim_input = float(
            np.clip(
                sequence_memory * 0.30
                + replay_score * 0.30
                + route_commitment * 0.25
                + max(0.0, route_commitment - abort_prediction) * 0.15,
                0.0,
                1.0,
            )
        )
        prospective_sim = max(
            float(getattr(self, "_b22_prospective_sim", 0.0))
            * float(params["b22_sim_decay"]),
            sim_input,
        )
        forward_model_score = float(
            np.clip(
                prospective_sim * float(params["b22_forward_gain"])
                + route_commitment * 0.35,
                0.0,
                1.0,
            )
        )
        viability_projection = float(
            np.clip(
                forward_model_score + sequence_memory * 0.20 - abort_prediction * 0.15,
                0.0,
                1.0,
            )
        )
        abort_projection = float(
            np.clip(abort_prediction - prospective_sim * 0.20, 0.0, 1.0)
        )
        sim_lock = int(getattr(self, "_b22_sim_lock", 0))
        sim_state = "non_corridor"
        decision_label = "preserve_b21"

        if corridor_map:
            if (
                b21_decision
                in {"hippocampal_replay_continue", "continue_replay_lock"}
                and viability_projection >= float(params["b22_viability_threshold"])
                and viability_projection >= abort_projection
            ):
                semantic_action = "MOVE_TO_FOOD"
                sim_lock = max(sim_lock, int(params["b22_sim_commit_ticks"]))
                sim_state = "prospective_sim_commits_route"
                decision_label = "prospective_replay_continue"
                reason = "b22_prospective_replay_continue"
            elif (
                abort_projection >= float(params["b22_abort_threshold"])
                and abort_projection > viability_projection
            ):
                semantic_action = "MOVE_TO_SHELTER"
                sim_state = "prospective_sim_aborts_route"
                decision_label = "prospective_replay_abort"
                reason = "b22_prospective_replay_abort"
            elif sim_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                sim_state = "prospective_lock_continues"
                decision_label = "continue_prospective_lock"
                reason = "b22_continue_prospective_lock"

        trace_payload.update(
            {
                "b22_controller_profile": profile,
                "b22_sim_state": sim_state,
                "b22_prospective_sim": round(float(prospective_sim), 6),
                "b22_forward_model_score": round(float(forward_model_score), 6),
                "b22_viability_projection": round(float(viability_projection), 6),
                "b22_abort_projection": round(float(abort_projection), 6),
                "b22_sim_lock": int(sim_lock),
                "b22_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b22_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b22_genetic_candidate"] = int(params["ga_candidate"])

        self._b22_prospective_sim = float(prospective_sim)
        self._b22_sim_lock = max(0, int(sim_lock) - 1)
        self._b22_last_tick = int(tick)
        return (
            semantic_action,
            B22_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b23_controller_params(self) -> dict[str, float]:
        params = self._b22_controller_params()
        defaults = {
            "b23_conflict_decay": 0.86,
            "b23_conflict_threshold": 0.24,
            "b23_abort_bias_threshold": 0.62,
            "b23_error_gain": 0.46,
            "b23_monitor_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "conflict_monitor"
        )
        if profile == "error_gated_replay":
            defaults.update(
                {"b23_conflict_threshold": 0.20, "b23_monitor_commit_ticks": 6.0}
            )
        elif profile == "abort_conflict_arbiter":
            defaults.update({"b23_abort_bias_threshold": 0.58, "b23_error_gain": 0.54})
        elif profile == "conflict_monitor_h56":
            defaults.update({"b23_conflict_decay": 0.88, "b23_monitor_commit_ticks": 6.0})
        elif profile == "genetic_conflict_monitor":
            defaults.update({"b23_conflict_threshold": 0.22, "b23_error_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b23_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b23_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b23_conflict_memory = 0.0
        self._b23_monitor_lock = 0
        self._b23_last_tick = int(tick)

    def _b23_conflict_monitor_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b22_prospective_replay_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b23_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "conflict_monitor"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b23_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b22_decision = str(trace_payload.get("b22_decision", "preserve_b22"))
        prospective_sim = float(trace_payload.get("b22_prospective_sim", 0.0) or 0.0)
        forward_model_score = float(
            trace_payload.get("b22_forward_model_score", 0.0) or 0.0
        )
        viability_projection = float(
            trace_payload.get("b22_viability_projection", 0.0) or 0.0
        )
        abort_projection = float(trace_payload.get("b22_abort_projection", 0.0) or 0.0)
        prediction_error = float(
            np.clip(
                abs(viability_projection - forward_model_score)
                + abort_projection * float(params["b23_error_gain"]),
                0.0,
                1.0,
            )
        )
        conflict_input = float(
            np.clip(
                prediction_error
                + max(0.0, abort_projection - viability_projection) * 0.35,
                0.0,
                1.0,
            )
        )
        conflict_memory = max(
            float(getattr(self, "_b23_conflict_memory", 0.0))
            * float(params["b23_conflict_decay"]),
            conflict_input,
        )
        stability_vote = float(
            np.clip(
                viability_projection
                + prospective_sim * 0.20
                - conflict_memory * 0.25,
                0.0,
                1.0,
            )
        )
        abort_bias = float(
            np.clip(
                abort_projection + conflict_memory * 0.25 - stability_vote * 0.15,
                0.0,
                1.0,
            )
        )
        monitor_lock = int(getattr(self, "_b23_monitor_lock", 0))
        conflict_state = "non_corridor"
        decision_label = "preserve_b22"

        if corridor_map:
            if (
                b22_decision
                in {"prospective_replay_continue", "continue_prospective_lock"}
                and stability_vote >= float(params["b23_conflict_threshold"])
                and abort_bias < float(params["b23_abort_bias_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                monitor_lock = max(monitor_lock, int(params["b23_monitor_commit_ticks"]))
                conflict_state = "conflict_monitor_stabilizes_route"
                decision_label = "conflict_monitor_continue"
                reason = "b23_conflict_monitor_continue"
            elif (
                abort_bias >= float(params["b23_abort_bias_threshold"])
                and abort_bias > stability_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                conflict_state = "conflict_monitor_predicts_abort"
                decision_label = "conflict_monitor_abort"
                reason = "b23_conflict_monitor_abort"
            elif monitor_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                conflict_state = "conflict_lock_continues"
                decision_label = "continue_conflict_lock"
                reason = "b23_continue_conflict_lock"

        trace_payload.update(
            {
                "b23_controller_profile": profile,
                "b23_conflict_state": conflict_state,
                "b23_prediction_error": round(float(prediction_error), 6),
                "b23_conflict_memory": round(float(conflict_memory), 6),
                "b23_stability_vote": round(float(stability_vote), 6),
                "b23_abort_bias": round(float(abort_bias), 6),
                "b23_monitor_lock": int(monitor_lock),
                "b23_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b23_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b23_genetic_candidate"] = int(params["ga_candidate"])

        self._b23_conflict_memory = float(conflict_memory)
        self._b23_monitor_lock = max(0, int(monitor_lock) - 1)
        self._b23_last_tick = int(tick)
        return (
            semantic_action,
            B23_CONFLICT_MONITOR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b24_controller_params(self) -> dict[str, float]:
        params = self._b23_controller_params()
        defaults = {
            "b24_precision_decay": 0.88,
            "b24_precision_threshold": 0.26,
            "b24_uncertainty_threshold": 0.64,
            "b24_precision_gain": 0.45,
            "b24_precision_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "precision_conflict"
        )
        if profile == "prediction_precision_gate":
            defaults.update(
                {"b24_precision_threshold": 0.22, "b24_precision_commit_ticks": 6.0}
            )
        elif profile == "reliability_abort":
            defaults.update({"b24_uncertainty_threshold": 0.60, "b24_precision_gain": 0.52})
        elif profile == "precision_conflict_h56":
            defaults.update({"b24_precision_decay": 0.90, "b24_precision_commit_ticks": 6.0})
        elif profile == "genetic_precision_conflict":
            defaults.update({"b24_precision_threshold": 0.24, "b24_precision_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b24_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b24_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b24_precision_memory = 0.0
        self._b24_precision_lock = 0
        self._b24_last_tick = int(tick)

    def _b24_precision_conflict_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b23_conflict_monitor_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b24_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "precision_conflict"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b24_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b23_decision = str(trace_payload.get("b23_decision", "preserve_b23"))
        prediction_error = float(trace_payload.get("b23_prediction_error", 0.0) or 0.0)
        conflict_memory = float(trace_payload.get("b23_conflict_memory", 0.0) or 0.0)
        stability_vote = float(trace_payload.get("b23_stability_vote", 0.0) or 0.0)
        abort_bias = float(trace_payload.get("b23_abort_bias", 0.0) or 0.0)
        reliability_input = float(
            np.clip(
                stability_vote * max(0.0, 1.0 - prediction_error)
                + max(0.0, stability_vote - abort_bias) * 0.25,
                0.0,
                1.0,
            )
        )
        precision_memory = max(
            float(getattr(self, "_b24_precision_memory", 0.0))
            * float(params["b24_precision_decay"]),
            reliability_input,
        )
        uncertainty_pressure = float(
            np.clip(
                prediction_error
                + conflict_memory * 0.35
                + abort_bias * 0.25
                - precision_memory * 0.25,
                0.0,
                1.0,
            )
        )
        precision_vote = float(
            np.clip(
                precision_memory * float(params["b24_precision_gain"])
                + stability_vote * 0.35
                - uncertainty_pressure * 0.15,
                0.0,
                1.0,
            )
        )
        abort_precision = float(
            np.clip(abort_bias + uncertainty_pressure * 0.20 - precision_vote * 0.15, 0.0, 1.0)
        )
        precision_lock = int(getattr(self, "_b24_precision_lock", 0))
        precision_state = "non_corridor"
        decision_label = "preserve_b23"

        if corridor_map:
            if (
                b23_decision in {"conflict_monitor_continue", "continue_conflict_lock"}
                and precision_vote >= float(params["b24_precision_threshold"])
                and abort_precision < float(params["b24_uncertainty_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                precision_lock = max(
                    precision_lock, int(params["b24_precision_commit_ticks"])
                )
                precision_state = "precision_conflict_stabilizes_route"
                decision_label = "precision_conflict_continue"
                reason = "b24_precision_conflict_continue"
            elif (
                abort_precision >= float(params["b24_uncertainty_threshold"])
                and abort_precision > precision_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                precision_state = "precision_conflict_predicts_abort"
                decision_label = "precision_conflict_abort"
                reason = "b24_precision_conflict_abort"
            elif precision_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                precision_state = "precision_lock_continues"
                decision_label = "continue_precision_lock"
                reason = "b24_continue_precision_lock"

        trace_payload.update(
            {
                "b24_controller_profile": profile,
                "b24_precision_state": precision_state,
                "b24_precision_memory": round(float(precision_memory), 6),
                "b24_precision_vote": round(float(precision_vote), 6),
                "b24_uncertainty_pressure": round(float(uncertainty_pressure), 6),
                "b24_abort_precision": round(float(abort_precision), 6),
                "b24_precision_lock": int(precision_lock),
                "b24_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b24_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b24_genetic_candidate"] = int(params["ga_candidate"])

        self._b24_precision_memory = float(precision_memory)
        self._b24_precision_lock = max(0, int(precision_lock) - 1)
        self._b24_last_tick = int(tick)
        return (
            semantic_action,
            B24_PRECISION_CONFLICT_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b25_controller_params(self) -> dict[str, float]:
        params = self._b24_controller_params()
        defaults = {
            "b25_confidence_decay": 0.90,
            "b25_confidence_threshold": 0.28,
            "b25_doubt_threshold": 0.66,
            "b25_control_gain": 0.44,
            "b25_meta_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "metacognitive_confidence"
        )
        if profile == "confidence_calibration":
            defaults.update(
                {"b25_confidence_threshold": 0.24, "b25_meta_commit_ticks": 6.0}
            )
        elif profile == "uncertainty_integrator":
            defaults.update({"b25_doubt_threshold": 0.62, "b25_control_gain": 0.52})
        elif profile == "metacognitive_confidence_h56":
            defaults.update({"b25_confidence_decay": 0.92, "b25_meta_commit_ticks": 6.0})
        elif profile == "genetic_metacognition":
            defaults.update({"b25_confidence_threshold": 0.26, "b25_control_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b25_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b25_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b25_confidence_memory = 0.0
        self._b25_meta_lock = 0
        self._b25_last_tick = int(tick)

    def _b25_metacognitive_confidence_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b24_precision_conflict_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b25_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "metacognitive_confidence"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b25_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b24_decision = str(trace_payload.get("b24_decision", "preserve_b24"))
        precision_memory = float(trace_payload.get("b24_precision_memory", 0.0) or 0.0)
        precision_vote = float(trace_payload.get("b24_precision_vote", 0.0) or 0.0)
        uncertainty_pressure = float(
            trace_payload.get("b24_uncertainty_pressure", 0.0) or 0.0
        )
        abort_precision = float(trace_payload.get("b24_abort_precision", 0.0) or 0.0)
        confidence_input = float(
            np.clip(
                precision_vote * max(0.0, 1.0 - uncertainty_pressure)
                + precision_memory * 0.25
                + max(0.0, precision_vote - abort_precision) * 0.20,
                0.0,
                1.0,
            )
        )
        confidence_memory = max(
            float(getattr(self, "_b25_confidence_memory", 0.0))
            * float(params["b25_confidence_decay"]),
            confidence_input,
        )
        doubt_pressure = float(
            np.clip(
                uncertainty_pressure
                + abort_precision * 0.35
                - confidence_memory * 0.25,
                0.0,
                1.0,
            )
        )
        control_gain = float(
            np.clip(
                confidence_memory * float(params["b25_control_gain"])
                + precision_vote * 0.35
                - doubt_pressure * 0.15,
                0.0,
                1.0,
            )
        )
        confidence_vote = float(
            np.clip(control_gain + confidence_memory * 0.25 - doubt_pressure * 0.20, 0.0, 1.0)
        )
        meta_lock = int(getattr(self, "_b25_meta_lock", 0))
        metacognitive_state = "non_corridor"
        decision_label = "preserve_b24"

        if corridor_map:
            if (
                b24_decision in {"precision_conflict_continue", "continue_precision_lock"}
                and confidence_vote >= float(params["b25_confidence_threshold"])
                and doubt_pressure < float(params["b25_doubt_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                meta_lock = max(meta_lock, int(params["b25_meta_commit_ticks"]))
                metacognitive_state = "metacognition_confirms_route"
                decision_label = "metacognitive_confidence_continue"
                reason = "b25_metacognitive_confidence_continue"
            elif (
                doubt_pressure >= float(params["b25_doubt_threshold"])
                and doubt_pressure > confidence_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                metacognitive_state = "metacognition_withholds_route"
                decision_label = "metacognitive_confidence_abort"
                reason = "b25_metacognitive_confidence_abort"
            elif meta_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                metacognitive_state = "meta_lock_continues"
                decision_label = "continue_meta_lock"
                reason = "b25_continue_meta_lock"

        trace_payload.update(
            {
                "b25_controller_profile": profile,
                "b25_metacognitive_state": metacognitive_state,
                "b25_confidence_memory": round(float(confidence_memory), 6),
                "b25_confidence_vote": round(float(confidence_vote), 6),
                "b25_doubt_pressure": round(float(doubt_pressure), 6),
                "b25_control_gain": round(float(control_gain), 6),
                "b25_meta_lock": int(meta_lock),
                "b25_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b25_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b25_genetic_candidate"] = int(params["ga_candidate"])

        self._b25_confidence_memory = float(confidence_memory)
        self._b25_meta_lock = max(0, int(meta_lock) - 1)
        self._b25_last_tick = int(tick)
        return (
            semantic_action,
            B25_METACOGNITIVE_CONFIDENCE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b26_controller_params(self) -> dict[str, float]:
        params = self._b25_controller_params()
        defaults = {
            "b26_error_decay": 0.88,
            "b26_prediction_threshold": 0.24,
            "b26_abort_threshold": 0.68,
            "b26_control_gain": 0.46,
            "b26_stability_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "allostatic_prediction"
        )
        if profile == "setpoint_drift":
            defaults.update(
                {"b26_prediction_threshold": 0.22, "b26_stability_commit_ticks": 6.0}
            )
        elif profile == "error_suppression":
            defaults.update({"b26_abort_threshold": 0.64, "b26_control_gain": 0.52})
        elif profile == "allostatic_prediction_h56":
            defaults.update({"b26_error_decay": 0.90, "b26_stability_commit_ticks": 6.0})
        elif profile == "genetic_allostasis":
            defaults.update({"b26_prediction_threshold": 0.23, "b26_control_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b26_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b26_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b26_prediction_error_memory = 0.0
        self._b26_stability_lock = 0
        self._b26_last_tick = int(tick)

    def _b26_allostatic_prediction_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b25_metacognitive_confidence_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b26_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "allostatic_prediction"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b26_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        b25_decision = str(trace_payload.get("b25_decision", "preserve_b25"))
        confidence_vote = float(trace_payload.get("b25_confidence_vote", 0.0) or 0.0)
        doubt_pressure = float(trace_payload.get("b25_doubt_pressure", 0.0) or 0.0)
        control_gain = float(trace_payload.get("b25_control_gain", 0.0) or 0.0)
        homeostatic_pressure = float(
            np.clip(hunger * 0.45 + sleep_debt * 0.25 + max(0.0, 1.0 - health) * 0.30, 0.0, 1.0)
        )
        route_stability = float(
            np.clip(confidence_vote * 0.45 + control_gain * 0.35 - doubt_pressure * 0.20, 0.0, 1.0)
        )
        prediction_error_input = float(
            np.clip(homeostatic_pressure * 0.55 + doubt_pressure * 0.35 - route_stability * 0.30, 0.0, 1.0)
        )
        prediction_error = max(
            float(getattr(self, "_b26_prediction_error_memory", 0.0))
            * float(params["b26_error_decay"]),
            prediction_error_input,
        )
        setpoint_pressure = float(
            np.clip(homeostatic_pressure + prediction_error * 0.25 - route_stability * 0.15, 0.0, 1.0)
        )
        allostatic_vote = float(
            np.clip(
                route_stability * float(params["b26_control_gain"])
                + confidence_vote * 0.35
                - prediction_error * 0.18,
                0.0,
                1.0,
            )
        )
        stability_lock = int(getattr(self, "_b26_stability_lock", 0))
        allostatic_state = "non_corridor"
        decision_label = "preserve_b25"

        if corridor_map:
            if (
                b25_decision
                in {"metacognitive_confidence_continue", "continue_meta_lock"}
                and allostatic_vote >= float(params["b26_prediction_threshold"])
                and prediction_error < float(params["b26_abort_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                stability_lock = max(
                    stability_lock,
                    int(params["b26_stability_commit_ticks"]),
                )
                allostatic_state = "allostasis_confirms_route"
                decision_label = "allostatic_prediction_continue"
                reason = "b26_allostatic_prediction_continue"
            elif (
                prediction_error >= float(params["b26_abort_threshold"])
                and prediction_error > allostatic_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                allostatic_state = "allostasis_predicts_abort"
                decision_label = "allostatic_prediction_abort"
                reason = "b26_allostatic_prediction_abort"
            elif stability_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                allostatic_state = "allostatic_lock_continues"
                decision_label = "continue_allostatic_lock"
                reason = "b26_continue_allostatic_lock"

        trace_payload.update(
            {
                "b26_controller_profile": profile,
                "b26_allostatic_state": allostatic_state,
                "b26_prediction_error": round(float(prediction_error), 6),
                "b26_setpoint_pressure": round(float(setpoint_pressure), 6),
                "b26_control_vote": round(float(allostatic_vote), 6),
                "b26_stability_lock": int(stability_lock),
                "b26_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b26_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b26_genetic_candidate"] = int(params["ga_candidate"])

        self._b26_prediction_error_memory = float(prediction_error)
        self._b26_stability_lock = max(0, int(stability_lock) - 1)
        self._b26_last_tick = int(tick)
        return (
            semantic_action,
            B26_ALLOSTATIC_PREDICTION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b27_controller_params(self) -> dict[str, float]:
        params = self._b26_controller_params()
        defaults = {
            "b27_arousal_decay": 0.86,
            "b27_gain_threshold": 0.24,
            "b27_stress_threshold": 0.70,
            "b27_modulation_gain": 0.48,
            "b27_arousal_commit_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "arousal_gain")
        if profile == "stress_modulation":
            defaults.update({"b27_stress_threshold": 0.66, "b27_modulation_gain": 0.54})
        elif profile == "energy_arousal":
            defaults.update({"b27_gain_threshold": 0.22, "b27_arousal_commit_ticks": 6.0})
        elif profile == "arousal_gain_h56":
            defaults.update({"b27_arousal_decay": 0.88, "b27_arousal_commit_ticks": 6.0})
        elif profile == "genetic_arousal":
            defaults.update({"b27_gain_threshold": 0.23, "b27_modulation_gain": 0.52})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b27_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b27_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b27_arousal_memory = 0.0
        self._b27_arousal_lock = 0
        self._b27_last_tick = int(tick)

    def _b27_arousal_gain_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b26_allostatic_prediction_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b27_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "arousal_gain")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b27_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        threat_arr = np.asarray(observation.get("threat", np.zeros(1)), dtype=float).ravel()
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        threat = float(threat_arr[0]) if threat_arr.size else 0.0
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        b26_decision = str(trace_payload.get("b26_decision", "preserve_b26"))
        prediction_error = float(trace_payload.get("b26_prediction_error", 0.0) or 0.0)
        setpoint_pressure = float(trace_payload.get("b26_setpoint_pressure", 0.0) or 0.0)
        control_vote = float(trace_payload.get("b26_control_vote", 0.0) or 0.0)
        arousal_input = float(
            np.clip(
                setpoint_pressure * 0.35
                + control_vote * 0.30
                + hunger * 0.15
                + max(0.0, 1.0 - health) * 0.10
                + sleep_debt * 0.10,
                0.0,
                1.0,
            )
        )
        arousal_level = max(
            float(getattr(self, "_b27_arousal_memory", 0.0))
            * float(params["b27_arousal_decay"]),
            arousal_input,
        )
        stress_pressure = float(
            np.clip(threat * 0.35 + prediction_error * 0.40 - arousal_level * 0.18, 0.0, 1.0)
        )
        gain_modulation = float(
            np.clip(
                arousal_level * float(params["b27_modulation_gain"])
                + control_vote * 0.38
                - stress_pressure * 0.16,
                0.0,
                1.0,
            )
        )
        arousal_lock = int(getattr(self, "_b27_arousal_lock", 0))
        arousal_state = "non_corridor"
        decision_label = "preserve_b26"

        if corridor_map:
            if (
                b26_decision
                in {"allostatic_prediction_continue", "continue_allostatic_lock"}
                and gain_modulation >= float(params["b27_gain_threshold"])
                and stress_pressure < float(params["b27_stress_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                arousal_lock = max(arousal_lock, int(params["b27_arousal_commit_ticks"]))
                arousal_state = "arousal_gain_stabilizes_route"
                decision_label = "arousal_gain_continue"
                reason = "b27_arousal_gain_continue"
            elif (
                stress_pressure >= float(params["b27_stress_threshold"])
                and stress_pressure > gain_modulation
            ):
                semantic_action = "MOVE_TO_SHELTER"
                arousal_state = "stress_modulation_aborts_route"
                decision_label = "arousal_gain_abort"
                reason = "b27_arousal_gain_abort"
            elif arousal_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                arousal_state = "arousal_lock_continues"
                decision_label = "continue_arousal_lock"
                reason = "b27_continue_arousal_lock"

        trace_payload.update(
            {
                "b27_controller_profile": profile,
                "b27_arousal_state": arousal_state,
                "b27_arousal_level": round(float(arousal_level), 6),
                "b27_gain_modulation": round(float(gain_modulation), 6),
                "b27_stress_pressure": round(float(stress_pressure), 6),
                "b27_arousal_lock": int(arousal_lock),
                "b27_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b27_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b27_genetic_candidate"] = int(params["ga_candidate"])

        self._b27_arousal_memory = float(arousal_level)
        self._b27_arousal_lock = max(0, int(arousal_lock) - 1)
        self._b27_last_tick = int(tick)
        return (
            semantic_action,
            B27_AROUSAL_GAIN_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b28_controller_params(self) -> dict[str, float]:
        params = self._b27_controller_params()
        defaults = {
            "b28_attention_decay": 0.86,
            "b28_focus_threshold": 0.24,
            "b28_distractor_threshold": 0.70,
            "b28_attention_gain": 0.48,
            "b28_attention_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "interoceptive_attention"
        )
        if profile == "threat_focus_attention":
            defaults.update(
                {"b28_distractor_threshold": 0.66, "b28_attention_gain": 0.54}
            )
        elif profile == "homeostatic_attention":
            defaults.update(
                {"b28_focus_threshold": 0.22, "b28_attention_commit_ticks": 6.0}
            )
        elif profile == "interoceptive_attention_h56":
            defaults.update(
                {"b28_attention_decay": 0.88, "b28_attention_commit_ticks": 6.0}
            )
        elif profile == "genetic_attention":
            defaults.update({"b28_focus_threshold": 0.23, "b28_attention_gain": 0.52})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b28_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b28_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b28_focus_memory = 0.0
        self._b28_attention_lock = 0
        self._b28_last_tick = int(tick)

    def _b28_interoceptive_attention_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b27_arousal_gain_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b28_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "interoceptive_attention"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b28_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        threat_arr = np.asarray(observation.get("threat", np.zeros(1)), dtype=float).ravel()
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        threat = float(threat_arr[0]) if threat_arr.size else 0.0
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        b27_decision = str(trace_payload.get("b27_decision", "preserve_b27"))
        arousal_level = float(trace_payload.get("b27_arousal_level", 0.0) or 0.0)
        gain_modulation = float(trace_payload.get("b27_gain_modulation", 0.0) or 0.0)
        stress_pressure = float(trace_payload.get("b27_stress_pressure", 0.0) or 0.0)
        interoceptive_focus = float(
            np.clip(
                hunger * 0.30
                + sleep_debt * 0.18
                + max(0.0, 1.0 - health) * 0.16
                + arousal_level * 0.18
                + gain_modulation * 0.18,
                0.0,
                1.0,
            )
        )
        attention_memory = max(
            float(getattr(self, "_b28_focus_memory", 0.0))
            * float(params["b28_attention_decay"]),
            interoceptive_focus,
        )
        distractor_pressure = float(
            np.clip(threat * 0.34 + stress_pressure * 0.40 - attention_memory * 0.18, 0.0, 1.0)
        )
        attention_gain = float(
            np.clip(
                attention_memory * float(params["b28_attention_gain"])
                + gain_modulation * 0.36
                - distractor_pressure * 0.16,
                0.0,
                1.0,
            )
        )
        attention_lock = int(getattr(self, "_b28_attention_lock", 0))
        attention_state = "non_corridor"
        decision_label = "preserve_b27"

        if corridor_map:
            if (
                b27_decision in {"arousal_gain_continue", "continue_arousal_lock"}
                and attention_gain >= float(params["b28_focus_threshold"])
                and distractor_pressure < float(params["b28_distractor_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                attention_lock = max(
                    attention_lock,
                    int(params["b28_attention_commit_ticks"]),
                )
                attention_state = "interoceptive_attention_stabilizes_route"
                decision_label = "interoceptive_attention_continue"
                reason = "b28_interoceptive_attention_continue"
            elif (
                distractor_pressure >= float(params["b28_distractor_threshold"])
                and distractor_pressure > attention_gain
            ):
                semantic_action = "MOVE_TO_SHELTER"
                attention_state = "attention_distractor_aborts_route"
                decision_label = "interoceptive_attention_abort"
                reason = "b28_interoceptive_attention_abort"
            elif attention_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                attention_state = "attention_lock_continues"
                decision_label = "continue_attention_lock"
                reason = "b28_continue_attention_lock"

        trace_payload.update(
            {
                "b28_controller_profile": profile,
                "b28_attention_state": attention_state,
                "b28_interoceptive_focus": round(float(interoceptive_focus), 6),
                "b28_attention_gain": round(float(attention_gain), 6),
                "b28_distractor_pressure": round(float(distractor_pressure), 6),
                "b28_attention_lock": int(attention_lock),
                "b28_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b28_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b28_genetic_candidate"] = int(params["ga_candidate"])

        self._b28_focus_memory = float(attention_memory)
        self._b28_attention_lock = max(0, int(attention_lock) - 1)
        self._b28_last_tick = int(tick)
        return (
            semantic_action,
            B28_INTEROCEPTIVE_ATTENTION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b29_controller_params(self) -> dict[str, float]:
        params = self._b28_controller_params()
        defaults = {
            "b29_salience_decay": 0.86,
            "b29_corridor_threshold": 0.24,
            "b29_threat_threshold": 0.70,
            "b29_homeostatic_gain": 0.40,
            "b29_competition_gain": 0.52,
            "b29_salience_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "salience_competition"
        )
        if profile == "threat_salience_gate":
            defaults.update(
                {"b29_threat_threshold": 0.66, "b29_competition_gain": 0.56}
            )
        elif profile == "homeostatic_salience_gate":
            defaults.update(
                {"b29_homeostatic_gain": 0.46, "b29_salience_commit_ticks": 6.0}
            )
        elif profile == "salience_competition_h56":
            defaults.update(
                {"b29_salience_decay": 0.88, "b29_salience_commit_ticks": 6.0}
            )
        elif profile == "genetic_salience":
            defaults.update({"b29_corridor_threshold": 0.23, "b29_competition_gain": 0.55})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b29_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b29_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b29_salience_memory = 0.0
        self._b29_salience_lock = 0
        self._b29_last_tick = int(tick)

    def _b29_salience_competition_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b28_interoceptive_attention_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b29_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "salience_competition"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b29_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        threat_arr = np.asarray(observation.get("threat", np.zeros(1)), dtype=float).ravel()
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        threat = float(threat_arr[0]) if threat_arr.size else 0.0
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        attention_gain = float(trace_payload.get("b28_attention_gain", 0.0) or 0.0)
        distractor = float(trace_payload.get("b28_distractor_pressure", 0.0) or 0.0)
        focus = float(trace_payload.get("b28_interoceptive_focus", 0.0) or 0.0)
        b28_decision = str(trace_payload.get("b28_decision", "preserve_b28"))

        homeostatic_salience = float(
            np.clip(
                hunger * 0.36
                + sleep_debt * 0.22
                + max(0.0, 1.0 - health) * 0.24
                + focus * float(params["b29_homeostatic_gain"]),
                0.0,
                1.0,
            )
        )
        threat_salience = float(
            np.clip(threat * 0.52 + distractor * 0.34 - attention_gain * 0.14, 0.0, 1.0)
        )
        salience_memory = max(
            float(getattr(self, "_b29_salience_memory", 0.0))
            * float(params["b29_salience_decay"]),
            attention_gain,
        )
        corridor_salience = float(
            np.clip(
                salience_memory * float(params["b29_competition_gain"])
                + homeostatic_salience * 0.28
                - threat_salience * 0.18,
                0.0,
                1.0,
            )
        )
        salience_map = {
            "corridor": corridor_salience,
            "homeostasis": homeostatic_salience,
            "threat": threat_salience,
        }
        winner_channel = max(salience_map, key=salience_map.get)
        salience_lock = int(getattr(self, "_b29_salience_lock", 0))
        salience_state = "non_corridor"
        decision_label = "preserve_b28"

        if corridor_map:
            if (
                b28_decision in {
                    "interoceptive_attention_continue",
                    "continue_attention_lock",
                }
                and winner_channel in {"corridor", "homeostasis"}
                and corridor_salience >= float(params["b29_corridor_threshold"])
                and threat_salience < float(params["b29_threat_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                salience_lock = max(
                    salience_lock,
                    int(params["b29_salience_commit_ticks"]),
                )
                salience_state = "corridor_salience_wins"
                decision_label = "salience_competition_continue"
                reason = "b29_salience_competition_continue"
            elif (
                winner_channel == "threat"
                and threat_salience >= float(params["b29_threat_threshold"])
            ):
                semantic_action = "MOVE_TO_SHELTER"
                salience_state = "threat_salience_aborts"
                decision_label = "salience_competition_abort"
                reason = "b29_salience_competition_abort"
            elif salience_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                salience_state = "salience_lock_continues"
                decision_label = "continue_salience_lock"
                reason = "b29_continue_salience_lock"

        trace_payload.update(
            {
                "b29_controller_profile": profile,
                "b29_salience_state": salience_state,
                "b29_threat_salience": round(float(threat_salience), 6),
                "b29_homeostatic_salience": round(float(homeostatic_salience), 6),
                "b29_corridor_salience": round(float(corridor_salience), 6),
                "b29_winner_channel": winner_channel,
                "b29_salience_lock": int(salience_lock),
                "b29_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b29_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b29_genetic_candidate"] = int(params["ga_candidate"])

        self._b29_salience_memory = float(corridor_salience)
        self._b29_salience_lock = max(0, int(salience_lock) - 1)
        self._b29_last_tick = int(tick)
        return (
            semantic_action,
            B29_SALIENCE_COMPETITION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b30_controller_params(self) -> dict[str, float]:
        params = self._b29_controller_params()
        defaults = {
            "b30_gate_decay": 0.86,
            "b30_go_threshold": 0.24,
            "b30_nogo_threshold": 0.70,
            "b30_go_gain": 0.52,
            "b30_nogo_gain": 0.44,
            "b30_gate_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "basal_ganglia_gate"
        )
        if profile == "go_nogo_balance":
            defaults.update({"b30_go_threshold": 0.22, "b30_gate_commit_ticks": 6.0})
        elif profile == "threat_inhibition_gate":
            defaults.update({"b30_nogo_threshold": 0.66, "b30_nogo_gain": 0.50})
        elif profile == "basal_ganglia_gate_h56":
            defaults.update({"b30_gate_decay": 0.88, "b30_gate_commit_ticks": 6.0})
        elif profile == "genetic_action_gate":
            defaults.update({"b30_go_threshold": 0.23, "b30_go_gain": 0.55})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b30_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b30_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b30_gate_memory = 0.0
        self._b30_gate_lock = 0
        self._b30_last_tick = int(tick)

    def _b30_basal_ganglia_gate_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b29_salience_competition_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b30_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "basal_ganglia_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b30_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        threat_arr = np.asarray(observation.get("threat", np.zeros(1)), dtype=float).ravel()
        threat = float(threat_arr[0]) if threat_arr.size else 0.0
        attention_gain = float(trace_payload.get("b28_attention_gain", 0.0) or 0.0)
        distractor = float(trace_payload.get("b28_distractor_pressure", 0.0) or 0.0)
        corridor_salience = float(trace_payload.get("b29_corridor_salience", 0.0) or 0.0)
        homeostatic_salience = float(
            trace_payload.get("b29_homeostatic_salience", 0.0) or 0.0
        )
        threat_salience = float(trace_payload.get("b29_threat_salience", 0.0) or 0.0)
        b29_decision = str(trace_payload.get("b29_decision", "preserve_b29"))

        gate_memory = max(
            float(getattr(self, "_b30_gate_memory", 0.0))
            * float(params["b30_gate_decay"]),
            corridor_salience,
        )
        no_go_signal = float(
            np.clip(
                threat * 0.34
                + threat_salience * float(params["b30_nogo_gain"])
                + distractor * 0.22
                - gate_memory * 0.10,
                0.0,
                1.0,
            )
        )
        go_signal = float(
            np.clip(
                gate_memory * float(params["b30_go_gain"])
                + homeostatic_salience * 0.30
                + attention_gain * 0.24
                - no_go_signal * 0.18,
                0.0,
                1.0,
            )
        )
        action_gate = "go" if go_signal >= no_go_signal else "no_go"
        gate_lock = int(getattr(self, "_b30_gate_lock", 0))
        gate_state = "non_corridor"
        decision_label = "preserve_b29"

        if corridor_map:
            if (
                b29_decision in {"salience_competition_continue", "continue_salience_lock"}
                and go_signal >= float(params["b30_go_threshold"])
                and no_go_signal < float(params["b30_nogo_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                gate_lock = max(gate_lock, int(params["b30_gate_commit_ticks"]))
                gate_state = "basal_go_gate_opens"
                decision_label = "basal_gate_go"
                reason = "b30_basal_gate_go"
            elif no_go_signal >= float(params["b30_nogo_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                action_gate = "no_go"
                gate_state = "basal_nogo_gate_inhibits"
                decision_label = "basal_gate_no_go"
                reason = "b30_basal_gate_no_go"
            elif gate_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                action_gate = "go"
                gate_state = "basal_gate_lock_continues"
                decision_label = "continue_basal_gate_lock"
                reason = "b30_continue_basal_gate_lock"

        trace_payload.update(
            {
                "b30_controller_profile": profile,
                "b30_gate_state": gate_state,
                "b30_go_signal": round(float(go_signal), 6),
                "b30_no_go_signal": round(float(no_go_signal), 6),
                "b30_action_gate": action_gate,
                "b30_gate_lock": int(gate_lock),
                "b30_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b30_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b30_genetic_candidate"] = int(params["ga_candidate"])

        self._b30_gate_memory = float(go_signal)
        self._b30_gate_lock = max(0, int(gate_lock) - 1)
        self._b30_last_tick = int(tick)
        return (
            semantic_action,
            B30_BASAL_GANGLIA_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b31_controller_params(self) -> dict[str, float]:
        params = self._b30_controller_params()
        defaults = {
            "b31_dopamine_decay": 0.86,
            "b31_go_threshold": 0.24,
            "b31_nogo_threshold": 0.70,
            "b31_prediction_gain": 0.46,
            "b31_phasic_gain": 0.52,
            "b31_tonic_gain": 0.42,
            "b31_dopamine_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "dopamine_prediction_error"
        )
        if profile == "tonic_dopamine_gate":
            defaults.update({"b31_tonic_gain": 0.48, "b31_dopamine_commit_ticks": 6.0})
        elif profile == "phasic_dopamine_gate":
            defaults.update({"b31_phasic_gain": 0.58, "b31_go_threshold": 0.22})
        elif profile == "dopamine_prediction_error_h56":
            defaults.update({"b31_dopamine_decay": 0.88, "b31_dopamine_commit_ticks": 6.0})
        elif profile == "genetic_dopamine_gate":
            defaults.update({"b31_prediction_gain": 0.50, "b31_phasic_gain": 0.55})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b31_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b31_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b31_value_prediction = 0.0
        self._b31_dopamine_memory = 0.0
        self._b31_dopamine_lock = 0
        self._b31_last_tick = int(tick)

    def _b31_dopamine_prediction_error_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b30_basal_ganglia_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b31_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "dopamine_prediction_error"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b31_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        go_signal = float(trace_payload.get("b30_go_signal", 0.0) or 0.0)
        no_go_signal = float(trace_payload.get("b30_no_go_signal", 0.0) or 0.0)
        action_gate = str(trace_payload.get("b30_action_gate", "go"))
        b30_decision = str(trace_payload.get("b30_decision", "preserve_b30"))
        corridor_salience = float(trace_payload.get("b29_corridor_salience", 0.0) or 0.0)
        homeostatic_salience = float(
            trace_payload.get("b29_homeostatic_salience", 0.0) or 0.0
        )

        reward_proxy = float(
            np.clip(
                corridor_salience * 0.40
                + homeostatic_salience * 0.30
                + go_signal * 0.24
                + max(0.0, health - 0.5) * 0.06
                - no_go_signal * 0.18
                - max(0.0, 0.70 - hunger) * 0.04,
                0.0,
                1.0,
            )
        )
        prediction = float(getattr(self, "_b31_value_prediction", 0.0))
        reward_prediction_error = reward_proxy - prediction
        tonic_dopamine = max(
            float(getattr(self, "_b31_dopamine_memory", 0.0))
            * float(params["b31_dopamine_decay"]),
            reward_proxy * float(params["b31_tonic_gain"]),
        )
        phasic_dopamine = float(
            np.clip(
                max(0.0, reward_prediction_error) * float(params["b31_phasic_gain"])
                + go_signal * 0.20
                - no_go_signal * 0.12,
                0.0,
                1.0,
            )
        )
        gate_bias = float(
            np.clip(
                tonic_dopamine
                + phasic_dopamine
                + reward_prediction_error * float(params["b31_prediction_gain"]),
                -1.0,
                1.0,
            )
        )
        dopamine_lock = int(getattr(self, "_b31_dopamine_lock", 0))
        dopamine_state = "non_corridor"
        decision_label = "preserve_b30"

        if corridor_map:
            if (
                b30_decision in {"basal_gate_go", "continue_basal_gate_lock"}
                and action_gate == "go"
                and gate_bias >= float(params["b31_go_threshold"])
                and no_go_signal < float(params["b31_nogo_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                dopamine_lock = max(
                    dopamine_lock,
                    int(params["b31_dopamine_commit_ticks"]),
                )
                dopamine_state = "dopamine_go_bias_stabilizes"
                decision_label = "dopamine_gate_go"
                reason = "b31_dopamine_gate_go"
            elif no_go_signal >= float(params["b31_nogo_threshold"]) and gate_bias < 0.25:
                semantic_action = "MOVE_TO_SHELTER"
                dopamine_state = "dopamine_no_go_inhibits"
                decision_label = "dopamine_gate_no_go"
                reason = "b31_dopamine_gate_no_go"
            elif dopamine_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                dopamine_state = "dopamine_lock_continues"
                decision_label = "continue_dopamine_lock"
                reason = "b31_continue_dopamine_lock"

        trace_payload.update(
            {
                "b31_controller_profile": profile,
                "b31_dopamine_state": dopamine_state,
                "b31_reward_prediction_error": round(float(reward_prediction_error), 6),
                "b31_tonic_dopamine": round(float(tonic_dopamine), 6),
                "b31_phasic_dopamine": round(float(phasic_dopamine), 6),
                "b31_gate_bias": round(float(gate_bias), 6),
                "b31_dopamine_lock": int(dopamine_lock),
                "b31_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b31_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b31_genetic_candidate"] = int(params["ga_candidate"])

        self._b31_value_prediction = float(
            prediction + 0.35 * reward_prediction_error
        )
        self._b31_dopamine_memory = float(tonic_dopamine)
        self._b31_dopamine_lock = max(0, int(dopamine_lock) - 1)
        self._b31_last_tick = int(tick)
        return (
            semantic_action,
            B31_DOPAMINE_PREDICTION_ERROR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b32_controller_params(self) -> dict[str, float]:
        params = self._b31_controller_params()
        defaults = {
            "b32_value_decay": 0.88,
            "b32_advantage_threshold": 0.20,
            "b32_abort_threshold": -0.18,
            "b32_critic_gain": 0.46,
            "b32_actor_gain": 0.50,
            "b32_value_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "actor_critic_value"
        )
        if profile == "advantage_value_gate":
            defaults.update({"b32_advantage_threshold": 0.18, "b32_actor_gain": 0.56})
        elif profile == "critic_stability":
            defaults.update({"b32_value_decay": 0.90, "b32_value_commit_ticks": 6.0})
        elif profile == "actor_critic_value_h56":
            defaults.update({"b32_value_decay": 0.90, "b32_value_commit_ticks": 6.0})
        elif profile == "genetic_actor_critic":
            defaults.update({"b32_advantage_threshold": 0.18, "b32_actor_gain": 0.54})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b32_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b32_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b32_critic_value = 0.0
        self._b32_value_lock = 0
        self._b32_last_tick = int(tick)

    def _b32_actor_critic_value_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b31_dopamine_prediction_error_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b32_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "actor_critic_value"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b32_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        gate_bias = float(trace_payload.get("b31_gate_bias", 0.0) or 0.0)
        tonic_dopamine = float(trace_payload.get("b31_tonic_dopamine", 0.0) or 0.0)
        phasic_dopamine = float(trace_payload.get("b31_phasic_dopamine", 0.0) or 0.0)
        b31_decision = str(trace_payload.get("b31_decision", "preserve_b31"))
        no_go_signal = float(trace_payload.get("b30_no_go_signal", 0.0) or 0.0)
        go_signal = float(trace_payload.get("b30_go_signal", 0.0) or 0.0)

        current_value = float(getattr(self, "_b32_critic_value", 0.0))
        reward_proxy = float(
            np.clip(
                gate_bias * 0.34
                + tonic_dopamine * 0.24
                + phasic_dopamine * 0.20
                + go_signal * 0.16
                + max(0.0, health - 0.5) * 0.06
                - no_go_signal * 0.22
                - max(0.0, 0.72 - hunger) * 0.05,
                -1.0,
                1.0,
            )
        )
        value_error = reward_proxy - current_value
        critic_value = float(
            np.clip(
                current_value * float(params["b32_value_decay"])
                + reward_proxy * float(params["b32_critic_gain"]),
                -1.0,
                1.0,
            )
        )
        actor_advantage = float(
            np.clip(
                value_error * float(params["b32_actor_gain"])
                + gate_bias * 0.35
                + go_signal * 0.18
                - no_go_signal * 0.20,
                -1.0,
                1.0,
            )
        )
        policy_bias = float(np.clip(critic_value + actor_advantage, -1.0, 1.0))
        value_lock = int(getattr(self, "_b32_value_lock", 0))
        decision_label = "preserve_b31"

        if corridor_map:
            if (
                b31_decision in {"dopamine_gate_go", "continue_dopamine_lock"}
                and actor_advantage >= float(params["b32_advantage_threshold"])
                and policy_bias > 0.0
            ):
                semantic_action = "MOVE_TO_FOOD"
                value_lock = max(value_lock, int(params["b32_value_commit_ticks"]))
                decision_label = "actor_critic_commit"
                reason = "b32_actor_critic_commit"
            elif policy_bias <= float(params["b32_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "critic_abort_negative_value"
                reason = "b32_critic_abort_negative_value"
            elif value_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_value_lock"
                reason = "b32_continue_value_lock"

        trace_payload.update(
            {
                "b32_controller_profile": profile,
                "b32_critic_value": round(float(critic_value), 6),
                "b32_actor_advantage": round(float(actor_advantage), 6),
                "b32_value_error": round(float(value_error), 6),
                "b32_policy_bias": round(float(policy_bias), 6),
                "b32_value_lock": int(value_lock),
                "b32_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b32_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b32_genetic_candidate"] = int(params["ga_candidate"])

        self._b32_critic_value = float(critic_value)
        self._b32_value_lock = max(0, int(value_lock) - 1)
        self._b32_last_tick = int(tick)
        return (
            semantic_action,
            B32_ACTOR_CRITIC_VALUE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b33_controller_params(self) -> dict[str, float]:
        params = self._b32_controller_params()
        defaults = {
            "b33_td_decay": 0.88,
            "b33_bootstrap_gain": 0.42,
            "b33_reward_trace_gain": 0.36,
            "b33_td_commit_ticks": 5.0,
            "b33_td_threshold": 0.16,
            "b33_abort_threshold": -0.20,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "td_error_decomposition"
        )
        if profile == "bootstrapped_value_gate":
            defaults.update({"b33_bootstrap_gain": 0.48, "b33_td_commit_ticks": 6.0})
        elif profile == "reward_trace_critic":
            defaults.update({"b33_reward_trace_gain": 0.44, "b33_td_threshold": 0.14})
        elif profile == "td_error_decomposition_h56":
            defaults.update({"b33_td_decay": 0.90, "b33_td_commit_ticks": 6.0})
        elif profile == "genetic_td_value":
            defaults.update({"b33_bootstrap_gain": 0.46, "b33_reward_trace_gain": 0.40})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b33_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b33_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b33_bootstrap_value = 0.0
        self._b33_reward_trace = 0.0
        self._b33_td_lock = 0
        self._b33_last_tick = int(tick)

    def _b33_td_error_decomposition_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b32_actor_critic_value_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b33_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "td_error_decomposition"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b33_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        critic_value = float(trace_payload.get("b32_critic_value", 0.0) or 0.0)
        actor_advantage = float(trace_payload.get("b32_actor_advantage", 0.0) or 0.0)
        policy_bias = float(trace_payload.get("b32_policy_bias", 0.0) or 0.0)
        value_error = float(trace_payload.get("b32_value_error", 0.0) or 0.0)
        b32_decision = str(trace_payload.get("b32_decision", "preserve_b32"))
        dopamine_error = float(
            trace_payload.get("b31_reward_prediction_error", 0.0) or 0.0
        )
        gate_bias = float(trace_payload.get("b31_gate_bias", 0.0) or 0.0)
        previous_bootstrap = float(getattr(self, "_b33_bootstrap_value", 0.0))
        previous_trace = float(getattr(self, "_b33_reward_trace", 0.0))
        reward_trace = float(
            np.clip(
                previous_trace * float(params["b33_td_decay"])
                + max(0.0, actor_advantage) * float(params["b33_reward_trace_gain"]),
                -1.0,
                1.0,
            )
        )
        bootstrap_value = float(
            np.clip(
                previous_bootstrap * float(params["b33_td_decay"])
                + critic_value * float(params["b33_bootstrap_gain"])
                + policy_bias * 0.22,
                -1.0,
                1.0,
            )
        )
        td_error = float(
            np.clip(
                value_error
                + dopamine_error * 0.35
                + reward_trace * 0.28
                + bootstrap_value * 0.20
                - previous_bootstrap * 0.18,
                -1.0,
                1.0,
            )
        )
        actor_update = float(
            np.clip(
                actor_advantage + td_error * 0.45 + gate_bias * 0.12,
                -1.0,
                1.0,
            )
        )
        td_lock = int(getattr(self, "_b33_td_lock", 0))
        decision_label = "preserve_b32"

        if corridor_map:
            if (
                b32_decision in {"actor_critic_commit", "continue_value_lock"}
                and actor_update >= float(params["b33_td_threshold"])
                and td_error > float(params["b33_abort_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                td_lock = max(td_lock, int(params["b33_td_commit_ticks"]))
                decision_label = "td_error_commit"
                reason = "b33_td_error_commit"
            elif td_error <= float(params["b33_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "td_error_abort"
                reason = "b33_td_error_abort"
            elif td_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_td_lock"
                reason = "b33_continue_td_lock"

        trace_payload.update(
            {
                "b33_controller_profile": profile,
                "b33_td_error": round(float(td_error), 6),
                "b33_bootstrap_value": round(float(bootstrap_value), 6),
                "b33_reward_trace": round(float(reward_trace), 6),
                "b33_actor_update": round(float(actor_update), 6),
                "b33_td_lock": int(td_lock),
                "b33_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b33_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b33_genetic_candidate"] = int(params["ga_candidate"])

        self._b33_bootstrap_value = float(bootstrap_value)
        self._b33_reward_trace = float(reward_trace)
        self._b33_td_lock = max(0, int(td_lock) - 1)
        self._b33_last_tick = int(tick)
        return (
            semantic_action,
            B33_TD_ERROR_DECOMPOSITION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b34_controller_params(self) -> dict[str, float]:
        params = self._b33_controller_params()
        defaults = {
            "b34_eligibility_decay": 0.86,
            "b34_credit_gain": 0.38,
            "b34_tag_gain": 0.32,
            "b34_credit_threshold": 0.12,
            "b34_abort_threshold": -0.18,
            "b34_credit_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "eligibility_credit"
        )
        if profile == "delayed_credit_gate":
            defaults.update({"b34_credit_gain": 0.44, "b34_credit_lock_ticks": 6.0})
        elif profile == "synaptic_tagging":
            defaults.update({"b34_tag_gain": 0.42, "b34_credit_threshold": 0.10})
        elif profile == "eligibility_credit_h56":
            defaults.update({"b34_eligibility_decay": 0.89, "b34_credit_lock_ticks": 6.0})
        elif profile == "genetic_eligibility":
            defaults.update({"b34_credit_gain": 0.42, "b34_tag_gain": 0.38})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b34_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b34_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b34_eligibility_trace = 0.0
        self._b34_synaptic_tag = 0.0
        self._b34_credit_lock = 0
        self._b34_last_tick = int(tick)

    def _b34_eligibility_credit_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b33_td_error_decomposition_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b34_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "eligibility_credit"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b34_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        td_error = float(trace_payload.get("b33_td_error", 0.0) or 0.0)
        actor_update = float(trace_payload.get("b33_actor_update", 0.0) or 0.0)
        reward_trace = float(trace_payload.get("b33_reward_trace", 0.0) or 0.0)
        b33_decision = str(trace_payload.get("b33_decision", "preserve_b33"))
        previous_eligibility = float(getattr(self, "_b34_eligibility_trace", 0.0))
        previous_tag = float(getattr(self, "_b34_synaptic_tag", 0.0))
        decay = float(params["b34_eligibility_decay"])
        eligibility_trace = float(
            np.clip(previous_eligibility * decay + max(0.0, actor_update) * 0.42, -1.0, 1.0)
        )
        synaptic_tag = float(
            np.clip(previous_tag * decay + max(0.0, reward_trace) * float(params["b34_tag_gain"]), -1.0, 1.0)
        )
        credit_assignment = float(
            np.clip(
                eligibility_trace * float(params["b34_credit_gain"])
                + synaptic_tag * 0.30
                + td_error * 0.34,
                -1.0,
                1.0,
            )
        )
        decay_memory = float(np.clip(previous_eligibility * decay, -1.0, 1.0))
        credit_lock = int(getattr(self, "_b34_credit_lock", 0))
        decision_label = "preserve_b33"

        if corridor_map:
            if (
                b33_decision in {"td_error_commit", "continue_td_lock"}
                and credit_assignment >= float(params["b34_credit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                credit_lock = max(credit_lock, int(params["b34_credit_lock_ticks"]))
                decision_label = "eligibility_credit_commit"
                reason = "b34_eligibility_credit_commit"
            elif credit_assignment <= float(params["b34_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "eligibility_credit_abort"
                reason = "b34_eligibility_credit_abort"
            elif credit_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_eligibility_lock"
                reason = "b34_continue_eligibility_lock"

        trace_payload.update(
            {
                "b34_controller_profile": profile,
                "b34_eligibility_trace": round(float(eligibility_trace), 6),
                "b34_credit_assignment": round(float(credit_assignment), 6),
                "b34_synaptic_tag": round(float(synaptic_tag), 6),
                "b34_decay_memory": round(float(decay_memory), 6),
                "b34_credit_lock": int(credit_lock),
                "b34_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b34_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b34_genetic_candidate"] = int(params["ga_candidate"])

        self._b34_eligibility_trace = float(eligibility_trace)
        self._b34_synaptic_tag = float(synaptic_tag)
        self._b34_credit_lock = max(0, int(credit_lock) - 1)
        self._b34_last_tick = int(tick)
        return (
            semantic_action,
            B34_ELIGIBILITY_CREDIT_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b35_controller_params(self) -> dict[str, float]:
        params = self._b34_controller_params()
        defaults = {
            "b35_model_decay": 0.84,
            "b35_prediction_gain": 0.40,
            "b35_transition_gain": 0.34,
            "b35_confidence_gain": 0.30,
            "b35_commit_threshold": 0.14,
            "b35_abort_threshold": -0.16,
            "b35_model_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "forward_model_value"
        )
        if profile == "transition_error_gate":
            defaults.update({"b35_transition_gain": 0.42, "b35_commit_threshold": 0.12})
        elif profile == "model_confidence":
            defaults.update({"b35_confidence_gain": 0.38, "b35_model_lock_ticks": 6.0})
        elif profile == "forward_model_value_h56":
            defaults.update({"b35_model_decay": 0.87, "b35_model_lock_ticks": 6.0})
        elif profile == "genetic_forward_model":
            defaults.update({"b35_prediction_gain": 0.44, "b35_confidence_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b35_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b35_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b35_prediction_memory = 0.0
        self._b35_model_confidence = 0.0
        self._b35_model_lock = 0
        self._b35_last_tick = int(tick)

    def _b35_forward_model_value_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b34_eligibility_credit_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b35_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "forward_model_value"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b35_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        food_dist = float(meta.get("food_dist", 0.0) or 0.0)
        shelter_dist = float(meta.get("shelter_dist", 0.0) or 0.0)
        credit_assignment = float(trace_payload.get("b34_credit_assignment", 0.0) or 0.0)
        eligibility_trace = float(trace_payload.get("b34_eligibility_trace", 0.0) or 0.0)
        td_error = float(trace_payload.get("b33_td_error", 0.0) or 0.0)
        b34_decision = str(trace_payload.get("b34_decision", "preserve_b34"))
        previous_prediction = float(getattr(self, "_b35_prediction_memory", 0.0))
        previous_confidence = float(getattr(self, "_b35_model_confidence", 0.0))
        distance_advantage = float(np.clip((shelter_dist - food_dist) / 12.0, -1.0, 1.0))
        predicted_delta = float(
            np.clip(
                credit_assignment * float(params["b35_prediction_gain"])
                + eligibility_trace * 0.24
                + distance_advantage * 0.18,
                -1.0,
                1.0,
            )
        )
        prediction_memory = float(
            np.clip(
                previous_prediction * float(params["b35_model_decay"]) + predicted_delta,
                -1.0,
                1.0,
            )
        )
        transition_error = float(
            np.clip(
                predicted_delta
                - previous_prediction * 0.28
                + td_error * float(params["b35_transition_gain"]),
                -1.0,
                1.0,
            )
        )
        model_confidence = float(
            np.clip(
                previous_confidence * float(params["b35_model_decay"])
                + max(0.0, credit_assignment) * float(params["b35_confidence_gain"])
                - max(0.0, -transition_error) * 0.18,
                -1.0,
                1.0,
            )
        )
        forward_value = float(
            np.clip(prediction_memory * 0.48 + transition_error * 0.32 + model_confidence * 0.20, -1.0, 1.0)
        )
        model_lock = int(getattr(self, "_b35_model_lock", 0))
        decision_label = "preserve_b34"

        if corridor_map:
            if (
                b34_decision in {"eligibility_credit_commit", "continue_eligibility_lock"}
                and forward_value >= float(params["b35_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                model_lock = max(model_lock, int(params["b35_model_lock_ticks"]))
                decision_label = "forward_model_commit"
                reason = "b35_forward_model_commit"
            elif forward_value <= float(params["b35_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "forward_model_abort"
                reason = "b35_forward_model_abort"
            elif model_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_model_lock"
                reason = "b35_continue_model_lock"

        trace_payload.update(
            {
                "b35_controller_profile": profile,
                "b35_forward_value": round(float(forward_value), 6),
                "b35_transition_error": round(float(transition_error), 6),
                "b35_model_confidence": round(float(model_confidence), 6),
                "b35_prediction_memory": round(float(prediction_memory), 6),
                "b35_model_lock": int(model_lock),
                "b35_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b35_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b35_genetic_candidate"] = int(params["ga_candidate"])

        self._b35_prediction_memory = float(prediction_memory)
        self._b35_model_confidence = float(model_confidence)
        self._b35_model_lock = max(0, int(model_lock) - 1)
        self._b35_last_tick = int(tick)
        return (
            semantic_action,
            B35_FORWARD_MODEL_VALUE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b36_controller_params(self) -> dict[str, float]:
        params = self._b35_controller_params()
        defaults = {
            "b36_belief_decay": 0.86,
            "b36_latent_gain": 0.38,
            "b36_context_gain": 0.32,
            "b36_error_gain": 0.34,
            "b36_commit_threshold": 0.14,
            "b36_abort_threshold": -0.16,
            "b36_belief_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "latent_belief_state"
        )
        if profile == "belief_error_gate":
            defaults.update({"b36_error_gain": 0.42, "b36_commit_threshold": 0.12})
        elif profile == "context_inference":
            defaults.update({"b36_context_gain": 0.40, "b36_belief_lock_ticks": 6.0})
        elif profile == "latent_belief_state_h56":
            defaults.update({"b36_belief_decay": 0.89, "b36_belief_lock_ticks": 6.0})
        elif profile == "genetic_belief_state":
            defaults.update({"b36_latent_gain": 0.42, "b36_context_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b36_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b36_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b36_latent_state = 0.0
        self._b36_context_memory = 0.0
        self._b36_state_confidence = 0.0
        self._b36_belief_lock = 0
        self._b36_last_tick = int(tick)

    def _b36_latent_belief_state_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b35_forward_model_value_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b36_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "latent_belief_state"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b36_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        food_dist = float(meta.get("food_dist", 0.0) or 0.0)
        shelter_dist = float(meta.get("shelter_dist", 0.0) or 0.0)
        forward_value = float(trace_payload.get("b35_forward_value", 0.0) or 0.0)
        transition_error = float(trace_payload.get("b35_transition_error", 0.0) or 0.0)
        model_confidence = float(trace_payload.get("b35_model_confidence", 0.0) or 0.0)
        prediction_memory = float(trace_payload.get("b35_prediction_memory", 0.0) or 0.0)
        b35_decision = str(trace_payload.get("b35_decision", "preserve_b35"))
        previous_latent = float(getattr(self, "_b36_latent_state", 0.0))
        previous_context = float(getattr(self, "_b36_context_memory", 0.0))
        previous_confidence = float(getattr(self, "_b36_state_confidence", 0.0))
        corridor_context = float(np.clip((shelter_dist - food_dist) / 12.0, -1.0, 1.0))
        context_memory = float(
            np.clip(
                previous_context * float(params["b36_belief_decay"])
                + corridor_context * float(params["b36_context_gain"]),
                -1.0,
                1.0,
            )
        )
        latent_state = float(
            np.clip(
                previous_latent * float(params["b36_belief_decay"])
                + forward_value * float(params["b36_latent_gain"])
                + prediction_memory * 0.22
                + context_memory * 0.18,
                -1.0,
                1.0,
            )
        )
        belief_error = float(
            np.clip(
                transition_error * float(params["b36_error_gain"])
                + latent_state
                - previous_latent * 0.30,
                -1.0,
                1.0,
            )
        )
        state_confidence = float(
            np.clip(
                previous_confidence * float(params["b36_belief_decay"])
                + max(0.0, model_confidence) * 0.34
                + max(0.0, latent_state) * 0.24
                - max(0.0, -belief_error) * 0.18,
                -1.0,
                1.0,
            )
        )
        belief_score = float(
            np.clip(latent_state * 0.42 + belief_error * 0.30 + state_confidence * 0.28, -1.0, 1.0)
        )
        belief_lock = int(getattr(self, "_b36_belief_lock", 0))
        decision_label = "preserve_b35"

        if corridor_map:
            if (
                b35_decision in {"forward_model_commit", "continue_model_lock"}
                and belief_score >= float(params["b36_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                belief_lock = max(belief_lock, int(params["b36_belief_lock_ticks"]))
                decision_label = "latent_belief_commit"
                reason = "b36_latent_belief_commit"
            elif belief_score <= float(params["b36_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "latent_belief_abort"
                reason = "b36_latent_belief_abort"
            elif belief_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_belief_lock"
                reason = "b36_continue_belief_lock"

        trace_payload.update(
            {
                "b36_controller_profile": profile,
                "b36_latent_state": round(float(latent_state), 6),
                "b36_belief_error": round(float(belief_error), 6),
                "b36_state_confidence": round(float(state_confidence), 6),
                "b36_context_memory": round(float(context_memory), 6),
                "b36_belief_lock": int(belief_lock),
                "b36_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b36_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b36_genetic_candidate"] = int(params["ga_candidate"])

        self._b36_latent_state = float(latent_state)
        self._b36_context_memory = float(context_memory)
        self._b36_state_confidence = float(state_confidence)
        self._b36_belief_lock = max(0, int(belief_lock) - 1)
        self._b36_last_tick = int(tick)
        return (
            semantic_action,
            B36_LATENT_BELIEF_STATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b37_controller_params(self) -> dict[str, float]:
        params = self._b36_controller_params()
        defaults = {
            "b37_factor_decay": 0.86,
            "b37_external_gain": 0.36,
            "b37_internal_gain": 0.34,
            "b37_factor_balance_gain": 0.32,
            "b37_commit_threshold": 0.14,
            "b37_abort_threshold": -0.16,
            "b37_factor_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "state_factor_gate"
        )
        if profile == "intero_extero_factor":
            defaults.update({"b37_external_gain": 0.40, "b37_internal_gain": 0.38})
        elif profile == "factor_confidence":
            defaults.update(
                {"b37_factor_balance_gain": 0.40, "b37_factor_lock_ticks": 6.0}
            )
        elif profile == "state_factor_gate_h56":
            defaults.update({"b37_factor_decay": 0.89, "b37_factor_lock_ticks": 6.0})
        elif profile == "genetic_state_factor":
            defaults.update({"b37_external_gain": 0.38, "b37_internal_gain": 0.38})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b37_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b37_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b37_external_state_factor = 0.0
        self._b37_internal_state_factor = 0.0
        self._b37_factor_confidence = 0.0
        self._b37_factor_lock = 0
        self._b37_last_tick = int(tick)

    def _b37_state_factor_gate_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b36_latent_belief_state_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b37_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "state_factor_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b37_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        food_dist = float(meta.get("food_dist", 0.0) or 0.0)
        shelter_dist = float(meta.get("shelter_dist", 0.0) or 0.0)
        corridor_context = float(np.clip((shelter_dist - food_dist) / 12.0, -1.0, 1.0))
        latent_state = float(trace_payload.get("b36_latent_state", 0.0) or 0.0)
        belief_error = float(trace_payload.get("b36_belief_error", 0.0) or 0.0)
        state_confidence = float(trace_payload.get("b36_state_confidence", 0.0) or 0.0)
        context_memory = float(trace_payload.get("b36_context_memory", 0.0) or 0.0)
        b36_decision = str(trace_payload.get("b36_decision", "preserve_b35"))

        decay = float(params["b37_factor_decay"])
        previous_external = float(getattr(self, "_b37_external_state_factor", 0.0))
        previous_internal = float(getattr(self, "_b37_internal_state_factor", 0.0))
        previous_confidence = float(getattr(self, "_b37_factor_confidence", 0.0))
        external_factor = float(
            np.clip(
                previous_external * decay
                + context_memory * float(params["b37_external_gain"])
                + corridor_context * 0.18,
                -1.0,
                1.0,
            )
        )
        internal_factor = float(
            np.clip(
                previous_internal * decay
                + latent_state * float(params["b37_internal_gain"])
                + state_confidence * 0.22,
                -1.0,
                1.0,
            )
        )
        balance_gain = float(params.get("b37_factor_balance_gain", 0.32))
        factor_alignment = float(
            np.clip(
                (external_factor + internal_factor) * 0.5
                + belief_error * balance_gain,
                -1.0,
                1.0,
            )
        )
        factor_confidence = float(
            np.clip(
                previous_confidence * decay
                + max(0.0, state_confidence) * 0.30
                + max(0.0, factor_alignment) * 0.24
                - max(0.0, -factor_alignment) * 0.18,
                -1.0,
                1.0,
            )
        )
        factor_score = float(
            np.clip(
                factor_alignment * 0.48
                + factor_confidence * 0.32
                + internal_factor * 0.20,
                -1.0,
                1.0,
            )
        )
        factor_lock = int(getattr(self, "_b37_factor_lock", 0))
        decision_label = "preserve_b36"

        if corridor_map:
            if (
                b36_decision in {"latent_belief_commit", "continue_belief_lock"}
                and factor_score >= float(params["b37_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                factor_lock = max(factor_lock, int(params["b37_factor_lock_ticks"]))
                decision_label = "state_factor_commit"
                reason = "b37_state_factor_commit"
            elif factor_score <= float(params["b37_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "state_factor_abort"
                reason = "b37_state_factor_abort"
            elif factor_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_factor_lock"
                reason = "b37_continue_factor_lock"

        trace_payload.update(
            {
                "b37_controller_profile": profile,
                "b37_external_state_factor": round(float(external_factor), 6),
                "b37_internal_state_factor": round(float(internal_factor), 6),
                "b37_factor_alignment": round(float(factor_alignment), 6),
                "b37_factor_confidence": round(float(factor_confidence), 6),
                "b37_factor_lock": int(factor_lock),
                "b37_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b37_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b37_genetic_candidate"] = int(params["ga_candidate"])

        self._b37_external_state_factor = float(external_factor)
        self._b37_internal_state_factor = float(internal_factor)
        self._b37_factor_confidence = float(factor_confidence)
        self._b37_factor_lock = max(0, int(factor_lock) - 1)
        self._b37_last_tick = int(tick)
        return (
            semantic_action,
            B37_STATE_FACTOR_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b38_controller_params(self) -> dict[str, float]:
        params = self._b37_controller_params()
        defaults = {
            "b38_attention_decay": 0.86,
            "b38_external_attention_gain": 0.34,
            "b38_internal_attention_gain": 0.36,
            "b38_confidence_attention_gain": 0.32,
            "b38_attention_threshold": 0.13,
            "b38_abort_threshold": -0.17,
            "b38_attention_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "factor_attention"
        )
        if profile == "interoceptive_attention":
            defaults.update(
                {"b38_internal_attention_gain": 0.42, "b38_attention_lock_ticks": 6.0}
            )
        elif profile == "confidence_attention":
            defaults.update(
                {"b38_confidence_attention_gain": 0.42, "b38_attention_threshold": 0.11}
            )
        elif profile == "factor_attention_h56":
            defaults.update({"b38_attention_decay": 0.89, "b38_attention_lock_ticks": 6.0})
        elif profile == "genetic_factor_attention":
            defaults.update(
                {"b38_external_attention_gain": 0.36, "b38_internal_attention_gain": 0.38}
            )
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b38_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b38_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b38_external_attention = 0.0
        self._b38_internal_attention = 0.0
        self._b38_attention_gain = 0.0
        self._b38_attention_lock = 0
        self._b38_last_tick = int(tick)

    def _b38_factor_attention_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b37_state_factor_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b38_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "factor_attention"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b38_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        external_factor = float(trace_payload.get("b37_external_state_factor", 0.0) or 0.0)
        internal_factor = float(trace_payload.get("b37_internal_state_factor", 0.0) or 0.0)
        factor_alignment = float(trace_payload.get("b37_factor_alignment", 0.0) or 0.0)
        factor_confidence = float(trace_payload.get("b37_factor_confidence", 0.0) or 0.0)
        b37_decision = str(trace_payload.get("b37_decision", "preserve_b36"))

        decay = float(params["b38_attention_decay"])
        previous_external = float(getattr(self, "_b38_external_attention", 0.0))
        previous_internal = float(getattr(self, "_b38_internal_attention", 0.0))
        previous_gain = float(getattr(self, "_b38_attention_gain", 0.0))
        external_attention = float(
            np.clip(
                previous_external * decay
                + abs(external_factor) * float(params["b38_external_attention_gain"]),
                -1.0,
                1.0,
            )
        )
        internal_attention = float(
            np.clip(
                previous_internal * decay
                + max(0.0, internal_factor) * float(params["b38_internal_attention_gain"]),
                -1.0,
                1.0,
            )
        )
        attention_balance = float(
            np.clip(
                (internal_attention - external_attention) * 0.32
                + factor_alignment * 0.42
                + factor_confidence * 0.26,
                -1.0,
                1.0,
            )
        )
        attention_gain = float(
            np.clip(
                previous_gain * decay
                + max(0.0, factor_confidence)
                * float(params["b38_confidence_attention_gain"])
                + max(0.0, attention_balance) * 0.24
                - max(0.0, -attention_balance) * 0.18,
                -1.0,
                1.0,
            )
        )
        attention_score = float(
            np.clip(attention_balance * 0.46 + attention_gain * 0.34 + internal_attention * 0.20, -1.0, 1.0)
        )
        attention_lock = int(getattr(self, "_b38_attention_lock", 0))
        decision_label = "preserve_b37"

        if corridor_map:
            if (
                b37_decision in {"state_factor_commit", "continue_factor_lock"}
                and attention_score >= float(params["b38_attention_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                attention_lock = max(
                    attention_lock,
                    int(params["b38_attention_lock_ticks"]),
                )
                decision_label = "factor_attention_commit"
                reason = "b38_factor_attention_commit"
            elif attention_score <= float(params["b38_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "factor_attention_abort"
                reason = "b38_factor_attention_abort"
            elif attention_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_attention_lock"
                reason = "b38_continue_attention_lock"

        trace_payload.update(
            {
                "b38_controller_profile": profile,
                "b38_external_attention": round(float(external_attention), 6),
                "b38_internal_attention": round(float(internal_attention), 6),
                "b38_attention_balance": round(float(attention_balance), 6),
                "b38_attention_gain": round(float(attention_gain), 6),
                "b38_attention_lock": int(attention_lock),
                "b38_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b38_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b38_genetic_candidate"] = int(params["ga_candidate"])

        self._b38_external_attention = float(external_attention)
        self._b38_internal_attention = float(internal_attention)
        self._b38_attention_gain = float(attention_gain)
        self._b38_attention_lock = max(0, int(attention_lock) - 1)
        self._b38_last_tick = int(tick)
        return (
            semantic_action,
            B38_FACTOR_ATTENTION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b39_controller_params(self) -> dict[str, float]:
        params = self._b38_controller_params()
        defaults = {
            "b39_binding_decay": 0.86,
            "b39_external_binding_gain": 0.32,
            "b39_internal_binding_gain": 0.34,
            "b39_context_binding_gain": 0.36,
            "b39_binding_threshold": 0.12,
            "b39_abort_threshold": -0.18,
            "b39_binding_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "attention_binding"
        )
        if profile == "cross_factor_binding":
            defaults.update(
                {"b39_external_binding_gain": 0.38, "b39_internal_binding_gain": 0.38}
            )
        elif profile == "context_binding_attention":
            defaults.update(
                {"b39_context_binding_gain": 0.44, "b39_binding_threshold": 0.10}
            )
        elif profile == "attention_binding_h56":
            defaults.update({"b39_binding_decay": 0.89, "b39_binding_lock_ticks": 6.0})
        elif profile == "genetic_attention_binding":
            defaults.update(
                {"b39_internal_binding_gain": 0.38, "b39_context_binding_gain": 0.40}
            )
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b39_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b39_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b39_binding_strength = 0.0
        self._b39_bound_context = 0.0
        self._b39_binding_gain = 0.0
        self._b39_binding_lock = 0
        self._b39_last_tick = int(tick)

    def _b39_attention_binding_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b38_factor_attention_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b39_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "attention_binding"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b39_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        external_attention = float(trace_payload.get("b38_external_attention", 0.0) or 0.0)
        internal_attention = float(trace_payload.get("b38_internal_attention", 0.0) or 0.0)
        attention_balance = float(trace_payload.get("b38_attention_balance", 0.0) or 0.0)
        attention_gain = float(trace_payload.get("b38_attention_gain", 0.0) or 0.0)
        b38_decision = str(trace_payload.get("b38_decision", "preserve_b37"))

        decay = float(params["b39_binding_decay"])
        previous_strength = float(getattr(self, "_b39_binding_strength", 0.0))
        previous_context = float(getattr(self, "_b39_bound_context", 0.0))
        previous_gain = float(getattr(self, "_b39_binding_gain", 0.0))
        cross_factor_coherence = float(
            np.clip(
                (external_attention + internal_attention) * 0.5
                - abs(external_attention - internal_attention) * 0.24
                + attention_balance * 0.28,
                -1.0,
                1.0,
            )
        )
        binding_strength = float(
            np.clip(
                previous_strength * decay
                + max(0.0, external_attention)
                * float(params["b39_external_binding_gain"])
                + max(0.0, internal_attention)
                * float(params["b39_internal_binding_gain"])
                + max(0.0, cross_factor_coherence) * 0.18,
                -1.0,
                1.0,
            )
        )
        bound_context = float(
            np.clip(
                previous_context * decay
                + binding_strength * float(params["b39_context_binding_gain"])
                + attention_balance * 0.22,
                -1.0,
                1.0,
            )
        )
        binding_gain = float(
            np.clip(
                previous_gain * decay
                + max(0.0, attention_gain) * 0.32
                + max(0.0, bound_context) * 0.28
                - max(0.0, -bound_context) * 0.18,
                -1.0,
                1.0,
            )
        )
        binding_score = float(
            np.clip(
                binding_strength * 0.36
                + cross_factor_coherence * 0.28
                + bound_context * 0.20
                + binding_gain * 0.16,
                -1.0,
                1.0,
            )
        )
        binding_lock = int(getattr(self, "_b39_binding_lock", 0))
        decision_label = "preserve_b38"

        if corridor_map:
            if (
                b38_decision in {"factor_attention_commit", "continue_attention_lock"}
                and binding_score >= float(params["b39_binding_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                binding_lock = max(binding_lock, int(params["b39_binding_lock_ticks"]))
                decision_label = "attention_binding_commit"
                reason = "b39_attention_binding_commit"
            elif binding_score <= float(params["b39_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "attention_binding_abort"
                reason = "b39_attention_binding_abort"
            elif binding_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_binding_lock"
                reason = "b39_continue_binding_lock"

        trace_payload.update(
            {
                "b39_controller_profile": profile,
                "b39_binding_strength": round(float(binding_strength), 6),
                "b39_cross_factor_coherence": round(float(cross_factor_coherence), 6),
                "b39_bound_context": round(float(bound_context), 6),
                "b39_binding_gain": round(float(binding_gain), 6),
                "b39_binding_lock": int(binding_lock),
                "b39_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b39_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b39_genetic_candidate"] = int(params["ga_candidate"])

        self._b39_binding_strength = float(binding_strength)
        self._b39_bound_context = float(bound_context)
        self._b39_binding_gain = float(binding_gain)
        self._b39_binding_lock = max(0, int(binding_lock) - 1)
        self._b39_last_tick = int(tick)
        return (
            semantic_action,
            B39_ATTENTION_BINDING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b40_controller_params(self) -> dict[str, float]:
        params = self._b39_controller_params()
        defaults = {
            "b40_workspace_decay": 0.86,
            "b40_activation_gain": 0.34,
            "b40_broadcast_gain": 0.32,
            "b40_context_gain": 0.36,
            "b40_workspace_threshold": 0.12,
            "b40_abort_threshold": -0.18,
            "b40_workspace_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "global_workspace"
        )
        if profile == "sensory_workspace":
            defaults.update({"b40_activation_gain": 0.40, "b40_broadcast_gain": 0.36})
        elif profile == "context_workspace":
            defaults.update({"b40_context_gain": 0.44, "b40_workspace_threshold": 0.10})
        elif profile == "global_workspace_h56":
            defaults.update({"b40_workspace_decay": 0.89, "b40_workspace_lock_ticks": 6.0})
        elif profile == "genetic_global_workspace":
            defaults.update({"b40_activation_gain": 0.38, "b40_context_gain": 0.40})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b40_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b40_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b40_workspace_activation = 0.0
        self._b40_context_availability = 0.0
        self._b40_workspace_stability = 0.0
        self._b40_workspace_lock = 0
        self._b40_last_tick = int(tick)

    def _b40_global_workspace_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b39_attention_binding_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b40_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "global_workspace"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b40_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        binding_strength = float(trace_payload.get("b39_binding_strength", 0.0) or 0.0)
        coherence = float(trace_payload.get("b39_cross_factor_coherence", 0.0) or 0.0)
        bound_context = float(trace_payload.get("b39_bound_context", 0.0) or 0.0)
        binding_gain = float(trace_payload.get("b39_binding_gain", 0.0) or 0.0)
        b39_decision = str(trace_payload.get("b39_decision", "preserve_b38"))

        decay = float(params["b40_workspace_decay"])
        previous_activation = float(getattr(self, "_b40_workspace_activation", 0.0))
        previous_context = float(getattr(self, "_b40_context_availability", 0.0))
        previous_stability = float(getattr(self, "_b40_workspace_stability", 0.0))
        workspace_activation = float(
            np.clip(
                previous_activation * decay
                + max(0.0, binding_strength) * float(params["b40_activation_gain"])
                + max(0.0, coherence) * 0.22,
                -1.0,
                1.0,
            )
        )
        broadcast_gain = float(
            np.clip(
                max(0.0, binding_gain) * float(params["b40_broadcast_gain"])
                + max(0.0, workspace_activation) * 0.26,
                -1.0,
                1.0,
            )
        )
        context_availability = float(
            np.clip(
                previous_context * decay
                + max(0.0, bound_context) * float(params["b40_context_gain"])
                + broadcast_gain * 0.22,
                -1.0,
                1.0,
            )
        )
        workspace_stability = float(
            np.clip(
                previous_stability * decay
                + workspace_activation * 0.30
                + context_availability * 0.26
                + broadcast_gain * 0.20
                - max(0.0, -coherence) * 0.18,
                -1.0,
                1.0,
            )
        )
        workspace_score = float(
            np.clip(
                workspace_activation * 0.34
                + broadcast_gain * 0.24
                + context_availability * 0.22
                + workspace_stability * 0.20,
                -1.0,
                1.0,
            )
        )
        workspace_lock = int(getattr(self, "_b40_workspace_lock", 0))
        decision_label = "preserve_b39"

        if corridor_map:
            if (
                b39_decision in {"attention_binding_commit", "continue_binding_lock"}
                and workspace_score >= float(params["b40_workspace_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                workspace_lock = max(
                    workspace_lock,
                    int(params["b40_workspace_lock_ticks"]),
                )
                decision_label = "global_workspace_commit"
                reason = "b40_global_workspace_commit"
            elif workspace_score <= float(params["b40_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "global_workspace_abort"
                reason = "b40_global_workspace_abort"
            elif workspace_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_workspace_lock"
                reason = "b40_continue_workspace_lock"

        trace_payload.update(
            {
                "b40_controller_profile": profile,
                "b40_workspace_activation": round(float(workspace_activation), 6),
                "b40_broadcast_gain": round(float(broadcast_gain), 6),
                "b40_context_availability": round(float(context_availability), 6),
                "b40_workspace_stability": round(float(workspace_stability), 6),
                "b40_workspace_lock": int(workspace_lock),
                "b40_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b40_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b40_genetic_candidate"] = int(params["ga_candidate"])

        self._b40_workspace_activation = float(workspace_activation)
        self._b40_context_availability = float(context_availability)
        self._b40_workspace_stability = float(workspace_stability)
        self._b40_workspace_lock = max(0, int(workspace_lock) - 1)
        self._b40_last_tick = int(tick)
        return (
            semantic_action,
            B40_GLOBAL_WORKSPACE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b41_controller_params(self) -> dict[str, float]:
        params = self._b40_controller_params()
        defaults = {
            "b41_executive_decay": 0.86,
            "b41_selection_gain": 0.34,
            "b41_inhibition_gain": 0.30,
            "b41_goal_context_gain": 0.32,
            "b41_selection_threshold": 0.12,
            "b41_abort_threshold": -0.18,
            "b41_executive_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "executive_workspace"
        )
        if profile == "inhibitory_control":
            defaults.update({"b41_inhibition_gain": 0.38, "b41_abort_threshold": -0.16})
        elif profile == "goal_context_selector":
            defaults.update(
                {"b41_goal_context_gain": 0.40, "b41_selection_threshold": 0.10}
            )
        elif profile == "executive_workspace_h56":
            defaults.update({"b41_executive_decay": 0.89, "b41_executive_lock_ticks": 6.0})
        elif profile == "genetic_executive_workspace":
            defaults.update({"b41_selection_gain": 0.38, "b41_goal_context_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b41_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b41_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b41_executive_selection = 0.0
        self._b41_goal_context = 0.0
        self._b41_executive_stability = 0.0
        self._b41_executive_lock = 0
        self._b41_last_tick = int(tick)

    def _b41_executive_workspace_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b40_global_workspace_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b41_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "executive_workspace"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b41_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        workspace_activation = float(
            trace_payload.get("b40_workspace_activation", 0.0) or 0.0
        )
        broadcast_gain = float(trace_payload.get("b40_broadcast_gain", 0.0) or 0.0)
        context_availability = float(
            trace_payload.get("b40_context_availability", 0.0) or 0.0
        )
        workspace_stability = float(
            trace_payload.get("b40_workspace_stability", 0.0) or 0.0
        )
        b40_decision = str(trace_payload.get("b40_decision", "preserve_b39"))

        decay = float(params["b41_executive_decay"])
        previous_selection = float(getattr(self, "_b41_executive_selection", 0.0))
        previous_context = float(getattr(self, "_b41_goal_context", 0.0))
        previous_stability = float(getattr(self, "_b41_executive_stability", 0.0))
        executive_selection = float(
            np.clip(
                previous_selection * decay
                + max(0.0, workspace_activation) * float(params["b41_selection_gain"])
                + max(0.0, broadcast_gain) * 0.24,
                -1.0,
                1.0,
            )
        )
        inhibitory_pressure = float(
            np.clip(
                max(0.0, -workspace_stability) * float(params["b41_inhibition_gain"])
                + max(0.0, -context_availability) * 0.22,
                0.0,
                1.0,
            )
        )
        goal_context = float(
            np.clip(
                previous_context * decay
                + max(0.0, context_availability) * float(params["b41_goal_context_gain"])
                + executive_selection * 0.22
                - inhibitory_pressure * 0.18,
                -1.0,
                1.0,
            )
        )
        executive_stability = float(
            np.clip(
                previous_stability * decay
                + executive_selection * 0.30
                + goal_context * 0.26
                + workspace_stability * 0.20
                - inhibitory_pressure * 0.20,
                -1.0,
                1.0,
            )
        )
        executive_score = float(
            np.clip(
                executive_selection * 0.34
                + goal_context * 0.26
                + executive_stability * 0.24
                - inhibitory_pressure * 0.16,
                -1.0,
                1.0,
            )
        )
        executive_lock = int(getattr(self, "_b41_executive_lock", 0))
        decision_label = "preserve_b40"

        if corridor_map:
            if (
                b40_decision in {"global_workspace_commit", "continue_workspace_lock"}
                and executive_score >= float(params["b41_selection_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                executive_lock = max(
                    executive_lock,
                    int(params["b41_executive_lock_ticks"]),
                )
                decision_label = "executive_workspace_select"
                reason = "b41_executive_workspace_select"
            elif executive_score <= float(params["b41_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "executive_workspace_abort"
                reason = "b41_executive_workspace_abort"
            elif executive_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_executive_lock"
                reason = "b41_continue_executive_lock"

        trace_payload.update(
            {
                "b41_controller_profile": profile,
                "b41_executive_selection": round(float(executive_selection), 6),
                "b41_inhibitory_pressure": round(float(inhibitory_pressure), 6),
                "b41_goal_context": round(float(goal_context), 6),
                "b41_executive_stability": round(float(executive_stability), 6),
                "b41_executive_lock": int(executive_lock),
                "b41_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b41_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b41_genetic_candidate"] = int(params["ga_candidate"])

        self._b41_executive_selection = float(executive_selection)
        self._b41_goal_context = float(goal_context)
        self._b41_executive_stability = float(executive_stability)
        self._b41_executive_lock = max(0, int(executive_lock) - 1)
        self._b41_last_tick = int(tick)
        return (
            semantic_action,
            B41_EXECUTIVE_WORKSPACE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b42_controller_params(self) -> dict[str, float]:
        params = self._b41_controller_params()
        defaults = {
            "b42_monitor_decay": 0.86,
            "b42_error_gain": 0.34,
            "b42_conflict_gain": 0.30,
            "b42_performance_gain": 0.32,
            "b42_commit_threshold": 0.12,
            "b42_abort_threshold": -0.18,
            "b42_monitor_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "error_monitor")
        if profile == "conflict_monitor":
            defaults.update({"b42_conflict_gain": 0.38, "b42_abort_threshold": -0.16})
        elif profile == "performance_monitor":
            defaults.update(
                {"b42_performance_gain": 0.40, "b42_commit_threshold": 0.10}
            )
        elif profile == "error_monitor_h56":
            defaults.update({"b42_monitor_decay": 0.89, "b42_monitor_lock_ticks": 6.0})
        elif profile == "genetic_error_monitor":
            defaults.update({"b42_error_gain": 0.38, "b42_performance_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b42_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b42_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b42_error_signal = 0.0
        self._b42_performance_context = 0.0
        self._b42_monitor_stability = 0.0
        self._b42_monitor_lock = 0
        self._b42_last_tick = int(tick)

    def _b42_error_monitor_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b41_executive_workspace_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b42_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "error_monitor")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b42_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        executive_selection = float(
            trace_payload.get("b41_executive_selection", 0.0) or 0.0
        )
        inhibitory_pressure = float(
            trace_payload.get("b41_inhibitory_pressure", 0.0) or 0.0
        )
        goal_context = float(trace_payload.get("b41_goal_context", 0.0) or 0.0)
        executive_stability = float(
            trace_payload.get("b41_executive_stability", 0.0) or 0.0
        )
        b41_decision = str(trace_payload.get("b41_decision", "preserve_b40"))

        decay = float(params["b42_monitor_decay"])
        previous_error = float(getattr(self, "_b42_error_signal", 0.0))
        previous_context = float(getattr(self, "_b42_performance_context", 0.0))
        previous_stability = float(getattr(self, "_b42_monitor_stability", 0.0))
        error_signal = float(
            np.clip(
                previous_error * decay
                + max(0.0, 1.0 - executive_stability) * float(params["b42_error_gain"]) * 0.20
                + inhibitory_pressure * 0.18,
                0.0,
                1.0,
            )
        )
        conflict_signal = float(
            np.clip(
                inhibitory_pressure * float(params["b42_conflict_gain"])
                + abs(executive_selection - goal_context) * 0.18,
                0.0,
                1.0,
            )
        )
        performance_context = float(
            np.clip(
                previous_context * decay
                + max(0.0, goal_context) * float(params["b42_performance_gain"])
                + max(0.0, executive_selection) * 0.24
                - conflict_signal * 0.12,
                -1.0,
                1.0,
            )
        )
        monitor_stability = float(
            np.clip(
                previous_stability * decay
                + performance_context * 0.28
                + executive_stability * 0.24
                - error_signal * 0.12
                - conflict_signal * 0.10,
                -1.0,
                1.0,
            )
        )
        monitor_score = float(
            np.clip(
                performance_context * 0.36
                + monitor_stability * 0.28
                + executive_selection * 0.24
                - error_signal * 0.06
                - conflict_signal * 0.06,
                -1.0,
                1.0,
            )
        )
        monitor_lock = int(getattr(self, "_b42_monitor_lock", 0))
        decision_label = "preserve_b41"

        if corridor_map:
            if (
                b41_decision
                in {"executive_workspace_select", "continue_executive_lock"}
                and monitor_score >= float(params["b42_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                monitor_lock = max(monitor_lock, int(params["b42_monitor_lock_ticks"]))
                decision_label = "error_monitor_commit"
                reason = "b42_error_monitor_commit"
            elif monitor_score <= float(params["b42_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "error_monitor_abort"
                reason = "b42_error_monitor_abort"
            elif monitor_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_monitor_lock"
                reason = "b42_continue_monitor_lock"

        trace_payload.update(
            {
                "b42_controller_profile": profile,
                "b42_error_signal": round(float(error_signal), 6),
                "b42_conflict_signal": round(float(conflict_signal), 6),
                "b42_performance_context": round(float(performance_context), 6),
                "b42_monitor_stability": round(float(monitor_stability), 6),
                "b42_monitor_lock": int(monitor_lock),
                "b42_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b42_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b42_genetic_candidate"] = int(params["ga_candidate"])

        self._b42_error_signal = float(error_signal)
        self._b42_performance_context = float(performance_context)
        self._b42_monitor_stability = float(monitor_stability)
        self._b42_monitor_lock = max(0, int(monitor_lock) - 1)
        self._b42_last_tick = int(tick)
        return (
            semantic_action,
            B42_ERROR_MONITOR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b43_controller_params(self) -> dict[str, float]:
        params = self._b42_controller_params()
        defaults = {
            "b43_precision_decay": 0.86,
            "b43_precision_gain": 0.34,
            "b43_arousal_gain": 0.30,
            "b43_threshold_gain": 0.32,
            "b43_commit_threshold": 0.12,
            "b43_abort_threshold": -0.18,
            "b43_precision_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "adaptive_precision"
        )
        if profile == "arousal_precision":
            defaults.update({"b43_arousal_gain": 0.38, "b43_abort_threshold": -0.16})
        elif profile == "threshold_adaptation":
            defaults.update({"b43_threshold_gain": 0.40, "b43_commit_threshold": 0.10})
        elif profile == "adaptive_precision_h56":
            defaults.update({"b43_precision_decay": 0.89, "b43_precision_lock_ticks": 6.0})
        elif profile == "genetic_adaptive_precision":
            defaults.update({"b43_precision_gain": 0.38, "b43_threshold_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b43_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b43_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b43_precision_signal = 0.0
        self._b43_arousal_context = 0.0
        self._b43_control_stability = 0.0
        self._b43_precision_lock = 0
        self._b43_last_tick = int(tick)

    def _b43_adaptive_precision_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b42_error_monitor_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b43_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "adaptive_precision"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b43_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        error_signal = float(trace_payload.get("b42_error_signal", 0.0) or 0.0)
        conflict_signal = float(trace_payload.get("b42_conflict_signal", 0.0) or 0.0)
        performance_context = float(
            trace_payload.get("b42_performance_context", 0.0) or 0.0
        )
        monitor_stability = float(
            trace_payload.get("b42_monitor_stability", 0.0) or 0.0
        )
        b42_decision = str(trace_payload.get("b42_decision", "preserve_b41"))

        decay = float(params["b43_precision_decay"])
        previous_precision = float(getattr(self, "_b43_precision_signal", 0.0))
        previous_arousal = float(getattr(self, "_b43_arousal_context", 0.0))
        previous_stability = float(getattr(self, "_b43_control_stability", 0.0))
        adaptive_threshold = float(
            np.clip(
                float(params["b43_commit_threshold"])
                + conflict_signal * float(params["b43_threshold_gain"]) * 0.10
                + error_signal * 0.08
                - max(0.0, performance_context) * 0.06,
                -0.25,
                0.45,
            )
        )
        precision_signal = float(
            np.clip(
                previous_precision * decay
                + max(0.0, performance_context) * float(params["b43_precision_gain"])
                + max(0.0, monitor_stability) * 0.22
                - error_signal * 0.08
                - conflict_signal * 0.06,
                -1.0,
                1.0,
            )
        )
        arousal_context = float(
            np.clip(
                previous_arousal * decay
                + (error_signal + conflict_signal) * float(params["b43_arousal_gain"]) * 0.20
                + max(0.0, precision_signal) * 0.18,
                0.0,
                1.0,
            )
        )
        control_stability = float(
            np.clip(
                previous_stability * decay
                + precision_signal * 0.30
                + monitor_stability * 0.24
                + performance_context * 0.20
                - arousal_context * 0.08,
                -1.0,
                1.0,
            )
        )
        precision_score = float(
            np.clip(
                precision_signal * 0.36
                + control_stability * 0.28
                + performance_context * 0.24
                - adaptive_threshold * 0.08
                - arousal_context * 0.04,
                -1.0,
                1.0,
            )
        )
        precision_lock = int(getattr(self, "_b43_precision_lock", 0))
        decision_label = "preserve_b42"

        if corridor_map:
            if (
                b42_decision in {"error_monitor_commit", "continue_monitor_lock"}
                and precision_score >= adaptive_threshold
            ):
                semantic_action = "MOVE_TO_FOOD"
                precision_lock = max(
                    precision_lock,
                    int(params["b43_precision_lock_ticks"]),
                )
                decision_label = "adaptive_precision_commit"
                reason = "b43_adaptive_precision_commit"
            elif precision_score <= float(params["b43_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "adaptive_precision_abort"
                reason = "b43_adaptive_precision_abort"
            elif precision_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_precision_lock"
                reason = "b43_continue_precision_lock"

        trace_payload.update(
            {
                "b43_controller_profile": profile,
                "b43_precision_signal": round(float(precision_signal), 6),
                "b43_adaptive_threshold": round(float(adaptive_threshold), 6),
                "b43_arousal_context": round(float(arousal_context), 6),
                "b43_control_stability": round(float(control_stability), 6),
                "b43_precision_lock": int(precision_lock),
                "b43_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b43_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b43_genetic_candidate"] = int(params["ga_candidate"])

        self._b43_precision_signal = float(precision_signal)
        self._b43_arousal_context = float(arousal_context)
        self._b43_control_stability = float(control_stability)
        self._b43_precision_lock = max(0, int(precision_lock) - 1)
        self._b43_last_tick = int(tick)
        return (
            semantic_action,
            B43_ADAPTIVE_PRECISION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b44_controller_params(self) -> dict[str, float]:
        params = self._b43_controller_params()
        defaults = {
            "b44_relay_decay": 0.86,
            "b44_gate_gain": 0.34,
            "b44_sensory_gain": 0.30,
            "b44_context_gain": 0.32,
            "b44_relay_threshold": 0.12,
            "b44_abort_threshold": -0.18,
            "b44_relay_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "thalamic_relay"
        )
        if profile == "sensory_relay":
            defaults.update({"b44_sensory_gain": 0.38, "b44_abort_threshold": -0.16})
        elif profile == "context_relay":
            defaults.update({"b44_context_gain": 0.40, "b44_relay_threshold": 0.10})
        elif profile == "thalamic_relay_h56":
            defaults.update({"b44_relay_decay": 0.89, "b44_relay_lock_ticks": 6.0})
        elif profile == "genetic_thalamic_relay":
            defaults.update({"b44_gate_gain": 0.38, "b44_context_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b44_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b44_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b44_relay_gate = 0.0
        self._b44_context_relay = 0.0
        self._b44_gate_stability = 0.0
        self._b44_relay_lock = 0
        self._b44_last_tick = int(tick)

    def _b44_thalamic_relay_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b43_adaptive_precision_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b44_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "thalamic_relay")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b44_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        precision_signal = float(trace_payload.get("b43_precision_signal", 0.0) or 0.0)
        adaptive_threshold = float(
            trace_payload.get("b43_adaptive_threshold", 0.0) or 0.0
        )
        arousal_context = float(trace_payload.get("b43_arousal_context", 0.0) or 0.0)
        control_stability = float(
            trace_payload.get("b43_control_stability", 0.0) or 0.0
        )
        b43_decision = str(trace_payload.get("b43_decision", "preserve_b42"))

        decay = float(params["b44_relay_decay"])
        previous_gate = float(getattr(self, "_b44_relay_gate", 0.0))
        previous_context = float(getattr(self, "_b44_context_relay", 0.0))
        previous_stability = float(getattr(self, "_b44_gate_stability", 0.0))
        sensory_precision = float(
            np.clip(
                max(0.0, precision_signal) * float(params["b44_sensory_gain"])
                + max(0.0, control_stability) * 0.24
                - max(0.0, arousal_context - 0.35) * 0.08,
                0.0,
                1.0,
            )
        )
        context_relay = float(
            np.clip(
                previous_context * decay
                + max(0.0, control_stability) * float(params["b44_context_gain"])
                + max(0.0, precision_signal - adaptive_threshold) * 0.22,
                -1.0,
                1.0,
            )
        )
        relay_gate = float(
            np.clip(
                previous_gate * decay
                + sensory_precision * float(params["b44_gate_gain"])
                + context_relay * 0.26
                - adaptive_threshold * 0.08,
                -1.0,
                1.0,
            )
        )
        gate_stability = float(
            np.clip(
                previous_stability * decay
                + relay_gate * 0.34
                + control_stability * 0.22
                + sensory_precision * 0.20
                - arousal_context * 0.06,
                -1.0,
                1.0,
            )
        )
        relay_score = float(
            np.clip(
                relay_gate * 0.34
                + gate_stability * 0.26
                + sensory_precision * 0.22
                + context_relay * 0.18,
                -1.0,
                1.0,
            )
        )
        relay_lock = int(getattr(self, "_b44_relay_lock", 0))
        decision_label = "preserve_b43"

        if corridor_map:
            if (
                b43_decision in {"adaptive_precision_commit", "continue_precision_lock"}
                and relay_score >= float(params["b44_relay_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                relay_lock = max(relay_lock, int(params["b44_relay_lock_ticks"]))
                decision_label = "thalamic_relay_commit"
                reason = "b44_thalamic_relay_commit"
            elif relay_score <= float(params["b44_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "thalamic_relay_abort"
                reason = "b44_thalamic_relay_abort"
            elif relay_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_relay_lock"
                reason = "b44_continue_relay_lock"

        trace_payload.update(
            {
                "b44_controller_profile": profile,
                "b44_relay_gate": round(float(relay_gate), 6),
                "b44_sensory_precision": round(float(sensory_precision), 6),
                "b44_context_relay": round(float(context_relay), 6),
                "b44_gate_stability": round(float(gate_stability), 6),
                "b44_relay_lock": int(relay_lock),
                "b44_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b44_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b44_genetic_candidate"] = int(params["ga_candidate"])

        self._b44_relay_gate = float(relay_gate)
        self._b44_context_relay = float(context_relay)
        self._b44_gate_stability = float(gate_stability)
        self._b44_relay_lock = max(0, int(relay_lock) - 1)
        self._b44_last_tick = int(tick)
        return (
            semantic_action,
            B44_THALAMIC_RELAY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b45_controller_params(self) -> dict[str, float]:
        params = self._b44_controller_params()
        defaults = {
            "b45_inhibition_decay": 0.86,
            "b45_inhibitory_gain": 0.34,
            "b45_sensory_filter_gain": 0.30,
            "b45_context_suppression_gain": 0.32,
            "b45_commit_threshold": 0.12,
            "b45_abort_threshold": -0.18,
            "b45_inhibition_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "reticular_inhibition"
        )
        if profile == "sensory_inhibition":
            defaults.update({"b45_sensory_filter_gain": 0.38, "b45_abort_threshold": -0.16})
        elif profile == "context_inhibition":
            defaults.update(
                {"b45_context_suppression_gain": 0.40, "b45_commit_threshold": 0.10}
            )
        elif profile == "reticular_inhibition_h56":
            defaults.update(
                {"b45_inhibition_decay": 0.89, "b45_inhibition_lock_ticks": 6.0}
            )
        elif profile == "genetic_reticular_inhibition":
            defaults.update({"b45_inhibitory_gain": 0.38, "b45_context_suppression_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b45_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b45_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b45_inhibitory_gate = 0.0
        self._b45_context_suppression = 0.0
        self._b45_loop_stability = 0.0
        self._b45_inhibition_lock = 0
        self._b45_last_tick = int(tick)

    def _b45_reticular_inhibition_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b44_thalamic_relay_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b45_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "reticular_inhibition"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b45_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        relay_gate = float(trace_payload.get("b44_relay_gate", 0.0) or 0.0)
        sensory_precision = float(trace_payload.get("b44_sensory_precision", 0.0) or 0.0)
        context_relay = float(trace_payload.get("b44_context_relay", 0.0) or 0.0)
        gate_stability = float(trace_payload.get("b44_gate_stability", 0.0) or 0.0)
        b44_decision = str(trace_payload.get("b44_decision", "preserve_b43"))

        decay = float(params["b45_inhibition_decay"])
        previous_gate = float(getattr(self, "_b45_inhibitory_gate", 0.0))
        previous_suppression = float(getattr(self, "_b45_context_suppression", 0.0))
        previous_stability = float(getattr(self, "_b45_loop_stability", 0.0))
        sensory_filter = float(
            np.clip(
                max(0.0, sensory_precision) * float(params["b45_sensory_filter_gain"])
                + max(0.0, relay_gate) * 0.22
                - max(0.0, -context_relay) * 0.08,
                0.0,
                1.0,
            )
        )
        context_suppression = float(
            np.clip(
                previous_suppression * decay
                + max(0.0, context_relay) * float(params["b45_context_suppression_gain"])
                + max(0.0, gate_stability) * 0.22,
                -1.0,
                1.0,
            )
        )
        inhibitory_gate = float(
            np.clip(
                previous_gate * decay
                + sensory_filter * float(params["b45_inhibitory_gain"])
                + context_suppression * 0.24
                + max(0.0, relay_gate) * 0.18,
                -1.0,
                1.0,
            )
        )
        loop_stability = float(
            np.clip(
                previous_stability * decay
                + inhibitory_gate * 0.32
                + gate_stability * 0.24
                + sensory_filter * 0.18
                - max(0.0, -context_suppression) * 0.06,
                -1.0,
                1.0,
            )
        )
        inhibition_score = float(
            np.clip(
                inhibitory_gate * 0.34
                + loop_stability * 0.28
                + sensory_filter * 0.20
                + context_suppression * 0.18,
                -1.0,
                1.0,
            )
        )
        inhibition_lock = int(getattr(self, "_b45_inhibition_lock", 0))
        decision_label = "preserve_b44"

        if corridor_map:
            if (
                b44_decision in {"thalamic_relay_commit", "continue_relay_lock"}
                and inhibition_score >= float(params["b45_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                inhibition_lock = max(
                    inhibition_lock,
                    int(params["b45_inhibition_lock_ticks"]),
                )
                decision_label = "reticular_inhibition_commit"
                reason = "b45_reticular_inhibition_commit"
            elif inhibition_score <= float(params["b45_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "reticular_inhibition_abort"
                reason = "b45_reticular_inhibition_abort"
            elif inhibition_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_inhibition_lock"
                reason = "b45_continue_inhibition_lock"

        trace_payload.update(
            {
                "b45_controller_profile": profile,
                "b45_inhibitory_gate": round(float(inhibitory_gate), 6),
                "b45_sensory_filter": round(float(sensory_filter), 6),
                "b45_context_suppression": round(float(context_suppression), 6),
                "b45_loop_stability": round(float(loop_stability), 6),
                "b45_inhibition_lock": int(inhibition_lock),
                "b45_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b45_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b45_genetic_candidate"] = int(params["ga_candidate"])

        self._b45_inhibitory_gate = float(inhibitory_gate)
        self._b45_context_suppression = float(context_suppression)
        self._b45_loop_stability = float(loop_stability)
        self._b45_inhibition_lock = max(0, int(inhibition_lock) - 1)
        self._b45_last_tick = int(tick)
        return (
            semantic_action,
            B45_RETICULAR_INHIBITION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b46_controller_params(self) -> dict[str, float]:
        params = self._b45_controller_params()
        defaults = {
            "b46_feedback_decay": 0.86,
            "b46_feedback_gain": 0.34,
            "b46_topdown_gain": 0.30,
            "b46_prediction_gain": 0.32,
            "b46_commit_threshold": 0.08,
            "b46_abort_threshold": -0.18,
            "b46_feedback_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "corticothalamic_feedback"
        )
        if profile == "feedback_gain":
            defaults.update({"b46_feedback_gain": 0.38, "b46_abort_threshold": -0.16})
        elif profile == "context_feedback":
            defaults.update({"b46_topdown_gain": 0.40, "b46_commit_threshold": 0.07})
        elif profile == "corticothalamic_feedback_h56":
            defaults.update({"b46_feedback_decay": 0.89, "b46_feedback_lock_ticks": 6.0})
        elif profile == "genetic_corticothalamic_feedback":
            defaults.update({"b46_feedback_gain": 0.38, "b46_prediction_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b46_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b46_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b46_feedback_gain = 0.0
        self._b46_topdown_context = 0.0
        self._b46_feedback_stability = 0.0
        self._b46_feedback_lock = 0
        self._b46_last_tick = int(tick)

    def _b46_corticothalamic_feedback_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b45_reticular_inhibition_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b46_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "corticothalamic_feedback"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b46_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        inhibitory_gate = float(trace_payload.get("b45_inhibitory_gate", 0.0) or 0.0)
        sensory_filter = float(trace_payload.get("b45_sensory_filter", 0.0) or 0.0)
        context_suppression = float(
            trace_payload.get("b45_context_suppression", 0.0) or 0.0
        )
        loop_stability = float(trace_payload.get("b45_loop_stability", 0.0) or 0.0)
        b45_decision = str(trace_payload.get("b45_decision", "preserve_b44"))

        decay = float(params["b46_feedback_decay"])
        previous_gain = float(getattr(self, "_b46_feedback_gain", 0.0))
        previous_context = float(getattr(self, "_b46_topdown_context", 0.0))
        previous_stability = float(getattr(self, "_b46_feedback_stability", 0.0))
        topdown_context = float(
            np.clip(
                previous_context * decay
                + max(0.0, context_suppression) * float(params["b46_topdown_gain"])
                + max(0.0, loop_stability) * 0.22,
                -1.0,
                1.0,
            )
        )
        prediction_match = float(
            np.clip(
                max(0.0, sensory_filter) * float(params["b46_prediction_gain"])
                + max(0.0, inhibitory_gate) * 0.24
                + max(0.0, topdown_context) * 0.18,
                0.0,
                1.0,
            )
        )
        feedback_gain = float(
            np.clip(
                previous_gain * decay
                + prediction_match * float(params["b46_feedback_gain"])
                + topdown_context * 0.24
                + max(0.0, loop_stability) * 0.18,
                -1.0,
                1.0,
            )
        )
        feedback_stability = float(
            np.clip(
                previous_stability * decay
                + feedback_gain * 0.34
                + prediction_match * 0.22
                + loop_stability * 0.24
                - max(0.0, -topdown_context) * 0.06,
                -1.0,
                1.0,
            )
        )
        feedback_score = float(
            np.clip(
                feedback_gain * 0.34
                + feedback_stability * 0.28
                + prediction_match * 0.20
                + topdown_context * 0.18,
                -1.0,
                1.0,
            )
        )
        feedback_lock = int(getattr(self, "_b46_feedback_lock", 0))
        decision_label = "preserve_b45"

        if corridor_map:
            if (
                b45_decision
                in {"reticular_inhibition_commit", "continue_inhibition_lock"}
                and feedback_score >= float(params["b46_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                feedback_lock = max(feedback_lock, int(params["b46_feedback_lock_ticks"]))
                decision_label = "corticothalamic_feedback_commit"
                reason = "b46_corticothalamic_feedback_commit"
            elif feedback_score <= float(params["b46_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "corticothalamic_feedback_abort"
                reason = "b46_corticothalamic_feedback_abort"
            elif feedback_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_feedback_lock"
                reason = "b46_continue_feedback_lock"

        trace_payload.update(
            {
                "b46_controller_profile": profile,
                "b46_feedback_gain": round(float(feedback_gain), 6),
                "b46_topdown_context": round(float(topdown_context), 6),
                "b46_prediction_match": round(float(prediction_match), 6),
                "b46_feedback_stability": round(float(feedback_stability), 6),
                "b46_feedback_lock": int(feedback_lock),
                "b46_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b46_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b46_genetic_candidate"] = int(params["ga_candidate"])

        self._b46_feedback_gain = float(feedback_gain)
        self._b46_topdown_context = float(topdown_context)
        self._b46_feedback_stability = float(feedback_stability)
        self._b46_feedback_lock = max(0, int(feedback_lock) - 1)
        self._b46_last_tick = int(tick)
        return (
            semantic_action,
            B46_CORTICOTHALAMIC_FEEDBACK_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b47_controller_params(self) -> dict[str, float]:
        params = self._b46_controller_params()
        defaults = {
            "b47_phase_decay": 0.86,
            "b47_phase_gain": 0.32,
            "b47_synchrony_gain": 0.34,
            "b47_coherence_gain": 0.30,
            "b47_commit_threshold": 0.08,
            "b47_abort_threshold": -0.18,
            "b47_phase_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "oscillatory_synchrony"
        )
        if profile == "phase_locking":
            defaults.update({"b47_phase_gain": 0.38, "b47_phase_lock_ticks": 6.0})
        elif profile == "coherence_gate":
            defaults.update({"b47_coherence_gain": 0.38, "b47_commit_threshold": 0.07})
        elif profile == "oscillatory_synchrony_h56":
            defaults.update({"b47_phase_decay": 0.89, "b47_phase_lock_ticks": 6.0})
        elif profile == "genetic_oscillatory_synchrony":
            defaults.update({"b47_synchrony_gain": 0.38, "b47_coherence_gain": 0.34})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b47_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b47_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b47_phase_alignment = 0.0
        self._b47_synchrony_gain = 0.0
        self._b47_cross_loop_coherence = 0.0
        self._b47_phase_lock = 0
        self._b47_last_tick = int(tick)

    def _b47_oscillatory_synchrony_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b46_corticothalamic_feedback_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b47_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "oscillatory_synchrony"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b47_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        feedback_gain = float(trace_payload.get("b46_feedback_gain", 0.0) or 0.0)
        topdown_context = float(trace_payload.get("b46_topdown_context", 0.0) or 0.0)
        prediction_match = float(trace_payload.get("b46_prediction_match", 0.0) or 0.0)
        feedback_stability = float(
            trace_payload.get("b46_feedback_stability", 0.0) or 0.0
        )
        b46_decision = str(trace_payload.get("b46_decision", "preserve_b45"))

        decay = float(params["b47_phase_decay"])
        previous_phase = float(getattr(self, "_b47_phase_alignment", 0.0))
        previous_synchrony = float(getattr(self, "_b47_synchrony_gain", 0.0))
        previous_coherence = float(getattr(self, "_b47_cross_loop_coherence", 0.0))
        phase_drive = float(np.cos(max(0, tick) * 0.37))
        phase_alignment = float(
            np.clip(
                previous_phase * decay
                + max(0.0, feedback_stability) * float(params["b47_phase_gain"])
                + max(0.0, prediction_match) * 0.22
                + max(0.0, phase_drive) * 0.04,
                -1.0,
                1.0,
            )
        )
        synchrony_gain = float(
            np.clip(
                previous_synchrony * decay
                + max(0.0, feedback_gain) * float(params["b47_synchrony_gain"])
                + max(0.0, phase_alignment) * 0.24,
                -1.0,
                1.0,
            )
        )
        cross_loop_coherence = float(
            np.clip(
                previous_coherence * decay
                + max(0.0, topdown_context) * float(params["b47_coherence_gain"])
                + max(0.0, synchrony_gain) * 0.26
                + max(0.0, prediction_match) * 0.18,
                -1.0,
                1.0,
            )
        )
        synchrony_score = float(
            np.clip(
                phase_alignment * 0.30
                + synchrony_gain * 0.32
                + cross_loop_coherence * 0.26
                + feedback_stability * 0.12,
                -1.0,
                1.0,
            )
        )
        phase_lock = int(getattr(self, "_b47_phase_lock", 0))
        decision_label = "preserve_b46"

        if corridor_map:
            if (
                b46_decision
                in {"corticothalamic_feedback_commit", "continue_feedback_lock"}
                and synchrony_score >= float(params["b47_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                phase_lock = max(phase_lock, int(params["b47_phase_lock_ticks"]))
                decision_label = "oscillatory_synchrony_commit"
                reason = "b47_oscillatory_synchrony_commit"
            elif synchrony_score <= float(params["b47_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "oscillatory_synchrony_abort"
                reason = "b47_oscillatory_synchrony_abort"
            elif phase_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_phase_lock"
                reason = "b47_continue_phase_lock"

        trace_payload.update(
            {
                "b47_controller_profile": profile,
                "b47_phase_alignment": round(float(phase_alignment), 6),
                "b47_synchrony_gain": round(float(synchrony_gain), 6),
                "b47_cross_loop_coherence": round(float(cross_loop_coherence), 6),
                "b47_phase_lock": int(phase_lock),
                "b47_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b47_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b47_genetic_candidate"] = int(params["ga_candidate"])

        self._b47_phase_alignment = float(phase_alignment)
        self._b47_synchrony_gain = float(synchrony_gain)
        self._b47_cross_loop_coherence = float(cross_loop_coherence)
        self._b47_phase_lock = max(0, int(phase_lock) - 1)
        self._b47_last_tick = int(tick)
        return (
            semantic_action,
            B47_OSCILLATORY_SYNCHRONY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b48_controller_params(self) -> dict[str, float]:
        params = self._b47_controller_params()
        defaults = {
            "b48_timing_decay": 0.86,
            "b48_error_gain": 0.30,
            "b48_prediction_gain": 0.34,
            "b48_corrective_gain": 0.32,
            "b48_commit_threshold": 0.08,
            "b48_abort_threshold": -0.18,
            "b48_calibration_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cerebellar_timing"
        )
        if profile == "timing_error_correction":
            defaults.update({"b48_error_gain": 0.38, "b48_corrective_gain": 0.36})
        elif profile == "predictive_timing":
            defaults.update({"b48_prediction_gain": 0.40, "b48_commit_threshold": 0.07})
        elif profile == "cerebellar_timing_h56":
            defaults.update({"b48_timing_decay": 0.89, "b48_calibration_lock_ticks": 6.0})
        elif profile == "genetic_cerebellar_timing":
            defaults.update({"b48_prediction_gain": 0.38, "b48_corrective_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b48_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b48_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b48_timing_error = 0.0
        self._b48_predictive_timing = 0.0
        self._b48_corrective_gain = 0.0
        self._b48_calibration_lock = 0
        self._b48_last_tick = int(tick)

    def _b48_cerebellar_timing_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b47_oscillatory_synchrony_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b48_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cerebellar_timing"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b48_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        phase_alignment = float(trace_payload.get("b47_phase_alignment", 0.0) or 0.0)
        synchrony_gain = float(trace_payload.get("b47_synchrony_gain", 0.0) or 0.0)
        cross_loop_coherence = float(
            trace_payload.get("b47_cross_loop_coherence", 0.0) or 0.0
        )
        phase_lock = float(trace_payload.get("b47_phase_lock", 0.0) or 0.0)
        b47_decision = str(trace_payload.get("b47_decision", "preserve_b46"))

        decay = float(params["b48_timing_decay"])
        previous_error = float(getattr(self, "_b48_timing_error", 0.0))
        previous_prediction = float(getattr(self, "_b48_predictive_timing", 0.0))
        previous_correction = float(getattr(self, "_b48_corrective_gain", 0.0))
        target_phase = 0.55 + 0.08 * float(np.sin(max(0, tick) * 0.23))
        raw_error = abs(target_phase - max(0.0, phase_alignment))
        timing_error = float(
            np.clip(
                previous_error * decay
                + (1.0 - min(1.0, raw_error)) * float(params["b48_error_gain"])
                + max(0.0, cross_loop_coherence) * 0.18,
                -1.0,
                1.0,
            )
        )
        predictive_timing = float(
            np.clip(
                previous_prediction * decay
                + max(0.0, synchrony_gain) * float(params["b48_prediction_gain"])
                + max(0.0, phase_lock) * 0.04
                + max(0.0, timing_error) * 0.18,
                -1.0,
                1.0,
            )
        )
        corrective_gain = float(
            np.clip(
                previous_correction * decay
                + max(0.0, predictive_timing) * float(params["b48_corrective_gain"])
                + max(0.0, cross_loop_coherence) * 0.22
                + max(0.0, timing_error) * 0.16,
                -1.0,
                1.0,
            )
        )
        timing_score = float(
            np.clip(
                timing_error * 0.24
                + predictive_timing * 0.32
                + corrective_gain * 0.32
                + cross_loop_coherence * 0.12,
                -1.0,
                1.0,
            )
        )
        calibration_lock = int(getattr(self, "_b48_calibration_lock", 0))
        decision_label = "preserve_b47"

        if corridor_map:
            if (
                b47_decision
                in {"oscillatory_synchrony_commit", "continue_phase_lock"}
                and timing_score >= float(params["b48_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                calibration_lock = max(
                    calibration_lock,
                    int(params["b48_calibration_lock_ticks"]),
                )
                decision_label = "cerebellar_timing_commit"
                reason = "b48_cerebellar_timing_commit"
            elif timing_score <= float(params["b48_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "cerebellar_timing_abort"
                reason = "b48_cerebellar_timing_abort"
            elif calibration_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_calibration_lock"
                reason = "b48_continue_calibration_lock"

        trace_payload.update(
            {
                "b48_controller_profile": profile,
                "b48_timing_error": round(float(timing_error), 6),
                "b48_predictive_timing": round(float(predictive_timing), 6),
                "b48_corrective_gain": round(float(corrective_gain), 6),
                "b48_calibration_lock": int(calibration_lock),
                "b48_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b48_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b48_genetic_candidate"] = int(params["ga_candidate"])

        self._b48_timing_error = float(timing_error)
        self._b48_predictive_timing = float(predictive_timing)
        self._b48_corrective_gain = float(corrective_gain)
        self._b48_calibration_lock = max(0, int(calibration_lock) - 1)
        self._b48_last_tick = int(tick)
        return (
            semantic_action,
            B48_CEREBELLAR_TIMING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b49_controller_params(self) -> dict[str, float]:
        params = self._b48_controller_params()
        defaults = {
            "b49_gate_decay": 0.86,
            "b49_go_gain": 0.34,
            "b49_no_go_gain": 0.30,
            "b49_balance_gain": 0.32,
            "b49_commit_threshold": 0.08,
            "b49_abort_threshold": -0.18,
            "b49_selection_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "striatal_action_gate"
        )
        if profile == "direct_path_facilitation":
            defaults.update({"b49_go_gain": 0.42, "b49_commit_threshold": 0.06})
        elif profile == "indirect_path_suppression":
            defaults.update({"b49_no_go_gain": 0.40, "b49_abort_threshold": -0.14})
        elif profile == "striatal_action_gate_h56":
            defaults.update({"b49_gate_decay": 0.89, "b49_selection_lock_ticks": 6.0})
        elif profile == "genetic_striatal_gate":
            defaults.update({"b49_go_gain": 0.38, "b49_balance_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b49_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b49_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b49_go_signal = 0.0
        self._b49_no_go_signal = 0.0
        self._b49_action_gate_balance = 0.0
        self._b49_selection_lock = 0
        self._b49_last_tick = int(tick)

    def _b49_striatal_action_gate_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b48_cerebellar_timing_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b49_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "striatal_action_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b49_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        timing_error = float(trace_payload.get("b48_timing_error", 0.0) or 0.0)
        predictive_timing = float(trace_payload.get("b48_predictive_timing", 0.0) or 0.0)
        corrective_gain = float(trace_payload.get("b48_corrective_gain", 0.0) or 0.0)
        calibration_lock = float(trace_payload.get("b48_calibration_lock", 0.0) or 0.0)
        b48_decision = str(trace_payload.get("b48_decision", "preserve_b47"))

        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        hunger = self._b_series_float(meta, "hunger")
        decay = float(params["b49_gate_decay"])
        previous_go = float(getattr(self, "_b49_go_signal", 0.0))
        previous_no_go = float(getattr(self, "_b49_no_go_signal", 0.0))
        previous_balance = float(getattr(self, "_b49_action_gate_balance", 0.0))
        go_signal = float(
            np.clip(
                previous_go * decay
                + max(0.0, predictive_timing) * float(params["b49_go_gain"])
                + max(0.0, corrective_gain) * 0.22
                + max(0.0, calibration_lock) * 0.04
                + max(0.0, hunger - 0.45) * 0.10,
                -1.0,
                1.0,
            )
        )
        no_go_signal = float(
            np.clip(
                previous_no_go * decay
                + max(0.0, current_threat) * float(params["b49_no_go_gain"])
                + max(0.0, 0.35 - timing_error) * 0.12
                + 0.03,
                -1.0,
                1.0,
            )
        )
        action_gate_balance = float(
            np.clip(
                previous_balance * decay
                + (go_signal - no_go_signal) * float(params["b49_balance_gain"])
                + max(0.0, corrective_gain) * 0.18,
                -1.0,
                1.0,
            )
        )
        selection_lock = int(getattr(self, "_b49_selection_lock", 0))
        decision_label = "preserve_b48"

        if corridor_map:
            if (
                b48_decision
                in {"cerebellar_timing_commit", "continue_calibration_lock"}
                and action_gate_balance >= float(params["b49_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                selection_lock = max(
                    selection_lock,
                    int(params["b49_selection_lock_ticks"]),
                )
                decision_label = "striatal_gate_commit"
                reason = "b49_striatal_gate_commit"
            elif action_gate_balance <= float(params["b49_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "striatal_gate_abort"
                reason = "b49_striatal_gate_abort"
            elif selection_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_selection_lock"
                reason = "b49_continue_selection_lock"

        trace_payload.update(
            {
                "b49_controller_profile": profile,
                "b49_go_signal": round(float(go_signal), 6),
                "b49_no_go_signal": round(float(no_go_signal), 6),
                "b49_action_gate_balance": round(float(action_gate_balance), 6),
                "b49_selection_lock": int(selection_lock),
                "b49_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b49_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b49_genetic_candidate"] = int(params["ga_candidate"])

        self._b49_go_signal = float(go_signal)
        self._b49_no_go_signal = float(no_go_signal)
        self._b49_action_gate_balance = float(action_gate_balance)
        self._b49_selection_lock = max(0, int(selection_lock) - 1)
        self._b49_last_tick = int(tick)
        return (
            semantic_action,
            B49_STRIATAL_ACTION_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b50_controller_params(self) -> dict[str, float]:
        params = self._b49_controller_params()
        defaults = {
            "b50_habit_decay": 0.86,
            "b50_habit_gain": 0.34,
            "b50_chunk_value_gain": 0.30,
            "b50_stability_gain": 0.32,
            "b50_commit_threshold": 0.08,
            "b50_abort_threshold": -0.18,
            "b50_chunk_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "habit_chunking"
        )
        if profile == "action_chunk_value":
            defaults.update({"b50_chunk_value_gain": 0.40, "b50_commit_threshold": 0.07})
        elif profile == "habit_stability":
            defaults.update({"b50_stability_gain": 0.40, "b50_chunk_lock_ticks": 6.0})
        elif profile == "habit_chunking_h56":
            defaults.update({"b50_habit_decay": 0.89, "b50_chunk_lock_ticks": 6.0})
        elif profile == "genetic_habit_chunking":
            defaults.update({"b50_habit_gain": 0.38, "b50_stability_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b50_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b50_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b50_habit_strength = 0.0
        self._b50_chunk_value = 0.0
        self._b50_habit_stability = 0.0
        self._b50_chunk_lock = 0
        self._b50_last_tick = int(tick)

    def _b50_habit_chunking_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b49_striatal_action_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b50_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "habit_chunking"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b50_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        go_signal = float(trace_payload.get("b49_go_signal", 0.0) or 0.0)
        no_go_signal = float(trace_payload.get("b49_no_go_signal", 0.0) or 0.0)
        gate_balance = float(trace_payload.get("b49_action_gate_balance", 0.0) or 0.0)
        selection_lock = float(trace_payload.get("b49_selection_lock", 0.0) or 0.0)
        b49_decision = str(trace_payload.get("b49_decision", "preserve_b48"))

        decay = float(params["b50_habit_decay"])
        previous_habit = float(getattr(self, "_b50_habit_strength", 0.0))
        previous_chunk = float(getattr(self, "_b50_chunk_value", 0.0))
        previous_stability = float(getattr(self, "_b50_habit_stability", 0.0))
        habit_strength = float(
            np.clip(
                previous_habit * decay
                + max(0.0, go_signal - no_go_signal) * float(params["b50_habit_gain"])
                + max(0.0, selection_lock) * 0.04,
                -1.0,
                1.0,
            )
        )
        chunk_value = float(
            np.clip(
                previous_chunk * decay
                + max(0.0, gate_balance) * float(params["b50_chunk_value_gain"])
                + max(0.0, habit_strength) * 0.18,
                -1.0,
                1.0,
            )
        )
        habit_stability = float(
            np.clip(
                previous_stability * decay
                + max(0.0, chunk_value) * float(params["b50_stability_gain"])
                + max(0.0, habit_strength) * 0.18,
                -1.0,
                1.0,
            )
        )
        chunk_score = float(
            np.clip(
                habit_strength * 0.30
                + chunk_value * 0.34
                + habit_stability * 0.30
                - no_go_signal * 0.06,
                -1.0,
                1.0,
            )
        )
        chunk_lock = int(getattr(self, "_b50_chunk_lock", 0))
        decision_label = "preserve_b49"

        if corridor_map:
            if (
                b49_decision in {"striatal_gate_commit", "continue_selection_lock"}
                and chunk_score >= float(params["b50_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                chunk_lock = max(chunk_lock, int(params["b50_chunk_lock_ticks"]))
                decision_label = "habit_chunk_commit"
                reason = "b50_habit_chunk_commit"
            elif chunk_score <= float(params["b50_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "habit_chunk_abort"
                reason = "b50_habit_chunk_abort"
            elif chunk_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_habit_chunk"
                reason = "b50_continue_habit_chunk"

        trace_payload.update(
            {
                "b50_controller_profile": profile,
                "b50_habit_strength": round(float(habit_strength), 6),
                "b50_chunk_value": round(float(chunk_value), 6),
                "b50_habit_stability": round(float(habit_stability), 6),
                "b50_chunk_lock": int(chunk_lock),
                "b50_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b50_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b50_genetic_candidate"] = int(params["ga_candidate"])

        self._b50_habit_strength = float(habit_strength)
        self._b50_chunk_value = float(chunk_value)
        self._b50_habit_stability = float(habit_stability)
        self._b50_chunk_lock = max(0, int(chunk_lock) - 1)
        self._b50_last_tick = int(tick)
        return (
            semantic_action,
            B50_HABIT_CHUNKING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b51_controller_params(self) -> dict[str, float]:
        params = self._b50_controller_params()
        defaults = {
            "b51_dopamine_decay": 0.86,
            "b51_prediction_error_gain": 0.32,
            "b51_dopamine_gain": 0.34,
            "b51_habit_modulation_gain": 0.30,
            "b51_commit_threshold": 0.08,
            "b51_abort_threshold": -0.18,
            "b51_modulation_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "dopaminergic_habit_modulation"
        )
        if profile == "reward_prediction_gain":
            defaults.update({"b51_prediction_error_gain": 0.40, "b51_commit_threshold": 0.07})
        elif profile == "novelty_modulated_habit":
            defaults.update({"b51_dopamine_gain": 0.40, "b51_habit_modulation_gain": 0.34})
        elif profile == "dopaminergic_habit_modulation_h56":
            defaults.update({"b51_dopamine_decay": 0.89, "b51_modulation_lock_ticks": 6.0})
        elif profile == "genetic_dopamine_habit":
            defaults.update({"b51_dopamine_gain": 0.38, "b51_habit_modulation_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b51_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b51_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b51_prediction_error = 0.0
        self._b51_dopamine_gain = 0.0
        self._b51_habit_modulation = 0.0
        self._b51_modulation_lock = 0
        self._b51_last_tick = int(tick)

    def _b51_dopaminergic_habit_modulation_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b50_habit_chunking_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b51_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "dopaminergic_habit_modulation"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b51_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        habit_strength = float(trace_payload.get("b50_habit_strength", 0.0) or 0.0)
        chunk_value = float(trace_payload.get("b50_chunk_value", 0.0) or 0.0)
        habit_stability = float(trace_payload.get("b50_habit_stability", 0.0) or 0.0)
        chunk_lock = float(trace_payload.get("b50_chunk_lock", 0.0) or 0.0)
        b50_decision = str(trace_payload.get("b50_decision", "preserve_b49"))
        hunger = self._b_series_float(meta, "hunger")
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b51_dopamine_decay"])
        previous_error = float(getattr(self, "_b51_prediction_error", 0.0))
        previous_gain = float(getattr(self, "_b51_dopamine_gain", 0.0))
        previous_modulation = float(getattr(self, "_b51_habit_modulation", 0.0))
        expected_chunk_value = 0.42 + 0.18 * max(0.0, habit_stability)
        reward_proxy = (
            max(0.0, chunk_value)
            + max(0.0, hunger - 0.45) * 0.20
            - max(0.0, current_threat) * 0.14
        )
        prediction_error = float(
            np.clip(
                previous_error * decay
                + max(0.0, reward_proxy - expected_chunk_value)
                * float(params["b51_prediction_error_gain"])
                + max(0.0, chunk_lock) * 0.03,
                -1.0,
                1.0,
            )
        )
        dopamine_gain = float(
            np.clip(
                previous_gain * decay
                + max(0.0, prediction_error) * float(params["b51_dopamine_gain"])
                + max(0.0, habit_strength) * 0.16,
                -1.0,
                1.0,
            )
        )
        habit_modulation = float(
            np.clip(
                previous_modulation * decay
                + max(0.0, dopamine_gain) * float(params["b51_habit_modulation_gain"])
                + max(0.0, habit_stability) * 0.18,
                -1.0,
                1.0,
            )
        )
        modulation_score = float(
            np.clip(
                prediction_error * 0.24
                + dopamine_gain * 0.34
                + habit_modulation * 0.34
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        modulation_lock = int(getattr(self, "_b51_modulation_lock", 0))
        decision_label = "preserve_b50"

        if corridor_map:
            if (
                b50_decision in {"habit_chunk_commit", "continue_habit_chunk"}
                and modulation_score >= float(params["b51_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                modulation_lock = max(
                    modulation_lock,
                    int(params["b51_modulation_lock_ticks"]),
                )
                decision_label = "dopamine_habit_commit"
                reason = "b51_dopamine_habit_commit"
            elif modulation_score <= float(params["b51_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "dopamine_habit_abort"
                reason = "b51_dopamine_habit_abort"
            elif modulation_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_dopamine_modulation"
                reason = "b51_continue_dopamine_modulation"

        trace_payload.update(
            {
                "b51_controller_profile": profile,
                "b51_prediction_error": round(float(prediction_error), 6),
                "b51_dopamine_gain": round(float(dopamine_gain), 6),
                "b51_habit_modulation": round(float(habit_modulation), 6),
                "b51_modulation_lock": int(modulation_lock),
                "b51_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b51_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b51_genetic_candidate"] = int(params["ga_candidate"])

        self._b51_prediction_error = float(prediction_error)
        self._b51_dopamine_gain = float(dopamine_gain)
        self._b51_habit_modulation = float(habit_modulation)
        self._b51_modulation_lock = max(0, int(modulation_lock) - 1)
        self._b51_last_tick = int(tick)
        return (
            semantic_action,
            B51_DOPAMINERGIC_HABIT_MODULATION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b52_controller_params(self) -> dict[str, float]:
        params = self._b51_controller_params()
        defaults = {
            "b52_acetylcholine_decay": 0.86,
            "b52_uncertainty_gain": 0.30,
            "b52_precision_gain": 0.34,
            "b52_attention_gain": 0.32,
            "b52_commit_threshold": 0.08,
            "b52_abort_threshold": -0.20,
            "b52_attention_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cholinergic_precision_gate"
        )
        if profile == "attention_gain":
            defaults.update({"b52_attention_gain": 0.40, "b52_commit_threshold": 0.07})
        elif profile == "uncertainty_release":
            defaults.update({"b52_uncertainty_gain": 0.38, "b52_abort_threshold": -0.16})
        elif profile == "cholinergic_precision_gate_h56":
            defaults.update(
                {"b52_acetylcholine_decay": 0.89, "b52_attention_lock_ticks": 6.0}
            )
        elif profile == "genetic_cholinergic_precision":
            defaults.update({"b52_precision_gain": 0.38, "b52_attention_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b52_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b52_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b52_acetylcholine_level = 0.0
        self._b52_precision_gain = 0.0
        self._b52_uncertainty_signal = 0.0
        self._b52_attention_lock = 0
        self._b52_last_tick = int(tick)

    def _b52_cholinergic_precision_gate_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b51_dopaminergic_habit_modulation_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b52_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cholinergic_precision_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b52_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b51_decision = str(trace_payload.get("b51_decision", "preserve_b50"))
        prediction_error = float(trace_payload.get("b51_prediction_error", 0.0) or 0.0)
        dopamine_gain = float(trace_payload.get("b51_dopamine_gain", 0.0) or 0.0)
        habit_modulation = float(trace_payload.get("b51_habit_modulation", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b52_acetylcholine_decay"])
        previous_acetylcholine = float(
            getattr(self, "_b52_acetylcholine_level", 0.0)
        )
        previous_precision = float(getattr(self, "_b52_precision_gain", 0.0))
        previous_uncertainty = float(getattr(self, "_b52_uncertainty_signal", 0.0))
        uncertainty_signal = float(
            np.clip(
                previous_uncertainty * decay
                + max(0.0, abs(prediction_error)) * float(params["b52_uncertainty_gain"])
                + max(0.0, current_threat) * 0.16
                + max(0.0, 0.42 - habit_modulation) * 0.10,
                0.0,
                1.0,
            )
        )
        acetylcholine_level = float(
            np.clip(
                previous_acetylcholine * decay
                + uncertainty_signal * 0.34
                + max(0.0, dopamine_gain) * 0.16,
                0.0,
                1.0,
            )
        )
        precision_gain = float(
            np.clip(
                previous_precision * decay
                + acetylcholine_level * float(params["b52_precision_gain"])
                + max(0.0, habit_modulation) * 0.18,
                0.0,
                1.0,
            )
        )
        attention_score = float(
            np.clip(
                precision_gain * float(params["b52_attention_gain"])
                + max(0.0, dopamine_gain) * 0.22
                + max(0.0, habit_modulation) * 0.20
                - uncertainty_signal * 0.06
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        attention_lock = int(getattr(self, "_b52_attention_lock", 0))
        decision_label = "preserve_b51"

        if corridor_map:
            if (
                b51_decision
                in {"dopamine_habit_commit", "continue_dopamine_modulation"}
                and attention_score >= float(params["b52_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                attention_lock = max(
                    attention_lock,
                    int(params["b52_attention_lock_ticks"]),
                )
                decision_label = "cholinergic_precision_commit"
                reason = "b52_cholinergic_precision_commit"
            elif attention_score <= float(params["b52_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "cholinergic_precision_abort"
                reason = "b52_cholinergic_precision_abort"
            elif attention_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_precision_attention"
                reason = "b52_continue_precision_attention"

        trace_payload.update(
            {
                "b52_controller_profile": profile,
                "b52_acetylcholine_level": round(float(acetylcholine_level), 6),
                "b52_precision_gain": round(float(precision_gain), 6),
                "b52_uncertainty_signal": round(float(uncertainty_signal), 6),
                "b52_attention_lock": int(attention_lock),
                "b52_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b52_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b52_genetic_candidate"] = int(params["ga_candidate"])

        self._b52_acetylcholine_level = float(acetylcholine_level)
        self._b52_precision_gain = float(precision_gain)
        self._b52_uncertainty_signal = float(uncertainty_signal)
        self._b52_attention_lock = max(0, int(attention_lock) - 1)
        self._b52_last_tick = int(tick)
        return (
            semantic_action,
            B52_CHOLINERGIC_PRECISION_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b53_controller_params(self) -> dict[str, float]:
        params = self._b52_controller_params()
        defaults = {
            "b53_norepinephrine_decay": 0.86,
            "b53_surprise_gain": 0.30,
            "b53_arousal_gain": 0.34,
            "b53_precision_mod_gain": 0.32,
            "b53_commit_threshold": 0.08,
            "b53_abort_threshold": -0.20,
            "b53_gain_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "noradrenergic_arousal_gain"
        )
        if profile == "surprise_gain":
            defaults.update({"b53_surprise_gain": 0.40, "b53_commit_threshold": 0.07})
        elif profile == "stress_precision":
            defaults.update({"b53_arousal_gain": 0.40, "b53_abort_threshold": -0.16})
        elif profile == "noradrenergic_arousal_gain_h56":
            defaults.update({"b53_norepinephrine_decay": 0.89, "b53_gain_lock_ticks": 6.0})
        elif profile == "genetic_arousal_precision":
            defaults.update({"b53_arousal_gain": 0.38, "b53_precision_mod_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b53_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b53_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b53_norepinephrine_level = 0.0
        self._b53_arousal_gain = 0.0
        self._b53_surprise_signal = 0.0
        self._b53_gain_lock = 0
        self._b53_last_tick = int(tick)

    def _b53_noradrenergic_arousal_gain_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b52_cholinergic_precision_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b53_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "noradrenergic_arousal_gain"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b53_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b52_decision = str(trace_payload.get("b52_decision", "preserve_b51"))
        acetylcholine_level = float(
            trace_payload.get("b52_acetylcholine_level", 0.0) or 0.0
        )
        precision_gain = float(trace_payload.get("b52_precision_gain", 0.0) or 0.0)
        uncertainty_signal = float(
            trace_payload.get("b52_uncertainty_signal", 0.0) or 0.0
        )
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b53_norepinephrine_decay"])
        previous_ne = float(getattr(self, "_b53_norepinephrine_level", 0.0))
        previous_arousal = float(getattr(self, "_b53_arousal_gain", 0.0))
        previous_surprise = float(getattr(self, "_b53_surprise_signal", 0.0))
        surprise_signal = float(
            np.clip(
                previous_surprise * decay
                + max(0.0, uncertainty_signal) * float(params["b53_surprise_gain"])
                + max(0.0, current_threat) * 0.14
                + max(0.0, 0.36 - precision_gain) * 0.10,
                0.0,
                1.0,
            )
        )
        norepinephrine_level = float(
            np.clip(
                previous_ne * decay
                + surprise_signal * 0.32
                + max(0.0, acetylcholine_level) * 0.16,
                0.0,
                1.0,
            )
        )
        arousal_gain = float(
            np.clip(
                previous_arousal * decay
                + norepinephrine_level * float(params["b53_arousal_gain"])
                + max(0.0, precision_gain) * float(params["b53_precision_mod_gain"]),
                0.0,
                1.0,
            )
        )
        gain_score = float(
            np.clip(
                arousal_gain * 0.34
                + max(0.0, precision_gain) * 0.26
                + max(0.0, acetylcholine_level) * 0.18
                - surprise_signal * 0.04
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        gain_lock = int(getattr(self, "_b53_gain_lock", 0))
        decision_label = "preserve_b52"

        if corridor_map:
            if (
                b52_decision
                in {"cholinergic_precision_commit", "continue_precision_attention"}
                and gain_score >= float(params["b53_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                gain_lock = max(gain_lock, int(params["b53_gain_lock_ticks"]))
                decision_label = "noradrenergic_arousal_commit"
                reason = "b53_noradrenergic_arousal_commit"
            elif gain_score <= float(params["b53_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "noradrenergic_arousal_abort"
                reason = "b53_noradrenergic_arousal_abort"
            elif gain_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_arousal_gain"
                reason = "b53_continue_arousal_gain"

        trace_payload.update(
            {
                "b53_controller_profile": profile,
                "b53_norepinephrine_level": round(float(norepinephrine_level), 6),
                "b53_arousal_gain": round(float(arousal_gain), 6),
                "b53_surprise_signal": round(float(surprise_signal), 6),
                "b53_gain_lock": int(gain_lock),
                "b53_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b53_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b53_genetic_candidate"] = int(params["ga_candidate"])

        self._b53_norepinephrine_level = float(norepinephrine_level)
        self._b53_arousal_gain = float(arousal_gain)
        self._b53_surprise_signal = float(surprise_signal)
        self._b53_gain_lock = max(0, int(gain_lock) - 1)
        self._b53_last_tick = int(tick)
        return (
            semantic_action,
            B53_NORADRENERGIC_AROUSAL_GAIN_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b54_controller_params(self) -> dict[str, float]:
        params = self._b53_controller_params()
        defaults = {
            "b54_serotonin_decay": 0.86,
            "b54_patience_gain": 0.34,
            "b54_impulse_suppression_gain": 0.32,
            "b54_arousal_balance_gain": 0.30,
            "b54_commit_threshold": 0.08,
            "b54_abort_threshold": -0.20,
            "b54_patience_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "serotonergic_patience_gate"
        )
        if profile == "impulse_suppression":
            defaults.update({"b54_impulse_suppression_gain": 0.40, "b54_abort_threshold": -0.16})
        elif profile == "patience_balance":
            defaults.update({"b54_patience_gain": 0.40, "b54_commit_threshold": 0.07})
        elif profile == "serotonergic_patience_gate_h56":
            defaults.update({"b54_serotonin_decay": 0.89, "b54_patience_lock_ticks": 6.0})
        elif profile == "genetic_serotonin_patience":
            defaults.update({"b54_patience_gain": 0.38, "b54_arousal_balance_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b54_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b54_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b54_serotonin_level = 0.0
        self._b54_patience_signal = 0.0
        self._b54_impulse_suppression = 0.0
        self._b54_patience_lock = 0
        self._b54_last_tick = int(tick)

    def _b54_serotonergic_patience_gate_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b53_noradrenergic_arousal_gain_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b54_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "serotonergic_patience_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b54_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b53_decision = str(trace_payload.get("b53_decision", "preserve_b52"))
        norepinephrine_level = float(
            trace_payload.get("b53_norepinephrine_level", 0.0) or 0.0
        )
        arousal_gain = float(trace_payload.get("b53_arousal_gain", 0.0) or 0.0)
        surprise_signal = float(trace_payload.get("b53_surprise_signal", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b54_serotonin_decay"])
        previous_serotonin = float(getattr(self, "_b54_serotonin_level", 0.0))
        previous_patience = float(getattr(self, "_b54_patience_signal", 0.0))
        previous_suppression = float(
            getattr(self, "_b54_impulse_suppression", 0.0)
        )
        balanced_arousal = max(0.0, arousal_gain - surprise_signal * 0.25)
        serotonin_level = float(
            np.clip(
                previous_serotonin * decay
                + max(0.0, balanced_arousal) * 0.30
                + max(0.0, 1.0 - current_threat) * 0.08,
                0.0,
                1.0,
            )
        )
        patience_signal = float(
            np.clip(
                previous_patience * decay
                + serotonin_level * float(params["b54_patience_gain"])
                + max(0.0, norepinephrine_level) * 0.14,
                0.0,
                1.0,
            )
        )
        impulse_suppression = float(
            np.clip(
                previous_suppression * decay
                + max(0.0, surprise_signal) * float(params["b54_impulse_suppression_gain"])
                + max(0.0, current_threat) * 0.12,
                0.0,
                1.0,
            )
        )
        patience_score = float(
            np.clip(
                patience_signal * 0.34
                + serotonin_level * 0.28
                + max(0.0, arousal_gain) * float(params["b54_arousal_balance_gain"])
                - impulse_suppression * 0.08
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        patience_lock = int(getattr(self, "_b54_patience_lock", 0))
        decision_label = "preserve_b53"

        if corridor_map:
            if (
                b53_decision
                in {"noradrenergic_arousal_commit", "continue_arousal_gain"}
                and patience_score >= float(params["b54_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                patience_lock = max(
                    patience_lock,
                    int(params["b54_patience_lock_ticks"]),
                )
                decision_label = "serotonergic_patience_commit"
                reason = "b54_serotonergic_patience_commit"
            elif patience_score <= float(params["b54_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "serotonergic_patience_abort"
                reason = "b54_serotonergic_patience_abort"
            elif patience_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_patience_lock"
                reason = "b54_continue_patience_lock"

        trace_payload.update(
            {
                "b54_controller_profile": profile,
                "b54_serotonin_level": round(float(serotonin_level), 6),
                "b54_patience_signal": round(float(patience_signal), 6),
                "b54_impulse_suppression": round(float(impulse_suppression), 6),
                "b54_patience_lock": int(patience_lock),
                "b54_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b54_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b54_genetic_candidate"] = int(params["ga_candidate"])

        self._b54_serotonin_level = float(serotonin_level)
        self._b54_patience_signal = float(patience_signal)
        self._b54_impulse_suppression = float(impulse_suppression)
        self._b54_patience_lock = max(0, int(patience_lock) - 1)
        self._b54_last_tick = int(tick)
        return (
            semantic_action,
            B54_SEROTONERGIC_PATIENCE_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b55_controller_params(self) -> dict[str, float]:
        params = self._b54_controller_params()
        defaults = {
            "b55_drive_decay": 0.86,
            "b55_hunger_gain": 0.34,
            "b55_satiety_gain": 0.28,
            "b55_recovery_gain": 0.30,
            "b55_threat_gate_gain": 0.26,
            "b55_commit_threshold": 0.08,
            "b55_abort_threshold": -0.20,
            "b55_drive_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "hypothalamic_drive_coupling"
        )
        if profile == "satiety_recovery_balance":
            defaults.update({"b55_satiety_gain": 0.34, "b55_recovery_gain": 0.36})
        elif profile == "sleep_hunger_arbiter":
            defaults.update({"b55_hunger_gain": 0.38, "b55_threat_gate_gain": 0.30})
        elif profile == "hypothalamic_drive_coupling_h56":
            defaults.update({"b55_drive_decay": 0.89, "b55_drive_lock_ticks": 6.0})
        elif profile == "genetic_hypothalamic_drive":
            defaults.update({"b55_hunger_gain": 0.36, "b55_recovery_gain": 0.34})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b55_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b55_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b55_hypothalamic_drive = 0.0
        self._b55_satiety_signal = 0.0
        self._b55_recovery_bias = 0.0
        self._b55_drive_balance = 0.0
        self._b55_drive_lock = 0
        self._b55_last_tick = int(tick)

    def _b55_hypothalamic_drive_coupling_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b54_serotonergic_patience_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b55_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "hypothalamic_drive_coupling"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b55_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        b54_decision = str(trace_payload.get("b54_decision", "preserve_b53"))
        serotonin_level = float(trace_payload.get("b54_serotonin_level", 0.0) or 0.0)
        patience_signal = float(trace_payload.get("b54_patience_signal", 0.0) or 0.0)
        impulse_suppression = float(
            trace_payload.get("b54_impulse_suppression", 0.0) or 0.0
        )
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b55_drive_decay"])
        previous_drive = float(getattr(self, "_b55_hypothalamic_drive", 0.0))
        previous_satiety = float(getattr(self, "_b55_satiety_signal", 0.0))
        previous_recovery = float(getattr(self, "_b55_recovery_bias", 0.0))
        previous_balance = float(getattr(self, "_b55_drive_balance", 0.0))
        hypothalamic_drive = float(
            np.clip(
                previous_drive * decay
                + max(0.0, hunger) * float(params["b55_hunger_gain"])
                + max(0.0, patience_signal) * 0.14,
                0.0,
                1.0,
            )
        )
        satiety_signal = float(
            np.clip(
                previous_satiety * decay
                + max(0.0, 1.0 - hunger) * float(params["b55_satiety_gain"])
                + max(0.0, serotonin_level) * 0.10,
                0.0,
                1.0,
            )
        )
        recovery_bias = float(
            np.clip(
                previous_recovery * decay
                + max(0.0, sleep_debt) * float(params["b55_recovery_gain"])
                + max(0.0, 1.0 - health) * 0.18,
                0.0,
                1.0,
            )
        )
        drive_balance = float(
            np.clip(
                previous_balance * decay
                + hypothalamic_drive * 0.34
                + max(0.0, patience_signal) * 0.24
                - satiety_signal * 0.08
                - recovery_bias * 0.10
                - current_threat * float(params["b55_threat_gate_gain"])
                - impulse_suppression * 0.04,
                -1.0,
                1.0,
            )
        )
        drive_lock = int(getattr(self, "_b55_drive_lock", 0))
        decision_label = "preserve_b54"

        if corridor_map:
            if (
                b54_decision
                in {"serotonergic_patience_commit", "continue_patience_lock"}
                and drive_balance >= float(params["b55_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                drive_lock = max(drive_lock, int(params["b55_drive_lock_ticks"]))
                decision_label = "hypothalamic_drive_commit"
                reason = "b55_hypothalamic_drive_commit"
            elif drive_balance <= float(params["b55_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "hypothalamic_drive_abort"
                reason = "b55_hypothalamic_drive_abort"
            elif drive_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_drive_lock"
                reason = "b55_continue_drive_lock"

        trace_payload.update(
            {
                "b55_controller_profile": profile,
                "b55_hypothalamic_drive": round(float(hypothalamic_drive), 6),
                "b55_satiety_signal": round(float(satiety_signal), 6),
                "b55_recovery_bias": round(float(recovery_bias), 6),
                "b55_drive_balance": round(float(drive_balance), 6),
                "b55_drive_lock": int(drive_lock),
                "b55_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b55_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b55_genetic_candidate"] = int(params["ga_candidate"])

        self._b55_hypothalamic_drive = float(hypothalamic_drive)
        self._b55_satiety_signal = float(satiety_signal)
        self._b55_recovery_bias = float(recovery_bias)
        self._b55_drive_balance = float(drive_balance)
        self._b55_drive_lock = max(0, int(drive_lock) - 1)
        self._b55_last_tick = int(tick)
        return (
            semantic_action,
            B55_HYPOTHALAMIC_DRIVE_COUPLING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b56_controller_params(self) -> dict[str, float]:
        params = self._b55_controller_params()
        defaults = {
            "b56_endocrine_decay": 0.88,
            "b56_cortisol_gain": 0.30,
            "b56_stress_load_gain": 0.32,
            "b56_recovery_signal_gain": 0.30,
            "b56_drive_mod_gain": 0.28,
            "b56_commit_threshold": 0.08,
            "b56_abort_threshold": -0.20,
            "b56_stress_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "hpa_stress_axis")
        if profile == "cortisol_recovery_balance":
            defaults.update({"b56_cortisol_gain": 0.34, "b56_recovery_signal_gain": 0.36})
        elif profile == "stress_load_gate":
            defaults.update({"b56_stress_load_gain": 0.38, "b56_drive_mod_gain": 0.32})
        elif profile == "hpa_stress_axis_h56":
            defaults.update({"b56_endocrine_decay": 0.90, "b56_stress_lock_ticks": 6.0})
        elif profile == "genetic_hpa_stress":
            defaults.update({"b56_cortisol_gain": 0.32, "b56_drive_mod_gain": 0.34})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b56_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b56_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b56_cortisol_level = 0.0
        self._b56_stress_load = 0.0
        self._b56_recovery_signal = 0.0
        self._b56_endocrine_balance = 0.0
        self._b56_stress_lock = 0
        self._b56_last_tick = int(tick)

    def _b56_hpa_stress_axis_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b55_hypothalamic_drive_coupling_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b56_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "hpa_stress_axis")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b56_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b55_decision = str(trace_payload.get("b55_decision", "preserve_b54"))
        hypothalamic_drive = float(trace_payload.get("b55_hypothalamic_drive", 0.0) or 0.0)
        recovery_bias = float(trace_payload.get("b55_recovery_bias", 0.0) or 0.0)
        drive_balance_input = float(trace_payload.get("b55_drive_balance", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b56_endocrine_decay"])
        previous_cortisol = float(getattr(self, "_b56_cortisol_level", 0.0))
        previous_stress = float(getattr(self, "_b56_stress_load", 0.0))
        previous_recovery = float(getattr(self, "_b56_recovery_signal", 0.0))
        previous_balance = float(getattr(self, "_b56_endocrine_balance", 0.0))
        stress_load = float(
            np.clip(
                previous_stress * decay
                + current_threat * float(params["b56_stress_load_gain"])
                + max(0.0, hypothalamic_drive) * 0.10,
                0.0,
                1.0,
            )
        )
        cortisol_level = float(
            np.clip(
                previous_cortisol * decay
                + stress_load * float(params["b56_cortisol_gain"])
                + max(0.0, current_threat) * 0.08,
                0.0,
                1.0,
            )
        )
        recovery_signal = float(
            np.clip(
                previous_recovery * decay
                + max(0.0, recovery_bias) * float(params["b56_recovery_signal_gain"])
                + max(0.0, 1.0 - current_threat) * 0.04,
                0.0,
                1.0,
            )
        )
        endocrine_balance = float(
            np.clip(
                previous_balance * decay
                + drive_balance_input * 0.34
                + max(0.0, hypothalamic_drive) * float(params["b56_drive_mod_gain"])
                + recovery_signal * 0.08
                - cortisol_level * 0.08
                - stress_load * 0.05,
                -1.0,
                1.0,
            )
        )
        stress_lock = int(getattr(self, "_b56_stress_lock", 0))
        decision_label = "preserve_b55"

        if corridor_map:
            if (
                b55_decision
                in {"hypothalamic_drive_commit", "continue_drive_lock"}
                and endocrine_balance >= float(params["b56_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                stress_lock = max(stress_lock, int(params["b56_stress_lock_ticks"]))
                decision_label = "hpa_stress_axis_commit"
                reason = "b56_hpa_stress_axis_commit"
            elif endocrine_balance <= float(params["b56_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "hpa_stress_axis_abort"
                reason = "b56_hpa_stress_axis_abort"
            elif stress_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_stress_axis_lock"
                reason = "b56_continue_stress_axis_lock"

        trace_payload.update(
            {
                "b56_controller_profile": profile,
                "b56_cortisol_level": round(float(cortisol_level), 6),
                "b56_stress_load": round(float(stress_load), 6),
                "b56_recovery_signal": round(float(recovery_signal), 6),
                "b56_endocrine_balance": round(float(endocrine_balance), 6),
                "b56_stress_lock": int(stress_lock),
                "b56_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b56_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b56_genetic_candidate"] = int(params["ga_candidate"])

        self._b56_cortisol_level = float(cortisol_level)
        self._b56_stress_load = float(stress_load)
        self._b56_recovery_signal = float(recovery_signal)
        self._b56_endocrine_balance = float(endocrine_balance)
        self._b56_stress_lock = max(0, int(stress_lock) - 1)
        self._b56_last_tick = int(tick)
        return (
            semantic_action,
            B56_HPA_STRESS_AXIS_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b57_controller_params(self) -> dict[str, float]:
        params = self._b56_controller_params()
        defaults = {
            "b57_awareness_decay": 0.87,
            "b57_visceral_salience_gain": 0.32,
            "b57_body_confidence_gain": 0.30,
            "b57_stress_awareness_gain": 0.28,
            "b57_drive_awareness_gain": 0.28,
            "b57_commit_threshold": 0.08,
            "b57_abort_threshold": -0.20,
            "b57_awareness_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "insular_interoceptive_awareness"
        )
        if profile == "visceral_salience_gate":
            defaults.update(
                {"b57_visceral_salience_gain": 0.38, "b57_body_confidence_gain": 0.34}
            )
        elif profile == "stress_drive_awareness":
            defaults.update(
                {"b57_stress_awareness_gain": 0.34, "b57_drive_awareness_gain": 0.34}
            )
        elif profile == "insular_interoceptive_awareness_h56":
            defaults.update({"b57_awareness_decay": 0.90, "b57_awareness_lock_ticks": 6.0})
        elif profile == "genetic_interoceptive_awareness":
            defaults.update(
                {"b57_visceral_salience_gain": 0.34, "b57_drive_awareness_gain": 0.32}
            )
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b57_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b57_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b57_interoceptive_awareness = 0.0
        self._b57_visceral_salience = 0.0
        self._b57_body_state_confidence = 0.0
        self._b57_awareness_balance = 0.0
        self._b57_awareness_lock = 0
        self._b57_last_tick = int(tick)

    def _b57_insular_interoceptive_awareness_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b56_hpa_stress_axis_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b57_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "insular_interoceptive_awareness"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b57_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b56_decision = str(trace_payload.get("b56_decision", "preserve_b55"))
        cortisol_level = float(trace_payload.get("b56_cortisol_level", 0.0) or 0.0)
        stress_load = float(trace_payload.get("b56_stress_load", 0.0) or 0.0)
        recovery_signal = float(trace_payload.get("b56_recovery_signal", 0.0) or 0.0)
        endocrine_balance = float(trace_payload.get("b56_endocrine_balance", 0.0) or 0.0)
        hypothalamic_drive = float(trace_payload.get("b55_hypothalamic_drive", 0.0) or 0.0)
        hunger = self._b_series_float(meta, "hunger")
        sleep_debt = self._b_series_float(meta, "sleep_debt")
        health_deficit = max(0.0, 1.0 - self._b_series_float(meta, "health"))
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b57_awareness_decay"])
        previous_awareness = float(getattr(self, "_b57_interoceptive_awareness", 0.0))
        previous_salience = float(getattr(self, "_b57_visceral_salience", 0.0))
        previous_confidence = float(getattr(self, "_b57_body_state_confidence", 0.0))
        previous_balance = float(getattr(self, "_b57_awareness_balance", 0.0))
        body_load = max(hunger, sleep_debt, health_deficit)
        visceral_salience = float(
            np.clip(
                previous_salience * decay
                + body_load * float(params["b57_visceral_salience_gain"])
                + max(0.0, hypothalamic_drive) * 0.08,
                0.0,
                1.0,
            )
        )
        body_state_confidence = float(
            np.clip(
                previous_confidence * decay
                + recovery_signal * float(params["b57_body_confidence_gain"])
                + max(0.0, 1.0 - current_threat) * 0.06,
                0.0,
                1.0,
            )
        )
        interoceptive_awareness = float(
            np.clip(
                previous_awareness * decay
                + visceral_salience * 0.24
                + stress_load * float(params["b57_stress_awareness_gain"])
                + max(0.0, endocrine_balance) * float(params["b57_drive_awareness_gain"]),
                0.0,
                1.0,
            )
        )
        awareness_balance = float(
            np.clip(
                previous_balance * decay
                + endocrine_balance * 0.32
                + interoceptive_awareness * 0.10
                + body_state_confidence * 0.10
                - cortisol_level * 0.08
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        awareness_lock = int(getattr(self, "_b57_awareness_lock", 0))
        decision_label = "preserve_b56"

        if corridor_map:
            if (
                b56_decision in {"hpa_stress_axis_commit", "continue_stress_axis_lock"}
                and awareness_balance >= float(params["b57_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                awareness_lock = max(awareness_lock, int(params["b57_awareness_lock_ticks"]))
                decision_label = "interoceptive_awareness_commit"
                reason = "b57_interoceptive_awareness_commit"
            elif awareness_balance <= float(params["b57_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "interoceptive_awareness_abort"
                reason = "b57_interoceptive_awareness_abort"
            elif awareness_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_awareness_lock"
                reason = "b57_continue_awareness_lock"

        trace_payload.update(
            {
                "b57_controller_profile": profile,
                "b57_interoceptive_awareness": round(float(interoceptive_awareness), 6),
                "b57_visceral_salience": round(float(visceral_salience), 6),
                "b57_body_state_confidence": round(float(body_state_confidence), 6),
                "b57_awareness_balance": round(float(awareness_balance), 6),
                "b57_awareness_lock": int(awareness_lock),
                "b57_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b57_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b57_genetic_candidate"] = int(params["ga_candidate"])

        self._b57_interoceptive_awareness = float(interoceptive_awareness)
        self._b57_visceral_salience = float(visceral_salience)
        self._b57_body_state_confidence = float(body_state_confidence)
        self._b57_awareness_balance = float(awareness_balance)
        self._b57_awareness_lock = max(0, int(awareness_lock) - 1)
        self._b57_last_tick = int(tick)
        return (
            semantic_action,
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b58_controller_params(self) -> dict[str, float]:
        params = self._b57_controller_params()
        defaults = {
            "b58_conflict_decay": 0.87,
            "b58_conflict_gain": 0.32,
            "b58_error_likelihood_gain": 0.30,
            "b58_control_allocation_gain": 0.30,
            "b58_awareness_mod_gain": 0.28,
            "b58_commit_threshold": 0.08,
            "b58_abort_threshold": -0.20,
            "b58_conflict_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "acc_conflict_monitor")
        if profile == "error_salience_gate":
            defaults.update({"b58_error_likelihood_gain": 0.36, "b58_conflict_gain": 0.34})
        elif profile == "conflict_resolution_balance":
            defaults.update(
                {"b58_control_allocation_gain": 0.36, "b58_awareness_mod_gain": 0.32}
            )
        elif profile == "acc_conflict_monitor_h56":
            defaults.update({"b58_conflict_decay": 0.90, "b58_conflict_lock_ticks": 6.0})
        elif profile == "genetic_acc_conflict":
            defaults.update({"b58_conflict_gain": 0.34, "b58_control_allocation_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b58_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b58_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b58_conflict_signal = 0.0
        self._b58_error_likelihood = 0.0
        self._b58_control_allocation = 0.0
        self._b58_resolution_balance = 0.0
        self._b58_conflict_lock = 0
        self._b58_last_tick = int(tick)

    def _b58_acc_conflict_monitor_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b57_insular_interoceptive_awareness_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b58_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "acc_conflict_monitor")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b58_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b57_decision = str(trace_payload.get("b57_decision", "preserve_b56"))
        awareness_balance = float(trace_payload.get("b57_awareness_balance", 0.0) or 0.0)
        interoceptive_awareness = float(
            trace_payload.get("b57_interoceptive_awareness", 0.0) or 0.0
        )
        visceral_salience = float(trace_payload.get("b57_visceral_salience", 0.0) or 0.0)
        body_confidence = float(trace_payload.get("b57_body_state_confidence", 0.0) or 0.0)
        cortisol_level = float(trace_payload.get("b56_cortisol_level", 0.0) or 0.0)
        stress_load = float(trace_payload.get("b56_stress_load", 0.0) or 0.0)
        food_dist = self._b_series_float(meta, "food_dist")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        route_conflict = float(np.clip((shelter_dist + 1.0) / (food_dist + 1.0), 0.0, 1.0))
        if food_dist <= 0.0:
            route_conflict = 0.0

        decay = float(params["b58_conflict_decay"])
        previous_conflict = float(getattr(self, "_b58_conflict_signal", 0.0))
        previous_error = float(getattr(self, "_b58_error_likelihood", 0.0))
        previous_control = float(getattr(self, "_b58_control_allocation", 0.0))
        previous_balance = float(getattr(self, "_b58_resolution_balance", 0.0))
        conflict_input = abs(awareness_balance) * 0.35 + current_threat * 0.25 + route_conflict * 0.20
        conflict_signal = float(
            np.clip(
                previous_conflict * decay
                + conflict_input * float(params["b58_conflict_gain"])
                + visceral_salience * 0.06,
                0.0,
                1.0,
            )
        )
        error_likelihood = float(
            np.clip(
                previous_error * decay
                + (current_threat + cortisol_level + stress_load) / 3.0
                * float(params["b58_error_likelihood_gain"])
                + route_conflict * 0.05,
                0.0,
                1.0,
            )
        )
        control_allocation = float(
            np.clip(
                previous_control * decay
                + body_confidence * float(params["b58_control_allocation_gain"])
                + interoceptive_awareness * float(params["b58_awareness_mod_gain"])
                + max(0.0, awareness_balance) * 0.08,
                0.0,
                1.0,
            )
        )
        resolution_balance = float(
            np.clip(
                previous_balance * decay
                + awareness_balance * 0.32
                + control_allocation * 0.14
                - error_likelihood * 0.08
                - conflict_signal * 0.06,
                -1.0,
                1.0,
            )
        )
        conflict_lock = int(getattr(self, "_b58_conflict_lock", 0))
        decision_label = "preserve_b57"

        if corridor_map:
            if (
                b57_decision
                in {"interoceptive_awareness_commit", "continue_awareness_lock"}
                and resolution_balance >= float(params["b58_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                conflict_lock = max(conflict_lock, int(params["b58_conflict_lock_ticks"]))
                decision_label = "acc_conflict_commit"
                reason = "b58_acc_conflict_commit"
            elif resolution_balance <= float(params["b58_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "acc_conflict_abort"
                reason = "b58_acc_conflict_abort"
            elif conflict_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_conflict_lock"
                reason = "b58_continue_conflict_lock"

        trace_payload.update(
            {
                "b58_controller_profile": profile,
                "b58_conflict_signal": round(float(conflict_signal), 6),
                "b58_error_likelihood": round(float(error_likelihood), 6),
                "b58_control_allocation": round(float(control_allocation), 6),
                "b58_resolution_balance": round(float(resolution_balance), 6),
                "b58_conflict_lock": int(conflict_lock),
                "b58_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b58_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b58_genetic_candidate"] = int(params["ga_candidate"])

        self._b58_conflict_signal = float(conflict_signal)
        self._b58_error_likelihood = float(error_likelihood)
        self._b58_control_allocation = float(control_allocation)
        self._b58_resolution_balance = float(resolution_balance)
        self._b58_conflict_lock = max(0, int(conflict_lock) - 1)
        self._b58_last_tick = int(tick)
        return (
            semantic_action,
            B58_ACC_CONFLICT_MONITOR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b59_controller_params(self) -> dict[str, float]:
        params = self._b58_controller_params()
        defaults = {
            "b59_context_decay": 0.87,
            "b59_goal_context_gain": 0.32,
            "b59_working_set_gain": 0.30,
            "b59_task_confidence_gain": 0.30,
            "b59_control_mod_gain": 0.28,
            "b59_commit_threshold": 0.08,
            "b59_abort_threshold": -0.20,
            "b59_executive_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "prefrontal_goal_context")
        if profile == "working_set_stability":
            defaults.update({"b59_working_set_gain": 0.36, "b59_task_confidence_gain": 0.34})
        elif profile == "executive_task_set":
            defaults.update({"b59_goal_context_gain": 0.34, "b59_control_mod_gain": 0.34})
        elif profile == "prefrontal_goal_context_h56":
            defaults.update({"b59_context_decay": 0.90, "b59_executive_lock_ticks": 6.0})
        elif profile == "genetic_prefrontal_control":
            defaults.update({"b59_goal_context_gain": 0.34, "b59_working_set_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b59_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b59_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b59_goal_context = 0.0
        self._b59_working_set_stability = 0.0
        self._b59_task_set_confidence = 0.0
        self._b59_executive_balance = 0.0
        self._b59_executive_lock = 0
        self._b59_last_tick = int(tick)

    def _b59_prefrontal_goal_context_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b58_acc_conflict_monitor_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b59_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "prefrontal_goal_context")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b59_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b58_decision = str(trace_payload.get("b58_decision", "preserve_b57"))
        resolution_balance = float(trace_payload.get("b58_resolution_balance", 0.0) or 0.0)
        control_allocation = float(trace_payload.get("b58_control_allocation", 0.0) or 0.0)
        conflict_signal = float(trace_payload.get("b58_conflict_signal", 0.0) or 0.0)
        error_likelihood = float(trace_payload.get("b58_error_likelihood", 0.0) or 0.0)
        awareness_balance = float(trace_payload.get("b57_awareness_balance", 0.0) or 0.0)
        body_confidence = float(trace_payload.get("b57_body_state_confidence", 0.0) or 0.0)
        food_dist = self._b_series_float(meta, "food_dist")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        corridor_goal = 1.0 if corridor_map and food_dist >= shelter_dist else 0.0

        decay = float(params["b59_context_decay"])
        previous_goal = float(getattr(self, "_b59_goal_context", 0.0))
        previous_stability = float(getattr(self, "_b59_working_set_stability", 0.0))
        previous_confidence = float(getattr(self, "_b59_task_set_confidence", 0.0))
        previous_balance = float(getattr(self, "_b59_executive_balance", 0.0))
        goal_context = float(
            np.clip(
                previous_goal * decay
                + corridor_goal * float(params["b59_goal_context_gain"])
                + max(0.0, resolution_balance) * 0.10,
                0.0,
                1.0,
            )
        )
        working_set_stability = float(
            np.clip(
                previous_stability * decay
                + control_allocation * float(params["b59_working_set_gain"])
                + body_confidence * 0.06,
                0.0,
                1.0,
            )
        )
        task_set_confidence = float(
            np.clip(
                previous_confidence * decay
                + max(0.0, awareness_balance) * float(params["b59_task_confidence_gain"])
                + max(0.0, 1.0 - current_threat) * 0.06,
                0.0,
                1.0,
            )
        )
        executive_balance = float(
            np.clip(
                previous_balance * decay
                + resolution_balance * 0.32
                + goal_context * 0.10
                + working_set_stability * 0.10
                + task_set_confidence * float(params["b59_control_mod_gain"])
                - conflict_signal * 0.05
                - error_likelihood * 0.07,
                -1.0,
                1.0,
            )
        )
        executive_lock = int(getattr(self, "_b59_executive_lock", 0))
        decision_label = "preserve_b58"

        if corridor_map:
            if (
                b58_decision in {"acc_conflict_commit", "continue_conflict_lock"}
                and executive_balance >= float(params["b59_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                executive_lock = max(executive_lock, int(params["b59_executive_lock_ticks"]))
                decision_label = "prefrontal_goal_commit"
                reason = "b59_prefrontal_goal_commit"
            elif executive_balance <= float(params["b59_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "prefrontal_goal_abort"
                reason = "b59_prefrontal_goal_abort"
            elif executive_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_executive_lock"
                reason = "b59_continue_executive_lock"

        trace_payload.update(
            {
                "b59_controller_profile": profile,
                "b59_goal_context": round(float(goal_context), 6),
                "b59_working_set_stability": round(float(working_set_stability), 6),
                "b59_task_set_confidence": round(float(task_set_confidence), 6),
                "b59_executive_balance": round(float(executive_balance), 6),
                "b59_executive_lock": int(executive_lock),
                "b59_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b59_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b59_genetic_candidate"] = int(params["ga_candidate"])

        self._b59_goal_context = float(goal_context)
        self._b59_working_set_stability = float(working_set_stability)
        self._b59_task_set_confidence = float(task_set_confidence)
        self._b59_executive_balance = float(executive_balance)
        self._b59_executive_lock = max(0, int(executive_lock) - 1)
        self._b59_last_tick = int(tick)
        return (
            semantic_action,
            B59_PREFRONTAL_GOAL_CONTEXT_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b60_controller_params(self) -> dict[str, float]:
        params = self._b59_controller_params()
        defaults = {
            "b60_value_decay": 0.88,
            "b60_outcome_value_gain": 0.32,
            "b60_reversal_signal_gain": 0.28,
            "b60_goal_value_confidence_gain": 0.30,
            "b60_prefrontal_mod_gain": 0.26,
            "b60_commit_threshold": 0.07,
            "b60_reversal_threshold": 0.24,
            "b60_value_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "orbitofrontal_outcome_value"
        )
        if profile == "reversal_value_gate":
            defaults.update({"b60_reversal_signal_gain": 0.34, "b60_reversal_threshold": 0.20})
        elif profile == "goal_outcome_prediction":
            defaults.update(
                {
                    "b60_outcome_value_gain": 0.36,
                    "b60_goal_value_confidence_gain": 0.34,
                }
            )
        elif profile == "orbitofrontal_outcome_value_h56":
            defaults.update({"b60_value_decay": 0.90, "b60_value_lock_ticks": 6.0})
        elif profile == "genetic_orbitofrontal_value":
            defaults.update({"b60_outcome_value_gain": 0.34, "b60_prefrontal_mod_gain": 0.30})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b60_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b60_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b60_outcome_value = 0.0
        self._b60_reversal_signal = 0.0
        self._b60_goal_value_confidence = 0.0
        self._b60_value_balance = 0.0
        self._b60_value_lock = 0
        self._b60_last_tick = int(tick)

    def _b60_orbitofrontal_outcome_value_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b59_prefrontal_goal_context_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b60_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "orbitofrontal_outcome_value"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b60_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b59_decision = str(trace_payload.get("b59_decision", "preserve_b58"))
        executive_balance = float(trace_payload.get("b59_executive_balance", 0.0) or 0.0)
        goal_context = float(trace_payload.get("b59_goal_context", 0.0) or 0.0)
        working_set = float(trace_payload.get("b59_working_set_stability", 0.0) or 0.0)
        task_confidence = float(trace_payload.get("b59_task_set_confidence", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        hunger = self._b_series_float(meta, "hunger")
        health = self._b_series_float(meta, "health")
        food_dist = self._b_series_float(meta, "food_dist")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        progress_value = 1.0 if food_dist >= shelter_dist else 0.0
        homeostatic_margin = float(np.clip((1.0 - hunger) * 0.45 + health * 0.55, 0.0, 1.0))
        threat_penalty = float(np.clip(current_threat + max(0.0, shelter_dist - food_dist) * 0.02, 0.0, 1.0))

        decay = float(params["b60_value_decay"])
        previous_value = float(getattr(self, "_b60_outcome_value", 0.0))
        previous_reversal = float(getattr(self, "_b60_reversal_signal", 0.0))
        previous_confidence = float(getattr(self, "_b60_goal_value_confidence", 0.0))
        previous_balance = float(getattr(self, "_b60_value_balance", 0.0))
        outcome_value = float(
            np.clip(
                previous_value * decay
                + executive_balance * float(params["b60_outcome_value_gain"])
                + progress_value * 0.08
                + homeostatic_margin * 0.06,
                -1.0,
                1.0,
            )
        )
        reversal_signal = float(
            np.clip(
                previous_reversal * decay
                + threat_penalty * float(params["b60_reversal_signal_gain"])
                + max(0.0, -executive_balance) * 0.12,
                0.0,
                1.0,
            )
        )
        goal_value_confidence = float(
            np.clip(
                previous_confidence * decay
                + task_confidence * float(params["b60_goal_value_confidence_gain"])
                + working_set * 0.08
                + goal_context * 0.07,
                0.0,
                1.0,
            )
        )
        value_balance = float(
            np.clip(
                previous_balance * decay
                + outcome_value * 0.36
                + goal_value_confidence * float(params["b60_prefrontal_mod_gain"])
                - reversal_signal * 0.30,
                -1.0,
                1.0,
            )
        )
        value_lock = int(getattr(self, "_b60_value_lock", 0))
        decision_label = "preserve_b59"

        if corridor_map:
            if (
                b59_decision in {"prefrontal_goal_commit", "continue_executive_lock"}
                and value_balance >= float(params["b60_commit_threshold"])
                and reversal_signal < float(params["b60_reversal_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                value_lock = max(value_lock, int(params["b60_value_lock_ticks"]))
                decision_label = "orbitofrontal_value_commit"
                reason = "b60_orbitofrontal_value_commit"
            elif reversal_signal >= float(params["b60_reversal_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "orbitofrontal_value_reversal"
                reason = "b60_orbitofrontal_value_reversal"
            elif value_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_value_lock"
                reason = "b60_continue_value_lock"

        trace_payload.update(
            {
                "b60_controller_profile": profile,
                "b60_outcome_value": round(float(outcome_value), 6),
                "b60_reversal_signal": round(float(reversal_signal), 6),
                "b60_goal_value_confidence": round(float(goal_value_confidence), 6),
                "b60_value_balance": round(float(value_balance), 6),
                "b60_value_lock": int(value_lock),
                "b60_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b60_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b60_genetic_candidate"] = int(params["ga_candidate"])

        self._b60_outcome_value = float(outcome_value)
        self._b60_reversal_signal = float(reversal_signal)
        self._b60_goal_value_confidence = float(goal_value_confidence)
        self._b60_value_balance = float(value_balance)
        self._b60_value_lock = max(0, int(value_lock) - 1)
        self._b60_last_tick = int(tick)
        return (
            semantic_action,
            B60_ORBITOFRONTAL_OUTCOME_VALUE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b61_controller_params(self) -> dict[str, float]:
        params = self._b60_controller_params()
        defaults = {
            "b61_affect_decay": 0.88,
            "b61_safety_value_gain": 0.32,
            "b61_threat_value_gain": 0.28,
            "b61_safety_confidence_gain": 0.30,
            "b61_ofc_mod_gain": 0.26,
            "b61_commit_threshold": 0.07,
            "b61_threat_threshold": 0.26,
            "b61_safety_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "amygdala_safety_value")
        if profile == "threat_value_tag":
            defaults.update({"b61_threat_value_gain": 0.34, "b61_threat_threshold": 0.22})
        elif profile == "safety_prediction_gate":
            defaults.update(
                {"b61_safety_value_gain": 0.36, "b61_safety_confidence_gain": 0.34}
            )
        elif profile == "amygdala_safety_value_h56":
            defaults.update({"b61_affect_decay": 0.90, "b61_safety_lock_ticks": 6.0})
        elif profile == "genetic_amygdala_safety":
            defaults.update({"b61_safety_value_gain": 0.34, "b61_ofc_mod_gain": 0.30})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b61_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b61_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b61_safety_value = 0.0
        self._b61_threat_value = 0.0
        self._b61_safety_confidence = 0.0
        self._b61_affective_balance = 0.0
        self._b61_safety_lock = 0
        self._b61_last_tick = int(tick)

    def _b61_amygdala_safety_value_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b60_orbitofrontal_outcome_value_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b61_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "amygdala_safety_value")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b61_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b60_decision = str(trace_payload.get("b60_decision", "preserve_b59"))
        outcome_value = float(trace_payload.get("b60_outcome_value", 0.0) or 0.0)
        reversal_signal = float(trace_payload.get("b60_reversal_signal", 0.0) or 0.0)
        goal_value_confidence = float(trace_payload.get("b60_goal_value_confidence", 0.0) or 0.0)
        value_balance = float(trace_payload.get("b60_value_balance", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        hunger = self._b_series_float(meta, "hunger")
        health = self._b_series_float(meta, "health")
        food_dist = self._b_series_float(meta, "food_dist")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        corridor_safety_cue = float(
            np.clip((1.0 - current_threat) * 0.65 + max(0.0, food_dist - shelter_dist) * 0.04, 0.0, 1.0)
        )
        vulnerability = float(np.clip(hunger * 0.35 + (1.0 - health) * 0.45 + reversal_signal * 0.20, 0.0, 1.0))

        decay = float(params["b61_affect_decay"])
        previous_safety = float(getattr(self, "_b61_safety_value", 0.0))
        previous_threat = float(getattr(self, "_b61_threat_value", 0.0))
        previous_confidence = float(getattr(self, "_b61_safety_confidence", 0.0))
        previous_balance = float(getattr(self, "_b61_affective_balance", 0.0))
        safety_value = float(
            np.clip(
                previous_safety * decay
                + max(0.0, outcome_value) * float(params["b61_safety_value_gain"])
                + corridor_safety_cue * 0.08,
                0.0,
                1.0,
            )
        )
        threat_value = float(
            np.clip(
                previous_threat * decay
                + current_threat * float(params["b61_threat_value_gain"])
                + vulnerability * 0.03,
                0.0,
                1.0,
            )
        )
        safety_confidence = float(
            np.clip(
                previous_confidence * decay
                + goal_value_confidence * float(params["b61_safety_confidence_gain"])
                + max(0.0, value_balance) * 0.08,
                0.0,
                1.0,
            )
        )
        affective_balance = float(
            np.clip(
                previous_balance * decay
                + safety_value * 0.32
                + safety_confidence * float(params["b61_ofc_mod_gain"])
                - threat_value * 0.30,
                -1.0,
                1.0,
            )
        )
        safety_lock = int(getattr(self, "_b61_safety_lock", 0))
        decision_label = "preserve_b60"

        if corridor_map:
            if (
                b60_decision in {"orbitofrontal_value_commit", "continue_value_lock"}
                and affective_balance >= float(params["b61_commit_threshold"])
                and threat_value < float(params["b61_threat_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                safety_lock = max(safety_lock, int(params["b61_safety_lock_ticks"]))
                decision_label = "amygdala_safety_commit"
                reason = "b61_amygdala_safety_commit"
            elif threat_value >= float(params["b61_threat_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "amygdala_threat_reversal"
                reason = "b61_amygdala_threat_reversal"
            elif safety_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_safety_lock"
                reason = "b61_continue_safety_lock"

        trace_payload.update(
            {
                "b61_controller_profile": profile,
                "b61_safety_value": round(float(safety_value), 6),
                "b61_threat_value": round(float(threat_value), 6),
                "b61_safety_confidence": round(float(safety_confidence), 6),
                "b61_affective_balance": round(float(affective_balance), 6),
                "b61_safety_lock": int(safety_lock),
                "b61_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b61_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b61_genetic_candidate"] = int(params["ga_candidate"])

        self._b61_safety_value = float(safety_value)
        self._b61_threat_value = float(threat_value)
        self._b61_safety_confidence = float(safety_confidence)
        self._b61_affective_balance = float(affective_balance)
        self._b61_safety_lock = max(0, int(safety_lock) - 1)
        self._b61_last_tick = int(tick)
        return (
            semantic_action,
            B61_AMYGDALA_SAFETY_VALUE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b62_controller_params(self) -> dict[str, float]:
        params = self._b61_controller_params()
        defaults = {
            "b62_defense_decay": 0.88,
            "b62_freeze_gain": 0.30,
            "b62_flee_gain": 0.34,
            "b62_shelter_bias_gain": 0.28,
            "b62_balance_gain": 0.26,
            "b62_freeze_threshold": 0.30,
            "b62_flee_threshold": 0.23,
            "b62_defense_lock_ticks": 4.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "defensive_mode_selector")
        if profile == "freeze_flee_balance":
            defaults.update({"b62_freeze_gain": 0.34, "b62_flee_gain": 0.36})
        elif profile == "shelter_defense_gate":
            defaults.update({"b62_shelter_bias_gain": 0.34, "b62_flee_threshold": 0.20})
        elif profile == "defensive_mode_selector_h56":
            defaults.update({"b62_defense_decay": 0.90, "b62_defense_lock_ticks": 5.0})
        elif profile == "genetic_defensive_mode":
            defaults.update({"b62_flee_gain": 0.36, "b62_balance_gain": 0.30})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b62_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b62_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b62_freeze_pressure = 0.0
        self._b62_flee_pressure = 0.0
        self._b62_shelter_bias = 0.0
        self._b62_defense_balance = 0.0
        self._b62_defense_lock = 0
        self._b62_last_tick = int(tick)

    def _b62_defensive_mode_selector_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b61_amygdala_safety_value_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b62_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "defensive_mode_selector")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b62_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        shelter_role = str(meta.get("shelter_role", "outside"))
        b61_decision = str(trace_payload.get("b61_decision", "preserve_b60"))
        safety_value = float(trace_payload.get("b61_safety_value", 0.0) or 0.0)
        threat_value = float(trace_payload.get("b61_threat_value", 0.0) or 0.0)
        safety_confidence = float(trace_payload.get("b61_safety_confidence", 0.0) or 0.0)
        affective_balance = float(trace_payload.get("b61_affective_balance", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        hunger = self._b_series_float(meta, "hunger")
        health = self._b_series_float(meta, "health")
        sleep_debt = self._b_series_float(meta, "sleep_debt")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        food_dist = self._b_series_float(meta, "food_dist")
        near_shelter = 1.0 if shelter_dist <= 2.0 or shelter_role in {"at_shelter", "deep_shelter"} else 0.0
        homeostatic_vulnerability = float(
            np.clip((1.0 - health) * 0.45 + sleep_debt * 0.25 + hunger * 0.20, 0.0, 1.0)
        )
        shelter_advantage = float(np.clip((food_dist - shelter_dist) * 0.05 + near_shelter * 0.18, 0.0, 1.0))

        decay = float(params["b62_defense_decay"])
        previous_freeze = float(getattr(self, "_b62_freeze_pressure", 0.0))
        previous_flee = float(getattr(self, "_b62_flee_pressure", 0.0))
        previous_shelter = float(getattr(self, "_b62_shelter_bias", 0.0))
        previous_balance = float(getattr(self, "_b62_defense_balance", 0.0))
        freeze_pressure = float(
            np.clip(
                previous_freeze * decay
                + (current_threat + threat_value) * 0.5 * float(params["b62_freeze_gain"])
                + near_shelter * 0.06,
                0.0,
                1.0,
            )
        )
        flee_pressure = float(
            np.clip(
                previous_flee * decay
                + (threat_value + homeostatic_vulnerability) * float(params["b62_flee_gain"])
                + max(0.0, -affective_balance) * 0.08,
                0.0,
                1.0,
            )
        )
        shelter_bias = float(
            np.clip(
                previous_shelter * decay
                + shelter_advantage * float(params["b62_shelter_bias_gain"])
                + threat_value * 0.08,
                0.0,
                1.0,
            )
        )
        defense_balance = float(
            np.clip(
                previous_balance * decay
                + flee_pressure * float(params["b62_balance_gain"])
                + shelter_bias * 0.24
                - safety_value * 0.16
                - safety_confidence * 0.10,
                -1.0,
                1.0,
            )
        )
        defense_lock = int(getattr(self, "_b62_defense_lock", 0))
        decision_label = "preserve_b61"
        defensive_mode = "preserve"

        if corridor_map:
            if (
                flee_pressure >= float(params["b62_flee_threshold"])
                and defense_balance > 0.0
                and hunger < 0.96
            ):
                semantic_action = "MOVE_TO_SHELTER"
                defense_lock = max(defense_lock, int(params["b62_defense_lock_ticks"]))
                defensive_mode = "flee_to_shelter"
                decision_label = "defensive_flee_to_shelter"
                reason = "b62_defensive_flee_to_shelter"
            elif (
                freeze_pressure >= float(params["b62_freeze_threshold"])
                and near_shelter > 0.0
                and hunger < 0.92
            ):
                semantic_action = "STAY"
                defense_lock = max(defense_lock, int(params["b62_defense_lock_ticks"]))
                defensive_mode = "freeze_hold"
                decision_label = "defensive_freeze_hold"
                reason = "b62_defensive_freeze_hold"
            elif defense_lock > 0 and hunger < 0.96:
                semantic_action = "MOVE_TO_SHELTER"
                defensive_mode = "continue_defense_lock"
                decision_label = "continue_defense_lock"
                reason = "b62_continue_defense_lock"
            elif (
                b61_decision in {"amygdala_safety_commit", "continue_safety_lock"}
                and safety_value >= max(threat_value, 0.0)
                and safety_confidence > 0.0
            ):
                semantic_action = "MOVE_TO_FOOD"
                defensive_mode = "safe_advance"
                decision_label = "defensive_safe_advance"
                reason = "b62_defensive_safe_advance"

        trace_payload.update(
            {
                "b62_controller_profile": profile,
                "b62_defensive_mode": defensive_mode,
                "b62_freeze_pressure": round(float(freeze_pressure), 6),
                "b62_flee_pressure": round(float(flee_pressure), 6),
                "b62_shelter_bias": round(float(shelter_bias), 6),
                "b62_defense_balance": round(float(defense_balance), 6),
                "b62_defense_lock": int(defense_lock),
                "b62_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b62_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b62_genetic_candidate"] = int(params["ga_candidate"])

        self._b62_freeze_pressure = float(freeze_pressure)
        self._b62_flee_pressure = float(flee_pressure)
        self._b62_shelter_bias = float(shelter_bias)
        self._b62_defense_balance = float(defense_balance)
        self._b62_defense_lock = max(0, int(defense_lock) - 1)
        self._b62_last_tick = int(tick)
        return (
            semantic_action,
            B62_DEFENSIVE_MODE_SELECTOR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    @staticmethod
    def _network_forward_macs(network: object) -> int:
        if isinstance(network, RecurrentProposalNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
            )
        if isinstance(network, RecurrentEventAttentionTrueMonolithicNetwork):
            recurrent_input_dim = (
                network.input_dim + network.event_context_dim
            )
            event_raw_dim = (
                network.event_embedding_dim + network.event_feature_dim + 1
            )
            return int(
                recurrent_input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.event_context_dim * (network.input_dim + network.hidden_dim)
                + 2 * (network.event_context_dim * event_raw_dim)
            )
        if isinstance(network, RecurrentOptionAffordanceTrueMonolithicNetwork):
            recurrent_input_dim = (
                network.input_dim + network.event_context_dim + network.option_dim
            )
            event_raw_dim = (
                network.event_embedding_dim + network.event_feature_dim + 1
            )
            return int(
                recurrent_input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.hidden_dim * network.option_dim
                + network.option_dim * network.output_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.hidden_dim * network.output_dim * network.affordance_role_dim
                + network.output_dim * network.affordance_role_dim
                + network.event_context_dim * (network.input_dim + network.hidden_dim)
                + 2 * (network.event_context_dim * event_raw_dim)
            )
        if isinstance(network, RecurrentOptionTrueMonolithicNetwork):
            recurrent_input_dim = (
                network.input_dim + network.event_context_dim + network.option_dim
            )
            event_raw_dim = (
                network.event_embedding_dim + network.event_feature_dim + 1
            )
            return int(
                recurrent_input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.hidden_dim * network.option_dim
                + network.option_dim * network.output_dim
                + network.event_context_dim * (network.input_dim + network.hidden_dim)
                + 2 * (network.event_context_dim * event_raw_dim)
            )
        if isinstance(network, ArbitrationNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.valence_dim
                + network.hidden_dim * network.gate_dim
                + network.hidden_dim
            )
        if isinstance(network, MotorNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
            )
        if isinstance(network, ProposalNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
            )
        return 0

    def estimate_compute_cost(self) -> Dict[str, object]:
        per_network: Dict[str, int] = {}
        if self.module_bank is not None:
            per_network.update(
                {
                    name: self._network_forward_macs(network)
                    for name, network in self.module_bank.modules.items()
                }
            )
        if self.monolithic_policy is not None:
            per_network[self.MONOLITHIC_POLICY_NAME] = self._network_forward_macs(
                self.monolithic_policy
            )
        if self.true_monolithic_policy is not None:
            per_network[self.TRUE_MONOLITHIC_POLICY_NAME] = self._network_forward_macs(
                self.true_monolithic_policy
            )
        if getattr(self, "b_series_policy", None) is not None:
            per_network[B_SERIES_POLICY_NAME] = self._network_forward_macs(
                self.b_series_policy
            )
        if self.arbitration_network is not None:
            per_network[self.ARBITRATION_NETWORK_NAME] = self._network_forward_macs(
                self.arbitration_network
            )
        if self.action_center is not None:
            per_network["action_center"] = self._network_forward_macs(
                self.action_center
            )
        if self.motor_cortex is not None:
            per_network["motor_cortex"] = self._network_forward_macs(
                self.motor_cortex
            )
        return {
            "unit": "approx_forward_macs",
            "per_network": per_network,
            "total": int(sum(per_network.values())),
        }

    def _true_monolithic_arbitration_decision(
        self,
        *,
        module_name: str,
        action_idx: int,
    ) -> ArbitrationDecision:
        """Build a trivial arbitration payload for the direct-control baseline."""
        return ArbitrationDecision(
            strategy="direct_control",
            winning_valence="exploration",
            valence_scores={
                "threat": 0.0,
                "hunger": 0.0,
                "sleep": 0.0,
                "exploration": 1.0,
            },
            module_gates={module_name: 1.0},
            suppressed_modules=[],
            evidence={},
            intent_before_gating_idx=int(action_idx),
            intent_after_gating_idx=int(action_idx),
            valence_logits={
                "threat": 0.0,
                "hunger": 0.0,
                "sleep": 0.0,
                "exploration": 1.0,
            },
            base_gates={module_name: 1.0},
            gate_adjustments={module_name: 1.0},
            arbitration_value=0.0,
            learned_adjustment=False,
            module_contribution_share={module_name: 1.0},
            dominant_module=module_name,
            dominant_module_share=1.0,
            effective_module_count=1.0,
            gate_entropy=0.0,
            dominance_rate=1.0,
            effective_proposer_count=1.0,
            module_counts={module_name: 1},
            module_agreement_rate=1.0,
            module_disagreement_rate=0.0,
        )

    def _food_direction_bias_action(
        self,
        observation: Dict[str, np.ndarray],
    ) -> str | None:
        """
        Return a locomotion action that follows the strongest available food cue.

        Preference order matches the modular inference path: visible food,
        then food trace, then fresh food memory, then smell.
        """
        hunger_obs = self._bound_observation("hunger_center", observation)
        food_dx = 0.0
        food_dy = 0.0
        if hunger_obs["food_visible"] > 0.0 and hunger_obs["food_certainty"] > 0.0:
            food_dx = hunger_obs["food_dx"]
            food_dy = hunger_obs["food_dy"]
        elif hunger_obs["food_trace_strength"] > 0.0:
            food_dx = hunger_obs["food_trace_dx"]
            food_dy = hunger_obs["food_trace_dy"]
        elif hunger_obs["food_memory_age"] < 1.0:
            food_dx = hunger_obs["food_memory_dx"]
            food_dy = hunger_obs["food_memory_dy"]
        elif hunger_obs["food_smell_strength"] > 0.0:
            food_dx = hunger_obs["food_smell_dx"]
            food_dy = hunger_obs["food_smell_dy"]
        action = direction_action(food_dx, food_dy)
        if action == "STAY":
            return None
        return action

    def _threat_escape_bias_action(
        self,
        observation: Dict[str, np.ndarray],
    ) -> str | None:
        """
        Return a locomotion action that biases the agent back toward shelter under threat.

        For the direct-control baseline this provides a lightweight escape prior:
        when recent contact/pain, a predator trace, or fresh predator memory is
        active, prefer the freshest shelter-memory direction and otherwise move
        away from the predator cue.
        """
        threat_obs = self._bound_observation("threat_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        sleep_rest_action = self._sleep_rest_bias_action(observation)
        smell_signal = threat_obs["predator_smell_strength"]
        predator_signal = max(
            threat_obs["predator_trace_strength"],
            max(0.0, 1.0 - threat_obs["predator_memory_age"]),
            threat_obs["predator_visible"] * threat_obs["predator_certainty"],
            threat_obs["recent_contact"],
            threat_obs["recent_pain"],
        )
        if predator_signal <= 0.0 and smell_signal < TRUE_MONOLITHIC_THREAT_ESCAPE_SMELL_THRESHOLD:
            return None
        if sleep_rest_action == "STAY" and predator_signal <= 0.0:
            return None
        if sleep_obs["shelter_memory_age"] < 1.0:
            shelter_action = direction_action(
                sleep_obs["shelter_memory_dx"],
                sleep_obs["shelter_memory_dy"],
            )
            if shelter_action != "STAY":
                return shelter_action
        predator_dx = 0.0
        predator_dy = 0.0
        if threat_obs["predator_trace_strength"] > 0.0:
            predator_dx = threat_obs["predator_trace_dx"]
            predator_dy = threat_obs["predator_trace_dy"]
        elif threat_obs["predator_memory_age"] < 1.0:
            predator_dx = threat_obs["predator_memory_dx"]
            predator_dy = threat_obs["predator_memory_dy"]
        elif threat_obs["predator_visible"] > 0.0 and threat_obs["predator_certainty"] > 0.0:
            predator_dx = threat_obs["predator_dx"]
            predator_dy = threat_obs["predator_dy"]
        elif smell_signal >= TRUE_MONOLITHIC_THREAT_ESCAPE_SMELL_THRESHOLD:
            predator_dx = threat_obs["predator_smell_dx"]
            predator_dy = threat_obs["predator_smell_dy"]
        escape_action = direction_action(-predator_dx, -predator_dy)
        if escape_action == "STAY":
            return None
        return escape_action

    def _sleep_rest_bias_action(
        self,
        observation: Dict[str, np.ndarray],
    ) -> str | None:
        """
        Return STAY when the spider is already sheltered and should rest in place.

        This mirrors the existing sleep-center reflex thresholds so the
        true-monolithic baseline can hold shelter and accumulate rest instead of
        pacing inside the burrow after a successful return.
        """
        sleep_obs = self._bound_observation("sleep_center", observation)
        thresholds = self.operational_profile.brain_reflex_thresholds["sleep_center"]
        if (
            sleep_obs["on_shelter"] > thresholds["on_shelter"]
            and sleep_obs["sleep_phase_level"] > thresholds["sleep_phase"]
            and sleep_obs["hunger"] < thresholds["rest_hunger"]
        ):
            return "STAY"
        if (
            sleep_obs["on_shelter"] > thresholds["on_shelter"]
            and sleep_obs["shelter_role_level"] > thresholds["deep_shelter_level"]
            and sleep_obs["rest_streak_norm"] > thresholds["rest_streak"]
            and sleep_obs["hunger"] < thresholds["rest_hunger"]
        ):
            return "STAY"
        if (
            sleep_obs["on_shelter"] > thresholds["on_shelter"]
            and (
                sleep_obs["night"] > thresholds["on_shelter"]
                or sleep_obs["fatigue"] > thresholds["fatigue_to_hold"]
                or sleep_obs["sleep_debt"] > thresholds["sleep_debt_to_hold"]
            )
            and sleep_obs["hunger"] < thresholds["rest_hunger"]
        ):
            return "STAY"
        return None

    def set_runtime_reflex_scale(self, scale: float) -> None:
        """
        Set the runtime multiplier applied to reflex strengths.
        
        Parameters:
            scale (float): Desired reflex scale; must be finite. Negative values are clamped to 0.0 and the value is stored as a float in `self.current_reflex_scale`.
        """
        value = float(scale)
        if not math.isfinite(value):
            raise ValueError("non-finite reflex scale")
        self.current_reflex_scale = max(0.0, value)

    def reset_runtime_reflex_scale(self) -> None:
        """
        Restore the runtime reflex scale to the configured default.
        
        Clamps the configured reflex scale to a minimum of 0.0 and sets self.current_reflex_scale accordingly. Raises ValueError if the configured reflex scale is not finite.
        """
        value = float(self.config.reflex_scale)
        if not math.isfinite(value):
            raise ValueError("non-finite reflex scale")
        self.current_reflex_scale = max(0.0, value)

    def _act_with_training(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None,
        *,
        sample: bool,
        policy_mode: str,
        training: bool,
    ) -> BrainStep:
        try:
            return self.act(
                observation,
                bus,
                sample=sample,
                policy_mode=policy_mode,
                training=training,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'training'" not in str(exc):
                raise
            return self.act(
                observation,
                bus,
                sample=sample,
                policy_mode=policy_mode,
            )

    def act_exploration(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        policy_mode: str = "normal",
    ) -> BrainStep:
        return self._act_with_training(
            observation, bus, sample=True, policy_mode=policy_mode, training=False
        )

    def act_inference(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = False,
        policy_mode: str = "normal",
    ) -> BrainStep:
        return self._act_with_training(
            observation, bus, sample=sample, policy_mode=policy_mode, training=False
        )

    def act_train(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = True,
        policy_mode: str = "normal",
    ) -> BrainStep:
        return self._act_with_training(
            observation, bus, sample=sample, policy_mode=policy_mode, training=True
        )

    def _effective_reflex_scale(self, module_name: str) -> float:
        """
        Compute the non-negative reflex scaling factor for the given module.
        
        If reflexes are disabled or the architecture is not modular, this returns 0.0. Otherwise
        returns the product of the current runtime reflex scale and the module's configured
        multiplier, clamped to be greater than or equal to 0.0.
        
        Returns:
            float: Effective scale applied to reflex strengths for the module (>= 0.0).
        """
        if not self.config.enable_reflexes or not self.config.is_modular:
            return 0.0
        return max(
            0.0,
            float(self.current_reflex_scale)
            * float(self.config.module_reflex_scales.get(module_name, 1.0)),
        )

    def _true_monolithic_allows_food_direction_bias(self) -> bool:
        return self.config.name not in TRUE_MONOLITHIC_NO_FOOD_DIRECTION_VARIANTS

    def act(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = True,
        policy_mode: str = "normal",
        training: bool | None = None,
    ) -> BrainStep:
        """
        Choose and execute an action for the provided observation.
        
        Builds per-module proposals, optionally applies reflexes, computes arbitration and priority gating, runs action-center and motor-cortex corrections (unless `policy_mode == "reflex_only"`), and returns a populated BrainStep describing module results, logits/policies, selected intent/action, overrides, controller inputs, and the arbitration decision.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Mapping of interface observation arrays consumed by proposal modules and context interfaces.
            bus (MessageBus | None): Optional message bus for publishing per-module proposal diagnostics and final selection/execution diagnostics; pass None to disable publishing.
            sample (bool): If True, sample the executed action from the final policy distribution; if False, select the greedy argmax action.
            policy_mode (str): Execution mode, either "normal" to apply learned action-center and motor-cortex corrections, or "reflex_only" to select directly from post-reflex modular proposals. "reflex_only" requires a modular architecture with reflexes enabled.
            training (bool | None): If provided, forces training mode on/off for internal network cache and learned-arbitration behavior; if None, training mode is inferred from `sample` or an internal override.
        
        Returns:
            BrainStep: Decision container populated with per-module ModuleResult entries, action-center and motor-cortex logits/policies, combined logits with and without reflexes, the final policy and value estimate, selected intent and action indices, override flags, controller input vectors, the active `policy_mode`, and the computed `arbitration_decision`.
        
        Raises:
            ValueError: If `policy_mode` is not "normal" or "reflex_only", or if `policy_mode == "reflex_only"` is requested but the brain is not modular or reflexes are disabled.
        """
        if policy_mode not in {"normal", "reflex_only"}:
            raise ValueError(
                "Invalid policy_mode. Use 'normal' or 'reflex_only'."
            )
        if policy_mode == "reflex_only" and not self.config.is_modular:
            raise ValueError(
                "policy_mode='reflex_only' requires the modular architecture."
            )
        if policy_mode == "reflex_only" and not self.config.enable_reflexes:
            raise ValueError(
                "policy_mode='reflex_only' requires reflexes to be enabled."
            )

        runtime_training = getattr(self, "_act_training_override", None)
        if training is None:
            training_mode = bool(sample if runtime_training is None else runtime_training)
        else:
            training_mode = bool(training)
        store_cache = training_mode and policy_mode == "normal"
        proposal_sum = np.zeros(self.action_dim, dtype=float)
        action_center_input = np.zeros(0, dtype=float)
        motor_input = np.zeros(0, dtype=float)
        action_center_correction_logits = np.zeros(self.action_dim, dtype=float)
        motor_correction_logits = np.zeros(self.action_dim, dtype=float)
        value = 0.0
        arbitration = None
        direct_policy_trace_payload: Dict[str, object] = {}
        phase_logits = np.zeros(0, dtype=float)
        phase_prediction: str | None = None
        phase_prediction_confidence = 0.0
        event_attention_top_type: str | None = None
        event_attention_top_age = -1
        event_attention_entropy = 0.0
        selected_option: str | None = None
        option_age = -1
        option_termination_reason = "none"
        option_logits = np.zeros(0, dtype=float)
        option_leaf_logits = np.zeros(0, dtype=float)
        option_owned_action: str | None = None
        safety_mask_applied = False
        safety_masked_actions: list[str] = []
        external_override_count = 0
        affordance_blocked_logits = np.zeros(0, dtype=float)
        affordance_role_logits = np.zeros(0, dtype=float)
        geometry_logits = np.zeros(0, dtype=float)
        shelter_column_logits = np.zeros(0, dtype=float)
        shelter_position_logits = np.zeros(0, dtype=float)
        transition_prediction_logits = np.zeros(0, dtype=float)
        transition_rollout_prediction_logits = np.zeros(0, dtype=float)
        if self.config.is_b_series:
            if self.b_series_policy is None:
                raise RuntimeError(
                    "B-series policy unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            semantic_logits, value = self.b_series_policy.forward(
                monolithic_observation,
                store_cache=store_cache,
            )
            semantic_policy = softmax(semantic_logits)
            if sample:
                learned_semantic_action_idx = int(
                    self.rng.choice(len(B_SEMANTIC_ACTIONS), p=semantic_policy)
                )
            else:
                learned_semantic_action_idx = int(np.argmax(semantic_policy))
            learned_semantic_action = B_SEMANTIC_ACTIONS[learned_semantic_action_idx]
            semantic_action = learned_semantic_action
            semantic_action_source = "network_policy"
            semantic_action_reason = "network_argmax_or_sample"
            semantic_override_count = 0
            b_temporal_threat_trace: dict[str, object] = {}
            b_level = int(getattr(self.config, "b_level", 0))
            b_effective_level = f"B{b_level}"
            if (
                b_level == 0
                and str(getattr(self.config, "b_mode", "")) == "current_bridge"
            ):
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                ) = self._b0_current_simple_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B_CURRENT_BRIDGE_EFFECTIVE_LEVEL
            elif (
                b_level == 1
                and str(getattr(self.config, "name", "")) == B1_THREAT_GUARD_POLICY_NAME
            ):
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                ) = self._b1_threat_guard_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B1_THREAT_GUARD_EFFECTIVE_LEVEL
            elif b_level == 2 and str(getattr(self.config, "name", "")) in {
                B2_TEMPORAL_THREAT_H48_POLICY_NAME,
                B2_TEMPORAL_THREAT_H56_POLICY_NAME,
                B2_TEMPORAL_THREAT_H64_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b2_temporal_threat_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL
            elif b_level == 3 and str(getattr(self.config, "name", "")) in {
                B3_CONTACT_MEMORY_H48_POLICY_NAME,
                B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME,
                B3_CONTACT_MEMORY_H56_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b3_contact_memory_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B3_CONTACT_MEMORY_EFFECTIVE_LEVEL
            elif b_level == 3 and str(getattr(self.config, "name", "")) == (
                B3_RECURRENT_GUARD_H48_POLICY_NAME
            ):
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b3_recurrent_guard_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B3_RECURRENT_GUARD_EFFECTIVE_LEVEL
            elif b_level == 4 and str(getattr(self.config, "name", "")) in {
                B4_RECOVERY_BALANCE_H48_POLICY_NAME,
                B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME,
                B4_RECOVERY_BALANCE_H56_POLICY_NAME,
                B4_GENETIC_RECOVERY_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b4_recovery_balance_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B4_RECOVERY_BALANCE_EFFECTIVE_LEVEL
            elif b_level == 5 and str(getattr(self.config, "name", "")) in {
                B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
                B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME,
                B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME,
                B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b5_homeostatic_arbiter_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B5_HOMEOSTATIC_ARBITER_EFFECTIVE_LEVEL
            elif b_level == 6 and str(getattr(self.config, "name", "")) in {
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
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b6_risk_corridor_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                if semantic_action_source == B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE:
                    b_effective_level = B6_FUSED_RISK_RECURRENT_EFFECTIVE_LEVEL
                elif semantic_action_source == B6_RECURRENT_MEMORY_SELECTION_SOURCE:
                    b_effective_level = B6_RECURRENT_MEMORY_EFFECTIVE_LEVEL
                else:
                    b_effective_level = B6_RISK_CORRIDOR_EFFECTIVE_LEVEL
            elif b_level == 7 and str(getattr(self.config, "name", "")) in {
                B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
                B7_ENERGY_BUDGET_CORRIDOR_H48_POLICY_NAME,
                B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME,
                B7_AFFORDANCE_BUDGET_H56_POLICY_NAME,
                B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b7_affordance_budget_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B7_AFFORDANCE_BUDGET_EFFECTIVE_LEVEL
            elif b_level == 8 and str(getattr(self.config, "name", "")) in {
                B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
                B8_RETURN_VECTOR_H48_POLICY_NAME,
                B8_CORRIDOR_PLACE_MEMORY_H48_POLICY_NAME,
                B8_SPATIAL_AFFORDANCE_MAP_H56_POLICY_NAME,
                B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b8_spatial_affordance_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B8_SPATIAL_AFFORDANCE_EFFECTIVE_LEVEL
            elif b_level == 9 and str(getattr(self.config, "name", "")) in {
                B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
                B9_PATH_INTEGRATION_H48_POLICY_NAME,
                B9_ROUTE_MEMORY_H48_POLICY_NAME,
                B9_WAYPOINT_PLANNER_H56_POLICY_NAME,
                B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b9_waypoint_planner_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B9_WAYPOINT_PLANNER_EFFECTIVE_LEVEL
            elif b_level == 10 and str(getattr(self.config, "name", "")) in {
                B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
                B10_VALUE_ROUTE_EVALUATOR_H48_POLICY_NAME,
                B10_REPLAY_PLANNER_H48_POLICY_NAME,
                B10_PROSPECTIVE_REPLAY_H56_POLICY_NAME,
                B10_GENETIC_REPLAY_PLANNER_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b10_prospective_replay_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B10_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL
            elif b_level == 11 and str(getattr(self.config, "name", "")) in {
                B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
                B11_UNCERTAINTY_GATE_H48_POLICY_NAME,
                B11_NEUROMODULATED_REPLAY_H48_POLICY_NAME,
                B11_CONFIDENCE_ARBITER_H56_POLICY_NAME,
                B11_GENETIC_CONFIDENCE_GATE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b11_confidence_arbiter_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B11_CONFIDENCE_ARBITER_EFFECTIVE_LEVEL
            elif b_level == 12 and str(getattr(self.config, "name", "")) in {
                B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
                B12_ACTIVE_INFERENCE_GATE_H48_POLICY_NAME,
                B12_AFFORDANCE_ATTENTION_H48_POLICY_NAME,
                B12_PREDICTIVE_ATTENTION_H56_POLICY_NAME,
                B12_GENETIC_ATTENTION_GATE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b12_predictive_attention_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B12_PREDICTIVE_ATTENTION_EFFECTIVE_LEVEL
            elif b_level == 13 and str(getattr(self.config, "name", "")) in {
                B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
                B13_COUNTERFACTUAL_ROUTE_H48_POLICY_NAME,
                B13_AFFORDANCE_SAMPLER_H48_POLICY_NAME,
                B13_LOCAL_AFFORDANCE_SEARCH_H56_POLICY_NAME,
                B13_GENETIC_LOCAL_SEARCH_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b13_local_affordance_search_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B13_LOCAL_SEARCH_EFFECTIVE_LEVEL
            elif b_level == 14 and str(getattr(self.config, "name", "")) in {
                B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
                B14_RISK_CALIBRATED_SEARCH_H48_POLICY_NAME,
                B14_CONFIDENCE_WEIGHTED_ROUTE_H48_POLICY_NAME,
                B14_AFFORDANCE_UNCERTAINTY_H56_POLICY_NAME,
                B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b14_affordance_uncertainty_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B14_AFFORDANCE_UNCERTAINTY_EFFECTIVE_LEVEL
            elif b_level == 15 and str(getattr(self.config, "name", "")) in {
                B15_OPTION_CRITIC_H48_POLICY_NAME,
                B15_PERSISTENCE_GATE_H48_POLICY_NAME,
                B15_VALUE_GATED_OPTION_H48_POLICY_NAME,
                B15_OPTION_CRITIC_H56_POLICY_NAME,
                B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b15_option_critic_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B15_OPTION_CRITIC_EFFECTIVE_LEVEL
            elif b_level == 16 and str(getattr(self.config, "name", "")) in {
                B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
                B16_COMPETING_OPTIONS_H48_POLICY_NAME,
                B16_ACTION_SET_VOTER_H48_POLICY_NAME,
                B16_OPTION_ENSEMBLE_H56_POLICY_NAME,
                B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b16_option_ensemble_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B16_OPTION_ENSEMBLE_EFFECTIVE_LEVEL
            elif b_level == 17 and str(getattr(self.config, "name", "")) in {
                B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
                B17_AROUSAL_GATED_OPTIONS_H48_POLICY_NAME,
                B17_HOMEOSTATIC_MODULATOR_H48_POLICY_NAME,
                B17_NEUROMODULATED_ENSEMBLE_H56_POLICY_NAME,
                B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b17_neuromodulated_ensemble_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B17_NEUROMODULATED_ENSEMBLE_EFFECTIVE_LEVEL
            elif b_level == 18 and str(getattr(self.config, "name", "")) in {
                B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
                B18_METASTABLE_AROUSAL_H48_POLICY_NAME,
                B18_SYNAPTIC_TRACE_MODULATOR_H48_POLICY_NAME,
                B18_ELIGIBILITY_TRACE_H56_POLICY_NAME,
                B18_GENETIC_ELIGIBILITY_TRACE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b18_eligibility_trace_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B18_ELIGIBILITY_TRACE_EFFECTIVE_LEVEL
            elif b_level == 19 and str(getattr(self.config, "name", "")) in {
                B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
                B19_STABILITY_MEMORY_H48_POLICY_NAME,
                B19_SWITCH_SUPPRESSION_H48_POLICY_NAME,
                B19_EPISODIC_META_MEMORY_H56_POLICY_NAME,
                B19_GENETIC_META_MEMORY_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b19_episodic_meta_memory_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B19_EPISODIC_META_MEMORY_EFFECTIVE_LEVEL
            elif b_level == 20 and str(getattr(self.config, "name", "")) in {
                B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
                B20_CONTEXT_BINDING_H48_POLICY_NAME,
                B20_STABILITY_BUFFER_H48_POLICY_NAME,
                B20_WORKING_MEMORY_GATE_H56_POLICY_NAME,
                B20_GENETIC_WORKING_MEMORY_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b20_working_memory_gate_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B20_WORKING_MEMORY_GATE_EFFECTIVE_LEVEL
            elif b_level == 21 and str(getattr(self.config, "name", "")) in {
                B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
                B21_SEQUENCE_BINDING_H48_POLICY_NAME,
                B21_ROUTE_REHEARSAL_H48_POLICY_NAME,
                B21_HIPPOCAMPAL_REPLAY_H56_POLICY_NAME,
                B21_GENETIC_REPLAY_GATE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b21_hippocampal_replay_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B21_HIPPOCAMPAL_REPLAY_EFFECTIVE_LEVEL
            elif b_level == 22 and str(getattr(self.config, "name", "")) in {
                B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
                B22_FORWARD_MODEL_GATE_H48_POLICY_NAME,
                B22_ROUTE_VIABILITY_SIM_H48_POLICY_NAME,
                B22_PROSPECTIVE_MAP_REPLAY_H56_POLICY_NAME,
                B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b22_prospective_replay_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B22_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL
            elif b_level == 23 and str(getattr(self.config, "name", "")) in {
                B23_CONFLICT_MONITOR_H48_POLICY_NAME,
                B23_ERROR_GATED_REPLAY_H48_POLICY_NAME,
                B23_ABORT_CONFLICT_ARBITER_H48_POLICY_NAME,
                B23_CONFLICT_MONITOR_H56_POLICY_NAME,
                B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b23_conflict_monitor_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B23_CONFLICT_MONITOR_EFFECTIVE_LEVEL
            elif b_level == 24 and str(getattr(self.config, "name", "")) in {
                B24_PRECISION_CONFLICT_H48_POLICY_NAME,
                B24_PREDICTION_PRECISION_GATE_H48_POLICY_NAME,
                B24_RELIABILITY_ABORT_H48_POLICY_NAME,
                B24_PRECISION_CONFLICT_H56_POLICY_NAME,
                B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b24_precision_conflict_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B24_PRECISION_CONFLICT_EFFECTIVE_LEVEL
            elif b_level == 25 and str(getattr(self.config, "name", "")) in {
                B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
                B25_CONFIDENCE_CALIBRATION_H48_POLICY_NAME,
                B25_UNCERTAINTY_INTEGRATOR_H48_POLICY_NAME,
                B25_METACOGNITIVE_CONFIDENCE_H56_POLICY_NAME,
                B25_GENETIC_METACOGNITION_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b25_metacognitive_confidence_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B25_METACOGNITIVE_CONFIDENCE_EFFECTIVE_LEVEL
            elif b_level == 26 and str(getattr(self.config, "name", "")) in {
                B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
                B26_SETPOINT_DRIFT_H48_POLICY_NAME,
                B26_ERROR_SUPPRESSION_H48_POLICY_NAME,
                B26_ALLOSTATIC_PREDICTION_H56_POLICY_NAME,
                B26_GENETIC_ALLOSTASIS_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b26_allostatic_prediction_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B26_ALLOSTATIC_PREDICTION_EFFECTIVE_LEVEL
            elif b_level == 27 and str(getattr(self.config, "name", "")) in {
                B27_AROUSAL_GAIN_H48_POLICY_NAME,
                B27_STRESS_MODULATION_H48_POLICY_NAME,
                B27_ENERGY_AROUSAL_H48_POLICY_NAME,
                B27_AROUSAL_GAIN_H56_POLICY_NAME,
                B27_GENETIC_AROUSAL_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b27_arousal_gain_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B27_AROUSAL_GAIN_EFFECTIVE_LEVEL
            elif b_level == 28 and str(getattr(self.config, "name", "")) in {
                B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                B28_THREAT_FOCUS_ATTENTION_H48_POLICY_NAME,
                B28_HOMEOSTATIC_ATTENTION_H48_POLICY_NAME,
                B28_INTEROCEPTIVE_ATTENTION_H56_POLICY_NAME,
                B28_GENETIC_ATTENTION_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b28_interoceptive_attention_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B28_INTEROCEPTIVE_ATTENTION_EFFECTIVE_LEVEL
            elif b_level == 29 and str(getattr(self.config, "name", "")) in {
                B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
                B29_THREAT_SALIENCE_GATE_H48_POLICY_NAME,
                B29_HOMEOSTATIC_SALIENCE_GATE_H48_POLICY_NAME,
                B29_SALIENCE_COMPETITION_H56_POLICY_NAME,
                B29_GENETIC_SALIENCE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b29_salience_competition_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B29_SALIENCE_COMPETITION_EFFECTIVE_LEVEL
            elif b_level == 30 and str(getattr(self.config, "name", "")) in {
                B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
                B30_GO_NOGO_BALANCE_H48_POLICY_NAME,
                B30_THREAT_INHIBITION_GATE_H48_POLICY_NAME,
                B30_BASAL_GANGLIA_GATE_H56_POLICY_NAME,
                B30_GENETIC_ACTION_GATE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b30_basal_ganglia_gate_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B30_BASAL_GANGLIA_GATE_EFFECTIVE_LEVEL
            elif b_level == 31 and str(getattr(self.config, "name", "")) in {
                B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
                B31_TONIC_DOPAMINE_GATE_H48_POLICY_NAME,
                B31_PHASIC_DOPAMINE_GATE_H48_POLICY_NAME,
                B31_DOPAMINE_PREDICTION_ERROR_H56_POLICY_NAME,
                B31_GENETIC_DOPAMINE_GATE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b31_dopamine_prediction_error_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B31_DOPAMINE_PREDICTION_ERROR_EFFECTIVE_LEVEL
            elif b_level == 32 and str(getattr(self.config, "name", "")) in {
                B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
                B32_ADVANTAGE_VALUE_GATE_H48_POLICY_NAME,
                B32_CRITIC_STABILITY_H48_POLICY_NAME,
                B32_ACTOR_CRITIC_VALUE_H56_POLICY_NAME,
                B32_GENETIC_ACTOR_CRITIC_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b32_actor_critic_value_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B32_ACTOR_CRITIC_VALUE_EFFECTIVE_LEVEL
            elif b_level == 33 and str(getattr(self.config, "name", "")) in {
                B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
                B33_BOOTSTRAPPED_VALUE_GATE_H48_POLICY_NAME,
                B33_REWARD_TRACE_CRITIC_H48_POLICY_NAME,
                B33_TD_ERROR_DECOMPOSITION_H56_POLICY_NAME,
                B33_GENETIC_TD_VALUE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b33_td_error_decomposition_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B33_TD_ERROR_DECOMPOSITION_EFFECTIVE_LEVEL
            elif b_level == 34 and str(getattr(self.config, "name", "")) in {
                B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
                B34_DELAYED_CREDIT_GATE_H48_POLICY_NAME,
                B34_SYNAPTIC_TAGGING_H48_POLICY_NAME,
                B34_ELIGIBILITY_CREDIT_H56_POLICY_NAME,
                B34_GENETIC_ELIGIBILITY_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b34_eligibility_credit_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B34_ELIGIBILITY_CREDIT_EFFECTIVE_LEVEL
            elif b_level == 35 and str(getattr(self.config, "name", "")) in {
                B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
                B35_TRANSITION_ERROR_GATE_H48_POLICY_NAME,
                B35_MODEL_CONFIDENCE_H48_POLICY_NAME,
                B35_FORWARD_MODEL_VALUE_H56_POLICY_NAME,
                B35_GENETIC_FORWARD_MODEL_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b35_forward_model_value_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B35_FORWARD_MODEL_VALUE_EFFECTIVE_LEVEL
            elif b_level == 36 and str(getattr(self.config, "name", "")) in {
                B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
                B36_BELIEF_ERROR_GATE_H48_POLICY_NAME,
                B36_CONTEXT_INFERENCE_H48_POLICY_NAME,
                B36_LATENT_BELIEF_STATE_H56_POLICY_NAME,
                B36_GENETIC_BELIEF_STATE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b36_latent_belief_state_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B36_LATENT_BELIEF_STATE_EFFECTIVE_LEVEL
            elif b_level == 37 and str(getattr(self.config, "name", "")) in {
                B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
                B37_INTERO_EXTERO_FACTOR_H48_POLICY_NAME,
                B37_FACTOR_CONFIDENCE_H48_POLICY_NAME,
                B37_STATE_FACTOR_GATE_H56_POLICY_NAME,
                B37_GENETIC_STATE_FACTOR_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b37_state_factor_gate_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B37_STATE_FACTOR_GATE_EFFECTIVE_LEVEL
            elif b_level == 38 and str(getattr(self.config, "name", "")) in {
                B38_FACTOR_ATTENTION_H48_POLICY_NAME,
                B38_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
                B38_CONFIDENCE_ATTENTION_H48_POLICY_NAME,
                B38_FACTOR_ATTENTION_H56_POLICY_NAME,
                B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b38_factor_attention_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B38_FACTOR_ATTENTION_EFFECTIVE_LEVEL
            elif b_level == 39 and str(getattr(self.config, "name", "")) in {
                B39_ATTENTION_BINDING_H48_POLICY_NAME,
                B39_CROSS_FACTOR_BINDING_H48_POLICY_NAME,
                B39_CONTEXT_BINDING_ATTENTION_H48_POLICY_NAME,
                B39_ATTENTION_BINDING_H56_POLICY_NAME,
                B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b39_attention_binding_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B39_ATTENTION_BINDING_EFFECTIVE_LEVEL
            elif b_level == 40 and str(getattr(self.config, "name", "")) in {
                B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
                B40_SENSORY_WORKSPACE_H48_POLICY_NAME,
                B40_CONTEXT_WORKSPACE_H48_POLICY_NAME,
                B40_GLOBAL_WORKSPACE_H56_POLICY_NAME,
                B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b40_global_workspace_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B40_GLOBAL_WORKSPACE_EFFECTIVE_LEVEL
            elif b_level == 41 and str(getattr(self.config, "name", "")) in {
                B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
                B41_INHIBITORY_CONTROL_H48_POLICY_NAME,
                B41_GOAL_CONTEXT_SELECTOR_H48_POLICY_NAME,
                B41_EXECUTIVE_WORKSPACE_H56_POLICY_NAME,
                B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b41_executive_workspace_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B41_EXECUTIVE_WORKSPACE_EFFECTIVE_LEVEL
            elif b_level == 42 and str(getattr(self.config, "name", "")) in {
                B42_ERROR_MONITOR_H48_POLICY_NAME,
                B42_CONFLICT_MONITOR_H48_POLICY_NAME,
                B42_PERFORMANCE_MONITOR_H48_POLICY_NAME,
                B42_ERROR_MONITOR_H56_POLICY_NAME,
                B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b42_error_monitor_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B42_ERROR_MONITOR_EFFECTIVE_LEVEL
            elif b_level == 43 and str(getattr(self.config, "name", "")) in {
                B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
                B43_AROUSAL_PRECISION_H48_POLICY_NAME,
                B43_THRESHOLD_ADAPTATION_H48_POLICY_NAME,
                B43_ADAPTIVE_PRECISION_H56_POLICY_NAME,
                B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b43_adaptive_precision_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B43_ADAPTIVE_PRECISION_EFFECTIVE_LEVEL
            elif b_level == 44 and str(getattr(self.config, "name", "")) in {
                B44_THALAMIC_RELAY_H48_POLICY_NAME,
                B44_SENSORY_RELAY_H48_POLICY_NAME,
                B44_CONTEXT_RELAY_H48_POLICY_NAME,
                B44_THALAMIC_RELAY_H56_POLICY_NAME,
                B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b44_thalamic_relay_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B44_THALAMIC_RELAY_EFFECTIVE_LEVEL
            elif b_level == 45 and str(getattr(self.config, "name", "")) in {
                B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
                B45_SENSORY_INHIBITION_H48_POLICY_NAME,
                B45_CONTEXT_INHIBITION_H48_POLICY_NAME,
                B45_RETICULAR_INHIBITION_H56_POLICY_NAME,
                B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b45_reticular_inhibition_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B45_RETICULAR_INHIBITION_EFFECTIVE_LEVEL
            elif b_level == 46 and str(getattr(self.config, "name", "")) in {
                B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
                B46_FEEDBACK_GAIN_H48_POLICY_NAME,
                B46_CONTEXT_FEEDBACK_H48_POLICY_NAME,
                B46_CORTICOTHALAMIC_FEEDBACK_H56_POLICY_NAME,
                B46_GENETIC_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b46_corticothalamic_feedback_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B46_CORTICOTHALAMIC_FEEDBACK_EFFECTIVE_LEVEL
            elif b_level == 47 and str(getattr(self.config, "name", "")) in {
                B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
                B47_PHASE_LOCKING_H48_POLICY_NAME,
                B47_COHERENCE_GATE_H48_POLICY_NAME,
                B47_OSCILLATORY_SYNCHRONY_H56_POLICY_NAME,
                B47_GENETIC_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b47_oscillatory_synchrony_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B47_OSCILLATORY_SYNCHRONY_EFFECTIVE_LEVEL
            elif b_level == 48 and str(getattr(self.config, "name", "")) in {
                B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
                B48_TIMING_ERROR_CORRECTION_H48_POLICY_NAME,
                B48_PREDICTIVE_TIMING_H48_POLICY_NAME,
                B48_CEREBELLAR_TIMING_H56_POLICY_NAME,
                B48_GENETIC_CEREBELLAR_TIMING_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b48_cerebellar_timing_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B48_CEREBELLAR_TIMING_EFFECTIVE_LEVEL
            elif b_level == 49 and str(getattr(self.config, "name", "")) in {
                B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
                B49_DIRECT_PATH_FACILITATION_H48_POLICY_NAME,
                B49_INDIRECT_PATH_SUPPRESSION_H48_POLICY_NAME,
                B49_STRIATAL_ACTION_GATE_H56_POLICY_NAME,
                B49_GENETIC_STRIATAL_GATE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b49_striatal_action_gate_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B49_STRIATAL_ACTION_GATE_EFFECTIVE_LEVEL
            elif b_level == 50 and str(getattr(self.config, "name", "")) in {
                B50_HABIT_CHUNKING_H48_POLICY_NAME,
                B50_ACTION_CHUNK_VALUE_H48_POLICY_NAME,
                B50_HABIT_STABILITY_H48_POLICY_NAME,
                B50_HABIT_CHUNKING_H56_POLICY_NAME,
                B50_GENETIC_HABIT_CHUNKING_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b50_habit_chunking_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B50_HABIT_CHUNKING_EFFECTIVE_LEVEL
            elif b_level == 51 and str(getattr(self.config, "name", "")) in {
                B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
                B51_REWARD_PREDICTION_GAIN_H48_POLICY_NAME,
                B51_NOVELTY_MODULATED_HABIT_H48_POLICY_NAME,
                B51_DOPAMINERGIC_HABIT_MODULATION_H56_POLICY_NAME,
                B51_GENETIC_DOPAMINE_HABIT_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b51_dopaminergic_habit_modulation_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B51_DOPAMINERGIC_HABIT_MODULATION_EFFECTIVE_LEVEL
            elif b_level == 52 and str(getattr(self.config, "name", "")) in {
                B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
                B52_ATTENTION_GAIN_H48_POLICY_NAME,
                B52_UNCERTAINTY_RELEASE_H48_POLICY_NAME,
                B52_CHOLINERGIC_PRECISION_GATE_H56_POLICY_NAME,
                B52_GENETIC_CHOLINERGIC_PRECISION_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b52_cholinergic_precision_gate_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B52_CHOLINERGIC_PRECISION_GATE_EFFECTIVE_LEVEL
            elif b_level == 53 and str(getattr(self.config, "name", "")) in {
                B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
                B53_SURPRISE_GAIN_H48_POLICY_NAME,
                B53_STRESS_PRECISION_H48_POLICY_NAME,
                B53_NORADRENERGIC_AROUSAL_GAIN_H56_POLICY_NAME,
                B53_GENETIC_AROUSAL_PRECISION_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b53_noradrenergic_arousal_gain_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B53_NORADRENERGIC_AROUSAL_GAIN_EFFECTIVE_LEVEL
            elif b_level == 54 and str(getattr(self.config, "name", "")) in {
                B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
                B54_IMPULSE_SUPPRESSION_H48_POLICY_NAME,
                B54_PATIENCE_BALANCE_H48_POLICY_NAME,
                B54_SEROTONERGIC_PATIENCE_GATE_H56_POLICY_NAME,
                B54_GENETIC_SEROTONIN_PATIENCE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b54_serotonergic_patience_gate_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B54_SEROTONERGIC_PATIENCE_GATE_EFFECTIVE_LEVEL
            elif b_level == 55 and str(getattr(self.config, "name", "")) in {
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
                B55_SATIETY_RECOVERY_BALANCE_H48_POLICY_NAME,
                B55_SLEEP_HUNGER_ARBITER_H48_POLICY_NAME,
                B55_HYPOTHALAMIC_DRIVE_COUPLING_H56_POLICY_NAME,
                B55_GENETIC_HYPOTHALAMIC_DRIVE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b55_hypothalamic_drive_coupling_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B55_HYPOTHALAMIC_DRIVE_COUPLING_EFFECTIVE_LEVEL
            elif b_level == 56 and str(getattr(self.config, "name", "")) in {
                B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
                B56_CORTISOL_RECOVERY_BALANCE_H48_POLICY_NAME,
                B56_STRESS_LOAD_GATE_H48_POLICY_NAME,
                B56_HPA_STRESS_AXIS_H56_POLICY_NAME,
                B56_GENETIC_HPA_STRESS_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b56_hpa_stress_axis_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B56_HPA_STRESS_AXIS_EFFECTIVE_LEVEL
            elif b_level == 57 and str(getattr(self.config, "name", "")) in {
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
                B57_VISCERAL_SALIENCE_GATE_H48_POLICY_NAME,
                B57_STRESS_DRIVE_AWARENESS_H48_POLICY_NAME,
                B57_INSULAR_INTEROCEPTIVE_AWARENESS_H56_POLICY_NAME,
                B57_GENETIC_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b57_insular_interoceptive_awareness_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B57_INSULAR_INTEROCEPTIVE_AWARENESS_EFFECTIVE_LEVEL
            elif b_level == 58 and str(getattr(self.config, "name", "")) in {
                B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
                B58_ERROR_SALIENCE_GATE_H48_POLICY_NAME,
                B58_CONFLICT_RESOLUTION_BALANCE_H48_POLICY_NAME,
                B58_ACC_CONFLICT_MONITOR_H56_POLICY_NAME,
                B58_GENETIC_ACC_CONFLICT_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b58_acc_conflict_monitor_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B58_ACC_CONFLICT_MONITOR_EFFECTIVE_LEVEL
            elif b_level == 59 and str(getattr(self.config, "name", "")) in {
                B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
                B59_WORKING_SET_STABILITY_H48_POLICY_NAME,
                B59_EXECUTIVE_TASK_SET_H48_POLICY_NAME,
                B59_PREFRONTAL_GOAL_CONTEXT_H56_POLICY_NAME,
                B59_GENETIC_PREFRONTAL_CONTROL_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b59_prefrontal_goal_context_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B59_PREFRONTAL_GOAL_CONTEXT_EFFECTIVE_LEVEL
            elif b_level == 60 and str(getattr(self.config, "name", "")) in {
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
                B60_REVERSAL_VALUE_GATE_H48_POLICY_NAME,
                B60_GOAL_OUTCOME_PREDICTION_H48_POLICY_NAME,
                B60_ORBITOFRONTAL_OUTCOME_VALUE_H56_POLICY_NAME,
                B60_GENETIC_ORBITOFRONTAL_VALUE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b60_orbitofrontal_outcome_value_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B60_ORBITOFRONTAL_OUTCOME_VALUE_EFFECTIVE_LEVEL
            elif b_level == 61 and str(getattr(self.config, "name", "")) in {
                B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
                B61_THREAT_VALUE_TAG_H48_POLICY_NAME,
                B61_SAFETY_PREDICTION_GATE_H48_POLICY_NAME,
                B61_AMYGDALA_SAFETY_VALUE_H56_POLICY_NAME,
                B61_GENETIC_AMYGDALA_SAFETY_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b61_amygdala_safety_value_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B61_AMYGDALA_SAFETY_VALUE_EFFECTIVE_LEVEL
            elif b_level == 62 and str(getattr(self.config, "name", "")) in {
                B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
                B62_FREEZE_FLEE_BALANCE_H48_POLICY_NAME,
                B62_SHELTER_DEFENSE_GATE_H48_POLICY_NAME,
                B62_DEFENSIVE_MODE_SELECTOR_H56_POLICY_NAME,
                B62_GENETIC_DEFENSIVE_MODE_H48_POLICY_NAME,
            }:
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                    b_temporal_threat_trace,
                ) = self._b62_defensive_mode_selector_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B62_DEFENSIVE_MODE_SELECTOR_EFFECTIVE_LEVEL
            semantic_action_idx = int(B_SEMANTIC_ACTION_TO_INDEX[semantic_action])
            bridge_observation = observation
            if (
                semantic_action == "MOVE_TO_SHELTER"
                and (
                    float(
                        b_temporal_threat_trace.get("b_predator_trace_pressure", 0.0)
                        or 0.0
                    )
                    >= 0.50
                    or float(
                        b_temporal_threat_trace.get(
                            "b_predator_memory_pressure",
                            0.0,
                        )
                        or 0.0
                    )
                    >= 0.85
                )
            ):
                meta = observation.get("meta")
                if isinstance(meta, dict):
                    bridge_meta = dict(meta)
                    bridge_meta["predator_smell_strength"] = max(
                        self._b_series_float(bridge_meta, "predator_smell_strength"),
                        float(
                            b_temporal_threat_trace[
                                "b_temporal_threat_pressure"
                            ]
                        ),
                    )
                    bridge_observation = dict(observation)
                    bridge_observation["meta"] = bridge_meta
            bridge_decision = bridge_b_semantic_action(
                semantic_action,
                bridge_observation,
                rng=self.rng,
                sample=bool(sample),
            )
            bridge_meta = bridge_observation.get("meta")
            if (
                b_level in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
                and semantic_action == "MOVE_TO_FOOD"
                and isinstance(bridge_meta, dict)
                and str(bridge_meta.get("map_template", "")) == "corridor_escape"
                and str(
                    b_temporal_threat_trace.get("b6_decision", "")
                ).startswith("corridor_commitment")
            ):
                transitions = bridge_meta.get("local_transition_consequences")
                transitions = transitions if isinstance(transitions, dict) else {}
                affordances = bridge_meta.get("local_affordances")
                affordances = affordances if isinstance(affordances, dict) else {}
                up_transition = transitions.get("MOVE_UP")
                up_transition = up_transition if isinstance(up_transition, dict) else {}
                right_transition = transitions.get("MOVE_RIGHT")
                right_transition = (
                    right_transition if isinstance(right_transition, dict) else {}
                )
                up_affordance = affordances.get("MOVE_UP")
                up_affordance = up_affordance if isinstance(up_affordance, dict) else {}
                right_affordance = affordances.get("MOVE_RIGHT")
                right_affordance = (
                    right_affordance if isinstance(right_affordance, dict) else {}
                )
                up_delta = self._b_series_float(up_transition, "food_dist_delta")
                right_delta = self._b_series_float(right_transition, "food_dist_delta")
                try:
                    food_dist = float(bridge_meta.get("food_dist", 0.0))
                except (TypeError, ValueError):
                    food_dist = 0.0
                if not np.isfinite(food_dist):
                    food_dist = 0.0
                if (
                    not bool(right_affordance.get("blocked", False))
                    and (
                        bool(right_transition.get("next_cell_has_food", False))
                        or (food_dist > 7.0 and right_delta >= 0.0)
                    )
                ):
                    bridge_decision = replace(
                        bridge_decision,
                        primitive_action="MOVE_RIGHT",
                        reason="b6_corridor_horizontal_commitment",
                        food_delta_used=float(right_delta),
                    )
                elif (
                    not bool(up_affordance.get("blocked", False))
                    and food_dist <= 7.0
                    and up_delta > 0.0
                ):
                    bridge_decision = replace(
                        bridge_decision,
                        primitive_action="MOVE_UP",
                        reason="b6_corridor_vertical_commitment",
                        food_delta_used=float(up_delta),
                    )
            action_idx = int(bridge_decision.primitive_action_idx)
            motor_action_idx = action_idx
            action_intent_idx = action_idx
            action_intent_without_reflex_idx = action_idx
            action_without_reflex_idx = action_idx
            primitive_logits = np.zeros(self.action_dim, dtype=float)
            primitive_logits[action_idx] = 6.0
            total_logits_without_reflex = primitive_logits.copy()
            total_logits = primitive_logits.copy()
            action_center_logits = primitive_logits.copy()
            action_center_policy = softmax(action_center_logits)
            policy = softmax(total_logits)
            proposal_sum = total_logits.copy()
            module_results = [
                ModuleResult(
                    interface=None,
                    name=B_SERIES_POLICY_NAME,
                    observation_key=B_SERIES_POLICY_NAME,
                    observation=monolithic_observation.copy(),
                    logits=primitive_logits.copy(),
                    probs=policy.copy(),
                    active=True,
                    reflex=None,
                    neural_logits=primitive_logits.copy(),
                    reflex_delta_logits=np.zeros_like(primitive_logits),
                    post_reflex_logits=primitive_logits.copy(),
                )
            ]
            module_results[0].valence_role = "semantic_bridge"
            module_results[0].gate_weight = 1.0
            module_results[0].gated_logits = primitive_logits.copy()
            module_results[0].contribution_share = 1.0
            module_results[0].intent_before_gating = bridge_decision.semantic_action
            module_results[0].intent_after_gating = bridge_decision.primitive_action
            arbitration = self._true_monolithic_arbitration_decision(
                module_name=B_SERIES_POLICY_NAME,
                action_idx=action_idx,
            )
            b_transfer_report = getattr(self, "b_series_transfer_report", None) or {}
            b_parent_level_raw = getattr(self.config, "b_parent_level", None)
            b_parent_level = (
                None if b_parent_level_raw is None else int(b_parent_level_raw)
            )
            b_transfer_coverage = b_transfer_report.get("coverage")
            try:
                b_transfer_coverage_value = (
                    None
                    if b_transfer_coverage is None
                    else round(float(b_transfer_coverage), 6)
                )
            except (TypeError, ValueError):
                b_transfer_coverage_value = None
            direct_policy_trace_payload = {
                "b_level": int(self.config.b_level),
                "b_effective_level": b_effective_level,
                "b_mode": str(self.config.b_mode),
                "b_parent_level": b_parent_level,
                "b_transfer_source_checkpoint": b_transfer_report.get(
                    "source_checkpoint"
                ),
                "b_transfer_coverage": b_transfer_coverage_value,
                **b_temporal_threat_trace,
                "semantic_action": semantic_action,
                "semantic_action_idx": int(semantic_action_idx),
                "learned_semantic_action": learned_semantic_action,
                "learned_semantic_action_idx": int(learned_semantic_action_idx),
                "semantic_action_source": semantic_action_source,
                "semantic_action_reason": semantic_action_reason,
                "semantic_override_count": int(semantic_override_count),
                "semantic_logits": np.asarray(semantic_logits, dtype=float)
                .round(6)
                .tolist(),
                "semantic_policy": np.asarray(semantic_policy, dtype=float)
                .round(6)
                .tolist(),
                "bridge_primitive_action": bridge_decision.primitive_action,
                "bridge_reason": bridge_decision.reason,
                "blocked_mask": dict(bridge_decision.blocked_mask),
                "food_delta_used": round(float(bridge_decision.food_delta_used), 6),
                "shelter_delta_used": round(
                    float(bridge_decision.shelter_delta_used),
                    6,
                ),
                "external_override_count": int(
                    bridge_decision.external_override_count
                ),
            }
            external_override_count = int(bridge_decision.external_override_count)
            motor_override = False
            final_reflex_override = False
        elif self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
            )
            monolithic_observation = self._build_monolithic_observation(observation)
            if hasattr(self.true_monolithic_policy, "set_runtime_observation_meta"):
                self.true_monolithic_policy.set_runtime_observation_meta(
                    observation.get("meta", {})
                )
            hidden_before = self.snapshot_direct_policy_hidden_state()
            direct_forward = self.true_monolithic_policy.forward(
                monolithic_observation,
                store_cache=store_cache,
            )
            if len(direct_forward) == 4:
                policy_logits, value, option_logits_raw, phase_logits_raw = direct_forward
                option_logits = np.asarray(option_logits_raw, dtype=float).copy()
            elif len(direct_forward) == 3:
                policy_logits, value, aux_logits_raw = direct_forward
                phase_logits_raw = (
                    aux_logits_raw if self.config.direct_policy_phase_head else None
                )
                if self.config.direct_policy_option_head:
                    option_logits = np.asarray(aux_logits_raw, dtype=float).copy()
            else:
                policy_logits, value = direct_forward
                phase_logits_raw = None
            if phase_logits_raw is not None:
                phase_logits = np.asarray(phase_logits_raw, dtype=float).copy()
                phase_probs = softmax(phase_logits)
                phase_prediction_idx = int(np.argmax(phase_probs))
                phase_prediction = PHASE_LABELS[phase_prediction_idx]
                phase_prediction_confidence = float(phase_probs[phase_prediction_idx])
            hidden_after = self.snapshot_direct_policy_hidden_state()
            if hidden_after is not None:
                hidden_before_array = (
                    hidden_before
                    if hidden_before is not None
                    else np.zeros_like(hidden_after, dtype=float)
                )
                direct_policy_trace_payload = {
                    "recurrent_hidden_norm": round(float(np.linalg.norm(hidden_after)), 6),
                    "recurrent_hidden_delta_norm": round(
                        float(np.linalg.norm(hidden_after - hidden_before_array)),
                        6,
                    ),
                    "hidden_reset_event": bool(self._direct_policy_hidden_reset_pending),
                    "architecture_metadata": {
                        "direct_policy_recurrent": bool(self.config.direct_policy_recurrent),
                        "direct_policy_hidden_dims": list(self.config.direct_policy_hidden_dims),
                        "direct_policy_phase_head": bool(self.config.direct_policy_phase_head),
                        "direct_policy_event_attention": bool(
                            self.config.direct_policy_event_attention
                        ),
                        "direct_policy_event_buffer_size": int(
                            self.config.direct_policy_event_buffer_size
                        ),
                        "direct_policy_option_head": bool(
                            self.config.direct_policy_option_head
                        ),
                        "direct_policy_owned_option_controller": bool(
                            self.config.direct_policy_owned_option_controller
                        ),
                        "direct_policy_option_ttl": int(
                            self.config.direct_policy_option_ttl
                        ),
                        "direct_policy_affordance_head": bool(
                            self.config.direct_policy_affordance_head
                        ),
                        "direct_policy_affordance_feedback": bool(
                            self.config.direct_policy_affordance_feedback
                        ),
                        "direct_policy_geometry_head": bool(
                            self.config.direct_policy_geometry_head
                        ),
                        "direct_policy_shelter_column_head": bool(
                            self.config.direct_policy_shelter_column_head
                        ),
                        "direct_policy_shelter_position_head": bool(
                            self.config.direct_policy_shelter_position_head
                        ),
                        "direct_policy_local_affordance_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_affordance_inputs",
                                False,
                            )
                        ),
                        "direct_policy_local_spatial_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_spatial_inputs",
                                False,
                            )
                        ),
                        "direct_policy_local_transition_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_transition_inputs",
                                False,
                            )
                        ),
                        "direct_policy_local_transition_rollout_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_transition_rollout_inputs",
                                False,
                            )
                        ),
                        "direct_policy_transition_prediction_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_prediction_head",
                                False,
                            )
                        ),
                        "direct_policy_transition_prediction_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_prediction_feedback",
                                False,
                            )
                        ),
                        "direct_policy_transition_rollout_prediction_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_rollout_prediction_head",
                                False,
                            )
                        ),
                        "direct_policy_transition_rollout_prediction_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_rollout_prediction_feedback",
                                False,
                            )
                        ),
                        "direct_policy_handoff_teacher": bool(
                            self.config.direct_policy_handoff_teacher
                        ),
                        "direct_policy_handoff_option_teacher": bool(
                            getattr(
                                self.config,
                                "direct_policy_handoff_option_teacher",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_action_teacher": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_action_teacher",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_release_sequence_teacher": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_release_sequence_teacher",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_release_sequence_replay_boost": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_release_sequence_replay_boost",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_release_sequence_distill": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_release_sequence_distill",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_sequence_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_sequence_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_family_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_family_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_handoff_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_handoff_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_trajectory_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_trajectory_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_cycle_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_cycle_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_trace_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_trace_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_rollout_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_rollout_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_frontier_teacher_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_frontier_teacher_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_replayable_teacher_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_replayable_teacher_distillation",
                                False,
                            )
                        ),
                        "direct_policy_continuation_replay_passes": int(
                            getattr(
                                self.config,
                                "direct_policy_continuation_replay_passes",
                                0,
                            )
                        ),
                        "direct_policy_continuation_replay_lr_scale": float(
                            getattr(
                                self.config,
                                "direct_policy_continuation_replay_lr_scale",
                                0.0,
                            )
                        ),
                        "direct_policy_continuation_margin_weight": float(
                            getattr(
                                self.config,
                                "direct_policy_continuation_margin_weight",
                                0.0,
                            )
                        ),
                        "direct_policy_phase_option_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_phase_option_feedback",
                                False,
                            )
                        ),
                        "direct_policy_option_transition_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_transition_feedback",
                                False,
                            )
                        ),
                        "direct_policy_option_termination_cooldown": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_termination_cooldown",
                                False,
                            )
                        ),
                        "direct_policy_option_action_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_head",
                                False,
                            )
                        ),
                        "direct_policy_option_decoder_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_decoder_state",
                                False,
                            )
                        ),
                        "direct_policy_option_recurrent_dynamics": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_recurrent_dynamics",
                                False,
                            )
                        ),
                        "direct_policy_option_sequence_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_sequence_head",
                                False,
                            )
                        ),
                        "direct_policy_option_decoder_recurrent_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_decoder_recurrent_state",
                                False,
                            )
                        ),
                        "direct_policy_option_action_transition_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_transition_state",
                                False,
                            )
                        ),
                        "direct_policy_option_action_controller_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_controller_state",
                                False,
                            )
                        ),
                        "direct_policy_option_action_token_decoder": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_token_decoder",
                                False,
                            )
                        ),
                        "direct_policy_option_action_recurrent_core": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_recurrent_core",
                                False,
                            )
                        ),
                        "direct_policy_option_action_separate_recurrent_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_separate_recurrent_head",
                                False,
                            )
                        ),
                        "direct_policy_option_action_separate_policy_path": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_separate_policy_path",
                                False,
                            )
                        ),
                        "direct_policy_option_action_separate_backbone": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_separate_backbone",
                                False,
                            )
                        ),
                        "direct_policy_executive_physiology_option_gating": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_physiology_option_gating",
                                False,
                            )
                        ),
                        "direct_policy_executive_affordance_action_gating": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_affordance_action_gating",
                                False,
                            )
                        ),
                        "direct_policy_executive_option_action_masking": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_option_action_masking",
                                False,
                            )
                        ),
                        "direct_policy_executive_event_release_latching": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_event_release_latching",
                                False,
                            )
                        ),
                        "direct_policy_executive_event_release_action_commitment": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_event_release_action_commitment",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_phase_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_phase_state",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_exit_contract": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_exit_contract",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_substate_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_substate_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_continuation": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_continuation",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_guidance": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_guidance",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_commitment": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_commitment",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_heading_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_heading_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_smell_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_smell_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_corridor_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_corridor_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_corridor_affordance_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_corridor_affordance_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_food_return": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_food_return",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_food_vector_return": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_food_vector_return",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_food_path_return": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_food_path_return",
                                False,
                            )
                        ),
                    },
                }
            if phase_prediction is not None:
                direct_policy_trace_payload.update(
                    {
                        "phase_prediction": phase_prediction,
                        "phase_prediction_confidence": round(
                            float(phase_prediction_confidence),
                            6,
                        ),
                    }
                )
            attention_summary = getattr(
                self.true_monolithic_policy,
                "last_attention_summary",
                None,
            )
            if isinstance(attention_summary, dict):
                event_attention_top_type = attention_summary.get(
                    "event_attention_top_type"
                )
                event_attention_top_age = int(
                    attention_summary.get("event_attention_top_age", -1)
                )
                event_attention_entropy = float(
                    attention_summary.get("event_attention_entropy", 0.0)
                )
                if event_attention_top_type is not None:
                    direct_policy_trace_payload.update(
                        {
                            "event_attention_top_type": str(
                                event_attention_top_type
                            ),
                            "event_attention_top_age": int(
                                event_attention_top_age
                            ),
                            "event_attention_entropy": round(
                                float(event_attention_entropy),
                                6,
                            ),
                        }
                    )
            option_summary = getattr(self.true_monolithic_policy, "last_option_summary", None)
            if isinstance(option_summary, dict):
                selected_option_raw = option_summary.get("selected_option")
                if selected_option_raw is not None:
                    selected_option = str(selected_option_raw)
                    option_age = int(option_summary.get("option_age", -1))
                    option_termination_reason = str(
                        option_summary.get("option_termination_reason", "none")
                    )
                    option_logits = np.asarray(
                        option_summary.get("option_logits", []),
                        dtype=float,
                    ).copy()
                    direct_policy_trace_payload.update(
                        {
                            "selected_option": selected_option,
                            "option_age": int(option_age),
                            "option_termination_reason": option_termination_reason,
                            "option_logits": option_logits.round(6).tolist(),
                        }
                    )
                    option_leaf_logits = np.asarray(
                        option_summary.get("option_leaf_logits", []),
                        dtype=float,
                    ).copy()
                    option_owned_action_raw = option_summary.get("option_owned_action")
                    option_owned_action = (
                        None
                        if option_owned_action_raw is None
                        else str(option_owned_action_raw)
                    )
                    safety_mask_applied = bool(
                        option_summary.get("safety_mask_applied", False)
                    )
                    safety_masked_actions = [
                        str(action)
                        for action in option_summary.get(
                            "safety_masked_actions",
                            [],
                        )
                    ]
                    external_override_count = int(
                        option_summary.get("external_override_count", 0)
                    )
                    if option_leaf_logits.size > 0:
                        direct_policy_trace_payload["option_leaf_logits"] = (
                            option_leaf_logits.round(6).tolist()
                        )
                    if option_owned_action is not None:
                        direct_policy_trace_payload["option_owned_action"] = (
                            option_owned_action
                        )
                    direct_policy_trace_payload["safety_mask_applied"] = bool(
                        safety_mask_applied
                    )
                    direct_policy_trace_payload["safety_masked_actions"] = list(
                        safety_masked_actions
                    )
                    direct_policy_trace_payload["external_override_count"] = int(
                        external_override_count
                    )
            affordance_summary = getattr(
                self.true_monolithic_policy,
                "last_affordance_summary",
                None,
            )
            if isinstance(affordance_summary, dict):
                affordance_blocked_logits = np.asarray(
                    affordance_summary.get("blocked_logits", []),
                    dtype=float,
                ).copy()
                affordance_role_logits = np.asarray(
                    affordance_summary.get("role_logits", []),
                    dtype=float,
                ).copy()
                if affordance_blocked_logits.size > 0:
                    direct_policy_trace_payload["affordance_blocked_logits"] = (
                        affordance_blocked_logits.round(6).tolist()
                    )
                if affordance_role_logits.size > 0:
                    direct_policy_trace_payload["affordance_role_logits"] = (
                        affordance_role_logits.round(6).tolist()
                    )
                geometry_logits = np.asarray(
                    affordance_summary.get("geometry_logits", []),
                    dtype=float,
                ).copy()
                if geometry_logits.size > 0:
                    direct_policy_trace_payload["geometry_logits"] = (
                        geometry_logits.round(6).tolist()
                    )
                shelter_column_logits = np.asarray(
                    affordance_summary.get("shelter_column_logits", []),
                    dtype=float,
                ).copy()
                if shelter_column_logits.size > 0:
                    direct_policy_trace_payload["shelter_column_logits"] = (
                        shelter_column_logits.round(6).tolist()
                    )
                shelter_position_logits = np.asarray(
                    affordance_summary.get("shelter_position_logits", []),
                    dtype=float,
                ).copy()
                if shelter_position_logits.size > 0:
                    direct_policy_trace_payload["shelter_position_logits"] = (
                        shelter_position_logits.round(6).tolist()
                    )
                transition_prediction_logits = np.asarray(
                    affordance_summary.get("transition_prediction_logits", []),
                    dtype=float,
                ).copy()
                if transition_prediction_logits.size > 0:
                    direct_policy_trace_payload["transition_prediction_logits"] = (
                        transition_prediction_logits.round(6).tolist()
                    )
                transition_rollout_prediction_logits = np.asarray(
                    affordance_summary.get(
                        "transition_rollout_prediction_logits",
                        [],
                    ),
                    dtype=float,
                ).copy()
                if transition_rollout_prediction_logits.size > 0:
                    direct_policy_trace_payload[
                        "transition_rollout_prediction_logits"
                    ] = transition_rollout_prediction_logits.round(6).tolist()
            direct_result = ModuleResult(
                interface=None,
                name=self.TRUE_MONOLITHIC_POLICY_NAME,
                observation_key=self.TRUE_MONOLITHIC_POLICY_NAME,
                observation=monolithic_observation.copy(),
                logits=policy_logits.copy(),
                probs=softmax(policy_logits),
                active=True,
                reflex=None,
                neural_logits=policy_logits.copy(),
                reflex_delta_logits=np.zeros_like(policy_logits),
                post_reflex_logits=policy_logits.copy(),
            )
            module_results = [direct_result]
            direct_result.valence_role = "integrated_policy"
            direct_result.gate_weight = 1.0
            direct_result.gated_logits = direct_result.logits.copy()
            direct_result.contribution_share = 1.0
            direct_result.intent_before_gating = ACTIONS[int(np.argmax(direct_result.logits))]
            direct_result.intent_after_gating = direct_result.intent_before_gating
            total_logits_without_reflex = direct_result.logits.copy()
            total_logits = direct_result.logits.copy()
            action_center_logits = total_logits.copy()
            action_center_policy = softmax(action_center_logits)
            proposal_sum = total_logits.copy()
            policy = softmax(total_logits)
            action_intent_idx = int(np.argmax(action_center_policy))
            action_intent_without_reflex_idx = action_intent_idx
            action_without_reflex_idx = action_intent_idx
            motor_action_idx = action_intent_idx
            arbitration = self._true_monolithic_arbitration_decision(
                module_name=self.TRUE_MONOLITHIC_POLICY_NAME,
                action_idx=action_intent_idx,
            )
            if (
                self.config.enable_food_direction_bias
                and not self.config.direct_policy_owned_option_controller
                and not training_mode
                and not sample
            ):
                bias_action = self._threat_escape_bias_action(observation)
                if bias_action is None:
                    bias_action = self._sleep_rest_bias_action(observation)
                if (
                    bias_action is None
                    and self._true_monolithic_allows_food_direction_bias()
                ):
                    bias_action = self._food_direction_bias_action(observation)
                if bias_action is not None:
                    total_logits = total_logits.copy()
                    threat_bias_action = self._threat_escape_bias_action(observation)
                    sleep_bias_action = self._sleep_rest_bias_action(observation)
                    if threat_bias_action is not None and bias_action == threat_bias_action:
                        bias_bonus = TRUE_MONOLITHIC_THREAT_ESCAPE_BIAS_LOGIT
                    elif sleep_bias_action is not None and bias_action == sleep_bias_action:
                        bias_bonus = TRUE_MONOLITHIC_SLEEP_REST_BIAS_LOGIT
                    else:
                        bias_bonus = TRUE_MONOLITHIC_DIRECTION_BIAS_LOGIT
                    total_logits[ACTION_TO_INDEX[bias_action]] += bias_bonus
                    policy = softmax(total_logits)
                    action_intent_idx = int(np.argmax(policy))
                    action_intent_without_reflex_idx = action_intent_idx
                    action_without_reflex_idx = action_intent_idx
                    motor_action_idx = action_intent_idx
                    action_center_logits = total_logits.copy()
                    action_center_policy = policy.copy()
                    proposal_sum = total_logits.copy()
                    arbitration = replace(
                        arbitration,
                        food_bias_applied=True,
                        food_bias_action=bias_action,
                        intent_before_gating_idx=int(np.argmax(total_logits_without_reflex)),
                        intent_after_gating_idx=action_intent_idx,
                    )
                    if hasattr(self.true_monolithic_policy, "record_external_override"):
                        self.true_monolithic_policy.record_external_override("final_bias")
                        external_override_count = int(
                            getattr(
                                self.true_monolithic_policy,
                                "external_override_count",
                                external_override_count,
                            )
                        )
                        direct_policy_trace_payload["external_override_count"] = int(
                            external_override_count
                        )
            if sample:
                action_idx = int(self.rng.choice(self.action_dim, p=policy))
            else:
                action_idx = motor_action_idx
            if hasattr(self.true_monolithic_policy, "record_executed_action"):
                self.true_monolithic_policy.record_executed_action(action_idx)
            motor_override = False
            final_reflex_override = False
        else:
            module_results = self._proposal_results(
                observation,
                store_cache=store_cache,
                training=training_mode,
            )
            arbitration_without_reflex = self._compute_arbitration(
                module_results,
                observation,
                training=False,
                store_cache=False,
            )
            gated_logits_without_reflex = [
                arbitration_gate_weight_for(arbitration_without_reflex, result.name) * result.logits
                for result in module_results
            ]
            proposal_sum_without_reflex = np.sum(
                np.stack(gated_logits_without_reflex, axis=0),
                axis=0,
            )
            if policy_mode == "reflex_only":
                action_center_logits_without_reflex = proposal_sum_without_reflex.copy()
                action_intent_without_reflex_idx = int(
                    np.argmax(action_center_logits_without_reflex)
                )
                total_logits_without_reflex = proposal_sum_without_reflex.copy()
            else:
                action_context_mapping = self._bound_action_context(observation)
                action_context = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_context_mapping)
                action_input_without_reflex = np.concatenate(
                    [np.concatenate(gated_logits_without_reflex, axis=0), action_context],
                    axis=0,
                )
                if self.action_center is None:
                    raise RuntimeError("Action center unavailable for the configured architecture.")
                action_center_correction_without_reflex, _ = self.action_center.forward(
                    action_input_without_reflex,
                    store_cache=False,
                )
                action_center_logits_without_reflex = (
                    proposal_sum_without_reflex + action_center_correction_without_reflex
                )
                action_intent_without_reflex_idx = int(
                    np.argmax(action_center_logits_without_reflex)
                )
                motor_input_without_reflex = self._build_motor_input(
                    one_hot(action_intent_without_reflex_idx, self.action_dim),
                    observation,
                )
                if self.motor_cortex is None:
                    raise RuntimeError("Motor cortex unavailable for the configured architecture.")
                motor_correction_without_reflex = self.motor_cortex.forward(
                    motor_input_without_reflex,
                    store_cache=False,
                )
                total_logits_without_reflex = (
                    action_center_logits_without_reflex + motor_correction_without_reflex
                )

            apply_reflex_path(
                module_results,
                ablation_config=self.config,
                operational_profile=self.operational_profile,
                interface_registry=self._interface_registry(),
                current_reflex_scale=self.current_reflex_scale,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )
            arbitration = self._compute_arbitration(
                module_results,
                observation,
                training=training_mode,
                store_cache=store_cache,
            )
            apply_priority_gating(
                module_results,
                arbitration,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )

        if bus is not None:
            for result in module_results:
                bus.publish(
                    sender=result.name,
                    topic="action.proposal",
                    payload={
                        "active": bool(result.active),
                        "action_logits": result.logits.round(6).tolist(),
                        "action_probs": result.probs.round(6).tolist(),
                        "neural_logits": result.neural_logits.round(6).tolist() if result.neural_logits is not None else None,
                        "reflex_delta_logits": result.reflex_delta_logits.round(6).tolist() if result.reflex_delta_logits is not None else None,
                        "post_reflex_logits": result.post_reflex_logits.round(6).tolist() if result.post_reflex_logits is not None else None,
                        "reflex_applied": bool(result.reflex_applied),
                        "effective_reflex_scale": round(float(result.effective_reflex_scale), 6),
                        "module_reflex_override": bool(result.module_reflex_override),
                        "module_reflex_dominance": round(float(result.module_reflex_dominance), 6),
                        "reflex": result.reflex.to_payload() if result.reflex is not None else None,
                        "valence_role": result.valence_role,
                        "gate_weight": round(float(result.gate_weight), 6),
                        "contribution_share": round(float(result.contribution_share), 6),
                        "gated_logits": result.gated_logits.round(6).tolist() if result.gated_logits is not None else None,
                        "intent_before_gating": result.intent_before_gating,
                        "intent_after_gating": result.intent_after_gating,
                        **(
                            direct_policy_trace_payload
                            if result.name
                            in {self.TRUE_MONOLITHIC_POLICY_NAME, B_SERIES_POLICY_NAME}
                            else {}
                        ),
                    },
                )

        if not (self.config.is_true_monolithic or self.config.is_b_series):
            proposal_sum = np.sum(
                np.stack(
                    [
                        result.gated_logits if result.gated_logits is not None else result.logits
                        for result in module_results
                    ],
                    axis=0,
                ),
                axis=0,
            )
            action_center_input = self._build_action_input(module_results, observation)
            if policy_mode == "reflex_only":
                action_center_logits = proposal_sum.copy()
                action_center_policy = softmax(action_center_logits)
                action_intent_idx = int(np.argmax(action_center_policy))
                motor_input = self._build_motor_input(
                    one_hot(action_intent_idx, self.action_dim),
                    observation,
                )
                total_logits = action_center_logits.copy()
                policy = softmax(total_logits)
                action_without_reflex_idx = int(np.argmax(total_logits_without_reflex))
                motor_action_idx = int(np.argmax(total_logits))
                if sample:
                    action_idx = int(self.rng.choice(self.action_dim, p=policy))
                else:
                    action_idx = motor_action_idx
                motor_override = False
                final_reflex_override = action_without_reflex_idx != motor_action_idx
            else:
                if self.action_center is None or self.motor_cortex is None:
                    raise RuntimeError(
                        "Action/motor pipeline unavailable for the configured architecture."
                    )
                action_center_correction_logits, value = self.action_center.forward(
                    action_center_input,
                    store_cache=store_cache,
                )
                action_center_logits = proposal_sum + action_center_correction_logits
                action_center_policy = softmax(action_center_logits)
                action_intent_idx = int(np.argmax(action_center_policy))
                motor_input = self._build_motor_input(
                    one_hot(action_intent_idx, self.action_dim),
                    observation,
                )
                motor_correction_logits = self.motor_cortex.forward(
                    motor_input,
                    store_cache=store_cache,
                )
                total_logits = action_center_logits + motor_correction_logits
                motor_action_idx_before_food_bias = int(np.argmax(total_logits))
                if (
                    self.config.enable_food_direction_bias
                    and not training_mode
                    and not sample
                    and arbitration is not None
                    and arbitration.winning_valence == "hunger"
                ):
                    food_bias_action = self._food_direction_bias_action(observation)
                    if food_bias_action is not None:
                        total_logits = total_logits.copy()
                        total_logits[ACTION_TO_INDEX[food_bias_action]] += (
                            MODULAR_DIRECTION_BIAS_LOGIT
                        )
                        arbitration = replace(
                            arbitration,
                            food_bias_applied=True,
                            food_bias_action=food_bias_action,
                        )
                policy = softmax(total_logits)
                action_without_reflex_idx = int(np.argmax(total_logits_without_reflex))
                motor_action_idx = int(np.argmax(total_logits))
                if sample:
                    action_idx = int(self.rng.choice(self.action_dim, p=policy))
                else:
                    action_idx = motor_action_idx
                motor_override = action_intent_idx != motor_action_idx
                final_reflex_override = (
                    action_intent_without_reflex_idx != action_intent_idx
                    or action_without_reflex_idx != motor_action_idx_before_food_bias
                )

        execution_diagnostics = self._motor_execution_diagnostics(
            observation,
            action_idx,
        )
        orientation_alignment = float(execution_diagnostics["orientation_alignment"])
        terrain_difficulty = float(execution_diagnostics["terrain_difficulty"])
        momentum = float(execution_diagnostics["momentum"])
        execution_difficulty = float(execution_diagnostics["execution_difficulty"])

        if bus is not None:
            if self.config.is_b_series:
                bus.publish(
                    sender=B_SERIES_POLICY_NAME,
                    topic="action.selection",
                    payload={
                        "policy_mode": policy_mode,
                        "direct_policy_logits": total_logits.round(6).tolist(),
                        "policy": policy.round(6).tolist(),
                        "selected_action": ACTIONS[motor_action_idx],
                        "executed_action": ACTIONS[action_idx],
                        "value_estimate": round(float(value), 6),
                        **direct_policy_trace_payload,
                    },
                )
                if bool(direct_policy_trace_payload.get("b6_emit_action_center_payload", False)):
                    b6_action_center_payload = direct_policy_trace_payload.get(
                        "b6_action_center_payload",
                        {},
                    )
                    b6_action_center_payload = (
                        b6_action_center_payload
                        if isinstance(b6_action_center_payload, dict)
                        else {}
                    )
                    bus.publish(
                        sender="action_center",
                        topic="action.selection",
                        payload={
                            "policy_mode": policy_mode,
                            "direct_policy_logits": total_logits.round(6).tolist(),
                            "policy": policy.round(6).tolist(),
                            "selected_intent": ACTIONS[motor_action_idx],
                            "selected_action": ACTIONS[motor_action_idx],
                            "executed_action": ACTIONS[action_idx],
                            "value_estimate": round(float(value), 6),
                            **b6_action_center_payload,
                            "b6_controller_family": direct_policy_trace_payload.get(
                                "b6_controller_family"
                            ),
                            "b6_controller_profile": direct_policy_trace_payload.get(
                                "b6_controller_profile"
                            ),
                            "b6_decision": direct_policy_trace_payload.get(
                                "b6_decision"
                            ),
                        },
                    )
            elif self.config.is_true_monolithic:
                bus.publish(
                    sender=self.TRUE_MONOLITHIC_POLICY_NAME,
                    topic="action.selection",
                    payload={
                        "policy_mode": policy_mode,
                        "direct_policy_logits": total_logits.round(6).tolist(),
                        "policy": policy.round(6).tolist(),
                        "selected_action": ACTIONS[motor_action_idx],
                        "executed_action": ACTIONS[action_idx],
                        "value_estimate": round(float(value), 6),
                        **direct_policy_trace_payload,
                    },
                )
                self._direct_policy_hidden_reset_pending = False
            else:
                arbitration_payload = arbitration.to_payload() if arbitration is not None else {}
                bus.publish(
                    sender="action_center",
                    topic="action.selection",
                    payload={
                        "policy_mode": policy_mode,
                        "proposal_sum_logits": proposal_sum.round(6).tolist(),
                        "action_center_correction_logits": action_center_correction_logits.round(6).tolist(),
                        "action_center_logits": action_center_logits.round(6).tolist(),
                        "action_center_policy": action_center_policy.round(6).tolist(),
                        "selected_intent": ACTIONS[action_intent_idx],
                        "selected_intent_without_reflex": ACTIONS[action_intent_without_reflex_idx],
                        "value_estimate": round(float(value), 6),
                        **arbitration_payload,
                    },
                )
                bus.publish(
                    sender="motor_cortex",
                    topic="action.execution",
                    payload={
                        "policy_mode": policy_mode,
                        "motor_correction_logits": motor_correction_logits.round(6).tolist(),
                        "total_logits_without_reflex": total_logits_without_reflex.round(6).tolist(),
                        "total_logits": total_logits.round(6).tolist(),
                        "policy": policy.round(6).tolist(),
                        "selected_intent": ACTIONS[action_intent_idx],
                        "selected_action": ACTIONS[motor_action_idx],
                        "executed_action": ACTIONS[action_idx],
                        "selected_action_without_reflex": ACTIONS[action_without_reflex_idx],
                        "motor_override": bool(motor_override),
                        "final_reflex_override": bool(final_reflex_override),
                        "orientation_alignment": round(float(orientation_alignment), 6),
                        "terrain_difficulty": round(float(terrain_difficulty), 6),
                        "momentum": round(float(momentum), 6),
                        "execution_difficulty": round(float(execution_difficulty), 6),
                        "execution_slip_occurred": False,
                        "slip_reason": "none",
                    },
                )

        step_observation: Dict[str, np.ndarray] = {}
        if policy_mode == "normal" and (
            self.config.is_true_monolithic or self.config.is_b_series
        ):
            brain_observation_keys = set(observation.keys())
            step_observation = {
                key: np.asarray(observation[key], dtype=float).copy()
                for key in brain_observation_keys
                if key in observation and key != "meta"
            }
        elif (
            policy_mode == "normal"
            and self.config.is_modular
            and self.config.uses_counterfactual_credit
        ):
            brain_observation_keys = {
                spec.observation_key for spec in MODULE_INTERFACES
            }
            brain_observation_keys.update(
                {
                    ACTION_CONTEXT_INTERFACE.observation_key,
                    MOTOR_CONTEXT_INTERFACE.observation_key,
                }
            )
            step_observation = {
                key: np.asarray(observation[key], dtype=float).copy()
                for key in brain_observation_keys
                if key in observation
            }

        return BrainStep(
            module_results=module_results,
            action_center_logits=action_center_logits,
            action_center_policy=action_center_policy,
            motor_correction_logits=motor_correction_logits,
            observation=step_observation,
            total_logits_without_reflex=total_logits_without_reflex,
            total_logits=total_logits,
            policy=policy,
            value=float(value),
            action_intent_idx=action_intent_idx,
            motor_action_idx=motor_action_idx,
            action_idx=action_idx,
            orientation_alignment=orientation_alignment,
            terrain_difficulty=terrain_difficulty,
            momentum=momentum,
            execution_difficulty=execution_difficulty,
            execution_slip_occurred=False,
            motor_slip_occurred=False,
            motor_noise_applied=False,
            slip_reason="none",
            motor_override=bool(motor_override),
            final_reflex_override=bool(final_reflex_override),
            action_center_input=action_center_input,
            motor_input=motor_input,
            policy_mode=policy_mode,
            arbitration_decision=arbitration,
            phase_logits=phase_logits,
            phase_prediction=phase_prediction,
            phase_prediction_confidence=float(phase_prediction_confidence),
            event_attention_top_type=event_attention_top_type,
            event_attention_top_age=int(event_attention_top_age),
            event_attention_entropy=float(event_attention_entropy),
            selected_option=selected_option,
            option_age=int(option_age),
            option_termination_reason=option_termination_reason,
            option_logits=np.asarray(option_logits, dtype=float).copy(),
            option_leaf_logits=np.asarray(option_leaf_logits, dtype=float).copy(),
            option_owned_action=option_owned_action,
            safety_mask_applied=bool(safety_mask_applied),
            safety_masked_actions=tuple(safety_masked_actions),
            external_override_count=int(external_override_count),
            affordance_blocked_logits=np.asarray(
                affordance_blocked_logits,
                dtype=float,
            ).copy(),
            affordance_role_logits=np.asarray(
                affordance_role_logits,
                dtype=float,
            ).copy(),
            geometry_logits=np.asarray(geometry_logits, dtype=float).copy(),
            shelter_column_logits=np.asarray(
                shelter_column_logits,
                dtype=float,
            ).copy(),
            shelter_position_logits=np.asarray(
                shelter_position_logits,
                dtype=float,
            ).copy(),
            transition_prediction_logits=np.asarray(
                transition_prediction_logits,
                dtype=float,
            ).copy(),
            transition_rollout_prediction_logits=np.asarray(
                transition_rollout_prediction_logits,
                dtype=float,
            ).copy(),
            b_level=int(self.config.b_level) if self.config.is_b_series else -1,
            b_effective_level=(
                str(direct_policy_trace_payload.get("b_effective_level"))
                if self.config.is_b_series
                and direct_policy_trace_payload.get("b_effective_level") is not None
                else None
            ),
            b_mode=str(self.config.b_mode) if self.config.is_b_series else None,
            b_parent_level=(
                direct_policy_trace_payload.get("b_parent_level")
                if self.config.is_b_series
                else None
            ),
            b_transfer_source_checkpoint=(
                direct_policy_trace_payload.get("b_transfer_source_checkpoint")
                if self.config.is_b_series
                else None
            ),
            b_transfer_coverage=(
                direct_policy_trace_payload.get("b_transfer_coverage")
                if self.config.is_b_series
                else None
            ),
            b_current_threat_pressure=(
                direct_policy_trace_payload.get("b_current_threat_pressure")
                if self.config.is_b_series
                else None
            ),
            b_temporal_threat_pressure=(
                direct_policy_trace_payload.get("b_temporal_threat_pressure")
                if self.config.is_b_series
                else None
            ),
            b_predator_memory_pressure=(
                direct_policy_trace_payload.get("b_predator_memory_pressure")
                if self.config.is_b_series
                else None
            ),
            b_predator_trace_pressure=(
                direct_policy_trace_payload.get("b_predator_trace_pressure")
                if self.config.is_b_series
                else None
            ),
            b3_contact_cooldown=(
                direct_policy_trace_payload.get("b3_contact_cooldown")
                if self.config.is_b_series
                else None
            ),
            b3_post_food_cooldown=(
                direct_policy_trace_payload.get("b3_post_food_cooldown")
                if self.config.is_b_series
                else None
            ),
            b3_hunger_drop=(
                direct_policy_trace_payload.get("b3_hunger_drop")
                if self.config.is_b_series
                else None
            ),
            b3_controller_profile=(
                direct_policy_trace_payload.get("b3_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b4_controller_profile=(
                direct_policy_trace_payload.get("b4_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b4_recovery_pressure=(
                direct_policy_trace_payload.get("b4_recovery_pressure")
                if self.config.is_b_series
                else None
            ),
            b4_sleep_hold=(
                direct_policy_trace_payload.get("b4_sleep_hold")
                if self.config.is_b_series
                else None
            ),
            b4_exit_blocked=(
                direct_policy_trace_payload.get("b4_exit_blocked")
                if self.config.is_b_series
                else None
            ),
            b4_hunger_release=(
                direct_policy_trace_payload.get("b4_hunger_release")
                if self.config.is_b_series
                else None
            ),
            b4_genetic_generation=(
                direct_policy_trace_payload.get("b4_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b4_genetic_candidate=(
                direct_policy_trace_payload.get("b4_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b5_controller_profile=(
                direct_policy_trace_payload.get("b5_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b5_hunger_urgency=(
                direct_policy_trace_payload.get("b5_hunger_urgency")
                if self.config.is_b_series
                else None
            ),
            b5_sleep_pressure=(
                direct_policy_trace_payload.get("b5_sleep_pressure")
                if self.config.is_b_series
                else None
            ),
            b5_recovery_debt=(
                direct_policy_trace_payload.get("b5_recovery_debt")
                if self.config.is_b_series
                else None
            ),
            b5_threat_gate=(
                direct_policy_trace_payload.get("b5_threat_gate")
                if self.config.is_b_series
                else None
            ),
            b5_sleep_bout_lock=(
                direct_policy_trace_payload.get("b5_sleep_bout_lock")
                if self.config.is_b_series
                else None
            ),
            b5_forage_commitment_lock=(
                direct_policy_trace_payload.get("b5_forage_commitment_lock")
                if self.config.is_b_series
                else None
            ),
            b5_homeostatic_decision=(
                direct_policy_trace_payload.get("b5_homeostatic_decision")
                if self.config.is_b_series
                else None
            ),
            b5_genetic_generation=(
                direct_policy_trace_payload.get("b5_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b5_genetic_candidate=(
                direct_policy_trace_payload.get("b5_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b6_controller_family=(
                direct_policy_trace_payload.get("b6_controller_family")
                if self.config.is_b_series
                else None
            ),
            b6_controller_profile=(
                direct_policy_trace_payload.get("b6_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b6_risk_pressure=(
                direct_policy_trace_payload.get("b6_risk_pressure")
                if self.config.is_b_series
                else None
            ),
            b6_threat_priority=(
                direct_policy_trace_payload.get("b6_threat_priority")
                if self.config.is_b_series
                else None
            ),
            b6_forage_suppressed=(
                direct_policy_trace_payload.get("b6_forage_suppressed")
                if self.config.is_b_series
                else None
            ),
            b6_corridor_commitment=(
                direct_policy_trace_payload.get("b6_corridor_commitment")
                if self.config.is_b_series
                else None
            ),
            b6_corridor_progress_memory=(
                direct_policy_trace_payload.get("b6_corridor_progress_memory")
                if self.config.is_b_series
                else None
            ),
            b6_recurrent_state=(
                direct_policy_trace_payload.get("b6_recurrent_state")
                if self.config.is_b_series
                else None
            ),
            b6_return_lock=(
                direct_policy_trace_payload.get("b6_return_lock")
                if self.config.is_b_series
                else None
            ),
            b6_decision=(
                direct_policy_trace_payload.get("b6_decision")
                if self.config.is_b_series
                else None
            ),
            b6_genetic_generation=(
                direct_policy_trace_payload.get("b6_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b6_genetic_candidate=(
                direct_policy_trace_payload.get("b6_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b7_controller_profile=(
                direct_policy_trace_payload.get("b7_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b7_affordance_state=(
                direct_policy_trace_payload.get("b7_affordance_state")
                if self.config.is_b_series
                else None
            ),
            b7_energy_budget=(
                direct_policy_trace_payload.get("b7_energy_budget")
                if self.config.is_b_series
                else None
            ),
            b7_budget_margin=(
                direct_policy_trace_payload.get("b7_budget_margin")
                if self.config.is_b_series
                else None
            ),
            b7_food_steps_estimate=(
                direct_policy_trace_payload.get("b7_food_steps_estimate")
                if self.config.is_b_series
                else None
            ),
            b7_return_steps_estimate=(
                direct_policy_trace_payload.get("b7_return_steps_estimate")
                if self.config.is_b_series
                else None
            ),
            b7_corridor_viability=(
                direct_policy_trace_payload.get("b7_corridor_viability")
                if self.config.is_b_series
                else None
            ),
            b7_abort_return=(
                direct_policy_trace_payload.get("b7_abort_return")
                if self.config.is_b_series
                else None
            ),
            b7_commitment_lock=(
                direct_policy_trace_payload.get("b7_commitment_lock")
                if self.config.is_b_series
                else None
            ),
            b7_decision=(
                direct_policy_trace_payload.get("b7_decision")
                if self.config.is_b_series
                else None
            ),
            b7_genetic_generation=(
                direct_policy_trace_payload.get("b7_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b7_genetic_candidate=(
                direct_policy_trace_payload.get("b7_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b8_controller_profile=(
                direct_policy_trace_payload.get("b8_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b8_spatial_map_state=(
                direct_policy_trace_payload.get("b8_spatial_map_state")
                if self.config.is_b_series
                else None
            ),
            b8_local_affordance_score=(
                direct_policy_trace_payload.get("b8_local_affordance_score")
                if self.config.is_b_series
                else None
            ),
            b8_return_vector_strength=(
                direct_policy_trace_payload.get("b8_return_vector_strength")
                if self.config.is_b_series
                else None
            ),
            b8_corridor_dead_end_risk=(
                direct_policy_trace_payload.get("b8_corridor_dead_end_risk")
                if self.config.is_b_series
                else None
            ),
            b8_abort_executed=(
                direct_policy_trace_payload.get("b8_abort_executed")
                if self.config.is_b_series
                else None
            ),
            b8_place_memory=(
                direct_policy_trace_payload.get("b8_place_memory")
                if self.config.is_b_series
                else None
            ),
            b8_decision=(
                direct_policy_trace_payload.get("b8_decision")
                if self.config.is_b_series
                else None
            ),
            b8_genetic_generation=(
                direct_policy_trace_payload.get("b8_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b8_genetic_candidate=(
                direct_policy_trace_payload.get("b8_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b9_controller_profile=(
                direct_policy_trace_payload.get("b9_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b9_route_state=(
                direct_policy_trace_payload.get("b9_route_state")
                if self.config.is_b_series
                else None
            ),
            b9_route_confidence=(
                direct_policy_trace_payload.get("b9_route_confidence")
                if self.config.is_b_series
                else None
            ),
            b9_waypoint_lock=(
                direct_policy_trace_payload.get("b9_waypoint_lock")
                if self.config.is_b_series
                else None
            ),
            b9_path_integrator=(
                direct_policy_trace_payload.get("b9_path_integrator")
                if self.config.is_b_series
                else None
            ),
            b9_replan_signal=(
                direct_policy_trace_payload.get("b9_replan_signal")
                if self.config.is_b_series
                else None
            ),
            b9_decision=(
                direct_policy_trace_payload.get("b9_decision")
                if self.config.is_b_series
                else None
            ),
            b9_genetic_generation=(
                direct_policy_trace_payload.get("b9_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b9_genetic_candidate=(
                direct_policy_trace_payload.get("b9_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b10_controller_profile=(
                direct_policy_trace_payload.get("b10_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b10_replay_state=(
                direct_policy_trace_payload.get("b10_replay_state")
                if self.config.is_b_series
                else None
            ),
            b10_prospective_value=(
                direct_policy_trace_payload.get("b10_prospective_value")
                if self.config.is_b_series
                else None
            ),
            b10_rollout_depth=(
                direct_policy_trace_payload.get("b10_rollout_depth")
                if self.config.is_b_series
                else None
            ),
            b10_replay_memory=(
                direct_policy_trace_payload.get("b10_replay_memory")
                if self.config.is_b_series
                else None
            ),
            b10_plan_commitment=(
                direct_policy_trace_payload.get("b10_plan_commitment")
                if self.config.is_b_series
                else None
            ),
            b10_abort_signal=(
                direct_policy_trace_payload.get("b10_abort_signal")
                if self.config.is_b_series
                else None
            ),
            b10_decision=(
                direct_policy_trace_payload.get("b10_decision")
                if self.config.is_b_series
                else None
            ),
            b10_genetic_generation=(
                direct_policy_trace_payload.get("b10_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b10_genetic_candidate=(
                direct_policy_trace_payload.get("b10_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b11_controller_profile=(
                direct_policy_trace_payload.get("b11_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b11_confidence_state=(
                direct_policy_trace_payload.get("b11_confidence_state")
                if self.config.is_b_series
                else None
            ),
            b11_plan_confidence=(
                direct_policy_trace_payload.get("b11_plan_confidence")
                if self.config.is_b_series
                else None
            ),
            b11_uncertainty=(
                direct_policy_trace_payload.get("b11_uncertainty")
                if self.config.is_b_series
                else None
            ),
            b11_neuromod_signal=(
                direct_policy_trace_payload.get("b11_neuromod_signal")
                if self.config.is_b_series
                else None
            ),
            b11_confidence_lock=(
                direct_policy_trace_payload.get("b11_confidence_lock")
                if self.config.is_b_series
                else None
            ),
            b11_decision=(
                direct_policy_trace_payload.get("b11_decision")
                if self.config.is_b_series
                else None
            ),
            b11_genetic_generation=(
                direct_policy_trace_payload.get("b11_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b11_genetic_candidate=(
                direct_policy_trace_payload.get("b11_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b12_controller_profile=(
                direct_policy_trace_payload.get("b12_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b12_attention_state=(
                direct_policy_trace_payload.get("b12_attention_state")
                if self.config.is_b_series
                else None
            ),
            b12_prediction_error=(
                direct_policy_trace_payload.get("b12_prediction_error")
                if self.config.is_b_series
                else None
            ),
            b12_attention_gain=(
                direct_policy_trace_payload.get("b12_attention_gain")
                if self.config.is_b_series
                else None
            ),
            b12_expected_progress=(
                direct_policy_trace_payload.get("b12_expected_progress")
                if self.config.is_b_series
                else None
            ),
            b12_search_lock=(
                direct_policy_trace_payload.get("b12_search_lock")
                if self.config.is_b_series
                else None
            ),
            b12_decision=(
                direct_policy_trace_payload.get("b12_decision")
                if self.config.is_b_series
                else None
            ),
            b12_genetic_generation=(
                direct_policy_trace_payload.get("b12_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b12_genetic_candidate=(
                direct_policy_trace_payload.get("b12_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b13_controller_profile=(
                direct_policy_trace_payload.get("b13_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b13_search_state=(
                direct_policy_trace_payload.get("b13_search_state")
                if self.config.is_b_series
                else None
            ),
            b13_local_route_score=(
                direct_policy_trace_payload.get("b13_local_route_score")
                if self.config.is_b_series
                else None
            ),
            b13_affordance_samples=(
                direct_policy_trace_payload.get("b13_affordance_samples")
                if self.config.is_b_series
                else None
            ),
            b13_search_memory=(
                direct_policy_trace_payload.get("b13_search_memory")
                if self.config.is_b_series
                else None
            ),
            b13_dead_end_score=(
                direct_policy_trace_payload.get("b13_dead_end_score")
                if self.config.is_b_series
                else None
            ),
            b13_search_lock=(
                direct_policy_trace_payload.get("b13_search_lock")
                if self.config.is_b_series
                else None
            ),
            b13_decision=(
                direct_policy_trace_payload.get("b13_decision")
                if self.config.is_b_series
                else None
            ),
            b13_genetic_generation=(
                direct_policy_trace_payload.get("b13_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b13_genetic_candidate=(
                direct_policy_trace_payload.get("b13_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b14_controller_profile=(
                direct_policy_trace_payload.get("b14_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b14_uncertainty_state=(
                direct_policy_trace_payload.get("b14_uncertainty_state")
                if self.config.is_b_series
                else None
            ),
            b14_affordance_confidence=(
                direct_policy_trace_payload.get("b14_affordance_confidence")
                if self.config.is_b_series
                else None
            ),
            b14_uncertainty=(
                direct_policy_trace_payload.get("b14_uncertainty")
                if self.config.is_b_series
                else None
            ),
            b14_risk_adjusted_score=(
                direct_policy_trace_payload.get("b14_risk_adjusted_score")
                if self.config.is_b_series
                else None
            ),
            b14_commitment_lock=(
                direct_policy_trace_payload.get("b14_commitment_lock")
                if self.config.is_b_series
                else None
            ),
            b14_decision=(
                direct_policy_trace_payload.get("b14_decision")
                if self.config.is_b_series
                else None
            ),
            b14_genetic_generation=(
                direct_policy_trace_payload.get("b14_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b14_genetic_candidate=(
                direct_policy_trace_payload.get("b14_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b15_controller_profile=(
                direct_policy_trace_payload.get("b15_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b15_option_state=(
                direct_policy_trace_payload.get("b15_option_state")
                if self.config.is_b_series
                else None
            ),
            b15_option_value=(
                direct_policy_trace_payload.get("b15_option_value")
                if self.config.is_b_series
                else None
            ),
            b15_termination_pressure=(
                direct_policy_trace_payload.get("b15_termination_pressure")
                if self.config.is_b_series
                else None
            ),
            b15_persistence_score=(
                direct_policy_trace_payload.get("b15_persistence_score")
                if self.config.is_b_series
                else None
            ),
            b15_option_lock=(
                direct_policy_trace_payload.get("b15_option_lock")
                if self.config.is_b_series
                else None
            ),
            b15_decision=(
                direct_policy_trace_payload.get("b15_decision")
                if self.config.is_b_series
                else None
            ),
            b15_genetic_generation=(
                direct_policy_trace_payload.get("b15_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b15_genetic_candidate=(
                direct_policy_trace_payload.get("b15_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b16_controller_profile=(
                direct_policy_trace_payload.get("b16_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b16_ensemble_state=(
                direct_policy_trace_payload.get("b16_ensemble_state")
                if self.config.is_b_series
                else None
            ),
            b16_continue_vote=(
                direct_policy_trace_payload.get("b16_continue_vote")
                if self.config.is_b_series
                else None
            ),
            b16_return_vote=(
                direct_policy_trace_payload.get("b16_return_vote")
                if self.config.is_b_series
                else None
            ),
            b16_option_votes=(
                direct_policy_trace_payload.get("b16_option_votes")
                if self.config.is_b_series
                else None
            ),
            b16_consensus_score=(
                direct_policy_trace_payload.get("b16_consensus_score")
                if self.config.is_b_series
                else None
            ),
            b16_conflict_score=(
                direct_policy_trace_payload.get("b16_conflict_score")
                if self.config.is_b_series
                else None
            ),
            b16_ensemble_lock=(
                direct_policy_trace_payload.get("b16_ensemble_lock")
                if self.config.is_b_series
                else None
            ),
            b16_decision=(
                direct_policy_trace_payload.get("b16_decision")
                if self.config.is_b_series
                else None
            ),
            b16_genetic_generation=(
                direct_policy_trace_payload.get("b16_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b16_genetic_candidate=(
                direct_policy_trace_payload.get("b16_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b17_controller_profile=(
                direct_policy_trace_payload.get("b17_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b17_modulator_state=(
                direct_policy_trace_payload.get("b17_modulator_state")
                if self.config.is_b_series
                else None
            ),
            b17_arousal_signal=(
                direct_policy_trace_payload.get("b17_arousal_signal")
                if self.config.is_b_series
                else None
            ),
            b17_homeostatic_gain=(
                direct_policy_trace_payload.get("b17_homeostatic_gain")
                if self.config.is_b_series
                else None
            ),
            b17_option_gain=(
                direct_policy_trace_payload.get("b17_option_gain")
                if self.config.is_b_series
                else None
            ),
            b17_conflict_release=(
                direct_policy_trace_payload.get("b17_conflict_release")
                if self.config.is_b_series
                else None
            ),
            b17_modulation_lock=(
                direct_policy_trace_payload.get("b17_modulation_lock")
                if self.config.is_b_series
                else None
            ),
            b17_decision=(
                direct_policy_trace_payload.get("b17_decision")
                if self.config.is_b_series
                else None
            ),
            b17_genetic_generation=(
                direct_policy_trace_payload.get("b17_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b17_genetic_candidate=(
                direct_policy_trace_payload.get("b17_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b18_controller_profile=(
                direct_policy_trace_payload.get("b18_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b18_trace_state=(
                direct_policy_trace_payload.get("b18_trace_state")
                if self.config.is_b_series
                else None
            ),
            b18_eligibility_trace=(
                direct_policy_trace_payload.get("b18_eligibility_trace")
                if self.config.is_b_series
                else None
            ),
            b18_reward_prediction_proxy=(
                direct_policy_trace_payload.get("b18_reward_prediction_proxy")
                if self.config.is_b_series
                else None
            ),
            b18_stability_bias=(
                direct_policy_trace_payload.get("b18_stability_bias")
                if self.config.is_b_series
                else None
            ),
            b18_switch_pressure=(
                direct_policy_trace_payload.get("b18_switch_pressure")
                if self.config.is_b_series
                else None
            ),
            b18_trace_lock=(
                direct_policy_trace_payload.get("b18_trace_lock")
                if self.config.is_b_series
                else None
            ),
            b18_decision=(
                direct_policy_trace_payload.get("b18_decision")
                if self.config.is_b_series
                else None
            ),
            b18_genetic_generation=(
                direct_policy_trace_payload.get("b18_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b18_genetic_candidate=(
                direct_policy_trace_payload.get("b18_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b19_controller_profile=(
                direct_policy_trace_payload.get("b19_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b19_memory_state=(
                direct_policy_trace_payload.get("b19_memory_state")
                if self.config.is_b_series
                else None
            ),
            b19_episode_memory=(
                direct_policy_trace_payload.get("b19_episode_memory")
                if self.config.is_b_series
                else None
            ),
            b19_consolidation_score=(
                direct_policy_trace_payload.get("b19_consolidation_score")
                if self.config.is_b_series
                else None
            ),
            b19_stability_vote=(
                direct_policy_trace_payload.get("b19_stability_vote")
                if self.config.is_b_series
                else None
            ),
            b19_switch_suppression=(
                direct_policy_trace_payload.get("b19_switch_suppression")
                if self.config.is_b_series
                else None
            ),
            b19_memory_lock=(
                direct_policy_trace_payload.get("b19_memory_lock")
                if self.config.is_b_series
                else None
            ),
            b19_decision=(
                direct_policy_trace_payload.get("b19_decision")
                if self.config.is_b_series
                else None
            ),
            b19_genetic_generation=(
                direct_policy_trace_payload.get("b19_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b19_genetic_candidate=(
                direct_policy_trace_payload.get("b19_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b20_controller_profile=(
                direct_policy_trace_payload.get("b20_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b20_buffer_state=(
                direct_policy_trace_payload.get("b20_buffer_state")
                if self.config.is_b_series
                else None
            ),
            b20_working_buffer=(
                direct_policy_trace_payload.get("b20_working_buffer")
                if self.config.is_b_series
                else None
            ),
            b20_context_binding=(
                direct_policy_trace_payload.get("b20_context_binding")
                if self.config.is_b_series
                else None
            ),
            b20_gate_vote=(
                direct_policy_trace_payload.get("b20_gate_vote")
                if self.config.is_b_series
                else None
            ),
            b20_release_vote=(
                direct_policy_trace_payload.get("b20_release_vote")
                if self.config.is_b_series
                else None
            ),
            b20_buffer_lock=(
                direct_policy_trace_payload.get("b20_buffer_lock")
                if self.config.is_b_series
                else None
            ),
            b20_decision=(
                direct_policy_trace_payload.get("b20_decision")
                if self.config.is_b_series
                else None
            ),
            b20_genetic_generation=(
                direct_policy_trace_payload.get("b20_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b20_genetic_candidate=(
                direct_policy_trace_payload.get("b20_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b21_controller_profile=(
                direct_policy_trace_payload.get("b21_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b21_replay_state=(
                direct_policy_trace_payload.get("b21_replay_state")
                if self.config.is_b_series
                else None
            ),
            b21_sequence_memory=(
                direct_policy_trace_payload.get("b21_sequence_memory")
                if self.config.is_b_series
                else None
            ),
            b21_replay_score=(
                direct_policy_trace_payload.get("b21_replay_score")
                if self.config.is_b_series
                else None
            ),
            b21_route_commitment=(
                direct_policy_trace_payload.get("b21_route_commitment")
                if self.config.is_b_series
                else None
            ),
            b21_abort_prediction=(
                direct_policy_trace_payload.get("b21_abort_prediction")
                if self.config.is_b_series
                else None
            ),
            b21_replay_lock=(
                direct_policy_trace_payload.get("b21_replay_lock")
                if self.config.is_b_series
                else None
            ),
            b21_decision=(
                direct_policy_trace_payload.get("b21_decision")
                if self.config.is_b_series
                else None
            ),
            b21_genetic_generation=(
                direct_policy_trace_payload.get("b21_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b21_genetic_candidate=(
                direct_policy_trace_payload.get("b21_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b22_controller_profile=(
                direct_policy_trace_payload.get("b22_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b22_sim_state=(
                direct_policy_trace_payload.get("b22_sim_state")
                if self.config.is_b_series
                else None
            ),
            b22_prospective_sim=(
                direct_policy_trace_payload.get("b22_prospective_sim")
                if self.config.is_b_series
                else None
            ),
            b22_forward_model_score=(
                direct_policy_trace_payload.get("b22_forward_model_score")
                if self.config.is_b_series
                else None
            ),
            b22_viability_projection=(
                direct_policy_trace_payload.get("b22_viability_projection")
                if self.config.is_b_series
                else None
            ),
            b22_abort_projection=(
                direct_policy_trace_payload.get("b22_abort_projection")
                if self.config.is_b_series
                else None
            ),
            b22_sim_lock=(
                direct_policy_trace_payload.get("b22_sim_lock")
                if self.config.is_b_series
                else None
            ),
            b22_decision=(
                direct_policy_trace_payload.get("b22_decision")
                if self.config.is_b_series
                else None
            ),
            b22_genetic_generation=(
                direct_policy_trace_payload.get("b22_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b22_genetic_candidate=(
                direct_policy_trace_payload.get("b22_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b23_controller_profile=(
                direct_policy_trace_payload.get("b23_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b23_conflict_state=(
                direct_policy_trace_payload.get("b23_conflict_state")
                if self.config.is_b_series
                else None
            ),
            b23_prediction_error=(
                direct_policy_trace_payload.get("b23_prediction_error")
                if self.config.is_b_series
                else None
            ),
            b23_conflict_memory=(
                direct_policy_trace_payload.get("b23_conflict_memory")
                if self.config.is_b_series
                else None
            ),
            b23_stability_vote=(
                direct_policy_trace_payload.get("b23_stability_vote")
                if self.config.is_b_series
                else None
            ),
            b23_abort_bias=(
                direct_policy_trace_payload.get("b23_abort_bias")
                if self.config.is_b_series
                else None
            ),
            b23_monitor_lock=(
                direct_policy_trace_payload.get("b23_monitor_lock")
                if self.config.is_b_series
                else None
            ),
            b23_decision=(
                direct_policy_trace_payload.get("b23_decision")
                if self.config.is_b_series
                else None
            ),
            b23_genetic_generation=(
                direct_policy_trace_payload.get("b23_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b23_genetic_candidate=(
                direct_policy_trace_payload.get("b23_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b24_controller_profile=(
                direct_policy_trace_payload.get("b24_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b24_precision_state=(
                direct_policy_trace_payload.get("b24_precision_state")
                if self.config.is_b_series
                else None
            ),
            b24_precision_memory=(
                direct_policy_trace_payload.get("b24_precision_memory")
                if self.config.is_b_series
                else None
            ),
            b24_precision_vote=(
                direct_policy_trace_payload.get("b24_precision_vote")
                if self.config.is_b_series
                else None
            ),
            b24_uncertainty_pressure=(
                direct_policy_trace_payload.get("b24_uncertainty_pressure")
                if self.config.is_b_series
                else None
            ),
            b24_abort_precision=(
                direct_policy_trace_payload.get("b24_abort_precision")
                if self.config.is_b_series
                else None
            ),
            b24_precision_lock=(
                direct_policy_trace_payload.get("b24_precision_lock")
                if self.config.is_b_series
                else None
            ),
            b24_decision=(
                direct_policy_trace_payload.get("b24_decision")
                if self.config.is_b_series
                else None
            ),
            b24_genetic_generation=(
                direct_policy_trace_payload.get("b24_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b24_genetic_candidate=(
                direct_policy_trace_payload.get("b24_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b25_controller_profile=(
                direct_policy_trace_payload.get("b25_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b25_metacognitive_state=(
                direct_policy_trace_payload.get("b25_metacognitive_state")
                if self.config.is_b_series
                else None
            ),
            b25_confidence_memory=(
                direct_policy_trace_payload.get("b25_confidence_memory")
                if self.config.is_b_series
                else None
            ),
            b25_confidence_vote=(
                direct_policy_trace_payload.get("b25_confidence_vote")
                if self.config.is_b_series
                else None
            ),
            b25_doubt_pressure=(
                direct_policy_trace_payload.get("b25_doubt_pressure")
                if self.config.is_b_series
                else None
            ),
            b25_control_gain=(
                direct_policy_trace_payload.get("b25_control_gain")
                if self.config.is_b_series
                else None
            ),
            b25_meta_lock=(
                direct_policy_trace_payload.get("b25_meta_lock")
                if self.config.is_b_series
                else None
            ),
            b25_decision=(
                direct_policy_trace_payload.get("b25_decision")
                if self.config.is_b_series
                else None
            ),
            b25_genetic_generation=(
                direct_policy_trace_payload.get("b25_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b25_genetic_candidate=(
                direct_policy_trace_payload.get("b25_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b26_controller_profile=(
                direct_policy_trace_payload.get("b26_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b26_allostatic_state=(
                direct_policy_trace_payload.get("b26_allostatic_state")
                if self.config.is_b_series
                else None
            ),
            b26_prediction_error=(
                direct_policy_trace_payload.get("b26_prediction_error")
                if self.config.is_b_series
                else None
            ),
            b26_setpoint_pressure=(
                direct_policy_trace_payload.get("b26_setpoint_pressure")
                if self.config.is_b_series
                else None
            ),
            b26_control_vote=(
                direct_policy_trace_payload.get("b26_control_vote")
                if self.config.is_b_series
                else None
            ),
            b26_stability_lock=(
                direct_policy_trace_payload.get("b26_stability_lock")
                if self.config.is_b_series
                else None
            ),
            b26_decision=(
                direct_policy_trace_payload.get("b26_decision")
                if self.config.is_b_series
                else None
            ),
            b26_genetic_generation=(
                direct_policy_trace_payload.get("b26_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b26_genetic_candidate=(
                direct_policy_trace_payload.get("b26_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b27_controller_profile=(
                direct_policy_trace_payload.get("b27_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b27_arousal_state=(
                direct_policy_trace_payload.get("b27_arousal_state")
                if self.config.is_b_series
                else None
            ),
            b27_arousal_level=(
                direct_policy_trace_payload.get("b27_arousal_level")
                if self.config.is_b_series
                else None
            ),
            b27_gain_modulation=(
                direct_policy_trace_payload.get("b27_gain_modulation")
                if self.config.is_b_series
                else None
            ),
            b27_stress_pressure=(
                direct_policy_trace_payload.get("b27_stress_pressure")
                if self.config.is_b_series
                else None
            ),
            b27_arousal_lock=(
                direct_policy_trace_payload.get("b27_arousal_lock")
                if self.config.is_b_series
                else None
            ),
            b27_decision=(
                direct_policy_trace_payload.get("b27_decision")
                if self.config.is_b_series
                else None
            ),
            b27_genetic_generation=(
                direct_policy_trace_payload.get("b27_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b27_genetic_candidate=(
                direct_policy_trace_payload.get("b27_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b28_controller_profile=(
                direct_policy_trace_payload.get("b28_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b28_attention_state=(
                direct_policy_trace_payload.get("b28_attention_state")
                if self.config.is_b_series
                else None
            ),
            b28_interoceptive_focus=(
                direct_policy_trace_payload.get("b28_interoceptive_focus")
                if self.config.is_b_series
                else None
            ),
            b28_attention_gain=(
                direct_policy_trace_payload.get("b28_attention_gain")
                if self.config.is_b_series
                else None
            ),
            b28_distractor_pressure=(
                direct_policy_trace_payload.get("b28_distractor_pressure")
                if self.config.is_b_series
                else None
            ),
            b28_attention_lock=(
                direct_policy_trace_payload.get("b28_attention_lock")
                if self.config.is_b_series
                else None
            ),
            b28_decision=(
                direct_policy_trace_payload.get("b28_decision")
                if self.config.is_b_series
                else None
            ),
            b28_genetic_generation=(
                direct_policy_trace_payload.get("b28_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b28_genetic_candidate=(
                direct_policy_trace_payload.get("b28_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b29_controller_profile=(
                direct_policy_trace_payload.get("b29_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b29_salience_state=(
                direct_policy_trace_payload.get("b29_salience_state")
                if self.config.is_b_series
                else None
            ),
            b29_threat_salience=(
                direct_policy_trace_payload.get("b29_threat_salience")
                if self.config.is_b_series
                else None
            ),
            b29_homeostatic_salience=(
                direct_policy_trace_payload.get("b29_homeostatic_salience")
                if self.config.is_b_series
                else None
            ),
            b29_corridor_salience=(
                direct_policy_trace_payload.get("b29_corridor_salience")
                if self.config.is_b_series
                else None
            ),
            b29_winner_channel=(
                direct_policy_trace_payload.get("b29_winner_channel")
                if self.config.is_b_series
                else None
            ),
            b29_salience_lock=(
                direct_policy_trace_payload.get("b29_salience_lock")
                if self.config.is_b_series
                else None
            ),
            b29_decision=(
                direct_policy_trace_payload.get("b29_decision")
                if self.config.is_b_series
                else None
            ),
            b29_genetic_generation=(
                direct_policy_trace_payload.get("b29_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b29_genetic_candidate=(
                direct_policy_trace_payload.get("b29_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b30_controller_profile=(
                direct_policy_trace_payload.get("b30_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b30_gate_state=(
                direct_policy_trace_payload.get("b30_gate_state")
                if self.config.is_b_series
                else None
            ),
            b30_go_signal=(
                direct_policy_trace_payload.get("b30_go_signal")
                if self.config.is_b_series
                else None
            ),
            b30_no_go_signal=(
                direct_policy_trace_payload.get("b30_no_go_signal")
                if self.config.is_b_series
                else None
            ),
            b30_action_gate=(
                direct_policy_trace_payload.get("b30_action_gate")
                if self.config.is_b_series
                else None
            ),
            b30_gate_lock=(
                direct_policy_trace_payload.get("b30_gate_lock")
                if self.config.is_b_series
                else None
            ),
            b30_decision=(
                direct_policy_trace_payload.get("b30_decision")
                if self.config.is_b_series
                else None
            ),
            b30_genetic_generation=(
                direct_policy_trace_payload.get("b30_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b30_genetic_candidate=(
                direct_policy_trace_payload.get("b30_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b31_controller_profile=(
                direct_policy_trace_payload.get("b31_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b31_dopamine_state=(
                direct_policy_trace_payload.get("b31_dopamine_state")
                if self.config.is_b_series
                else None
            ),
            b31_reward_prediction_error=(
                direct_policy_trace_payload.get("b31_reward_prediction_error")
                if self.config.is_b_series
                else None
            ),
            b31_tonic_dopamine=(
                direct_policy_trace_payload.get("b31_tonic_dopamine")
                if self.config.is_b_series
                else None
            ),
            b31_phasic_dopamine=(
                direct_policy_trace_payload.get("b31_phasic_dopamine")
                if self.config.is_b_series
                else None
            ),
            b31_gate_bias=(
                direct_policy_trace_payload.get("b31_gate_bias")
                if self.config.is_b_series
                else None
            ),
            b31_dopamine_lock=(
                direct_policy_trace_payload.get("b31_dopamine_lock")
                if self.config.is_b_series
                else None
            ),
            b31_decision=(
                direct_policy_trace_payload.get("b31_decision")
                if self.config.is_b_series
                else None
            ),
            b31_genetic_generation=(
                direct_policy_trace_payload.get("b31_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b31_genetic_candidate=(
                direct_policy_trace_payload.get("b31_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b32_controller_profile=(
                direct_policy_trace_payload.get("b32_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b32_critic_value=(
                direct_policy_trace_payload.get("b32_critic_value")
                if self.config.is_b_series
                else None
            ),
            b32_actor_advantage=(
                direct_policy_trace_payload.get("b32_actor_advantage")
                if self.config.is_b_series
                else None
            ),
            b32_value_error=(
                direct_policy_trace_payload.get("b32_value_error")
                if self.config.is_b_series
                else None
            ),
            b32_policy_bias=(
                direct_policy_trace_payload.get("b32_policy_bias")
                if self.config.is_b_series
                else None
            ),
            b32_value_lock=(
                direct_policy_trace_payload.get("b32_value_lock")
                if self.config.is_b_series
                else None
            ),
            b32_decision=(
                direct_policy_trace_payload.get("b32_decision")
                if self.config.is_b_series
                else None
            ),
            b32_genetic_generation=(
                direct_policy_trace_payload.get("b32_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b32_genetic_candidate=(
                direct_policy_trace_payload.get("b32_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b33_controller_profile=(
                direct_policy_trace_payload.get("b33_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b33_td_error=(
                direct_policy_trace_payload.get("b33_td_error")
                if self.config.is_b_series
                else None
            ),
            b33_bootstrap_value=(
                direct_policy_trace_payload.get("b33_bootstrap_value")
                if self.config.is_b_series
                else None
            ),
            b33_reward_trace=(
                direct_policy_trace_payload.get("b33_reward_trace")
                if self.config.is_b_series
                else None
            ),
            b33_actor_update=(
                direct_policy_trace_payload.get("b33_actor_update")
                if self.config.is_b_series
                else None
            ),
            b33_td_lock=(
                direct_policy_trace_payload.get("b33_td_lock")
                if self.config.is_b_series
                else None
            ),
            b33_decision=(
                direct_policy_trace_payload.get("b33_decision")
                if self.config.is_b_series
                else None
            ),
            b33_genetic_generation=(
                direct_policy_trace_payload.get("b33_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b33_genetic_candidate=(
                direct_policy_trace_payload.get("b33_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b34_controller_profile=(
                direct_policy_trace_payload.get("b34_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b34_eligibility_trace=(
                direct_policy_trace_payload.get("b34_eligibility_trace")
                if self.config.is_b_series
                else None
            ),
            b34_credit_assignment=(
                direct_policy_trace_payload.get("b34_credit_assignment")
                if self.config.is_b_series
                else None
            ),
            b34_synaptic_tag=(
                direct_policy_trace_payload.get("b34_synaptic_tag")
                if self.config.is_b_series
                else None
            ),
            b34_decay_memory=(
                direct_policy_trace_payload.get("b34_decay_memory")
                if self.config.is_b_series
                else None
            ),
            b34_credit_lock=(
                direct_policy_trace_payload.get("b34_credit_lock")
                if self.config.is_b_series
                else None
            ),
            b34_decision=(
                direct_policy_trace_payload.get("b34_decision")
                if self.config.is_b_series
                else None
            ),
            b34_genetic_generation=(
                direct_policy_trace_payload.get("b34_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b34_genetic_candidate=(
                direct_policy_trace_payload.get("b34_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b35_controller_profile=(
                direct_policy_trace_payload.get("b35_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b35_forward_value=(
                direct_policy_trace_payload.get("b35_forward_value")
                if self.config.is_b_series
                else None
            ),
            b35_transition_error=(
                direct_policy_trace_payload.get("b35_transition_error")
                if self.config.is_b_series
                else None
            ),
            b35_model_confidence=(
                direct_policy_trace_payload.get("b35_model_confidence")
                if self.config.is_b_series
                else None
            ),
            b35_prediction_memory=(
                direct_policy_trace_payload.get("b35_prediction_memory")
                if self.config.is_b_series
                else None
            ),
            b35_model_lock=(
                direct_policy_trace_payload.get("b35_model_lock")
                if self.config.is_b_series
                else None
            ),
            b35_decision=(
                direct_policy_trace_payload.get("b35_decision")
                if self.config.is_b_series
                else None
            ),
            b35_genetic_generation=(
                direct_policy_trace_payload.get("b35_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b35_genetic_candidate=(
                direct_policy_trace_payload.get("b35_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b36_controller_profile=(
                direct_policy_trace_payload.get("b36_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b36_latent_state=(
                direct_policy_trace_payload.get("b36_latent_state")
                if self.config.is_b_series
                else None
            ),
            b36_belief_error=(
                direct_policy_trace_payload.get("b36_belief_error")
                if self.config.is_b_series
                else None
            ),
            b36_state_confidence=(
                direct_policy_trace_payload.get("b36_state_confidence")
                if self.config.is_b_series
                else None
            ),
            b36_context_memory=(
                direct_policy_trace_payload.get("b36_context_memory")
                if self.config.is_b_series
                else None
            ),
            b36_belief_lock=(
                direct_policy_trace_payload.get("b36_belief_lock")
                if self.config.is_b_series
                else None
            ),
            b36_decision=(
                direct_policy_trace_payload.get("b36_decision")
                if self.config.is_b_series
                else None
            ),
            b36_genetic_generation=(
                direct_policy_trace_payload.get("b36_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b36_genetic_candidate=(
                direct_policy_trace_payload.get("b36_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b37_controller_profile=(
                direct_policy_trace_payload.get("b37_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b37_external_state_factor=(
                direct_policy_trace_payload.get("b37_external_state_factor")
                if self.config.is_b_series
                else None
            ),
            b37_internal_state_factor=(
                direct_policy_trace_payload.get("b37_internal_state_factor")
                if self.config.is_b_series
                else None
            ),
            b37_factor_alignment=(
                direct_policy_trace_payload.get("b37_factor_alignment")
                if self.config.is_b_series
                else None
            ),
            b37_factor_confidence=(
                direct_policy_trace_payload.get("b37_factor_confidence")
                if self.config.is_b_series
                else None
            ),
            b37_factor_lock=(
                direct_policy_trace_payload.get("b37_factor_lock")
                if self.config.is_b_series
                else None
            ),
            b37_decision=(
                direct_policy_trace_payload.get("b37_decision")
                if self.config.is_b_series
                else None
            ),
            b37_genetic_generation=(
                direct_policy_trace_payload.get("b37_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b37_genetic_candidate=(
                direct_policy_trace_payload.get("b37_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b38_controller_profile=(
                direct_policy_trace_payload.get("b38_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b38_external_attention=(
                direct_policy_trace_payload.get("b38_external_attention")
                if self.config.is_b_series
                else None
            ),
            b38_internal_attention=(
                direct_policy_trace_payload.get("b38_internal_attention")
                if self.config.is_b_series
                else None
            ),
            b38_attention_balance=(
                direct_policy_trace_payload.get("b38_attention_balance")
                if self.config.is_b_series
                else None
            ),
            b38_attention_gain=(
                direct_policy_trace_payload.get("b38_attention_gain")
                if self.config.is_b_series
                else None
            ),
            b38_attention_lock=(
                direct_policy_trace_payload.get("b38_attention_lock")
                if self.config.is_b_series
                else None
            ),
            b38_decision=(
                direct_policy_trace_payload.get("b38_decision")
                if self.config.is_b_series
                else None
            ),
            b38_genetic_generation=(
                direct_policy_trace_payload.get("b38_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b38_genetic_candidate=(
                direct_policy_trace_payload.get("b38_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b39_controller_profile=(
                direct_policy_trace_payload.get("b39_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b39_binding_strength=(
                direct_policy_trace_payload.get("b39_binding_strength")
                if self.config.is_b_series
                else None
            ),
            b39_cross_factor_coherence=(
                direct_policy_trace_payload.get("b39_cross_factor_coherence")
                if self.config.is_b_series
                else None
            ),
            b39_bound_context=(
                direct_policy_trace_payload.get("b39_bound_context")
                if self.config.is_b_series
                else None
            ),
            b39_binding_gain=(
                direct_policy_trace_payload.get("b39_binding_gain")
                if self.config.is_b_series
                else None
            ),
            b39_binding_lock=(
                direct_policy_trace_payload.get("b39_binding_lock")
                if self.config.is_b_series
                else None
            ),
            b39_decision=(
                direct_policy_trace_payload.get("b39_decision")
                if self.config.is_b_series
                else None
            ),
            b39_genetic_generation=(
                direct_policy_trace_payload.get("b39_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b39_genetic_candidate=(
                direct_policy_trace_payload.get("b39_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b40_controller_profile=(
                direct_policy_trace_payload.get("b40_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b40_workspace_activation=(
                direct_policy_trace_payload.get("b40_workspace_activation")
                if self.config.is_b_series
                else None
            ),
            b40_broadcast_gain=(
                direct_policy_trace_payload.get("b40_broadcast_gain")
                if self.config.is_b_series
                else None
            ),
            b40_context_availability=(
                direct_policy_trace_payload.get("b40_context_availability")
                if self.config.is_b_series
                else None
            ),
            b40_workspace_stability=(
                direct_policy_trace_payload.get("b40_workspace_stability")
                if self.config.is_b_series
                else None
            ),
            b40_workspace_lock=(
                direct_policy_trace_payload.get("b40_workspace_lock")
                if self.config.is_b_series
                else None
            ),
            b40_decision=(
                direct_policy_trace_payload.get("b40_decision")
                if self.config.is_b_series
                else None
            ),
            b40_genetic_generation=(
                direct_policy_trace_payload.get("b40_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b40_genetic_candidate=(
                direct_policy_trace_payload.get("b40_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b41_controller_profile=(
                direct_policy_trace_payload.get("b41_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b41_executive_selection=(
                direct_policy_trace_payload.get("b41_executive_selection")
                if self.config.is_b_series
                else None
            ),
            b41_inhibitory_pressure=(
                direct_policy_trace_payload.get("b41_inhibitory_pressure")
                if self.config.is_b_series
                else None
            ),
            b41_goal_context=(
                direct_policy_trace_payload.get("b41_goal_context")
                if self.config.is_b_series
                else None
            ),
            b41_executive_stability=(
                direct_policy_trace_payload.get("b41_executive_stability")
                if self.config.is_b_series
                else None
            ),
            b41_executive_lock=(
                direct_policy_trace_payload.get("b41_executive_lock")
                if self.config.is_b_series
                else None
            ),
            b41_decision=(
                direct_policy_trace_payload.get("b41_decision")
                if self.config.is_b_series
                else None
            ),
            b41_genetic_generation=(
                direct_policy_trace_payload.get("b41_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b41_genetic_candidate=(
                direct_policy_trace_payload.get("b41_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b42_controller_profile=(
                direct_policy_trace_payload.get("b42_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b42_error_signal=(
                direct_policy_trace_payload.get("b42_error_signal")
                if self.config.is_b_series
                else None
            ),
            b42_conflict_signal=(
                direct_policy_trace_payload.get("b42_conflict_signal")
                if self.config.is_b_series
                else None
            ),
            b42_performance_context=(
                direct_policy_trace_payload.get("b42_performance_context")
                if self.config.is_b_series
                else None
            ),
            b42_monitor_stability=(
                direct_policy_trace_payload.get("b42_monitor_stability")
                if self.config.is_b_series
                else None
            ),
            b42_monitor_lock=(
                direct_policy_trace_payload.get("b42_monitor_lock")
                if self.config.is_b_series
                else None
            ),
            b42_decision=(
                direct_policy_trace_payload.get("b42_decision")
                if self.config.is_b_series
                else None
            ),
            b42_genetic_generation=(
                direct_policy_trace_payload.get("b42_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b42_genetic_candidate=(
                direct_policy_trace_payload.get("b42_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b43_controller_profile=(
                direct_policy_trace_payload.get("b43_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b43_precision_signal=(
                direct_policy_trace_payload.get("b43_precision_signal")
                if self.config.is_b_series
                else None
            ),
            b43_adaptive_threshold=(
                direct_policy_trace_payload.get("b43_adaptive_threshold")
                if self.config.is_b_series
                else None
            ),
            b43_arousal_context=(
                direct_policy_trace_payload.get("b43_arousal_context")
                if self.config.is_b_series
                else None
            ),
            b43_control_stability=(
                direct_policy_trace_payload.get("b43_control_stability")
                if self.config.is_b_series
                else None
            ),
            b43_precision_lock=(
                direct_policy_trace_payload.get("b43_precision_lock")
                if self.config.is_b_series
                else None
            ),
            b43_decision=(
                direct_policy_trace_payload.get("b43_decision")
                if self.config.is_b_series
                else None
            ),
            b43_genetic_generation=(
                direct_policy_trace_payload.get("b43_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b43_genetic_candidate=(
                direct_policy_trace_payload.get("b43_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b44_controller_profile=(
                direct_policy_trace_payload.get("b44_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b44_relay_gate=(
                direct_policy_trace_payload.get("b44_relay_gate")
                if self.config.is_b_series
                else None
            ),
            b44_sensory_precision=(
                direct_policy_trace_payload.get("b44_sensory_precision")
                if self.config.is_b_series
                else None
            ),
            b44_context_relay=(
                direct_policy_trace_payload.get("b44_context_relay")
                if self.config.is_b_series
                else None
            ),
            b44_gate_stability=(
                direct_policy_trace_payload.get("b44_gate_stability")
                if self.config.is_b_series
                else None
            ),
            b44_relay_lock=(
                direct_policy_trace_payload.get("b44_relay_lock")
                if self.config.is_b_series
                else None
            ),
            b44_decision=(
                direct_policy_trace_payload.get("b44_decision")
                if self.config.is_b_series
                else None
            ),
            b44_genetic_generation=(
                direct_policy_trace_payload.get("b44_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b44_genetic_candidate=(
                direct_policy_trace_payload.get("b44_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b45_controller_profile=(
                direct_policy_trace_payload.get("b45_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b45_inhibitory_gate=(
                direct_policy_trace_payload.get("b45_inhibitory_gate")
                if self.config.is_b_series
                else None
            ),
            b45_sensory_filter=(
                direct_policy_trace_payload.get("b45_sensory_filter")
                if self.config.is_b_series
                else None
            ),
            b45_context_suppression=(
                direct_policy_trace_payload.get("b45_context_suppression")
                if self.config.is_b_series
                else None
            ),
            b45_loop_stability=(
                direct_policy_trace_payload.get("b45_loop_stability")
                if self.config.is_b_series
                else None
            ),
            b45_inhibition_lock=(
                direct_policy_trace_payload.get("b45_inhibition_lock")
                if self.config.is_b_series
                else None
            ),
            b45_decision=(
                direct_policy_trace_payload.get("b45_decision")
                if self.config.is_b_series
                else None
            ),
            b45_genetic_generation=(
                direct_policy_trace_payload.get("b45_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b45_genetic_candidate=(
                direct_policy_trace_payload.get("b45_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b46_controller_profile=(
                direct_policy_trace_payload.get("b46_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b46_feedback_gain=(
                direct_policy_trace_payload.get("b46_feedback_gain")
                if self.config.is_b_series
                else None
            ),
            b46_topdown_context=(
                direct_policy_trace_payload.get("b46_topdown_context")
                if self.config.is_b_series
                else None
            ),
            b46_prediction_match=(
                direct_policy_trace_payload.get("b46_prediction_match")
                if self.config.is_b_series
                else None
            ),
            b46_feedback_stability=(
                direct_policy_trace_payload.get("b46_feedback_stability")
                if self.config.is_b_series
                else None
            ),
            b46_feedback_lock=(
                direct_policy_trace_payload.get("b46_feedback_lock")
                if self.config.is_b_series
                else None
            ),
            b46_decision=(
                direct_policy_trace_payload.get("b46_decision")
                if self.config.is_b_series
                else None
            ),
            b46_genetic_generation=(
                direct_policy_trace_payload.get("b46_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b46_genetic_candidate=(
                direct_policy_trace_payload.get("b46_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b47_controller_profile=(
                direct_policy_trace_payload.get("b47_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b47_phase_alignment=(
                direct_policy_trace_payload.get("b47_phase_alignment")
                if self.config.is_b_series
                else None
            ),
            b47_synchrony_gain=(
                direct_policy_trace_payload.get("b47_synchrony_gain")
                if self.config.is_b_series
                else None
            ),
            b47_cross_loop_coherence=(
                direct_policy_trace_payload.get("b47_cross_loop_coherence")
                if self.config.is_b_series
                else None
            ),
            b47_phase_lock=(
                direct_policy_trace_payload.get("b47_phase_lock")
                if self.config.is_b_series
                else None
            ),
            b47_decision=(
                direct_policy_trace_payload.get("b47_decision")
                if self.config.is_b_series
                else None
            ),
            b47_genetic_generation=(
                direct_policy_trace_payload.get("b47_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b47_genetic_candidate=(
                direct_policy_trace_payload.get("b47_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b48_controller_profile=(
                direct_policy_trace_payload.get("b48_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b48_timing_error=(
                direct_policy_trace_payload.get("b48_timing_error")
                if self.config.is_b_series
                else None
            ),
            b48_predictive_timing=(
                direct_policy_trace_payload.get("b48_predictive_timing")
                if self.config.is_b_series
                else None
            ),
            b48_corrective_gain=(
                direct_policy_trace_payload.get("b48_corrective_gain")
                if self.config.is_b_series
                else None
            ),
            b48_calibration_lock=(
                direct_policy_trace_payload.get("b48_calibration_lock")
                if self.config.is_b_series
                else None
            ),
            b48_decision=(
                direct_policy_trace_payload.get("b48_decision")
                if self.config.is_b_series
                else None
            ),
            b48_genetic_generation=(
                direct_policy_trace_payload.get("b48_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b48_genetic_candidate=(
                direct_policy_trace_payload.get("b48_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b49_controller_profile=(
                direct_policy_trace_payload.get("b49_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b49_go_signal=(
                direct_policy_trace_payload.get("b49_go_signal")
                if self.config.is_b_series
                else None
            ),
            b49_no_go_signal=(
                direct_policy_trace_payload.get("b49_no_go_signal")
                if self.config.is_b_series
                else None
            ),
            b49_action_gate_balance=(
                direct_policy_trace_payload.get("b49_action_gate_balance")
                if self.config.is_b_series
                else None
            ),
            b49_selection_lock=(
                direct_policy_trace_payload.get("b49_selection_lock")
                if self.config.is_b_series
                else None
            ),
            b49_decision=(
                direct_policy_trace_payload.get("b49_decision")
                if self.config.is_b_series
                else None
            ),
            b49_genetic_generation=(
                direct_policy_trace_payload.get("b49_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b49_genetic_candidate=(
                direct_policy_trace_payload.get("b49_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b50_controller_profile=(
                direct_policy_trace_payload.get("b50_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b50_habit_strength=(
                direct_policy_trace_payload.get("b50_habit_strength")
                if self.config.is_b_series
                else None
            ),
            b50_chunk_value=(
                direct_policy_trace_payload.get("b50_chunk_value")
                if self.config.is_b_series
                else None
            ),
            b50_habit_stability=(
                direct_policy_trace_payload.get("b50_habit_stability")
                if self.config.is_b_series
                else None
            ),
            b50_chunk_lock=(
                direct_policy_trace_payload.get("b50_chunk_lock")
                if self.config.is_b_series
                else None
            ),
            b50_decision=(
                direct_policy_trace_payload.get("b50_decision")
                if self.config.is_b_series
                else None
            ),
            b50_genetic_generation=(
                direct_policy_trace_payload.get("b50_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b50_genetic_candidate=(
                direct_policy_trace_payload.get("b50_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b51_controller_profile=(
                direct_policy_trace_payload.get("b51_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b51_prediction_error=(
                direct_policy_trace_payload.get("b51_prediction_error")
                if self.config.is_b_series
                else None
            ),
            b51_dopamine_gain=(
                direct_policy_trace_payload.get("b51_dopamine_gain")
                if self.config.is_b_series
                else None
            ),
            b51_habit_modulation=(
                direct_policy_trace_payload.get("b51_habit_modulation")
                if self.config.is_b_series
                else None
            ),
            b51_modulation_lock=(
                direct_policy_trace_payload.get("b51_modulation_lock")
                if self.config.is_b_series
                else None
            ),
            b51_decision=(
                direct_policy_trace_payload.get("b51_decision")
                if self.config.is_b_series
                else None
            ),
            b51_genetic_generation=(
                direct_policy_trace_payload.get("b51_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b51_genetic_candidate=(
                direct_policy_trace_payload.get("b51_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b52_controller_profile=(
                direct_policy_trace_payload.get("b52_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b52_acetylcholine_level=(
                direct_policy_trace_payload.get("b52_acetylcholine_level")
                if self.config.is_b_series
                else None
            ),
            b52_precision_gain=(
                direct_policy_trace_payload.get("b52_precision_gain")
                if self.config.is_b_series
                else None
            ),
            b52_uncertainty_signal=(
                direct_policy_trace_payload.get("b52_uncertainty_signal")
                if self.config.is_b_series
                else None
            ),
            b52_attention_lock=(
                direct_policy_trace_payload.get("b52_attention_lock")
                if self.config.is_b_series
                else None
            ),
            b52_decision=(
                direct_policy_trace_payload.get("b52_decision")
                if self.config.is_b_series
                else None
            ),
            b52_genetic_generation=(
                direct_policy_trace_payload.get("b52_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b52_genetic_candidate=(
                direct_policy_trace_payload.get("b52_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b53_controller_profile=(
                direct_policy_trace_payload.get("b53_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b53_norepinephrine_level=(
                direct_policy_trace_payload.get("b53_norepinephrine_level")
                if self.config.is_b_series
                else None
            ),
            b53_arousal_gain=(
                direct_policy_trace_payload.get("b53_arousal_gain")
                if self.config.is_b_series
                else None
            ),
            b53_surprise_signal=(
                direct_policy_trace_payload.get("b53_surprise_signal")
                if self.config.is_b_series
                else None
            ),
            b53_gain_lock=(
                direct_policy_trace_payload.get("b53_gain_lock")
                if self.config.is_b_series
                else None
            ),
            b53_decision=(
                direct_policy_trace_payload.get("b53_decision")
                if self.config.is_b_series
                else None
            ),
            b53_genetic_generation=(
                direct_policy_trace_payload.get("b53_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b53_genetic_candidate=(
                direct_policy_trace_payload.get("b53_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b54_controller_profile=(
                direct_policy_trace_payload.get("b54_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b54_serotonin_level=(
                direct_policy_trace_payload.get("b54_serotonin_level")
                if self.config.is_b_series
                else None
            ),
            b54_patience_signal=(
                direct_policy_trace_payload.get("b54_patience_signal")
                if self.config.is_b_series
                else None
            ),
            b54_impulse_suppression=(
                direct_policy_trace_payload.get("b54_impulse_suppression")
                if self.config.is_b_series
                else None
            ),
            b54_patience_lock=(
                direct_policy_trace_payload.get("b54_patience_lock")
                if self.config.is_b_series
                else None
            ),
            b54_decision=(
                direct_policy_trace_payload.get("b54_decision")
                if self.config.is_b_series
                else None
            ),
            b54_genetic_generation=(
                direct_policy_trace_payload.get("b54_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b54_genetic_candidate=(
                direct_policy_trace_payload.get("b54_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b55_controller_profile=(
                direct_policy_trace_payload.get("b55_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b55_hypothalamic_drive=(
                direct_policy_trace_payload.get("b55_hypothalamic_drive")
                if self.config.is_b_series
                else None
            ),
            b55_satiety_signal=(
                direct_policy_trace_payload.get("b55_satiety_signal")
                if self.config.is_b_series
                else None
            ),
            b55_recovery_bias=(
                direct_policy_trace_payload.get("b55_recovery_bias")
                if self.config.is_b_series
                else None
            ),
            b55_drive_balance=(
                direct_policy_trace_payload.get("b55_drive_balance")
                if self.config.is_b_series
                else None
            ),
            b55_drive_lock=(
                direct_policy_trace_payload.get("b55_drive_lock")
                if self.config.is_b_series
                else None
            ),
            b55_decision=(
                direct_policy_trace_payload.get("b55_decision")
                if self.config.is_b_series
                else None
            ),
            b55_genetic_generation=(
                direct_policy_trace_payload.get("b55_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b55_genetic_candidate=(
                direct_policy_trace_payload.get("b55_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b56_controller_profile=(
                direct_policy_trace_payload.get("b56_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b56_cortisol_level=(
                direct_policy_trace_payload.get("b56_cortisol_level")
                if self.config.is_b_series
                else None
            ),
            b56_stress_load=(
                direct_policy_trace_payload.get("b56_stress_load")
                if self.config.is_b_series
                else None
            ),
            b56_recovery_signal=(
                direct_policy_trace_payload.get("b56_recovery_signal")
                if self.config.is_b_series
                else None
            ),
            b56_endocrine_balance=(
                direct_policy_trace_payload.get("b56_endocrine_balance")
                if self.config.is_b_series
                else None
            ),
            b56_stress_lock=(
                direct_policy_trace_payload.get("b56_stress_lock")
                if self.config.is_b_series
                else None
            ),
            b56_decision=(
                direct_policy_trace_payload.get("b56_decision")
                if self.config.is_b_series
                else None
            ),
            b56_genetic_generation=(
                direct_policy_trace_payload.get("b56_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b56_genetic_candidate=(
                direct_policy_trace_payload.get("b56_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b57_controller_profile=(
                direct_policy_trace_payload.get("b57_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b57_interoceptive_awareness=(
                direct_policy_trace_payload.get("b57_interoceptive_awareness")
                if self.config.is_b_series
                else None
            ),
            b57_visceral_salience=(
                direct_policy_trace_payload.get("b57_visceral_salience")
                if self.config.is_b_series
                else None
            ),
            b57_body_state_confidence=(
                direct_policy_trace_payload.get("b57_body_state_confidence")
                if self.config.is_b_series
                else None
            ),
            b57_awareness_balance=(
                direct_policy_trace_payload.get("b57_awareness_balance")
                if self.config.is_b_series
                else None
            ),
            b57_awareness_lock=(
                direct_policy_trace_payload.get("b57_awareness_lock")
                if self.config.is_b_series
                else None
            ),
            b57_decision=(
                direct_policy_trace_payload.get("b57_decision")
                if self.config.is_b_series
                else None
            ),
            b57_genetic_generation=(
                direct_policy_trace_payload.get("b57_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b57_genetic_candidate=(
                direct_policy_trace_payload.get("b57_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b58_controller_profile=(
                direct_policy_trace_payload.get("b58_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b58_conflict_signal=(
                direct_policy_trace_payload.get("b58_conflict_signal")
                if self.config.is_b_series
                else None
            ),
            b58_error_likelihood=(
                direct_policy_trace_payload.get("b58_error_likelihood")
                if self.config.is_b_series
                else None
            ),
            b58_control_allocation=(
                direct_policy_trace_payload.get("b58_control_allocation")
                if self.config.is_b_series
                else None
            ),
            b58_resolution_balance=(
                direct_policy_trace_payload.get("b58_resolution_balance")
                if self.config.is_b_series
                else None
            ),
            b58_conflict_lock=(
                direct_policy_trace_payload.get("b58_conflict_lock")
                if self.config.is_b_series
                else None
            ),
            b58_decision=(
                direct_policy_trace_payload.get("b58_decision")
                if self.config.is_b_series
                else None
            ),
            b58_genetic_generation=(
                direct_policy_trace_payload.get("b58_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b58_genetic_candidate=(
                direct_policy_trace_payload.get("b58_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b59_controller_profile=(
                direct_policy_trace_payload.get("b59_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b59_goal_context=(
                direct_policy_trace_payload.get("b59_goal_context")
                if self.config.is_b_series
                else None
            ),
            b59_working_set_stability=(
                direct_policy_trace_payload.get("b59_working_set_stability")
                if self.config.is_b_series
                else None
            ),
            b59_task_set_confidence=(
                direct_policy_trace_payload.get("b59_task_set_confidence")
                if self.config.is_b_series
                else None
            ),
            b59_executive_balance=(
                direct_policy_trace_payload.get("b59_executive_balance")
                if self.config.is_b_series
                else None
            ),
            b59_executive_lock=(
                direct_policy_trace_payload.get("b59_executive_lock")
                if self.config.is_b_series
                else None
            ),
            b59_decision=(
                direct_policy_trace_payload.get("b59_decision")
                if self.config.is_b_series
                else None
            ),
            b59_genetic_generation=(
                direct_policy_trace_payload.get("b59_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b59_genetic_candidate=(
                direct_policy_trace_payload.get("b59_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b60_controller_profile=(
                direct_policy_trace_payload.get("b60_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b60_outcome_value=(
                direct_policy_trace_payload.get("b60_outcome_value")
                if self.config.is_b_series
                else None
            ),
            b60_reversal_signal=(
                direct_policy_trace_payload.get("b60_reversal_signal")
                if self.config.is_b_series
                else None
            ),
            b60_goal_value_confidence=(
                direct_policy_trace_payload.get("b60_goal_value_confidence")
                if self.config.is_b_series
                else None
            ),
            b60_value_balance=(
                direct_policy_trace_payload.get("b60_value_balance")
                if self.config.is_b_series
                else None
            ),
            b60_value_lock=(
                direct_policy_trace_payload.get("b60_value_lock")
                if self.config.is_b_series
                else None
            ),
            b60_decision=(
                direct_policy_trace_payload.get("b60_decision")
                if self.config.is_b_series
                else None
            ),
            b60_genetic_generation=(
                direct_policy_trace_payload.get("b60_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b60_genetic_candidate=(
                direct_policy_trace_payload.get("b60_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b61_controller_profile=(
                direct_policy_trace_payload.get("b61_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b61_safety_value=(
                direct_policy_trace_payload.get("b61_safety_value")
                if self.config.is_b_series
                else None
            ),
            b61_threat_value=(
                direct_policy_trace_payload.get("b61_threat_value")
                if self.config.is_b_series
                else None
            ),
            b61_safety_confidence=(
                direct_policy_trace_payload.get("b61_safety_confidence")
                if self.config.is_b_series
                else None
            ),
            b61_affective_balance=(
                direct_policy_trace_payload.get("b61_affective_balance")
                if self.config.is_b_series
                else None
            ),
            b61_safety_lock=(
                direct_policy_trace_payload.get("b61_safety_lock")
                if self.config.is_b_series
                else None
            ),
            b61_decision=(
                direct_policy_trace_payload.get("b61_decision")
                if self.config.is_b_series
                else None
            ),
            b61_genetic_generation=(
                direct_policy_trace_payload.get("b61_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b61_genetic_candidate=(
                direct_policy_trace_payload.get("b61_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            b62_controller_profile=(
                direct_policy_trace_payload.get("b62_controller_profile")
                if self.config.is_b_series
                else None
            ),
            b62_defensive_mode=(
                direct_policy_trace_payload.get("b62_defensive_mode")
                if self.config.is_b_series
                else None
            ),
            b62_freeze_pressure=(
                direct_policy_trace_payload.get("b62_freeze_pressure")
                if self.config.is_b_series
                else None
            ),
            b62_flee_pressure=(
                direct_policy_trace_payload.get("b62_flee_pressure")
                if self.config.is_b_series
                else None
            ),
            b62_shelter_bias=(
                direct_policy_trace_payload.get("b62_shelter_bias")
                if self.config.is_b_series
                else None
            ),
            b62_defense_balance=(
                direct_policy_trace_payload.get("b62_defense_balance")
                if self.config.is_b_series
                else None
            ),
            b62_defense_lock=(
                direct_policy_trace_payload.get("b62_defense_lock")
                if self.config.is_b_series
                else None
            ),
            b62_decision=(
                direct_policy_trace_payload.get("b62_decision")
                if self.config.is_b_series
                else None
            ),
            b62_genetic_generation=(
                direct_policy_trace_payload.get("b62_genetic_generation")
                if self.config.is_b_series
                else None
            ),
            b62_genetic_candidate=(
                direct_policy_trace_payload.get("b62_genetic_candidate")
                if self.config.is_b_series
                else None
            ),
            semantic_action=(
                direct_policy_trace_payload.get("semantic_action")
                if self.config.is_b_series
                else None
            ),
            semantic_action_idx=(
                int(direct_policy_trace_payload.get("semantic_action_idx", -1))
                if self.config.is_b_series
                else -1
            ),
            learned_semantic_action=(
                direct_policy_trace_payload.get("learned_semantic_action")
                if self.config.is_b_series
                else None
            ),
            learned_semantic_action_idx=(
                int(direct_policy_trace_payload.get("learned_semantic_action_idx", -1))
                if self.config.is_b_series
                else -1
            ),
            semantic_action_source=(
                direct_policy_trace_payload.get("semantic_action_source")
                if self.config.is_b_series
                else None
            ),
            semantic_action_reason=(
                direct_policy_trace_payload.get("semantic_action_reason")
                if self.config.is_b_series
                else None
            ),
            semantic_override_count=(
                int(direct_policy_trace_payload.get("semantic_override_count", 0))
                if self.config.is_b_series
                else 0
            ),
            semantic_logits=(
                np.asarray(semantic_logits, dtype=float).copy()
                if self.config.is_b_series
                else np.zeros(0, dtype=float)
            ),
            semantic_policy=(
                np.asarray(semantic_policy, dtype=float).copy()
                if self.config.is_b_series
                else np.zeros(0, dtype=float)
            ),
            bridge_primitive_action=(
                direct_policy_trace_payload.get("bridge_primitive_action")
                if self.config.is_b_series
                else None
            ),
            bridge_reason=(
                direct_policy_trace_payload.get("bridge_reason")
                if self.config.is_b_series
                else None
            ),
            blocked_mask=(
                dict(direct_policy_trace_payload.get("blocked_mask", {}))
                if self.config.is_b_series
                else {}
            ),
            food_delta_used=(
                float(direct_policy_trace_payload.get("food_delta_used", 0.0))
                if self.config.is_b_series
                else 0.0
            ),
            shelter_delta_used=(
                float(direct_policy_trace_payload.get("shelter_delta_used", 0.0))
                if self.config.is_b_series
                else 0.0
            ),
        )

    def estimate_value(self, observation: Dict[str, np.ndarray]) -> float:
        """
        Estimate the action-center state value for a single observation.
        
        Builds proposals, applies reflex and arbitration gating as used at inference time, and returns the scalar value produced by the action-center. If the brain contains recurrent modules, their hidden states are snapshot and restored so this call does not mutate recurrent state.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Observation arrays keyed by interface names used to produce proposals and construct the action-center input.
        
        Returns:
            float: Scalar value estimate for the provided observation.
        """
        if self.config.is_b_series:
            if self.b_series_policy is None:
                raise RuntimeError(
                    "B-series policy unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            return float(
                self.b_series_policy.value_only(monolithic_observation)
            )
        if self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            if hasattr(self.true_monolithic_policy, "set_runtime_observation_meta"):
                self.true_monolithic_policy.set_runtime_observation_meta(
                    observation.get("meta", {})
                )
            runtime_state_snapshot = self.snapshot_direct_policy_runtime_state()
            try:
                direct_forward = self.true_monolithic_policy.forward(
                    monolithic_observation,
                    store_cache=False,
                )
                if len(direct_forward) >= 3:
                    _, value, *_ = direct_forward
                else:
                    _, value = direct_forward
                return float(value)
            finally:
                self.restore_direct_policy_runtime_state(runtime_state_snapshot)
        hidden_state_snapshot: Dict[str, np.ndarray] | None = None
        if self.module_bank is not None and self.module_bank.has_recurrent_modules:
            hidden_state_snapshot = self.module_bank.snapshot_hidden_states()
        try:
            module_results = self._proposal_results(
                observation,
                store_cache=False,
                training=False,
            )
            apply_reflex_path(
                module_results,
                ablation_config=self.config,
                operational_profile=self.operational_profile,
                interface_registry=self._interface_registry(),
                current_reflex_scale=self.current_reflex_scale,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )
            arbitration = self._compute_arbitration(
                module_results,
                observation,
                training=False,
                store_cache=False,
            )
            apply_priority_gating(
                module_results,
                arbitration,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )
            action_input = self._build_action_input(module_results, observation)
            if self.action_center is None:
                raise RuntimeError("Action center unavailable for the configured architecture.")
            _, value = self.action_center.forward(action_input, store_cache=False)
            return float(value)
        finally:
            if hidden_state_snapshot is not None and self.module_bank is not None:
                self.module_bank.restore_hidden_states(hidden_state_snapshot)

    def _proposal_stage_names(self) -> List[str]:
        """
        Return the ordered proposal sources that feed the action-center input.
        """
        if self.module_bank is not None:
            return [spec.name for spec in self.module_bank.enabled_specs]
        if self.true_monolithic_policy is not None:
            return [self.TRUE_MONOLITHIC_POLICY_NAME]
        if getattr(self, "b_series_policy", None) is not None:
            return [B_SERIES_POLICY_NAME]
        return [self.MONOLITHIC_POLICY_NAME]

    def _architecture_signature(self) -> dict[str, object]:
        """
        Compute the runtime architecture signature for the active proposal backend.
        
        Returns:
            signature (dict): A mapping describing architecture identifiers and configuration used for compatibility/fingerprinting (includes proposal backend name and order, whether learned arbitration is enabled, and arbitration network input/hidden dims and regularization weight).
        """
        arbitration_input_dim = (
            self.arbitration_network.input_dim
            if self.arbitration_network is not None
            else 0
        )
        arbitration_hidden_dim = (
            self.arbitration_network.hidden_dim
            if self.arbitration_network is not None
            else 0
        )
        return architecture_signature(
            proposal_backend=self.config.architecture,
            proposal_order=self._proposal_stage_names(),
            module_variants=getattr(self.config, "module_variants", None),
            learned_arbitration=(
                self.config.use_learned_arbitration and self.arbitration_network is not None
            ),
            arbitration_input_dim=arbitration_input_dim,
            arbitration_hidden_dim=arbitration_hidden_dim,
            arbitration_regularization_weight=self.arbitration_regularization_weight,
            capacity_profile_name=self.config.capacity_profile_name,
            module_hidden_dims=self.config.module_hidden_dims,
            action_center_hidden_dim=self.config.action_center_hidden_dim,
            motor_hidden_dim=self.config.motor_hidden_dim,
            integration_hidden_dim=self.config.integration_hidden_dim,
            monolithic_hidden_dim=self.config.monolithic_hidden_dim,
            b_level=getattr(self.config, "b_level", 0),
            b_mode=getattr(self.config, "b_mode", "current_bridge"),
            direct_policy_hidden_dims=self.config.direct_policy_hidden_dims or None,
            direct_policy_recurrent=bool(self.config.direct_policy_recurrent),
            direct_policy_phase_head=bool(self.config.direct_policy_phase_head),
            direct_policy_event_attention=bool(
                self.config.direct_policy_event_attention
            ),
            direct_policy_event_buffer_size=int(
                self.config.direct_policy_event_buffer_size
            ),
            direct_policy_option_head=bool(self.config.direct_policy_option_head),
            direct_policy_owned_option_controller=bool(
                self.config.direct_policy_owned_option_controller
            ),
            direct_policy_option_ttl=int(self.config.direct_policy_option_ttl),
            direct_policy_affordance_head=bool(
                self.config.direct_policy_affordance_head
            ),
            direct_policy_affordance_feedback=bool(
                self.config.direct_policy_affordance_feedback
            ),
            direct_policy_geometry_head=bool(
                self.config.direct_policy_geometry_head
            ),
            direct_policy_shelter_column_head=bool(
                self.config.direct_policy_shelter_column_head
            ),
            direct_policy_shelter_position_head=bool(
                self.config.direct_policy_shelter_position_head
            ),
            direct_policy_local_affordance_inputs=bool(
                getattr(self.config, "direct_policy_local_affordance_inputs", False)
            ),
            direct_policy_local_spatial_inputs=bool(
                getattr(self.config, "direct_policy_local_spatial_inputs", False)
            ),
            direct_policy_local_transition_inputs=bool(
                getattr(self.config, "direct_policy_local_transition_inputs", False)
            ),
            direct_policy_local_transition_rollout_inputs=bool(
                getattr(
                    self.config,
                    "direct_policy_local_transition_rollout_inputs",
                    False,
                )
            ),
            direct_policy_transition_prediction_head=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_prediction_head",
                    False,
                )
            ),
            direct_policy_transition_prediction_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_prediction_feedback",
                    False,
                )
            ),
            direct_policy_transition_rollout_prediction_head=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_rollout_prediction_head",
                    False,
                )
            ),
            direct_policy_transition_rollout_prediction_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_rollout_prediction_feedback",
                    False,
                )
            ),
            direct_policy_handoff_teacher=bool(
                self.config.direct_policy_handoff_teacher
            ),
            direct_policy_handoff_option_teacher=bool(
                getattr(self.config, "direct_policy_handoff_option_teacher", False)
            ),
            direct_policy_post_rest_action_teacher=bool(
                getattr(self.config, "direct_policy_post_rest_action_teacher", False)
            ),
            direct_policy_post_rest_release_sequence_teacher=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_teacher",
                    False,
                )
            ),
            direct_policy_post_rest_release_sequence_replay_boost=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_replay_boost",
                    False,
                )
            ),
            direct_policy_post_rest_release_sequence_distill=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_distill",
                    False,
                )
            ),
            direct_policy_post_rest_probe_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_sequence_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_sequence_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_family_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_family_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_handoff_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_handoff_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_trajectory_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_trajectory_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_cycle_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_cycle_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_trace_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_trace_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_rollout_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_rollout_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_frontier_teacher_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_frontier_teacher_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_replayable_teacher_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_replayable_teacher_distillation",
                    False,
                )
            ),
            direct_policy_continuation_replay_passes=int(
                getattr(
                    self.config,
                    "direct_policy_continuation_replay_passes",
                    0,
                )
            ),
            direct_policy_continuation_replay_lr_scale=float(
                getattr(
                    self.config,
                    "direct_policy_continuation_replay_lr_scale",
                    0.0,
                )
            ),
            direct_policy_continuation_margin_weight=float(
                getattr(
                    self.config,
                    "direct_policy_continuation_margin_weight",
                    0.0,
                )
            ),
            direct_policy_phase_option_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_phase_option_feedback",
                    False,
                )
            ),
            direct_policy_option_transition_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_option_transition_feedback",
                    False,
                )
            ),
            direct_policy_option_termination_cooldown=bool(
                getattr(
                    self.config,
                    "direct_policy_option_termination_cooldown",
                    False,
                )
            ),
            direct_policy_option_action_head=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_head",
                    False,
                )
            ),
            direct_policy_option_decoder_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_decoder_state",
                    False,
                )
            ),
            direct_policy_option_recurrent_dynamics=bool(
                getattr(
                    self.config,
                    "direct_policy_option_recurrent_dynamics",
                    False,
                )
            ),
            direct_policy_option_sequence_head=bool(
                getattr(
                    self.config,
                    "direct_policy_option_sequence_head",
                    False,
                )
            ),
            direct_policy_option_decoder_recurrent_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_decoder_recurrent_state",
                    False,
                )
            ),
            direct_policy_option_action_transition_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_transition_state",
                    False,
                )
            ),
            direct_policy_option_action_controller_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_controller_state",
                    False,
                )
            ),
            direct_policy_option_action_token_decoder=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_token_decoder",
                    False,
                )
            ),
            direct_policy_option_action_recurrent_core=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_recurrent_core",
                    False,
                )
            ),
            direct_policy_option_action_separate_recurrent_head=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_separate_recurrent_head",
                    False,
                )
            ),
            direct_policy_option_action_separate_policy_path=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_separate_policy_path",
                    False,
                )
            ),
            direct_policy_option_action_separate_backbone=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_separate_backbone",
                    False,
                )
            ),
            direct_policy_executive_physiology_option_gating=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_physiology_option_gating",
                    False,
                )
            ),
            direct_policy_executive_affordance_action_gating=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_affordance_action_gating",
                    False,
                )
            ),
            direct_policy_executive_option_action_masking=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_option_action_masking",
                    False,
                )
            ),
            direct_policy_executive_event_release_latching=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_event_release_latching",
                    False,
                )
            ),
            direct_policy_executive_event_release_action_commitment=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_event_release_action_commitment",
                    False,
                )
            ),
            direct_policy_executive_release_phase_state=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_phase_state",
                    False,
                )
            ),
            direct_policy_executive_release_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_progression",
                    False,
                )
            ),
            direct_policy_executive_release_exit_contract=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_exit_contract",
                    False,
                )
            ),
            direct_policy_executive_release_substate_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_substate_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_continuation=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_continuation",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_guidance=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_guidance",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_commitment=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_commitment",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_heading_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_heading_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_smell_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_smell_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_corridor_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_corridor_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_corridor_affordance_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_corridor_affordance_progression",
                    False,
                )
            ),
            direct_policy_executive_post_food_return=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_food_return",
                    False,
                )
            ),
            direct_policy_executive_post_food_vector_return=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_food_vector_return",
                    False,
                )
            ),
            direct_policy_executive_post_food_path_return=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_food_path_return",
                    False,
                )
            ),
            capacity_profile=self.config.capacity_profile.to_summary(),
        )

    def _interface_registry(self) -> dict[str, object]:
        """
        Retrieve the runtime-governed interface registry used by this brain.
        
        Returns:
            dict[str, object]: Mapping of interface names to their interface objects as provided by the active runtime registry.
        """
        return interface_registry()

    def _architecture_fingerprint(self) -> str:
        """
        Get the stable fingerprint of the brain's runtime architecture signature.
        
        Returns:
            fingerprint (str): String representation of the architecture signature's `fingerprint` field.
        """
        return str(self._architecture_signature()["fingerprint"])

    def parameter_norms(self) -> Dict[str, float]:
        """
        Compute the L2 norm of parameters for each trainable network component.
        
        Returns:
            Mapping from component name to its L2 parameter norm for each active
            trainable network in the configured topology.
        """
        norms: Dict[str, float] = {}
        if self.module_bank is not None:
            norms.update(self.module_bank.parameter_norms())
        if self.monolithic_policy is not None:
            norms[self.MONOLITHIC_POLICY_NAME] = self.monolithic_policy.parameter_norm()
        if self.true_monolithic_policy is not None:
            norms[self.TRUE_MONOLITHIC_POLICY_NAME] = (
                self.true_monolithic_policy.parameter_norm()
            )
        if getattr(self, "b_series_policy", None) is not None:
            norms[B_SERIES_POLICY_NAME] = self.b_series_policy.parameter_norm()
        if self.arbitration_network is not None:
            norms[self.ARBITRATION_NETWORK_NAME] = self.arbitration_network.parameter_norm()
        if self.action_center is not None:
            norms["action_center"] = self.action_center.parameter_norm()
        if self.motor_cortex is not None:
            norms["motor_cortex"] = self.motor_cortex.parameter_norm()
        return norms

    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable parameters for each trainable network component.

        Returns:
            Mapping from component name to trainable parameter count for each
            active trainable network in the configured topology.
        """
        counts: Dict[str, int] = {}
        if self.module_bank is not None:
            counts.update(self.module_bank.parameter_counts())
        if self.monolithic_policy is not None:
            counts[self.MONOLITHIC_POLICY_NAME] = self.monolithic_policy.count_parameters()
        if self.true_monolithic_policy is not None:
            counts[self.TRUE_MONOLITHIC_POLICY_NAME] = (
                self.true_monolithic_policy.count_parameters()
            )
        if getattr(self, "b_series_policy", None) is not None:
            counts[B_SERIES_POLICY_NAME] = self.b_series_policy.count_parameters()
        if self.arbitration_network is not None:
            counts[self.ARBITRATION_NETWORK_NAME] = self.arbitration_network.count_parameters()
        if self.action_center is not None:
            counts["action_center"] = self.action_center.count_parameters()
        if self.motor_cortex is not None:
            counts["motor_cortex"] = self.motor_cortex.count_parameters()
        return counts

    def _module_names(self) -> List[str]:
        """
        List module names present in the brain for inspection.
        
        When the brain is modular, returns the module spec names in their
        configured order followed by the active downstream controllers.
        Monolithic variants return only the networks present in that topology.
        
        Returns:
            names (List[str]): Ordered list of module names and the two controller component names.
        """
        if self.module_bank is not None:
            return [
                spec.name for spec in self.module_bank.enabled_specs
            ] + [self.ARBITRATION_NETWORK_NAME, "action_center", "motor_cortex"]
        if self.true_monolithic_policy is not None:
            return [self.TRUE_MONOLITHIC_POLICY_NAME]
        if getattr(self, "b_series_policy", None) is not None:
            return [B_SERIES_POLICY_NAME]
        return [self.MONOLITHIC_POLICY_NAME, self.ARBITRATION_NETWORK_NAME, "action_center", "motor_cortex"]
