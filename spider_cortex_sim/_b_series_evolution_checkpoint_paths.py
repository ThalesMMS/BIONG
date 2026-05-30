from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

def b0_current_bridge_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B0_CURRENT_BRIDGE_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b1_threat_guard_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B1_THREAT_GUARD_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b2_temporal_threat_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B2_TEMPORAL_THREAT_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b3_recurrent_guard_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B3_RECURRENT_GUARD_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b4_genetic_recovery_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B4_GENETIC_RECOVERY_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b5_genetic_homeostasis_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b6_fused_risk_recurrent_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b7_affordance_budget_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B7_AFFORDANCE_BUDGET_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b8_spatial_affordance_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b9_waypoint_planner_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B9_WAYPOINT_PLANNER_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b10_prospective_replay_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b11_confidence_arbiter_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B11_CONFIDENCE_ARBITER_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b12_predictive_attention_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b13_local_affordance_search_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b14_affordance_uncertainty_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b15_option_critic_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B15_OPTION_CRITIC_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b16_option_ensemble_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B16_OPTION_ENSEMBLE_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b17_neuromodulated_ensemble_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b18_eligibility_trace_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B18_ELIGIBILITY_TRACE_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b19_episodic_meta_memory_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B19_EPISODIC_META_MEMORY_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b20_working_memory_gate_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B20_WORKING_MEMORY_GATE_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b21_hippocampal_replay_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b22_prospective_replay_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b23_conflict_monitor_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B23_CONFLICT_MONITOR_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b24_precision_conflict_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B24_PRECISION_CONFLICT_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b25_metacognitive_confidence_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b26_allostatic_prediction_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b27_arousal_gain_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return root_path / B27_AROUSAL_GAIN_H48_POLICY_NAME / f"seed_{int(seed)}" / "best"


def b28_interoceptive_attention_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b29_salience_competition_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B29_SALIENCE_COMPETITION_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b30_basal_ganglia_gate_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b31_dopamine_prediction_error_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b32_actor_critic_value_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b33_td_error_decomposition_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b34_eligibility_credit_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b35_forward_model_value_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b36_latent_belief_state_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B36_LATENT_BELIEF_STATE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b37_state_factor_gate_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B37_STATE_FACTOR_GATE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b38_factor_attention_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B38_FACTOR_ATTENTION_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b39_attention_binding_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B39_ATTENTION_BINDING_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b40_global_workspace_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B40_GLOBAL_WORKSPACE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b41_executive_workspace_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b42_error_monitor_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B42_ERROR_MONITOR_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b43_adaptive_precision_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B43_ADAPTIVE_PRECISION_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b44_thalamic_relay_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B44_THALAMIC_RELAY_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b45_reticular_inhibition_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B45_RETICULAR_INHIBITION_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b46_corticothalamic_feedback_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b47_oscillatory_synchrony_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b48_cerebellar_timing_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B48_CEREBELLAR_TIMING_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b49_striatal_action_gate_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b50_habit_chunking_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B50_HABIT_CHUNKING_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b51_dopaminergic_habit_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b52_cholinergic_precision_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b53_noradrenergic_arousal_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b54_serotonergic_patience_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b55_hypothalamic_drive_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b56_hpa_stress_axis_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B56_HPA_STRESS_AXIS_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b57_insular_interoceptive_awareness_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b58_acc_conflict_monitor_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b59_prefrontal_goal_context_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b60_orbitofrontal_outcome_value_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )


def b61_amygdala_safety_value_checkpoint_path(
    *,
    root: str | Path = B_SERIES_EVOLUTION_ROOT,
    seed: int = 7,
) -> Path:
    root_path = Path(root)
    return (
        root_path
        / B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME
        / f"seed_{int(seed)}"
        / "best"
    )
