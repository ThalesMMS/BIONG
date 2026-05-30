from __future__ import annotations

from ._b_series_evolution_shared import *
from ._b_series_evolution_constants import *

def build_b_evolution_config(
    variant_name: str,
    *,
    source_checkpoint: str | Path,
    expected_level: int,
) -> BrainAblationConfig:
    config = resolve_ablation_configs([variant_name])[0]
    if int(config.b_level) != int(expected_level):
        raise ValueError(f"{variant_name!r} is not a B{int(expected_level)} config.")
    return replace(config, b_transfer_source_checkpoint=str(source_checkpoint))


def build_b1_capacity_config(
    variant_name: str = B1_CAPACITY_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B0_CURRENT_BRIDGE_DEFAULT_CHECKPOINT,
) -> BrainAblationConfig:
    return build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=1,
    )


def build_b2_temporal_threat_config(
    variant_name: str = B2_TEMPORAL_THREAT_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B1_THREAT_GUARD_DEFAULT_CHECKPOINT,
) -> BrainAblationConfig:
    return build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=2,
    )


def build_b3_contact_memory_config(
    variant_name: str = B3_CONTACT_MEMORY_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B2_TEMPORAL_THREAT_DEFAULT_CHECKPOINT,
) -> BrainAblationConfig:
    return build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=3,
    )


def build_b4_recovery_balance_config(
    variant_name: str = B4_RECOVERY_BALANCE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B3_RECURRENT_GUARD_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=4,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        replacements["b_controller_params"] = dict(controller_params)
    return replace(config, **replacements) if replacements else config


def build_b5_homeostatic_arbiter_config(
    variant_name: str = B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B4_GENETIC_RECOVERY_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=5,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        replacements["b_controller_params"] = dict(controller_params)
    return replace(config, **replacements) if replacements else config


def build_b6_risk_corridor_config(
    variant_name: str = B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B5_GENETIC_HOMEOSTASIS_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=6,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b7_affordance_budget_config(
    variant_name: str = B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B6_FUSED_RISK_RECURRENT_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=7,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b8_spatial_affordance_config(
    variant_name: str = B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B7_AFFORDANCE_BUDGET_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=8,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b9_waypoint_planner_config(
    variant_name: str = B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B8_SPATIAL_AFFORDANCE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=9,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b10_prospective_replay_config(
    variant_name: str = B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B9_WAYPOINT_PLANNER_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=10,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b11_confidence_arbiter_config(
    variant_name: str = B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B10_PROSPECTIVE_REPLAY_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=11,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b12_predictive_attention_config(
    variant_name: str = B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B11_CONFIDENCE_ARBITER_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=12,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b13_local_affordance_search_config(
    variant_name: str = B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B12_PREDICTIVE_ATTENTION_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=13,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b14_affordance_uncertainty_config(
    variant_name: str = B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B13_LOCAL_AFFORDANCE_SEARCH_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=14,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b15_option_critic_config(
    variant_name: str = B15_OPTION_CRITIC_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B14_AFFORDANCE_UNCERTAINTY_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=15,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b16_option_ensemble_config(
    variant_name: str = B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B15_OPTION_CRITIC_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=16,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b17_neuromodulated_ensemble_config(
    variant_name: str = B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B16_OPTION_ENSEMBLE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=17,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b18_eligibility_trace_config(
    variant_name: str = B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B17_NEUROMODULATED_ENSEMBLE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=18,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b19_episodic_meta_memory_config(
    variant_name: str = B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B18_ELIGIBILITY_TRACE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=19,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b20_working_memory_gate_config(
    variant_name: str = B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B19_EPISODIC_META_MEMORY_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=20,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b21_hippocampal_replay_config(
    variant_name: str = B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B20_WORKING_MEMORY_GATE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=21,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b22_prospective_replay_config(
    variant_name: str = B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B21_HIPPOCAMPAL_REPLAY_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=22,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b23_conflict_monitor_config(
    variant_name: str = B23_CONFLICT_MONITOR_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B22_PROSPECTIVE_REPLAY_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=23,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b24_precision_conflict_config(
    variant_name: str = B24_PRECISION_CONFLICT_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B23_CONFLICT_MONITOR_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=24,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b25_metacognitive_confidence_config(
    variant_name: str = B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B24_PRECISION_CONFLICT_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=25,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b26_allostatic_prediction_config(
    variant_name: str = B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B25_METACOGNITIVE_CONFIDENCE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=26,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b27_arousal_gain_config(
    variant_name: str = B27_AROUSAL_GAIN_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B26_ALLOSTATIC_PREDICTION_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=27,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b28_interoceptive_attention_config(
    variant_name: str = B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B27_AROUSAL_GAIN_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=28,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b29_salience_competition_config(
    variant_name: str = B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B28_INTEROCEPTIVE_ATTENTION_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=29,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b30_basal_ganglia_gate_config(
    variant_name: str = B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B29_SALIENCE_COMPETITION_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=30,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b31_dopamine_prediction_error_config(
    variant_name: str = B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B30_BASAL_GANGLIA_GATE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=31,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b32_actor_critic_value_config(
    variant_name: str = B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B31_DOPAMINE_PREDICTION_ERROR_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=32,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b33_td_error_decomposition_config(
    variant_name: str = B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B32_ACTOR_CRITIC_VALUE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=33,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b34_eligibility_credit_config(
    variant_name: str = B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B33_TD_ERROR_DECOMPOSITION_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=34,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b35_forward_model_value_config(
    variant_name: str = B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B34_ELIGIBILITY_CREDIT_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=35,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b36_latent_belief_state_config(
    variant_name: str = B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B35_FORWARD_MODEL_VALUE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=36,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b37_state_factor_gate_config(
    variant_name: str = B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B36_LATENT_BELIEF_STATE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=37,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b38_factor_attention_config(
    variant_name: str = B38_FACTOR_ATTENTION_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B37_STATE_FACTOR_GATE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=38,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b39_attention_binding_config(
    variant_name: str = B39_ATTENTION_BINDING_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B38_FACTOR_ATTENTION_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=39,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b40_global_workspace_config(
    variant_name: str = B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B39_ATTENTION_BINDING_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=40,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b41_executive_workspace_config(
    variant_name: str = B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B40_GLOBAL_WORKSPACE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=41,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b42_error_monitor_config(
    variant_name: str = B42_ERROR_MONITOR_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B41_EXECUTIVE_WORKSPACE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=42,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b43_adaptive_precision_config(
    variant_name: str = B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B42_ERROR_MONITOR_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=43,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b44_thalamic_relay_config(
    variant_name: str = B44_THALAMIC_RELAY_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B43_ADAPTIVE_PRECISION_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=44,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b45_reticular_inhibition_config(
    variant_name: str = B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B44_THALAMIC_RELAY_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=45,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b46_corticothalamic_feedback_config(
    variant_name: str = B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B45_RETICULAR_INHIBITION_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=46,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b47_oscillatory_synchrony_config(
    variant_name: str = B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B46_CORTICOTHALAMIC_FEEDBACK_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=47,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b48_cerebellar_timing_config(
    variant_name: str = B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B47_OSCILLATORY_SYNCHRONY_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=48,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b49_striatal_action_gate_config(
    variant_name: str = B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B48_CEREBELLAR_TIMING_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=49,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b50_habit_chunking_config(
    variant_name: str = B50_HABIT_CHUNKING_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B49_STRIATAL_ACTION_GATE_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=50,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b51_dopaminergic_habit_modulation_config(
    variant_name: str = B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = B50_HABIT_CHUNKING_DEFAULT_CHECKPOINT,
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=51,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b52_cholinergic_precision_gate_config(
    variant_name: str = B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b51_dopaminergic_habit_modulation_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=52,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b53_noradrenergic_arousal_gain_config(
    variant_name: str = B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b52_cholinergic_precision_gate_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=53,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b54_serotonergic_patience_gate_config(
    variant_name: str = B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b53_noradrenergic_arousal_gain_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=54,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b55_hypothalamic_drive_coupling_config(
    variant_name: str = B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b54_serotonergic_patience_gate_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=55,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b56_hpa_stress_axis_config(
    variant_name: str = B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b55_hypothalamic_drive_coupling_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=56,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b57_insular_interoceptive_awareness_config(
    variant_name: str = B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b56_hpa_stress_axis_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=57,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b58_acc_conflict_monitor_config(
    variant_name: str = B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b57_insular_interoceptive_awareness_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=58,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b59_prefrontal_goal_context_config(
    variant_name: str = B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b58_acc_conflict_monitor_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=59,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b60_orbitofrontal_outcome_value_config(
    variant_name: str = B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b59_prefrontal_goal_context_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=60,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b61_amygdala_safety_value_config(
    variant_name: str = B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b60_orbitofrontal_outcome_value_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=61,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config


def build_b62_defensive_mode_selector_config(
    variant_name: str = B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
    *,
    source_checkpoint: str | Path = (
        "artifacts/b_series/evolution/"
        "b61_amygdala_safety_value_h48_bridge_policy/seed_7/best"
    ),
    controller_profile: str | None = None,
    controller_params: dict[str, float] | None = None,
) -> BrainAblationConfig:
    config = build_b_evolution_config(
        variant_name,
        source_checkpoint=source_checkpoint,
        expected_level=62,
    )
    replacements: dict[str, object] = {}
    if controller_profile is not None:
        replacements["b_controller_profile"] = str(controller_profile)
    if controller_params is not None:
        merged = dict(config.b_controller_params)
        merged.update(dict(controller_params))
        replacements["b_controller_params"] = merged
    return replace(config, **replacements) if replacements else config
