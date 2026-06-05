from __future__ import annotations

from ._catalog_shared import *


B72_SOURCE_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b71_tectal_orienting_gate_h48_bridge_policy/seed_7/best"
)

B72_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B72_PULVINAR_ATTENTION_GATE_H48_POLICY_NAME,
        72,
        48,
        parent_level=71,
        source_checkpoint=B72_SOURCE_CHECKPOINT,
        controller_profile="pulvinar_attention_gate",
        controller_params=B72_BASE_PULVINAR_ATTENTION_PARAMS,
    ),
    b_series_diagnostic_variant(
        B72_DISTRACTOR_FILTER_PACING_H48_POLICY_NAME,
        72,
        48,
        parent_level=71,
        source_checkpoint=B72_SOURCE_CHECKPOINT,
        controller_profile="distractor_filter_pacing",
        controller_params=(
            B72_BASE_PULVINAR_ATTENTION_PARAMS
            | {"b72_distractor_gain": 0.35, "b72_hold_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B72_FOCUS_LOCK_RECOVERY_H48_POLICY_NAME,
        72,
        48,
        parent_level=71,
        source_checkpoint=B72_SOURCE_CHECKPOINT,
        controller_profile="focus_lock_recovery",
        controller_params=(
            B72_BASE_PULVINAR_ATTENTION_PARAMS
            | {"b72_focus_gain": 0.36, "b72_attention_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B72_PULVINAR_ATTENTION_GATE_H56_POLICY_NAME,
        72,
        56,
        parent_level=71,
        source_checkpoint=B72_SOURCE_CHECKPOINT,
        controller_profile="pulvinar_attention_gate_h56",
        controller_params=(
            B72_BASE_PULVINAR_ATTENTION_PARAMS
            | {"b72_attention_decay": 0.92, "b72_attention_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B72_GENETIC_PULVINAR_ATTENTION_H48_POLICY_NAME,
        72,
        48,
        parent_level=71,
        source_checkpoint=B72_SOURCE_CHECKPOINT,
        controller_profile="genetic_pulvinar_attention",
        controller_params=B72_BASE_PULVINAR_ATTENTION_PARAMS,
    ),
)


def diagnostic_b72_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B72_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
