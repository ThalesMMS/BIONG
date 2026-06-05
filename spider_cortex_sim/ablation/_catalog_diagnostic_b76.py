from __future__ import annotations

from ._catalog_shared import *


B76_SOURCE_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b75_basal_thalamic_release_h48_bridge_policy/seed_7/best"
)

B76_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B76_CEREBELLAR_STRIDE_GATE_H48_POLICY_NAME,
        76,
        48,
        parent_level=75,
        source_checkpoint=B76_SOURCE_CHECKPOINT,
        controller_profile="cerebellar_stride_gate",
        controller_params=B76_BASE_CEREBELLAR_STRIDE_PARAMS,
    ),
    b_series_diagnostic_variant(
        B76_STRIDE_ERROR_PACING_H48_POLICY_NAME,
        76,
        48,
        parent_level=75,
        source_checkpoint=B76_SOURCE_CHECKPOINT,
        controller_profile="stride_error_pacing",
        controller_params=(
            B76_BASE_CEREBELLAR_STRIDE_PARAMS
            | {"b76_error_gain": 0.35, "b76_hold_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B76_BURST_SMOOTHING_RECOVERY_H48_POLICY_NAME,
        76,
        48,
        parent_level=75,
        source_checkpoint=B76_SOURCE_CHECKPOINT,
        controller_profile="burst_smoothing_recovery",
        controller_params=(
            B76_BASE_CEREBELLAR_STRIDE_PARAMS
            | {"b76_smoothing_gain": 0.34, "b76_stride_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B76_CEREBELLAR_STRIDE_GATE_H56_POLICY_NAME,
        76,
        56,
        parent_level=75,
        source_checkpoint=B76_SOURCE_CHECKPOINT,
        controller_profile="cerebellar_stride_gate_h56",
        controller_params=(
            B76_BASE_CEREBELLAR_STRIDE_PARAMS
            | {"b76_timing_decay": 0.92, "b76_stride_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B76_GENETIC_CEREBELLAR_STRIDE_H48_POLICY_NAME,
        76,
        48,
        parent_level=75,
        source_checkpoint=B76_SOURCE_CHECKPOINT,
        controller_profile="genetic_cerebellar_stride",
        controller_params=B76_BASE_CEREBELLAR_STRIDE_PARAMS,
    ),
)


def diagnostic_b76_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B76_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
