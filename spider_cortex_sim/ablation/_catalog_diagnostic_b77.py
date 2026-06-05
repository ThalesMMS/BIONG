from __future__ import annotations

from ._catalog_shared import *


B77_SOURCE_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b76_cerebellar_stride_gate_h48_bridge_policy/seed_7/best"
)

B77_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B77_OLIVARY_ERROR_CORRECTION_H48_POLICY_NAME,
        77,
        48,
        parent_level=76,
        source_checkpoint=B77_SOURCE_CHECKPOINT,
        controller_profile="olivary_error_correction",
        controller_params=B77_BASE_OLIVARY_ERROR_PARAMS,
    ),
    b_series_diagnostic_variant(
        B77_ERROR_PREDICTION_PACING_H48_POLICY_NAME,
        77,
        48,
        parent_level=76,
        source_checkpoint=B77_SOURCE_CHECKPOINT,
        controller_profile="error_prediction_pacing",
        controller_params=(
            B77_BASE_OLIVARY_ERROR_PARAMS
            | {"b77_prediction_gain": 0.35, "b77_release_threshold": 0.28}
        ),
    ),
    b_series_diagnostic_variant(
        B77_CLIMBING_FIBER_RECOVERY_H48_POLICY_NAME,
        77,
        48,
        parent_level=76,
        source_checkpoint=B77_SOURCE_CHECKPOINT,
        controller_profile="climbing_fiber_recovery",
        controller_params=(
            B77_BASE_OLIVARY_ERROR_PARAMS
            | {"b77_climbing_gain": 0.35, "b77_error_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B77_OLIVARY_ERROR_CORRECTION_H56_POLICY_NAME,
        77,
        56,
        parent_level=76,
        source_checkpoint=B77_SOURCE_CHECKPOINT,
        controller_profile="olivary_error_correction_h56",
        controller_params=(
            B77_BASE_OLIVARY_ERROR_PARAMS
            | {"b77_error_decay": 0.92, "b77_error_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B77_GENETIC_OLIVARY_ERROR_H48_POLICY_NAME,
        77,
        48,
        parent_level=76,
        source_checkpoint=B77_SOURCE_CHECKPOINT,
        controller_profile="genetic_olivary_error",
        controller_params=B77_BASE_OLIVARY_ERROR_PARAMS,
    ),
)


def diagnostic_b77_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B77_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
