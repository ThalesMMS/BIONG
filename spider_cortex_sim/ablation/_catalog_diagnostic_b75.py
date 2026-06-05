from __future__ import annotations

from ._catalog_shared import *


B75_SOURCE_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b74_thalamic_rebound_gate_h48_bridge_policy/seed_7/best"
)

B75_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B75_BASAL_THALAMIC_RELEASE_H48_POLICY_NAME,
        75,
        48,
        parent_level=74,
        source_checkpoint=B75_SOURCE_CHECKPOINT,
        controller_profile="basal_thalamic_release",
        controller_params=B75_BASE_BASAL_THALAMIC_RELEASE_PARAMS,
    ),
    b_series_diagnostic_variant(
        B75_RELEASE_BURST_PACING_H48_POLICY_NAME,
        75,
        48,
        parent_level=74,
        source_checkpoint=B75_SOURCE_CHECKPOINT,
        controller_profile="release_burst_pacing",
        controller_params=(
            B75_BASE_BASAL_THALAMIC_RELEASE_PARAMS
            | {"b75_burst_gain": 0.34, "b75_release_threshold": 0.28}
        ),
    ),
    b_series_diagnostic_variant(
        B75_GO_NOGO_REBOUND_TIMING_H48_POLICY_NAME,
        75,
        48,
        parent_level=74,
        source_checkpoint=B75_SOURCE_CHECKPOINT,
        controller_profile="go_nogo_rebound_timing",
        controller_params=(
            B75_BASE_BASAL_THALAMIC_RELEASE_PARAMS
            | {"b75_go_gain": 0.35, "b75_nogo_gain": 0.35, "b75_release_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B75_BASAL_THALAMIC_RELEASE_H56_POLICY_NAME,
        75,
        56,
        parent_level=74,
        source_checkpoint=B75_SOURCE_CHECKPOINT,
        controller_profile="basal_thalamic_release_h56",
        controller_params=(
            B75_BASE_BASAL_THALAMIC_RELEASE_PARAMS
            | {"b75_timing_decay": 0.92, "b75_release_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B75_GENETIC_BASAL_THALAMIC_RELEASE_H48_POLICY_NAME,
        75,
        48,
        parent_level=74,
        source_checkpoint=B75_SOURCE_CHECKPOINT,
        controller_profile="genetic_basal_thalamic_release",
        controller_params=B75_BASE_BASAL_THALAMIC_RELEASE_PARAMS,
    ),
)


def diagnostic_b75_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B75_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
