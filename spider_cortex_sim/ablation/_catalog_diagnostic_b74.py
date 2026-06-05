from __future__ import annotations

from ._catalog_shared import *


B74_SOURCE_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b73_reticular_inhibition_gate_h48_bridge_policy/seed_7/best"
)

B74_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B74_THALAMIC_REBOUND_GATE_H48_POLICY_NAME,
        74,
        48,
        parent_level=73,
        source_checkpoint=B74_SOURCE_CHECKPOINT,
        controller_profile="thalamic_rebound_gate",
        controller_params=B74_BASE_THALAMIC_REBOUND_PARAMS,
    ),
    b_series_diagnostic_variant(
        B74_REBOUND_RELEASE_PACING_H48_POLICY_NAME,
        74,
        48,
        parent_level=73,
        source_checkpoint=B74_SOURCE_CHECKPOINT,
        controller_profile="rebound_release_pacing",
        controller_params=(
            B74_BASE_THALAMIC_REBOUND_PARAMS
            | {"b74_release_gain": 0.34, "b74_release_threshold": 0.28}
        ),
    ),
    b_series_diagnostic_variant(
        B74_POST_INHIBITION_RECOVERY_H48_POLICY_NAME,
        74,
        48,
        parent_level=73,
        source_checkpoint=B74_SOURCE_CHECKPOINT,
        controller_profile="post_inhibition_recovery",
        controller_params=(
            B74_BASE_THALAMIC_REBOUND_PARAMS
            | {"b74_aftereffect_gain": 0.35, "b74_rebound_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B74_THALAMIC_REBOUND_GATE_H56_POLICY_NAME,
        74,
        56,
        parent_level=73,
        source_checkpoint=B74_SOURCE_CHECKPOINT,
        controller_profile="thalamic_rebound_gate_h56",
        controller_params=(
            B74_BASE_THALAMIC_REBOUND_PARAMS
            | {"b74_rebound_decay": 0.92, "b74_rebound_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B74_GENETIC_THALAMIC_REBOUND_H48_POLICY_NAME,
        74,
        48,
        parent_level=73,
        source_checkpoint=B74_SOURCE_CHECKPOINT,
        controller_profile="genetic_thalamic_rebound",
        controller_params=B74_BASE_THALAMIC_REBOUND_PARAMS,
    ),
)


def diagnostic_b74_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B74_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
