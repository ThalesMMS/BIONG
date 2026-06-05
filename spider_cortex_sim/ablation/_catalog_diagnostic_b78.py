from __future__ import annotations

from ._catalog_shared import *


B78_SOURCE_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b77_olivary_error_correction_h48_bridge_policy/seed_7/best"
)

B78_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B78_VESTIBULAR_BALANCE_H48_POLICY_NAME,
        78,
        48,
        parent_level=77,
        source_checkpoint=B78_SOURCE_CHECKPOINT,
        controller_profile="vestibular_balance",
        controller_params=B78_BASE_VESTIBULAR_BALANCE_PARAMS,
    ),
    b_series_diagnostic_variant(
        B78_HEAD_STABILIZATION_H48_POLICY_NAME,
        78,
        48,
        parent_level=77,
        source_checkpoint=B78_SOURCE_CHECKPOINT,
        controller_profile="head_stabilization",
        controller_params=(
            B78_BASE_VESTIBULAR_BALANCE_PARAMS
            | {"b78_stabilization_gain": 0.35, "b78_release_threshold": 0.28}
        ),
    ),
    b_series_diagnostic_variant(
        B78_LOCOMOTOR_CONFIDENCE_H48_POLICY_NAME,
        78,
        48,
        parent_level=77,
        source_checkpoint=B78_SOURCE_CHECKPOINT,
        controller_profile="locomotor_confidence",
        controller_params=(
            B78_BASE_VESTIBULAR_BALANCE_PARAMS
            | {"b78_confidence_gain": 0.35, "b78_balance_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B78_VESTIBULAR_BALANCE_H56_POLICY_NAME,
        78,
        56,
        parent_level=77,
        source_checkpoint=B78_SOURCE_CHECKPOINT,
        controller_profile="vestibular_balance_h56",
        controller_params=(
            B78_BASE_VESTIBULAR_BALANCE_PARAMS
            | {"b78_balance_decay": 0.92, "b78_balance_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B78_GENETIC_VESTIBULAR_BALANCE_H48_POLICY_NAME,
        78,
        48,
        parent_level=77,
        source_checkpoint=B78_SOURCE_CHECKPOINT,
        controller_profile="genetic_vestibular_balance",
        controller_params=B78_BASE_VESTIBULAR_BALANCE_PARAMS,
    ),
)


def diagnostic_b78_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B78_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
