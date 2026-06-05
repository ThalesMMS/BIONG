from __future__ import annotations

from ._catalog_shared import *


B63_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B63_PERIAQUEDUCTAL_ESCAPE_SEQUENCE_H48_POLICY_NAME,
        63,
        48,
        parent_level=62,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b62_defensive_mode_selector_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="periaqueductal_escape_sequence",
        controller_params=B63_BASE_PERIAQUEDUCTAL_ESCAPE_PARAMS,
    ),
    b_series_diagnostic_variant(
        B63_FREEZE_RELEASE_ESCAPE_H48_POLICY_NAME,
        63,
        48,
        parent_level=62,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b62_defensive_mode_selector_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="freeze_release_escape",
        controller_params=(
            B63_BASE_PERIAQUEDUCTAL_ESCAPE_PARAMS
            | {"b63_freeze_release_gain": 0.30, "b63_freeze_release_threshold": 0.18}
        ),
    ),
    b_series_diagnostic_variant(
        B63_SHELTER_VECTOR_SEQUENCE_H48_POLICY_NAME,
        63,
        48,
        parent_level=62,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b62_defensive_mode_selector_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="shelter_vector_sequence",
        controller_params=(
            B63_BASE_PERIAQUEDUCTAL_ESCAPE_PARAMS
            | {"b63_shelter_vector_gain": 0.36, "b63_escape_threshold": 0.20}
        ),
    ),
    b_series_diagnostic_variant(
        B63_PERIAQUEDUCTAL_ESCAPE_SEQUENCE_H56_POLICY_NAME,
        63,
        56,
        parent_level=62,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b62_defensive_mode_selector_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="periaqueductal_escape_sequence_h56",
        controller_params=(
            B63_BASE_PERIAQUEDUCTAL_ESCAPE_PARAMS
            | {"b63_escape_decay": 0.90, "b63_escape_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B63_GENETIC_ESCAPE_SEQUENCE_H48_POLICY_NAME,
        63,
        48,
        parent_level=62,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b62_defensive_mode_selector_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="genetic_escape_sequence",
        controller_params=B63_BASE_PERIAQUEDUCTAL_ESCAPE_PARAMS,
    ),
)


def diagnostic_b63_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B63_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
