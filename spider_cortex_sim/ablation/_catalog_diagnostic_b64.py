from __future__ import annotations

from ._catalog_shared import *


B64_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B64_VAGAL_RECOVERY_BRAKE_H48_POLICY_NAME,
        64,
        48,
        parent_level=63,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b63_periaqueductal_escape_sequence_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="vagal_recovery_brake",
        controller_params=B64_BASE_VAGAL_RECOVERY_PARAMS,
    ),
    b_series_diagnostic_variant(
        B64_POST_ESCAPE_REST_GATE_H48_POLICY_NAME,
        64,
        48,
        parent_level=63,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b63_periaqueductal_escape_sequence_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="post_escape_rest_gate",
        controller_params=(
            B64_BASE_VAGAL_RECOVERY_PARAMS
            | {"b64_recovery_tone_gain": 0.38, "b64_rest_threshold": 0.18}
        ),
    ),
    b_series_diagnostic_variant(
        B64_SHELTER_RECOVERY_COUPLING_H48_POLICY_NAME,
        64,
        48,
        parent_level=63,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b63_periaqueductal_escape_sequence_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="shelter_recovery_coupling",
        controller_params=(
            B64_BASE_VAGAL_RECOVERY_PARAMS
            | {"b64_post_escape_gain": 0.36, "b64_recovery_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B64_VAGAL_RECOVERY_BRAKE_H56_POLICY_NAME,
        64,
        56,
        parent_level=63,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b63_periaqueductal_escape_sequence_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="vagal_recovery_brake_h56",
        controller_params=(
            B64_BASE_VAGAL_RECOVERY_PARAMS
            | {"b64_recovery_decay": 0.92, "b64_recovery_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B64_GENETIC_RECOVERY_BRAKE_H48_POLICY_NAME,
        64,
        48,
        parent_level=63,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b63_periaqueductal_escape_sequence_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="genetic_recovery_brake",
        controller_params=B64_BASE_VAGAL_RECOVERY_PARAMS,
    ),
)


def diagnostic_b64_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B64_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
