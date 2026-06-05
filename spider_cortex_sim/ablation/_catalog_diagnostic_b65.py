from __future__ import annotations

from ._catalog_shared import *


B65_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B65_ENTERIC_ASSIMILATION_GATE_H48_POLICY_NAME,
        65,
        48,
        parent_level=64,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b64_vagal_recovery_brake_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="enteric_assimilation_gate",
        controller_params=B65_BASE_ENTERIC_ASSIMILATION_PARAMS,
    ),
    b_series_diagnostic_variant(
        B65_POST_RECOVERY_FORAGE_RELEASE_H48_POLICY_NAME,
        65,
        48,
        parent_level=64,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b64_vagal_recovery_brake_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="post_recovery_forage_release",
        controller_params=(
            B65_BASE_ENTERIC_ASSIMILATION_PARAMS
            | {"b65_forage_readiness_gain": 0.36, "b65_release_threshold": 0.30}
        ),
    ),
    b_series_diagnostic_variant(
        B65_SHELTER_DIGESTIVE_COUPLING_H48_POLICY_NAME,
        65,
        48,
        parent_level=64,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b64_vagal_recovery_brake_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="shelter_digestive_coupling",
        controller_params=(
            B65_BASE_ENTERIC_ASSIMILATION_PARAMS
            | {"b65_recovery_coupling_gain": 0.36, "b65_digestive_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B65_ENTERIC_ASSIMILATION_GATE_H56_POLICY_NAME,
        65,
        56,
        parent_level=64,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b64_vagal_recovery_brake_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="enteric_assimilation_gate_h56",
        controller_params=(
            B65_BASE_ENTERIC_ASSIMILATION_PARAMS
            | {"b65_enteric_decay": 0.92, "b65_digestive_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B65_GENETIC_ENTERIC_ASSIMILATION_H48_POLICY_NAME,
        65,
        48,
        parent_level=64,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b64_vagal_recovery_brake_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="genetic_enteric_assimilation",
        controller_params=B65_BASE_ENTERIC_ASSIMILATION_PARAMS,
    ),
)


def diagnostic_b65_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B65_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
