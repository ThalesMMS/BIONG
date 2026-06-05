from __future__ import annotations

from ._catalog_shared import *


B66_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B66_IMMUNE_MALAISE_GATE_H48_POLICY_NAME,
        66,
        48,
        parent_level=65,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b65_enteric_assimilation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="immune_malaise_gate",
        controller_params=B66_BASE_IMMUNE_MALAISE_PARAMS,
    ),
    b_series_diagnostic_variant(
        B66_INFLAMMATORY_RECOVERY_HOLD_H48_POLICY_NAME,
        66,
        48,
        parent_level=65,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b65_enteric_assimilation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="inflammatory_recovery_hold",
        controller_params=(
            B66_BASE_IMMUNE_MALAISE_PARAMS
            | {"b66_malaise_gain": 0.36, "b66_hold_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B66_DAMAGE_AWARE_FORAGE_RELEASE_H48_POLICY_NAME,
        66,
        48,
        parent_level=65,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b65_enteric_assimilation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="damage_aware_forage_release",
        controller_params=(
            B66_BASE_IMMUNE_MALAISE_PARAMS
            | {"b66_recovery_veto_gain": 0.26, "b66_release_threshold": 0.26}
        ),
    ),
    b_series_diagnostic_variant(
        B66_IMMUNE_MALAISE_GATE_H56_POLICY_NAME,
        66,
        56,
        parent_level=65,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b65_enteric_assimilation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="immune_malaise_gate_h56",
        controller_params=(
            B66_BASE_IMMUNE_MALAISE_PARAMS
            | {"b66_immune_decay": 0.92, "b66_inflammation_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B66_GENETIC_IMMUNE_MALAISE_H48_POLICY_NAME,
        66,
        48,
        parent_level=65,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b65_enteric_assimilation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="genetic_immune_malaise",
        controller_params=B66_BASE_IMMUNE_MALAISE_PARAMS,
    ),
)


def diagnostic_b66_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B66_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
