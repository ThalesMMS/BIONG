from __future__ import annotations

from ._catalog_shared import *


B67_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B67_GLIAL_ENERGY_GATE_H48_POLICY_NAME,
        67,
        48,
        parent_level=66,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b66_immune_malaise_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="glial_energy_gate",
        controller_params=B67_BASE_GLIAL_ENERGY_PARAMS,
    ),
    b_series_diagnostic_variant(
        B67_LACTATE_RECOVERY_SUPPORT_H48_POLICY_NAME,
        67,
        48,
        parent_level=66,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b66_immune_malaise_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="lactate_recovery_support",
        controller_params=(
            B67_BASE_GLIAL_ENERGY_PARAMS
            | {"b67_lactate_support_gain": 0.34, "b67_hold_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B67_FATIGUE_AWARE_RELEASE_H48_POLICY_NAME,
        67,
        48,
        parent_level=66,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b66_immune_malaise_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="fatigue_aware_release",
        controller_params=(
            B67_BASE_GLIAL_ENERGY_PARAMS
            | {"b67_fatigue_gain": 0.30, "b67_release_threshold": 0.26}
        ),
    ),
    b_series_diagnostic_variant(
        B67_GLIAL_ENERGY_GATE_H56_POLICY_NAME,
        67,
        56,
        parent_level=66,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b66_immune_malaise_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="glial_energy_gate_h56",
        controller_params=(
            B67_BASE_GLIAL_ENERGY_PARAMS
            | {"b67_glial_decay": 0.92, "b67_recovery_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B67_GENETIC_GLIAL_ENERGY_H48_POLICY_NAME,
        67,
        48,
        parent_level=66,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b66_immune_malaise_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="genetic_glial_energy",
        controller_params=B67_BASE_GLIAL_ENERGY_PARAMS,
    ),
)


def diagnostic_b67_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B67_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
