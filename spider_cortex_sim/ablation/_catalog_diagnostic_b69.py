from __future__ import annotations

from ._catalog_shared import *


B69_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B69_VESTIBULAR_ORIENTATION_GATE_H48_POLICY_NAME,
        69,
        48,
        parent_level=68,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b68_proprioceptive_pacing_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="vestibular_orientation_gate",
        controller_params=B69_BASE_VESTIBULAR_ORIENTATION_PARAMS,
    ),
    b_series_diagnostic_variant(
        B69_HEADING_STABILITY_PACING_H48_POLICY_NAME,
        69,
        48,
        parent_level=68,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b68_proprioceptive_pacing_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="heading_stability_pacing",
        controller_params=(
            B69_BASE_VESTIBULAR_ORIENTATION_PARAMS
            | {"b69_heading_confidence_gain": 0.35, "b69_release_threshold": 0.28}
        ),
    ),
    b_series_diagnostic_variant(
        B69_TURN_ERROR_RECOVERY_H48_POLICY_NAME,
        69,
        48,
        parent_level=68,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b68_proprioceptive_pacing_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="turn_error_recovery",
        controller_params=(
            B69_BASE_VESTIBULAR_ORIENTATION_PARAMS
            | {"b69_turn_error_gain": 0.34, "b69_hold_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B69_VESTIBULAR_ORIENTATION_GATE_H56_POLICY_NAME,
        69,
        56,
        parent_level=68,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b68_proprioceptive_pacing_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="vestibular_orientation_gate_h56",
        controller_params=(
            B69_BASE_VESTIBULAR_ORIENTATION_PARAMS
            | {"b69_orientation_decay": 0.92, "b69_orientation_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B69_GENETIC_ORIENTATION_GATE_H48_POLICY_NAME,
        69,
        48,
        parent_level=68,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b68_proprioceptive_pacing_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="genetic_orientation_gate",
        controller_params=B69_BASE_VESTIBULAR_ORIENTATION_PARAMS,
    ),
)


def diagnostic_b69_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B69_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
