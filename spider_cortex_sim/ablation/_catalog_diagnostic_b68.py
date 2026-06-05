from __future__ import annotations

from ._catalog_shared import *


B68_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B68_PROPRIOCEPTIVE_PACING_GATE_H48_POLICY_NAME,
        68,
        48,
        parent_level=67,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b67_glial_energy_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="proprioceptive_pacing_gate",
        controller_params=B68_BASE_MOTOR_PACING_PARAMS,
    ),
    b_series_diagnostic_variant(
        B68_STRIDE_RECOVERY_PACING_H48_POLICY_NAME,
        68,
        48,
        parent_level=67,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b67_glial_energy_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="stride_recovery_pacing",
        controller_params=(
            B68_BASE_MOTOR_PACING_PARAMS
            | {"b68_stride_pacing_gain": 0.34, "b68_hold_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B68_OVEREXERTION_AWARE_RELEASE_H48_POLICY_NAME,
        68,
        48,
        parent_level=67,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b67_glial_energy_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="overexertion_aware_release",
        controller_params=(
            B68_BASE_MOTOR_PACING_PARAMS
            | {"b68_overexertion_gain": 0.30, "b68_release_threshold": 0.26}
        ),
    ),
    b_series_diagnostic_variant(
        B68_PROPRIOCEPTIVE_PACING_GATE_H56_POLICY_NAME,
        68,
        56,
        parent_level=67,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b67_glial_energy_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="proprioceptive_pacing_gate_h56",
        controller_params=(
            B68_BASE_MOTOR_PACING_PARAMS
            | {"b68_motor_decay": 0.92, "b68_pacing_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B68_GENETIC_MOTOR_PACING_H48_POLICY_NAME,
        68,
        48,
        parent_level=67,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b67_glial_energy_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="genetic_motor_pacing",
        controller_params=B68_BASE_MOTOR_PACING_PARAMS,
    ),
)


def diagnostic_b68_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B68_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
