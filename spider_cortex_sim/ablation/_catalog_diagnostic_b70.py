from __future__ import annotations

from ._catalog_shared import *


B70_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B70_OPTIC_FLOW_STABILIZATION_GATE_H48_POLICY_NAME,
        70,
        48,
        parent_level=69,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b69_vestibular_orientation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="optic_flow_stabilization_gate",
        controller_params=B70_BASE_OPTIC_FLOW_PARAMS,
    ),
    b_series_diagnostic_variant(
        B70_LATERAL_FLOW_PACING_H48_POLICY_NAME,
        70,
        48,
        parent_level=69,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b69_vestibular_orientation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="lateral_flow_pacing",
        controller_params=(
            B70_BASE_OPTIC_FLOW_PARAMS
            | {"b70_lateral_drift_gain": 0.34, "b70_hold_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B70_LOOMING_RISK_RECOVERY_H48_POLICY_NAME,
        70,
        48,
        parent_level=69,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b69_vestibular_orientation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="looming_risk_recovery",
        controller_params=(
            B70_BASE_OPTIC_FLOW_PARAMS
            | {"b70_looming_risk_gain": 0.32, "b70_release_threshold": 0.28}
        ),
    ),
    b_series_diagnostic_variant(
        B70_OPTIC_FLOW_STABILIZATION_GATE_H56_POLICY_NAME,
        70,
        56,
        parent_level=69,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b69_vestibular_orientation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="optic_flow_stabilization_gate_h56",
        controller_params=(
            B70_BASE_OPTIC_FLOW_PARAMS
            | {"b70_flow_decay": 0.92, "b70_flow_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B70_GENETIC_OPTIC_FLOW_GATE_H48_POLICY_NAME,
        70,
        48,
        parent_level=69,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b69_vestibular_orientation_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="genetic_optic_flow_gate",
        controller_params=B70_BASE_OPTIC_FLOW_PARAMS,
    ),
)


def diagnostic_b70_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B70_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
