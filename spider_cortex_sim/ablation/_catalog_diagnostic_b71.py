from __future__ import annotations

from ._catalog_shared import *


B71_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B71_TECTAL_ORIENTING_GATE_H48_POLICY_NAME,
        71,
        48,
        parent_level=70,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b70_optic_flow_stabilization_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="tectal_orienting_gate",
        controller_params=B71_BASE_TECTAL_ORIENTING_PARAMS,
    ),
    b_series_diagnostic_variant(
        B71_SALIENCE_MAP_PACING_H48_POLICY_NAME,
        71,
        48,
        parent_level=70,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b70_optic_flow_stabilization_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="salience_map_pacing",
        controller_params=(
            B71_BASE_TECTAL_ORIENTING_PARAMS
            | {"b71_target_salience_gain": 0.35, "b71_release_threshold": 0.28}
        ),
    ),
    b_series_diagnostic_variant(
        B71_COLLISION_VETO_RECOVERY_H48_POLICY_NAME,
        71,
        48,
        parent_level=70,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b70_optic_flow_stabilization_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="collision_veto_recovery",
        controller_params=(
            B71_BASE_TECTAL_ORIENTING_PARAMS
            | {"b71_collision_veto_gain": 0.32, "b71_hold_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B71_TECTAL_ORIENTING_GATE_H56_POLICY_NAME,
        71,
        56,
        parent_level=70,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b70_optic_flow_stabilization_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="tectal_orienting_gate_h56",
        controller_params=(
            B71_BASE_TECTAL_ORIENTING_PARAMS
            | {"b71_orienting_decay": 0.92, "b71_orienting_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B71_GENETIC_TECTAL_ORIENTING_H48_POLICY_NAME,
        71,
        48,
        parent_level=70,
        source_checkpoint=(
            "artifacts/b_series/evolution/"
            "b70_optic_flow_stabilization_gate_h48_bridge_policy/seed_7/best"
        ),
        controller_profile="genetic_tectal_orienting",
        controller_params=B71_BASE_TECTAL_ORIENTING_PARAMS,
    ),
)


def diagnostic_b71_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B71_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
