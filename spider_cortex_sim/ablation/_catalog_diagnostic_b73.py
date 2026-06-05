from __future__ import annotations

from ._catalog_shared import *


B73_SOURCE_CHECKPOINT = (
    "artifacts/b_series/evolution/"
    "b72_pulvinar_attention_gate_h48_bridge_policy/seed_7/best"
)

B73_DIAGNOSTIC_VARIANT_SPECS: tuple[BSeriesDiagnosticVariantSpec, ...] = (
    b_series_diagnostic_variant(
        B73_RETICULAR_INHIBITION_GATE_H48_POLICY_NAME,
        73,
        48,
        parent_level=72,
        source_checkpoint=B73_SOURCE_CHECKPOINT,
        controller_profile="reticular_inhibition_gate",
        controller_params=B73_BASE_RETICULAR_INHIBITION_PARAMS,
    ),
    b_series_diagnostic_variant(
        B73_SURROUND_SUPPRESSION_PACING_H48_POLICY_NAME,
        73,
        48,
        parent_level=72,
        source_checkpoint=B73_SOURCE_CHECKPOINT,
        controller_profile="surround_suppression_pacing",
        controller_params=(
            B73_BASE_RETICULAR_INHIBITION_PARAMS
            | {"b73_suppression_gain": 0.35, "b73_hold_threshold": 0.16}
        ),
    ),
    b_series_diagnostic_variant(
        B73_FOCUS_RELEASE_RECOVERY_H48_POLICY_NAME,
        73,
        48,
        parent_level=72,
        source_checkpoint=B73_SOURCE_CHECKPOINT,
        controller_profile="focus_release_recovery",
        controller_params=(
            B73_BASE_RETICULAR_INHIBITION_PARAMS
            | {"b73_release_gain": 0.34, "b73_inhibition_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B73_RETICULAR_INHIBITION_GATE_H56_POLICY_NAME,
        73,
        56,
        parent_level=72,
        source_checkpoint=B73_SOURCE_CHECKPOINT,
        controller_profile="reticular_inhibition_gate_h56",
        controller_params=(
            B73_BASE_RETICULAR_INHIBITION_PARAMS
            | {"b73_inhibition_decay": 0.92, "b73_inhibition_lock_ticks": 5.0}
        ),
    ),
    b_series_diagnostic_variant(
        B73_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME,
        73,
        48,
        parent_level=72,
        source_checkpoint=B73_SOURCE_CHECKPOINT,
        controller_profile="genetic_reticular_inhibition",
        controller_params=B73_BASE_RETICULAR_INHIBITION_PARAMS,
    ),
)


def diagnostic_b73_configs(profile_fields: dict[str, object]) -> Dict[str, BrainAblationConfig]:
    return build_b_series_diagnostic_configs(B73_DIAGNOSTIC_VARIANT_SPECS, profile_fields)
