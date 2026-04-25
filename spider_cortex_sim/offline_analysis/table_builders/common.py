from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..constants import (
    AUSTERE_SURVIVAL_THRESHOLD,
    CLASSIFICATION_LABELS,
    DEFAULT_CONFIDENCE_LEVEL,
    LADDER_ADJACENT_COMPARISONS,
    LADDER_PRIMARY_VARIANT_BY_RUNG,
    LADDER_PROTOCOL_NAMES,
    LADDER_RUNG_MAPPING,
    MODULAR_CREDIT_RUNGS,
)
from ..extractors import (
    compare_capacity_totals,
    extract_architecture_capacity,
    extract_capacity_sweeps,
    extract_credit_metrics,
    extract_reward_profile_ladder,
    _format_failure_indicator,
    _variant_with_minimal_reflex_support,
    build_primary_benchmark,
    build_reflex_dependence_indicators,
    extract_ablations,
)
from ..uncertainty import (
    _ci_row_fields,
    _cohens_d_row,
    _payload_has_zero_reflex_scale,
    _payload_metric_seed_items,
    _payload_uncertainty,
    _primary_benchmark_scenario_success,
    _primary_benchmark_source_payload,
    _uncertainty_or_empty,
    _uncertainty_or_seed_delta,
)
from ..utils import (
    _coerce_float,
    _coerce_optional_float,
    _dominant_module_by_score,
    _mapping_or_empty,
)
from ..writers import _table

CAPACITY_SWEEP_INTERPRETATION_GUIDANCE = (
    "Improvement with capacity -> possible undercapacity; "
    "no improvement -> likely interface/connection/reward issue."
)
CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD = 0.05
CREDIT_ASSIGNMENT_STRATEGY_ORDER: tuple[str, ...] = (
    "broadcast",
    "local_only",
    "counterfactual",
)
CREDIT_ASSIGNMENT_VARIANT_CANDIDATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("A2", "broadcast"): ("three_center_modular",),
    ("A2", "local_only"): (
        "three_center_modular_local_credit",
        "three_center_local_credit",
    ),
    ("A2", "counterfactual"): (
        "three_center_modular_counterfactual",
        "three_center_counterfactual_credit",
    ),
    ("A3", "broadcast"): ("four_center_modular",),
    ("A3", "local_only"): ("four_center_modular_local_credit",),
    ("A3", "counterfactual"): ("four_center_modular_counterfactual",),
    ("A4", "broadcast"): ("modular_full",),
    ("A4", "local_only"): ("local_credit_only",),
    ("A4", "counterfactual"): ("counterfactual_credit",),
}

REWARD_PROFILE_LADDER_CONFIDENCE_LEVEL = DEFAULT_CONFIDENCE_LEVEL

UNIFIED_LADDER_HIGH_PRIORITY_EXPERIMENTS: frozenset[str] = frozenset(
    {
        "complete_architectural_ladder",
        "capacity_matched_ladder",
        "module_interface_sufficiency_suite",
        "no_reflex_competence_validation",
        "credit_assignment_variants",
    }
)
UNIFIED_LADDER_MEDIUM_PRIORITY_EXPERIMENTS: frozenset[str] = frozenset(
    {
        "cross_profile_ladder_runs",
        "five_seed_ladder_replication",
    }
)


def _credit_assignment_rung(variant_name: str) -> str | None:
    """
    Map a credit-assignment variant name to its ladder rung identifier.
    
    Parameters:
        variant_name (str): Variant identifier string (e.g., experiment variant key).
    
    Returns:
        str | None: `"A2"`, `"A3"`, or `"A4"` when `variant_name` matches a known rung; `None` if unknown.
    """
    if variant_name in {
        "three_center_modular",
        "three_center_modular_local_credit",
        "three_center_modular_counterfactual",
        "three_center_local_credit",
        "three_center_counterfactual_credit",
    }:
        return "A2"
    if variant_name in {
        "four_center_modular",
        "four_center_modular_local_credit",
        "four_center_modular_counterfactual",
    }:
        return "A3"
    if variant_name in {
        "modular_full",
        "local_credit_only",
        "counterfactual_credit",
    }:
        return "A4"
    return None


def _unified_ladder_priority(experiment_name: str) -> str:
    """
    Classifies an experiment name into a unified ladder priority tier.
    
    Returns:
        str: `'high'` if the experiment is listed as high priority, `'medium'` if listed as medium priority, `'low'` otherwise.
    """
    if experiment_name in UNIFIED_LADDER_HIGH_PRIORITY_EXPERIMENTS:
        return "high"
    if experiment_name in UNIFIED_LADDER_MEDIUM_PRIORITY_EXPERIMENTS:
        return "medium"
    return "low"


def _unified_ladder_interpretation(row: Mapping[str, object]) -> str:
    """
    Determine a human-readable interpretation label for a result row using delta, confidence interval, and effect-size information.
    
    Parameters:
        row (Mapping[str, object]): A mapping containing keys the function reads:
            - "delta": observed difference or change (numeric or coercible to float)
            - "cohens_d": Cohen's d effect size (numeric or coercible to float)
            - "ci_lower": lower bound of a confidence interval (numeric or coercible to float)
            - "ci_upper": upper bound of a confidence interval (numeric or coercible to float)
            - "effect_magnitude": optional string label (e.g., "small", "medium", "large")
    
    Returns:
        str: One of:
            - "insufficient data" if `delta` is missing or not coercible to a float.
            - "confidence interval crosses zero" if both `ci_lower` and `ci_upper` are present and the interval contains 0.
            - "effect size unavailable" if `cohens_d` is missing or not coercible to a float.
            - "{direction} {label} effect" where `direction` is "positive" when delta >= 0.0 else "negative", and `label` is the `effect_magnitude` value or "unlabeled" when that field is empty.
    """
    delta = _coerce_optional_float(row.get("delta"))
    effect_size = _coerce_optional_float(row.get("cohens_d"))
    ci_lower = _coerce_optional_float(row.get("ci_lower"))
    ci_upper = _coerce_optional_float(row.get("ci_upper"))
    magnitude = str(row.get("effect_magnitude") or "")
    if delta is None:
        return "insufficient data"
    if (
        ci_lower is not None
        and ci_upper is not None
        and ci_lower <= 0.0 <= ci_upper
    ):
        return "confidence interval crosses zero"
    if effect_size is None:
        return "effect size unavailable"
    direction = "positive" if delta >= 0.0 else "negative"
    label = magnitude or "unlabeled"
    return f"{direction} {label} effect"

__all__ = [name for name in globals() if not name.startswith("__")]
