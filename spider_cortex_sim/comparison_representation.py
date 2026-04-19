"""Cross-run comparison and condensed reporting helpers.

``condense_robustness_summary`` is the public home for the CLI helper formerly
named ``_short_robustness_matrix_summary``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Dict

from .ablations import (
    compare_predator_type_ablation_performance,
    resolve_ablation_scenario_group,
)
from .benchmark_types import SeedLevelResult

from .comparison_utils import aggregate_with_uncertainty, fallback_seed_values, metric_seed_values_from_payload, paired_seed_delta_rows, predator_type_specialization_score, safe_float, seed_level_dicts

def representation_specialization_from_payload(
    payload: Dict[str, object] | None,
) -> Dict[str, object]:
    """
    Extract representation-specialization evidence from a payload.
    
    Prefer top-level aggregate metrics when present; otherwise, average legacy- and
    suite-level sources (from `legacy_scenarios` and `suite`) to produce per-module
    metrics and an overall specialization score.
    
    Parameters:
        payload (dict | None): Payload containing either top-level metrics or
            nested legacy/suite entries.
    
    Returns:
        dict: A normalized result with the following keys:
            - "available" (bool): True when any metric was found.
            - "proposer_divergence_by_module" (dict[str, float]): Per-module proposer
              divergence values rounded to 6 decimals.
            - "action_center_gate_differential" (dict[str, float]): Per-module gate
              differential values rounded to 6 decimals.
            - "action_center_contribution_differential" (dict[str, float]): Per-module
              contribution differential values rounded to 6 decimals.
            - "representation_specialization_score" (float): Aggregated specialization
              score rounded to 6 decimals (0.0 when absent).
    """

    def _mapping_metric(
        source: Dict[str, object],
        mean_key: str,
        raw_key: str,
    ) -> tuple[Dict[str, float], bool]:
        """
        Extracts and normalizes a mapping metric from a source dictionary.
        
        Looks up the metric under `mean_key` first and `raw_key` second. If a non-empty mapping is found, converts each value with `safe_float`, rounds to 6 decimal places, and coerces keys to strings.
        
        Parameters:
            source (Dict[str, object]): Dictionary to read metrics from.
            mean_key (str): Preferred key name for mean-derived metrics.
            raw_key (str): Fallback key name for raw metrics.
        
        Returns:
            tuple[Dict[str, float], bool]: A pair where the first element is the normalized mapping of metric name → float and the second element is `True` if a usable mapping was found, otherwise an empty mapping and `False`.
        """
        for key in (mean_key, raw_key):
            if key not in source:
                continue
            value = source.get(key)
            if value is None:
                continue
            if not isinstance(value, Mapping) or len(value) == 0:
                continue
            normalized = {
                str(name): round(safe_float(metric), 6)
                for name, metric in value.items()
            }
            if normalized:
                return normalized, True
        return {}, False

    def _scalar_metric(
        source: Dict[str, object],
        mean_key: str,
        raw_key: str,
    ) -> tuple[float, bool]:
        """
        Select a scalar metric from a source dict, preferring the mean key and falling back to the raw key.
        
        Parameters:
            source (Dict[str, object]): Mapping containing metric values.
            mean_key (str): Primary key to check (typically a mean/aggregated value).
            raw_key (str): Secondary key to check if the primary is absent or None.
        
        Returns:
            tuple[float, bool]: A pair where the first element is the metric converted to a float and rounded to 6 decimals (or 0.0 if not found), and the second element is `True` if a valid value was found, `False` otherwise.
        """
        for key in (mean_key, raw_key):
            if key in source:
                value = source.get(key)
                if value is None:
                    continue
                return round(safe_float(value), 6), True
        return 0.0, False

    def _from_source(source: Dict[str, object]) -> Dict[str, object]:
        """
        Extracts representation-specialization metrics from a single payload source.
        
        Parameters:
            source (Dict[str, object]): Mapping representing a payload or metrics source to read.
        
        Returns:
            Dict[str, object]: A dictionary with the following keys:
                - "available": True if any proposer/gate/contribution mapping or a specialization score was found, False otherwise.
                - "proposer_divergence_by_module": Mapping of proposer divergence per module (floats) or an empty mapping.
                - "action_center_gate_differential": Mapping of gate differential per module (floats) or an empty mapping.
                - "action_center_contribution_differential": Mapping of contribution differential per module (floats) or an empty mapping.
                - "representation_specialization_score": Scalar specialization score (float) or 0.0.
        """
        proposer_divergence, has_proposer = _mapping_metric(
            source,
            "mean_proposer_divergence_by_module",
            "proposer_divergence_by_module",
        )
        gate_differential, has_gate = _mapping_metric(
            source,
            "mean_action_center_gate_differential",
            "action_center_gate_differential",
        )
        contribution_differential, has_contribution = _mapping_metric(
            source,
            "mean_action_center_contribution_differential",
            "action_center_contribution_differential",
        )
        score, has_score = _scalar_metric(
            source,
            "mean_representation_specialization_score",
            "representation_specialization_score",
        )
        available = bool(
            has_proposer or has_gate or has_contribution or has_score
        )
        return {
            "available": available,
            "proposer_divergence_by_module": proposer_divergence,
            "action_center_gate_differential": gate_differential,
            "action_center_contribution_differential": (
                contribution_differential
            ),
            "representation_specialization_score": score,
        }

    if not isinstance(payload, dict):
        return {
            "available": False,
            "proposer_divergence_by_module": {},
            "action_center_gate_differential": {},
            "action_center_contribution_differential": {},
            "representation_specialization_score": 0.0,
        }

    top_level = _from_source(payload)
    if bool(top_level["available"]):
        return top_level

    sources: list[Dict[str, object]] = []
    legacy_scenarios = payload.get("legacy_scenarios", {})
    if isinstance(legacy_scenarios, dict):
        for source in legacy_scenarios.values():
            if isinstance(source, dict):
                sources.append(source)
    suite = payload.get("suite", {})
    if isinstance(suite, dict):
        for scenario_payload in suite.values():
            if not isinstance(scenario_payload, dict):
                continue
            legacy_metrics = scenario_payload.get("legacy_metrics")
            if isinstance(legacy_metrics, dict):
                sources.append(legacy_metrics)
            sources.append(scenario_payload)

    proposer_grouped: Dict[str, list[float]] = {}
    gate_grouped: Dict[str, list[float]] = {}
    contribution_grouped: Dict[str, list[float]] = {}
    scores: list[float] = []
    for source in sources:
        source_metrics = _from_source(source)
        if not bool(source_metrics["available"]):
            continue
        for name, value in dict(
            source_metrics["proposer_divergence_by_module"]
        ).items():
            proposer_grouped.setdefault(str(name), []).append(
                safe_float(value)
            )
        for name, value in dict(
            source_metrics["action_center_gate_differential"]
        ).items():
            gate_grouped.setdefault(str(name), []).append(safe_float(value))
        for name, value in dict(
            source_metrics["action_center_contribution_differential"]
        ).items():
            contribution_grouped.setdefault(str(name), []).append(
                safe_float(value)
            )
        if (
            (
                "mean_representation_specialization_score" in source
                and source.get("mean_representation_specialization_score") is not None
            )
            or (
                "representation_specialization_score" in source
                and source.get("representation_specialization_score") is not None
            )
        ):
            scores.append(
                safe_float(
                    source_metrics["representation_specialization_score"]
                )
            )

    if not proposer_grouped and not gate_grouped and not contribution_grouped and not scores:
        return top_level

    return {
        "available": True,
        "proposer_divergence_by_module": {
            name: round(sum(values) / len(values), 6)
            for name, values in sorted(proposer_grouped.items())
            if values
        },
        "action_center_gate_differential": {
            name: round(sum(values) / len(values), 6)
            for name, values in sorted(gate_grouped.items())
            if values
        },
        "action_center_contribution_differential": {
            name: round(sum(values) / len(values), 6)
            for name, values in sorted(contribution_grouped.items())
            if values
        },
        "representation_specialization_score": round(
            sum(scores) / len(scores),
            6,
        )
        if scores
        else 0.0,
    }

def visual_minus_olfactory_seed_rows(
    payload: Dict[str, object] | None,
    *,
    condition: str,
    metric_name: str = "visual_minus_olfactory_delta",
    fallback_value: object | None = None,
) -> list[SeedLevelResult]:
    """
    Compute per-seed deltas of visual minus olfactory scenario success rates.
    
    For each seed, averages success rates across scenarios classified as "visual" and "olfactory"
    and emits a SeedLevelResult with the difference (visual mean minus olfactory mean) rounded
    to six decimals. If payload is not a dict or no seeds contain both families, returns rows
    derived from fallback_seed_values(fallback_value).
    
    Parameters:
        payload (Dict[str, object] | None): Payload containing a "seed_level" list of metric rows.
        condition (str): Condition label to set on each returned SeedLevelResult.
        metric_name (str): Metric name to assign to returned SeedLevelResult entries.
        fallback_value (object | None): Value(s) passed to fallback_seed_values when no real rows exist.
    
    Returns:
        list[SeedLevelResult]: One entry per seed with both visual and olfactory data containing the
        averaged visual-minus-olfactory delta, or fallback rows when no computed rows are available.
    """
    if not isinstance(payload, dict):
        return [
            SeedLevelResult(
                metric_name=metric_name,
                seed=seed,
                value=value,
                condition=condition,
            )
            for seed, value in fallback_seed_values(fallback_value)
        ]
    visual_scenarios = set(resolve_ablation_scenario_group("visual_predator_scenarios"))
    olfactory_scenarios = set(
        resolve_ablation_scenario_group("olfactory_predator_scenarios")
    )
    by_seed: Dict[int, Dict[str, list[float]]] = {}
    for item in payload.get("seed_level", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("metric_name", "")) != "scenario_success_rate":
            continue
        scenario = item.get("scenario")
        if scenario is None:
            continue
        scenario_name = str(scenario)
        buckets: list[str] = []
        if scenario_name in visual_scenarios:
            buckets.append("visual")
        if scenario_name in olfactory_scenarios:
            buckets.append("olfactory")
        if not buckets:
            continue
        seed_raw = item.get("seed")
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            for bucket in buckets:
                by_seed.setdefault(seed, {}).setdefault(
                    bucket,
                    [],
                ).append(value)
    rows = []
    for seed, buckets in sorted(by_seed.items()):
        visual_values = buckets.get("visual", [])
        olfactory_values = buckets.get("olfactory", [])
        if not visual_values or not olfactory_values:
            continue
        rows.append(
            SeedLevelResult(
                metric_name=metric_name,
                seed=seed,
                value=round(
                    (sum(visual_values) / len(visual_values))
                    - (sum(olfactory_values) / len(olfactory_values)),
                    6,
                ),
                condition=condition,
            )
        )
    if rows:
        return rows
    return [
        SeedLevelResult(
            metric_name=metric_name,
            seed=seed,
            value=value,
            condition=condition,
        )
        for seed, value in fallback_seed_values(fallback_value)
    ]

def build_predator_type_specialization_summary(
    variants: Dict[str, Dict[str, object]],
    *,
    reference_variant: str,
    deltas_vs_reference: Dict[str, object],
) -> Dict[str, object]:
    """
    Compose a predator-type specialization report combining ablation comparisons, per-seed visual-minus-olfactory rows, specialization-score rows, paired deltas against a reference, and per-variant uncertainty aggregates.
    
    Parameters:
    	variants (Dict[str, Dict[str, object]]): Mapping from variant name to its payload dictionary.
    	reference_variant (str): Variant name used as the baseline for paired delta calculations.
    	deltas_vs_reference (Dict[str, object]): Precomputed deltas against the reference used to contextualize ablation comparisons.
    
    Returns:
    	summary (Dict[str, object]): A dictionary merging the ablation comparison summary with:
    		- "reference_variant": the provided reference_variant.
    		- "seed_level": list of seed-level result dictionaries combining visual-minus-olfactory rows, specialization_score rows, and paired delta rows.
    		- "uncertainty": mapping from variant name to aggregated uncertainty objects for
    		  "visual_minus_olfactory_success_rate", "visual_minus_olfactory_success_rate_delta", and "specialization_score".
    """
    comparison_summary = compare_predator_type_ablation_performance(
        {
            "variants": variants,
            "deltas_vs_reference": deltas_vs_reference,
        },
        variant_names=tuple(variants),
    )
    comparisons = comparison_summary.get("comparisons", {})
    uncertainty: Dict[str, object] = {}
    seed_level: list[Dict[str, object]] = []
    reference_payload = variants.get(reference_variant)
    reference_rows = visual_minus_olfactory_seed_rows(
        reference_payload,
        condition=reference_variant,
        metric_name="visual_minus_olfactory_success_rate",
        fallback_value=(
            dict(comparisons.get(reference_variant, {})).get(
                "visual_minus_olfactory_success_rate"
            )
            if isinstance(comparisons, dict)
            else None
        ),
    )
    for variant_name, payload in variants.items():
        comparison_payload = (
            dict(comparisons.get(variant_name, {}))
            if isinstance(comparisons, dict)
            and isinstance(comparisons.get(variant_name), dict)
            else {}
        )
        visual_rows = visual_minus_olfactory_seed_rows(
            payload,
            condition=variant_name,
            metric_name="visual_minus_olfactory_success_rate",
            fallback_value=comparison_payload.get(
                "visual_minus_olfactory_success_rate"
            ),
        )
        specialization_seed_values = metric_seed_values_from_payload(
            payload,
            metric_name="specialization_score",
            fallback_value=predator_type_specialization_score(payload),
        )
        if not specialization_seed_values:
            specialization_seed_values = [
                (0, predator_type_specialization_score(payload))
            ]
        specialization_rows = [
            SeedLevelResult(
                metric_name="specialization_score",
                seed=seed,
                value=value,
                condition=variant_name,
            )
            for seed, value in specialization_seed_values
        ]
        delta_rows = paired_seed_delta_rows(
            [(row.seed, row.value) for row in reference_rows],
            [(row.seed, row.value) for row in visual_rows],
            metric_name="visual_minus_olfactory_success_rate_delta",
            condition=variant_name,
            fallback_delta=comparison_payload.get(
                "visual_minus_olfactory_success_rate_delta"
            ),
        )
        seed_level.extend(
            seed_level_dicts([*visual_rows, *specialization_rows, *delta_rows])
        )
        uncertainty[variant_name] = {
            "visual_minus_olfactory_success_rate": (
                aggregate_with_uncertainty(visual_rows)
            ),
            "visual_minus_olfactory_success_rate_delta": (
                aggregate_with_uncertainty(delta_rows)
            ),
            "specialization_score": aggregate_with_uncertainty(
                specialization_rows
            ),
        }
    return {
        **comparison_summary,
        "reference_variant": reference_variant,
        "seed_level": seed_level,
        "uncertainty": uncertainty,
    }
