from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

from ...ablations import compare_predator_type_ablation_performance
from ..constants import (
    CANONICAL_NOISE_CONDITIONS,
    DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    DEFAULT_MODULE_NAMES,
    REFLEX_DOMINANCE_WARNING_THRESHOLD,
    REFLEX_OVERRIDE_WARNING_THRESHOLD,
    SHAPING_DEPENDENCE_WARNING_THRESHOLD,
)
from ..uncertainty import _payload_has_zero_reflex_scale, _payload_uncertainty
from ..utils import (
    _coerce_bool,
    _coerce_float,
    _coerce_optional_float,
    _dominant_module_by_score,
    _mapping_or_empty,
    _mean,
    _safe_divide,
)

def build_primary_benchmark(
    summary: Mapping[str, object],
    scenario_success: Mapping[str, object],
    ablations: Mapping[str, object],
) -> dict[str, object]:
    """
    Selects a primary "no-reflex" scenario success benchmark from available ablation, summary, or scenario data.
    
    Parameters:
    	summary (Mapping[str, object]): Parsed report summary (may contain evaluation blocks and fallbacks).
    	scenario_success (Mapping[str, object]): Scenario-level success data and metadata, used as a final fallback.
    	ablations (Mapping[str, object]): Ablation variants and metadata; preferred source for a no-reflex benchmark.
    
    Returns:
    	dict[str, object]: A payload describing the chosen primary benchmark with these keys:
    		available (bool): True when a benchmark value was found.
    		primary (bool): Always True for this payload.
    		label (str): Human-friendly label for the metric (always "No-reflex scenario_success_rate").
    		metric (str): Metric name ("scenario_success_rate").
    		scenario_success_rate (float): The benchmark score (0.0 when unavailable).
    		eval_reflex_scale (float|None): The eval reflex scale associated with the benchmark, or None when unknown.
    		reference_variant (str): Reference ablation variant name when sourced from ablations, else empty string.
    		source (str): Dot-path or descriptor indicating where the benchmark was sourced.
    		interpretation (str): Brief guidance about how to interpret the metric.
    """
    metric_name = "scenario_success_rate"
    label = "No-reflex scenario_success_rate"
    variants = ablations.get("variants", {})
    reference_variant = str(ablations.get("reference_variant") or "")
    if isinstance(variants, Mapping) and variants:
        if not reference_variant and "modular_full" in variants:
            reference_variant = "modular_full"
        reference_payload = variants.get(reference_variant)
        if isinstance(reference_payload, Mapping):
            summary_payload = reference_payload.get("summary", {})
            if isinstance(summary_payload, Mapping) and metric_name in summary_payload:
                return {
                    "available": True,
                    "primary": True,
                    "label": label,
                    "metric": metric_name,
                    "scenario_success_rate": _coerce_float(summary_payload.get(metric_name)),
                    "uncertainty": dict(
                        _payload_uncertainty(reference_payload, metric_name)
                    ),
                    "eval_reflex_scale": _coerce_float(
                        summary_payload.get(
                            "eval_reflex_scale",
                            reference_payload.get("eval_reflex_scale"),
                        ),
                        0.0,
                    ),
                    "reference_variant": reference_variant,
                    "source": (
                        f"{ablations.get('source')}.variants.{reference_variant}.summary"
                    ),
                    "interpretation": "Higher is better; this benchmark is evaluated with reflex support disabled.",
                }

    no_reflex_eval = summary.get("evaluation_without_reflex_support", {})
    if isinstance(no_reflex_eval, Mapping):
        no_reflex_summary = no_reflex_eval.get("summary", {})
        if isinstance(no_reflex_summary, Mapping) and metric_name in no_reflex_summary:
            return {
                "available": True,
                "primary": True,
                "label": label,
                "metric": metric_name,
                "scenario_success_rate": _coerce_float(no_reflex_summary.get(metric_name)),
                "uncertainty": dict(_payload_uncertainty(no_reflex_eval, metric_name)),
                "eval_reflex_scale": _coerce_float(
                    no_reflex_summary.get(
                        "eval_reflex_scale",
                        no_reflex_eval.get("eval_reflex_scale"),
                    ),
                    0.0,
                ),
                "reference_variant": "",
                "source": "summary.evaluation_without_reflex_support.summary",
                "interpretation": "Higher is better; this benchmark is evaluated with reflex support disabled.",
            }

    behavior_evaluation = summary.get("behavior_evaluation", {})
    if isinstance(behavior_evaluation, Mapping):
        behavior_summary = behavior_evaluation.get("summary", {})
        if (
            isinstance(behavior_summary, Mapping)
            and metric_name in behavior_summary
            and _payload_has_zero_reflex_scale(behavior_evaluation)
        ):
            return {
                "available": True,
                "primary": True,
                "label": label,
                "metric": metric_name,
                "scenario_success_rate": _coerce_float(
                    behavior_summary.get(metric_name)
                ),
                "uncertainty": dict(
                    _payload_uncertainty(behavior_evaluation, metric_name)
                ),
                "eval_reflex_scale": _coerce_float(
                    behavior_summary.get(
                        "eval_reflex_scale",
                        behavior_evaluation.get("eval_reflex_scale"),
                    ),
                    0.0,
                ),
                "reference_variant": "",
                "source": "summary.behavior_evaluation.summary",
                "interpretation": "Higher is better; this benchmark is evaluated with reflex support disabled when eval_reflex_scale is 0.0.",
            }

    evaluation = summary.get("evaluation", {})
    if isinstance(evaluation, Mapping) and metric_name in evaluation:
        eval_reflex_scale_raw = evaluation.get("eval_reflex_scale")
        if eval_reflex_scale_raw is not None:
            eval_reflex_scale = _coerce_float(eval_reflex_scale_raw, math.nan)
            if math.isfinite(eval_reflex_scale) and abs(eval_reflex_scale) <= 1e-6:
                return {
                    "available": True,
                    "primary": True,
                    "label": label,
                    "metric": metric_name,
                    "scenario_success_rate": _coerce_float(evaluation.get(metric_name)),
                    "uncertainty": dict(_payload_uncertainty(evaluation, metric_name)),
                    "eval_reflex_scale": eval_reflex_scale,
                    "reference_variant": "",
                    "source": "summary.evaluation",
                    "interpretation": "Higher is better; this benchmark is evaluated with reflex support disabled when eval_reflex_scale is 0.0.",
                }

    scenarios = scenario_success.get("scenarios", [])
    if isinstance(scenarios, list) and scenarios:
        values = [
            _coerce_float(item.get("success_rate"))
            for item in scenarios
            if isinstance(item, Mapping)
        ]
        if values:
            return {
                "available": True,
                "primary": True,
                "label": label,
                "metric": metric_name,
                "scenario_success_rate": _mean(values),
                "uncertainty": {},
                "eval_reflex_scale": None,
                "reference_variant": "",
                "source": str(scenario_success.get("source") or ""),
                "interpretation": "Higher is better; eval_reflex_scale was unavailable for this fallback scenario aggregate.",
            }

    return {
        "available": False,
        "primary": True,
        "label": label,
        "metric": metric_name,
        "scenario_success_rate": 0.0,
        "uncertainty": {},
        "eval_reflex_scale": None,
        "reference_variant": "",
        "source": "none",
        "interpretation": "No no-reflex scenario_success_rate benchmark was available.",
    }

def build_reflex_dependence_indicators(
    summary: Mapping[str, object],
    reflex_frequency: Mapping[str, object],
) -> dict[str, object]:
    """
    Compute reflex override and dominance indicators and their warning statuses using data from an evaluation summary and optional trace-derived reflex frequency.
    
    Parameters:
        summary (Mapping[str, object]): Report summary that may contain
            `evaluation_with_reflex_support` (with a `summary` mapping) or
            `evaluation` entries that include `mean_final_reflex_override_rate`
            and/or `mean_reflex_dominance`.
        reflex_frequency (Mapping[str, object]): Trace-derived reflex frequency
            payload (typically from `extract_reflex_frequency`) that may include a
            `modules` list with per-module `override_rate` and `mean_dominance`.
    
    Returns:
        dict[str, object]: A mapping with keys:
          - `available` (bool): `True` if any reflex indicator was found.
          - `source` (str): Source summary/trace fields used (e.g. `"summary.evaluation"`, `"trace.reflex_frequency"`, or `"none"`).
          - `failure_indicators` (dict): Contains two indicator entries:
              - `override_rate`: mapping with `value` (float), `warning_threshold` (float), and `status` (`"warning"` or `"ok"`).
              - `dominance`: mapping with `value` (float), `warning_threshold` (float), and `status` (`"warning"` or `"ok"`).
    """
    source_parts: list[str] = []
    override_rate = 0.0
    dominance = 0.0
    available = False

    reflex_eval = summary.get("evaluation_with_reflex_support", {})
    reflex_summary = (
        reflex_eval.get("summary", {})
        if isinstance(reflex_eval, Mapping)
        else {}
    )
    if isinstance(reflex_summary, Mapping) and (
        "mean_final_reflex_override_rate" in reflex_summary
        or "mean_reflex_dominance" in reflex_summary
    ):
        override_rate = _coerce_float(
            reflex_summary.get("mean_final_reflex_override_rate"),
            0.0,
        )
        dominance = _coerce_float(reflex_summary.get("mean_reflex_dominance"), 0.0)
        available = True
        source_parts.append("summary.evaluation_with_reflex_support.summary")
    else:
        evaluation = summary.get("evaluation", {})
        if isinstance(evaluation, Mapping) and (
            "mean_final_reflex_override_rate" in evaluation
            or "mean_reflex_dominance" in evaluation
        ):
            override_rate = _coerce_float(
                evaluation.get("mean_final_reflex_override_rate"),
                0.0,
            )
            dominance = _coerce_float(evaluation.get("mean_reflex_dominance"), 0.0)
            available = True
            source_parts.append("summary.evaluation")

    modules = reflex_frequency.get("modules", [])
    trace_has_debug_reflexes = bool(reflex_frequency.get("uses_debug_reflexes"))
    if trace_has_debug_reflexes and isinstance(modules, list) and modules:
        module_override_rates = [
            _coerce_float(module.get("override_rate"))
            for module in modules
            if isinstance(module, Mapping)
        ]
        module_dominance = [
            _coerce_float(module.get("mean_dominance"))
            for module in modules
            if isinstance(module, Mapping)
        ]
        if module_override_rates:
            override_rate = max(override_rate, max(module_override_rates))
            available = True
        if module_dominance:
            dominance = max(dominance, max(module_dominance))
            available = True
        source_parts.append("trace.reflex_frequency")

    override_status = (
        "warning"
        if override_rate >= REFLEX_OVERRIDE_WARNING_THRESHOLD
        else "ok"
    )
    dominance_status = (
        "warning"
        if dominance >= REFLEX_DOMINANCE_WARNING_THRESHOLD
        else "ok"
    )
    return {
        "available": available,
        "source": "+".join(source_parts) if source_parts else "none",
        "failure_indicators": {
            "override_rate": {
                "value": override_rate,
                "warning_threshold": REFLEX_OVERRIDE_WARNING_THRESHOLD,
                "status": override_status,
            },
            "dominance": {
                "value": dominance,
                "warning_threshold": REFLEX_DOMINANCE_WARNING_THRESHOLD,
                "status": dominance_status,
            },
        },
    }

def _format_failure_indicator(
    value: float,
    threshold: float,
    status: str,
) -> str:
    """
    Format a numeric failure indicator into a concise human-readable string.
    
    Parameters:
        value (float): Numeric indicator value to format.
        threshold (float): Threshold used in the warning message.
        status (str): Status label (e.g., "ok" or "warning").
    
    Returns:
        str: Formatted string like "0.123456 (warning; warning >= 0.10)" where `value` is shown with six decimal places and `threshold` with two decimal places.
    """
    return f"{value:.6f} ({status}; warning >= {threshold:.2f})"

def _fallback_group_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    key_name: str,
) -> dict[str, object]:
    """
    Aggregate per-group success statistics and counts from behavior CSV rows.
    
    Groups input rows by the string value of the given key name (skipping rows with no key or empty string). For each group computes:
    - `scenario_success_rate`: mean of per-row `success` (coerced to boolean then 1.0/0.0)
    - `episode_success_rate`: same mean computed over episodes
    - `scenario_count`: number of distinct non-empty `scenario` values in the group
    - `episode_count`: total number of rows in the group
    
    Parameters:
        rows (Sequence[Mapping[str, object]]): Iterable of row mappings (e.g., csv.DictReader rows).
        key_name (str): Column/key name to group rows by; its string value is used as the group key.
    
    Returns:
        dict[str, object]: Mapping from each group key to a dict containing a `summary` mapping with
        `scenario_success_rate`, `episode_success_rate`, `scenario_count`, and `episode_count`.
    """
    by_key: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(key_name) or "")
        if key:
            by_key[key].append(row)
    summary: dict[str, object] = {}
    for key, items in sorted(by_key.items()):
        scenario_names = sorted({str(row.get("scenario") or "") for row in items if row.get("scenario")})
        per_scenario_success_rates = []
        for scenario_name in scenario_names:
            scenario_success_values = [
                1.0 if _coerce_bool(row.get("success")) else 0.0
                for row in items
                if str(row.get("scenario") or "") == scenario_name
            ]
            per_scenario_success_rates.append(_mean(scenario_success_values))
        success_values = [1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items]
        summary[key] = {
            "summary": {
                "scenario_success_rate": _mean(per_scenario_success_rates),
                "episode_success_rate": _mean(success_values),
                "scenario_count": len(scenario_names),
                "episode_count": len(items),
            }
        }
    return summary
