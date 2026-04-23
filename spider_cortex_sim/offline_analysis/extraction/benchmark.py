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
from .shaping import _variant_with_minimal_reflex_support

CAPACITY_MATCH_RATIO_THRESHOLD = 1.2
PARAMETER_COUNT_METADATA_KEYS = frozenset(
    {
        "architecture",
        "by_network",
        "per_network",
        "proportions",
        "total",
        "total_trainable",
    }
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


def _parameter_counts_payload(value: object) -> dict[str, object]:
    payload = _mapping_or_empty(value)
    raw_by_network = _mapping_or_empty(payload.get("per_network"))
    if not raw_by_network:
        raw_by_network = _mapping_or_empty(payload.get("by_network"))
    by_network = {
        str(name): int(_coerce_float(count, 0.0))
        for name, count in raw_by_network.items()
    }
    if not by_network and payload and all(
        not isinstance(item, Mapping) for item in payload.values()
    ):
        by_network = {
            str(name): int(_coerce_float(count, 0.0))
            for name, count in payload.items()
            if str(name) not in PARAMETER_COUNT_METADATA_KEYS
        }
    total_trainable = int(_coerce_float(payload.get("total"), 0.0))
    if total_trainable <= 0:
        total_trainable = int(_coerce_float(payload.get("total_trainable"), 0.0))
    if total_trainable <= 0 and by_network:
        total_trainable = int(sum(by_network.values()))
    raw_proportions = _mapping_or_empty(payload.get("proportions"))
    proportions = {
        str(name): _coerce_float(value)
        for name, value in raw_proportions.items()
        if str(name) in by_network
    }
    if not proportions and total_trainable > 0:
        proportions = {
            name: round(count / total_trainable, 6)
            for name, count in by_network.items()
        }
    return {
        "architecture": str(payload.get("architecture") or ""),
        "by_network": by_network,
        "total_trainable": total_trainable,
        "proportions": proportions,
    }


def compare_capacity_totals(
    total_parameters_by_variant: Mapping[str, object],
    *,
    match_ratio_threshold: float = CAPACITY_MATCH_RATIO_THRESHOLD,
) -> dict[str, object]:
    """
    Compare architecture capacities using the ratio between the largest and smallest totals.

    Parameters:
        total_parameters_by_variant (Mapping[str, object]): Mapping from
            architecture/variant name to total trainable parameter count.
        match_ratio_threshold (float): Largest/smallest ratio considered a
            capacity match. The default allows up to a 20% spread.

    Returns:
        dict[str, object]: Capacity comparison result with `capacity_matched`,
        `ratio`, and `status`, plus supporting metadata describing the largest
        and smallest variants used in the comparison.
    """
    normalized = {
        str(name): int(_coerce_float(total, 0.0))
        for name, total in total_parameters_by_variant.items()
        if _coerce_float(total, 0.0) > 0.0
    }
    if not normalized:
        return {
            "available": False,
            "capacity_matched": False,
            "ratio": 0.0,
            "status": "unavailable",
            "largest_variant": "",
            "smallest_variant": "",
            "largest_total": 0,
            "smallest_total": 0,
            "match_ratio_threshold": float(match_ratio_threshold),
            "variant_count": 0,
        }
    if len(normalized) == 1:
        only_variant, only_total = next(iter(normalized.items()))
        return {
            "available": True,
            "capacity_matched": True,
            "ratio": 1.0,
            "status": "single architecture",
            "largest_variant": only_variant,
            "smallest_variant": only_variant,
            "largest_total": only_total,
            "smallest_total": only_total,
            "match_ratio_threshold": float(match_ratio_threshold),
            "variant_count": 1,
        }

    smallest_variant, smallest_total = min(
        normalized.items(),
        key=lambda item: (item[1], item[0]),
    )
    largest_variant, largest_total = max(
        normalized.items(),
        key=lambda item: (item[1], item[0]),
    )
    ratio = round(_safe_divide(largest_total, smallest_total), 6)
    capacity_matched = bool(ratio <= float(match_ratio_threshold))
    status = (
        "matched"
        if capacity_matched
        else f"{largest_variant} {ratio:.1f}x larger"
    )
    return {
        "available": True,
        "capacity_matched": capacity_matched,
        "ratio": ratio,
        "status": status,
        "largest_variant": largest_variant,
        "smallest_variant": smallest_variant,
        "largest_total": largest_total,
        "smallest_total": smallest_total,
        "match_ratio_threshold": float(match_ratio_threshold),
        "variant_count": len(normalized),
    }


def extract_model_capacity(summary: Mapping[str, object]) -> dict[str, object]:
    """
    Extract the current run's trainable-parameter breakdown from the summary.

    Returns:
        dict[str, object]: A mapping with `available`, run metadata, ordered
        per-network rows, and limitations when parameter-count data is absent.
    """
    config = _mapping_or_empty(summary.get("config"))
    brain = _mapping_or_empty(config.get("brain"))
    parameter_counts = _parameter_counts_payload(summary.get("parameter_counts"))
    by_network = parameter_counts["by_network"]
    total_trainable = int(parameter_counts["total_trainable"])
    if not by_network:
        capacity_sweeps = extract_capacity_sweeps(summary)
        capacity_rows = (
            capacity_sweeps.get("rows", [])
            if isinstance(capacity_sweeps.get("rows"), list)
            else []
        )
        preferred_variant = str(brain.get("name") or "")
        selected_row: Mapping[str, object] | None = None
        for row in capacity_rows:
            if not isinstance(row, Mapping):
                continue
            if (
                str(row.get("variant") or "") == preferred_variant
                and str(row.get("capacity_profile") or "") == "current"
            ):
                selected_row = row
                break
        if selected_row is None:
            for row in capacity_rows:
                if not isinstance(row, Mapping):
                    continue
                if str(row.get("capacity_profile") or "") == "current":
                    selected_row = row
                    break
        if selected_row is None:
            for row in capacity_rows:
                if not isinstance(row, Mapping):
                    continue
                if str(row.get("variant") or "") == preferred_variant:
                    selected_row = row
                    break
        if selected_row is None:
            for row in capacity_rows:
                if isinstance(row, Mapping):
                    selected_row = row
                    break
        if selected_row is not None:
            by_network = {
                str(name): int(_coerce_float(count, 0.0))
                for name, count in _mapping_or_empty(
                    selected_row.get("parameter_counts_by_network")
                ).items()
            }
            total_trainable = int(
                _coerce_float(selected_row.get("total_trainable"), 0.0)
            )
            proportions = (
                {
                    name: round(count / total_trainable, 6)
                    for name, count in by_network.items()
                }
                if total_trainable > 0
                else {}
            )
            sorted_networks = sorted(
                by_network.items(),
                key=lambda item: (-item[1], item[0]),
            )
            return {
                "available": bool(by_network),
                "variant": str(selected_row.get("variant") or preferred_variant),
                "architecture": str(
                    selected_row.get("architecture")
                    or parameter_counts["architecture"]
                    or brain.get("architecture")
                    or ""
                ),
                "total_trainable": total_trainable,
                "networks": [
                    {
                        "network": name,
                        "parameters": count,
                        "proportion": _coerce_float(proportions.get(name)),
                    }
                    for name, count in sorted_networks
                ],
                "top_components": [
                    {
                        "network": name,
                        "parameters": count,
                        "proportion": _coerce_float(proportions.get(name)),
                    }
                    for name, count in sorted_networks[:3]
                ],
                "source": str(selected_row.get("source") or ""),
                "limitations": [],
            }
    if not by_network:
        return {
            "available": False,
            "variant": str(brain.get("name") or ""),
            "architecture": str(
                parameter_counts["architecture"] or brain.get("architecture") or ""
            ),
            "total_trainable": total_trainable,
            "networks": [],
            "source": "summary.parameter_counts",
            "limitations": ["No summary parameter-count payload was available."],
        }
    proportions = parameter_counts["proportions"]
    sorted_networks = sorted(
        by_network.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "available": True,
        "variant": str(brain.get("name") or ""),
        "architecture": str(
            parameter_counts["architecture"] or brain.get("architecture") or ""
        ),
        "total_trainable": total_trainable,
        "networks": [
            {
                "network": name,
                "parameters": count,
                "proportion": _coerce_float(proportions.get(name)),
            }
            for name, count in sorted_networks
        ],
        "top_components": [
            {
                "network": name,
                "parameters": count,
                "proportion": _coerce_float(proportions.get(name)),
            }
            for name, count in sorted_networks[:3]
        ],
        "source": "summary.parameter_counts",
        "limitations": [],
    }


def extract_architecture_capacity(summary: Mapping[str, object]) -> dict[str, object]:
    """
    Extract architecture-capacity rows from a summary or ablation payload.

    Returns:
        dict[str, object]: A mapping with `available`, `reference_variant`,
        ordered `rows`, and limitation notes when capacity data is missing.
    """
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    ablations = _mapping_or_empty(behavior_evaluation.get("ablations"))
    variants = _mapping_or_empty(ablations.get("variants"))
    if variants:
        reference_variant = str(ablations.get("reference_variant") or "")
        if not reference_variant and "modular_full" in variants:
            reference_variant = "modular_full"
        rows: list[dict[str, object]] = []
        totals_by_variant: dict[str, int] = {}
        invalid_variants: list[str] = []
        for variant_name, payload in variants.items():
            if not isinstance(payload, Mapping):
                invalid_variants.append(
                    f"Variant {variant_name} had a non-mapping payload."
                )
                continue
            counts = _parameter_counts_payload(payload.get("parameter_counts"))
            if not counts["by_network"]:
                invalid_variants.append(
                    f"Variant {variant_name} was missing parameter counts."
                )
                continue
            config_payload = _mapping_or_empty(payload.get("config"))
            sorted_components = sorted(
                counts["by_network"].items(),
                key=lambda item: (-item[1], item[0]),
            )
            totals_by_variant[str(variant_name)] = int(counts["total_trainable"])
            rows.append(
                {
                    "variant": str(variant_name),
                    "architecture": str(
                        counts["architecture"]
                        or config_payload.get("architecture")
                        or ""
                    ),
                    "total_trainable": int(counts["total_trainable"]),
                    "key_components": ", ".join(
                        f"{name}={count}"
                        for name, count in sorted_components[:3]
                    ),
                    "source": (
                        "summary.behavior_evaluation.ablations."
                        f"variants.{variant_name}.parameter_counts"
                    ),
                }
            )
        if invalid_variants:
            return {
                "available": False,
                "reference_variant": reference_variant,
                "capacity_analysis": compare_capacity_totals({}),
                "rows": [],
                "limitations": invalid_variants,
            }
        if not rows:
            return {
                "available": False,
                "reference_variant": reference_variant,
                "capacity_analysis": compare_capacity_totals({}),
                "rows": [],
                "limitations": [
                    "Ablation variants were present but did not include parameter counts."
                ],
            }
        row_by_variant = {str(row["variant"]): row for row in rows}
        if reference_variant not in row_by_variant:
            reference_variant = str(rows[0]["variant"])
        reference_total = int(
            _coerce_float(row_by_variant[reference_variant].get("total_trainable"), 0.0)
        )
        capacity_analysis = compare_capacity_totals(totals_by_variant)
        ordered_variants = [
            reference_variant,
            *sorted(
                variant
                for variant in row_by_variant
                if variant != reference_variant
            ),
        ]
        ordered_rows = []
        for variant_name in ordered_variants:
            row = dict(row_by_variant[variant_name])
            total_trainable = int(
                _coerce_float(row.get("total_trainable"), 0.0)
            )
            if variant_name == reference_variant:
                row["capacity_matched"] = True
                row["capacity_status"] = "reference"
                row["total_ratio_vs_reference"] = 1.0
            else:
                matched = bool(
                    round(_safe_divide(max(total_trainable, reference_total), min(total_trainable, reference_total)), 6)
                    <= float(capacity_analysis["match_ratio_threshold"])
                )
                row["capacity_matched"] = matched
                row["capacity_status"] = "matched" if matched else "not matched"
                row["total_ratio_vs_reference"] = round(
                    _safe_divide(total_trainable, reference_total),
                    6,
                )
            row["reference_variant"] = reference_variant
            row["reference_total_trainable"] = reference_total
            ordered_rows.append(row)
        return {
            "available": True,
            "reference_variant": reference_variant,
            "capacity_analysis": capacity_analysis,
            "rows": ordered_rows,
            "limitations": [],
        }

    model_capacity = extract_model_capacity(summary)
    if not model_capacity["available"]:
        return {
            "available": False,
            "reference_variant": "",
            "capacity_analysis": compare_capacity_totals({}),
            "rows": [],
            "limitations": list(model_capacity.get("limitations", [])),
        }
    capacity_analysis = compare_capacity_totals(
        {
            str(model_capacity.get("variant") or ""): int(
                _coerce_float(model_capacity.get("total_trainable"), 0.0)
            )
        }
    )
    return {
        "available": True,
        "reference_variant": str(model_capacity.get("variant") or ""),
        "capacity_analysis": capacity_analysis,
        "rows": [
            {
                "variant": str(model_capacity.get("variant") or ""),
                "architecture": str(model_capacity.get("architecture") or ""),
                "total_trainable": int(
                    _coerce_float(model_capacity.get("total_trainable"), 0.0)
                ),
                "key_components": ", ".join(
                    f"{item.get('network')}={int(_coerce_float(item.get('parameters'), 0.0))}"
                    for item in model_capacity.get("top_components", [])
                    if isinstance(item, Mapping)
                ),
                "capacity_matched": True,
                "capacity_status": "reference",
                "reference_variant": str(model_capacity.get("variant") or ""),
                "reference_total_trainable": int(
                    _coerce_float(model_capacity.get("total_trainable"), 0.0)
                ),
                "total_ratio_vs_reference": 1.0,
                "source": str(model_capacity.get("source") or ""),
            }
        ],
        "limitations": [],
    }


def _capacity_probe_success_rate(
    suite: Mapping[str, object],
) -> float | None:
    probe_rates = [
        _coerce_float(item.get("success_rate"))
        for item in suite.values()
        if isinstance(item, Mapping) and bool(item.get("is_capability_probe"))
    ]
    if not probe_rates:
        return None
    return round(_mean(probe_rates), 6)


def extract_capacity_sweeps(summary: Mapping[str, object]) -> dict[str, object]:
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    capacity_sweeps = _mapping_or_empty(behavior_evaluation.get("capacity_sweeps"))
    profiles = _mapping_or_empty(capacity_sweeps.get("profiles"))
    if not profiles:
        return {
            "available": False,
            "profiles": [],
            "rows": [],
            "interpretations": [],
            "limitations": ["No capacity-sweep payload was available."],
            "source": "summary.behavior_evaluation.capacity_sweeps",
        }

    rows: list[dict[str, object]] = []
    by_variant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for profile_name, payload in profiles.items():
        if not isinstance(payload, Mapping):
            continue
        profile_summary = _mapping_or_empty(payload.get("capacity_profile"))
        profile_scale_factor = _coerce_float(
            profile_summary.get("scale_factor"),
            0.0,
        )
        variants = _mapping_or_empty(payload.get("variants"))
        for variant_name, variant_payload in variants.items():
            if not isinstance(variant_payload, Mapping):
                continue
            minimal_payload = _variant_with_minimal_reflex_support(variant_payload)
            summary_payload = _mapping_or_empty(minimal_payload.get("summary"))
            suite_payload = _mapping_or_empty(minimal_payload.get("suite"))
            counts = _parameter_counts_payload(minimal_payload.get("parameter_counts"))
            compute_cost = _mapping_or_empty(
                minimal_payload.get("approximate_compute_cost")
            )
            row = {
                "variant": str(variant_name),
                "architecture": str(
                    _mapping_or_empty(minimal_payload.get("config")).get("architecture")
                    or ""
                ),
                "capacity_profile": str(profile_name),
                "capacity_profile_version": str(profile_summary.get("version") or ""),
                "scale_factor": profile_scale_factor,
                "scenario_success_rate": round(
                    _coerce_float(summary_payload.get("scenario_success_rate")),
                    6,
                ),
                "episode_success_rate": round(
                    _coerce_float(summary_payload.get("episode_success_rate")),
                    6,
                ),
                "capability_probe_success_rate": _capacity_probe_success_rate(
                    suite_payload
                ),
                "total_trainable": int(counts["total_trainable"]),
                "parameter_counts_by_network": dict(counts["by_network"]),
                "approximate_compute_cost_total": int(
                    _coerce_float(compute_cost.get("total"), 0.0)
                ),
                "approximate_compute_cost_unit": str(
                    compute_cost.get("unit") or ""
                ),
                "approximate_compute_cost_by_network": dict(
                    _mapping_or_empty(compute_cost.get("per_network"))
                ),
                "source": (
                    "summary.behavior_evaluation.capacity_sweeps."
                    f"profiles.{profile_name}.variants.{variant_name}"
                ),
            }
            rows.append(row)
            by_variant[str(variant_name)].append(row)

    interpretations: list[dict[str, object]] = []
    for variant_name, variant_rows in sorted(by_variant.items()):
        ordered = sorted(
            variant_rows,
            key=lambda item: (
                _coerce_float(item.get("scale_factor"), 0.0),
                str(item.get("capacity_profile") or ""),
            ),
        )
        if len(ordered) < 2:
            status = "insufficient_data"
            interpretation = "Insufficient capacity sweep data."
        else:
            first = _coerce_float(ordered[0].get("scenario_success_rate"), 0.0)
            best = max(
                _coerce_float(item.get("scenario_success_rate"), 0.0)
                for item in ordered
            )
            improvement = round(best - first, 6)
            if improvement > 0.05:
                status = "improves_with_capacity"
                interpretation = (
                    "Performance improves with capacity; possible undercapacity."
                )
            else:
                status = "flat_with_capacity"
                interpretation = (
                    "Performance does not materially improve with capacity; "
                    "interface, connection, or reward issues are more likely."
                )
        interpretations.append(
            {
                "variant": variant_name,
                "status": status,
                "interpretation": interpretation,
            }
        )

    return {
        "available": bool(rows),
        "profiles": sorted(
            {
                str(item.get("capacity_profile") or "")
                for item in rows
                if str(item.get("capacity_profile") or "")
            }
        ),
        "rows": rows,
        "interpretations": interpretations,
        "limitations": []
        if rows
        else ["Capacity-sweep payload was present but no usable rows were found."],
        "source": "summary.behavior_evaluation.capacity_sweeps",
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
