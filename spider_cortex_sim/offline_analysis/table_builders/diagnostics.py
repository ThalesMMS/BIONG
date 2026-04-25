from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

from .common import (
    _coerce_float,
    _coerce_optional_float,
    _format_failure_indicator,
    _mapping_or_empty,
    _variant_with_minimal_reflex_support,
    build_reflex_dependence_indicators,
    compare_capacity_totals,
)

def _numeric_metric(item: Mapping[str, object], key: str) -> float | None:
    """
    Extract and coerce a numeric metric from a mapping by key.
    
    Parameters:
        item (Mapping[str, object]): Mapping containing the metric.
        key (str): Key whose value should be coerced to a float.
    
    Returns:
        float: The coerced float value of item[key], or `None` if the key is missing or the value is `None`.
    """
    if key not in item or item.get(key) is None:
        return None
    metric = _coerce_optional_float(item.get(key))
    if metric is None or not math.isfinite(metric):
        return None
    return metric


def build_reward_component_rows(
    summary: Mapping[str, object],
    scenario_success: Mapping[str, object],
    trace: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """
    Builds rows describing reward-component mean values aggregated from summary blocks, per-scenario legacy metrics, and trace totals.
    
    Parameters:
        summary (Mapping[str, object]): Top-level payload containing optional `training`, `training_last_window`, and `evaluation` blocks; each block may include `mean_reward_components` (a mapping of component -> numeric value).
        scenario_success (Mapping[str, object]): Payload containing a `scenarios` list; each scenario may contain `legacy_metrics.mean_reward_components`.
        trace (Sequence[Mapping[str, object]]): Sequence of trace items; each item may include `reward_components` (a mapping of component -> numeric value) which will be summed across the trace.
    
    Returns:
        list[dict[str, object]]: A list of rows where each row has keys:
            - "source": origin of the component (e.g., "summary", "behavior_evaluation.suite", or "trace")
            - "scope": scope identifier (e.g., "training", "evaluation", "scenario:<id>", or "trace_total")
            - "component": component name as a string
            - "value": numeric component value coerced to a float and rounded to 6 decimal places
    
    Notes:
        - Only mapping-valued `mean_reward_components` / `reward_components` are processed; non-mapping entries are ignored.
        - Trace component values are aggregated (summed) before being added as a "trace_total" row set.
    """
    rows: list[dict[str, object]] = []

    def add_components(source: str, scope: str, components: object) -> None:
        """
        Append a row for each entry in `components` to the surrounding `rows` collection.
        
        Parameters:
            source (str): Label identifying the origin of these components (for example, "training", "evaluation", or a scenario path).
            scope (str): Scope identifier applied to each appended row (for example, "trace" or "scenario:<id>").
            components (Mapping|object): Mapping of component name to numeric value; if not a mapping, no rows are appended.
        
        Notes:
            For each component entry, a row with keys `source`, `scope`, `component`, and `value` is appended. The `component` is converted to a string and `value` is coerced to a float and rounded to six decimal places.
        """
        if not isinstance(components, Mapping):
            return
        for component, value in sorted(components.items()):
            coerced_value = _coerce_optional_float(value)
            if coerced_value is None or not math.isfinite(coerced_value):
                continue
            rows.append(
                {
                    "source": source,
                    "scope": scope,
                    "component": str(component),
                    "value": round(coerced_value, 6),
                }
            )

    for aggregate_name in ("training", "training_last_window", "evaluation"):
        aggregate = summary.get(aggregate_name, {})
        if isinstance(aggregate, Mapping):
            add_components(
                "summary",
                aggregate_name,
                aggregate.get("mean_reward_components"),
            )

    scenarios_list = scenario_success.get("scenarios")
    for scenario in (scenarios_list if isinstance(scenarios_list, list) else []):
        if not isinstance(scenario, Mapping):
            continue
        legacy_metrics = scenario.get("legacy_metrics")
        if isinstance(legacy_metrics, Mapping):
            add_components(
                "behavior_evaluation.suite",
                f"scenario:{scenario.get('scenario')}",
                legacy_metrics.get("mean_reward_components"),
            )

    trace_totals: defaultdict[str, float] = defaultdict(float)
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        components = item.get("reward_components")
        if not isinstance(components, Mapping):
            continue
        for component, value in components.items():
            coerced_value = _coerce_optional_float(value)
            if coerced_value is None or not math.isfinite(coerced_value):
                continue
            trace_totals[str(component)] += coerced_value
    if trace_totals:
        add_components("trace", "trace_total", trace_totals)
    return rows


def build_diagnostics(
    summary: Mapping[str, object],
    scenario_success: Mapping[str, object],
    ablations: Mapping[str, object],
    reflex_frequency: Mapping[str, object],
) -> list[dict[str, object]]:
    """
    Builds a list of diagnostic metric entries summarizing evaluation metrics, reflex-dependence indicators, weakest scenario, best ablation variant, and most frequent reflex source.
    
    Parameters:
        summary (Mapping[str, object]): Parsed evaluation/training summary data used to extract aggregate evaluation metrics and evaluation-derived reflex indicators.
        scenario_success (Mapping[str, object]): Normalized scenario success payload containing a `scenarios` list used to identify the weakest scenario.
        ablations (Mapping[str, object]): Ablation variants payload used to rank variants by minimal-reflex `scenario_success_rate`.
        reflex_frequency (Mapping[str, object]): Trace-derived reflex frequency payload containing `modules` used to identify the most frequent reflex source.
    
    Returns:
        list[dict[str, object]]: A list of diagnostic rows. Each entry contains at least:
            - `label` (str): Human-readable metric name.
            - `value` (str|number): Display value for the metric.
          Failure-indicator entries (for reflex dependence) additionally include:
            - `status` (str), `warning_threshold` (float), `failure_indicator` (bool), and `source` (str).
    """
    diagnostics: list[dict[str, object]] = []
    evaluation = summary.get("evaluation", {})
    if isinstance(evaluation, Mapping):
        diagnostics.extend(
            [
                {
                    "label": "Evaluation mean reward",
                    "value": round(_coerce_float(evaluation.get("mean_reward")), 6),
                },
                {
                    "label": "Evaluation mean food distance delta",
                    "value": round(
                        _coerce_float(evaluation.get("mean_food_distance_delta")),
                        6,
                    ),
                },
                {
                    "label": "Evaluation mean shelter distance delta",
                    "value": round(
                        _coerce_float(evaluation.get("mean_shelter_distance_delta")),
                        6,
                    ),
                },
                {
                    "label": "Evaluation predator mode transitions",
                    "value": round(
                        _coerce_float(evaluation.get("mean_predator_mode_transitions")),
                        6,
                    ),
                },
                {
                    "label": "Evaluation dominant predator state",
                    "value": str(evaluation.get("dominant_predator_state") or ""),
                },
            ]
        )
    parameter_counts = _mapping_or_empty(summary.get("parameter_counts"))
    total_trainable = int(
        _coerce_float(
            parameter_counts.get("total"),
            _coerce_float(parameter_counts.get("total_trainable"), 0.0),
        )
    )
    if total_trainable > 0:
        diagnostics.append(
            {
                "label": "Trainable parameters",
                "value": total_trainable,
            }
        )
    per_network = _mapping_or_empty(parameter_counts.get("per_network"))
    if not per_network:
        per_network = _mapping_or_empty(parameter_counts.get("by_network"))
    sorted_components = sorted(
        (
            (str(name), int(coerced_count))
            for name, count in per_network.items()
            if (coerced_count := _coerce_float(count, 0.0)) > 0.0
        ),
        key=lambda item: (-item[1], item[0]),
    )
    for component_name, component_count in sorted_components[:3]:
        proportion = (
            0.0
            if total_trainable <= 0
            else float(component_count / total_trainable)
        )
        diagnostics.append(
            {
                "label": f"Parameters: {component_name}",
                "value": f"{component_count} ({proportion:.2%})",
            }
        )

    reflex_dependence = build_reflex_dependence_indicators(summary, reflex_frequency)
    if reflex_dependence.get("available"):
        indicators = reflex_dependence.get("failure_indicators", {})

        def append_reflex_indicator(
            label: str,
            indicator: Mapping[str, object],
        ) -> None:
            status = str(indicator.get("status") or "ok")
            warning_threshold = _coerce_float(indicator.get("warning_threshold"))
            diagnostics.append(
                {
                    "label": label,
                    "value": _format_failure_indicator(
                        _coerce_float(indicator.get("value")),
                        warning_threshold,
                        status,
                    ),
                    "status": status,
                    "warning_threshold": warning_threshold,
                    "failure_indicator": True,
                    "source": str(reflex_dependence.get("source") or ""),
                }
            )

        def top_level_indicator(
            value_key: str,
            warning_key: str,
            threshold_key: str,
            status_key: str,
        ) -> dict[str, object] | None:
            value = _coerce_optional_float(reflex_dependence.get(value_key))
            if value is None:
                return None
            raw_warning = reflex_dependence.get(warning_key)
            warning_threshold = _coerce_optional_float(
                reflex_dependence.get(threshold_key)
            )
            if warning_threshold is None:
                warning_threshold = _coerce_optional_float(raw_warning)
            status = str(reflex_dependence.get(status_key) or "")
            if not status:
                status = "warning" if raw_warning is True else "ok"
            return {
                "value": value,
                "warning_threshold": (
                    warning_threshold if warning_threshold is not None else 0.0
                ),
                "status": status,
            }

        override = top_level_indicator(
            "override_rate",
            "override_warning",
            "override_warning_threshold",
            "override_status",
        )
        dominance = top_level_indicator(
            "dominance_rate",
            "dominance_warning",
            "dominance_warning_threshold",
            "dominance_status",
        )
        if override is None and isinstance(indicators, Mapping):
            legacy_override = indicators.get("override_rate", {})
            if isinstance(legacy_override, Mapping):
                override = dict(legacy_override)
        if dominance is None and isinstance(indicators, Mapping):
            legacy_dominance = indicators.get("dominance", {})
            if isinstance(legacy_dominance, Mapping):
                dominance = dict(legacy_dominance)
        if override is not None:
            append_reflex_indicator("Reflex Dependence: override rate", override)
        if dominance is not None:
            append_reflex_indicator("Reflex Dependence: dominance", dominance)

    scenarios = scenario_success.get("scenarios", [])
    if isinstance(scenarios, list) and scenarios:
        filtered_scenarios = [
            (scenario, value)
            for scenario in scenarios
            if isinstance(scenario, Mapping)
            and (value := _numeric_metric(scenario, "success_rate")) is not None
        ]
        if filtered_scenarios:
            weakest, weakest_success_rate = min(filtered_scenarios, key=lambda item: item[1])
            diagnostics.append(
                {
                    "label": "Weakest scenario",
                    "value": f"{weakest.get('scenario')} ({weakest_success_rate:.2f})",
                }
            )

    variants = ablations.get("variants", {})
    if isinstance(variants, Mapping) and variants:
        capacity_totals: dict[str, int] = {}
        sortable = []
        for variant_name, payload in variants.items():
            if not isinstance(payload, Mapping):
                continue
            minimal_payload = _variant_with_minimal_reflex_support(payload)
            variant_parameter_counts = _mapping_or_empty(
                minimal_payload.get("parameter_counts")
            )
            variant_total = int(
                _coerce_float(
                    variant_parameter_counts.get("total"),
                    _coerce_float(
                        variant_parameter_counts.get("total_trainable"),
                        0.0,
                    ),
                )
            )
            if variant_total > 0:
                capacity_totals[str(variant_name)] = variant_total
            summary_payload = minimal_payload.get("summary", {})
            if not isinstance(summary_payload, Mapping):
                continue
            scenario_success_rate = _numeric_metric(
                summary_payload,
                "scenario_success_rate",
            )
            if scenario_success_rate is not None:
                sortable.append((scenario_success_rate, str(variant_name)))
        if sortable:
            best_score, best_variant = max(sortable)
            diagnostics.append(
                {
                    "label": "Best ablation variant",
                    "value": f"{best_variant} ({best_score:.2f})",
                }
            )
        capacity_summary = compare_capacity_totals(capacity_totals)
        if bool(capacity_summary.get("capacity_matched")) or _coerce_float(
            capacity_summary.get("ratio"),
            0.0,
        ) > 1.0:
            diagnostics.append(
                {
                    "label": "Architecture capacity match",
                    "value": str(capacity_summary.get("status") or ""),
                }
            )

    modules = reflex_frequency.get("modules", [])
    if isinstance(modules, list) and modules:
        valid_modules = [
            (module, value)
            for module in modules
            if isinstance(module, Mapping)
            and (value := _numeric_metric(module, "reflex_events")) is not None
        ]
        if valid_modules:
            highest, reflex_events = max(valid_modules, key=lambda item: item[1])
            diagnostics.append(
                {
                    "label": "Most frequent reflex source",
                    "value": f"{highest.get('module')} ({int(reflex_events)} events)",
                }
            )
    return diagnostics

__all__ = ["build_diagnostics", "build_reward_component_rows"]
