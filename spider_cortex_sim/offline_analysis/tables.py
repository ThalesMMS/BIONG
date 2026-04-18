from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .extractors import (
    _format_failure_indicator,
    _variant_with_minimal_reflex_support,
    build_primary_benchmark,
    build_reflex_dependence_indicators,
    extract_ablations,
)
from .uncertainty import (
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
from .utils import _coerce_float, _coerce_optional_float, _mapping_or_empty
from .writers import _table

def build_scenario_checks_rows(
    scenario_success: Mapping[str, object],
) -> list[dict[str, object]]:
    """
    Builds a list of per-scenario check result rows from a scenario-success payload.
    
    Parameters:
        scenario_success (Mapping[str, object]): Payload containing a "scenarios" list (each scenario is a mapping) and an optional top-level "source" string.
    
    Returns:
        list[dict[str, object]]: A list of rows; each row is a mapping with the keys:
            - `scenario` (str): scenario identifier or empty string.
            - `check_name` (str): name of the check.
            - `pass_rate` (float): check pass rate coerced to float and rounded to 6 decimals.
            - `mean_value` (float): check mean value coerced to float and rounded to 6 decimals.
            - `expected` (str): expected value string (empty if missing).
            - `description` (str): description string (empty if missing).
            - `source` (str): source string taken from `scenario_success["source"]` (empty if missing).
    """
    rows: list[dict[str, object]] = []
    scenarios_list = scenario_success.get("scenarios")
    for scenario in (scenarios_list if isinstance(scenarios_list, list) else []):
        if not isinstance(scenario, Mapping):
            continue
        checks = scenario.get("checks", {})
        if not isinstance(checks, Mapping):
            continue
        for check_name, payload in sorted(checks.items()):
            if not isinstance(payload, Mapping):
                continue
            rows.append(
                {
                    "scenario": str(scenario.get("scenario") or ""),
                    "check_name": str(check_name),
                    "pass_rate": round(_coerce_float(payload.get("pass_rate")), 6),
                    "mean_value": round(_coerce_float(payload.get("mean_value")), 6),
                    "expected": str(payload.get("expected") or ""),
                    "description": str(payload.get("description") or ""),
                    "source": str(scenario_success.get("source") or ""),
                }
            )
    return rows


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
        Append one row per entry in `components` into the module's rows collection for later table assembly.
        
        Parameters:
            source (str): Label identifying the source of these components (e.g., "training", "evaluation", or scenario path).
            scope (str): Scope identifier applied to each row (e.g., "trace", "scenario:<id>", or similar).
            components (Mapping|object): Mapping of component name -> numeric value. If `components` is not a mapping, the function does nothing.
        
        Description:
            For each key/value pair in `components` (sorted by key), this function stringifies the component name, coerces the value to a float, rounds it to 6 decimal places, and appends a row dictionary with keys `source`, `scope`, `component`, and `value` to the module-level rows collection.
        """
        if not isinstance(components, Mapping):
            return
        for component, value in sorted(components.items()):
            rows.append(
                {
                    "source": source,
                    "scope": scope,
                    "component": str(component),
                    "value": round(_coerce_float(value), 6),
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
        components = item.get("reward_components")
        if not isinstance(components, Mapping):
            continue
        for component, value in components.items():
            trace_totals[str(component)] += _coerce_float(value)
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

    reflex_dependence = build_reflex_dependence_indicators(summary, reflex_frequency)
    indicators = reflex_dependence.get("failure_indicators", {})
    if reflex_dependence.get("available") and isinstance(indicators, Mapping):
        override = indicators.get("override_rate", {})
        if isinstance(override, Mapping):
            diagnostics.append(
                {
                    "label": "Reflex Dependence: override rate",
                    "value": _format_failure_indicator(
                        _coerce_float(override.get("value")),
                        _coerce_float(override.get("warning_threshold")),
                        str(override.get("status") or "ok"),
                    ),
                    "status": str(override.get("status") or "ok"),
                    "warning_threshold": _coerce_float(
                        override.get("warning_threshold")
                    ),
                    "failure_indicator": True,
                    "source": str(reflex_dependence.get("source") or ""),
                }
            )
        dominance = indicators.get("dominance", {})
        if isinstance(dominance, Mapping):
            diagnostics.append(
                {
                    "label": "Reflex Dependence: dominance",
                    "value": _format_failure_indicator(
                        _coerce_float(dominance.get("value")),
                        _coerce_float(dominance.get("warning_threshold")),
                        str(dominance.get("status") or "ok"),
                    ),
                    "status": str(dominance.get("status") or "ok"),
                    "warning_threshold": _coerce_float(
                        dominance.get("warning_threshold")
                    ),
                    "failure_indicator": True,
                    "source": str(reflex_dependence.get("source") or ""),
                }
            )

    scenarios = scenario_success.get("scenarios", [])
    if isinstance(scenarios, list) and scenarios:
        filtered_scenarios = [
            scenario for scenario in scenarios if isinstance(scenario, Mapping)
        ]
        if filtered_scenarios:
            weakest = min(
                filtered_scenarios,
                key=lambda item: _coerce_float(item.get("success_rate"), 0.0),
            )
            diagnostics.append(
                {
                    "label": "Weakest scenario",
                    "value": f"{weakest.get('scenario')} ({_coerce_float(weakest.get('success_rate')):.2f})",
                }
            )

    variants = ablations.get("variants", {})
    if isinstance(variants, Mapping) and variants:
        sortable = []
        for variant_name, payload in variants.items():
            if not isinstance(payload, Mapping):
                continue
            minimal_payload = _variant_with_minimal_reflex_support(payload)
            summary_payload = minimal_payload.get("summary", {})
            if not isinstance(summary_payload, Mapping):
                continue
            sortable.append(
                (
                    _coerce_float(
                        summary_payload.get("scenario_success_rate"),
                        0.0,
                    ),
                    str(variant_name),
                )
            )
        if sortable:
            best_score, best_variant = max(sortable)
            diagnostics.append(
                {
                    "label": "Best ablation variant",
                    "value": f"{best_variant} ({best_score:.2f})",
                }
            )

    modules = reflex_frequency.get("modules", [])
    if isinstance(modules, list) and modules:
        valid_modules = [module for module in modules if isinstance(module, Mapping)]
        if valid_modules:
            highest = max(
                valid_modules,
                key=lambda item: _coerce_float(item.get("reflex_events"), 0.0),
            )
            diagnostics.append(
                {
                    "label": "Most frequent reflex source",
                    "value": f"{highest.get('module')} ({int(_coerce_float(highest.get('reflex_events'), 0.0))} events)",
                }
            )
    return diagnostics


def build_aggregate_benchmark_tables(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """
    Build aggregate benchmark tables with uncertainty-aware confidence-interval fields.
    
    Parameters:
        summary (Mapping[str, object]): Evaluation and training payload from which
            benchmark rows and uncertainty blocks are extracted.
    
    Returns:
        dict[str, object]: A dictionary containing:
            - available (bool): True if any table rows were produced.
            - confidence_level (float): Confidence level used for CI columns (defaults to 0.95 if unavailable).
            - primary_benchmark (dict): Table dict with columns for metric, label, value, CI bounds, std error, n_seeds, confidence_level, reference_variant, and source.
            - per_scenario_success_rates (dict): Table dict with per-scenario rows and CI columns for scenario_success_rate.
            - learning_evidence_deltas (dict): Table dict with delta rows (e.g., scenario_success_rate_delta) and CI columns for learning-evidence comparisons.
            - limitations (list[str]): Human-readable messages explaining missing data when any of the tables are empty.
    """
    scenario_success = _primary_benchmark_scenario_success(summary)
    primary_benchmark = build_primary_benchmark(
        summary,
        scenario_success,
        extract_ablations(summary, ()),
    )
    source, reference_variant, payload = _primary_benchmark_source_payload(summary)
    primary_rows: list[dict[str, object]] = []
    if primary_benchmark.get("available"):
        row = {
            "metric": str(primary_benchmark.get("metric") or "scenario_success_rate"),
            "label": str(
                primary_benchmark.get("label") or "No-reflex scenario_success_rate"
            ),
            "reference_variant": str(
                primary_benchmark.get("reference_variant") or reference_variant
            ),
            "source": str(primary_benchmark.get("source") or source),
        }
        row.update(
            _ci_row_fields(
                _mapping_or_empty(primary_benchmark.get("uncertainty")),
                value=primary_benchmark.get("scenario_success_rate"),
            )
        )
        primary_rows.append(row)

    scenario_rows: list[dict[str, object]] = []
    suite = _mapping_or_empty(payload.get("suite"))
    if not suite:
        behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
        if _payload_has_zero_reflex_scale(behavior_evaluation):
            suite = _mapping_or_empty(behavior_evaluation.get("suite"))
            payload = behavior_evaluation
            source = "summary.behavior_evaluation"
    for scenario_name, scenario_payload in sorted(suite.items()):
        if not isinstance(scenario_payload, Mapping):
            continue
        row = {
            "scenario": str(scenario_name),
            "metric": "scenario_success_rate",
            "source": f"{source}.suite.{scenario_name}",
        }
        row.update(
            _ci_row_fields(
                _payload_uncertainty(payload, "scenario_success_rate", scenario=str(scenario_name)),
                value=scenario_payload.get("success_rate"),
            )
        )
        scenario_rows.append(row)

    learning_rows: list[dict[str, object]] = []
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    learning = _mapping_or_empty(behavior_evaluation.get("learning_evidence"))
    evidence_summary = _mapping_or_empty(learning.get("evidence_summary"))
    evidence_uncertainty = _mapping_or_empty(evidence_summary.get("uncertainty"))
    for comparison_key, comparison_label in (
        ("trained_vs_random_init", "random_init"),
        ("trained_vs_reflex_only", "reflex_only"),
    ):
        scalar_block = _mapping_or_empty(evidence_summary.get(comparison_key))
        uncertainty_block = _mapping_or_empty(evidence_uncertainty.get(comparison_key))
        for metric_name in (
            "scenario_success_rate_delta",
            "episode_success_rate_delta",
            "mean_reward_delta",
        ):
            if metric_name not in scalar_block and metric_name not in uncertainty_block:
                continue
            row = {
                "comparison": comparison_label,
                "metric": metric_name,
                "source": f"summary.behavior_evaluation.learning_evidence.evidence_summary.{comparison_key}",
            }
            row.update(
                _ci_row_fields(
                    _uncertainty_or_empty(uncertainty_block.get(metric_name)),
                    value=scalar_block.get(metric_name),
                )
            )
            learning_rows.append(row)

    limitations: list[str] = []
    if not primary_rows:
        limitations.append("No primary benchmark row with uncertainty was available.")
    if not scenario_rows:
        limitations.append("No per-scenario benchmark rows with uncertainty were available.")
    if not learning_rows:
        limitations.append("No learning-evidence delta uncertainty rows were available.")
    return {
        "available": bool(primary_rows or scenario_rows or learning_rows),
        "confidence_level": (
            primary_rows[0].get("confidence_level")
            if primary_rows and primary_rows[0].get("confidence_level") is not None
            else 0.95
        ),
        "primary_benchmark": _table(
            (
                "metric",
                "label",
                "value",
                "ci_lower",
                "ci_upper",
                "std_error",
                "n_seeds",
                "confidence_level",
                "reference_variant",
                "source",
            ),
            primary_rows,
        ),
        "per_scenario_success_rates": _table(
            (
                "scenario",
                "metric",
                "value",
                "ci_lower",
                "ci_upper",
                "std_error",
                "n_seeds",
                "confidence_level",
                "source",
            ),
            scenario_rows,
        ),
        "learning_evidence_deltas": _table(
            (
                "comparison",
                "metric",
                "value",
                "ci_lower",
                "ci_upper",
                "std_error",
                "n_seeds",
                "confidence_level",
                "source",
            ),
            learning_rows,
        ),
        "limitations": limitations,
    }


def _claim_uncertainty_for_condition(
    uncertainty: object,
    condition: str,
) -> Mapping[str, object]:
    """
    Select the uncertainty mapping relevant to a named condition.
    
    If `uncertainty` contains a mapping for `condition`, that nested mapping is returned; otherwise, if the top-level uncertainty mapping contains a `ci_lower` or `mean` key it is returned; otherwise an empty mapping is returned.
    
    Parameters:
        uncertainty (object): Uncertainty payload that will be coerced to a mapping.
        condition (str): Condition name to look up within the uncertainty payload.
    
    Returns:
        Mapping[str, object]: The uncertainty mapping for `condition`, the top-level uncertainty mapping if it contains uncertainty fields, or an empty mapping.
    """
    payload = _mapping_or_empty(uncertainty)
    nested = payload.get(condition)
    if isinstance(nested, Mapping):
        return nested
    return payload if "ci_lower" in payload or "mean" in payload else {}


def build_claim_test_tables(summary: Mapping[str, object]) -> dict[str, object]:
    """
    Build a table of claim-test uncertainty rows covering reference, comparison, delta, and effect-size roles.
    
    Parameters:
    	summary (Mapping[str, object]): Evaluation payload containing `behavior_evaluation.claim_tests`.
    
    Returns:
    	dict: A dictionary with:
    		- available (bool): True if any claim-test rows were produced.
    		- claim_results (dict): A table-like mapping containing rows for each claim and role. Each row includes columns such as `claim`, `status`, `passed`, `role`, `condition`, `metric`, `value`, CI fields (`ci_lower`, `ci_upper`, `std_error`, `n_seeds`, `confidence_level`), `cohens_d`, and `effect_magnitude`.
    		- limitations (list[str]): Human-readable limitation messages when no rows are available.
    """
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    claim_payload = _mapping_or_empty(behavior_evaluation.get("claim_tests"))
    claims = _mapping_or_empty(claim_payload.get("claims"))
    rows: list[dict[str, object]] = []
    for claim_name, raw_result in sorted(claims.items()):
        if not isinstance(raw_result, Mapping):
            continue
        result = raw_result
        status = str(result.get("status") or "")
        primary_metric = str(result.get("primary_metric") or "")
        passed = bool(result.get("passed", False))

        reference_value = result.get("reference_value")
        reference_uncertainty = _mapping_or_empty(result.get("reference_uncertainty"))
        if isinstance(reference_value, Mapping):
            for metric_name, value in sorted(reference_value.items()):
                metric_uncertainty = (
                    reference_uncertainty.get(metric_name)
                    if isinstance(reference_uncertainty.get(metric_name), Mapping)
                    else reference_uncertainty
                )
                row = {
                    "claim": str(claim_name),
                    "status": status,
                    "passed": passed,
                    "role": "reference",
                    "condition": "reference",
                    "metric": str(metric_name),
                    "cohens_d": None,
                    "effect_magnitude": "",
                }
                row.update(_ci_row_fields(metric_uncertainty, value=value))
                rows.append(row)
        else:
            row = {
                "claim": str(claim_name),
                "status": status,
                "passed": passed,
                "role": "reference",
                "condition": "reference",
                "metric": primary_metric,
                "cohens_d": None,
                "effect_magnitude": "",
            }
            row.update(_ci_row_fields(reference_uncertainty, value=reference_value))
            rows.append(row)

        comparison_values = _mapping_or_empty(result.get("comparison_values"))
        comparison_uncertainty = _mapping_or_empty(result.get("comparison_uncertainty"))
        for condition, value in sorted(comparison_values.items()):
            row = {
                "claim": str(claim_name),
                "status": status,
                "passed": passed,
                "role": "comparison",
                "condition": str(condition),
                "metric": primary_metric,
                "cohens_d": None,
                "effect_magnitude": "",
            }
            row.update(
                _ci_row_fields(
                    _claim_uncertainty_for_condition(comparison_uncertainty, str(condition)),
                    value=value,
                )
            )
            rows.append(row)

        delta_values = _mapping_or_empty(result.get("delta"))
        delta_uncertainty = _mapping_or_empty(result.get("delta_uncertainty"))
        for condition, value in sorted(delta_values.items()):
            row = {
                "claim": str(claim_name),
                "status": status,
                "passed": passed,
                "role": "delta",
                "condition": str(condition),
                "metric": f"{primary_metric}_delta" if primary_metric else "delta",
                "cohens_d": None,
                "effect_magnitude": "",
            }
            row.update(
                _ci_row_fields(
                    _claim_uncertainty_for_condition(delta_uncertainty, str(condition)),
                    value=value,
                )
            )
            rows.append(row)

        effect_values = result.get("effect_size")
        effect_uncertainty = _mapping_or_empty(result.get("effect_size_uncertainty"))
        cohens_d_values = _mapping_or_empty(result.get("cohens_d"))
        magnitude_values = _mapping_or_empty(result.get("effect_magnitude"))
        if isinstance(effect_values, Mapping):
            effect_items = sorted(effect_values.items())
        else:
            effect_items = [("effect_size", effect_values)]
        for condition, value in effect_items:
            condition_name = str(condition)
            row = {
                "claim": str(claim_name),
                "status": status,
                "passed": passed,
                "role": "effect_size",
                "condition": condition_name,
                "metric": "effect_size",
                "cohens_d": _coerce_optional_float(cohens_d_values.get(condition_name)),
                "effect_magnitude": str(magnitude_values.get(condition_name) or ""),
            }
            row.update(
                _ci_row_fields(
                    _claim_uncertainty_for_condition(effect_uncertainty, condition_name),
                    value=value,
                )
            )
            rows.append(row)

    limitations: list[str] = []
    if not rows:
        limitations.append("No claim-test uncertainty rows were available.")
    return {
        "available": bool(rows),
        "claim_results": _table(
            (
                "claim",
                "status",
                "passed",
                "role",
                "condition",
                "metric",
                "value",
                "ci_lower",
                "ci_upper",
                "std_error",
                "n_seeds",
                "confidence_level",
                "cohens_d",
                "effect_magnitude",
            ),
            rows,
        ),
        "limitations": limitations,
    }


def build_effect_size_tables(summary: Mapping[str, object]) -> dict[str, object]:
    """
    Builds effect-size rows for learning baselines, ablation comparisons, and claim-test-derived effect sizes.
    
    Parameters:
        summary (Mapping[str, object]): Evaluation/training summary payload containing
            behavior_evaluation, learning_evidence, ablations, and claim_tests.
    
    Returns:
        dict[str, object]: A dictionary with:
            - `available` (bool): True when any effect-size rows were produced.
            - `effect_sizes` (mapping): A table-like mapping of effect-size rows including
              Cohen's d, raw deltas, confidence intervals, standard errors, sample sizes,
              and source paths.
            - `limitations` (list[str]): Human-readable limitations when no rows are available.
    """
    rows: list[dict[str, object]] = []
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))

    learning = _mapping_or_empty(behavior_evaluation.get("learning_evidence"))
    learning_conditions = _mapping_or_empty(learning.get("conditions"))
    reference_condition = str(
        learning.get("reference_condition")
        or _mapping_or_empty(learning.get("evidence_summary")).get("reference_condition")
        or "trained_without_reflex_support"
    )
    reference_payload = _mapping_or_empty(learning_conditions.get(reference_condition))
    evidence_summary = _mapping_or_empty(learning.get("evidence_summary"))
    evidence_uncertainty = _mapping_or_empty(evidence_summary.get("uncertainty"))
    for baseline in ("random_init", "reflex_only"):
        baseline_payload = _mapping_or_empty(learning_conditions.get(baseline))
        if not reference_payload or not baseline_payload:
            continue
        comparison_key = (
            "trained_vs_random_init"
            if baseline == "random_init"
            else "trained_vs_reflex_only"
        )
        scalar_block = _mapping_or_empty(evidence_summary.get(comparison_key))
        uncertainty_block = _mapping_or_empty(evidence_uncertainty.get(comparison_key))
        row = _cohens_d_row(
            domain="learning_evidence",
            baseline=baseline,
            comparison=reference_condition,
            metric="scenario_success_rate",
            baseline_values=_payload_metric_seed_items(
                baseline_payload,
                "scenario_success_rate",
            ),
            comparison_values=_payload_metric_seed_items(
                reference_payload,
                "scenario_success_rate",
            ),
            raw_delta=scalar_block.get("scenario_success_rate_delta"),
            uncertainty=_uncertainty_or_empty(
                uncertainty_block.get("scenario_success_rate_delta")
            ),
            source=f"summary.behavior_evaluation.learning_evidence.{comparison_key}",
        )
        if row is not None:
            rows.append(row)

    ablations = _mapping_or_empty(behavior_evaluation.get("ablations"))
    variants_raw = _mapping_or_empty(ablations.get("variants"))
    variants = {
        str(variant_name): _variant_with_minimal_reflex_support(payload)
        for variant_name, payload in variants_raw.items()
        if isinstance(payload, Mapping)
    }
    reference_variant = str(ablations.get("reference_variant") or "")
    if not reference_variant and "modular_full" in variants:
        reference_variant = "modular_full"
    deltas_vs_reference = _mapping_or_empty(ablations.get("deltas_vs_reference"))
    for baseline in ("modular_full", "monolithic_policy"):
        baseline_payload = _mapping_or_empty(variants.get(baseline))
        if not baseline_payload:
            continue
        baseline_summary = _mapping_or_empty(baseline_payload.get("summary"))
        baseline_values = _payload_metric_seed_items(
            baseline_payload,
            "scenario_success_rate",
        )
        for comparison, comparison_payload in sorted(variants.items()):
            if comparison == baseline or not isinstance(comparison_payload, Mapping):
                continue
            comparison_summary = _mapping_or_empty(comparison_payload.get("summary"))
            comparison_values = _payload_metric_seed_items(
                comparison_payload,
                "scenario_success_rate",
            )
            scalar_block: Mapping[str, object] = {}
            uncertainty_block: Mapping[str, object] = {}
            if baseline == reference_variant:
                delta_payload = _mapping_or_empty(deltas_vs_reference.get(comparison))
                scalar_block = _mapping_or_empty(delta_payload.get("summary"))
                uncertainty_block = _mapping_or_empty(delta_payload.get("uncertainty"))
            raw_delta: object = scalar_block.get("scenario_success_rate_delta")
            if raw_delta is None:
                raw_delta = round(
                    _coerce_float(comparison_summary.get("scenario_success_rate"))
                    - _coerce_float(baseline_summary.get("scenario_success_rate")),
                    6,
                )
            row = _cohens_d_row(
                domain="ablation",
                baseline=baseline,
                comparison=str(comparison),
                metric="scenario_success_rate",
                baseline_values=baseline_values,
                comparison_values=comparison_values,
                raw_delta=raw_delta,
                uncertainty=_uncertainty_or_seed_delta(
                    uncertainty_block.get("scenario_success_rate_delta"),
                    baseline_values,
                    comparison_values,
                    raw_delta=raw_delta,
                ),
                source=(
                    "summary.behavior_evaluation.ablations.variants."
                    f"{comparison}.summary"
                ),
            )
            if row is not None:
                rows.append(row)

    claim_tables = _mapping_or_empty(build_claim_test_tables(summary).get("claim_results"))
    claim_rows = claim_tables.get("rows", [])
    if isinstance(claim_rows, list):
        for item in claim_rows:
            if not isinstance(item, Mapping) or str(item.get("role")) != "effect_size":
                continue
            cohens_value = _coerce_optional_float(item.get("cohens_d"))
            if cohens_value is None:
                continue
            effect_ci_lower = _coerce_optional_float(item.get("ci_lower"))
            effect_ci_upper = _coerce_optional_float(item.get("ci_upper"))
            effect_std_error = _coerce_optional_float(item.get("std_error"))
            effect_n_seeds = int(_coerce_float(item.get("n_seeds"), 0.0))
            effect_confidence_level = _coerce_optional_float(
                item.get("confidence_level")
            )
            rows.append(
                {
                    "domain": "claim_test",
                    "baseline": "reference",
                    "comparison": str(item.get("condition") or ""),
                    "metric": str(item.get("metric") or "effect_size"),
                    "raw_delta": _coerce_optional_float(item.get("value")),
                    "cohens_d": cohens_value,
                    "magnitude_label": str(item.get("effect_magnitude") or ""),
                    "source": f"summary.behavior_evaluation.claim_tests.claims.{item.get('claim')}",
                    "value": cohens_value,
                    "ci_lower": effect_ci_lower,
                    "ci_upper": effect_ci_upper,
                    "std_error": effect_std_error,
                    "n_seeds": effect_n_seeds,
                    "confidence_level": effect_confidence_level,
                    "effect_size_ci_lower": effect_ci_lower,
                    "effect_size_ci_upper": effect_ci_upper,
                    "effect_size_std_error": effect_std_error,
                    "effect_size_n_seeds": effect_n_seeds,
                    "effect_size_confidence_level": effect_confidence_level,
                    "delta_ci_lower": None,
                    "delta_ci_upper": None,
                    "delta_std_error": None,
                    "delta_n_seeds": 0,
                    "delta_confidence_level": None,
                }
            )

    limitations: list[str] = []
    if not rows:
        limitations.append("No effect-size rows were available for the main baselines.")
    return {
        "available": bool(rows),
        "effect_sizes": _table(
            (
                "domain",
                "baseline",
                "comparison",
                "metric",
                "raw_delta",
                "cohens_d",
                "magnitude_label",
                "value",
                "ci_lower",
                "ci_upper",
                "std_error",
                "n_seeds",
                "confidence_level",
                "effect_size_ci_lower",
                "effect_size_ci_upper",
                "effect_size_std_error",
                "effect_size_n_seeds",
                "effect_size_confidence_level",
                "delta_ci_lower",
                "delta_ci_upper",
                "delta_std_error",
                "delta_n_seeds",
                "delta_confidence_level",
                "source",
            ),
            rows,
        ),
        "limitations": limitations,
    }