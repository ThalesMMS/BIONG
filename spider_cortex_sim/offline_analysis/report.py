from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .constants import SHAPING_DEPENDENCE_WARNING_THRESHOLD
from .extractors import (
    _noise_robustness_marginal,
    _noise_robustness_rate,
    _normalize_float_map,
    _representation_interpretation,
    build_primary_benchmark,
    build_reflex_dependence_indicators,
    extract_ablations,
    extract_comparisons,
    extract_noise_robustness,
    extract_predator_type_specialization,
    extract_reflex_frequency,
    extract_representation_specialization,
    extract_scenario_success,
    extract_shaping_audit,
    extract_training_eval_series,
)
from .ingestion import normalize_behavior_rows
from .renderers import (
    render_bar_chart,
    render_line_chart,
    render_matrix_heatmap,
    render_placeholder_svg,
)
from .tables import (
    build_aggregate_benchmark_tables,
    build_claim_test_tables,
    build_diagnostics,
    build_effect_size_tables,
    build_reward_component_rows,
    build_scenario_checks_rows,
)
from .utils import (
    _coerce_float,
    _coerce_optional_float,
    _format_optional_metric,
    _mapping_or_empty,
)
from .writers import (
    _markdown_table,
    _markdown_value,
    _table_rows,
    _write_csv,
    _write_svg,
)



from .report_data import build_report_data

def write_report(output_dir: str | Path, report: Mapping[str, object]) -> dict[str, str]:
    """
    Write filesystem artifacts (SVG, CSV, JSON, and Markdown) that represent the provided analysis report.
    
    Parameters:
        output_dir (str | Path): Target directory where artifact files will be created; created if missing.
        report (Mapping[str, object]): Report data used to populate charts, tables, and markdown. Expected keys include (but are not limited to) "training_eval", "scenario_success", "shaping_program", "ablations", "reflex_frequency", "scenario_checks", "reward_components", "diagnostics", "primary_benchmark", "limitations", and "inputs".
    
    Returns:
        dict[str, str]: Mapping of artifact identifiers to their filesystem paths. Keys include "report_md", "report_json", "training_eval_svg", "scenario_success_svg", "scenario_checks_csv", "reward_components_csv", "ablation_comparison_svg", and "reflex_frequency_svg".
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_eval = report.get("training_eval", {})
    scenario_success = report.get("scenario_success", {})
    primary_benchmark = report.get("primary_benchmark", {})
    shaping_program = report.get("shaping_program", {})
    noise_robustness = report.get("noise_robustness", {})
    ablations = report.get("ablations", {})
    predator_type_specialization = report.get("predator_type_specialization", {})
    representation_specialization = report.get("representation_specialization", {})
    reflex_frequency = report.get("reflex_frequency", {})
    aggregate_benchmark_tables = _mapping_or_empty(
        report.get("aggregate_benchmark_tables")
    )
    claim_test_tables = _mapping_or_empty(report.get("claim_test_tables"))
    effect_size_tables = _mapping_or_empty(report.get("effect_size_tables"))

    training_series = []
    if isinstance(training_eval, Mapping):
        training_series.extend(
            training_eval.get("training_points", [])
            if isinstance(training_eval.get("training_points"), list)
            else []
        )
        evaluation_points = (
            training_eval.get("evaluation_points", [])
            if isinstance(training_eval.get("evaluation_points"), list)
            else []
        )
    else:
        evaluation_points = []

    training_svg_path = output_path / "training_eval.svg"
    if training_series:
        _write_svg(
            training_svg_path,
            render_line_chart("Training reward trajectory", training_series),
        )
    elif evaluation_points:
        _write_svg(
            training_svg_path,
            render_line_chart("Evaluation reward trajectory", evaluation_points),
        )
    else:
        _write_svg(
            training_svg_path,
            render_placeholder_svg("Training / evaluation", "No summary reward series was available."),
        )

    scenario_svg_path = output_path / "scenario_success.svg"
    scenario_items = (
        scenario_success.get("scenarios", [])
        if isinstance(scenario_success, Mapping)
        else []
    )
    scenario_chart_items = [
        {"scenario": item.get("scenario"), "success_rate": item.get("success_rate")}
        for item in scenario_items
        if isinstance(item, Mapping)
    ]
    _write_svg(
        scenario_svg_path,
        render_bar_chart(
            "Scenario success rate",
            scenario_chart_items,
            label_key="scenario",
            value_key="success_rate",
        ),
    )

    robustness_svg_path = output_path / "robustness_matrix.svg"
    _nr = _mapping_or_empty(noise_robustness)
    robustness_matrix_spec = _mapping_or_empty(_nr.get("matrix_spec"))
    _train_conds_raw = robustness_matrix_spec.get("train_conditions")
    _eval_conds_raw = robustness_matrix_spec.get("eval_conditions")
    robustness_train_conditions = list(_train_conds_raw) if isinstance(_train_conds_raw, list) else []
    robustness_eval_conditions = list(_eval_conds_raw) if isinstance(_eval_conds_raw, list) else []
    robustness_matrix = _mapping_or_empty(_nr.get("matrix"))
    robustness_train_marginals = _mapping_or_empty(_nr.get("train_marginals"))
    robustness_eval_marginals = _mapping_or_empty(_nr.get("eval_marginals"))
    _write_svg(
        robustness_svg_path,
        render_matrix_heatmap(
            "Noise robustness matrix",
            train_conditions=robustness_train_conditions,
            eval_conditions=robustness_eval_conditions,
            matrix=robustness_matrix,
            train_marginals=robustness_train_marginals,
            eval_marginals=robustness_eval_marginals,
        ),
    )

    ablation_svg_path = ""
    if isinstance(ablations, Mapping) and ablations.get("available"):
        variants = []
        variants_payload = ablations.get("variants", {})
        if isinstance(variants_payload, Mapping):
            for variant_name, payload in sorted(variants_payload.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_payload = payload.get("summary", {})
                variants.append(
                    {
                        "variant": variant_name,
                        "score": _coerce_float(
                            summary_payload.get("scenario_success_rate"),
                            0.0,
                        ),
                    }
                )
        if variants:
            ablation_svg_path = str(output_path / "ablation_comparison.svg")
            _write_svg(
                Path(ablation_svg_path),
                render_bar_chart(
                    "Ablation scenario success rate",
                    variants,
                    label_key="variant",
                    value_key="score",
                ),
            )

    reflex_svg_path = ""
    if isinstance(reflex_frequency, Mapping) and reflex_frequency.get("available"):
        modules = reflex_frequency.get("modules", [])
        if isinstance(modules, list) and modules:
            reflex_svg_path = str(output_path / "reflex_frequency.svg")
            _write_svg(
                Path(reflex_svg_path),
                render_bar_chart(
                    "Reflex events per module",
                    modules,
                    label_key="module",
                    value_key="reflex_events",
                ),
            )

    representation_svg_path = output_path / "representation_specialization.svg"
    representation_payload = _mapping_or_empty(representation_specialization)
    representation_chart_items = [
        {
            "module": str(module_name),
            "score": _coerce_float(score),
        }
        for module_name, score in sorted(
            _normalize_float_map(representation_payload.get("proposer_divergence")).items()
        )
    ]
    if representation_chart_items:
        _write_svg(
            representation_svg_path,
            render_bar_chart(
                "Representation specialization by module",
                representation_chart_items,
                label_key="module",
                value_key="score",
            ),
        )
    else:
        _write_svg(
            representation_svg_path,
            render_placeholder_svg(
                "Representation specialization",
                "No representation-specialization divergence scores were available.",
            ),
        )

    scenario_checks_path = output_path / "scenario_checks.csv"
    _write_csv(
        scenario_checks_path,
        report.get("scenario_checks", []) if isinstance(report.get("scenario_checks"), list) else [],
        ("scenario", "check_name", "pass_rate", "mean_value", "expected", "description", "source"),
    )

    reward_components_path = output_path / "reward_components.csv"
    _write_csv(
        reward_components_path,
        report.get("reward_components", []) if isinstance(report.get("reward_components"), list) else [],
        ("source", "scope", "component", "value"),
    )

    report_json_path = output_path / "report.json"
    report_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    diagnostics = report.get("diagnostics", [])
    diagnostic_rows = [
        (
            str(item.get("label") or ""),
            str(item.get("value") or ""),
        )
        for item in diagnostics
        if isinstance(item, Mapping)
    ]
    primary_benchmark_rows = []
    if isinstance(primary_benchmark, Mapping) and primary_benchmark.get("available"):
        primary_benchmark_rows.append(
            (
                str(primary_benchmark.get("label") or ""),
                f"{_coerce_float(primary_benchmark.get('scenario_success_rate')):.2f}",
                (
                    "n/a"
                    if primary_benchmark.get("eval_reflex_scale") is None
                    else f"{_coerce_float(primary_benchmark.get('eval_reflex_scale')):.2f}"
                ),
                str(primary_benchmark.get("reference_variant") or ""),
                str(primary_benchmark.get("source") or ""),
            )
        )

    aggregate_primary_rows = []
    aggregate_primary_table = _mapping_or_empty(
        aggregate_benchmark_tables.get("primary_benchmark")
    )
    for item in _table_rows(aggregate_primary_table):
        if not isinstance(item, Mapping):
            continue
        aggregate_primary_rows.append(
            (
                str(item.get("metric") or ""),
                _markdown_value(item.get("value")),
                _markdown_value(item.get("ci_lower")),
                _markdown_value(item.get("ci_upper")),
                str(int(_coerce_float(item.get("n_seeds"), 0.0))),
                str(item.get("reference_variant") or ""),
                str(item.get("source") or ""),
            )
        )
    aggregate_scenario_rows = []
    aggregate_scenario_table = _mapping_or_empty(
        aggregate_benchmark_tables.get("per_scenario_success_rates")
    )
    for item in _table_rows(aggregate_scenario_table):
        if not isinstance(item, Mapping):
            continue
        aggregate_scenario_rows.append(
            (
                str(item.get("scenario") or ""),
                _markdown_value(item.get("value")),
                _markdown_value(item.get("ci_lower")),
                _markdown_value(item.get("ci_upper")),
                str(int(_coerce_float(item.get("n_seeds"), 0.0))),
            )
        )
    aggregate_learning_rows = []
    aggregate_learning_table = _mapping_or_empty(
        aggregate_benchmark_tables.get("learning_evidence_deltas")
    )
    for item in _table_rows(aggregate_learning_table):
        if not isinstance(item, Mapping):
            continue
        aggregate_learning_rows.append(
            (
                str(item.get("comparison") or ""),
                str(item.get("metric") or ""),
                _markdown_value(item.get("value")),
                _markdown_value(item.get("ci_lower")),
                _markdown_value(item.get("ci_upper")),
                str(int(_coerce_float(item.get("n_seeds"), 0.0))),
            )
        )

    claim_uncertainty_rows = []
    claim_results_table = _mapping_or_empty(claim_test_tables.get("claim_results"))
    for item in _table_rows(claim_results_table):
        if not isinstance(item, Mapping):
            continue
        claim_uncertainty_rows.append(
            (
                str(item.get("claim") or ""),
                str(item.get("role") or ""),
                str(item.get("condition") or ""),
                str(item.get("metric") or ""),
                _markdown_value(item.get("value")),
                _markdown_value(item.get("ci_lower")),
                _markdown_value(item.get("ci_upper")),
                str(int(_coerce_float(item.get("n_seeds"), 0.0))),
                _markdown_value(item.get("cohens_d")),
                str(item.get("effect_magnitude") or ""),
            )
        )

    effect_size_rows = []
    effect_size_table = _mapping_or_empty(effect_size_tables.get("effect_sizes"))
    for item in _table_rows(effect_size_table):
        if not isinstance(item, Mapping):
            continue
        effect_size_rows.append(
            (
                str(item.get("domain") or ""),
                str(item.get("baseline") or ""),
                str(item.get("comparison") or ""),
                str(item.get("metric") or ""),
                _markdown_value(item.get("raw_delta")),
                _markdown_value(item.get("delta_ci_lower")),
                _markdown_value(item.get("delta_ci_upper")),
                _markdown_value(item.get("cohens_d")),
                str(item.get("magnitude_label") or ""),
                _markdown_value(item.get("ci_lower")),
                _markdown_value(item.get("ci_upper")),
                str(int(_coerce_float(item.get("n_seeds"), 0.0))),
            )
        )

    shaping_gap_rows = []
    shaping_component_rows = []
    shaping_profile_rows = []
    shaping_disposition_rows = []
    shaping_survival_rows = []
    shaping_summary_line = "_No shaping minimization audit was available._"
    shaping_removed_gap_line = "_No removed disposition weight gap was available._"
    shaping_warning_line = ""
    shaping_program = _mapping_or_empty(shaping_program)
    if shaping_program and bool(shaping_program.get("available")):
        dense_profile = str(shaping_program.get("dense_profile") or "classic")
        minimal_profile = str(shaping_program.get("minimal_profile") or "austere")
        gap_metrics = _mapping_or_empty(shaping_program.get("gap_metrics"))
        flags = _mapping_or_empty(shaping_program.get("interpretive_flags"))
        thresholds = _mapping_or_empty(shaping_program.get("thresholds"))
        if gap_metrics and bool(flags.get("gap_available")):
            scenario_delta = _coerce_float(
                gap_metrics.get("scenario_success_rate_delta")
            )
            episode_delta = _coerce_float(
                gap_metrics.get("episode_success_rate_delta")
            )
            reward_delta = _coerce_float(gap_metrics.get("mean_reward_delta"))
            shaping_gap_rows.append(
                (
                    f"{dense_profile} - {minimal_profile}",
                    f"{scenario_delta:.2f}",
                    f"{episode_delta:.2f}",
                    f"{reward_delta:.2f}",
                    str(shaping_program.get("interpretation") or ""),
                )
            )
        if bool(flags.get("shaping_dependent")):
            shaping_warning_line = (
                "> WARNING: High shaping dependence detected. Dense-profile "
                "scenario success exceeds the minimal-shaping warning threshold of "
                f"{_coerce_float(thresholds.get('shaping_dependence'), SHAPING_DEPENDENCE_WARNING_THRESHOLD):.2f}."
            )
        profile_weight_breakdown = _mapping_or_empty(
            shaping_program.get("profile_weight_breakdown")
        )
        for profile_name, weights in sorted(profile_weight_breakdown.items()):
            if not isinstance(weights, Mapping):
                continue
            for disposition, weight in sorted(weights.items()):
                shaping_profile_rows.append(
                    (
                        str(profile_name),
                        str(disposition),
                        f"{_coerce_float(weight):.2f}",
                    )
                )
        disposition_summary = _mapping_or_empty(
            shaping_program.get("disposition_summary")
        )
        for disposition, payload in sorted(disposition_summary.items()):
            if not isinstance(payload, Mapping):
                continue
            components = payload.get("components", [])
            if isinstance(components, list):
                component_names = ", ".join(str(component) for component in components)
                default_count = len(components)
            else:
                component_names = ""
                default_count = 0
            shaping_disposition_rows.append(
                (
                    str(disposition),
                    str(int(_coerce_float(payload.get("component_count"), default_count))),
                    component_names,
                )
            )
        shaping_removed_gap_line = (
            "Removed disposition weight gap "
            f"({dense_profile} - {minimal_profile}): "
            f"{_coerce_float(shaping_program.get('removed_weight_gap')):.2f}."
        )
        components = shaping_program.get("component_classification", [])
        if isinstance(components, list):
            shaping_component_rows = [
                (
                    str(item.get("component") or ""),
                    str(item.get("category") or ""),
                    str(item.get("risk") or ""),
                    str(item.get("disposition") or ""),
                    str(item.get("rationale") or ""),
                )
                for item in components
                if isinstance(item, Mapping)
            ]
        behavior_survival = _mapping_or_empty(shaping_program.get("behavior_survival"))
        survival_scenarios = behavior_survival.get("scenarios", [])
        if isinstance(survival_scenarios, list):
            shaping_survival_rows = [
                (
                    str(item.get("scenario") or ""),
                    f"{_coerce_float(item.get('austere_success_rate')):.2f}",
                    "yes" if bool(item.get("survives")) else "no",
                    str(int(_coerce_float(item.get("episodes"), 0.0))),
                )
                for item in survival_scenarios
                if isinstance(item, Mapping)
            ]
        if not bool(behavior_survival.get("available")):
            shaping_summary_line = "_No shaping survival data available._"
        else:
            shaping_summary_line = (
                f"{int(_coerce_float(behavior_survival.get('surviving_scenario_count'), 0.0))}/"
                f"{int(_coerce_float(behavior_survival.get('scenario_count'), 0.0))} "
                "scenarios survive minimal shaping "
                f"({_coerce_float(behavior_survival.get('survival_rate')):.2f})."
            )
    scenario_rows = [
        (
            str(item.get("scenario") or ""),
            f"{_coerce_float(item.get('success_rate')):.2f}",
            str(len(item.get("checks", {})) if isinstance(item.get("checks"), Mapping) else 0),
        )
        for item in scenario_items
        if isinstance(item, Mapping)
    ]
    ablation_rows = []
    ablation_predator_type_rows = []
    if isinstance(ablations, Mapping):
        variants_payload = ablations.get("variants", {})
        if isinstance(variants_payload, Mapping):
            for variant_name, payload in sorted(variants_payload.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_payload = payload.get("summary", {})
                config_payload = payload.get("config", {})
                if not isinstance(summary_payload, Mapping):
                    summary_payload = {}
                if not isinstance(config_payload, Mapping):
                    config_payload = {}
                ablation_rows.append(
                    (
                        str(variant_name),
                        str(config_payload.get("architecture") or ""),
                        f"{_coerce_float(summary_payload.get('eval_reflex_scale'), _coerce_float(payload.get('eval_reflex_scale'), 0.0)):.2f}",
                        f"{_coerce_float(summary_payload.get('scenario_success_rate')):.2f}",
                    )
                )
        predator_type_comparisons = ablations.get("predator_type_comparisons", {})
        if isinstance(predator_type_comparisons, Mapping):
            comparisons = predator_type_comparisons.get("comparisons", {})
            if isinstance(comparisons, Mapping):
                for variant_name, payload in sorted(comparisons.items()):
                    if not isinstance(payload, Mapping):
                        continue
                    visual_group = _mapping_or_empty(payload.get("visual_predator_scenarios"))
                    olfactory_group = _mapping_or_empty(payload.get("olfactory_predator_scenarios"))
                    ablation_predator_type_rows.append(
                        (
                            str(variant_name),
                            f"{_coerce_float(visual_group.get('mean_success_rate')):.2f}",
                            f"{_coerce_float(olfactory_group.get('mean_success_rate')):.2f}",
                            f"{_coerce_float(payload.get('visual_minus_olfactory_success_rate')):.2f}",
                        )
                    )

    specialization_rows = []
    specialization_summary_line = "_No predator-type specialization metrics were available._"
    specialization = _mapping_or_empty(predator_type_specialization)
    if specialization and bool(specialization.get("available")):
        predator_types = _mapping_or_empty(specialization.get("predator_types"))
        for predator_type in ("visual", "olfactory"):
            payload = _mapping_or_empty(predator_types.get(predator_type))
            specialization_rows.append(
                (
                    predator_type,
                    f"{_coerce_float(payload.get('visual_cortex_activation')):.2f}",
                    f"{_coerce_float(payload.get('sensory_cortex_activation')):.2f}",
                    str(payload.get("dominant_module") or ""),
                )
            )
        specialization_summary_line = (
            "Specialization score: "
            f"{_coerce_float(specialization.get('specialization_score')):.2f} "
            f"({str(specialization.get('interpretation') or 'unknown')}). "
            "Type-module correlation proxy: "
            f"{_coerce_float(specialization.get('type_module_correlation')):.2f}."
        )
    differential_activation = _mapping_or_empty(
        _mapping_or_empty(predator_type_specialization).get("differential_activation")
    )
    specialization_delta_rows = []
    if differential_activation:
        specialization_delta_rows = [
            (
                "visual_cortex (visual - olfactory)",
                f"{_coerce_float(differential_activation.get('visual_cortex_visual_minus_olfactory')):.2f}",
            ),
            (
                "sensory_cortex (olfactory - visual)",
                f"{_coerce_float(differential_activation.get('sensory_cortex_olfactory_minus_visual')):.2f}",
            ),
        ]

    representation_divergence_rows = []
    representation_action_center_rows = []
    representation_summary_line = (
        "_No representation-specialization metrics were available._"
    )
    representation_chart_line = "Chart: `representation_specialization.svg`."
    representation_section = _mapping_or_empty(representation_specialization)
    if representation_section and bool(representation_section.get("available")):
        proposer_divergence = _normalize_float_map(
            representation_section.get("proposer_divergence")
        )
        for module_name, score in sorted(proposer_divergence.items()):
            representation_divergence_rows.append(
                (
                    str(module_name),
                    f"{_coerce_float(score):.2f}",
                    _representation_interpretation(_coerce_float(score)),
                )
            )
        gate_differential = _normalize_float_map(
            representation_section.get("action_center_gate_differential")
        )
        contribution_differential = _normalize_float_map(
            representation_section.get("action_center_contribution_differential")
        )
        differential_modules = sorted(
            set(gate_differential) | set(contribution_differential)
        )
        for module_name in differential_modules:
            representation_action_center_rows.append(
                (
                    str(module_name),
                    f"{_coerce_float(gate_differential.get(module_name)):.2f}",
                    f"{_coerce_float(contribution_differential.get(module_name)):.2f}",
                )
            )
        representation_summary_line = (
            "Representation specialization score: "
            f"{_coerce_float(representation_section.get('representation_specialization_score')):.2f} "
            f"({representation_section.get('interpretation') or 'unknown'!s}). "
            f"Source: {representation_section.get('source') or 'unknown'!s}."
        )

    robustness_rows = []
    if robustness_train_conditions and robustness_eval_conditions:
        for train_condition in robustness_train_conditions:
            row = [train_condition]
            for eval_condition in robustness_eval_conditions:
                row.append(
                    _format_optional_metric(
                        _noise_robustness_rate(
                            robustness_matrix,
                            train_condition=train_condition,
                            eval_condition=eval_condition,
                        )
                    )
                )
            row.append(
                _format_optional_metric(
                    _noise_robustness_marginal(
                        robustness_train_marginals,
                        train_condition,
                    )
                )
            )
            robustness_rows.append(tuple(row))
        marginal_row = ["mean"]
        for eval_condition in robustness_eval_conditions:
            marginal_row.append(
                _format_optional_metric(
                    _noise_robustness_marginal(
                        robustness_eval_marginals,
                        eval_condition,
                    )
                )
            )
        marginal_row.append(
            _format_optional_metric(
                _coerce_optional_float(noise_robustness.get("robustness_score"))
            )
        )
        robustness_rows.append(tuple(marginal_row))

    markdown_lines = [
        "# Offline Analysis Report",
        "",
        "## Inputs",
        "",
        f"- `summary`: {report['inputs']['summary_path']}",
        f"- `trace`: {report['inputs']['trace_path']}",
        f"- `behavior_csv`: {report['inputs']['behavior_csv_path']}",
        "",
        "## Diagnostics",
        "",
        _markdown_table(diagnostic_rows, ("metric", "value")),
        "",
        "## Primary Benchmark",
        "",
        _markdown_table(
            primary_benchmark_rows,
            ("metric", "scenario_success_rate", "eval_reflex_scale", "reference_variant", "source"),
        ),
        "",
        "## Benchmark-of-Record Summary",
        "",
        _markdown_table(
            aggregate_primary_rows,
            (
                "primary_metric",
                "value",
                "ci_lower",
                "ci_upper",
                "n_seeds",
                "reference_variant",
                "source",
            ),
        ),
        "",
        "Per-scenario success rates with CI[^ci-method]:",
        "",
        _markdown_table(
            aggregate_scenario_rows,
            ("scenario", "success_rate", "ci_lower", "ci_upper", "n_seeds"),
        ),
        "",
        "Learning-evidence deltas with CI[^ci-method]:",
        "",
        _markdown_table(
            aggregate_learning_rows,
            ("comparison", "metric", "delta", "ci_lower", "ci_upper", "n_seeds"),
        ),
        "",
        "## Claim Test Results with Uncertainty",
        "",
        _markdown_table(
            claim_uncertainty_rows,
            (
                "claim",
                "role",
                "condition",
                "metric",
                "value",
                "ci_lower",
                "ci_upper",
                "n_seeds",
                "cohens_d",
                "magnitude",
            ),
        ),
        "",
        "## Effect Sizes Against Baselines",
        "",
        "Cohen's d and magnitude labels are reported for the main baselines[^effect-size].",
        "",
        _markdown_table(
            effect_size_rows,
            (
                "domain",
                "baseline",
                "comparison",
                "metric",
                "raw_delta",
                "delta_ci_lower",
                "delta_ci_upper",
                "cohens_d",
                "magnitude",
                "effect_ci_lower",
                "effect_ci_upper",
                "n_seeds",
            ),
        ),
        "",
        "## Shaping Minimization Program",
        "",
    ]
    if shaping_warning_line:
        markdown_lines.extend([shaping_warning_line, ""])
    markdown_lines.extend(
        [
            "### Dense vs Minimal Gap",
            "",
            _markdown_table(
                shaping_gap_rows,
                (
                    "comparison",
                    "scenario_success_rate_delta",
                    "episode_success_rate_delta",
                    "mean_reward_delta",
                    "interpretation",
                ),
            ),
            "",
            "### Profile-Level Summary",
            "",
            shaping_removed_gap_line,
            "",
            "Profile disposition weight proxies:",
            "",
            _markdown_table(
                shaping_profile_rows,
                ("profile", "disposition", "total_weight_proxy"),
            ),
            "",
            "Disposition component summary:",
            "",
            _markdown_table(
                shaping_disposition_rows,
                ("disposition", "component_count", "components"),
            ),
            "",
            "### Component Dispositions",
            "",
            _markdown_table(
                shaping_component_rows,
                ("Component", "Category", "Risk", "Disposition", "Rationale"),
            ),
            "",
            "### Behavior Survival",
            "",
            shaping_summary_line,
            "",
            _markdown_table(
                shaping_survival_rows,
                ("scenario", "austere_success_rate", "survives", "episodes"),
            ),
            "",
            "## Scenario Success",
            "",
            _markdown_table(scenario_rows, ("scenario", "success_rate", "check_count")),
            "",
            "## Ablations",
            "",
            _markdown_table(
                ablation_rows,
                ("variant", "architecture", "eval_reflex_scale", "scenario_success_rate"),
            ),
            "",
            "Ablation predator-type comparisons:",
            "",
            _markdown_table(
                ablation_predator_type_rows,
                ("variant", "visual_scenarios", "olfactory_scenarios", "visual_minus_olfactory"),
            ),
            "",
            "## Predator Type Specialization",
            "",
            specialization_summary_line,
            "",
            _markdown_table(
                specialization_rows,
                (
                    "predator_type",
                    "visual_cortex_activation",
                    "sensory_cortex_activation",
                    "dominant_module",
                ),
            ),
            "",
            _markdown_table(
                specialization_delta_rows,
                ("differential", "value"),
            ),
            "",
            "## Representation Specialization",
            "",
            representation_summary_line,
            "",
            representation_chart_line,
            "",
            _markdown_table(
                representation_divergence_rows,
                ("module", "js_divergence", "interpretation"),
            ),
            "",
            _markdown_table(
                representation_action_center_rows,
                ("module", "gate_differential", "contribution_differential"),
            ),
            "",
            "## Noise Robustness Matrix",
            "",
            (
                _markdown_table(
                    robustness_rows,
                    ("train \\ eval", *robustness_eval_conditions, "mean"),
                )
                if robustness_rows
                else "_No noise robustness matrix was available._"
            ),
            "",
            "Overall robustness score: "
            f"{_format_optional_metric(_coerce_optional_float(_mapping_or_empty(noise_robustness).get('robustness_score')))}.",
            "Diagonal score: "
            f"{_format_optional_metric(_coerce_optional_float(_mapping_or_empty(noise_robustness).get('diagonal_score')))}.",
            "Off-diagonal score: "
            f"{_format_optional_metric(_coerce_optional_float(_mapping_or_empty(noise_robustness).get('off_diagonal_score')))}.",
            "",
            "## Limitations",
            "",
        ]
    )
    limitations = report.get("limitations", [])
    if isinstance(limitations, list) and limitations:
        markdown_lines.extend(f"- {item}" for item in limitations)
    else:
        markdown_lines.append("- None.")
    markdown_lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "[^ci-method]: Confidence intervals use percentile bootstrap resampling over seed-level metric values at the reported confidence level; benchmark-of-record tables default to 95%.",
            "[^effect-size]: Cohen's d uses the pooled sample standard deviation. Magnitude labels follow common absolute-value thresholds: negligible < 0.2, small < 0.5, medium < 0.8, and large otherwise.",
            "",
            "## Generated Files",
            "",
            "- `training_eval.svg`",
            "- `scenario_success.svg`",
            "- `robustness_matrix.svg`",
            "- `representation_specialization.svg`",
            "- `scenario_checks.csv`",
            "- `reward_components.csv`",
            "- `report.json`",
        ]
    )
    if ablation_svg_path:
        markdown_lines.append("- `ablation_comparison.svg`")
    if reflex_svg_path:
        markdown_lines.append("- `reflex_frequency.svg`")

    report_md_path = output_path / "report.md"
    report_md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {
        "report_md": str(report_md_path),
        "report_json": str(report_json_path),
        "training_eval_svg": str(training_svg_path),
        "scenario_success_svg": str(scenario_svg_path),
        "robustness_matrix_svg": str(robustness_svg_path),
        "representation_specialization_svg": str(representation_svg_path),
        "scenario_checks_csv": str(scenario_checks_path),
        "reward_components_csv": str(reward_components_path),
        "ablation_comparison_svg": ablation_svg_path,
        "reflex_frequency_svg": reflex_svg_path,
    }
