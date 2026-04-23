from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .constants import (
    LADDER_ACTIVE_RUNGS,
    LADDER_ADJACENT_COMPARISONS,
    LADDER_PROTOCOL_NAMES,
    LADDER_RUNG_DESCRIPTIONS,
    LADDER_RUNG_MAPPING,
    SHAPING_DEPENDENCE_WARNING_THRESHOLD,
)
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
from .plots import build_capacity_comparison_plot
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


def _section_artifact_state(section: Mapping[str, object]) -> str:
    if bool(section.get("available")):
        return "available"
    return str(section.get("artifact_state") or "expected_but_missing")


def _unavailable_section_message(
    section: Mapping[str, object],
    *,
    expected_but_missing: str,
    not_in_artifact: str,
) -> str:
    return (
        not_in_artifact
        if _section_artifact_state(section) == "not_in_this_artifact"
        else expected_but_missing
    )


def _markdown_table_or_message(
    rows: Sequence[Sequence[object]],
    columns: Sequence[str],
    *,
    section: Mapping[str, object],
    expected_but_missing: str,
    not_in_artifact: str,
) -> str:
    if rows:
        return _markdown_table(rows, columns)
    return _unavailable_section_message(
        section,
        expected_but_missing=expected_but_missing,
        not_in_artifact=not_in_artifact,
    )


def write_report(output_dir: str | Path, report: Mapping[str, object]) -> dict[str, str]:
    """
    Write filesystem artifacts (SVG, CSV, JSON, and Markdown) that represent the provided analysis report.
    
    Parameters:
        output_dir (str | Path): Target directory where artifact files will be created; created if missing.
        report (Mapping[str, object]): Report data used to populate charts, tables, and markdown. Expected keys include (but are not limited to) "training_eval", "scenario_success", "shaping_program", "ablations", "ladder_comparison", "ladder_profile_comparison", "reflex_frequency", "scenario_checks", "reward_components", "diagnostics", "primary_benchmark", "limitations", and "inputs".
    
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
    ladder_comparison = _mapping_or_empty(report.get("ladder_comparison"))
    ladder_profile_comparison = _mapping_or_empty(
        report.get("ladder_profile_comparison")
    )
    unified_ladder_report = _mapping_or_empty(report.get("unified_ladder_report"))
    predator_type_specialization = report.get("predator_type_specialization", {})
    representation_specialization = report.get("representation_specialization", {})
    capacity_analysis = _mapping_or_empty(report.get("capacity_analysis"))
    model_capacity = _mapping_or_empty(report.get("model_capacity"))
    capacity_sweeps = _mapping_or_empty(report.get("capacity_sweeps"))
    reflex_frequency = report.get("reflex_frequency", {})
    credit_assignment = _mapping_or_empty(report.get("credit_assignment"))
    credit_analysis = _mapping_or_empty(report.get("credit_analysis"))
    aggregate_benchmark_tables = _mapping_or_empty(
        report.get("aggregate_benchmark_tables")
    )
    claim_test_tables = _mapping_or_empty(report.get("claim_test_tables"))
    effect_size_tables = _mapping_or_empty(report.get("effect_size_tables"))
    module_local_sufficiency = _mapping_or_empty(
        report.get("module_local_sufficiency")
    )
    distillation_analysis = _mapping_or_empty(report.get("distillation_analysis"))
    if not module_local_sufficiency:
        module_local_sufficiency = {
            "available": False,
            "artifact_state": "not_in_this_artifact",
        }
    if not distillation_analysis:
        distillation_analysis = {
            "available": False,
            "artifact_state": "not_in_this_artifact",
        }

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

    capacity_comparison_svg_path = ""
    capacity_sweep_rows = (
        capacity_sweeps.get("rows", [])
        if isinstance(capacity_sweeps.get("rows"), list)
        else []
    )
    if capacity_sweep_rows:
        capacity_comparison_plot = build_capacity_comparison_plot(capacity_sweep_rows)
        capacity_comparison_svg = str(capacity_comparison_plot.get("svg") or "")
        if capacity_comparison_plot.get("available") and capacity_comparison_svg:
            capacity_comparison_svg_path = str(output_path / "capacity_comparison.svg")
            _write_svg(
                Path(capacity_comparison_svg_path),
                capacity_comparison_svg,
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
    combined_inputs = _mapping_or_empty(report.get("combined_inputs"))
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
    model_capacity_rows = []
    for item in (
        model_capacity.get("networks", [])
        if isinstance(model_capacity.get("networks"), list)
        else []
    ):
        if not isinstance(item, Mapping):
            continue
        model_capacity_rows.append(
            (
                str(item.get("network") or ""),
                str(int(_coerce_float(item.get("parameters"), 0.0))),
                _markdown_value(item.get("proportion")),
            )
        )
    aggregate_architecture_rows = []
    aggregate_architecture_table = _mapping_or_empty(
        aggregate_benchmark_tables.get("architecture_capacity")
    )
    for item in _table_rows(aggregate_architecture_table):
        if not isinstance(item, Mapping):
            continue
        aggregate_architecture_rows.append(
            (
                str(item.get("variant") or ""),
                str(item.get("architecture") or ""),
                str(int(_coerce_float(item.get("total_trainable"), 0.0))),
                str(item.get("key_components") or ""),
                str(item.get("capacity_status") or ""),
                _markdown_value(item.get("total_ratio_vs_reference")),
            )
        )
    capacity_reference_variant = ""
    architecture_capacity_rows = _table_rows(aggregate_architecture_table)
    if architecture_capacity_rows:
        capacity_reference_variant = str(
            architecture_capacity_rows[0].get("reference_variant") or ""
        )
    capacity_sweep_curve_rows = []
    aggregate_capacity_sweep_table = _mapping_or_empty(
        aggregate_benchmark_tables.get("capacity_sweep_curves")
    )
    for item in _table_rows(aggregate_capacity_sweep_table):
        if not isinstance(item, Mapping):
            continue
        capacity_sweep_curve_rows.append(
            (
                str(item.get("variant") or ""),
                str(item.get("capacity_profile") or ""),
                _markdown_value(item.get("scale_factor")),
                str(int(_coerce_float(item.get("total_trainable"), 0.0))),
                str(int(_coerce_float(item.get("approximate_compute_cost_total"), 0.0))),
                _markdown_value(item.get("scenario_success_rate")),
                _markdown_value(item.get("capability_probe_success_rate")),
            )
        )
    capacity_sweep_interpretation_rows = [
        (
            str(item.get("variant") or ""),
            str(item.get("status") or ""),
            str(item.get("interpretation") or ""),
        )
        for item in (
            capacity_sweeps.get("interpretations", [])
            if isinstance(capacity_sweeps.get("interpretations"), list)
            else []
        )
        if isinstance(item, Mapping)
    ]
    if capacity_sweep_curve_rows:
        capacity_sweep_summary_block = _markdown_table(
            capacity_sweep_curve_rows,
            (
                "variant",
                "capacity_profile",
                "scale_factor",
                "total_trainable",
                "approx_compute",
                "scenario_success_rate",
                "capability_probe_success_rate",
            ),
        )
    elif capacity_sweep_interpretation_rows:
        capacity_sweep_summary_block = (
            "_Raw capacity-sweep curve rows were not exported in this artifact; "
            "fallback interpretations are reported below._"
        )
    else:
        capacity_sweep_summary_block = _unavailable_section_message(
            capacity_sweeps,
            expected_but_missing="_Capacity-sweep payload was expected in this artifact but missing._",
            not_in_artifact="_Capacity-sweep payload was not included in this artifact._",
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

    ladder_description_rows = []
    ladder_comparison_rows = []
    ladder_summary_base = (
        "The ladder validation asks whether gains or regressions come from adding "
        "the decision-execution pipeline itself or from later sensory/proposer modularization."
    )
    ladder_summary_line = ladder_summary_base
    ladder_rungs_payload = _mapping_or_empty(_mapping_or_empty(ablations).get("ladder_rungs"))
    recognized_rungs = ladder_rungs_payload.get("recognized_rungs", [])
    recognized_rung_set = (
        {str(item) for item in recognized_rungs}
        if isinstance(recognized_rungs, list)
        else set()
    )
    for rung in LADDER_ACTIVE_RUNGS:
        if rung not in recognized_rung_set:
            continue
        ladder_description_rows.append(
            (
                rung,
                LADDER_PROTOCOL_NAMES.get(rung, rung),
                LADDER_RUNG_DESCRIPTIONS.get(rung, ""),
            )
        )
    ladder_effect_size_lookup: dict[tuple[str, str], Mapping[str, object]] = {}
    for item in _table_rows(effect_size_table):
        if not isinstance(item, Mapping):
            continue
        if str(item.get("domain") or "") != "ladder":
            continue
        ladder_effect_size_lookup[
            (
                str(item.get("baseline") or ""),
                str(item.get("comparison") or ""),
            )
        ] = item
    comparison_items = ladder_comparison.get("comparisons", [])
    produced_pair_labels: set[str] = set()
    if isinstance(comparison_items, list):
        for item in comparison_items:
            if not isinstance(item, Mapping):
                continue
            produced_pair_labels.add(
                f"{str(item.get('baseline_rung') or '')}->{str(item.get('comparison_rung') or '')}"
            )
            metrics_payload = _mapping_or_empty(item.get("metrics"))
            deltas_payload = _mapping_or_empty(item.get("deltas"))
            baseline_variant = str(metrics_payload.get("baseline_variant") or "")
            comparison_variant = str(metrics_payload.get("comparison_variant") or "")
            effect_row = ladder_effect_size_lookup.get(
                (baseline_variant, comparison_variant),
                {},
            )
            effect_size = _coerce_optional_float(effect_row.get("cohens_d"))
            effect_magnitude = str(effect_row.get("magnitude_label") or "")
            scenario_delta = _coerce_float(
                deltas_payload.get("scenario_success_rate_delta")
            )
            interpretation = (
                (
                    f"{'positive' if scenario_delta >= 0.0 else 'negative'} "
                    f"{effect_magnitude or 'unlabeled'} effect"
                )
                if effect_size is not None
                else "Effect size unavailable"
            )
            ladder_comparison_rows.append(
                (
                    str(item.get("baseline_rung") or ""),
                    str(item.get("comparison_rung") or ""),
                    f"{scenario_delta:.2f}",
                    f"{effect_size:.2f}" if effect_size is not None else "n/a",
                    interpretation,
                )
            )
    present_variants = {
        str(variant_name)
        for variant_name in _mapping_or_empty(ablations.get("variants")).keys()
    }
    required_variants = {
        variant_name
        for variant_name, rung in LADDER_RUNG_MAPPING.items()
        if rung in set(LADDER_ACTIVE_RUNGS)
    }
    missing_variants = sorted(required_variants - present_variants)
    expected_pair_labels = {
        f"{baseline_rung}->{comparison_rung}"
        for baseline_rung, comparison_rung in LADDER_ADJACENT_COMPARISONS
    }
    missing_pair_labels = sorted(expected_pair_labels - produced_pair_labels)
    ladder_limitations = [
        str(item)
        for item in ladder_comparison.get("limitations", [])
        if item
    ]
    if missing_variants or missing_pair_labels or ladder_limitations:
        summary_parts = [ladder_summary_base]
        if missing_variants:
            summary_parts.append(
                "Missing variants: "
                + ", ".join(f"`{name}`" for name in missing_variants)
                + "."
            )
        if missing_pair_labels:
            summary_parts.append(
                "Missing adjacent ladder pairs: "
                + ", ".join(f"`{label}`" for label in missing_pair_labels)
                + "."
            )
        if ladder_limitations:
            summary_parts.append(
                "Notes: " + " ".join(ladder_limitations)
            )
        ladder_summary_line = " ".join(summary_parts)

    ladder_profile_summary_rows = []
    ladder_profile_gap_rows = []
    ladder_profile_summary_line = _unavailable_section_message(
        ladder_profile_comparison,
        expected_but_missing="_Cross-profile architectural ladder comparison was expected in this artifact but missing._",
        not_in_artifact="_Cross-profile architectural ladder comparison was not included in this artifact._",
    )
    ladder_profile_rows_payload = _mapping_or_empty(
        ladder_profile_comparison.get("classification_summary")
    ).get("rows", [])
    if (
        bool(ladder_profile_comparison.get("available"))
        and isinstance(ladder_profile_rows_payload, list)
        and ladder_profile_rows_payload
    ):
        ladder_profile_summary_line = (
            "The cross-profile ladder re-runs A0-A4 under `classic`, "
            "`ecological`, and `austere` while preserving no-reflex "
            "`scenario_success_rate` as the benchmark surface. "
            "Classifications use the four-term taxonomy: "
            "`shaping_dependent`, `austere_survivor`, "
            "`fails_with_shaping`, and `mixed_or_borderline`."
        )
        for item in ladder_profile_rows_payload:
            if not isinstance(item, Mapping):
                continue
            ladder_profile_summary_rows.append(
                (
                    str(item.get("protocol_name") or ""),
                    str(item.get("classification") or ""),
                    _markdown_value(item.get("classic_success_rate")),
                    _markdown_value(item.get("ecological_success_rate")),
                    _markdown_value(item.get("austere_success_rate")),
                    _markdown_value(item.get("austere_competence_gap")),
                    str(item.get("reason") or ""),
                )
            )
            ladder_profile_gap_rows.append(
                (
                    str(item.get("protocol_name") or ""),
                    _markdown_value(item.get("classic_minus_austere")),
                    _markdown_value(item.get("classic_gap_ci_lower")),
                    _markdown_value(item.get("classic_gap_ci_upper")),
                    _markdown_value(item.get("classic_gap_effect_size")),
                    str(int(_coerce_float(item.get("classic_gap_n_seeds"), 0.0))),
                    _markdown_value(item.get("ecological_minus_austere")),
                    _markdown_value(item.get("ecological_gap_ci_lower")),
                    _markdown_value(item.get("ecological_gap_ci_upper")),
                    _markdown_value(item.get("ecological_gap_effect_size")),
                    str(int(_coerce_float(item.get("ecological_gap_n_seeds"), 0.0))),
                )
            )

    ladder_report_summary_line = _unavailable_section_message(
        unified_ladder_report,
        expected_but_missing="_Unified architectural ladder report was expected in this artifact but missing._",
        not_in_artifact="_Unified architectural ladder report was not included in this artifact._",
    )
    ladder_report_rows = []
    ladder_report_adjacent_rows = []
    ladder_report_capacity_rows = []
    ladder_report_credit_rows = []
    ladder_report_shaping_rows = []
    ladder_report_no_reflex_rows = []
    ladder_report_capability_rows = []
    ladder_report_conclusion_rows = []
    ladder_report_missing_rows = []
    ladder_report_limitation_lines = ["- None."]
    if unified_ladder_report and bool(unified_ladder_report.get("available")):
        conclusion = str(
            unified_ladder_report.get("conclusion") or "modularity inconclusive"
        )
        rationale = str(unified_ladder_report.get("conclusion_rationale") or "")
        ladder_report_summary_line = (
            f"Conclusion: `{conclusion}`. {rationale}".strip()
        )
        ladder_report_tables = _mapping_or_empty(unified_ladder_report.get("tables"))

        for item in _table_rows(
            _mapping_or_empty(ladder_report_tables.get("primary_rung_table"))
        ):
            if not isinstance(item, Mapping):
                continue
            ladder_report_rows.append(
                (
                    str(item.get("rung") or ""),
                    str(item.get("protocol_name") or ""),
                    str(item.get("variant") or ""),
                    str(int(_coerce_float(item.get("total_parameters"), 0.0))),
                    _markdown_value(item.get("scenario_success_rate")),
                    _markdown_value(item.get("ci_lower")),
                    _markdown_value(item.get("ci_upper")),
                    _markdown_value(item.get("effect_size_vs_a0")),
                    str(item.get("effect_magnitude") or ""),
                    str(int(_coerce_float(item.get("n_seeds"), 0.0))),
                )
            )
        for item in _table_rows(
            _mapping_or_empty(ladder_report_tables.get("adjacent_comparison_table"))
        ):
            if not isinstance(item, Mapping):
                continue
            ladder_report_adjacent_rows.append(
                (
                    str(item.get("baseline_rung") or ""),
                    str(item.get("comparison_rung") or ""),
                    _markdown_value(item.get("delta")),
                    _markdown_value(item.get("ci_lower")),
                    _markdown_value(item.get("ci_upper")),
                    _markdown_value(item.get("cohens_d")),
                    str(item.get("effect_magnitude") or ""),
                    str(item.get("interpretation") or ""),
                )
            )
        for item in _table_rows(
            _mapping_or_empty(ladder_report_tables.get("capacity_summary_table"))
        ):
            if not isinstance(item, Mapping):
                continue
            ladder_report_capacity_rows.append(
                (
                    "yes" if bool(item.get("capacity_matched")) else "no",
                    _markdown_value(item.get("ratio")),
                    str(item.get("largest_variant") or ""),
                    str(item.get("smallest_variant") or ""),
                )
            )
        for item in _table_rows(
            _mapping_or_empty(ladder_report_tables.get("credit_assignment_summary"))
        ):
            if not isinstance(item, Mapping):
                continue
            ladder_report_credit_rows.append(
                (
                    str(item.get("rung") or ""),
                    _markdown_value(item.get("local_only_delta_vs_broadcast")),
                    _markdown_value(item.get("counterfactual_delta_vs_broadcast")),
                    str(item.get("strategies_present") or ""),
                )
            )
        for item in _table_rows(
            _mapping_or_empty(
                ladder_report_tables.get("reward_shaping_sensitivity_summary")
            )
        ):
            if not isinstance(item, Mapping):
                continue
            ladder_report_shaping_rows.append(
                (
                    str(item.get("rung") or ""),
                    _markdown_value(item.get("classic_minus_austere")),
                    _markdown_value(item.get("classic_effect_size")),
                    _markdown_value(item.get("ecological_minus_austere")),
                    _markdown_value(item.get("ecological_effect_size")),
                )
            )
        for item in _table_rows(
            _mapping_or_empty(ladder_report_tables.get("no_reflex_competence_summary"))
        ):
            if not isinstance(item, Mapping):
                continue
            ladder_report_no_reflex_rows.append(
                (
                    str(item.get("rung") or ""),
                    "yes" if bool(item.get("no_reflex_evaluated")) else "no",
                    _markdown_value(item.get("eval_reflex_scale")),
                    _markdown_value(item.get("scenario_success_rate")),
                )
            )
        for item in _table_rows(
            _mapping_or_empty(ladder_report_tables.get("capability_probe_boundaries"))
        ):
            if not isinstance(item, Mapping):
                continue
            ladder_report_capability_rows.append(
                (
                    str(item.get("scenario") or ""),
                    str(item.get("requirement_level") or ""),
                    _markdown_value(item.get("first_competent_rung")),
                    _markdown_value(item.get("highest_competent_rung")),
                    str(item.get("rationale") or ""),
                )
            )
        for item in _table_rows(
            _mapping_or_empty(ladder_report_tables.get("conclusion_table"))
        ):
            if not isinstance(item, Mapping):
                continue
            ladder_report_conclusion_rows.append(
                (
                    str(item.get("conclusion") or ""),
                    str(item.get("rationale") or ""),
                    _markdown_value(item.get("confidence_level")),
                    str(item.get("confounding_factors") or ""),
                )
            )
        for item in _table_rows(
            _mapping_or_empty(ladder_report_tables.get("missing_experiments_table"))
        ):
            if not isinstance(item, Mapping):
                continue
            ladder_report_missing_rows.append(
                (
                    str(item.get("experiment") or ""),
                    str(item.get("description") or ""),
                    str(item.get("priority") or ""),
                )
            )
        limitations = unified_ladder_report.get("limitations", [])
        if isinstance(limitations, list) and limitations:
            ladder_report_limitation_lines = [
                f"- {str(item)}"
                for item in limitations
                if item
            ]
            if not ladder_report_limitation_lines:
                ladder_report_limitation_lines = ["- No limitations reported."]

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
        architecture_capacity_by_variant = {
            str(item.get("variant") or ""): item
            for item in architecture_capacity_rows
            if isinstance(item, Mapping)
        }
        if isinstance(variants_payload, Mapping):
            for variant_name, payload in sorted(variants_payload.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_payload = payload.get("summary", {})
                config_payload = payload.get("config", {})
                capacity_payload = _mapping_or_empty(
                    architecture_capacity_by_variant.get(str(variant_name))
                )
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
                        str(capacity_payload.get("capacity_status") or ""),
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

    credit_strategy_rows = []
    for item in _table_rows(
        _mapping_or_empty(credit_analysis.get("strategy_comparison_table"))
    ):
        if not isinstance(item, Mapping):
            continue
        credit_strategy_rows.append(
            (
                str(item.get("rung") or item.get("architecture_rung") or ""),
                str(item.get("credit_strategy") or ""),
                _markdown_value(item.get("scenario_success_rate")),
                _markdown_value(item.get("scenario_success_delta_vs_broadcast")),
                str(item.get("dominant_module") or ""),
            )
        )
    credit_architecture_rows = []
    for item in _table_rows(
        _mapping_or_empty(credit_analysis.get("architecture_strategy_matrix"))
    ):
        if not isinstance(item, Mapping):
            continue
        credit_architecture_rows.append(
            (
                str(item.get("architecture_rung") or ""),
                str(item.get("credit_strategy") or ""),
                str(item.get("variant") or ""),
                _markdown_value(item.get("scenario_success_rate")),
                _markdown_value(item.get("mean_effective_module_count")),
            )
        )
    credit_module_rows = []
    for item in _table_rows(_mapping_or_empty(credit_assignment.get("module_credit"))):
        if not isinstance(item, Mapping):
            continue
        credit_module_rows.append(
            (
                str(item.get("rung") or ""),
                str(item.get("credit_strategy") or ""),
                str(item.get("module") or ""),
                _markdown_value(item.get("mean_module_credit_weight")),
                _markdown_value(item.get("mean_counterfactual_credit_weight")),
                _markdown_value(item.get("mean_module_gradient_norm")),
                _markdown_value(item.get("dominant_module_rate")),
            )
        )
    credit_scenario_rows = []
    for item in _table_rows(_mapping_or_empty(credit_assignment.get("scenario_success"))):
        if not isinstance(item, Mapping):
            continue
        credit_scenario_rows.append(
            (
                str(item.get("rung") or ""),
                str(item.get("credit_strategy") or ""),
                str(item.get("scenario") or ""),
                _markdown_value(item.get("success_rate")),
            )
        )
    credit_interpretation_lines = [
        (
            "- "
            + " / ".join(
                part
                for part in (
                    str(item.get("scope") or ""),
                    str(item.get("strategy") or ""),
                    str(item.get("pattern") or ""),
                )
                if part
            )
            + f": {str(item.get('interpretation') or '')}"
        )
        for item in (
            credit_analysis.get("findings", [])
            if isinstance(credit_analysis.get("findings"), list)
            else []
        )
        if isinstance(item, Mapping)
        and (item.get("pattern") or item.get("interpretation"))
    ]
    if not credit_interpretation_lines:
        credit_interpretation_lines = [
            _unavailable_section_message(
                credit_analysis,
                expected_but_missing="_Credit diagnostics were expected in this artifact but missing._",
                not_in_artifact="_Credit diagnostics were not included in this artifact._",
            )
        ]

    module_local_rows = []
    module_local_summary_line = ""
    if module_local_sufficiency and bool(module_local_sufficiency.get("available")):
        partial_coverage = [
            str(name)
            for name in module_local_sufficiency.get("partial_variant_coverage", [])
            if name
        ]
        summary_parts = [
            "Paper gate: "
            + (
                "pass"
                if bool(module_local_sufficiency.get("paper_gate_pass"))
                else "blocked"
            )
            + "."
        ]
        if partial_coverage:
            summary_parts.append(
                "Partial variant coverage: "
                + ", ".join(f"`{name}`" for name in partial_coverage)
                + "."
            )
        blocked_reasons = [
            str(reason)
            for reason in module_local_sufficiency.get("blocked_reasons", [])
            if reason
        ]
        if blocked_reasons:
            summary_parts.append("Blocked reasons: " + " ".join(blocked_reasons))
        module_local_summary_line = " ".join(summary_parts)
        for item in module_local_sufficiency.get("rows", []):
            if not isinstance(item, Mapping):
                continue
            module_local_rows.append(
                (
                    str(item.get("module") or ""),
                    str(item.get("coverage_mode") or ""),
                    str(item.get("seeds") or ""),
                    _markdown_value(item.get("minimal_sufficient_level")),
                    "yes" if bool(item.get("canonical_v4_pass")) else "no",
                    "yes" if bool(item.get("partial_variant_coverage")) else "no",
                )
            )

    distillation_rows = []
    distillation_summary_line = ""
    distillation_assessment = _mapping_or_empty(
        distillation_analysis.get("assessment")
    )
    if distillation_analysis and bool(distillation_analysis.get("available")):
        answer = str(distillation_assessment.get("answer") or "unknown")
        rationale = str(distillation_assessment.get("rationale") or "")
        distillation_summary_line = f"Assessment: `{answer}`. {rationale}".strip()
        for item in distillation_analysis.get("rows", []):
            if not isinstance(item, Mapping):
                continue
            distillation_rows.append(
                (
                    str(item.get("condition") or ""),
                    str(int(_coerce_float(item.get("episodes"), 0.0))),
                    _markdown_value(item.get("survival_rate")),
                    _markdown_value(item.get("mean_reward")),
                    _markdown_value(item.get("mean_food_distance_delta")),
                )
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
        *(
            [
                f"- `ablation_summary`: {combined_inputs.get('ablation_summary_path')}",
                f"- `ablation_behavior_csv`: {combined_inputs.get('ablation_behavior_csv_path')}",
                f"- `profile_summary`: {combined_inputs.get('profile_summary_path')}",
                f"- `profile_behavior_csv`: {combined_inputs.get('profile_behavior_csv_path')}",
                f"- `capacity_summary`: {combined_inputs.get('capacity_summary_path')}",
                f"- `capacity_behavior_csv`: {combined_inputs.get('capacity_behavior_csv_path')}",
                f"- `module_local_report`: {combined_inputs.get('module_local_report_path')}",
                f"- `distillation_summary`: {combined_inputs.get('distillation_summary_path')}",
            ]
            if combined_inputs
            else []
        ),
        "",
        "## Diagnostics",
        "",
        _markdown_table(diagnostic_rows, ("metric", "value")),
        "",
        "## Architecture Capacity",
        "",
        (
            "Current run "
            f"({model_capacity.get('variant') or 'unknown'} / "
            f"{model_capacity.get('architecture') or 'unknown'}): "
            f"{int(_coerce_float(model_capacity.get('total_trainable'), 0.0))} "
            "trainable parameters."
            if model_capacity.get("available")
            else _unavailable_section_message(
                model_capacity,
                expected_but_missing="_Trainable-parameter payload was expected in this artifact but missing._",
                not_in_artifact="_Trainable-parameter payload was not included in this artifact._",
            )
        ),
        "",
        (
            "Capacity status: "
            f"{capacity_analysis.get('status') or 'unavailable'} "
            f"(ratio {_markdown_value(capacity_analysis.get('ratio'))})."
            if capacity_analysis.get("available")
            else "Capacity status: unavailable."
        ),
        "",
        _markdown_table(
            model_capacity_rows,
            ("network", "parameters", "proportion"),
        ),
        "",
        (
            "Capacity comparison against the reference variant "
            f"`{capacity_reference_variant}`:"
            if capacity_reference_variant
            else "Capacity comparison:"
        ),
        "",
        _markdown_table(
            aggregate_architecture_rows,
            (
                "variant",
                "architecture",
                "total_trainable",
                "key_components",
                "capacity_status",
                "ratio_vs_reference",
            ),
        ),
        "",
        "## Capacity Sweep",
        "",
        capacity_sweep_summary_block,
        "",
        _markdown_table(
            capacity_sweep_interpretation_rows,
            ("variant", "status", "interpretation"),
        ),
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
        _markdown_table_or_message(
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
            section=aggregate_benchmark_tables,
            expected_but_missing="_Benchmark-of-record summary rows were expected in this artifact but missing._",
            not_in_artifact="_Benchmark-of-record summary rows were not included in this artifact._",
        ),
        "",
        "Per-scenario success rates with CI[^ci-method]:",
        "",
        _markdown_table_or_message(
            aggregate_scenario_rows,
            ("scenario", "success_rate", "ci_lower", "ci_upper", "n_seeds"),
            section=aggregate_benchmark_tables,
            expected_but_missing="_Per-scenario benchmark rows were expected in this artifact but missing._",
            not_in_artifact="_Per-scenario benchmark rows were not included in this artifact._",
        ),
        "",
        "Learning-evidence deltas with CI[^ci-method]:",
        "",
        _markdown_table_or_message(
            aggregate_learning_rows,
            ("comparison", "metric", "delta", "ci_lower", "ci_upper", "n_seeds"),
            section=aggregate_benchmark_tables,
            expected_but_missing="_Learning-evidence benchmark rows were expected in this artifact but missing._",
            not_in_artifact="_Learning-evidence benchmark rows were not included in this artifact._",
        ),
        "",
        "## Claim Test Results with Uncertainty",
        "",
        _markdown_table_or_message(
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
            section=claim_test_tables,
            expected_but_missing="_Claim-test uncertainty rows were expected in this artifact but missing._",
            not_in_artifact="_Claim-test uncertainty rows were not included in this artifact._",
        ),
        "",
        "## Effect Sizes Against Baselines",
        "",
        "Cohen's d and magnitude labels are reported for the main baselines[^effect-size].",
        "",
        _markdown_table_or_message(
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
            section=effect_size_tables,
            expected_but_missing="_Effect-size rows were expected in this artifact but missing._",
            not_in_artifact="_Effect-size rows were not included in this artifact._",
        ),
        "",
        "## Architectural Ladder Comparison",
        "",
        ladder_summary_line,
        "",
        _markdown_table(
            ladder_description_rows,
            (
                "rung",
                "protocol_name",
                "experimental_isolation_question",
            ),
        ),
        "",
        "Pairwise ladder comparisons:",
        "",
        _markdown_table(
            ladder_comparison_rows,
            (
                "baseline_rung",
                "comparison_rung",
                "scenario_success_rate_delta",
                "effect_size",
                "interpretation",
            ),
        ),
        "",
        "## Cross-Profile Ladder Comparison",
        "",
        ladder_profile_summary_line,
        "",
        _markdown_table(
            ladder_profile_summary_rows,
            (
                "protocol_name",
                "classification",
                "classic_success_rate",
                "ecological_success_rate",
                "austere_success_rate",
                "austere_competence_gap",
                "interpretation",
            ),
        ),
        "",
        "Scenario-success shaping gaps:",
        "",
        _markdown_table(
            ladder_profile_gap_rows,
            (
                "protocol_name",
                "classic_minus_austere",
                "classic_ci_lower",
                "classic_ci_upper",
                "classic_effect_size",
                "classic_n_seeds",
                "ecological_minus_austere",
                "ecological_ci_lower",
                "ecological_ci_upper",
                "ecological_effect_size",
                "ecological_n_seeds",
            ),
        ),
        "",
        "## Unified Architectural Ladder Report",
        "",
        ladder_report_summary_line,
        "",
        _markdown_table(
            ladder_report_rows,
            (
                "rung",
                "protocol_name",
                "variant",
                "total_parameters",
                "scenario_success_rate",
                "ci_lower",
                "ci_upper",
                "effect_size_vs_a0",
                "effect_magnitude",
                "n_seeds",
            ),
        ),
        "",
        "Adjacent rung comparisons:",
        "",
        _markdown_table(
            ladder_report_adjacent_rows,
            (
                "baseline_rung",
                "comparison_rung",
                "delta",
                "ci_lower",
                "ci_upper",
                "cohens_d",
                "effect_magnitude",
                "interpretation",
            ),
        ),
        "",
        "Capacity matching summary:",
        "",
        _markdown_table(
            ladder_report_capacity_rows,
            (
                "capacity_matched",
                "ratio",
                "largest_variant",
                "smallest_variant",
            ),
        ),
        "",
        "Credit assignment comparison summary:",
        "",
        _markdown_table(
            ladder_report_credit_rows,
            (
                "rung",
                "local_only_delta_vs_broadcast",
                "counterfactual_delta_vs_broadcast",
                "strategies_present",
            ),
        ),
        "",
        "Reward shaping sensitivity:",
        "",
        _markdown_table(
            ladder_report_shaping_rows,
            (
                "rung",
                "classic_minus_austere",
                "classic_effect_size",
                "ecological_minus_austere",
                "ecological_effect_size",
            ),
        ),
        "",
        "No-reflex competence:",
        "",
        _markdown_table(
            ladder_report_no_reflex_rows,
            (
                "rung",
                "no_reflex_evaluated",
                "eval_reflex_scale",
                "scenario_success_rate",
            ),
        ),
        "",
        "Capability probe boundaries:",
        "",
        _markdown_table(
            ladder_report_capability_rows,
            (
                "scenario",
                "requirement_level",
                "first_competent_rung",
                "highest_competent_rung",
                "rationale",
            ),
        ),
        "",
        "> Conclusion",
        "",
        _markdown_table(
            ladder_report_conclusion_rows,
            (
                "conclusion",
                "rationale",
                "confidence_level",
                "confounding_factors",
            ),
        ),
        "",
        "Missing experiments before asserting modular emergence:",
        "",
        _markdown_table(
            ladder_report_missing_rows,
            ("experiment", "description", "priority"),
        ),
        "",
        "Unified ladder limitations:",
        "",
        *ladder_report_limitation_lines,
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
                (
                    "variant",
                    "architecture",
                    "eval_reflex_scale",
                    "scenario_success_rate",
                    "capacity_status",
                ),
            ),
            "",
            "Ablation predator-type comparisons:",
            "",
            _markdown_table(
                ablation_predator_type_rows,
                ("variant", "visual_scenarios", "olfactory_scenarios", "visual_minus_olfactory"),
            ),
            "",
            "## Credit Assignment Analysis",
            "",
            _markdown_table(
                credit_strategy_rows,
                (
                    "rung",
                    "strategy",
                    "scenario_success_rate",
                    "delta_vs_broadcast",
                    "dominant_module",
                ),
            ),
            "",
            _markdown_table(
                credit_architecture_rows,
                (
                    "architecture_rung",
                    "strategy",
                    "variant",
                    "scenario_success_rate",
                    "effective_module_count",
                ),
            ),
            "",
            _markdown_table(
                credit_module_rows,
                (
                    "rung",
                    "strategy",
                    "module",
                    "module_credit_weight",
                    "counterfactual_credit_weight",
                    "module_gradient_norm",
                    "dominant_module_rate",
                ),
            ),
            "",
            _markdown_table(
                credit_scenario_rows,
                ("rung", "strategy", "scenario", "success_rate"),
            ),
            "",
            *credit_interpretation_lines,
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
            "## Module-Local Sufficiency",
            "",
            (
                module_local_summary_line
                if module_local_summary_line
                else _unavailable_section_message(
                    module_local_sufficiency,
                    expected_but_missing="_Module-local sufficiency summary was expected in this artifact but missing._",
                    not_in_artifact="_Module-local sufficiency summary was not included in this artifact._",
                )
            ),
            "",
            _markdown_table_or_message(
                module_local_rows,
                (
                    "module",
                    "coverage_mode",
                    "seeds",
                    "minimal_sufficient_level",
                    "canonical_v4_pass",
                    "partial_variant_coverage",
                ),
                section=module_local_sufficiency,
                expected_but_missing="_Module-local sufficiency rows were expected in this artifact but missing._",
                not_in_artifact="_Module-local sufficiency rows were not included in this artifact._",
            ),
            "",
            "## Distillation Diagnostic",
            "",
            (
                distillation_summary_line
                if distillation_summary_line
                else _unavailable_section_message(
                    distillation_analysis,
                    expected_but_missing="_Distillation diagnostic summary was expected in this artifact but missing._",
                    not_in_artifact="_Distillation diagnostic summary was not included in this artifact._",
                )
            ),
            "",
            _markdown_table_or_message(
                distillation_rows,
                (
                    "condition",
                    "episodes",
                    "survival_rate",
                    "mean_reward",
                    "mean_food_distance_delta",
                ),
                section=distillation_analysis,
                expected_but_missing="_Distillation diagnostic rows were expected in this artifact but missing._",
                not_in_artifact="_Distillation diagnostic rows were not included in this artifact._",
            ),
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
    if capacity_comparison_svg_path:
        markdown_lines.append("- `capacity_comparison.svg`")
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
        "capacity_comparison_svg": capacity_comparison_svg_path,
        # Keep the legacy key as an alias for API stability while callers migrate.
        "capacity_sweep_svg": capacity_comparison_svg_path,
        "reflex_frequency_svg": reflex_svg_path,
    }
