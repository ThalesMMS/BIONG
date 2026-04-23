from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .constants import MODULAR_CREDIT_RUNGS, SHAPING_DEPENDENCE_WARNING_THRESHOLD
from .extractors import (
    _compare_credit_across_architectures,
    extract_architecture_capacity,
    extract_capacity_sweeps,
    extract_model_capacity,
    _noise_robustness_marginal,
    _noise_robustness_rate,
    _normalize_float_map,
    _representation_interpretation,
    _interpret_credit_failure,
    build_primary_benchmark,
    build_reflex_dependence_indicators,
    extract_ablations,
    extract_reward_profile_ladder,
    extract_comparisons,
    extract_credit_metrics,
    extract_noise_robustness,
    extract_predator_type_specialization,
    extract_reflex_frequency,
    extract_representation_specialization,
    extract_scenario_success,
    extract_shaping_audit,
    extract_training_eval_series,
    extract_unified_ladder_report,
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
    build_credit_assignment_tables,
    build_diagnostics,
    build_effect_size_tables,
    build_reward_profile_ladder_tables,
    build_unified_ladder_tables,
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
    _table,
    _table_rows,
    _write_csv,
    _write_svg,
)


def _artifact_state(*, available: bool, included: bool) -> str:
    if available:
        return "available"
    if included:
        return "expected_but_missing"
    return "not_in_this_artifact"


def _section_with_artifact_state(
    section: Mapping[str, object],
    *,
    included: bool,
) -> dict[str, object]:
    available = bool(section.get("available"))
    enriched = dict(section)
    enriched["artifact_state"] = _artifact_state(
        available=available,
        included=included,
    )
    return enriched


def _summary_parameter_counts_payload(
    summary: Mapping[str, object],
    model_capacity: Mapping[str, object],
) -> dict[str, object]:
    parameter_counts = dict(_mapping_or_empty(summary.get("parameter_counts")))
    if parameter_counts:
        return parameter_counts
    if not model_capacity.get("available"):
        return {}
    networks = (
        model_capacity.get("networks", [])
        if isinstance(model_capacity.get("networks"), list)
        else []
    )
    return {
        "architecture": str(model_capacity.get("architecture") or ""),
        "total_trainable": int(_coerce_float(model_capacity.get("total_trainable"), 0.0)),
        "by_network": {
            str(item.get("network") or ""): int(
                _coerce_float(item.get("parameters"), 0.0)
            )
            for item in networks
            if isinstance(item, Mapping) and str(item.get("network") or "")
        },
        "proportions": {
            str(item.get("network") or ""): _coerce_float(item.get("proportion"))
            for item in networks
            if isinstance(item, Mapping) and str(item.get("network") or "")
        },
        "source": str(model_capacity.get("source") or ""),
    }


def _credit_payloads_from_assignment(
    credit_assignment: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    strategy_rows = _table_rows(
        _mapping_or_empty(credit_assignment.get("strategy_summary"))
    )
    module_rows = _table_rows(
        _mapping_or_empty(credit_assignment.get("module_credit"))
    )
    payloads: dict[str, dict[str, object]] = {}
    payloads_by_key: dict[tuple[str, str], dict[str, object]] = {}

    for item in strategy_rows:
        if not isinstance(item, Mapping):
            continue
        rung = str(item.get("rung") or "")
        strategy = str(item.get("credit_strategy") or "")
        variant = str(item.get("variant") or "")
        if not rung or not strategy or not variant:
            continue
        payload = {
            "architecture_rung": rung,
            "strategy": strategy,
            "scenario_success_rate": item.get("scenario_success_rate"),
            "mean_effective_module_count": item.get("mean_effective_module_count"),
            "dominant_module": str(item.get("dominant_module") or ""),
            "weights": {},
            "gradient_norms": {},
            "counterfactual_weights": {},
        }
        payloads[variant] = payload
        payloads_by_key[(rung, strategy)] = payload

    for item in module_rows:
        if not isinstance(item, Mapping):
            continue
        rung = str(item.get("rung") or "")
        strategy = str(item.get("credit_strategy") or "")
        module_name = str(item.get("module") or "")
        payload = payloads_by_key.get((rung, strategy))
        if payload is None or not module_name:
            continue
        payload["weights"][module_name] = _coerce_float(
            item.get("mean_module_credit_weight")
        )
        payload["gradient_norms"][module_name] = _coerce_float(
            item.get("mean_module_gradient_norm")
        )
        payload["counterfactual_weights"][module_name] = _coerce_float(
            item.get("mean_counterfactual_credit_weight")
        )

    return payloads


def build_report_data(
    *,
    summary: Mapping[str, object],
    trace: Sequence[Mapping[str, object]],
    behavior_rows: Sequence[Mapping[str, object]],
    summary_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    behavior_csv_path: str | Path | None = None,
) -> dict[str, object]:
    """
    Assembles analysis sections from provided inputs and returns the consolidated report payload.
    
    Parameters:
        summary: Parsed summary JSON mapping (may be empty) used as the primary source for evaluations and ablations.
        trace: Sequence of parsed trace records (JSONL entries); used to compute reflex frequency and trace-derived indicators.
        behavior_rows: Sequence of behavior CSV row mappings; normalized and used to reconstruct scenario/ablation/comparison data when summary is incomplete.
        summary_path: Optional original path for the summary input (used only for reporting).
        trace_path: Optional original path for the trace input (used only for reporting).
        behavior_csv_path: Optional original path for the behavior CSV input (used only for reporting).
    
    Returns:
        A dict containing assembled report sections and metadata:
        - inputs: dict with input paths and booleans indicating which inputs produced data.
        - training_eval: training/evaluation series payload.
        - scenario_success: normalized per-scenario success payload.
        - primary_benchmark: selected "no-reflex" benchmark payload and provenance.
        - shaping_program: reward-shaping minimization audit, gap metrics, and minimal-profile survival indicators.
        - comparisons: reward-profile and map-template comparison payloads.
        - ablations: ablation variants, reference selection, deltas vs reference, ladder metadata, and limitations.
        - ladder_comparison: adjacent ladder-rung comparisons derived from the ablation payload.
        - ladder_profile_comparison: A0-A4 cross-profile comparison derived from the summary behavior_evaluation payload.
        - reward_profile_ladder_tables: structured tables derived from the cross-profile ladder comparison.
        - unified_ladder_report: structured unified ladder tables and conclusion payload combining rung, capacity, shaping, credit, no-reflex, probe, and missing-experiment reads.
        - architectural_ladder: named ladder-rung summaries and adjacent ladder comparisons derived from the ablation payload.
        - representation_specialization: representation-level specialization aggregates and interpretation.
        - reflex_frequency: per-module reflex event metrics derived from trace.
        - credit_analysis: credit-strategy comparison tables and diagnostic findings across architecture stages.
        - model_capacity: current-run trainable-parameter totals and per-network proportions.
        - capacity_analysis: capacity-match result derived from the available architecture totals.
        - reflex_dependence: override/dominance indicators, thresholds, and statuses.
        - scenario_checks: list of CSV-like rows describing scenario checks.
        - reward_components: list of rows for reward component values aggregated from summary, scenarios, and trace.
        - diagnostics: diagnostic metric entries aggregated from the various analyses.
        - limitations: list of textual limitation messages surfaced from the assembled sections.
    """
    normalized_rows = normalize_behavior_rows(behavior_rows)
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    ablations_included = "ablations" in behavior_evaluation
    capacity_sweeps_included = "capacity_sweeps" in behavior_evaluation
    ladder_profile_included = (
        "ladder_profile_comparison" in behavior_evaluation
        or "ladder_under_profiles" in behavior_evaluation
    )
    claim_tests_included = "claim_tests" in behavior_evaluation
    uncertainty_tables_included = bool(
        claim_tests_included
        or ablations_included
        or _mapping_or_empty(summary.get("evaluation")).get("seed_level")
        or _mapping_or_empty(summary.get("evaluation_without_reflex_support")).get(
            "seed_level"
        )
    )
    training_eval = extract_training_eval_series(summary)
    scenario_success = extract_scenario_success(summary, normalized_rows)
    comparisons = extract_comparisons(summary, normalized_rows)
    noise_robustness = extract_noise_robustness(summary, normalized_rows)
    ablations = extract_ablations(summary, normalized_rows)
    ladder_comparison = _mapping_or_empty(ablations.get("ladder_comparison"))
    ladder_profile_comparison = extract_reward_profile_ladder(summary)
    shaping_program = extract_shaping_audit(summary)
    reflex_frequency = extract_reflex_frequency(trace)
    predator_type_specialization = extract_predator_type_specialization(
        summary,
        normalized_rows,
        trace,
    )
    representation_specialization = extract_representation_specialization(
        summary,
        normalized_rows,
    )
    model_capacity = _section_with_artifact_state(
        extract_model_capacity(summary),
        included=bool(
            _mapping_or_empty(summary.get("parameter_counts")) or capacity_sweeps_included
        ),
    )
    architecture_capacity = _section_with_artifact_state(
        extract_architecture_capacity(summary),
        included=bool(
            ablations_included
            or _mapping_or_empty(summary.get("parameter_counts"))
            or capacity_sweeps_included
        ),
    )
    capacity_sweeps = _section_with_artifact_state(
        extract_capacity_sweeps(summary),
        included=capacity_sweeps_included,
    )
    capacity_analysis = _mapping_or_empty(
        architecture_capacity.get("capacity_analysis")
    )
    primary_benchmark = build_primary_benchmark(
        summary,
        scenario_success,
        ablations,
    )
    reflex_dependence = build_reflex_dependence_indicators(
        summary,
        reflex_frequency,
    )
    scenario_check_rows = build_scenario_checks_rows(scenario_success)
    reward_component_rows = build_reward_component_rows(
        summary,
        scenario_success,
        trace,
    )
    diagnostics = build_diagnostics(
        summary,
        scenario_success,
        ablations,
        reflex_frequency,
    )
    extracted_credit_metrics = extract_credit_metrics(summary, normalized_rows)
    credit_assignment = build_credit_assignment_tables(ablations)
    fallback_credit_metrics = _credit_payloads_from_assignment(credit_assignment)
    credit_metrics = dict(fallback_credit_metrics)
    credit_metrics.update(
        {
            str(variant_name): dict(payload)
            for variant_name, payload in extracted_credit_metrics.items()
            if isinstance(payload, Mapping)
        }
    )

    credit_variants_by_rung: dict[str, dict[str, Mapping[str, object]]] = {
        rung: {}
        for rung in MODULAR_CREDIT_RUNGS
    }
    credit_variant_names: dict[tuple[str, str], str] = {}
    credit_findings: list[dict[str, object]] = []
    credit_limitations: list[str] = []
    architecture_strategy_rows: list[dict[str, object]] = []

    for variant_name, credit_payload in sorted(credit_metrics.items()):
        if not isinstance(credit_payload, Mapping):
            continue
        rung = str(credit_payload.get("architecture_rung") or "")
        strategy = str(credit_payload.get("strategy") or "broadcast")
        if rung in credit_variants_by_rung and strategy:
            credit_variants_by_rung[rung][strategy] = credit_payload
            credit_variant_names[(rung, strategy)] = str(variant_name)
            architecture_strategy_rows.append(
                {
                    "architecture_rung": rung,
                    "credit_strategy": strategy,
                    "variant": str(variant_name),
                    "scenario_success_rate": credit_payload.get(
                        "scenario_success_rate"
                    ),
                    "mean_effective_module_count": credit_payload.get(
                        "mean_effective_module_count"
                    ),
                }
            )

    for rung in MODULAR_CREDIT_RUNGS:
        rung_variants = credit_variants_by_rung.get(rung, {})
        if not rung_variants:
            continue
        broadcast_reference = rung_variants.get("broadcast", {})
        if not broadcast_reference:
            credit_limitations.append(
                f"Credit diagnostics for {rung} lacked a broadcast reference."
            )
        for strategy in ("broadcast", "local_only", "counterfactual"):
            variant_payload = rung_variants.get(strategy)
            if not isinstance(variant_payload, Mapping):
                continue
            variant_name = credit_variant_names.get((rung, strategy), "")
            interpretation = _interpret_credit_failure(
                variant_payload,
                broadcast_reference if isinstance(broadcast_reference, Mapping) else {},
            )
            for finding in interpretation.get("findings", []):
                if isinstance(finding, Mapping):
                    credit_findings.append(
                        {
                            "scope": rung,
                            "variant": variant_name,
                            "strategy": strategy,
                            "pattern": str(finding.get("pattern") or ""),
                            "interpretation": str(finding.get("interpretation") or ""),
                            "evidence": dict(finding.get("evidence", {}))
                            if isinstance(finding.get("evidence"), Mapping)
                            else {},
                        }
                    )
            for limitation in interpretation.get("limitations", []):
                if limitation:
                    credit_limitations.append(str(limitation))

    cross_architecture_credit = _compare_credit_across_architectures(
        credit_variants_by_rung,
    )
    for finding in cross_architecture_credit.get("findings", []):
        if isinstance(finding, Mapping):
            credit_findings.append(
                {
                    "scope": "/".join(MODULAR_CREDIT_RUNGS),
                    "variant": "",
                    "strategy": "",
                    "pattern": str(finding.get("pattern") or ""),
                    "interpretation": str(finding.get("interpretation") or ""),
                    "evidence": dict(finding.get("evidence", {}))
                    if isinstance(finding.get("evidence"), Mapping)
                    else {},
                }
            )
    credit_limitations.extend(
        str(item)
        for item in cross_architecture_credit.get("limitations", [])
        if item
    )
    credit_analysis = _section_with_artifact_state(
        {
        "available": bool(credit_metrics or credit_assignment.get("available")),
        "strategy_comparison_table": _mapping_or_empty(
            credit_assignment.get("strategy_summary")
        ),
        "architecture_strategy_matrix": _table(
            (
                "architecture_rung",
                "credit_strategy",
                "variant",
                "scenario_success_rate",
                "mean_effective_module_count",
            ),
            architecture_strategy_rows,
        ),
        "findings": credit_findings,
        "comparisons": _mapping_or_empty(cross_architecture_credit.get("comparisons")),
        "limitations": credit_limitations,
        },
        included=ablations_included,
    )
    aggregate_benchmark_tables = _section_with_artifact_state(
        build_aggregate_benchmark_tables(summary),
        included=uncertainty_tables_included,
    )
    claim_test_tables = _section_with_artifact_state(
        build_claim_test_tables(summary),
        included=claim_tests_included,
    )
    effect_size_tables = _section_with_artifact_state(
        build_effect_size_tables(summary, normalized_rows),
        included=uncertainty_tables_included,
    )
    ladder_profile_comparison = _section_with_artifact_state(
        ladder_profile_comparison,
        included=ladder_profile_included,
    )
    reward_profile_ladder_tables = _section_with_artifact_state(
        build_reward_profile_ladder_tables(
            ladder_profile_comparison=ladder_profile_comparison,
        ),
        included=ladder_profile_included,
    )
    architectural_ladder = _mapping_or_empty(ablations.get("ladder"))
    unified_ladder_data = extract_unified_ladder_report(summary, normalized_rows)
    unified_ladder_report = _section_with_artifact_state(
        build_unified_ladder_tables(unified_ladder_data),
        included=bool(ablations_included or ladder_profile_included or capacity_sweeps_included),
    )

    limitations: list[str] = []
    seen_limitations: set[str] = set()

    def extend_limitations(section: Mapping[str, object]) -> None:
        for item in section.get("limitations", []):
            if not item:
                continue
            text = str(item)
            if text in seen_limitations:
                continue
            seen_limitations.add(text)
            limitations.append(text)

    for section in (
        training_eval,
        scenario_success,
        comparisons,
        noise_robustness,
        ablations,
        ladder_comparison,
        ladder_profile_comparison,
        shaping_program,
        reflex_frequency,
        credit_assignment,
        credit_analysis,
        predator_type_specialization,
        representation_specialization,
        model_capacity,
        architecture_capacity,
        capacity_sweeps,
        aggregate_benchmark_tables,
        claim_test_tables,
        effect_size_tables,
        reward_profile_ladder_tables,
        unified_ladder_report,
    ):
        if isinstance(section, Mapping):
            extend_limitations(section)

    inputs = {
        "summary_path": str(summary_path) if summary_path is not None else None,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "behavior_csv_path": str(behavior_csv_path) if behavior_csv_path is not None else None,
        "summary_loaded": bool(summary),
        "trace_loaded": bool(trace),
        "behavior_csv_loaded": bool(normalized_rows),
    }
    return {
        "inputs": inputs,
        "training_eval": training_eval,
        "scenario_success": scenario_success,
        "primary_benchmark": primary_benchmark,
        "shaping_program": shaping_program,
        "comparisons": comparisons,
        "noise_robustness": noise_robustness,
        "ablations": ablations,
        "ladder_comparison": ladder_comparison,
        "ladder_profile_comparison": ladder_profile_comparison,
        "architectural_ladder": dict(architectural_ladder),
        "predator_type_specialization": predator_type_specialization,
        "representation_specialization": representation_specialization,
        "parameter_counts": _summary_parameter_counts_payload(summary, model_capacity),
        "model_capacity": model_capacity,
        "capacity_sweeps": capacity_sweeps,
        "capacity_analysis": dict(capacity_analysis),
        "aggregate_benchmark_tables": aggregate_benchmark_tables,
        "claim_test_tables": claim_test_tables,
        "effect_size_tables": effect_size_tables,
        "reward_profile_ladder_tables": reward_profile_ladder_tables,
        "unified_ladder_report": unified_ladder_report,
        "reflex_frequency": reflex_frequency,
        "credit_assignment": credit_assignment,
        "credit_analysis": credit_analysis,
        "reflex_dependence": reflex_dependence,
        "scenario_checks": scenario_check_rows,
        "reward_components": reward_component_rows,
        "diagnostics": diagnostics,
        "limitations": limitations,
    }
