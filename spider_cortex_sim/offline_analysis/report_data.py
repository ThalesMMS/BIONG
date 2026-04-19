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
        - ablations: ablation variants, reference selection, deltas vs reference, and limitations.
        - representation_specialization: representation-level specialization aggregates and interpretation.
        - reflex_frequency: per-module reflex event metrics derived from trace.
        - reflex_dependence: override/dominance indicators, thresholds, and statuses.
        - scenario_checks: list of CSV-like rows describing scenario checks.
        - reward_components: list of rows for reward component values aggregated from summary, scenarios, and trace.
        - diagnostics: diagnostic metric entries aggregated from the various analyses.
        - limitations: list of textual limitation messages surfaced from the assembled sections.
    """
    normalized_rows = normalize_behavior_rows(behavior_rows)
    training_eval = extract_training_eval_series(summary)
    scenario_success = extract_scenario_success(summary, normalized_rows)
    comparisons = extract_comparisons(summary, normalized_rows)
    noise_robustness = extract_noise_robustness(summary, normalized_rows)
    ablations = extract_ablations(summary, normalized_rows)
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
    aggregate_benchmark_tables = build_aggregate_benchmark_tables(summary)
    claim_test_tables = build_claim_test_tables(summary)
    effect_size_tables = build_effect_size_tables(summary)

    limitations: list[str] = []
    for section in (
        training_eval,
        scenario_success,
        comparisons,
        noise_robustness,
        ablations,
        shaping_program,
        reflex_frequency,
        predator_type_specialization,
        representation_specialization,
        aggregate_benchmark_tables,
        claim_test_tables,
        effect_size_tables,
    ):
        if isinstance(section, Mapping):
            limitations.extend(
                str(item)
                for item in section.get("limitations", [])
                if item
            )

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
        "predator_type_specialization": predator_type_specialization,
        "representation_specialization": representation_specialization,
        "aggregate_benchmark_tables": aggregate_benchmark_tables,
        "claim_test_tables": claim_test_tables,
        "effect_size_tables": effect_size_tables,
        "reflex_frequency": reflex_frequency,
        "reflex_dependence": reflex_dependence,
        "scenario_checks": scenario_check_rows,
        "reward_components": reward_component_rows,
        "diagnostics": diagnostics,
        "limitations": limitations,
    }
