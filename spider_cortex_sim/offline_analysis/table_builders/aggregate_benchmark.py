from __future__ import annotations

from .common import *
from .capacity import build_capacity_sweep_tables

def build_aggregate_benchmark_tables(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """
    Builds aggregate benchmark tables from an evaluation/training summary, including confidence-interval fields when uncertainty is present.
    
    Parameters:
        summary (Mapping[str, object]): Evaluation and training payload containing primary benchmark data, per-scenario suite entries, learning-evidence summaries, and capacity reporting.
    
    Returns:
        dict[str, object]: A dictionary with the following keys:
            - available (bool): True if any table rows were produced.
            - confidence_level (float): Confidence level used for CI columns (defaults to 0.95 if unavailable).
            - primary_benchmark (dict): Table dict containing primary benchmark row(s) and CI/metadata columns.
            - per_scenario_success_rates (dict): Table dict with per-scenario success-rate rows and CI/metadata columns.
            - learning_evidence_deltas (dict): Table dict with learning-evidence delta rows (e.g., scenario_success_rate_delta) and CI/metadata columns.
            - architecture_capacity (dict): Table dict with per-architecture capacity rows and related metadata.
            - capacity_sweep_curves (dict): Capacity-sweep curve data as a dict (from capacity sweep tables).
            - capacity_sweep_metadata (dict): Metadata for capacity-sweep results.
            - capacity_analysis (dict): Capacity analysis information extracted from architecture capacity.
            - limitations (list[str]): Human-readable messages describing missing or incomplete data used to build the tables.
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

    architecture_capacity = extract_architecture_capacity(summary)
    capacity_sweep_tables = build_capacity_sweep_tables(summary)
    architecture_rows = list(architecture_capacity.get("rows", []))
    capacity_sweep_rows = list(
        _mapping_or_empty(capacity_sweep_tables.get("curves")).get("rows", [])
    )
    capacity_analysis = _mapping_or_empty(
        architecture_capacity.get("capacity_analysis")
    )

    limitations: list[str] = []
    if not primary_rows:
        limitations.append("No primary benchmark row with uncertainty was available.")
    if not scenario_rows:
        limitations.append("No per-scenario benchmark rows with uncertainty were available.")
    if not learning_rows:
        limitations.append("No learning-evidence delta uncertainty rows were available.")
    if not architecture_rows:
        limitations.extend(
            str(item)
            for item in architecture_capacity.get("limitations", [])
            if item
        )
    if not capacity_sweep_rows:
        limitations.extend(
            str(item)
            for item in capacity_sweep_tables.get("limitations", [])
            if item
        )
    return {
        "available": bool(
            primary_rows
            or scenario_rows
            or learning_rows
            or architecture_rows
            or capacity_sweep_rows
        ),
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
        "architecture_capacity": _table(
            (
                "variant",
                "architecture",
                "total_trainable",
                "key_components",
                "capacity_status",
                "capacity_matched",
                "reference_variant",
                "reference_total_trainable",
                "total_ratio_vs_reference",
                "source",
            ),
            architecture_rows,
        ),
        "capacity_sweep_curves": dict(
            _mapping_or_empty(capacity_sweep_tables.get("curves"))
        ),
        "capacity_sweep_metadata": dict(
            _mapping_or_empty(capacity_sweep_tables.get("metadata"))
        ),
        "capacity_analysis": dict(capacity_analysis),
        "limitations": limitations,
    }

__all__ = [name for name in globals() if not name.startswith("__")]
