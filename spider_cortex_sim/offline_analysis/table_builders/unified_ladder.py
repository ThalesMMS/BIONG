from __future__ import annotations

from collections.abc import Mapping

from .common import (
    DEFAULT_CONFIDENCE_LEVEL,
    _coerce_float,
    _coerce_optional_float,
    _mapping_or_empty,
    _unified_ladder_interpretation,
    _unified_ladder_priority,
)
from ..writers import _table


UNIFIED_LADDER_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "primary_rung_table": (
        "rung",
        "protocol_name",
        "variant",
        "present",
        "total_parameters",
        "scenario_success_rate",
        "ci_lower",
        "ci_upper",
        "effect_size_vs_a0",
        "effect_magnitude",
        "n_seeds",
        "limitations",
    ),
    "adjacent_comparison_table": (
        "baseline_rung",
        "comparison_rung",
        "delta",
        "ci_lower",
        "ci_upper",
        "cohens_d",
        "effect_magnitude",
        "interpretation",
    ),
    "capacity_summary_table": (
        "capacity_matched",
        "ratio",
        "largest_variant",
        "smallest_variant",
    ),
    "conclusion_table": (
        "conclusion",
        "rationale",
        "confidence_level",
        "confounding_factors",
    ),
    "missing_experiments_table": ("experiment", "description", "priority"),
    "credit_assignment_summary": (
        "rung",
        "local_only_delta_vs_broadcast",
        "counterfactual_delta_vs_broadcast",
        "strategies_present",
    ),
    "reward_shaping_sensitivity_summary": (
        "rung",
        "classic_minus_austere",
        "classic_effect_size",
        "ecological_minus_austere",
        "ecological_effect_size",
    ),
    "no_reflex_competence_summary": (
        "rung",
        "no_reflex_evaluated",
        "eval_reflex_scale",
        "scenario_success_rate",
    ),
    "capability_probe_boundaries": (
        "scenario",
        "requirement_level",
        "first_competent_rung",
        "highest_competent_rung",
        "rationale",
    ),
}

def build_unified_ladder_tables(
    unified_ladder_report: Mapping[str, object],
) -> dict[str, object]:
    """
    Builds a normalized set of row dictionaries and corresponding table descriptors from a unified architectural ladder report.
    
    Given a mapping representing a unified ladder report, returns a structured dictionary containing per-section row lists and _table(...) objects, top-level metadata (conclusion, rationale, confounds, limitations, missing experiments), raw comparison sub-mappings, and the original source report. If the input report is missing or marked unavailable, the result has "available": False, empty primary/adjacent/capacity/conclusion/summary tables, a populated missing-experiments table, and a derived limitations list.
    
    Parameters:
        unified_ladder_report (Mapping[str, object]): The source unified ladder report mapping to normalize. May be any mapping-like object or a non-mapping (treated as empty).
    
    Returns:
        dict[str, object]: A dictionary with keys including:
            - "available": bool indicating whether the source report was available.
            - "rows": mapping of table name -> list of normalized row dicts for each section.
            - "tables": mapping of table name -> table descriptor produced by _table(header_tuple, rows).
            - "conclusion": conclusion string from the report (or empty string).
            - "conclusion_rationale": rationale string from the report (or empty string).
            - "confounds": list of confound entries (or empty list).
            - "missing_experiments": list of missing-experiment row dicts.
            - "limitations": list of limitation strings (may include generated messages when sections are absent).
            - "credit_assignment_comparison", "reward_shaping_sensitivity", "no_reflex_competence", "capability_probe_boundaries_data": raw sub-mappings from the report (empty dicts when not present).
            - "source_report": the normalized source report mapping used to build the output.
    """
    ladder = (
        dict(unified_ladder_report)
        if isinstance(unified_ladder_report, Mapping)
        else {}
    )
    if not ladder.get("available"):
        missing_experiments = [
            item
            for item in ladder.get("missing_experiments", [])
            if isinstance(item, Mapping)
        ]
        missing_experiments_rows = [
            {
                "experiment": str(item.get("experiment_name") or ""),
                "description": str(item.get("description") or ""),
                "priority": _unified_ladder_priority(
                    str(item.get("experiment_name") or "")
                ),
            }
            for item in missing_experiments
        ]
        limitations = [
            str(item)
            for item in ladder.get("limitations", [])
            if item
        ]
        if not limitations:
            limitations.append("No unified architectural ladder report was available.")
        empty_rows: dict[str, list[dict[str, object]]] = {
            "primary_rung_table": [],
            "adjacent_comparison_table": [],
            "capacity_summary_table": [],
            "conclusion_table": [],
            "missing_experiments_table": missing_experiments_rows,
            "credit_assignment_summary": [],
            "reward_shaping_sensitivity_summary": [],
            "no_reflex_competence_summary": [],
            "capability_probe_boundaries": [],
        }
        empty_tables = {
            name: _table(UNIFIED_LADDER_TABLE_COLUMNS[name], ())
            for name in empty_rows
            if name != "missing_experiments_table"
        }
        empty_tables["missing_experiments_table"] = _table(
            UNIFIED_LADDER_TABLE_COLUMNS["missing_experiments_table"],
            missing_experiments_rows,
        )
        return {
            "available": False,
            "rows": empty_rows,
            "tables": empty_tables,
            "conclusion": "",
            "conclusion_rationale": "",
            "confounds": [],
            "missing_experiments": missing_experiments_rows,
            "limitations": limitations,
            "credit_assignment_comparison": {},
            "reward_shaping_sensitivity": {},
            "no_reflex_competence": {},
            "capability_probe_boundaries_data": {},
            "source_report": ladder,
        }

    ladder_table = _mapping_or_empty(ladder.get("ladder_table"))
    ladder_rows = [
        row
        for row in ladder_table.get("rows", [])
        if isinstance(row, Mapping)
    ]
    primary_rung_rows = [
        {
            "rung": str(row.get("rung") or ""),
            "protocol_name": str(row.get("protocol_name") or ""),
            "variant": str(row.get("variant") or ""),
            "present": bool(row.get("present")),
            "total_parameters": int(_coerce_float(row.get("total_trainable"), 0.0)),
            "scenario_success_rate": _coerce_optional_float(
                row.get("scenario_success_rate")
            ),
            "ci_lower": _coerce_optional_float(
                row.get("scenario_success_rate_ci_lower")
            ),
            "ci_upper": _coerce_optional_float(
                row.get("scenario_success_rate_ci_upper")
            ),
            "effect_size_vs_a0": _coerce_optional_float(row.get("effect_size_vs_a0")),
            "effect_magnitude": str(row.get("effect_magnitude_vs_a0") or ""),
            "n_seeds": int(_coerce_float(row.get("n_seeds"), 0.0)),
            "limitations": (
                list(row.get("limitations", []))
                if isinstance(row.get("limitations"), list)
                else str(row.get("limitations") or "")
            ),
        }
        for row in ladder_rows
    ]

    adjacent_comparison_rows = [
        {
            "baseline_rung": str(item.get("baseline_rung") or ""),
            "comparison_rung": str(item.get("comparison_rung") or ""),
            "delta": _coerce_optional_float(item.get("scenario_success_rate_delta")),
            "ci_lower": _coerce_optional_float(item.get("delta_ci_lower")),
            "ci_upper": _coerce_optional_float(item.get("delta_ci_upper")),
            "cohens_d": _coerce_optional_float(item.get("cohens_d")),
            "effect_magnitude": str(item.get("magnitude_label") or ""),
            "interpretation": _unified_ladder_interpretation(
                {
                    "delta": item.get("scenario_success_rate_delta"),
                    "ci_lower": item.get("delta_ci_lower"),
                    "ci_upper": item.get("delta_ci_upper"),
                    "cohens_d": item.get("cohens_d"),
                    "effect_magnitude": item.get("magnitude_label"),
                }
            ),
        }
        for item in ladder.get("adjacent_comparisons", [])
        if isinstance(item, Mapping)
    ]

    capacity = _mapping_or_empty(ladder.get("capacity_matched_comparison"))
    capacity_summary_rows = (
        [
            {
                "capacity_matched": bool(capacity.get("capacity_matched")),
                "ratio": _coerce_optional_float(capacity.get("ratio")),
                "largest_variant": str(capacity.get("largest_variant") or ""),
                "smallest_variant": str(capacity.get("smallest_variant") or ""),
            }
        ]
        if bool(capacity.get("available"))
        else []
    )

    confounds = ladder.get("confounds", [])
    confounding_factors = (
        "; ".join(str(item) for item in confounds if item)
        if isinstance(confounds, list) and confounds
        else ""
    )
    overall = _mapping_or_empty(ladder.get("overall_comparison"))
    confidence_level = _coerce_optional_float(overall.get("confidence_level"))
    conclusion_table_rows = (
        [
            {
                "conclusion": str(ladder.get("conclusion") or ""),
                "rationale": str(ladder.get("conclusion_rationale") or ""),
                "confidence_level": (
                    confidence_level
                    if confidence_level is not None
                    else DEFAULT_CONFIDENCE_LEVEL
                ),
                "confounding_factors": confounding_factors,
            }
        ]
        if bool(ladder.get("conclusion"))
        else []
    )

    missing_experiment_rows = (
        [
            {
                "experiment": str(item.get("experiment_name") or ""),
                "description": str(item.get("description") or ""),
                "priority": _unified_ladder_priority(
                    str(item.get("experiment_name") or "")
                ),
            }
            for item in ladder.get("missing_experiments", [])
            if isinstance(item, Mapping)
        ]
        if bool(ladder.get("available"))
        else []
    )

    credit_assignment = _mapping_or_empty(ladder.get("credit_assignment_comparison"))
    credit_assignment_rows = (
        [
            {
                "rung": str(item.get("rung") or ""),
                "local_only_delta_vs_broadcast": _coerce_optional_float(
                    item.get("local_only_delta_vs_broadcast")
                ),
                "counterfactual_delta_vs_broadcast": _coerce_optional_float(
                    item.get("counterfactual_delta_vs_broadcast")
                ),
                "strategies_present": ", ".join(
                    str(entry)
                    for entry in item.get("strategies_present", [])
                )
                if isinstance(item.get("strategies_present"), list)
                else "",
            }
            for item in credit_assignment.get("rows", [])
            if isinstance(item, Mapping)
        ]
        if bool(credit_assignment.get("available"))
        else []
    )

    reward_shaping = _mapping_or_empty(ladder.get("reward_shaping_sensitivity"))
    reward_shaping_rows = (
        [
            {
                "rung": str(item.get("rung") or ""),
                "classic_minus_austere": _coerce_optional_float(
                    item.get("classic_minus_austere")
                ),
                "classic_effect_size": _coerce_optional_float(
                    item.get("classic_effect_size")
                ),
                "ecological_minus_austere": _coerce_optional_float(
                    item.get("ecological_minus_austere")
                ),
                "ecological_effect_size": _coerce_optional_float(
                    item.get("ecological_effect_size")
                ),
            }
            for item in reward_shaping.get("rows", [])
            if isinstance(item, Mapping)
        ]
        if bool(reward_shaping.get("available"))
        else []
    )

    no_reflex = _mapping_or_empty(ladder.get("no_reflex_competence"))
    no_reflex_rows = (
        [
            {
                "rung": str(item.get("rung") or ""),
                "no_reflex_evaluated": bool(item.get("no_reflex_evaluated")),
                "eval_reflex_scale": _coerce_optional_float(
                    item.get("eval_reflex_scale")
                ),
                "scenario_success_rate": _coerce_optional_float(
                    item.get("scenario_success_rate")
                ),
            }
            for item in no_reflex.get("rungs", [])
            if isinstance(item, Mapping)
        ]
        if bool(no_reflex.get("available"))
        else []
    )

    capability_probes = _mapping_or_empty(ladder.get("capability_probe_boundaries"))
    capability_probe_rows = (
        [
            {
                "scenario": str(item.get("scenario") or ""),
                "requirement_level": str(item.get("requirement_level") or ""),
                "first_competent_rung": str(item.get("first_competent_rung") or ""),
                "highest_competent_rung": str(item.get("highest_competent_rung") or ""),
                "rationale": str(item.get("rationale") or ""),
            }
            for item in capability_probes.get("rows", [])
            if isinstance(item, Mapping)
        ]
        if bool(capability_probes.get("available"))
        else []
    )

    limitations = [
        str(item)
        for item in ladder.get("limitations", [])
        if item
    ]
    if not primary_rung_rows:
        limitations.append("Unified ladder primary rung table had no rows.")
    if not adjacent_comparison_rows:
        limitations.append("Unified ladder adjacent comparison table had no rows.")

    rows = {
        "primary_rung_table": primary_rung_rows,
        "adjacent_comparison_table": adjacent_comparison_rows,
        "capacity_summary_table": capacity_summary_rows,
        "conclusion_table": conclusion_table_rows,
        "missing_experiments_table": missing_experiment_rows,
        "credit_assignment_summary": credit_assignment_rows,
        "reward_shaping_sensitivity_summary": reward_shaping_rows,
        "no_reflex_competence_summary": no_reflex_rows,
        "capability_probe_boundaries": capability_probe_rows,
    }
    tables = {
        "primary_rung_table": _table(
            (
                "rung",
                "protocol_name",
                "variant",
                "present",
                "total_parameters",
                "scenario_success_rate",
                "ci_lower",
                "ci_upper",
                "effect_size_vs_a0",
                "effect_magnitude",
                "n_seeds",
                "limitations",
            ),
            primary_rung_rows,
        ),
        "adjacent_comparison_table": _table(
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
            adjacent_comparison_rows,
        ),
        "capacity_summary_table": _table(
            (
                "capacity_matched",
                "ratio",
                "largest_variant",
                "smallest_variant",
            ),
            capacity_summary_rows,
        ),
        "conclusion_table": _table(
            (
                "conclusion",
                "rationale",
                "confidence_level",
                "confounding_factors",
            ),
            conclusion_table_rows,
        ),
        "missing_experiments_table": _table(
            ("experiment", "description", "priority"),
            missing_experiment_rows,
        ),
        "credit_assignment_summary": _table(
            (
                "rung",
                "local_only_delta_vs_broadcast",
                "counterfactual_delta_vs_broadcast",
                "strategies_present",
            ),
            credit_assignment_rows,
        ),
        "reward_shaping_sensitivity_summary": _table(
            (
                "rung",
                "classic_minus_austere",
                "classic_effect_size",
                "ecological_minus_austere",
                "ecological_effect_size",
            ),
            reward_shaping_rows,
        ),
        "no_reflex_competence_summary": _table(
            (
                "rung",
                "no_reflex_evaluated",
                "eval_reflex_scale",
                "scenario_success_rate",
            ),
            no_reflex_rows,
        ),
        "capability_probe_boundaries": _table(
            (
                "scenario",
                "requirement_level",
                "first_competent_rung",
                "highest_competent_rung",
                "rationale",
            ),
            capability_probe_rows,
        ),
    }
    return {
        "available": True,
        "rows": rows,
        "tables": tables,
        "conclusion": str(ladder.get("conclusion") or ""),
        "conclusion_rationale": str(ladder.get("conclusion_rationale") or ""),
        "confounds": list(confounds) if isinstance(confounds, list) else [],
        "missing_experiments": list(missing_experiment_rows),
        "limitations": limitations,
        "credit_assignment_comparison": dict(
            _mapping_or_empty(ladder.get("credit_assignment_comparison"))
        ),
        "reward_shaping_sensitivity": dict(
            _mapping_or_empty(ladder.get("reward_shaping_sensitivity"))
        ),
        "no_reflex_competence": dict(
            _mapping_or_empty(ladder.get("no_reflex_competence"))
        ),
        "capability_probe_boundaries_data": dict(
            _mapping_or_empty(ladder.get("capability_probe_boundaries"))
        ),
        "source_report": ladder,
    }

__all__ = ["build_unified_ladder_tables"]
