from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .constants import (
    AUSTERE_SURVIVAL_THRESHOLD,
    CLASSIFICATION_LABELS,
    DEFAULT_CONFIDENCE_LEVEL,
    LADDER_ADJACENT_COMPARISONS,
    LADDER_PRIMARY_VARIANT_BY_RUNG,
    LADDER_PROTOCOL_NAMES,
    LADDER_RUNG_MAPPING,
    MODULAR_CREDIT_RUNGS,
)
from .extractors import (
    compare_capacity_totals,
    extract_architecture_capacity,
    extract_capacity_sweeps,
    extract_credit_metrics,
    extract_reward_profile_ladder,
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
from .utils import (
    _coerce_float,
    _coerce_optional_float,
    _dominant_module_by_score,
    _mapping_or_empty,
)
from .writers import _table

CAPACITY_SWEEP_INTERPRETATION_GUIDANCE = (
    "Improvement with capacity -> possible undercapacity; "
    "no improvement -> likely interface/connection/reward issue."
)
CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD = 0.05
CREDIT_ASSIGNMENT_STRATEGY_ORDER: tuple[str, ...] = (
    "broadcast",
    "local_only",
    "counterfactual",
)
CREDIT_ASSIGNMENT_VARIANT_CANDIDATES: dict[tuple[str, str], tuple[str, ...]] = {
    ("A2", "broadcast"): ("three_center_modular",),
    ("A2", "local_only"): (
        "three_center_modular_local_credit",
        "three_center_local_credit",
    ),
    ("A2", "counterfactual"): (
        "three_center_modular_counterfactual",
        "three_center_counterfactual_credit",
    ),
    ("A3", "broadcast"): ("four_center_modular",),
    ("A3", "local_only"): ("four_center_modular_local_credit",),
    ("A3", "counterfactual"): ("four_center_modular_counterfactual",),
    ("A4", "broadcast"): ("modular_full",),
    ("A4", "local_only"): ("local_credit_only",),
    ("A4", "counterfactual"): ("counterfactual_credit",),
}

REWARD_PROFILE_LADDER_CONFIDENCE_LEVEL = DEFAULT_CONFIDENCE_LEVEL

UNIFIED_LADDER_HIGH_PRIORITY_EXPERIMENTS: frozenset[str] = frozenset(
    {
        "complete_architectural_ladder",
        "capacity_matched_ladder",
        "module_interface_sufficiency_suite",
        "no_reflex_competence_validation",
        "credit_assignment_variants",
    }
)
UNIFIED_LADDER_MEDIUM_PRIORITY_EXPERIMENTS: frozenset[str] = frozenset(
    {
        "cross_profile_ladder_runs",
        "five_seed_ladder_replication",
    }
)


def _credit_assignment_rung(variant_name: str) -> str | None:
    if variant_name in {
        "three_center_modular",
        "three_center_modular_local_credit",
        "three_center_modular_counterfactual",
        "three_center_local_credit",
        "three_center_counterfactual_credit",
    }:
        return "A2"
    if variant_name in {
        "four_center_modular",
        "four_center_modular_local_credit",
        "four_center_modular_counterfactual",
    }:
        return "A3"
    if variant_name in {
        "modular_full",
        "local_credit_only",
        "counterfactual_credit",
    }:
        return "A4"
    return None


def _unified_ladder_priority(experiment_name: str) -> str:
    if experiment_name in UNIFIED_LADDER_HIGH_PRIORITY_EXPERIMENTS:
        return "high"
    if experiment_name in UNIFIED_LADDER_MEDIUM_PRIORITY_EXPERIMENTS:
        return "medium"
    return "low"


def _unified_ladder_interpretation(row: Mapping[str, object]) -> str:
    delta = _coerce_optional_float(row.get("delta"))
    effect_size = _coerce_optional_float(row.get("cohens_d"))
    ci_lower = _coerce_optional_float(row.get("ci_lower"))
    ci_upper = _coerce_optional_float(row.get("ci_upper"))
    magnitude = str(row.get("effect_magnitude") or "")
    if delta is None:
        return "insufficient data"
    if (
        ci_lower is not None
        and ci_upper is not None
        and ci_lower <= 0.0 <= ci_upper
    ):
        return "confidence interval crosses zero"
    if effect_size is None:
        return "effect size unavailable"
    direction = "positive" if delta >= 0.0 else "negative"
    label = magnitude or "unlabeled"
    return f"{direction} {label} effect"


def build_unified_ladder_tables(
    unified_ladder_report: Mapping[str, object],
) -> dict[str, object]:
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
            name: _table((), ())
            for name in empty_rows
            if name != "missing_experiments_table"
        }
        empty_tables["missing_experiments_table"] = _table(
            ("experiment", "description", "priority"),
            missing_experiments_rows,
        )
        return {
            "available": False,
            "rows": empty_rows,
            "tables": empty_tables,
            "conclusion": "",
            "conclusion_rationale": "",
            "missing_experiments": missing_experiments_rows,
            "limitations": limitations,
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
    conclusion_table_rows = (
        [
            {
                "conclusion": str(ladder.get("conclusion") or ""),
                "rationale": str(ladder.get("conclusion_rationale") or ""),
                "confidence_level": _coerce_optional_float(
                    overall.get("confidence_level")
                )
                or DEFAULT_CONFIDENCE_LEVEL,
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


def _credit_assignment_strategy(
    variant_name: str,
    payload: Mapping[str, object],
) -> str:
    config = _mapping_or_empty(payload.get("config"))
    strategy = str(config.get("credit_strategy") or "")
    if strategy in CREDIT_ASSIGNMENT_STRATEGY_ORDER:
        return strategy
    if variant_name in {
        "three_center_modular_local_credit",
        "three_center_local_credit",
        "four_center_modular_local_credit",
        "local_credit_only",
    }:
        return "local_only"
    if variant_name in {
        "three_center_modular_counterfactual",
        "three_center_counterfactual_credit",
        "four_center_modular_counterfactual",
        "counterfactual_credit",
    }:
        return "counterfactual"
    return "broadcast"


def _credit_assignment_variant_rows(
    ablations: Mapping[str, object],
) -> tuple[
    dict[tuple[str, str], tuple[str, Mapping[str, object]]],
    list[str],
]:
    variants = {
        str(variant_name): _variant_with_minimal_reflex_support(payload)
        for variant_name, payload in _mapping_or_empty(ablations.get("variants")).items()
        if isinstance(payload, Mapping)
    }
    selected: dict[tuple[str, str], tuple[str, Mapping[str, object]]] = {}
    limitations: list[str] = []
    for key, candidates in CREDIT_ASSIGNMENT_VARIANT_CANDIDATES.items():
        selected_variant: tuple[str, Mapping[str, object]] | None = None
        for candidate in candidates:
            payload = variants.get(candidate)
            if isinstance(payload, Mapping):
                selected_variant = (candidate, payload)
                break
        if selected_variant is None:
            rung, strategy = key
            for variant_name, payload in sorted(variants.items()):
                if (
                    _credit_assignment_rung(variant_name) == rung
                    and _credit_assignment_strategy(variant_name, payload) == strategy
                ):
                    selected_variant = (variant_name, payload)
                    break
        if selected_variant is None:
            limitations.append(
                f"Credit-assignment comparison is missing {key[0]} {key[1]}."
            )
            continue
        selected[key] = selected_variant
    return selected, limitations


def _credit_assignment_interpretations(
    strategy_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows_by_key = {
        (str(row.get("rung") or ""), str(row.get("credit_strategy") or "")): row
        for row in strategy_rows
    }
    interpretations: list[dict[str, object]] = []

    local_failures: list[str] = []
    global_credit_failures: list[str] = []
    counterfactual_deltas: dict[str, float] = {}

    for rung in MODULAR_CREDIT_RUNGS:
        broadcast = rows_by_key.get((rung, "broadcast"))
        local_only = rows_by_key.get((rung, "local_only"))
        counterfactual = rows_by_key.get((rung, "counterfactual"))
        if isinstance(broadcast, Mapping) and isinstance(local_only, Mapping):
            local_delta = _coerce_float(local_only.get("scenario_success_delta_vs_broadcast"))
            if local_delta <= -CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD:
                local_failures.append(rung)
                interpretations.append(
                    {
                        "scope": rung,
                        "finding": "Failure by local credit insufficiency",
                        "evidence": round(local_delta, 6),
                        "interpretation": (
                            f"`local_only` trails broadcast by {local_delta:.2f} on {rung}, "
                            "so purely local gradients look insufficient at this rung."
                        ),
                    }
                )
            elif local_delta >= CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD:
                global_credit_failures.append(rung)
                interpretations.append(
                    {
                        "scope": rung,
                        "finding": "Failure by excessive global credit",
                        "evidence": round(local_delta, 6),
                        "interpretation": (
                            f"`local_only` beats broadcast by {local_delta:.2f} on {rung}, "
                            "so uniform global broadcast looks over-distributed here."
                        ),
                    }
                )
        if isinstance(broadcast, Mapping) and isinstance(counterfactual, Mapping):
            counterfactual_delta = _coerce_float(
                counterfactual.get("scenario_success_delta_vs_broadcast")
            )
            counterfactual_deltas[rung] = counterfactual_delta
            if counterfactual_delta >= CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD:
                interpretations.append(
                    {
                        "scope": rung,
                        "finding": "Counterfactual credit improvement",
                        "evidence": round(counterfactual_delta, 6),
                        "interpretation": (
                            f"`counterfactual` beats broadcast by {counterfactual_delta:.2f} "
                            f"on {rung}, suggesting targeted global credit is more useful than "
                            "uniform broadcast at this architecture."
                        ),
                    }
                )

    if local_failures:
        if set(local_failures) == set(MODULAR_CREDIT_RUNGS):
            scope_text = "`local_only` fails across every modular rung that was evaluated."
        elif len(local_failures) == 1:
            scope_text = (
                f"`local_only` fails primarily at {local_failures[0]} in the modular ladder."
            )
        else:
            scope_text = (
                "`local_only` shows mixed failures across the modular ladder: "
                + ", ".join(local_failures)
                + "."
            )
        interpretations.append(
            {
                "scope": "/".join(MODULAR_CREDIT_RUNGS),
                "finding": "Local-only failure scope",
                "evidence": ", ".join(local_failures),
                "interpretation": scope_text,
            }
        )

    if global_credit_failures:
        interpretations.append(
            {
                "scope": "/".join(MODULAR_CREDIT_RUNGS),
                "finding": "Global-credit failure scope",
                "evidence": ", ".join(global_credit_failures),
                "interpretation": (
                    "Uniform broadcast underperforms a local-only baseline at "
                    + ", ".join(global_credit_failures)
                    + "."
                ),
            }
        )

    available_counterfactual_rungs = [
        rung for rung in MODULAR_CREDIT_RUNGS if rung in counterfactual_deltas
    ]
    if len(available_counterfactual_rungs) >= 2:
        first_rung = available_counterfactual_rungs[0]
        last_rung = available_counterfactual_rungs[-1]
        cf_first = counterfactual_deltas[first_rung]
        cf_last = counterfactual_deltas[last_rung]
        diff = cf_last - cf_first
        if diff >= CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD:
            text = (
                f"`counterfactual` helps more in {last_rung} than {first_rung} "
                f"({cf_last:.2f} vs {cf_first:.2f} "
                "delta vs broadcast), consistent with larger gains when there are more modules."
            )
        elif diff <= -CREDIT_ASSIGNMENT_SUCCESS_DELTA_THRESHOLD:
            text = (
                f"`counterfactual` helps more in {first_rung} than {last_rung} "
                f"({cf_first:.2f} vs {cf_last:.2f} "
                "delta vs broadcast), so extra modularity did not amplify its benefit here."
            )
        else:
            text = (
                f"`counterfactual` has similar impact across {first_rung} and {last_rung} "
                f"({cf_first:.2f} vs {cf_last:.2f} delta vs broadcast)."
            )
        interpretations.append(
            {
                "scope": f"{first_rung}/{last_rung}",
                "finding": "Counterfactual scaling across the ladder",
                "evidence": round(diff, 6),
                "interpretation": text,
            }
        )
    return interpretations


def build_credit_assignment_tables(
    ablations: Mapping[str, object],
) -> dict[str, object]:
    """
    Build credit-assignment tables for the modular ladder stages.

    The output is organized as three tables:
    - `strategy_summary`: one row per architecture rung and credit strategy.
    - `module_credit`: one row per architecture rung, credit strategy, and module.
    - `scenario_success`: one row per architecture rung, credit strategy, and scenario.
    """
    selected, limitations = _credit_assignment_variant_rows(ablations)
    strategy_rows: list[dict[str, object]] = []
    module_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []

    for rung in MODULAR_CREDIT_RUNGS:
        broadcast_payload_tuple = selected.get((rung, "broadcast"))
        broadcast_summary = (
            _mapping_or_empty(broadcast_payload_tuple[1].get("summary"))
            if broadcast_payload_tuple is not None
            else {}
        )
        broadcast_success = _coerce_optional_float(
            broadcast_summary.get("scenario_success_rate")
        )
        for strategy in CREDIT_ASSIGNMENT_STRATEGY_ORDER:
            selected_payload = selected.get((rung, strategy))
            if selected_payload is None:
                continue
            variant_name, payload = selected_payload
            summary = _mapping_or_empty(payload.get("summary"))
            suite = _mapping_or_empty(payload.get("suite"))
            summary_row = {
                "rung": rung,
                "protocol_name": LADDER_PROTOCOL_NAMES.get(rung, rung),
                "variant": variant_name,
                "architecture": str(
                    _mapping_or_empty(payload.get("config")).get("architecture") or ""
                ),
                "credit_strategy": strategy,
                "scenario_success_rate": round(
                    _coerce_float(summary.get("scenario_success_rate")),
                    6,
                ),
                "episode_success_rate": round(
                    _coerce_float(summary.get("episode_success_rate")),
                    6,
                ),
                "scenario_success_delta_vs_broadcast": (
                    round(
                        _coerce_float(summary.get("scenario_success_rate"))
                        - broadcast_success,
                        6,
                    )
                    if broadcast_success is not None
                    else None
                ),
                "dominant_module": str(
                    summary.get("dominant_module")
                    or _dominant_module_by_score(
                        _mapping_or_empty(summary.get("dominant_module_distribution"))
                    )
                    or ""
                ),
                "mean_dominant_module_share": round(
                    _coerce_float(summary.get("mean_dominant_module_share")),
                    6,
                ),
                "mean_effective_module_count": round(
                    _coerce_float(summary.get("mean_effective_module_count")),
                    6,
                ),
                "source": f"ablations.variants.{variant_name}.summary",
            }
            strategy_rows.append(summary_row)

            module_credit_weights = _mapping_or_empty(
                summary.get("mean_module_credit_weights")
            )
            module_gradient_norms = _mapping_or_empty(
                summary.get("module_gradient_norm_means")
            )
            counterfactual_credit_weights = _mapping_or_empty(
                summary.get("mean_counterfactual_credit_weights")
            )
            module_contribution_share = _mapping_or_empty(
                summary.get("mean_module_contribution_share")
            )
            dominant_module_distribution = _mapping_or_empty(
                summary.get("dominant_module_distribution")
            )
            dominant_module_name = str(summary.get("dominant_module") or "")
            module_names = sorted(
                {
                    *module_credit_weights.keys(),
                    *module_gradient_norms.keys(),
                    *counterfactual_credit_weights.keys(),
                    *module_contribution_share.keys(),
                    *dominant_module_distribution.keys(),
                    *((dominant_module_name,) if dominant_module_name else ()),
                }
            )
            for module_name in module_names:
                module_rows.append(
                    {
                        "rung": rung,
                        "protocol_name": summary_row["protocol_name"],
                        "variant": variant_name,
                        "credit_strategy": strategy,
                        "module": str(module_name),
                        "mean_module_credit_weight": round(
                            _coerce_float(module_credit_weights.get(module_name)),
                            6,
                        ),
                        "mean_module_gradient_norm": round(
                            _coerce_float(module_gradient_norms.get(module_name)),
                            6,
                        ),
                        "mean_counterfactual_credit_weight": round(
                            _coerce_float(counterfactual_credit_weights.get(module_name)),
                            6,
                        ),
                        "mean_module_contribution_share": round(
                            _coerce_float(module_contribution_share.get(module_name)),
                            6,
                        ),
                        "dominant_module_rate": round(
                            _coerce_float(dominant_module_distribution.get(module_name)),
                            6,
                        ),
                        "scenario_success_rate": summary_row["scenario_success_rate"],
                    }
                )

            for scenario_name, scenario_payload in sorted(suite.items()):
                if not isinstance(scenario_payload, Mapping):
                    continue
                scenario_rows.append(
                    {
                        "rung": rung,
                        "protocol_name": summary_row["protocol_name"],
                        "variant": variant_name,
                        "credit_strategy": strategy,
                        "scenario": str(scenario_name),
                        "success_rate": round(
                            _coerce_float(scenario_payload.get("success_rate")),
                            6,
                        ),
                        "source": f"ablations.variants.{variant_name}.suite.{scenario_name}",
                    }
                )

    interpretation_rows = _credit_assignment_interpretations(strategy_rows)
    if not strategy_rows:
        limitations.append(
            "No modular-ladder credit-assignment variants were available in the ablation payload."
        )
    return {
        "available": bool(strategy_rows),
        "strategy_summary": _table(
            (
                "rung",
                "protocol_name",
                "variant",
                "architecture",
                "credit_strategy",
                "scenario_success_rate",
                "episode_success_rate",
                "scenario_success_delta_vs_broadcast",
                "dominant_module",
                "mean_dominant_module_share",
                "mean_effective_module_count",
                "source",
            ),
            strategy_rows,
        ),
        "module_credit": _table(
            (
                "rung",
                "protocol_name",
                "variant",
                "credit_strategy",
                "module",
                "mean_module_credit_weight",
                "mean_module_gradient_norm",
                "mean_counterfactual_credit_weight",
                "mean_module_contribution_share",
                "dominant_module_rate",
                "scenario_success_rate",
            ),
            module_rows,
        ),
        "scenario_success": _table(
            (
                "rung",
                "protocol_name",
                "variant",
                "credit_strategy",
                "scenario",
                "success_rate",
                "source",
            ),
            scenario_rows,
        ),
        "interpretations": list(interpretation_rows),
        "limitations": limitations,
    }


def build_credit_table(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """
    Build a machine-readable credit metrics table for benchmark packages.

    Rows are emitted per variant and module using the credit metrics available in
    summary blocks. The table is intentionally flat and light on interpretation.
    """
    credit_metrics = extract_credit_metrics(summary, ())
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    ablations = _mapping_or_empty(behavior_evaluation.get("ablations"))
    ablation_variants = _mapping_or_empty(ablations.get("variants"))
    config = _mapping_or_empty(summary.get("config"))
    brain_config = _mapping_or_empty(config.get("brain"))

    rows: list[dict[str, object]] = []
    mean_credit_by_strategy_module: list[dict[str, object]] = []
    concentration_rows: list[dict[str, object]] = []
    limitations: list[str] = []

    credit_means_by_strategy_module: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    effective_module_count_by_strategy: defaultdict[str, list[float]] = defaultdict(list)

    def variant_summary_payload(variant_name: str) -> Mapping[str, object]:
        ablation_payload = _mapping_or_empty(ablation_variants.get(variant_name))
        if ablation_payload:
            normalized_payload = _variant_with_minimal_reflex_support(ablation_payload)
            return _mapping_or_empty(normalized_payload.get("summary"))
        current_name = str(
            brain_config.get("name")
            or config.get("name")
            or "current_run"
        )
        if variant_name != current_name:
            return {}
        for block in (
            _mapping_or_empty(summary.get("evaluation_without_reflex_support")).get("summary"),
            summary.get("evaluation"),
            behavior_evaluation.get("summary"),
            summary.get("training"),
        ):
            payload = _mapping_or_empty(block)
            if payload:
                return payload
        return {}

    for variant_name, payload in sorted(credit_metrics.items()):
        summary_payload = variant_summary_payload(variant_name)
        config_payload = _mapping_or_empty(
            _mapping_or_empty(ablation_variants.get(variant_name)).get("config")
        )
        brain_variant_name = str(brain_config.get("name") or config.get("name") or "")
        architecture_rung = str(
            payload.get("architecture_rung")
            or _credit_assignment_rung(variant_name)
            or LADDER_RUNG_MAPPING.get(variant_name)
            or _credit_assignment_rung(brain_variant_name)
            or LADDER_RUNG_MAPPING.get(brain_variant_name)
            or ""
        )
        inferred_strategy = ""
        if config_payload or _credit_assignment_rung(variant_name):
            inferred_strategy = _credit_assignment_strategy(
                variant_name,
                {"config": config_payload},
            )
        elif brain_variant_name:
            inferred_strategy = _credit_assignment_strategy(
                brain_variant_name,
                {"config": brain_config},
            )
        strategy = str(
            payload.get("strategy")
            or config_payload.get("credit_strategy")
            or config.get("credit_strategy")
            or brain_config.get("credit_strategy")
            or inferred_strategy
            or "broadcast"
        )
        scenario_success_rate = _coerce_optional_float(
            summary_payload.get("scenario_success_rate")
        )
        mean_effective_module_count = _coerce_optional_float(
            summary_payload.get("mean_effective_module_count")
        )
        weights = _mapping_or_empty(payload.get("weights"))
        gradient_norms = _mapping_or_empty(payload.get("gradient_norms"))
        counterfactual_weights = _mapping_or_empty(
            payload.get("counterfactual_weights")
        )
        raw_weight_keys = set(weights.keys())
        module_names = sorted(
            {
                *weights.keys(),
                *gradient_norms.keys(),
                *counterfactual_weights.keys(),
            }
        )
        for module_name in module_names:
            credit_weight = _coerce_float(weights.get(module_name))
            gradient_norm = _coerce_float(gradient_norms.get(module_name))
            counterfactual_weight = _coerce_float(
                counterfactual_weights.get(module_name)
            )
            rows.append(
                {
                    "variant": variant_name,
                    "architecture_rung": architecture_rung,
                    "credit_strategy": strategy,
                    "module_name": str(module_name),
                    "credit_weight": round(credit_weight, 6),
                    "gradient_norm": round(gradient_norm, 6),
                    "counterfactual_weight": round(counterfactual_weight, 6),
                    "scenario_success_rate": (
                        round(float(scenario_success_rate), 6)
                        if scenario_success_rate is not None
                        else None
                    ),
                }
            )
            if module_name in raw_weight_keys:
                credit_means_by_strategy_module[(strategy, str(module_name))].append(
                    credit_weight
                )
        if mean_effective_module_count is not None:
            effective_module_count_by_strategy[strategy].append(
                float(mean_effective_module_count)
            )

    for (strategy, module_name), values in sorted(credit_means_by_strategy_module.items()):
        mean_credit_by_strategy_module.append(
            {
                "credit_strategy": strategy,
                "module_name": module_name,
                "mean_credit_weight": round(
                    sum(values) / len(values),
                    6,
                ),
                "variant_count": len(values),
            }
        )
    for strategy, values in sorted(effective_module_count_by_strategy.items()):
        concentration_rows.append(
            {
                "credit_strategy": strategy,
                "mean_effective_module_count": round(
                    sum(values) / len(values),
                    6,
                ),
                "variant_count": len(values),
            }
        )

    if not credit_metrics:
        limitations.append("No credit metrics were available in the summary payload.")

    return {
        "available": bool(rows),
        "table": _table(
            (
                "variant",
                "architecture_rung",
                "credit_strategy",
                "module_name",
                "credit_weight",
                "gradient_norm",
                "counterfactual_weight",
                "scenario_success_rate",
            ),
            rows,
        ),
        "summary_statistics": {
            "mean_credit_per_module_by_strategy": _table(
                ("credit_strategy", "module_name", "mean_credit_weight", "variant_count"),
                mean_credit_by_strategy_module,
            ),
            "credit_concentration": _table(
                ("credit_strategy", "mean_effective_module_count", "variant_count"),
                concentration_rows,
            ),
        },
        "limitations": limitations,
    }


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
            (str(name), int(_coerce_float(count, 0.0)))
            for name, count in per_network.items()
            if _coerce_float(count, 0.0) > 0.0
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
        capacity_summary = compare_capacity_totals(capacity_totals)
        if capacity_summary.get("available") and int(
            _coerce_float(capacity_summary.get("variant_count"), 0.0)
        ) > 1:
            diagnostics.append(
                {
                    "label": "Architecture capacity match",
                    "value": str(capacity_summary.get("status") or ""),
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


def build_capacity_sweep_tables(
    summary: Mapping[str, object],
) -> dict[str, object]:
    capacity_sweeps = extract_capacity_sweeps(summary)
    capacity_sweep_rows = list(capacity_sweeps.get("rows", []))
    if not capacity_sweep_rows:
        return {
            "available": False,
            "curves": _table(
                (
                    "variant",
                    "architecture",
                    "capacity_profile",
                    "capacity_profile_version",
                    "scale_factor",
                    "total_trainable",
                    "approximate_compute_cost_total",
                    "approximate_compute_cost_unit",
                    "scenario_success_rate",
                    "episode_success_rate",
                    "capability_probe_success_rate",
                    "source",
                ),
                (),
            ),
            "matrices": {},
            "interpretations": list(capacity_sweeps.get("interpretations", [])),
            "metadata": {
                "interpretation_guidance": CAPACITY_SWEEP_INTERPRETATION_GUIDANCE,
            },
            "limitations": [
                str(item)
                for item in capacity_sweeps.get("limitations", [])
                if item
            ],
        }

    matrices: dict[str, dict[str, dict[str, object]]] = {
        "scenario_success_rate": {},
        "episode_success_rate": {},
        "capability_probe_success_rate": {},
        "total_trainable": {},
        "approximate_compute_cost_total": {},
    }
    for row in capacity_sweep_rows:
        if not isinstance(row, Mapping):
            continue
        variant_name = str(row.get("variant") or "")
        profile_name = str(row.get("capacity_profile") or "")
        if not variant_name or not profile_name:
            continue
        matrices["scenario_success_rate"].setdefault(variant_name, {})[profile_name] = (
            row.get("scenario_success_rate")
        )
        matrices["episode_success_rate"].setdefault(variant_name, {})[profile_name] = (
            row.get("episode_success_rate")
        )
        matrices["capability_probe_success_rate"].setdefault(
            variant_name,
            {},
        )[profile_name] = row.get("capability_probe_success_rate")
        matrices["total_trainable"].setdefault(variant_name, {})[profile_name] = (
            row.get("total_trainable")
        )
        matrices["approximate_compute_cost_total"].setdefault(
            variant_name,
            {},
        )[profile_name] = row.get("approximate_compute_cost_total")

    return {
        "available": True,
        "curves": _table(
            (
                "variant",
                "architecture",
                "capacity_profile",
                "capacity_profile_version",
                "scale_factor",
                "total_trainable",
                "approximate_compute_cost_total",
                "approximate_compute_cost_unit",
                "scenario_success_rate",
                "episode_success_rate",
                "capability_probe_success_rate",
                "source",
            ),
            capacity_sweep_rows,
        ),
        "matrices": matrices,
        "interpretations": list(capacity_sweeps.get("interpretations", [])),
        "metadata": {
            "interpretation_guidance": CAPACITY_SWEEP_INTERPRETATION_GUIDANCE,
        },
        "limitations": [],
    }


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
            - architecture_capacity (dict): Table dict with per-architecture trainable-parameter totals and capacity-match status versus the reference variant.
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


def build_reward_profile_ladder_tables(
    summary: Mapping[str, object] | None = None,
    *,
    ladder_profile_comparison: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """
    Build structured tables from the cross-profile architectural ladder payload.

    Returns four tables:
    - ladder_by_profile
    - shaping_gap_by_variant
    - architecture_classifications
    - austere_survival_by_variant
    """
    ladder = (
        dict(ladder_profile_comparison)
        if isinstance(ladder_profile_comparison, Mapping)
        else extract_reward_profile_ladder(summary or {})
    )
    if not ladder.get("available"):
        limitations = list(ladder.get("limitations", []))
        if not limitations:
            limitations.append(
                "No cross-profile architectural ladder comparison was available."
            )
        return {
            "available": False,
            "confidence_level": REWARD_PROFILE_LADDER_CONFIDENCE_LEVEL,
            "rows": {
                "ladder_by_profile": [],
                "shaping_gap_by_variant": [],
                "architecture_classifications": [],
                "austere_survival_by_variant": [],
            },
            "ladder_by_profile": _table(
                (
                    "variant",
                    "protocol_name",
                    "profile",
                    "scenario_success_rate",
                    "scenario_success_rate_ci_lower",
                    "scenario_success_rate_ci_upper",
                    "episode_success_rate",
                    "episode_success_rate_ci_lower",
                    "episode_success_rate_ci_upper",
                    "mean_reward",
                    "mean_reward_ci_lower",
                    "mean_reward_ci_upper",
                    "n_seeds",
                ),
                (),
            ),
            "shaping_gap_by_variant": _table(
                (
                    "variant",
                    "protocol_name",
                    "classic_vs_austere_delta",
                    "classic_vs_austere_ci_lower",
                    "classic_vs_austere_ci_upper",
                    "classic_vs_austere_effect_size",
                    "ecological_vs_austere_delta",
                    "ecological_vs_austere_ci_lower",
                    "ecological_vs_austere_ci_upper",
                    "ecological_vs_austere_effect_size",
                    "n_seeds",
                ),
                (),
            ),
            "architecture_classifications": _table(
                (
                    "variant",
                    "protocol_name",
                    "classification",
                    "austere_success_rate",
                    "classic_success_rate",
                    "ecological_success_rate",
                    "classic_minus_austere",
                    "ecological_minus_austere",
                    "austere_survival_threshold",
                    "classic_gap_limit",
                    "ecological_gap_limit",
                    "reason",
                ),
                (),
            ),
            "austere_survival_by_variant": _table(
                (
                    "variant",
                    "protocol_name",
                    "survives",
                    "austere_success_rate",
                    "threshold",
                ),
                (),
            ),
            "limitations": limitations,
        }

    profiles = _mapping_or_empty(ladder.get("profiles"))
    cross_profile_deltas = _mapping_or_empty(ladder.get("cross_profile_deltas"))
    classifications = _mapping_or_empty(ladder.get("classifications"))
    raw_payload = _mapping_or_empty(ladder.get("raw_payload"))
    raw_variants = _mapping_or_empty(raw_payload.get("variants"))

    ladder_by_profile_rows: list[dict[str, object]] = []
    for profile_name, profile_payload in sorted(profiles.items()):
        if not isinstance(profile_payload, Mapping):
            continue
        variants = _mapping_or_empty(profile_payload.get("variants"))
        for variant_name, variant_payload in sorted(variants.items()):
            if not isinstance(variant_payload, Mapping):
                continue
            summary_payload = _mapping_or_empty(variant_payload.get("summary"))
            uncertainty = _mapping_or_empty(variant_payload.get("uncertainty"))
            scenario_uncertainty = _mapping_or_empty(
                uncertainty.get("scenario_success_rate")
            )
            episode_uncertainty = _mapping_or_empty(
                uncertainty.get("episode_success_rate")
            )
            reward_uncertainty = _mapping_or_empty(uncertainty.get("mean_reward"))
            protocol_name = str(
                _mapping_or_empty(raw_variants.get(variant_name)).get("protocol_name")
                or LADDER_PROTOCOL_NAMES.get(LADDER_RUNG_MAPPING.get(variant_name, ""), "")
            )
            ladder_by_profile_rows.append(
                {
                    "variant": str(variant_name),
                    "protocol_name": protocol_name,
                    "profile": str(profile_name),
                    "scenario_success_rate": _coerce_float(
                        summary_payload.get("scenario_success_rate"),
                    ),
                    "scenario_success_rate_ci_lower": _coerce_optional_float(
                        scenario_uncertainty.get("ci_lower")
                    ),
                    "scenario_success_rate_ci_upper": _coerce_optional_float(
                        scenario_uncertainty.get("ci_upper")
                    ),
                    "episode_success_rate": _coerce_float(
                        summary_payload.get("episode_success_rate"),
                    ),
                    "episode_success_rate_ci_lower": _coerce_optional_float(
                        episode_uncertainty.get("ci_lower")
                    ),
                    "episode_success_rate_ci_upper": _coerce_optional_float(
                        episode_uncertainty.get("ci_upper")
                    ),
                    "mean_reward": _coerce_optional_float(
                        summary_payload.get("mean_reward")
                    ),
                    "mean_reward_ci_lower": _coerce_optional_float(
                        reward_uncertainty.get("ci_lower")
                    ),
                    "mean_reward_ci_upper": _coerce_optional_float(
                        reward_uncertainty.get("ci_upper")
                    ),
                    "n_seeds": int(
                        _coerce_float(scenario_uncertainty.get("n_seeds"), 0.0)
                    ),
                }
            )

    shaping_gap_rows: list[dict[str, object]] = []
    for variant_name, deltas_payload in sorted(cross_profile_deltas.items()):
        if not isinstance(deltas_payload, Mapping):
            continue
        classic_payload = _mapping_or_empty(deltas_payload.get("classic"))
        ecological_payload = _mapping_or_empty(deltas_payload.get("ecological"))
        classic_metric = _mapping_or_empty(
            _mapping_or_empty(classic_payload.get("metrics")).get(
                "scenario_success_rate"
            )
        )
        ecological_metric = _mapping_or_empty(
            _mapping_or_empty(ecological_payload.get("metrics")).get(
                "scenario_success_rate"
            )
        )
        classic_uncertainty = _mapping_or_empty(classic_metric.get("delta_uncertainty"))
        ecological_uncertainty = _mapping_or_empty(
            ecological_metric.get("delta_uncertainty")
        )
        protocol_name = str(
            _mapping_or_empty(raw_variants.get(variant_name)).get("protocol_name")
            or LADDER_PROTOCOL_NAMES.get(LADDER_RUNG_MAPPING.get(variant_name, ""), "")
        )
        shaping_gap_rows.append(
            {
                "variant": str(variant_name),
                "protocol_name": protocol_name,
                "classic_vs_austere_delta": _coerce_optional_float(
                    _mapping_or_empty(classic_payload.get("summary")).get(
                        "scenario_success_rate_delta"
                    )
                ),
                "classic_vs_austere_ci_lower": _coerce_optional_float(
                    classic_uncertainty.get("ci_lower")
                ),
                "classic_vs_austere_ci_upper": _coerce_optional_float(
                    classic_uncertainty.get("ci_upper")
                ),
                "classic_vs_austere_effect_size": _coerce_optional_float(
                    classic_metric.get("cohens_d")
                ),
                "ecological_vs_austere_delta": _coerce_optional_float(
                    _mapping_or_empty(ecological_payload.get("summary")).get(
                        "scenario_success_rate_delta"
                    )
                ),
                "ecological_vs_austere_ci_lower": _coerce_optional_float(
                    ecological_uncertainty.get("ci_lower")
                ),
                "ecological_vs_austere_ci_upper": _coerce_optional_float(
                    ecological_uncertainty.get("ci_upper")
                ),
                "ecological_vs_austere_effect_size": _coerce_optional_float(
                    ecological_metric.get("cohens_d")
                ),
                "n_seeds": max(
                    int(_coerce_float(classic_uncertainty.get("n_seeds"), 0.0)),
                    int(_coerce_float(ecological_uncertainty.get("n_seeds"), 0.0)),
                ),
            }
        )

    classification_rows: list[dict[str, object]] = []
    austere_survival_rows: list[dict[str, object]] = []
    for variant_name, classification_payload in sorted(classifications.items()):
        if not isinstance(classification_payload, Mapping):
            continue
        classification = _mapping_or_empty(classification_payload.get("classification"))
        austere_survival = _mapping_or_empty(
            classification_payload.get("austere_survival")
        )
        label = str(classification.get("label") or "")
        if not label:
            label = "unknown"
        elif label not in CLASSIFICATION_LABELS:
            label = f"nonstandard:{label}"
        protocol_name = str(
            classification_payload.get("protocol_name")
            or LADDER_PROTOCOL_NAMES.get(LADDER_RUNG_MAPPING.get(variant_name, ""), "")
        )
        classification_rows.append(
            {
                "variant": str(variant_name),
                "protocol_name": protocol_name,
                "classification": label,
                "austere_success_rate": _coerce_optional_float(
                    classification.get("austere_success_rate")
                ),
                "classic_success_rate": _coerce_optional_float(
                    classification.get("classic_success_rate")
                ),
                "ecological_success_rate": _coerce_optional_float(
                    classification.get("ecological_success_rate")
                ),
                "classic_minus_austere": _coerce_optional_float(
                    classification.get("classic_minus_austere")
                ),
                "ecological_minus_austere": _coerce_optional_float(
                    classification.get("ecological_minus_austere")
                ),
                "austere_survival_threshold": _coerce_optional_float(
                    classification.get("austere_survival_threshold")
                )
                or AUSTERE_SURVIVAL_THRESHOLD,
                "classic_gap_limit": _coerce_optional_float(
                    classification.get("classic_gap_limit")
                ),
                "ecological_gap_limit": _coerce_optional_float(
                    classification.get("ecological_gap_limit")
                ),
                "reason": str(classification.get("reason") or ""),
            }
        )
        austere_survival_rows.append(
            {
                "variant": str(variant_name),
                "protocol_name": protocol_name,
                "survives": bool(austere_survival.get("passed", False)),
                "austere_success_rate": _coerce_optional_float(
                    austere_survival.get("success_rate")
                ),
                "threshold": _coerce_optional_float(
                    austere_survival.get("survival_threshold")
                )
                or AUSTERE_SURVIVAL_THRESHOLD,
            }
        )

    limitations = list(ladder.get("limitations", []))
    if not ladder_by_profile_rows:
        limitations.append("No reward-profile ladder rows were available.")
    if not shaping_gap_rows:
        limitations.append("No shaping-gap rows were available for the ladder.")
    if not classification_rows:
        limitations.append("No architecture classification rows were available.")
    if not austere_survival_rows:
        limitations.append("No austere survival rows were available.")

    return {
        "available": bool(
            ladder_by_profile_rows
            or shaping_gap_rows
            or classification_rows
            or austere_survival_rows
        ),
        "confidence_level": REWARD_PROFILE_LADDER_CONFIDENCE_LEVEL,
        "rows": {
            "ladder_by_profile": ladder_by_profile_rows,
            "shaping_gap_by_variant": shaping_gap_rows,
            "architecture_classifications": classification_rows,
            "austere_survival_by_variant": austere_survival_rows,
        },
        "ladder_by_profile": _table(
            (
                "variant",
                "protocol_name",
                "profile",
                "scenario_success_rate",
                "scenario_success_rate_ci_lower",
                "scenario_success_rate_ci_upper",
                "episode_success_rate",
                "episode_success_rate_ci_lower",
                "episode_success_rate_ci_upper",
                "mean_reward",
                "mean_reward_ci_lower",
                "mean_reward_ci_upper",
                "n_seeds",
            ),
            ladder_by_profile_rows,
        ),
        "shaping_gap_by_variant": _table(
            (
                "variant",
                "protocol_name",
                "classic_vs_austere_delta",
                "classic_vs_austere_ci_lower",
                "classic_vs_austere_ci_upper",
                "classic_vs_austere_effect_size",
                "ecological_vs_austere_delta",
                "ecological_vs_austere_ci_lower",
                "ecological_vs_austere_ci_upper",
                "ecological_vs_austere_effect_size",
                "n_seeds",
            ),
            shaping_gap_rows,
        ),
        "architecture_classifications": _table(
            (
                "variant",
                "protocol_name",
                "classification",
                "austere_success_rate",
                "classic_success_rate",
                "ecological_success_rate",
                "classic_minus_austere",
                "ecological_minus_austere",
                "austere_survival_threshold",
                "classic_gap_limit",
                "ecological_gap_limit",
                "reason",
            ),
            classification_rows,
        ),
        "austere_survival_by_variant": _table(
            (
                "variant",
                "protocol_name",
                "survives",
                "austere_success_rate",
                "threshold",
            ),
            austere_survival_rows,
        ),
        "limitations": limitations,
    }


def build_effect_size_tables(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """
    Builds effect-size rows for learning baselines, ablation comparisons, and claim-test-derived effect sizes.
    
    Parameters:
        summary (Mapping[str, object]): Evaluation/training summary payload containing
            behavior_evaluation, learning_evidence, ablations, and claim_tests.
        behavior_rows (Sequence[Mapping[str, object]]): Behavior rows used as fallback when
            summary ablations are unavailable.
    
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
    extracted_ablations = extract_ablations(summary, behavior_rows)

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

    variants = {
        str(variant_name): payload
        for variant_name, payload in _mapping_or_empty(
            extracted_ablations.get("variants")
        ).items()
        if isinstance(payload, Mapping)
    }
    reference_variant = str(extracted_ablations.get("reference_variant") or "")
    deltas_vs_reference = _mapping_or_empty(
        extracted_ablations.get("deltas_vs_reference")
    )
    for baseline in ("modular_full", "true_monolithic_policy", "monolithic_policy"):
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

    rung_to_variant = dict(LADDER_PRIMARY_VARIANT_BY_RUNG)
    for baseline_rung, comparison_rung in LADDER_ADJACENT_COMPARISONS:
        baseline_variant = rung_to_variant.get(baseline_rung, "")
        comparison_variant = rung_to_variant.get(comparison_rung, "")
        baseline_payload = _mapping_or_empty(variants.get(baseline_variant))
        comparison_payload = _mapping_or_empty(variants.get(comparison_variant))
        if not baseline_payload or not comparison_payload:
            continue
        baseline_summary = _mapping_or_empty(baseline_payload.get("summary"))
        comparison_summary = _mapping_or_empty(comparison_payload.get("summary"))
        raw_delta = round(
            _coerce_float(comparison_summary.get("scenario_success_rate"))
            - _coerce_float(baseline_summary.get("scenario_success_rate")),
            6,
        )
        row = _cohens_d_row(
            domain="ladder",
            baseline=baseline_variant,
            comparison=comparison_variant,
            metric="scenario_success_rate",
            baseline_values=_payload_metric_seed_items(
                baseline_payload,
                "scenario_success_rate",
            ),
            comparison_values=_payload_metric_seed_items(
                comparison_payload,
                "scenario_success_rate",
            ),
            raw_delta=raw_delta,
            uncertainty=None,
            source=(
                "summary.behavior_evaluation.ablations.ladder."
                f"{baseline_rung}_vs_{comparison_rung}"
            ),
        )
        if row is not None:
            rows.append(row)

    ladder_profile_base_path = "summary.behavior_evaluation.ladder_under_profiles"
    ladder_profile_comparison = _mapping_or_empty(
        behavior_evaluation.get("ladder_under_profiles")
    )
    if not ladder_profile_comparison:
        ladder_profile_base_path = (
            "summary.behavior_evaluation.ladder_profile_comparison"
        )
        ladder_profile_comparison = _mapping_or_empty(
            behavior_evaluation.get("ladder_profile_comparison")
        )
    ladder_profile_variants = _mapping_or_empty(ladder_profile_comparison.get("variants"))
    for variant_name, variant_payload in sorted(ladder_profile_variants.items()):
        if not isinstance(variant_payload, Mapping):
            continue
        shaping_gap = _mapping_or_empty(variant_payload.get("shaping_gap"))
        for profile_name, comparison_payload in sorted(shaping_gap.items()):
            if not isinstance(comparison_payload, Mapping):
                continue
            metric_blocks = _mapping_or_empty(comparison_payload.get("metrics"))
            for metric_name, metric_payload in sorted(metric_blocks.items()):
                if not isinstance(metric_payload, Mapping):
                    continue
                delta_uncertainty = _mapping_or_empty(
                    metric_payload.get("delta_uncertainty")
                )
                effect_uncertainty = _mapping_or_empty(
                    metric_payload.get("effect_size_uncertainty")
                )
                rows.append(
                    {
                        "domain": "ladder_profile",
                        "baseline": f"{variant_name}@austere",
                        "comparison": f"{variant_name}@{profile_name}",
                        "metric": str(metric_name),
                        "raw_delta": _coerce_optional_float(
                            metric_payload.get("delta")
                        ),
                        "cohens_d": _coerce_optional_float(
                            metric_payload.get("cohens_d")
                        ),
                        "magnitude_label": str(
                            metric_payload.get("effect_magnitude") or ""
                        ),
                        "source": (
                            f"{ladder_profile_base_path}."
                            f"variants.{variant_name}.shaping_gap.{profile_name}."
                            f"metrics.{metric_name}"
                        ),
                        "value": _coerce_optional_float(
                            metric_payload.get("cohens_d")
                        ),
                        "ci_lower": _coerce_optional_float(
                            effect_uncertainty.get("ci_lower")
                        ),
                        "ci_upper": _coerce_optional_float(
                            effect_uncertainty.get("ci_upper")
                        ),
                        "std_error": _coerce_optional_float(
                            effect_uncertainty.get("std_error")
                        ),
                        "n_seeds": int(
                            _coerce_float(effect_uncertainty.get("n_seeds"), 0.0)
                        ),
                        "confidence_level": _coerce_optional_float(
                            effect_uncertainty.get("confidence_level")
                        ),
                        "effect_size_ci_lower": _coerce_optional_float(
                            effect_uncertainty.get("ci_lower")
                        ),
                        "effect_size_ci_upper": _coerce_optional_float(
                            effect_uncertainty.get("ci_upper")
                        ),
                        "effect_size_std_error": _coerce_optional_float(
                            effect_uncertainty.get("std_error")
                        ),
                        "effect_size_n_seeds": int(
                            _coerce_float(effect_uncertainty.get("n_seeds"), 0.0)
                        ),
                        "effect_size_confidence_level": _coerce_optional_float(
                            effect_uncertainty.get("confidence_level")
                        ),
                        "delta_ci_lower": _coerce_optional_float(
                            delta_uncertainty.get("ci_lower")
                        ),
                        "delta_ci_upper": _coerce_optional_float(
                            delta_uncertainty.get("ci_upper")
                        ),
                        "delta_std_error": _coerce_optional_float(
                            delta_uncertainty.get("std_error")
                        ),
                        "delta_n_seeds": int(
                            _coerce_float(delta_uncertainty.get("n_seeds"), 0.0)
                        ),
                        "delta_confidence_level": _coerce_optional_float(
                            delta_uncertainty.get("confidence_level")
                        ),
                    }
                )

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
