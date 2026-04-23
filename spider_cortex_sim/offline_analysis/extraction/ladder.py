from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...reward import SCENARIO_AUSTERE_REQUIREMENTS
from ..constants import (
    LADDER_ACTIVE_RUNGS,
    LADDER_ADJACENT_COMPARISONS,
    LADDER_PRIMARY_VARIANT_BY_RUNG,
    LADDER_PROTOCOL_NAMES,
    MODULAR_CREDIT_RUNGS,
)
from ..uncertainty import (
    _cohens_d_row,
    _payload_metric_seed_items,
    _payload_uncertainty,
)
from ..utils import (
    _coerce_bool,
    _coerce_float,
    _coerce_optional_float,
    _mapping_or_empty,
)
from .ablation import extract_ablations, extract_reward_profile_ladder
from .benchmark import CAPACITY_MATCH_RATIO_THRESHOLD, compare_capacity_totals, extract_architecture_capacity
from .credit import _compare_credit_across_architectures, extract_credit_metrics

MODULARITY_SUPPORTED = "modularity supported"
MODULARITY_INCONCLUSIVE = "modularity inconclusive"
MODULARITY_HARMFUL = "modularity currently harmful"
CAPACITY_INTERFACE_CONFOUNDED = "capacity/interface confounded"


def _table(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "columns": [str(column) for column in columns],
        "rows": [dict(row) for row in rows],
    }


def _metric_uncertainty_fields(
    payload: Mapping[str, object],
    metric_name: str,
) -> dict[str, object]:
    uncertainty = _mapping_or_empty(_payload_uncertainty(payload, metric_name))
    return {
        "ci_lower": _coerce_optional_float(uncertainty.get("ci_lower")),
        "ci_upper": _coerce_optional_float(uncertainty.get("ci_upper")),
        "std_error": _coerce_optional_float(uncertainty.get("std_error")),
        "n_seeds": int(_coerce_float(uncertainty.get("n_seeds"), 0.0)),
        "confidence_level": _coerce_optional_float(uncertainty.get("confidence_level")),
    }


def _effect_summary(
    *,
    baseline_variant: str,
    comparison_variant: str,
    baseline_payload: Mapping[str, object],
    comparison_payload: Mapping[str, object],
    source: str,
) -> dict[str, object]:
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
        source=source,
        uncertainty=None,
    )
    if row is None:
        return {
            "delta": raw_delta,
            "delta_ci_lower": None,
            "delta_ci_upper": None,
            "cohens_d": None,
            "magnitude_label": "",
            "n_seeds": 0,
            "confidence_level": None,
        }
    return {
        "delta": _coerce_optional_float(row.get("raw_delta")),
        "delta_ci_lower": _coerce_optional_float(row.get("delta_ci_lower")),
        "delta_ci_upper": _coerce_optional_float(row.get("delta_ci_upper")),
        "cohens_d": _coerce_optional_float(row.get("cohens_d")),
        "magnitude_label": str(row.get("magnitude_label") or ""),
        "n_seeds": int(_coerce_float(row.get("delta_n_seeds"), 0.0)),
        "confidence_level": _coerce_optional_float(row.get("delta_confidence_level")),
    }


def _extract_adjacent_comparisons(
    ablations: Mapping[str, object],
) -> list[dict[str, object]]:
    variants = {
        str(name): payload
        for name, payload in _mapping_or_empty(ablations.get("variants")).items()
        if isinstance(payload, Mapping)
    }
    ladder_comparison = _mapping_or_empty(ablations.get("ladder_comparison"))
    rows: list[dict[str, object]] = []
    for item in ladder_comparison.get("comparisons", []):
        if not isinstance(item, Mapping):
            continue
        metrics = _mapping_or_empty(item.get("metrics"))
        baseline_variant = str(metrics.get("baseline_variant") or "")
        comparison_variant = str(metrics.get("comparison_variant") or "")
        baseline_payload = _mapping_or_empty(variants.get(baseline_variant))
        comparison_payload = _mapping_or_empty(variants.get(comparison_variant))
        effect = (
            _effect_summary(
                baseline_variant=baseline_variant,
                comparison_variant=comparison_variant,
                baseline_payload=baseline_payload,
                comparison_payload=comparison_payload,
                source=str(item.get("source") or ""),
            )
            if baseline_payload and comparison_payload
            else {}
        )
        deltas = _mapping_or_empty(item.get("deltas"))
        rows.append(
            {
                "baseline_rung": str(item.get("baseline_rung") or ""),
                "comparison_rung": str(item.get("comparison_rung") or ""),
                "baseline_variant": baseline_variant,
                "comparison_variant": comparison_variant,
                "scenario_success_rate_delta": _coerce_optional_float(
                    deltas.get("scenario_success_rate_delta")
                ),
                "delta_ci_lower": _coerce_optional_float(effect.get("delta_ci_lower")),
                "delta_ci_upper": _coerce_optional_float(effect.get("delta_ci_upper")),
                "cohens_d": _coerce_optional_float(effect.get("cohens_d")),
                "magnitude_label": str(effect.get("magnitude_label") or ""),
                "n_seeds": int(_coerce_float(effect.get("n_seeds"), 0.0)),
                "source": str(item.get("source") or ""),
            }
        )
    return rows


def _build_ladder_rows(
    *,
    ablations: Mapping[str, object],
    architecture_capacity: Mapping[str, object],
    reward_profile_ladder: Mapping[str, object],
) -> list[dict[str, object]]:
    variants = {
        str(name): payload
        for name, payload in _mapping_or_empty(ablations.get("variants")).items()
        if isinstance(payload, Mapping)
    }
    capacity_rows = {
        str(row.get("variant") or ""): row
        for row in architecture_capacity.get("rows", [])
        if isinstance(row, Mapping)
    }
    classifications = _mapping_or_empty(reward_profile_ladder.get("classifications"))
    adjacent_by_rung = {
        str(row.get("comparison_rung") or ""): row
        for row in _extract_adjacent_comparisons(ablations)
        if isinstance(row, Mapping)
    }
    a0_variant = LADDER_PRIMARY_VARIANT_BY_RUNG.get("A0", "")
    a0_payload = _mapping_or_empty(variants.get(a0_variant))
    rows: list[dict[str, object]] = []
    for rung in LADDER_ACTIVE_RUNGS:
        variant_name = LADDER_PRIMARY_VARIANT_BY_RUNG.get(rung, "")
        variant_payload = _mapping_or_empty(variants.get(variant_name))
        summary_payload = _mapping_or_empty(variant_payload.get("summary"))
        capacity_payload = _mapping_or_empty(capacity_rows.get(variant_name))
        classification_payload = _mapping_or_empty(
            _mapping_or_empty(classifications.get(variant_name)).get("classification")
        )
        uncertainty_fields = _metric_uncertainty_fields(
            variant_payload,
            "scenario_success_rate",
        )
        previous_effect = _mapping_or_empty(adjacent_by_rung.get(rung))
        vs_a0_effect = (
            _effect_summary(
                baseline_variant=a0_variant,
                comparison_variant=variant_name,
                baseline_payload=a0_payload,
                comparison_payload=variant_payload,
                source=f"summary.behavior_evaluation.ablations.ladder.A0_vs_{rung}",
            )
            if variant_payload and a0_payload
            else {}
        )
        limitations: list[str] = []
        if not variant_payload:
            limitations.append(
                f"Primary ladder variant {variant_name!r} for rung {rung} was unavailable."
            )
        if variant_payload and int(uncertainty_fields["n_seeds"]) <= 0:
            limitations.append(
                f"Rung {rung} did not include seed-level uncertainty for scenario_success_rate."
            )
        rows.append(
            {
                "rung": rung,
                "protocol_name": LADDER_PROTOCOL_NAMES.get(rung, rung),
                "variant": variant_name,
                "present": bool(variant_payload),
                "total_trainable": int(
                    _coerce_float(capacity_payload.get("total_trainable"), 0.0)
                ),
                "scenario_success_rate": _coerce_optional_float(
                    summary_payload.get("scenario_success_rate")
                ),
                "scenario_success_rate_ci_lower": uncertainty_fields["ci_lower"],
                "scenario_success_rate_ci_upper": uncertainty_fields["ci_upper"],
                "n_seeds": int(uncertainty_fields["n_seeds"]),
                "confidence_level": uncertainty_fields["confidence_level"],
                "eval_reflex_scale": _coerce_optional_float(
                    summary_payload.get("eval_reflex_scale")
                ),
                "capacity_status": str(capacity_payload.get("capacity_status") or ""),
                "total_ratio_vs_reference": _coerce_optional_float(
                    capacity_payload.get("total_ratio_vs_reference")
                ),
                "shaping_classification": str(
                    classification_payload.get("label")
                    or classification_payload.get("classification")
                    or ""
                ),
                "delta_vs_previous": _coerce_optional_float(
                    previous_effect.get("scenario_success_rate_delta")
                ),
                "delta_ci_lower_vs_previous": _coerce_optional_float(
                    previous_effect.get("delta_ci_lower")
                ),
                "delta_ci_upper_vs_previous": _coerce_optional_float(
                    previous_effect.get("delta_ci_upper")
                ),
                "cohens_d_vs_previous": _coerce_optional_float(
                    previous_effect.get("cohens_d")
                ),
                "effect_magnitude_vs_previous": str(
                    previous_effect.get("magnitude_label") or ""
                ),
                "delta_vs_a0": _coerce_optional_float(vs_a0_effect.get("delta")),
                "delta_ci_lower_vs_a0": _coerce_optional_float(
                    vs_a0_effect.get("delta_ci_lower")
                ),
                "delta_ci_upper_vs_a0": _coerce_optional_float(
                    vs_a0_effect.get("delta_ci_upper")
                ),
                "effect_size_vs_a0": _coerce_optional_float(
                    vs_a0_effect.get("cohens_d")
                ),
                "effect_magnitude_vs_a0": str(
                    vs_a0_effect.get("magnitude_label") or ""
                ),
                "limitations": limitations,
            }
        )
    return rows


def _extract_capacity_comparison(
    ladder_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    totals = {
        str(row.get("variant") or ""): int(_coerce_float(row.get("total_trainable"), 0.0))
        for row in ladder_rows
        if bool(row.get("present")) and _coerce_float(row.get("total_trainable"), 0.0) > 0.0
    }
    analysis = compare_capacity_totals(totals)
    return {
        **analysis,
        "rows": [
            {
                "rung": str(row.get("rung") or ""),
                "variant": str(row.get("variant") or ""),
                "total_trainable": int(_coerce_float(row.get("total_trainable"), 0.0)),
            }
            for row in ladder_rows
        ],
    }


def _extract_interface_sufficiency_results(
    summary: Mapping[str, object],
) -> dict[str, object]:
    modules: list[dict[str, object]] = []
    insufficient_results: list[dict[str, object]] = []

    def visit(value: object, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            if (
                "module_name" in value
                and "minimal_sufficient_level" in value
                and isinstance(value.get("levels"), list)
            ):
                module_name = str(value.get("module_name") or "")
                levels = [
                    level
                    for level in value.get("levels", [])
                    if isinstance(level, Mapping)
                ]
                interface_insufficient_count = 0
                for level in levels:
                    for task_detail in level.get("task_details", []):
                        if not isinstance(task_detail, Mapping):
                            continue
                        if str(task_detail.get("status") or "") != "interface_insufficient":
                            continue
                        interface_insufficient_count += 1
                        insufficient_results.append(
                            {
                                "module_name": module_name,
                                "level": int(_coerce_float(level.get("level"), 0.0)),
                                "task_name": str(task_detail.get("task_name") or ""),
                                "status": "interface_insufficient",
                                "source": ".".join(path),
                            }
                        )
                modules.append(
                    {
                        "module_name": module_name,
                        "minimal_sufficient_level": (
                            int(_coerce_float(value.get("minimal_sufficient_level"), 0.0))
                            if value.get("minimal_sufficient_level") is not None
                            else None
                        ),
                        "minimal_sufficient_interface": str(
                            value.get("minimal_sufficient_interface") or ""
                        ),
                        "levels_evaluated": list(value.get("levels_evaluated", []))
                        if isinstance(value.get("levels_evaluated"), list)
                        else [],
                        "interface_insufficient_count": interface_insufficient_count,
                        "source": ".".join(path),
                    }
                )
            for key, nested in value.items():
                visit(nested, (*path, str(key)))
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, (*path, str(index)))

    visit(summary, ("summary",))
    if not modules:
        return {
            "available": False,
            "modules": [],
            "interface_insufficient_results": [],
            "any_interface_insufficient": False,
            "limitations": [
                "Interface sufficiency results were unavailable in the summary JSON."
            ],
        }
    return {
        "available": True,
        "modules": modules,
        "interface_insufficient_results": insufficient_results,
        "any_interface_insufficient": bool(insufficient_results),
        "limitations": [],
    }


def _extract_credit_assignment_comparison(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    credit_metrics = extract_credit_metrics(summary, behavior_rows)
    variants_by_rung: dict[str, dict[str, Mapping[str, object]]] = {
        rung: {}
        for rung in MODULAR_CREDIT_RUNGS
    }
    for _variant_name, payload in credit_metrics.items():
        if not isinstance(payload, Mapping):
            continue
        rung = str(payload.get("architecture_rung") or "")
        strategy = str(payload.get("strategy") or "")
        if rung in variants_by_rung and strategy:
            variants_by_rung[rung][strategy] = payload
    cross_architecture = _compare_credit_across_architectures(variants_by_rung)
    rows = [
        {
            "rung": rung,
            "local_only_delta_vs_broadcast": _coerce_optional_float(
                _mapping_or_empty(cross_architecture.get("comparisons")).get(
                    rung,
                    {},
                ).get("local_only_delta_vs_broadcast")
            ),
            "counterfactual_delta_vs_broadcast": _coerce_optional_float(
                _mapping_or_empty(cross_architecture.get("comparisons")).get(
                    rung,
                    {},
                ).get("counterfactual_delta_vs_broadcast")
            ),
            "strategies_present": sorted(variants_by_rung.get(rung, {}).keys()),
        }
        for rung in MODULAR_CREDIT_RUNGS
    ]
    limitations = [
        str(item)
        for item in cross_architecture.get("limitations", [])
        if item
    ]
    if not credit_metrics:
        limitations.append("Credit-assignment comparison data was unavailable.")
    return {
        "available": bool(credit_metrics),
        "rows": rows,
        "findings": list(cross_architecture.get("findings", [])),
        "strategies_by_rung": {
            rung: sorted(variants_by_rung.get(rung, {}).keys())
            for rung in MODULAR_CREDIT_RUNGS
        },
        "limitations": limitations,
    }


def _extract_reward_shaping_sensitivity(
    reward_profile_ladder: Mapping[str, object],
) -> dict[str, object]:
    cross_profile_deltas = _mapping_or_empty(reward_profile_ladder.get("cross_profile_deltas"))
    profiles = _mapping_or_empty(reward_profile_ladder.get("profiles"))
    rows: list[dict[str, object]] = []
    for rung in LADDER_ACTIVE_RUNGS:
        variant_name = LADDER_PRIMARY_VARIANT_BY_RUNG.get(rung, "")
        delta_payload = _mapping_or_empty(cross_profile_deltas.get(variant_name))
        classic_metric = _mapping_or_empty(
            _mapping_or_empty(
                _mapping_or_empty(delta_payload.get("classic")).get("metrics")
            ).get("scenario_success_rate")
        )
        ecological_metric = _mapping_or_empty(
            _mapping_or_empty(
                _mapping_or_empty(delta_payload.get("ecological")).get("metrics")
            ).get("scenario_success_rate")
        )
        classic_uncertainty = _mapping_or_empty(classic_metric.get("delta_uncertainty"))
        ecological_uncertainty = _mapping_or_empty(
            ecological_metric.get("delta_uncertainty")
        )
        rows.append(
            {
                "rung": rung,
                "variant": variant_name,
                "classic_minus_austere": _coerce_optional_float(
                    _mapping_or_empty(delta_payload.get("classic")).get(
                        "summary",
                        {},
                    ).get("scenario_success_rate_delta")
                ),
                "classic_ci_lower": _coerce_optional_float(
                    classic_uncertainty.get("ci_lower")
                ),
                "classic_ci_upper": _coerce_optional_float(
                    classic_uncertainty.get("ci_upper")
                ),
                "classic_effect_size": _coerce_optional_float(
                    classic_metric.get("cohens_d")
                ),
                "ecological_minus_austere": _coerce_optional_float(
                    _mapping_or_empty(delta_payload.get("ecological")).get(
                        "summary",
                        {},
                    ).get("scenario_success_rate_delta")
                ),
                "ecological_ci_lower": _coerce_optional_float(
                    ecological_uncertainty.get("ci_lower")
                ),
                "ecological_ci_upper": _coerce_optional_float(
                    ecological_uncertainty.get("ci_upper")
                ),
                "ecological_effect_size": _coerce_optional_float(
                    ecological_metric.get("cohens_d")
                ),
            }
        )
    limitations = [
        str(item)
        for item in reward_profile_ladder.get("limitations", [])
        if item
    ]
    if not reward_profile_ladder.get("available"):
        limitations.append("Cross-profile ladder results were unavailable.")
    return {
        "available": bool(reward_profile_ladder.get("available")),
        "rows": rows,
        "profiles_available": sorted(profiles.keys()),
        "minimal_profile": str(reward_profile_ladder.get("minimal_profile") or "austere"),
        "limitations": limitations,
    }


def _extract_no_reflex_competence(
    summary: Mapping[str, object],
    ablations: Mapping[str, object],
) -> dict[str, object]:
    claims = _mapping_or_empty(
        _mapping_or_empty(
            _mapping_or_empty(summary.get("behavior_evaluation")).get("claim_tests")
        ).get("claims")
    )
    escape_claim = _mapping_or_empty(claims.get("escape_without_reflex_support"))
    claim_result = {}
    if escape_claim:
        comparison_values = _mapping_or_empty(escape_claim.get("comparison_values"))
        deltas = _mapping_or_empty(escape_claim.get("delta"))
        cohens_d = _mapping_or_empty(escape_claim.get("cohens_d"))
        claim_result = {
            "status": str(escape_claim.get("status") or ""),
            "passed": _coerce_bool(escape_claim.get("passed")),
            "reference_value": _coerce_optional_float(escape_claim.get("reference_value")),
            "comparison_value": _coerce_optional_float(
                comparison_values.get("trained_without_reflex_support")
            ),
            "delta": _coerce_optional_float(
                deltas.get("trained_without_reflex_support")
            ),
            "cohens_d": _coerce_optional_float(
                cohens_d.get("trained_without_reflex_support")
            ),
            "primary_metric": str(escape_claim.get("primary_metric") or ""),
            "scenarios_evaluated": list(escape_claim.get("scenarios_evaluated", []))
            if isinstance(escape_claim.get("scenarios_evaluated"), list)
            else [],
        }
    variants = _mapping_or_empty(ablations.get("variants"))
    rows = []
    for rung in LADDER_ACTIVE_RUNGS:
        variant_name = LADDER_PRIMARY_VARIANT_BY_RUNG.get(rung, "")
        payload = _mapping_or_empty(variants.get(variant_name))
        summary_payload = _mapping_or_empty(payload.get("summary"))
        if not payload:
            continue
        eval_reflex_scale = _coerce_optional_float(summary_payload.get("eval_reflex_scale"))
        if eval_reflex_scale is None:
            continue
        rows.append(
            {
                "rung": rung,
                "variant": variant_name,
                "eval_reflex_scale": eval_reflex_scale,
                "no_reflex_evaluated": abs(eval_reflex_scale) <= 1e-6,
                "scenario_success_rate": _coerce_optional_float(
                    summary_payload.get("scenario_success_rate")
                ),
            }
        )
    limitations: list[str] = []
    if not claim_result:
        limitations.append(
            "Claim test result for escape_without_reflex_support was unavailable."
        )
    if not rows:
        limitations.append(
            "No ladder variants exposed eval_reflex_scale information for no-reflex competence."
        )
    return {
        "available": bool(claim_result or rows),
        "claim_result": claim_result,
        "rungs": rows,
        "limitations": limitations,
    }


def _extract_capability_probe_boundaries(
    reward_profile_ladder: Mapping[str, object],
) -> dict[str, object]:
    ordered_scenarios = [
        scenario_name
        for level in ("gate", "warning", "diagnostic")
        for scenario_name, requirement in SCENARIO_AUSTERE_REQUIREMENTS.items()
        if str(_mapping_or_empty(requirement).get("requirement_level") or "") == level
    ]
    minimal_profile = str(reward_profile_ladder.get("minimal_profile") or "austere")
    profile_payload = _mapping_or_empty(
        _mapping_or_empty(reward_profile_ladder.get("profiles")).get(minimal_profile)
    )
    variants = _mapping_or_empty(profile_payload.get("variants"))
    rows: list[dict[str, object]] = []
    for scenario_name in ordered_scenarios:
        rung_success: list[dict[str, object]] = []
        for rung in LADDER_ACTIVE_RUNGS:
            variant_name = LADDER_PRIMARY_VARIANT_BY_RUNG.get(rung, "")
            variant_payload = _mapping_or_empty(variants.get(variant_name))
            suite = _mapping_or_empty(variant_payload.get("suite"))
            scenario_payload = _mapping_or_empty(suite.get(scenario_name))
            success_rate = _coerce_optional_float(scenario_payload.get("success_rate"))
            if success_rate is None:
                continue
            rung_success.append(
                {
                    "rung": rung,
                    "success_rate": success_rate,
                }
            )
        competent_rungs = [
            item["rung"]
            for item in rung_success
            if _coerce_float(item.get("success_rate"), 0.0) >= 0.5
        ]
        rows.append(
            {
                "scenario": scenario_name,
                "requirement_level": str(
                    _mapping_or_empty(
                        SCENARIO_AUSTERE_REQUIREMENTS.get(scenario_name)
                    ).get("requirement_level")
                    or ""
                ),
                "first_competent_rung": competent_rungs[0] if competent_rungs else None,
                "highest_competent_rung": competent_rungs[-1] if competent_rungs else None,
                "rung_success": rung_success,
                "rationale": str(
                    _mapping_or_empty(
                        SCENARIO_AUSTERE_REQUIREMENTS.get(scenario_name)
                    ).get("rationale")
                    or ""
                ),
            }
        )
    limitations: list[str] = []
    if not variants:
        limitations.append(
            "Capability probe boundaries were unavailable because austere ladder profile results were missing."
        )
    return {
        "available": bool(variants),
        "minimal_profile": minimal_profile,
        "rows": rows,
        "limitations": limitations,
    }


def _overall_modularity_comparison(
    ablations: Mapping[str, object],
) -> dict[str, object]:
    variants = {
        str(name): payload
        for name, payload in _mapping_or_empty(ablations.get("variants")).items()
        if isinstance(payload, Mapping)
    }
    baseline_variant = LADDER_PRIMARY_VARIANT_BY_RUNG["A0"]
    comparison_variant = LADDER_PRIMARY_VARIANT_BY_RUNG["A4"]
    baseline_payload = _mapping_or_empty(variants.get(baseline_variant))
    comparison_payload = _mapping_or_empty(variants.get(comparison_variant))
    if not baseline_payload or not comparison_payload:
        return {
            "available": False,
            "baseline_rung": "A0",
            "comparison_rung": "A4",
            "baseline_variant": baseline_variant,
            "comparison_variant": comparison_variant,
            "delta": None,
            "delta_ci_lower": None,
            "delta_ci_upper": None,
            "cohens_d": None,
            "magnitude_label": "",
            "n_seeds": 0,
            "limitations": [
                "A0->A4 comparison was unavailable because one of the endpoint rungs was missing."
            ],
        }
    effect = _effect_summary(
        baseline_variant=baseline_variant,
        comparison_variant=comparison_variant,
        baseline_payload=baseline_payload,
        comparison_payload=comparison_payload,
        source="summary.behavior_evaluation.ablations.ladder.A0_vs_A4",
    )
    return {
        "available": True,
        "baseline_rung": "A0",
        "comparison_rung": "A4",
        "baseline_variant": baseline_variant,
        "comparison_variant": comparison_variant,
        "delta": _coerce_optional_float(effect.get("delta")),
        "delta_ci_lower": _coerce_optional_float(effect.get("delta_ci_lower")),
        "delta_ci_upper": _coerce_optional_float(effect.get("delta_ci_upper")),
        "cohens_d": _coerce_optional_float(effect.get("cohens_d")),
        "magnitude_label": str(effect.get("magnitude_label") or ""),
        "n_seeds": int(_coerce_float(effect.get("n_seeds"), 0.0)),
        "confidence_level": _coerce_optional_float(effect.get("confidence_level")),
        "limitations": [],
    }


def compute_modularity_conclusion(
    unified_ladder_report: Mapping[str, object],
) -> dict[str, object]:
    capacity = _mapping_or_empty(unified_ladder_report.get("capacity_matched_comparison"))
    interface = _mapping_or_empty(
        unified_ladder_report.get("interface_sufficiency_results")
    )
    overall = _mapping_or_empty(unified_ladder_report.get("overall_comparison"))
    confounds: list[str] = []

    capacity_ratio = _coerce_optional_float(capacity.get("ratio"))
    if (
        _coerce_bool(capacity.get("available"))
        and capacity_ratio is not None
        and capacity_ratio > CAPACITY_MATCH_RATIO_THRESHOLD
    ):
        confounds.append(
            f"Capacity ratio across ladder rungs was {capacity_ratio:.2f}x, above the {CAPACITY_MATCH_RATIO_THRESHOLD:.2f}x match threshold."
        )
    if _coerce_bool(interface.get("any_interface_insufficient")):
        confounds.append(
            "At least one module-local sufficiency result reported interface_insufficient."
        )
    if confounds:
        return {
            "conclusion": CAPACITY_INTERFACE_CONFOUNDED,
            "conclusion_rationale": " ".join(confounds),
            "confounds": confounds,
        }

    delta = _coerce_optional_float(overall.get("delta"))
    ci_lower = _coerce_optional_float(overall.get("delta_ci_lower"))
    ci_upper = _coerce_optional_float(overall.get("delta_ci_upper"))
    effect_size = _coerce_optional_float(overall.get("cohens_d"))
    if delta is None or effect_size is None or ci_lower is None or ci_upper is None:
        return {
            "conclusion": MODULARITY_INCONCLUSIVE,
            "conclusion_rationale": (
                "A0->A4 effect estimation was incomplete, so modularity could not be adjudicated."
            ),
            "confounds": [],
        }

    ci_excludes_zero = ci_lower > 0.0 or ci_upper < 0.0
    if delta > 0.0 and effect_size >= 0.2 and ci_lower > 0.0 and ci_excludes_zero:
        return {
            "conclusion": MODULARITY_SUPPORTED,
            "conclusion_rationale": (
                f"A4 exceeded A0 by {delta:.2f} scenario_success_rate with Cohen's d {effect_size:.2f} "
                f"and CI [{ci_lower:.2f}, {ci_upper:.2f}] excluding zero."
            ),
            "confounds": [],
        }
    if delta < 0.0 and effect_size <= -0.2 and ci_upper < 0.0 and ci_excludes_zero:
        return {
            "conclusion": MODULARITY_HARMFUL,
            "conclusion_rationale": (
                f"A4 underperformed A0 by {abs(delta):.2f} scenario_success_rate with Cohen's d {effect_size:.2f} "
                f"and CI [{ci_lower:.2f}, {ci_upper:.2f}] excluding zero."
            ),
            "confounds": [],
        }
    return {
        "conclusion": MODULARITY_INCONCLUSIVE,
        "conclusion_rationale": (
            f"A0->A4 delta was {delta:.2f} with Cohen's d {effect_size:.2f} and CI "
            f"[{ci_lower:.2f}, {ci_upper:.2f}], which is too small or too uncertain to support a strong modularity claim."
        ),
        "confounds": [],
    }


def detect_missing_experiments(
    unified_ladder_report: Mapping[str, object],
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    ladder_table = _mapping_or_empty(unified_ladder_report.get("ladder_table"))
    ladder_rows = [
        row
        for row in ladder_table.get("rows", [])
        if isinstance(row, Mapping)
    ]
    missing_rungs = [
        f"{(row.get('rung') or '')!s}"
        for row in ladder_rows
        if not _coerce_bool(row.get("present"))
    ]
    if missing_rungs:
        missing.append(
            {
                "experiment_name": "complete_architectural_ladder",
                "description": "Run the missing ladder rungs: " + ", ".join(missing_rungs) + ".",
                "impact_on_conclusion": "Without A0-A4 coverage, modularity claims remain partial.",
            }
        )

    shaping = _mapping_or_empty(unified_ladder_report.get("reward_shaping_sensitivity"))
    missing_profiles = sorted(
        {"classic", "ecological", "austere"}
        - {str(item) for item in shaping.get("profiles_available", [])}
    )
    if missing_profiles:
        missing.append(
            {
                "experiment_name": "cross_profile_ladder_runs",
                "description": "Add ladder runs for reward profiles: " + ", ".join(missing_profiles) + ".",
                "impact_on_conclusion": "Reward shaping sensitivity remains under-measured.",
            }
        )

    low_seed_rungs = [
        f"{(row.get('rung') or '')!s}({int(_coerce_float(row.get('n_seeds'), 0.0))})"
        for row in ladder_rows
        if _coerce_bool(row.get("present")) and int(_coerce_float(row.get("n_seeds"), 0.0)) < 5
    ]
    if low_seed_rungs:
        missing.append(
            {
                "experiment_name": "five_seed_ladder_replication",
                "description": "Increase seed coverage to at least 5 for: " + ", ".join(low_seed_rungs) + ".",
                "impact_on_conclusion": "Confidence intervals and effect sizes are too fragile for strong claims.",
            }
        )

    interface = _mapping_or_empty(
        unified_ladder_report.get("interface_sufficiency_results")
    )
    if not _coerce_bool(interface.get("available")):
        missing.append(
            {
                "experiment_name": "module_interface_sufficiency_suite",
                "description": "Run module-local interface sufficiency tests and attach the per-module reports to the summary JSON.",
                "impact_on_conclusion": "Interface bottlenecks cannot be separated cleanly from architectural effects.",
            }
        )

    no_reflex = _mapping_or_empty(unified_ladder_report.get("no_reflex_competence"))
    no_reflex_rungs = [
        f"{(row.get('rung') or '')!s}"
        for row in no_reflex.get("rungs", [])
        if isinstance(row, Mapping) and not _coerce_bool(row.get("no_reflex_evaluated"))
    ]
    if no_reflex_rungs or not _mapping_or_empty(no_reflex.get("claim_result")):
        parts: list[str] = []
        if no_reflex_rungs:
            parts.append(
                "evaluate zero-reflex ladder variants for " + ", ".join(no_reflex_rungs)
            )
        if not _mapping_or_empty(no_reflex.get("claim_result")):
            parts.append("add escape_without_reflex_support claim results")
        missing.append(
            {
                "experiment_name": "no_reflex_competence_validation",
                "description": "; ".join(parts) + ".",
                "impact_on_conclusion": "Modularity cannot be distinguished from reflex support dependence.",
            }
        )

    credit = _mapping_or_empty(
        unified_ladder_report.get("credit_assignment_comparison")
    )
    missing_credit = []
    for rung in MODULAR_CREDIT_RUNGS:
        present = {
            str(item)
            for item in _mapping_or_empty(credit.get("strategies_by_rung")).get(rung, [])
        }
        absent = sorted({"broadcast", "local_only", "counterfactual"} - present)
        if absent:
            missing_credit.append(f"{rung}:{','.join(absent)}")
    if missing_credit:
        missing.append(
            {
                "experiment_name": "credit_assignment_variants",
                "description": "Add missing credit variants by rung: " + "; ".join(missing_credit) + ".",
                "impact_on_conclusion": "Credit-assignment differences remain entangled with architectural changes.",
            }
        )

    capacity = _mapping_or_empty(unified_ladder_report.get("capacity_matched_comparison"))
    if _coerce_bool(capacity.get("available")) and not _coerce_bool(
        capacity.get("capacity_matched")
    ):
        missing.append(
            {
                "experiment_name": "capacity_matched_ladder",
                "description": "Re-run A0-A4 with matched parameter counts or an explicit capacity sweep at each rung.",
                "impact_on_conclusion": "Capacity differences can dominate the ladder outcome.",
            }
        )
    return missing


def extract_unified_ladder_report(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    ablations = extract_ablations(summary, behavior_rows)
    architecture_capacity = extract_architecture_capacity(summary)
    reward_profile_ladder = extract_reward_profile_ladder(summary)

    ladder_rows = _build_ladder_rows(
        ablations=ablations,
        architecture_capacity=architecture_capacity,
        reward_profile_ladder=reward_profile_ladder,
    )
    adjacent_comparisons = _extract_adjacent_comparisons(ablations)
    capacity_comparison = _extract_capacity_comparison(ladder_rows)
    interface_results = _extract_interface_sufficiency_results(summary)
    credit_comparison = _extract_credit_assignment_comparison(summary, behavior_rows)
    reward_shaping_sensitivity = _extract_reward_shaping_sensitivity(
        reward_profile_ladder
    )
    no_reflex_competence = _extract_no_reflex_competence(summary, ablations)
    capability_probe_boundaries = _extract_capability_probe_boundaries(
        reward_profile_ladder
    )
    overall_comparison = _overall_modularity_comparison(ablations)

    limitations: list[str] = []
    for section in (
        ablations,
        architecture_capacity,
        reward_profile_ladder,
        interface_results,
        credit_comparison,
        reward_shaping_sensitivity,
        no_reflex_competence,
        capability_probe_boundaries,
        overall_comparison,
    ):
        limitations.extend(
            str(item)
            for item in section.get("limitations", [])
            if item
        )

    report = {
        "available": bool(
            any(_coerce_bool(row.get("present")) for row in ladder_rows)
            or adjacent_comparisons
        ),
        "ladder_table": _table(
            (
                "rung",
                "protocol_name",
                "variant",
                "total_trainable",
                "scenario_success_rate",
                "scenario_success_rate_ci_lower",
                "scenario_success_rate_ci_upper",
                "n_seeds",
                "delta_vs_previous",
                "delta_ci_lower_vs_previous",
                "delta_ci_upper_vs_previous",
                "cohens_d_vs_previous",
                "effect_magnitude_vs_previous",
                "capacity_status",
                "shaping_classification",
                "limitations",
            ),
            ladder_rows,
        ),
        "adjacent_comparisons": adjacent_comparisons,
        "overall_comparison": overall_comparison,
        "capacity_matched_comparison": capacity_comparison,
        "interface_sufficiency_results": interface_results,
        "credit_assignment_comparison": credit_comparison,
        "reward_shaping_sensitivity": reward_shaping_sensitivity,
        "no_reflex_competence": no_reflex_competence,
        "capability_probe_boundaries": capability_probe_boundaries,
        "limitations": limitations,
    }
    report["missing_experiments"] = detect_missing_experiments(report)
    report.update(compute_modularity_conclusion(report))
    return report
