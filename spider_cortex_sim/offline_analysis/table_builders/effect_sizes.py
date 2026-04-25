from __future__ import annotations

from .common import *
from .claims import build_claim_test_tables

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

__all__ = [name for name in globals() if not name.startswith("__")]
