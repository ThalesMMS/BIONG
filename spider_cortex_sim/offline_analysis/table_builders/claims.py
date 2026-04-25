from __future__ import annotations

from .common import *

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
        if effect_values is not None:
            effect_uncertainty = _mapping_or_empty(
                result.get("effect_size_uncertainty")
            )
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
                    "cohens_d": _coerce_optional_float(
                        cohens_d_values.get(condition_name)
                    ),
                    "effect_magnitude": str(magnitude_values.get(condition_name) or ""),
                }
                row.update(
                    _ci_row_fields(
                        _claim_uncertainty_for_condition(
                            effect_uncertainty,
                            condition_name,
                        ),
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

__all__ = [name for name in globals() if not name.startswith("__")]
