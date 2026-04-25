from __future__ import annotations

import math
from typing import Mapping, Sequence

from .config import *

def _safe_float(value: object) -> float:
    """
    Coerces a value to a finite float, falling back to 0.0 for invalid or non-finite inputs.
    
    Parameters:
        value (object): The value to convert to float.
    
    Returns:
        float: The finite float representation of `value`, or `0.0` if `value` cannot be converted or is not finite.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    return result


def _finite_float_or_none(value: object) -> float | None:
    """
    Convert a value to a finite float if possible.
    
    Returns:
        float | None: the converted finite float, or `None` if the value cannot be converted to a float or the resulting float is not finite.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _mean(values: Sequence[float]) -> float:
    """
    Compute the arithmetic mean of a sequence of floats.
    
    If the sequence is empty, returns 0.0.
    
    Parameters:
        values (Sequence[float]): Sequence of numeric values to average.
    
    Returns:
        float: The arithmetic mean of the input values, or 0.0 if the sequence is empty.
    """
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _scenario_success_rate(payload: Mapping[str, object], scenario_name: str) -> float | None:
    """
    Return a scenario's success rate from a payload that may use current ("suite") or legacy ("legacy_scenarios") structures.
    
    Looks for the scenario under payload["suite"][scenario_name] first, then payload["legacy_scenarios"][scenario_name]; if a mapping entry contains a "success_rate" key that can be converted to a finite float, that float is returned.
    
    Parameters:
        payload (Mapping[str, object]): Mapping that may contain a "suite" mapping or a "legacy_scenarios" mapping where scenario names map to mappings that may include "success_rate".
        scenario_name (str): Name of the scenario to look up.
    
    Returns:
        The scenario's success rate as a finite float if present and valid, `None` otherwise.
    """
    suite = payload.get("suite", {})
    if isinstance(suite, Mapping):
        scenario_payload = suite.get(scenario_name)
        if isinstance(scenario_payload, Mapping):
            if "success_rate" in scenario_payload:
                return _finite_float_or_none(scenario_payload.get("success_rate"))
    legacy_scenarios = payload.get("legacy_scenarios", {})
    if isinstance(legacy_scenarios, Mapping):
        scenario_payload = legacy_scenarios.get(scenario_name)
        if isinstance(scenario_payload, Mapping):
            if "success_rate" in scenario_payload:
                return _finite_float_or_none(scenario_payload.get("success_rate"))
    return None


def _safe_diff(a: float | None, b: float | None, decimals: int = 6) -> float | None:
    """
    Compute round(a - b, decimals) when both values are present, otherwise return None.
    """
    if a is None or b is None:
        return None
    return round(a - b, decimals)


def _group_metric_value(
    group: object,
    metric_name: str,
    *,
    count_key: str,
) -> float | None:
    """
    Retrieve a finite metric value from a mapping only when the associated count is greater than zero.
    
    Parameters:
        group (Mapping): Mapping expected to contain the metric and a count keyed by `count_key`.
        metric_name (str): Key name of the metric to read from `group`.
        count_key (str): Key name of the integer count in `group` that must be > 0 for the metric to be considered valid.
    
    Returns:
        float | None: The metric converted to a finite float if present and `int(group[count_key]) > 0`, `None` otherwise.
    """
    if not isinstance(group, Mapping):
        return None
    try:
        count = int(group.get(count_key, 0))
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    return _finite_float_or_none(group.get(metric_name))


def _variant_delta_payload(
    deltas_vs_reference: object,
    variant_name: str,
) -> Mapping[str, object]:
    """Return the delta payload for a variant across current and legacy shapes."""
    if not isinstance(deltas_vs_reference, Mapping):
        return {}
    prefix = f"{variant_name}_vs_"
    for delta_key, delta_payload in deltas_vs_reference.items():
        if str(delta_key).startswith(prefix) and isinstance(delta_payload, Mapping):
            return delta_payload
    legacy_payload = deltas_vs_reference.get(variant_name, {})
    if isinstance(legacy_payload, Mapping):
        return legacy_payload
    return {}


def compare_predator_type_ablation_performance(
    ablations_payload: Mapping[str, object],
    *,
    variant_names: Sequence[str] = ("drop_visual_cortex", "drop_sensory_cortex"),
) -> dict[str, object]:
    """
    Compute per-variant comparisons of mean success rates and success-rate deltas across visual and olfactory predator scenario groups.
    
    Given an ablations payload mapping (expected to contain "variants" and optional "deltas_vs_reference"), this function iterates the requested variant names and, for each variant present, computes for each multi-predator scenario group:
    - scenario_names: list of scenarios in the group
    - scenario_count: number of scenarios with a valid success rate
    - mean_success_rate: arithmetic mean of available success rates
    - mean_success_rate_delta: arithmetic mean of available success-rate deltas
    
    For each variant the result also includes:
    - visual_minus_olfactory_success_rate: difference between the visual and olfactory group mean success rates, rounded to 6 decimals
    - visual_minus_olfactory_success_rate_delta: difference between the visual and olfactory group mean deltas, rounded to 6 decimals
    
    If the payload does not contain a mapping under "variants", the function returns available=False and an empty comparisons mapping.
    
    Parameters:
        ablations_payload (Mapping[str, object]): Mapping expected to contain "variants" (mapping of variant name to variant payload) and optionally "deltas_vs_reference" (mapping of variant name to per-scenario deltas).
        variant_names (Sequence[str]): Ordered sequence of variant names to process; only variants present and mapping-like in the payload are included.
    
    Returns:
        dict[str, object]: A dictionary with:
            - "available" (bool): True if any comparisons were produced, False otherwise.
            - "scenario_groups" (dict): The canonical mapping of named scenario groups to their scenario tuples.
            - "comparisons" (dict): Mapping from variant name to the computed per-group rows and the visual-minus-olfactory differences.
    """
    variants = ablations_payload.get("variants", {})
    deltas_vs_reference = ablations_payload.get("deltas_vs_reference", {})
    if not isinstance(variants, Mapping):
        return {
            "available": False,
            "scenario_groups": dict(MULTI_PREDATOR_SCENARIO_GROUPS),
            "comparisons": {},
        }

    comparisons: dict[str, object] = {}
    for variant_name in variant_names:
        payload = variants.get(variant_name, {})
        if not isinstance(payload, Mapping):
            continue
        variant_delta_payload = _variant_delta_payload(
            deltas_vs_reference,
            str(variant_name),
        )
        variant_delta_scenarios = variant_delta_payload.get("scenarios", {})
        group_rows: dict[str, object] = {}
        for group_name, scenario_names in MULTI_PREDATOR_SCENARIO_GROUPS.items():
            scenario_rates = [
                rate
                for scenario_name in scenario_names
                if (rate := _scenario_success_rate(payload, scenario_name)) is not None
            ]
            scenario_deltas: list[float] = []
            for scenario_name in scenario_names:
                if not isinstance(variant_delta_scenarios, Mapping):
                    continue
                scenario_delta = variant_delta_scenarios.get(scenario_name, {})
                if not isinstance(scenario_delta, Mapping):
                    continue
                raw_delta = scenario_delta.get("scenario_success_rate_delta")
                parsed_delta = _finite_float_or_none(raw_delta)
                if parsed_delta is not None:
                    scenario_deltas.append(parsed_delta)
            group_rows[group_name] = {
                "scenario_names": list(scenario_names),
                "scenario_count": len(scenario_rates),
                "scenario_delta_count": len(scenario_deltas),
                "mean_success_rate": _mean(scenario_rates),
                "mean_success_rate_delta": _mean(scenario_deltas),
            }
        if not any(
            int(group.get("scenario_count", 0)) > 0
            for group in group_rows.values()
            if isinstance(group, Mapping)
        ):
            continue
        visual_group = group_rows.get("visual_predator_scenarios", {})
        olfactory_group = group_rows.get("olfactory_predator_scenarios", {})
        visual_success_rate = _group_metric_value(
            visual_group,
            "mean_success_rate",
            count_key="scenario_count",
        )
        olfactory_success_rate = _group_metric_value(
            olfactory_group,
            "mean_success_rate",
            count_key="scenario_count",
        )
        visual_success_rate_delta = _group_metric_value(
            visual_group,
            "mean_success_rate_delta",
            count_key="scenario_delta_count",
        )
        olfactory_success_rate_delta = _group_metric_value(
            olfactory_group,
            "mean_success_rate_delta",
            count_key="scenario_delta_count",
        )
        comparisons[str(variant_name)] = {
            **group_rows,
            "visual_minus_olfactory_success_rate": _safe_diff(
                visual_success_rate, olfactory_success_rate
            ),
            "visual_minus_olfactory_success_rate_delta": _safe_diff(
                visual_success_rate_delta, olfactory_success_rate_delta
            ),
        }

    return {
        "available": bool(comparisons),
        "scenario_groups": dict(MULTI_PREDATOR_SCENARIO_GROUPS),
        "comparisons": comparisons,
    }

__all__ = ["compare_predator_type_ablation_performance"]
