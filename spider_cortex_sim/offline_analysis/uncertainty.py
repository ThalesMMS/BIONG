from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from numbers import Real

from ..statistics import bootstrap_confidence_interval, cohens_d
from .constants import (
    BOOTSTRAP_RANDOM_SEED,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
)
from .utils import (
    _coerce_float,
    _coerce_optional_float,
    _mapping_or_empty,
    _mean,
)

def _uncertainty_or_empty(value: object) -> Mapping[str, object]:
    """
    Provide a mapping representation of `value`, or an empty mapping when `value` is not a mapping.
    
    Returns:
        mapping (Mapping[str, object]): `value` unchanged if it is a `Mapping`, otherwise an empty dictionary.
    """
    return value if isinstance(value, Mapping) else {}


def _ci_row_fields(
    uncertainty: Mapping[str, object] | None,
    *,
    value: object | None = None,
) -> dict[str, object]:
    """
    Build a standardized CI row dictionary from an uncertainty mapping or explicit value.
    
    Parameters:
        uncertainty (Mapping[str, object] | None): Optional uncertainty mapping containing keys like
            "mean", "ci_lower", "ci_upper", "std_error", "n_seeds", and "confidence_level".
            If not a mapping or None, it is treated as empty.
        value (object | None): Optional explicit numeric value to use for the row's `value` field.
            If coercible to a float, this overrides `uncertainty["mean"]`.
    
    Returns:
        dict[str, object]: A dictionary with the following keys:
            - "value": float | None — coerced numeric value (explicit `value` if provided, else `uncertainty["mean"]`).
            - "ci_lower": float | None — coerced lower confidence bound from `uncertainty`.
            - "ci_upper": float | None — coerced upper confidence bound from `uncertainty`.
            - "std_error": float | None — coerced standard error from `uncertainty`.
            - "n_seeds": int — integer number of seeds (defaults to 0 if missing).
            - "confidence_level": float | None — coerced confidence level from `uncertainty`.
    """
    payload = _uncertainty_or_empty(uncertainty)
    row_value = _coerce_optional_float(value)
    if row_value is None:
        row_value = _coerce_optional_float(payload.get("mean"))
    return {
        "value": row_value,
        "ci_lower": _coerce_optional_float(payload.get("ci_lower")),
        "ci_upper": _coerce_optional_float(payload.get("ci_upper")),
        "std_error": _coerce_optional_float(payload.get("std_error")),
        "n_seeds": int(_coerce_float(payload.get("n_seeds"), 0.0)),
        "confidence_level": _coerce_optional_float(payload.get("confidence_level")),
    }


def _seed_key(value: object) -> str | None:
    """
    Normalize a seed identifier into a string key or None.
    
    Parameters:
    	value (object): Seed identifier to normalize. If `None` or an empty string, it is treated as missing.
    
    Returns:
    	seed_key (str | None): String representation of `value`, or `None` when `value` is `None` or `""`.
    """
    if value in (None, ""):
        return None
    return str(value)


def _seed_value_items(values: Sequence[object]) -> list[tuple[str | None, float]]:
    """
    Extract numeric seed/value pairs from a sequence of heterogeneous items.
    
    Parameters:
        values (Sequence[object]): Sequence of items in any of these forms:
            - Mapping with keys "seed" and "value"
            - 2-element tuple/list (seed, value)
            - a raw value (seed will be None)
        Only entries whose value coerces to a numeric float are included.
    
    Returns:
        list[tuple[str | None, float]]: List of (seed, value) pairs where `seed` is a string or None
        (empty or missing seeds yield None) and `value` is a float.
    """
    items: list[tuple[str | None, float]] = []
    for item in values:
        seed: object | None = None
        raw_value: object = item
        if isinstance(item, Mapping):
            summary = item.get("summary")
            if isinstance(summary, Mapping):
                seed = summary.get("seed", item.get("seed"))
                raw_value = summary.get("value", item.get("value"))
            else:
                seed = item.get("seed")
                raw_value = item.get("value")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            seed, raw_value = item
        numeric_value = _coerce_optional_float(raw_value)
        if numeric_value is not None:
            items.append((_seed_key(seed), numeric_value))
    return items


def _values_from_seed_items(items: Sequence[tuple[str | None, float]]) -> list[float]:
    """
    Extract numeric values from a sequence of (seed, value) pairs preserving their order.
    
    Parameters:
    	items (Sequence[tuple[str | None, float]]): Sequence of pairs where the first element is a seed identifier (string or None) and the second is a numeric value.
    
    Returns:
    	list[float]: List of the numeric values extracted from each pair, in the same order as provided.
    """
    return [value for _seed, value in items]


def _sample_std(values: Sequence[float]) -> float:
    """
    Compute the sample standard deviation of a sequence of numbers.
    
    Parameters:
        values (Sequence[float]): Numeric observations.
    
    Returns:
        float: Sample standard deviation computed with denominator (n - 1); returns 0.0 if fewer than two values.
    """
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """
    Compute the quantile value from an ascending-sorted sequence using linear interpolation.
    
    Parameters:
        sorted_values (Sequence[float]): Ascending-sorted numeric samples to query.
        quantile (float): Desired quantile in [0, 1], where 0 yields the minimum and 1 yields the maximum.
    
    Returns:
        float: The quantile value computed by linear interpolation between neighboring samples; if `sorted_values` contains a single element, that element is returned.
    """
    if not sorted_values:
        raise ValueError("sorted_values must be non-empty")
    if (
        not isinstance(quantile, Real)
        or isinstance(quantile, bool)
        or not 0.0 <= quantile <= 1.0
    ):
        raise ValueError("quantile must be between 0 and 1")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = float(quantile) * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return (
        float(sorted_values[lower_index]) * (1.0 - fraction)
        + float(sorted_values[upper_index]) * fraction
    )


def _bootstrap_distribution_fields(values: Sequence[float]) -> dict[str, object]:
    """
    Builds bootstrap-style interval fields from a sequence of numeric samples.
    
    Returns:
        A dictionary with keys:
          - `ci_lower`: lower confidence bound computed at the lower tail of `DEFAULT_CONFIDENCE_LEVEL`.
          - `ci_upper`: upper confidence bound computed at the upper tail of `DEFAULT_CONFIDENCE_LEVEL`.
          - `std_error`: standard error of the samples (sample standard deviation using n-1).
        Returns an empty dictionary if `values` is empty.
    """
    if not values:
        return {}
    sorted_values = sorted(float(value) for value in values)
    tail_probability = (1.0 - DEFAULT_CONFIDENCE_LEVEL) / 2.0
    return {
        "ci_lower": _percentile(sorted_values, tail_probability),
        "ci_upper": _percentile(sorted_values, 1.0 - tail_probability),
        "std_error": _sample_std(sorted_values),
    }


def _unpaired_delta_uncertainty_from_seed_items(
    baseline_items: Sequence[tuple[str | None, float]],
    comparison_items: Sequence[tuple[str | None, float]],
    *,
    raw_delta: object | None = None,
) -> dict[str, object]:
    """
    Compute a bootstrap-based confidence interval for the unpaired difference in means between two groups of seeded values.
    
    Parameters:
        baseline_items (Sequence[tuple[str | None, float]]): Sequence of (seed, value) pairs for the baseline group; seed may be None or empty string.
        comparison_items (Sequence[tuple[str | None, float]]): Sequence of (seed, value) pairs for the comparison group; seed may be None or empty string.
        raw_delta (object | None): Optional explicit point estimate to use for the mean difference; if not coercible to a float, the difference of group means is used.
    
    Returns:
        dict[str, object]: A mapping with bootstrap-derived uncertainty fields, or an empty dict when either group has no numeric values. When non-empty, the dictionary contains:
            - "mean": the point estimate for the mean difference (uses `raw_delta` if provided/coercible, otherwise comparison_mean - baseline_mean).
            - "ci_lower": lower bound of the confidence interval (or None if not available).
            - "ci_upper": upper bound of the confidence interval (or None if not available).
            - "std_error": bootstrap standard error of the difference (or None if not available).
            - "n_seeds": the effective sample size used (minimum of baseline and comparison sample sizes).
            - "confidence_level": the confidence level used to compute the interval.
            - "seed_values": an empty list (pairwise seed deltas are not produced for the unpaired case).
    """
    baseline_values = _values_from_seed_items(baseline_items)
    comparison_values = _values_from_seed_items(comparison_items)
    if not baseline_values or not comparison_values:
        return {}
    rng = random.Random(BOOTSTRAP_RANDOM_SEED)
    bootstrap_deltas = [
        _mean(rng.choices(comparison_values, k=len(comparison_values)))
        - _mean(rng.choices(baseline_values, k=len(baseline_values)))
        for _ in range(DEFAULT_BOOTSTRAP_RESAMPLES)
    ]
    interval = _bootstrap_distribution_fields(bootstrap_deltas)
    point_estimate = _coerce_optional_float(raw_delta)
    if point_estimate is None:
        point_estimate = _mean(comparison_values) - _mean(baseline_values)
    return {
        "mean": point_estimate,
        "ci_lower": interval.get("ci_lower"),
        "ci_upper": interval.get("ci_upper"),
        "std_error": interval.get("std_error"),
        "n_seeds": min(len(baseline_values), len(comparison_values)),
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "seed_values": [],
    }


def _delta_uncertainty_from_seed_values(
    baseline_values: Sequence[object],
    comparison_values: Sequence[object],
    *,
    raw_delta: object | None = None,
) -> dict[str, object]:
    """
    Compute a bootstrap confidence interval for the delta (comparison - baseline) using seeded values, preferring paired seeds.
    
    Parameters:
        baseline_values (Sequence[object]): Sequence of items representing baseline seed/value entries (accepted shapes are those parseable by _seed_value_items).
        comparison_values (Sequence[object]): Sequence of items representing comparison seed/value entries (accepted shapes are those parseable by _seed_value_items).
        raw_delta (object | None): Optional value to use as the point estimate instead of the bootstrap mean (coerced if possible).
    
    Returns:
        dict[str, object]: A mapping containing:
            - "mean": the point estimate (raw_delta if provided and coercible, otherwise bootstrap mean),
            - "ci_lower": lower confidence bound,
            - "ci_upper": upper confidence bound,
            - "std_error": standard error of the bootstrap distribution,
            - "n_seeds": number of paired seeds used,
            - "confidence_level": confidence level used for intervals,
            - "seed_values": list of per-seed delta objects with keys "seed" and "value".
        Returns an empty dict if either input has no usable numeric entries; if no paired seeds are found, an unpaired bootstrap uncertainty is returned instead.
    """
    baseline_items = _seed_value_items(baseline_values)
    comparison_items = _seed_value_items(comparison_values)
    if not baseline_items or not comparison_items:
        return {}

    baseline_by_seed = {
        seed: value
        for seed, value in baseline_items
        if seed is not None
    }
    comparison_by_seed = {
        seed: value
        for seed, value in comparison_items
        if seed is not None
    }
    shared_seeds = sorted(set(baseline_by_seed) & set(comparison_by_seed))
    if not shared_seeds:
        return _unpaired_delta_uncertainty_from_seed_items(
            baseline_items,
            comparison_items,
            raw_delta=raw_delta,
        )

    seed_deltas = [
        {
            "seed": seed,
            "value": comparison_by_seed[seed] - baseline_by_seed[seed],
        }
        for seed in shared_seeds
    ]
    deltas = [float(item["value"]) for item in seed_deltas]
    mean, ci_lower, ci_upper, std_error = bootstrap_confidence_interval(
        deltas,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        n_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
        random_seed=BOOTSTRAP_RANDOM_SEED,
    )
    point_estimate = _coerce_optional_float(raw_delta)
    if point_estimate is None:
        point_estimate = mean
    return {
        "mean": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "n_seeds": len(deltas),
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "seed_values": seed_deltas,
    }


def _uncertainty_or_seed_delta(
    uncertainty: object,
    baseline_values: Sequence[object],
    comparison_values: Sequence[object],
    *,
    raw_delta: object | None = None,
) -> Mapping[str, object]:
    """
    Return the provided uncertainty mapping if present; otherwise compute delta uncertainty from seeded baseline and comparison values.
    
    Parameters:
        uncertainty (object): An uncertainty payload or any value; if it is a mapping and non-empty it will be returned as-is.
        baseline_values (Sequence[object]): Sequence of baseline seed/value items (mappings, tuples, or raw values) to parse when computing delta uncertainty.
        comparison_values (Sequence[object]): Sequence of comparison seed/value items to parse when computing delta uncertainty.
        raw_delta (object | None): Optional raw delta value to prefer as the point estimate when computing uncertainty.
    
    Returns:
        Mapping[str, object]: A mapping with uncertainty fields (e.g., `mean`, `ci_lower`, `ci_upper`, `std_error`, `n_seeds`, `confidence_level`, `seed_values`) or an empty mapping when no usable data is available.
    """
    payload = _uncertainty_or_empty(uncertainty)
    if payload:
        return payload
    return _delta_uncertainty_from_seed_values(
        baseline_values,
        comparison_values,
        raw_delta=raw_delta,
    )


def _payload_uncertainty(
    payload: Mapping[str, object] | None,
    metric_name: str,
    *,
    scenario: str | None = None,
) -> Mapping[str, object]:
    """
    Extracts the uncertainty mapping for a given metric from a payload, optionally limited to a specific scenario.
    
    If a scenario is provided, the function first looks under payload["suite"][scenario]["uncertainty"] for an entry matching the metric name or "success_rate" (in that order). If not found there, it then checks payload["uncertainty"] for the same keys. If the payload is not a mapping or no matching uncertainty mapping is found, an empty mapping is returned.
    
    Parameters:
        payload: The result payload to inspect; may be None or a nested mapping.
        metric_name: The metric key to look up (checked before "success_rate").
        scenario: Optional scenario name to scope the search to the suite's scenario entry.
    
    Returns:
        A mapping containing the uncertainty data for the metric, or an empty mapping if none is found.
    """
    if not isinstance(payload, Mapping):
        return {}
    if scenario is not None:
        suite = _mapping_or_empty(payload.get("suite"))
        scenario_payload = _mapping_or_empty(suite.get(scenario))
        uncertainty = _mapping_or_empty(scenario_payload.get("uncertainty"))
        for key in (metric_name, "success_rate"):
            nested = uncertainty.get(key)
            if isinstance(nested, Mapping):
                return nested
    uncertainty = _mapping_or_empty(payload.get("uncertainty"))
    for key in (metric_name, "success_rate"):
        nested = uncertainty.get(key)
        if isinstance(nested, Mapping):
            return nested
    return {}


def _payload_metric_seed_items(
    payload: Mapping[str, object] | None,
    metric_name: str,
    *,
    scenario: str | None = None,
) -> list[tuple[str | None, float]]:
    """
    Extract seed/value pairs for a given metric from a payload, optionally restricted to a scenario.
    
    Parameters:
        payload (Mapping | None): Result payload containing top-level and optional suite scenario data.
        metric_name (str): Metric name to match against item["metric_name"].
        scenario (str | None): If provided, only include seed items for that scenario; if None, include only items with no scenario.
    
    Returns:
        list[tuple[str | None, float]]: A list of (seed_key, value) tuples where `seed_key` is a string or `None`
        (empty/None seeds normalized to `None`) and `value` is a numeric float. If no direct seed-level items
        are found, the function will look for `seed_values` inside the payload's uncertainty section and return
        parsed entries from there. An empty list is returned if no usable seed/value pairs are available.
    """
    if not isinstance(payload, Mapping):
        return []
    if scenario is not None:
        suite = _mapping_or_empty(payload.get("suite"))
        scenario_payload = _mapping_or_empty(suite.get(scenario))
        scenario_seed_level = scenario_payload.get("seed_level", [])
        if isinstance(scenario_seed_level, list):
            sources: list[object] = [
                scenario_seed_level,
                payload.get("seed_level", []),
            ]
        else:
            sources = [payload.get("seed_level", [])]
    else:
        sources = [payload.get("seed_level", [])]
    source_items: list[tuple[str | None, float]] = []
    for source in sources:
        if not isinstance(source, list):
            continue
        source_items = []
        for item in source:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("metric_name") or "") != metric_name:
                continue
            item_scenario = item.get("scenario")
            if scenario is None and item_scenario not in (None, ""):
                continue
            if (
                scenario is not None
                and item_scenario not in (None, "")
                and str(item_scenario) != str(scenario)
            ):
                continue
            numeric_value = _coerce_optional_float(item.get("value"))
            if numeric_value is not None:
                source_items.append((_seed_key(item.get("seed")), numeric_value))
        if source_items:
            return source_items
    items: list[tuple[str | None, float]] = []
    uncertainty = _payload_uncertainty(payload, metric_name, scenario=scenario)
    seed_values = uncertainty.get("seed_values") if isinstance(uncertainty, Mapping) else None
    if isinstance(seed_values, list):
        items = []
        for item in seed_values:
            seed = item.get("seed") if isinstance(item, Mapping) else None
            raw_value = item.get("value") if isinstance(item, Mapping) else item
            numeric_value = _coerce_optional_float(raw_value)
            if numeric_value is not None:
                items.append((_seed_key(seed), numeric_value))
        return items
    return []


def _payload_metric_seed_values(
    payload: Mapping[str, object] | None,
    metric_name: str,
    *,
    scenario: str | None = None,
) -> list[float]:
    """
    Extract numeric seed-level values for a given metric (and optional scenario) from a payload.
    
    Parameters:
        payload: The payload mapping to extract seed-level metric entries from; may be None.
        metric_name: Metric name to match against seed-level entries.
        scenario: If provided, only include seed entries for this scenario; if None, include top-level/no-scenario entries.
    
    Returns:
        A list of numeric metric values extracted from the payload's seed-level entries, or an empty list if none are found.
    """
    return _values_from_seed_items(
        _payload_metric_seed_items(payload, metric_name, scenario=scenario)
    )


def _mean_or_none(values: Sequence[float]) -> float | None:
    """
    Return the arithmetic mean of the sequence, or `None` if the sequence is empty.
    
    Returns:
        The arithmetic mean of `values` if `values` contains one or more numbers, `None` otherwise.
    """
    if not values:
        return None
    return _mean(values)


def _cohens_d_uncertainty_from_seed_values(
    baseline_values: Sequence[object],
    comparison_values: Sequence[object],
    *,
    effect_size: object | None = None,
) -> dict[str, object]:
    """
    Estimate Cohen's d and its bootstrap confidence interval from two groups of seeded values.
    
    Parameters:
    	baseline_values (Sequence[object]): Sequence of seed/value entries for the baseline group. Supported entry shapes include mappings with keys `seed` and `value`, 2-element (seed, value) pairs, or raw values (seeds will be treated as None). Values that cannot be coerced to numbers are ignored.
    	comparison_values (Sequence[object]): Sequence of seed/value entries for the comparison group, using the same accepted shapes as `baseline_values`.
    	effect_size (object | None): Optional numeric effect-size override to use as the point estimate; if not provided or not coercible to a float, the bootstrap mean is used.
    
    Returns:
    	details (dict[str, object]): A dictionary containing:
    		- `mean`: the point estimate (the provided `effect_size` if valid, otherwise the bootstrap mean).
    		- `ci_lower`, `ci_upper`: bootstrap confidence interval bounds (or `None` if not available).
    		- `std_error`: bootstrap standard error (or `None` if not available).
    		- `n_seeds`: the number of paired seeds used (minimum of group sizes).
    		- `confidence_level`: the confidence level used for the interval.
    		- `seed_values`: an empty list (placeholder for per-seed bootstrap values).
    	An empty dict is returned if either group contains no numeric values.
    """
    baseline_items = _seed_value_items(baseline_values)
    comparison_items = _seed_value_items(comparison_values)
    baseline_numeric = _values_from_seed_items(baseline_items)
    comparison_numeric = _values_from_seed_items(comparison_items)
    if not baseline_numeric or not comparison_numeric:
        return {}
    rng = random.Random(BOOTSTRAP_RANDOM_SEED)
    bootstrap_effects: list[float] = []
    for _ in range(DEFAULT_BOOTSTRAP_RESAMPLES):
        effect, _magnitude = cohens_d(
            rng.choices(comparison_numeric, k=len(comparison_numeric)),
            rng.choices(baseline_numeric, k=len(baseline_numeric)),
        )
        bootstrap_effects.append(effect)
    interval = _bootstrap_distribution_fields(bootstrap_effects)
    point_estimate = _coerce_optional_float(effect_size)
    if point_estimate is None:
        point_estimate = _mean(bootstrap_effects)
    return {
        "mean": point_estimate,
        "ci_lower": interval.get("ci_lower"),
        "ci_upper": interval.get("ci_upper"),
        "std_error": interval.get("std_error"),
        "n_seeds": min(len(baseline_numeric), len(comparison_numeric)),
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "seed_values": [],
    }


def _cohens_d_row(
    *,
    domain: str,
    baseline: str,
    comparison: str,
    metric: str,
    baseline_values: Sequence[object],
    comparison_values: Sequence[object],
    raw_delta: object | None,
    source: str,
    uncertainty: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """
    Assembles a combined Cohen's d effect-size row and delta confidence-interval fields from seeded baseline and comparison data.
    
    Parameters:
        domain (str): Domain identifier for the row.
        baseline (str): Baseline variant identifier.
        comparison (str): Comparison variant identifier.
        metric (str): Metric name.
        baseline_values (Sequence[object]): Sequence of seed-level baseline entries (mapping with `seed`/`value`, 2-tuple `(seed, value)`, or raw values).
        comparison_values (Sequence[object]): Sequence of seed-level comparison entries (same accepted shapes as `baseline_values`).
        raw_delta (object | None): Optional raw delta value to use as the delta point estimate if coercible to a number.
        source (str): Source identifier describing where the data came from.
        uncertainty (Mapping[str, object] | None): Optional precomputed delta uncertainty mapping; if not provided or empty, delta uncertainty will be computed from seed values when possible.
    
    Returns:
        dict[str, object] | None: A row dictionary containing identifiers and computed fields, or `None` when neither group has numeric values and `raw_delta` is not provided. Returned keys include:
          - "domain", "baseline", "comparison", "metric", "raw_delta", "cohens_d", "magnitude_label", "source"
          - Effect-size CI and stats: "effect_size_ci_lower", "effect_size_ci_upper", "effect_size_std_error", "effect_size_n_seeds", "effect_size_confidence_level" (and any additional effect uncertainty fields)
          - Delta CI and stats: "delta_ci_lower", "delta_ci_upper", "delta_std_error", "delta_n_seeds", "delta_confidence_level"
    """
    baseline_items = _seed_value_items(baseline_values)
    comparison_items = _seed_value_items(comparison_values)
    baseline_numeric = _values_from_seed_items(baseline_items)
    comparison_numeric = _values_from_seed_items(comparison_items)
    delta_value = _coerce_optional_float(raw_delta)
    if not baseline_numeric and not comparison_numeric and delta_value is None:
        return None
    if baseline_numeric and comparison_numeric:
        effect_size, magnitude = cohens_d(comparison_numeric, baseline_numeric)
        effect_size_value: float | None = round(effect_size, 6)
    else:
        effect_size_value = None
        magnitude = ""
    if delta_value is None:
        baseline_mean = _mean_or_none(baseline_numeric)
        comparison_mean = _mean_or_none(comparison_numeric)
        if baseline_mean is not None and comparison_mean is not None:
            delta_value = round(comparison_mean - baseline_mean, 6)
    effect_uncertainty = _cohens_d_uncertainty_from_seed_values(
        baseline_items,
        comparison_items,
        effect_size=effect_size_value,
    )
    effect_fields = _ci_row_fields(effect_uncertainty, value=effect_size_value)
    delta_uncertainty = _uncertainty_or_seed_delta(
        uncertainty,
        baseline_items,
        comparison_items,
        raw_delta=delta_value,
    )
    delta_fields = _ci_row_fields(delta_uncertainty, value=delta_value)
    row = {
        "domain": domain,
        "baseline": baseline,
        "comparison": comparison,
        "metric": metric,
        "raw_delta": delta_value,
        "cohens_d": effect_size_value,
        "magnitude_label": magnitude,
        "source": source,
        "effect_size_ci_lower": effect_fields["ci_lower"],
        "effect_size_ci_upper": effect_fields["ci_upper"],
        "effect_size_std_error": effect_fields["std_error"],
        "effect_size_n_seeds": effect_fields["n_seeds"],
        "effect_size_confidence_level": effect_fields["confidence_level"],
        "delta_ci_lower": delta_fields["ci_lower"],
        "delta_ci_upper": delta_fields["ci_upper"],
        "delta_std_error": delta_fields["std_error"],
        "delta_n_seeds": delta_fields["n_seeds"],
        "delta_confidence_level": delta_fields["confidence_level"],
    }
    # Keep legacy effect_fields names alongside the namespaced effect_size_* keys.
    row.update(effect_fields)
    return row


def _primary_benchmark_source_payload(
    summary: Mapping[str, object],
) -> tuple[str, str, Mapping[str, object]]:
    """
    Selects a primary benchmark payload and identifies its source within the provided summary.
    
    Given a top-level summary mapping, determines the most appropriate payload to use as the primary benchmark according to the module's priority rules:
    1. The reference variant under behavior_evaluation.ablations.variants (falling back to "modular_full" when present).
    2. evaluation_without_reflex_support.summary.
    3. behavior_evaluation.summary if the payload is marked as zero-reflex scale.
    4. evaluation.summary if the payload is marked as zero-reflex scale.
    If none of these yield a suitable payload, returns a sentinel indicating no selection.
    
    Parameters:
        summary (Mapping[str, object]): The top-level summary structure containing evaluation and behavior_evaluation entries.
    
    Returns:
        tuple[str, str, Mapping[str, object]]: A 3-tuple of
            - source: a dotted-path string identifying where the chosen payload came from (or "none" if no payload),
            - reference_variant: the reference variant name when selected, or an empty string when not applicable,
            - payload: the selected payload mapping (empty mapping if none was selected).
    """
    # Local import of _variant_with_minimal_reflex_support avoids an extractors.py cycle.
    from .extractors import _variant_with_minimal_reflex_support

    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    ablations = _mapping_or_empty(behavior_evaluation.get("ablations"))
    variants = _mapping_or_empty(ablations.get("variants"))
    reference_variant = str(ablations.get("reference_variant") or "")
    if not reference_variant and "modular_full" in variants:
        reference_variant = "modular_full"
    if (
        reference_variant
        and not variants.get(reference_variant)
        and "modular_full" in variants
    ):
        reference_variant = "modular_full"
    reference_payload = _mapping_or_empty(variants.get(reference_variant))
    if reference_payload:
        payload = _variant_with_minimal_reflex_support(reference_payload)
        summary_payload = _mapping_or_empty(payload.get("summary"))
        if "scenario_success_rate" in summary_payload:
            return (
                f"summary.behavior_evaluation.ablations.variants.{reference_variant}",
                reference_variant,
                payload,
            )

    no_reflex_eval = _mapping_or_empty(summary.get("evaluation_without_reflex_support"))
    no_reflex_summary = _mapping_or_empty(no_reflex_eval.get("summary"))
    if "scenario_success_rate" in no_reflex_summary:
        return (
            "summary.evaluation_without_reflex_support",
            "",
            no_reflex_eval,
        )

    behavior_summary = _mapping_or_empty(behavior_evaluation.get("summary"))
    if (
        "scenario_success_rate" in behavior_summary
        and _payload_has_zero_reflex_scale(behavior_evaluation)
    ):
        return (
            "summary.behavior_evaluation",
            "",
            behavior_evaluation,
        )

    evaluation = _mapping_or_empty(summary.get("evaluation"))
    evaluation_summary = _mapping_or_empty(evaluation.get("summary"))
    if (
        "scenario_success_rate" in evaluation_summary
        and _payload_has_zero_reflex_scale(evaluation)
    ):
        return ("summary.evaluation", "", evaluation)

    return ("none", "", {})


def _is_zero_reflex_scale(value: object) -> bool:
    """
    Determine whether a numeric reflex scale is effectively zero.
    
    Parameters:
        value (object): Value to coerce to float (e.g., numeric, numeric string, or None).
    
    Returns:
        `true` if the coerced value is finite and its absolute value is less than or equal to 1e-6, `false` otherwise.
    """
    scale = _coerce_float(value, math.nan)
    return math.isfinite(scale) and abs(scale) <= 1e-6


def _payload_has_zero_reflex_scale(payload: Mapping[str, object]) -> bool:
    """
    Determine whether a payload's reflex scale is effectively zero.
    
    Checks `summary["eval_reflex_scale"]` first (if `summary` is a mapping), then falls back to `payload["eval_reflex_scale"]`. The scale is considered "zero" if it can be coerced to a finite number whose absolute value is <= 1e-6.
    
    Parameters:
        payload (Mapping[str, object]): Payload mapping that may contain a nested `summary` mapping and/or an `eval_reflex_scale` key.
    
    Returns:
        bool: `true` if the reflex scale is finite and its absolute value is <= 1e-6, `false` otherwise.
    """
    summary_payload = _mapping_or_empty(payload.get("summary"))
    scale = summary_payload.get("eval_reflex_scale")
    if scale is None:
        scale = payload.get("eval_reflex_scale")
    return _is_zero_reflex_scale(scale)


def _primary_benchmark_scenario_success(
    summary: Mapping[str, object],
) -> dict[str, object]:
    """
    Select the scenario-success structure to use as the primary benchmark, disabling the primary-benchmark fallback when the behavior-evaluation suite is not marked as zero-reflex.
    
    Parameters:
        summary (Mapping[str, object]): The overall summary payload containing evaluation and behavior_evaluation sections.
    
    Returns:
        dict[str, object]: The chosen scenario-success structure. Typically this is the output of `extract_scenario_success(summary, ())`. If that output originates from `summary.behavior_evaluation.suite` but the suite is not marked with a zero reflex scale, returns an explicit unavailable structure with:
            - "available" (bool): False
            - "source" (str): source identifier (original or "summary.behavior_evaluation.suite")
            - "scenarios" (list): empty list
            - "limitations" (list): human-readable explanation why the fallback was disabled.
    """
    # Local import of extract_scenario_success avoids an extractors.py cycle.
    from .extractors import extract_scenario_success

    scenario_success = extract_scenario_success(summary, ())
    unavailable_suite: dict[str, object] | None = None
    if scenario_success.get("source") == "summary.behavior_evaluation.suite":
        behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
        if _payload_has_zero_reflex_scale(behavior_evaluation):
            return scenario_success
        unavailable_suite = {
            "available": False,
            "source": str(
                scenario_success.get("source") or "summary.behavior_evaluation.suite"
            ),
            "scenarios": [],
            "limitations": [
                "Behavior-evaluation suite was not marked as zero-reflex and was "
                "excluded from primary benchmark fallback."
            ],
        }
    elif scenario_success.get("available"):
        return scenario_success

    evaluation = _mapping_or_empty(summary.get("evaluation"))
    evaluation_summary = _mapping_or_empty(evaluation.get("summary"))
    if (
        "scenario_success_rate" in evaluation_summary
        and _payload_has_zero_reflex_scale(evaluation)
    ):
        return {
            "available": True,
            "source": "summary.evaluation",
            "scenarios": [
                {
                    "scenario": "aggregate",
                    "description": "",
                    "objective": "",
                    "success_rate": _coerce_float(
                        evaluation_summary.get("scenario_success_rate")
                    ),
                    "episodes": int(
                        _coerce_float(evaluation_summary.get("episode_count"), 0.0)
                    ),
                    "failures": [],
                    "checks": {},
                    "legacy_metrics": {},
                }
            ],
            "limitations": [],
        }

    if unavailable_suite is not None:
        return unavailable_suite
    return scenario_success
