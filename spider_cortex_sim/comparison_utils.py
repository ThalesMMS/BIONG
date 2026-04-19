"""Cross-run comparison and condensed reporting helpers.

``condense_robustness_summary`` is the public home for the CLI helper formerly
named ``_short_robustness_matrix_summary``.
"""

from __future__ import annotations

import json
import math
from typing import Dict, Sequence

from .benchmark_types import SeedLevelResult, UncertaintyEstimate
from .checkpointing import mean_reward_from_behavior_payload
from .noise import NoiseConfig
from .statistics import bootstrap_confidence_interval

def noise_profile_metadata(noise_profile: NoiseConfig) -> Dict[str, object]:
    """
    Aggregate-safe metadata for a resolved noise profile.
    
    Returns:
        dict: Mapping with keys:
            - "noise_profile": profile name (str)
            - "noise_profile_config": profile summary (dict) suitable for aggregation/serialization
    """
    return {
        "noise_profile": noise_profile.name,
        "noise_profile_config": noise_profile.to_summary(),
    }

def noise_profile_csv_value(noise_profile: NoiseConfig) -> str:
    """
    Produce a stable, compact JSON representation of a noise profile suitable for embedding in CSV fields.
    
    Parameters:
        noise_profile (NoiseConfig): A resolved noise profile whose summary will be serialized.
    
    Returns:
        json_value (str): A JSON string with sorted keys and compact separators representing the profile summary.
    """
    return json.dumps(
        noise_profile.to_summary(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

def safe_float(value: object) -> float:
    """
    Safely convert a value to float, returning 0.0 for invalid inputs.

    Parameters:
        value: Any value to attempt conversion to float.

    Returns:
        float: The converted float value, or 0.0 if conversion fails.
    """
    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return 0.0
    if not math.isfinite(numeric_value):
        return 0.0
    return numeric_value

def aggregate_with_uncertainty(
    seed_results: Sequence[SeedLevelResult | Dict[str, object] | tuple[int, float]],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
) -> Dict[str, object]:
    """
    Aggregate seed-level numeric values into a JSON-safe uncertainty estimate.
    
    Parses each entry of `seed_results` as either a `SeedLevelResult`, a dict with
    `"seed"` and `"value"`, or a `(seed, value)` tuple. Non-finite values and
    malformed entries are ignored. If no valid values remain, returns an estimate
    with zeros and `n_seeds = 0`.
    
    Parameters:
        seed_results: Sequence of seed-level rows to aggregate. Each element may be a
            `SeedLevelResult`, a `dict` with `"seed"` and `"value"`, or a `(seed, value)` tuple.
        confidence_level: Two-sided confidence level used for the bootstrap interval.
        n_resamples: Number of bootstrap resamples to use when computing the interval.
    
    Returns:
        dict: An UncertaintyEstimate-like dictionary with keys
        `mean`, `ci_lower`, `ci_upper`, `std_error`, `n_seeds`, `confidence_level`,
        and `seed_values`.
    """
    confidence = float(confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0.0 and 1.0.")
    resamples = int(n_resamples)
    if resamples < 1:
        raise ValueError("n_resamples must be at least 1.")
    parsed: list[tuple[int, float]] = []
    for item in seed_results:
        seed: object
        value: object
        if isinstance(item, SeedLevelResult):
            seed = item.seed
            value = item.value
        elif isinstance(item, dict):
            seed = item.get("seed")
            value = item.get("value")
        else:
            try:
                seed, value = item
            except (TypeError, ValueError):
                continue
        try:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                continue
            parsed.append((int(seed), numeric_value))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return UncertaintyEstimate(
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            std_error=0.0,
            n_seeds=0,
            confidence_level=confidence,
            seed_values=(),
        ).to_dict()
    values = tuple(value for _, value in parsed)
    mean, ci_lower, ci_upper, std_error = bootstrap_confidence_interval(
        values,
        confidence_level=confidence,
        n_resamples=resamples,
    )
    return UncertaintyEstimate(
        mean=mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_error=std_error,
        n_seeds=len(parsed),
        confidence_level=confidence,
        seed_values=values,
    ).to_dict()

def metric_seed_values_from_payload(
    payload: Dict[str, object] | None,
    *,
    metric_name: str,
    scenario: str | None = None,
    fallback_value: object | None = None,
) -> list[tuple[int, float]]:
    """
    Extract per-seed numeric values for a named metric from a compact behavior payload.
    
    Parses seed-level rows from payload["seed_level"] and, if `scenario` is provided, from payload["suite"][scenario]["seed_level"] when present. Selects entries whose "metric_name" matches `metric_name` and whose "scenario" matches the `scenario` constraint. Converts "value" to a finite float and "seed" to an int, skipping malformed or non-finite entries. When multiple entries exist for the same seed the last encountered value is used.
    
    Parameters:
        metric_name (str): Name of the metric to extract.
        scenario (str | None): If provided, restricts matches to rows for this scenario; if None, only rows without a "scenario" field are accepted.
        fallback_value (object | None): Accepted for API compatibility but ignored by this function.
    
    Returns:
        list[tuple[int, float]]: Sorted list of (seed, value) pairs for the requested metric; empty if none found.
    """
    if not isinstance(payload, dict):
        return []
    rows: list[tuple[int, float]] = []
    row_sources: list[object] = []
    if scenario is not None:
        suite = payload.get("suite", {})
        if isinstance(suite, dict):
            scenario_payload = suite.get(scenario, {})
            if isinstance(scenario_payload, dict):
                row_sources.append(scenario_payload.get("seed_level", []))
    row_sources.append(payload.get("seed_level", []))
    for source in row_sources:
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            if str(item.get("metric_name", "")) != metric_name:
                continue
            item_scenario = item.get("scenario")
            if scenario is None and item_scenario is not None:
                continue
            if scenario is not None and str(item_scenario) != str(scenario):
                continue
            try:
                value = float(item.get("value"))
                if not math.isfinite(value):
                    continue
                rows.append((int(item.get("seed")), value))
            except (TypeError, ValueError):
                continue
    if rows:
        deduped: dict[int, float] = {}
        for seed, value in rows:
            deduped[seed] = value
        return sorted(deduped.items())
    return []

def fallback_seed_values(value: object | None) -> list[tuple[int, float]]:
    """
    Convert a single value into a one-element seed-value pair or return an empty list when conversion fails.
    
    Parameters:
        value (object | None): A value to coerce to a finite float. If `None`, not convertible, or not finite, it is treated as absent.
    
    Returns:
        list[tuple[int, float]]: `[(0, numeric_value)]` when `value` can be converted to a finite float, otherwise `[]`.
    """
    if value is None:
        return []
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(numeric_value):
        return []
    return [(0, numeric_value)]

def paired_seed_delta_rows(
    reference_values: Sequence[tuple[int, float]],
    comparison_values: Sequence[tuple[int, float]],
    *,
    metric_name: str,
    condition: str,
    fallback_delta: object | None = None,
    scenario: str | None = None,
) -> list[SeedLevelResult]:
    """
    Compute per-seed differences between comparison and reference runs for a specific metric and condition.
    
    If both inputs contain the same seed, a row is produced with value = comparison - reference rounded to 6 decimal places. If there are no overlapping seeds, returns rows derived from `fallback_delta` (e.g., a single seed 0 entry when a numeric fallback is provided).
    
    Parameters:
        reference_values (Sequence[tuple[int, float]]): Sequence of (seed, value) pairs from the reference run.
        comparison_values (Sequence[tuple[int, float]]): Sequence of (seed, value) pairs from the comparison run.
        metric_name (str): Metric name to assign to each resulting row.
        condition (str): Condition identifier to assign to each resulting row.
        fallback_delta (object | None): Fallback value used when there are no common seeds; passed to `fallback_seed_values`.
        scenario (str | None): Optional scenario name to assign to each resulting row.
    
    Returns:
        list[SeedLevelResult]: Seed-level rows containing per-seed deltas (or fallback rows) with `metric_name`, `condition`, and optional `scenario` set.
    """
    reference_by_seed = {int(seed): float(value) for seed, value in reference_values}
    comparison_by_seed = {
        int(seed): float(value) for seed, value in comparison_values
    }
    common_seeds = sorted(set(reference_by_seed) & set(comparison_by_seed))
    rows = [
        SeedLevelResult(
            metric_name=metric_name,
            seed=seed,
            value=round(comparison_by_seed[seed] - reference_by_seed[seed], 6),
            condition=condition,
            scenario=scenario,
        )
        for seed in common_seeds
    ]
    if rows:
        return rows
    return [
        SeedLevelResult(
            metric_name=metric_name,
            seed=seed,
            value=value,
            condition=condition,
            scenario=scenario,
        )
        for seed, value in fallback_seed_values(fallback_delta)
    ]

def paired_seed_effect_size_rows(
    reference_values: Sequence[tuple[int, float]],
    comparison_values: Sequence[tuple[int, float]],
    *,
    condition: str,
    point_effect_size: float,
) -> list[SeedLevelResult]:
    """
    Compute per-seed standardized effect-size values labeled as "cohens_d".
    
    Given two sequences of (seed, value) pairs, returns a list of SeedLevelResult rows for the seeds present in both inputs where each row's value is a standardized delta for that seed. If the two sequences share no seeds, returns fallback seed rows derived from `point_effect_size`. If either group contains no numeric values, returns an empty list.
    
    Parameters:
        reference_values (Sequence[tuple[int, float]]): Sequence of (seed, value) pairs for the reference group.
        comparison_values (Sequence[tuple[int, float]]): Sequence of (seed, value) pairs for the comparison group.
        condition (str): Condition label to set on each returned SeedLevelResult.
        point_effect_size (float): Point estimate of the effect size used to scale deltas; when zero or when the computed pooled standardization is invalid, per-seed standardized values are set to `0.0`. Also used to produce fallback rows when no seeds overlap.
    
    Returns:
        list[SeedLevelResult]: One `SeedLevelResult` per common seed with `metric_name="cohens_d"`, `seed` set to the seed id, `value` set to the per-seed standardized delta (rounded to 6 decimals) or `0.0` in fallback/degenerate cases."""
    reference_by_seed = {int(seed): float(value) for seed, value in reference_values}
    comparison_by_seed = {
        int(seed): float(value) for seed, value in comparison_values
    }
    common_seeds = sorted(set(reference_by_seed) & set(comparison_by_seed))
    if not common_seeds:
        return [
            SeedLevelResult(
                metric_name="cohens_d",
                seed=seed,
                value=value,
                condition=condition,
            )
            for seed, value in fallback_seed_values(point_effect_size)
        ]

    reference_group = values_only(reference_values)
    comparison_group = values_only(comparison_values)
    if not reference_group or not comparison_group:
        return []
    mean_delta = (
        sum(comparison_group) / len(comparison_group)
        - sum(reference_group) / len(reference_group)
    )
    if point_effect_size == 0.0 or mean_delta == 0.0:
        standardized_values = {seed: 0.0 for seed in common_seeds}
    else:
        pooled_std = abs(mean_delta / point_effect_size)
        if pooled_std == 0.0 or not math.isfinite(pooled_std):
            standardized_values = {seed: 0.0 for seed in common_seeds}
        else:
            standardized_values = {
                seed: round(
                    (comparison_by_seed[seed] - reference_by_seed[seed])
                    / pooled_std,
                    6,
                )
                for seed in common_seeds
            }
    return [
        SeedLevelResult(
            metric_name="cohens_d",
            seed=seed,
            value=standardized_values[seed],
            condition=condition,
        )
        for seed in common_seeds
    ]

def values_only(seed_values: Sequence[tuple[int, float]]) -> list[float]:
    """
    Extract the numeric values from a sequence of (seed, value) pairs.
    
    Converts each pair's value to float and preserves input order.
    
    Returns:
        list[float]: Float values corresponding to each input pair, in the same order.
    """
    return [float(value) for _, value in seed_values]

def seed_level_dicts(rows: Sequence[SeedLevelResult]) -> list[Dict[str, object]]:
    """
    Convert a sequence of seed-level result objects into a list of dictionaries.
    
    Each output dictionary is the serialized representation of the corresponding input result.
    
    Parameters:
    	rows (Sequence[SeedLevelResult]): Sequence of seed-level result objects to convert.
    
    Returns:
    	list[Dict[str, object]]: A list of dictionaries representing each seed-level result.
    """
    return [row.to_dict() for row in rows]

def behavior_metric_seed_rows(
    seed_payloads: Sequence[tuple[int, Dict[str, object]]],
    *,
    metric_name: str,
    condition: str,
    scenario: str | None = None,
) -> list[SeedLevelResult]:
    """
    Collect per-seed metric rows for a given metric and condition from seed-associated payloads.
    
    Parameters:
        seed_payloads (Sequence[tuple[int, Dict[str, object]]]): Sequence of (seed, payload) pairs where payload is a dict produced for that seed.
        metric_name (str): Name of the metric to extract from the compact summary or scenario payload.
        condition (str): Condition label to attach to each returned SeedLevelResult.
        scenario (str | None): If None, read `metric_name` from the compact condition summary; if provided, read `success_rate` from `payload["suite"][scenario]`.
    
    Returns:
        list[SeedLevelResult]: A list of SeedLevelResult objects for seeds with valid numeric values. Seeds with missing or malformed payloads or missing metric data are omitted.
    """
    rows: list[SeedLevelResult] = []
    for seed, payload in seed_payloads:
        if scenario is None:
            if not isinstance(payload, dict):
                continue
            summary_payload = payload.get("summary", {})
            if not isinstance(summary_payload, dict) or metric_name not in summary_payload:
                continue
            raw_value = summary_payload.get(metric_name)
        else:
            suite = payload.get("suite", {})
            if not isinstance(suite, dict):
                continue
            scenario_payload = suite.get(scenario, {})
            if not isinstance(scenario_payload, dict):
                continue
            if "success_rate" not in scenario_payload:
                continue
            raw_value = scenario_payload.get("success_rate")
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        rows.append(
            SeedLevelResult(
                metric_name=metric_name,
                seed=seed,
                value=value,
                condition=condition,
                scenario=scenario,
            )
        )
    return rows

def attach_behavior_seed_statistics(
    payload: Dict[str, object],
    seed_payloads: Sequence[tuple[int, Dict[str, object]]],
    *,
    condition: str,
    scenario_names: Sequence[str],
) -> None:
    """
    Attach per-seed metric rows and aggregated uncertainty estimates onto a behavior payload in place.
    
    Computes per-seed rows for "scenario_success_rate", "episode_success_rate", and "mean_reward" (and a per-seed "specialization_score"), aggregates uncertainty for each metric, and sets payload["seed_level"] to the flattened list of seed-level row dicts and payload["uncertainty"] to a dict of per-metric uncertainty summaries. If payload["suite"] is a dict, also computes and attaches scenario-specific seed rows and uncertainty under each listed scenario name as:
    - scenario_payload["seed_level"]: list of seed-level row dicts for that scenario
    - scenario_payload["uncertainty"]["success_rate"]: uncertainty summary for the scenario's success rate
    
    Parameters:
        payload (Dict[str, object]): The behavior payload to mutate; will receive "seed_level" and "uncertainty" entries.
        seed_payloads (Sequence[tuple[int, Dict[str, object]]]): Sequence of (seed, per-seed payload) pairs from which per-seed metrics are extracted.
        condition (str): Condition label assigned to generated seed-level rows.
        scenario_names (Sequence[str]): Sequence of scenario names to process from payload["suite"] for scenario-specific attachments.
    """
    seed_level_rows: list[SeedLevelResult] = []
    uncertainty: Dict[str, object] = {}
    for metric_name in (
        "scenario_success_rate",
        "episode_success_rate",
        "mean_reward",
    ):
        metric_rows = behavior_metric_seed_rows(
            seed_payloads,
            metric_name=metric_name,
            condition=condition,
        )
        seed_level_rows.extend(metric_rows)
        uncertainty[metric_name] = aggregate_with_uncertainty(metric_rows)
    specialization_rows = [
        SeedLevelResult(
            metric_name="specialization_score",
            seed=seed,
            value=predator_type_specialization_score(seed_payload),
            condition=condition,
        )
        for seed, seed_payload in seed_payloads
    ]
    if specialization_rows:
        seed_level_rows.extend(specialization_rows)
        uncertainty["specialization_score"] = aggregate_with_uncertainty(
            specialization_rows
        )
    suite = payload.get("suite", {})
    if isinstance(suite, dict):
        for scenario_name in scenario_names:
            scenario_payload = suite.get(scenario_name)
            if not isinstance(scenario_payload, dict):
                continue
            scenario_rows = behavior_metric_seed_rows(
                seed_payloads,
                metric_name="scenario_success_rate",
                condition=condition,
                scenario=scenario_name,
            )
            seed_level_rows.extend(scenario_rows)
            scenario_payload["seed_level"] = seed_level_dicts(scenario_rows)
            scenario_payload["uncertainty"] = {
                "success_rate": aggregate_with_uncertainty(scenario_rows),
            }
    payload["seed_level"] = seed_level_dicts(seed_level_rows)
    payload["uncertainty"] = uncertainty

def condition_compact_summary(
    payload: Dict[str, object] | None,
) -> Dict[str, float]:
    """
    Extract a compact numeric summary of scenario/episode success rates and mean reward from a behavior-suite payload.

    Parameters:
        payload (Dict[str, object] | None): A behavior-suite payload (expected to contain a top-level "summary" mapping) or None.

    Returns:
        Dict[str, float]: A mapping with keys `"scenario_success_rate"`, `"episode_success_rate"`, and `"mean_reward"`, each cast to float. Missing or malformed input yields zeros for all three fields.
    """
    if not isinstance(payload, dict):
        return {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.0,
        }
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return {
            "scenario_success_rate": 0.0,
            "episode_success_rate": 0.0,
            "mean_reward": 0.0,
        }
    return {
        "scenario_success_rate": safe_float(
            summary.get("scenario_success_rate", 0.0)
        ),
        "episode_success_rate": safe_float(
            summary.get("episode_success_rate", 0.0)
        ),
        "mean_reward": condition_mean_reward(payload),
    }

def condition_mean_reward(payload: Dict[str, object] | None) -> float:
    """
    Resolve the mean reward for a condition payload, preferring summary["mean_reward"].
    
    If `summary["mean_reward"]` is present and convertible to float, that value is returned.
    Otherwise the function attempts `mean_reward_from_behavior_payload(payload)` and returns its float conversion.
    If `payload` is not a dict or conversions fail, returns 0.0.
    
    Parameters:
        payload (dict | None): Condition payload, typically containing a `summary` dict or legacy scenario data.
    
    Returns:
        float: The resolved mean reward, or 0.0 when missing or invalid.
    """
    if not isinstance(payload, dict):
        return 0.0
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        summary_mean_reward = summary.get("mean_reward")
        if summary_mean_reward is not None:
            try:
                value = float(summary_mean_reward)
            except (TypeError, ValueError):
                pass
            else:
                return value if math.isfinite(value) else 0.0
    try:
        value = float(mean_reward_from_behavior_payload(payload))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0

def module_response_by_predator_type_from_payload(
    payload: Dict[str, object] | None,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate module response mappings by predator type from suite or legacy payloads.

    Accepts compact behavior payloads with legacy scenario summaries or suite
    entries and returns rounded mean module responses per predator type.
    """
    if not isinstance(payload, dict):
        return {}
    sources: list[object] = []
    legacy_scenarios = payload.get("legacy_scenarios", {})
    if isinstance(legacy_scenarios, dict):
        sources.extend(legacy_scenarios.values())
    suite = payload.get("suite", {})
    if isinstance(suite, dict):
        for scenario_payload in suite.values():
            if isinstance(scenario_payload, dict):
                legacy_metrics = scenario_payload.get("legacy_metrics")
                if isinstance(legacy_metrics, dict):
                    sources.append(legacy_metrics)
                sources.append(scenario_payload)
    grouped: Dict[str, Dict[str, list[float]]] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        response = (
            source.get("mean_module_response_by_predator_type")
            or source.get("module_response_by_predator_type")
        )
        if not isinstance(response, dict):
            continue
        for predator_type, module_values in response.items():
            if not isinstance(module_values, dict):
                continue
            predator_key = str(predator_type)
            grouped.setdefault(predator_key, {})
            for module_name, value in module_values.items():
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric_value):
                    continue
                grouped[predator_key].setdefault(str(module_name), []).append(
                    numeric_value
                )
    return {
        predator_type: {
            module_name: round(sum(values) / len(values), 6)
            for module_name, values in sorted(module_values.items())
            if values
        }
        for predator_type, module_values in sorted(grouped.items())
    }

def predator_type_specialization_score(
    payload: Dict[str, object] | None,
) -> float:
    """
    Compute a bounded specialization score comparing visual and olfactory module responses.
    
    Returns:
        A float in the range [0.0, 1.0] representing the specialization score, rounded to 6 decimals. Returns 0.0 when either visual or olfactory predator-type responses lack any positive data.
    """
    response = module_response_by_predator_type_from_payload(payload)
    visual_modules = response.get("visual", {})
    olfactory_modules = response.get("olfactory", {})
    has_visual_data = any(safe_float(value) > 0.0 for value in visual_modules.values())
    has_olfactory_data = any(
        safe_float(value) > 0.0 for value in olfactory_modules.values()
    )
    if not has_visual_data or not has_olfactory_data:
        return 0.0
    visual_visual = safe_float(visual_modules.get("visual_cortex"))
    visual_sensory = safe_float(visual_modules.get("sensory_cortex"))
    olfactory_visual = safe_float(olfactory_modules.get("visual_cortex"))
    olfactory_sensory = safe_float(olfactory_modules.get("sensory_cortex"))
    return round(
        max(
            0.0,
            min(
                1.0,
                (
                    max(0.0, visual_visual - olfactory_visual)
                    + max(0.0, olfactory_sensory - visual_sensory)
                )
                / 2.0,
            ),
        ),
        6,
    )
