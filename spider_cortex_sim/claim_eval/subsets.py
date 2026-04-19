"""Claim-test evaluation workflows and helpers.

``condense_claim_test_summary`` is the public home for the CLI helper formerly
named ``_short_claim_test_suite_summary``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List, Sequence

from ..ablations import BrainAblationConfig, compare_predator_type_ablation_performance
from ..benchmark_types import SeedLevelResult
from ..budget_profiles import BudgetProfile
from ..checkpointing import CheckpointPenaltyMode
from ..claim_tests import (
    ClaimTestSpec,
    assess_scaffold_support,
    primary_claim_test_names,
    resolve_claim_tests,
)
from ..comparison import (
    aggregate_with_uncertainty,
    austere_comparison_from_payloads,
    compare_ablation_suite,
    compare_behavior_suite,
    compare_learning_evidence,
    compare_noise_robustness,
    fallback_seed_values,
    metric_seed_values_from_payload,
    paired_seed_delta_rows,
    paired_seed_effect_size_rows,
    representation_specialization_from_payload,
    safe_float,
    values_only,
    visual_minus_olfactory_seed_rows,
)
from ..memory import memory_leakage_audit
from ..noise import NoiseConfig, RobustnessMatrixSpec
from ..operational_profiles import OperationalProfile
from ..perception import observation_leakage_audit
from ..reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD, SCENARIO_AUSTERE_REQUIREMENTS
from ..statistics import cohens_d

from .thresholds import claim_registry_entry

def claim_subset_scenario_success_rate(
    payload: Dict[str, object] | None,
    *,
    scenarios: Sequence[str],
) -> tuple[float | None, str | None]:
    """
    Compute the mean scenario success rate across a required subset of scenarios from a behavior-suite payload.

    Parameters:
        payload (Dict[str, object] | None): Behavior payload expected to contain a top-level "suite" mapping where each scenario maps to a dict with a numeric "success_rate" key.
        scenarios (Sequence[str]): Sequence of scenario names required for the subset.

    Returns:
        tuple[float | None, str | None]: A pair (score, error). `score` is the mean of the per-scenario
        `success_rate` values rounded to six decimal places when all required scenarios are present; `None`
        if the payload is missing or invalid. `error` is a human-readable error message when `score` is
        `None`, otherwise `None`.
    """
    if not isinstance(payload, Mapping):
        return None, "Missing behavior payload."
    suite = payload.get("suite", {})
    if not isinstance(suite, Mapping):
        return None, "Behavior payload is missing suite results."
    if not scenarios:
        return 0.0, None
    missing = [
        name
        for name in scenarios
        if not isinstance(suite.get(name), Mapping)
    ]
    if missing:
        return None, f"Missing required scenarios: {missing}."
    scenario_passes = [
        safe_float(suite[name].get("success_rate", 0.0))
        for name in scenarios
    ]
    return round(sum(scenario_passes) / len(scenarios), 6), None

def claim_subset_seed_values(
    payload: Dict[str, object] | None,
    *,
    scenarios: Sequence[str],
    fallback_value: object | None = None,
) -> list[tuple[int, float]]:
    """
    Compute per-seed average success rates across the specified scenarios.
    
    Only seeds that have a recorded `scenario_success_rate` for every requested scenario are included. Each returned tuple is (seed, average_success_rate) where the average is rounded to six decimal places. If `scenarios` is empty or no seeds have complete data for all scenarios, the function returns fallback seed rows derived from `fallback_value`.
    
    Parameters:
        payload (dict | None): Behavior payload to extract per-seed metric values from.
        scenarios (Sequence[str]): Scenario names to average for each seed.
        fallback_value (object | None): Value used to generate fallback seed rows when no complete seed rows are available.
    
    Returns:
        list[tuple[int, float]]: List of (seed, average_success_rate) tuples sorted by seed.
    """
    if not scenarios:
        return fallback_seed_values(fallback_value if fallback_value is not None else 0.0)
    by_seed: Dict[int, list[float]] = {}
    for scenario_name in scenarios:
        for seed, value in metric_seed_values_from_payload(
            payload,
            metric_name="scenario_success_rate",
            scenario=scenario_name,
        ):
            by_seed.setdefault(seed, []).append(value)
    rows = [
        (seed, round(sum(values) / len(values), 6))
        for seed, values in sorted(by_seed.items())
        if len(values) == len(scenarios)
    ]
    if rows:
        return rows
    return fallback_seed_values(fallback_value)

def claim_comparison_statistics(
    *,
    reference_seed_values: Sequence[tuple[int, float]],
    comparison_seed_values: Sequence[tuple[int, float]],
    comparison_name: str,
    fallback_delta: object | None,
) -> Dict[str, object]:
    """
    Compute paired-delta and effect-size statistics comparing reference and comparison seed values.
    
    Parameters:
        reference_seed_values (Sequence[tuple[int, float]]): Per-seed (seed, value) pairs for the reference condition.
        comparison_seed_values (Sequence[tuple[int, float]]): Per-seed (seed, value) pairs for the comparison condition.
        comparison_name (str): Label for the comparison condition (used to annotate returned rows).
        fallback_delta (object | None): Value to use for delta when a paired comparison is missing; passed through to delta row construction.
    
    Returns:
        dict: A dictionary containing:
            - "delta_rows": List of per-seed delta rows comparing reference → comparison.
            - "delta_uncertainty": Aggregated uncertainty summary for the deltas.
            - "effect_size_uncertainty": Aggregated uncertainty summary for paired effect-size rows.
            - "cohens_d": Cohen's d for comparison vs. reference, rounded to 6 decimals.
            - "effect_magnitude": Categorical magnitude label for the effect size.
    """
    delta_rows = paired_seed_delta_rows(
        reference_seed_values,
        comparison_seed_values,
        metric_name="score_delta",
        condition=comparison_name,
        fallback_delta=fallback_delta,
    )
    comparison_values = values_only(comparison_seed_values) or [0.0]
    reference_values = values_only(reference_seed_values) or [0.0]
    effect_size, effect_magnitude = cohens_d(
        comparison_values,
        reference_values,
    )
    effect_size_rows = paired_seed_effect_size_rows(
        reference_seed_values,
        comparison_seed_values,
        condition=comparison_name,
        point_effect_size=round(float(effect_size), 6),
    )
    return {
        "delta_rows": delta_rows,
        "delta_uncertainty": aggregate_with_uncertainty(delta_rows),
        "effect_size_uncertainty": aggregate_with_uncertainty(
            effect_size_rows
        ),
        "cohens_d": round(float(effect_size), 6),
        "effect_magnitude": effect_magnitude,
    }

def claim_noise_subset_seed_values(
    payload: Dict[str, object] | None,
    *,
    scenarios: Sequence[str],
    diagonal_fallback: object | None,
    off_diagonal_fallback: object | None,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    Extracts per-seed averaged success-rate rows for diagonal (train==eval) and off-diagonal (train!=eval) cells from a noise-robustness matrix payload.
    
    Interprets `payload` as a dict containing a `matrix` mapping train_condition -> eval_condition -> cell_payload. For each cell, collects per-seed scenario-subset values (via claim_subset_seed_values), groups values by seed into diagonal or off-diagonal buckets, and returns per-seed averages rounded to six decimals. When `payload` or its `matrix` is malformed or when no rows exist for a category, returns fallback seed rows produced by `fallback_seed_values(...)` using the corresponding fallback argument.
    
    Parameters:
        payload (dict | None): Noise-robustness payload expected to contain a `matrix` mapping train -> eval -> cell payload.
        scenarios (Sequence[str]): Scenario names used to compute per-cell seed values.
        diagonal_fallback (object | None): Value forwarded to `fallback_seed_values` when diagonal rows are absent or payload is invalid.
        off_diagonal_fallback (object | None): Value forwarded to `fallback_seed_values` when off-diagonal rows are absent or payload is invalid.
    
    Returns:
        tuple[list[tuple[int, float]], list[tuple[int, float]]]: A (diagonal_rows, off_diagonal_rows) pair where each is a list of (seed, averaged_success_rate) with rates rounded to six decimals.
    """
    if not isinstance(payload, dict):
        return (
            fallback_seed_values(diagonal_fallback),
            fallback_seed_values(off_diagonal_fallback),
        )
    matrix = payload.get("matrix", {})
    if not isinstance(matrix, dict):
        return (
            fallback_seed_values(diagonal_fallback),
            fallback_seed_values(off_diagonal_fallback),
        )
    diagonal_by_seed: Dict[int, list[float]] = {}
    off_diagonal_by_seed: Dict[int, list[float]] = {}
    for train_condition, row in matrix.items():
        if not isinstance(row, dict):
            continue
        for eval_condition, cell_payload in row.items():
            subset_values = claim_subset_seed_values(
                cell_payload if isinstance(cell_payload, dict) else None,
                scenarios=scenarios,
            )
            target = (
                diagonal_by_seed
                if train_condition == eval_condition
                else off_diagonal_by_seed
            )
            for seed, value in subset_values:
                target.setdefault(seed, []).append(value)
    diagonal_rows = [
        (seed, round(sum(values) / len(values), 6))
        for seed, values in sorted(diagonal_by_seed.items())
        if values
    ]
    off_diagonal_rows = [
        (seed, round(sum(values) / len(values), 6))
        for seed, values in sorted(off_diagonal_by_seed.items())
        if values
    ]
    return (
        diagonal_rows or fallback_seed_values(diagonal_fallback),
        off_diagonal_rows or fallback_seed_values(off_diagonal_fallback),
    )

def claim_noise_subset_scores(
    payload: Dict[str, object] | None,
    *,
    scenarios: Sequence[str],
) -> tuple[float | None, float | None, str | None]:
    """
    Compute mean diagonal and off-diagonal subset success rates for a noise-robustness matrix.

    Parameters:
        payload (dict | None): Robustness comparison payload containing a "matrix" mapping
            train_condition -> eval_condition -> cell payload. If missing or malformed,
            an error reason is returned.
        scenarios (Sequence[str]): Sequence of scenario names to include when computing
            per-cell subset success rates.

    Returns:
        tuple[float | None, float | None, str | None]:
            diagonal_mean: Mean subset success rate across cells where train_condition == eval_condition,
                rounded to 6 decimal places, or `None` if unavailable.
            off_diagonal_mean: Mean subset success rate across cells where train_condition != eval_condition,
                rounded to 6 decimal places, or `None` if unavailable.
            error_reason: `None` on success; otherwise a short string explaining why the computation
                could not be completed (e.g., missing matrix, malformed rows, or missing cell data).
    """
    if not isinstance(payload, dict):
        return None, None, "Missing noise_robustness payload."
    matrix = payload.get("matrix", {})
    if not isinstance(matrix, dict):
        return None, None, "Noise robustness payload is missing matrix results."
    diagonal_scores: list[float] = []
    off_diagonal_scores: list[float] = []
    for train_condition, row in matrix.items():
        if not isinstance(row, dict):
            return None, None, f"Noise robustness row {train_condition!r} is malformed."
        for eval_condition, cell_payload in row.items():
            metric_value, reason = claim_subset_scenario_success_rate(
                cell_payload,
                scenarios=scenarios,
            )
            if metric_value is None:
                return None, None, (
                    f"Missing data for noise cell {train_condition!r} -> "
                    f"{eval_condition!r}: {reason}"
                )
            if train_condition == eval_condition:
                diagonal_scores.append(metric_value)
            else:
                off_diagonal_scores.append(metric_value)
    if not diagonal_scores:
        return None, None, "Noise robustness payload has no diagonal cells."
    if not off_diagonal_scores:
        return None, None, "Noise robustness payload has no off-diagonal cells."
    return (
        round(sum(diagonal_scores) / len(diagonal_scores), 6),
        round(sum(off_diagonal_scores) / len(off_diagonal_scores), 6),
        None,
    )

SPECIALIZATION_ENGAGEMENT_CHECKS: dict[str, str] = {
    "visual_olfactory_pincer": "type_specific_response",
    "olfactory_ambush": "sensory_cortex_engaged",
    "visual_hunter_open_field": "visual_cortex_engaged",
}

def claim_specialization_engagement(
    payload: Dict[str, object] | None,
    *,
    variant_name: str,
    scenarios: Sequence[str],
) -> tuple[int | None, Dict[str, float] | None, str | None]:
    """
    Determine how many scenarios for a variant show full specialization engagement according to type-specific checks.

    Examines the `variants` registry entry in `payload` for `variant_name`, reads each `scenario`'s check named in SPECIALIZATION_ENGAGEMENT_CHECKS, and records each check's pass rate. A scenario is counted as "engaged" when its pass rate is greater than or equal to 1.0.

    Parameters:
        payload (dict | None): Comparison payload containing a `variants` registry (may be None).
        variant_name (str): Name of the variant whose suite to inspect.
        scenarios (Sequence[str]): Sequence of scenario names to evaluate for specialization engagement.

    Returns:
        tuple:
            engaged_count (int | None): Number of scenarios with pass rate >= 1.0, or `None` if the check could not be performed.
            pass_rates (dict[str, float] | None): Mapping from scenario name to its recorded `pass_rate` (rounded to 6 decimals), or `None` when unavailable.
            reason (str | None): `None` on success; otherwise a short explanatory message describing why the result is unavailable (missing registry entry, missing scenario/check, or unregistered engagement check).
    """
    variant_payload, reason = claim_registry_entry(
        payload,
        registry_key="variants",
        entry_name=variant_name,
    )
    if variant_payload is None:
        return None, None, reason
    suite = variant_payload.get("suite", {})
    if not isinstance(suite, dict):
        return None, None, f"Variant {variant_name!r} is missing suite results."
    pass_rates: Dict[str, float] = {}
    engaged_count = 0
    for scenario_name in scenarios:
        scenario_payload = suite.get(scenario_name)
        if not isinstance(scenario_payload, dict):
            return None, None, (
                f"Variant {variant_name!r} is missing scenario {scenario_name!r}."
            )
        checks = scenario_payload.get("checks", {})
        if not isinstance(checks, dict):
            return None, None, (
                f"Variant {variant_name!r} scenario {scenario_name!r} is missing checks."
            )
        check_name = SPECIALIZATION_ENGAGEMENT_CHECKS.get(scenario_name)
        if check_name is None:
            return None, None, (
                f"No specialization engagement check is registered for scenario {scenario_name!r}."
            )
        check_payload = checks.get(check_name)
        if not isinstance(check_payload, dict):
            return None, None, (
                f"Variant {variant_name!r} scenario {scenario_name!r} is missing "
                f"check {check_name!r}."
            )
        raw_pass_rate = safe_float(check_payload.get("pass_rate", 0.0))
        pass_rates[scenario_name] = round(raw_pass_rate, 6)
        if raw_pass_rate >= 1.0:
            engaged_count += 1
    return engaged_count, pass_rates, None

def claim_austere_survival_gate(
    spec: ClaimTestSpec,
    payloads: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    Assess whether required austere survival gate scenarios are available and pass for a claim test.
    
    Parameters:
        spec (ClaimTestSpec): Claim test specification; function reads `spec.austere_survival_required`
            and `spec.scenarios` to determine gate scenarios and requirement status.
        payloads (Dict[str, Dict[str, object]]): Mapping of primitive comparison payloads used to
            derive austere survival data; may be produced by comparison helpers or reused inputs.
    
    Returns:
        Dict[str, object]: A gate report containing:
            - required (bool): Whether an austere survival gate is required for this claim.
            - available (bool): Whether austere survival data was available to evaluate the gate.
            - passed (bool): Whether all required gate scenarios were present and survived.
            - gate_scenarios (list[str]): List of scenario names designated as gate scenarios.
            - failed_scenarios (list[str]): Gate scenarios that were present but did not survive.
            - missing_scenarios (list[str]): Gate scenarios that were not present in the austere data.
            - scenario_results (dict): Per-scenario entries with availability, `survives` flag,
              and when available an `austere_success_rate` and `survival_threshold`.
            - reason (str): Human-readable reason when the gate could not be evaluated or when it failed.
    """
    gate_scenarios = [
        scenario_name
        for scenario_name in spec.scenarios
        if SCENARIO_AUSTERE_REQUIREMENTS.get(
            scenario_name,
            {},
        ).get("requirement_level") == "gate"
    ]
    gate_result: Dict[str, object] = {
        "required": bool(spec.austere_survival_required),
        "available": False,
        "passed": not bool(spec.austere_survival_required),
        "gate_scenarios": gate_scenarios,
        "failed_scenarios": [],
        "missing_scenarios": [],
        "scenario_results": {},
        "reason": "",
    }
    if not spec.austere_survival_required:
        return gate_result
    comparison = austere_comparison_from_payloads(payloads)
    if comparison is None:
        gate_result["reason"] = "Austere survival comparison data is unavailable."
        return gate_result
    behavior_survival = comparison.get("behavior_survival", {})
    if not isinstance(behavior_survival, dict) or not bool(
        behavior_survival.get("available", False)
    ):
        gate_result["reason"] = "Austere behavior survival data is unavailable."
        return gate_result
    survival_scenarios = behavior_survival.get("scenarios", {})
    if not isinstance(survival_scenarios, dict):
        gate_result["reason"] = "Austere behavior survival scenarios are missing."
        return gate_result
    scenario_results: Dict[str, object] = {}
    failed: list[str] = []
    missing: list[str] = []
    for scenario_name in gate_scenarios:
        scenario_payload = survival_scenarios.get(scenario_name)
        if not isinstance(scenario_payload, dict):
            missing.append(scenario_name)
            scenario_results[scenario_name] = {
                "available": False,
                "survives": False,
            }
            continue
        survives = bool(scenario_payload.get("survives", False))
        austere_success_rate = round(
            safe_float(scenario_payload.get("austere_success_rate")),
            6,
        )
        scenario_results[scenario_name] = {
            "available": True,
            "survives": survives,
            "austere_success_rate": austere_success_rate,
            "survival_threshold": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        }
        if not survives:
            failed.append(scenario_name)
    gate_result.update(
        {
            "available": True,
            "passed": not failed and not missing,
            "failed_scenarios": failed,
            "missing_scenarios": missing,
            "scenario_results": scenario_results,
        }
    )
    if failed or missing:
        reasons: list[str] = []
        if failed:
            reasons.append(f"failed austere scenarios: {failed}")
        if missing:
            reasons.append(f"missing austere scenarios: {missing}")
        gate_result["reason"] = "; ".join(reasons)
    return gate_result

def claim_payload_config_summary(
    payload: object,
) -> Dict[str, object]:
    """
    Produce a shallow copy of the payload's "config" mapping when present; otherwise return an empty dict.
    
    Returns:
        dict: A shallow copy of payload["config"] if `payload` is a dict and contains a dict under the "config" key; otherwise an empty dict.
    """
    if not isinstance(payload, dict):
        return {}
    config_summary = payload.get("config", {})
    if not isinstance(config_summary, dict):
        return {}
    return dict(config_summary)

def claim_payload_eval_reflex_scale(
    payload: object,
) -> float | None:
    """
    Return the finite evaluation reflex scale value from a payload if present.
    
    Attempts to read `eval_reflex_scale` from the top level of `payload` or, if absent, from `payload["summary"]["eval_reflex_scale"]` when `summary` is a dict; parses the value as a float and rejects non-finite values.
    
    Returns:
        float: The finite `eval_reflex_scale` value when present and parseable.
        None: If the payload is not a mapping, the value is missing, cannot be converted to float, or is not finite.
    """
    if not isinstance(payload, dict):
        return None
    raw_scale = payload.get("eval_reflex_scale")
    if raw_scale is None:
        summary = payload.get("summary", {})
        if isinstance(summary, dict):
            raw_scale = summary.get("eval_reflex_scale")
    if raw_scale is None:
        return None
    try:
        value = float(raw_scale)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value
