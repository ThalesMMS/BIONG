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

from .ablations import BrainAblationConfig, compare_predator_type_ablation_performance
from .benchmark_types import SeedLevelResult
from .budget_profiles import BudgetProfile
from .checkpointing import CheckpointPenaltyMode
from .claim_tests import (
    ClaimTestSpec,
    assess_scaffold_support,
    primary_claim_test_names,
    resolve_claim_tests,
)
from .comparison import (
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
from .memory import memory_leakage_audit
from .noise import NoiseConfig, RobustnessMatrixSpec
from .operational_profiles import OperationalProfile
from .perception import observation_leakage_audit
from .reward import MINIMAL_SHAPING_SURVIVAL_THRESHOLD, SCENARIO_AUSTERE_REQUIREMENTS
from .statistics import cohens_d


SPECIALIZATION_ENGAGEMENT_CHECKS: dict[str, str] = {
    "visual_olfactory_pincer": "type_specific_response",
    "olfactory_ambush": "sensory_cortex_engaged",
    "visual_hunter_open_field": "visual_cortex_engaged",
}


def claim_test_source(spec: ClaimTestSpec) -> str | None:
    """
    Map a claim-test specification to the primitive payload type it requires.

    Parameters:
        spec (ClaimTestSpec): Claim-test specification object with a `.protocol` attribute.

    Returns:
        str | None: `'learning_evidence'`, `'noise_robustness'`, or `'ablation'` when the protocol name references that primitive; `None` if no known primitive is referenced.
    """
    protocol = spec.protocol.lower()
    if "learning-evidence" in protocol:
        return "learning_evidence"
    if "noise-robustness" in protocol:
        return "noise_robustness"
    if "ablation" in protocol:
        return "ablation"
    return None


def claim_skip_result(spec: ClaimTestSpec, reason: str) -> Dict[str, object]:
    """
    Produce a standardized "skipped" result record for a claim test.

    Parameters:
        spec (ClaimTestSpec): The claim test specification; used to populate primary metric, scenarios, and success criterion.
        reason (str): Human-readable explanation for why the claim test was skipped.

    Returns:
        Dict[str, object]: A normalized result dictionary with keys:
            - "status": "skipped"
            - "passed": False
            - "reason": the provided reason string
            - "reference_value": None
            - "comparison_values": empty dict
            - "delta": empty dict
            - "effect_size": None
            - "primary_metric": value from spec.primary_metric
            - "scenarios_evaluated": list of scenarios from spec.scenarios
            - "notes": list containing spec.success_criterion
    """
    return {
        "status": "skipped",
        "passed": False,
        "reason": str(reason),
        "reference_value": None,
        "comparison_values": {},
        "delta": {},
        "effect_size": None,
        "reference_uncertainty": None,
        "comparison_uncertainty": {},
        "delta_uncertainty": {},
        "effect_size_uncertainty": {},
        "cohens_d": {},
        "effect_magnitude": {},
        "primary_metric": spec.primary_metric,
        "scenarios_evaluated": list(spec.scenarios),
        "notes": [spec.success_criterion],
    }


def claim_threshold_from_operator(
    success_criterion: str,
    operator: str,
) -> float | None:
    """
    Parses the first numeric literal immediately following a given operator token in a criterion string.
    
    Parameters:
        success_criterion (str): Criterion text to search.
        operator (str): Literal operator token to match (e.g., ">=", "<", "==").
    
    Returns:
        float | None: Parsed number (supports negative and decimal values) if found, otherwise None.
    """
    match = re.search(
        rf"{re.escape(operator)}\s*(-?\d+(?:\.\d+)?)",
        success_criterion,
    )
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def claim_threshold_from_phrase(
    success_criterion: str,
    phrase: str,
) -> float | None:
    """
    Parses a numeric threshold immediately following a literal phrase in a criterion string.
    
    Parameters:
        success_criterion (str): Criterion text to search for a numeric value.
        phrase (str): Literal phrase to locate; the function looks for a number directly after this phrase.
    
    Returns:
        float | None: The parsed numeric threshold (supports an optional leading `-` and decimals) if found, `None` otherwise.
    """
    match = re.search(
        rf"{re.escape(phrase)}\s*(-?\d+(?:\.\d+)?)",
        success_criterion,
    )
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def claim_count_threshold(success_criterion: str) -> int | None:
    """
    Parse an integer count from a phrase of the form "at least N of the M".
    
    Parameters:
        success_criterion (str): Text to search for the "at least N of the M" pattern.
    
    Returns:
        int | None: The extracted integer N when the pattern is present and valid, otherwise `None`.
    """
    match = re.search(r"at least\s+(\d+)\s+of\s+the\s+\d+", success_criterion)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


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
    Compute per-seed average success rates across a specified set of scenarios.
    
    If `scenarios` is empty, returns fallback seed rows generated by `fallback_seed_values`.
    Only seeds that have a recorded `scenario_success_rate` for every requested scenario are included; each returned tuple is (seed, average) where average is the mean of that seed's per-scenario success rates rounded to six decimal places. If no complete seeds are found, returns the fallback seed rows produced from `fallback_value`.
    
    Parameters:
        payload (dict | None): Behavior payload from which per-seed metric values are extracted.
        scenarios (Sequence[str]): Scenario names to average per seed.
        fallback_value (object | None): Value forwarded to `fallback_seed_values` when no complete seed rows are available.
    
    Returns:
        list[tuple[int, float]]: A list of (seed, average_success_rate) tuples sorted by seed.
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
        metric_name="effect_size",
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


def claim_registry_entry(
    payload: Dict[str, object] | None,
    *,
    registry_key: str,
    entry_name: str,
) -> tuple[Dict[str, object] | None, str | None]:
    """
    Locate a named entry in a registry payload and verify it was evaluated.

    Parameters:
        registry_key (str): Top-level key in `payload` expected to contain the registry mapping.
        entry_name (str): Name of the entry to fetch from the registry.

    Returns:
        tuple[Dict[str, object] | None, str | None]: A pair `(entry, error)`. `entry` is the found registry entry dict when present and not marked skipped; otherwise `None`. `error` is a human-readable message when the registry, entry, or evaluation result is missing or the entry was skipped; otherwise `None`.
    """
    if not isinstance(payload, dict):
        return None, f"Missing {registry_key} payload."
    registry = payload.get(registry_key, {})
    if not isinstance(registry, dict):
        return None, f"{registry_key!r} payload is missing its registry."
    entry = registry.get(entry_name)
    if not isinstance(entry, dict):
        return None, f"Missing {registry_key[:-1]} {entry_name!r}."
    if bool(entry.get("skipped")):
        return None, str(
            entry.get("reason", f"{registry_key[:-1].capitalize()} {entry_name!r} was skipped.")
        )
    return entry, None


def claim_leakage_audit_summary() -> Dict[str, object]:
    """
    Summarizes unresolved privileged or world-derived leakage findings.

    Returns:
        summary (dict): A mapping with:
            - finding_count (int): Number of unresolved findings.
            - findings (list[str]): List of findings formatted as "<audit_name>:<signal_name>".
    """
    findings: list[str] = []
    for audit_name, audit_entries in (
        ("observation", observation_leakage_audit()),
        ("memory", memory_leakage_audit()),
    ):
        for signal_name, metadata in audit_entries.items():
            classification = str(metadata.get("classification", ""))
            risk = str(metadata.get("risk", ""))
            if classification in {
                "privileged_world_signal",
                "world_derived_navigation_hint",
            } and risk != "resolved":
                findings.append(f"{audit_name}:{signal_name}")
    return {
        "finding_count": len(findings),
        "findings": findings,
    }


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
        pass_rate = round(
            safe_float(check_payload.get("pass_rate", 0.0)),
            6,
        )
        pass_rates[scenario_name] = pass_rate
        if pass_rate >= 1.0:
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
    Extracts the evaluation reflex scale from a payload, falling back to payload["summary"]["eval_reflex_scale"] and returning it as a finite float.
    
    Parameters:
        payload (object): A payload object expected to be a dict containing an optional
            "eval_reflex_scale" key or a "summary" dict with "eval_reflex_scale".
    
    Returns:
        float | None: The parsed finite `eval_reflex_scale` value, or `None` if the value is missing,
        not parseable as a float, or not finite.
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


def extract_claim_config_for_scaffold_assessment(
    spec: ClaimTestSpec,
    payloads: dict,
) -> tuple[dict, float | None]:
    """
    Extract the scaffold configuration summary and the evaluation reflex scale needed for scaffold assessment.
    
    Parameters:
        spec (ClaimTestSpec): Claim test specification whose protocol determines the payload source and which reference/comparison entries to consult.
        payloads (dict): Mapping of primitive payload types ("learning_evidence", "ablation", "noise_robustness") to their loaded payload dictionaries.
    
    Returns:
        tuple[dict, float | None]: A pair where the first element is a shallow config summary dictionary extracted from the selected reference/baseline payload (empty dict if missing or invalid), and the second element is the extracted `eval_reflex_scale` as a finite float or `None` when unavailable.
    """
    source = claim_test_source(spec)
    if source == "learning_evidence":
        payload = payloads.get("learning_evidence")
        reference_payload, _ = claim_registry_entry(
            payload,
            registry_key="conditions",
            entry_name=spec.reference_condition,
        )
        config_summary = claim_payload_config_summary(reference_payload)
        if not config_summary:
            return {}, None
        determining_comparison_name: str | None = None
        if "trained_without_reflex_support" in spec.comparison_conditions:
            determining_comparison_name = "trained_without_reflex_support"
        elif spec.comparison_conditions:
            determining_comparison_name = spec.comparison_conditions[0]
        determining_comparison_payload = None
        if determining_comparison_name is not None:
            determining_comparison_payload, _ = claim_registry_entry(
                payload,
                registry_key="conditions",
                entry_name=determining_comparison_name,
            )
        eval_reflex_scale = claim_payload_eval_reflex_scale(
            determining_comparison_payload
        )
        if eval_reflex_scale is None:
            eval_reflex_scale = claim_payload_eval_reflex_scale(
                reference_payload
            )
        return (config_summary, eval_reflex_scale)

    if source == "ablation":
        payload = payloads.get("ablation")
        reference_payload, _ = claim_registry_entry(
            payload,
            registry_key="variants",
            entry_name=spec.reference_condition,
        )
        config_summary = claim_payload_config_summary(reference_payload)
        eval_reflex_scale = claim_payload_eval_reflex_scale(reference_payload)
        if not config_summary and isinstance(reference_payload, dict):
            without_reflex_payload = reference_payload.get(
                "without_reflex_support"
            )
            config_summary = claim_payload_config_summary(
                without_reflex_payload
            )
            if eval_reflex_scale is None:
                eval_reflex_scale = claim_payload_eval_reflex_scale(
                    without_reflex_payload
                )
        if not config_summary:
            return {}, None
        return config_summary, eval_reflex_scale

    if source == "noise_robustness":
        payload = payloads.get("noise_robustness")
        if not isinstance(payload, dict):
            return {}, None
        matrix = payload.get("matrix", {})
        if not isinstance(matrix, dict):
            return {}, None
        baseline_condition: str | None = None
        matrix_spec = payload.get("matrix_spec", {})
        if isinstance(matrix_spec, dict):
            train_conditions = matrix_spec.get("train_conditions", [])
            eval_conditions = matrix_spec.get("eval_conditions", [])
            if isinstance(train_conditions, (list, tuple)) and isinstance(
                eval_conditions,
                (list, tuple),
            ):
                train_names = [str(condition) for condition in train_conditions]
                eval_names = [str(condition) for condition in eval_conditions]
                if "none" in train_names and "none" in eval_names:
                    baseline_condition = "none"
                else:
                    eval_name_set = set(eval_names)
                    for condition in train_names:
                        if condition in eval_name_set:
                            baseline_condition = condition
                            break
        if baseline_condition is None:
            for train_condition, row in matrix.items():
                if isinstance(row, dict) and str(train_condition) in row:
                    baseline_condition = str(train_condition)
                    break
        if baseline_condition is None:
            return {}, None
        row = matrix.get(baseline_condition, {})
        if not isinstance(row, dict):
            return {}, None
        baseline_payload = row.get(baseline_condition)
        config_summary = claim_payload_config_summary(baseline_payload)
        if not config_summary:
            return {}, None
        return (
            config_summary,
            claim_payload_eval_reflex_scale(baseline_payload),
        )

    return {}, None


def finalize_claim_result(
    spec: ClaimTestSpec,
    result: Dict[str, object],
    payloads: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    Finalize a claim result by applying austere survival gating and scaffold-support assessment.
    
    Attaches the computed austere survival gate to the result, forces failure when the claim requires an austere gate that did not pass, performs scaffold-support assessment (extracting a config summary and eval reflex scale from provided payloads), populates scaffold-related metadata (`scaffold_support_level`, `scaffold_findings`, `benchmark_of_record_eligible`, `claim_severity`), and appends scaffold findings to the result notes. If the claim is a primary claim that only passes under `scaffolded_runtime` support, the result is downgraded to failed and an explanatory reason is appended.
    
    Parameters:
        spec (ClaimTestSpec): The claim test specification driving gating and scaffold extraction.
        result (Dict[str, object]): The intermediate claim result to finalize; a copy is returned and original is not mutated.
        payloads (Dict[str, Dict[str, object]]): Available primitive payloads used to compute the austere gate and scaffold assessment inputs.
    
    Returns:
        Dict[str, object]: A finalized claim result dictionary with added keys:
            - `austere_survival_required` (bool)
            - `austere_survival_gate` (dict)
            - `scaffold_support_level` (str)
            - `scaffold_findings` (list[str])
            - `benchmark_of_record_eligible` (bool)
            - `claim_severity` (str)
          The returned result may have `status`, `passed`, `reason`, and `notes` modified when gates or scaffold rules change the outcome.
    """
    finalized = dict(result)
    gate = claim_austere_survival_gate(spec, payloads)
    finalized["austere_survival_required"] = bool(
        spec.austere_survival_required
    )
    finalized["austere_survival_gate"] = gate
    if (
        spec.austere_survival_required
        and str(finalized.get("status")) != "skipped"
        and not bool(gate.get("passed", False))
    ):
        finalized["status"] = "failed"
        finalized["passed"] = False
        finalized["reason"] = (
            "Austere survival gate failed: "
            f"{gate.get('reason') or 'required gate did not pass'}."
        )
        notes = list(finalized.get("notes", []))
        notes.append(str(finalized["reason"]))
        finalized["notes"] = notes
    config_summary, eval_reflex_scale = (
        extract_claim_config_for_scaffold_assessment(spec, payloads)
    )
    severity_by_support_level = {
        "minimal_manual": "full",
        "standard_constrained": "qualified",
        "scaffolded_runtime": "non_benchmark",
        "missing_inputs": "non_benchmark",
    }
    if not config_summary or eval_reflex_scale is None:
        support_level = "missing_inputs"
        scaffold_findings: list[str] = []
        if not config_summary:
            scaffold_findings.append("scaffold_config_missing")
        if eval_reflex_scale is None:
            scaffold_findings.append("scaffold_eval_reflex_scale_missing")
        if not scaffold_findings:
            scaffold_findings.append("scaffold_assessment_inputs_missing")
        benchmark_of_record_eligible = False
    else:
        scaffold_assessment = assess_scaffold_support(
            config_summary,
            eval_reflex_scale,
        )
        support_level = scaffold_assessment.support_level.value
        scaffold_findings = list(scaffold_assessment.findings)
        benchmark_of_record_eligible = bool(
            scaffold_assessment.benchmark_of_record_eligible
        )
    finalized["scaffold_support_level"] = support_level
    finalized["scaffold_findings"] = list(scaffold_findings)
    finalized["benchmark_of_record_eligible"] = benchmark_of_record_eligible
    finalized["claim_severity"] = severity_by_support_level[support_level]
    notes = list(finalized.get("notes", []))
    notes.extend(
        f"Scaffold finding: {finding}."
        for finding in scaffold_findings
    )
    finalized["notes"] = notes
    if (
        spec.primary
        and str(finalized.get("status")) == "passed"
        and support_level == "scaffolded_runtime"
    ):
        findings_clause = ""
        if scaffold_findings:
            findings_clause = (
                f" Findings: {', '.join(scaffold_findings)}."
            )
        finalized["status"] = "failed"
        finalized["passed"] = False
        finalized["reason"] = (
            "Primary claim passed only under scaffolded runtime conditions "
            "and is not eligible for benchmark-of-record evidence."
            f"{findings_clause}"
        )
        notes.append(str(finalized["reason"]))
        finalized["notes"] = notes
    return finalized


def evaluate_claim_test(
    spec: ClaimTestSpec,
    payloads: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    Evaluate a single claim test specification using provided primitive payloads and produce a standardized result record.
    
    This function selects the primitive payload type required by `spec`, validates and extracts the needed reference and comparison data from `payloads`, computes reference/comparison metrics, per-seed uncertainties, deltas, effect-size statistics (including Cohen's d and magnitude), and determines pass/fail/skipped status according to the claim's `success_criterion` and any claim-specific rules. The returned result is finalized with austere-survival and scaffold-support assessments.
    
    Parameters:
        spec (ClaimTestSpec): The canonical claim test specification to evaluate; provides protocol, name, primary metric, scenarios, reference and comparison condition names, and success criterion.
        payloads (Dict[str, Dict[str, object]]): Mapping of primitive source names ("learning_evidence", "ablation", "noise_robustness", etc.) to their payload registries used to compute metrics and seed-level values.
    
    Returns:
        Dict[str, object]: A standardized claim result record containing at least:
          - `status` ("passed", "failed", or "skipped") and `passed` (bool),
          - metric values (`reference_value`, `comparison_values`, `delta`, `effect_size`),
          - uncertainty summaries (`reference_uncertainty`, `comparison_uncertainty`, `delta_uncertainty`, `effect_size_uncertainty`),
          - effect statistics (`cohens_d`, `effect_magnitude`),
          - `primary_metric`, `scenarios_evaluated`, `notes`, and scaffold/austere-related fields added by finalization.
    """
    def finish(result: Dict[str, object]) -> Dict[str, object]:
        """
        Finalize a single claim result by applying austere survival gating, scaffold-support assessment, and primary-claim eligibility rules.
        
        Parameters:
            result (Dict[str, object]): Partial claim result produced by evaluation (may be mutated or extended).
        
        Returns:
            Dict[str, object]: The finalized claim result record, with `austere_survival_gate`, `scaffold_support_level`, `scaffold_findings`, `benchmark_of_record_eligible`, `claim_severity`, and potentially updated `status`, `passed`, `reason`, and `notes`.
        """
        return finalize_claim_result(spec, result, payloads)

    def skip(reason: str) -> Dict[str, object]:
        """
        Create a finalized skipped claim result for the current claim spec using the provided reason.
        
        Parameters:
        	reason (str): Explanation for skipping the claim.
        
        Returns:
        	Dict[str, object]: A claim result record with `status` set to "skipped", `passed` set to False, `reason` populated with the given text, and standardized empty or placeholder fields for comparison, delta, effect-size, and uncertainty.
        """
        return finish(claim_skip_result(spec, reason))

    source = claim_test_source(spec)
    if source is None:
        return skip(
            f"Could not determine a primitive payload source from protocol {spec.protocol!r}.",
        )

    if source == "learning_evidence":
        payload = payloads.get("learning_evidence")
        reference_payload, reason = claim_registry_entry(
            payload,
            registry_key="conditions",
            entry_name=spec.reference_condition,
        )
        if reference_payload is None:
            return skip(str(reason))
        reference_value, reason = claim_subset_scenario_success_rate(
            reference_payload,
            scenarios=spec.scenarios,
        )
        if reference_value is None:
            return skip(str(reason))
        reference_seed_values = claim_subset_seed_values(
            reference_payload,
            scenarios=spec.scenarios,
            fallback_value=reference_value,
        )
        reference_uncertainty = aggregate_with_uncertainty(
            [
                SeedLevelResult(
                    metric_name=spec.primary_metric,
                    seed=seed,
                    value=value,
                    condition=spec.reference_condition,
                )
                for seed, value in reference_seed_values
            ]
        )
        comparison_values: Dict[str, float] = {}
        comparison_uncertainty: Dict[str, object] = {}
        deltas: Dict[str, float] = {}
        delta_uncertainty: Dict[str, object] = {}
        effect_size_uncertainty: Dict[str, object] = {}
        cohens_d_values: Dict[str, float] = {}
        effect_magnitudes: Dict[str, str] = {}
        for comparison_name in spec.comparison_conditions:
            comparison_payload, reason = claim_registry_entry(
                payload,
                registry_key="conditions",
                entry_name=comparison_name,
            )
            if comparison_payload is None:
                return skip(str(reason))
            comparison_value, reason = claim_subset_scenario_success_rate(
                comparison_payload,
                scenarios=spec.scenarios,
            )
            if comparison_value is None:
                return skip(str(reason))
            comparison_values[comparison_name] = comparison_value
            deltas[comparison_name] = round(comparison_value - reference_value, 6)
            comparison_seed_values = claim_subset_seed_values(
                comparison_payload,
                scenarios=spec.scenarios,
                fallback_value=comparison_value,
            )
            comparison_uncertainty[comparison_name] = (
                aggregate_with_uncertainty(
                    [
                        SeedLevelResult(
                            metric_name=spec.primary_metric,
                            seed=seed,
                            value=value,
                            condition=comparison_name,
                        )
                        for seed, value in comparison_seed_values
                    ]
                )
            )
            stats = claim_comparison_statistics(
                reference_seed_values=reference_seed_values,
                comparison_seed_values=comparison_seed_values,
                comparison_name=comparison_name,
                fallback_delta=deltas[comparison_name],
            )
            delta_uncertainty[comparison_name] = stats["delta_uncertainty"]
            effect_size_uncertainty[comparison_name] = stats[
                "effect_size_uncertainty"
            ]
            cohens_d_values[comparison_name] = float(stats["cohens_d"])
            effect_magnitudes[comparison_name] = str(stats["effect_magnitude"])

        if spec.name == "learning_without_privileged_signals":
            delta_threshold = claim_threshold_from_phrase(
                spec.success_criterion,
                "by at least",
            )
            if delta_threshold is None:
                return skip(
                    f"Could not parse success criterion {spec.success_criterion!r}.",
                )
            leakage_audit = claim_leakage_audit_summary()
            trained_value = comparison_values.get("trained_without_reflex_support")
            trained_delta = deltas.get("trained_without_reflex_support")
            if trained_value is None or trained_delta is None:
                return skip(
                    "Missing trained_without_reflex_support comparison data.",
                )
            passed = bool(
                trained_delta >= delta_threshold
                and int(leakage_audit["finding_count"]) == 0
            )
            return finish({
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reference_value": reference_value,
                "comparison_values": comparison_values,
                "delta": deltas,
                "effect_size": dict(deltas),
                "reference_uncertainty": reference_uncertainty,
                "comparison_uncertainty": comparison_uncertainty,
                "delta_uncertainty": delta_uncertainty,
                "effect_size_uncertainty": effect_size_uncertainty,
                "cohens_d": cohens_d_values,
                "effect_magnitude": effect_magnitudes,
                "primary_metric": spec.primary_metric,
                "scenarios_evaluated": list(spec.scenarios),
                "notes": [
                    spec.success_criterion,
                    f"Leakage audit unresolved findings: {leakage_audit['finding_count']}.",
                ],
            })

        if spec.name == "escape_without_reflex_support":
            minimum_success = claim_threshold_from_operator(
                spec.success_criterion,
                ">=",
            )
            delta_threshold = claim_threshold_from_phrase(
                spec.success_criterion,
                "by at least",
            )
            if minimum_success is None or delta_threshold is None:
                return skip(
                    f"Could not parse success criterion {spec.success_criterion!r}.",
                )
            trained_value = comparison_values.get("trained_without_reflex_support")
            trained_delta = deltas.get("trained_without_reflex_support")
            if trained_value is None or trained_delta is None:
                return skip(
                    "Missing trained_without_reflex_support comparison data.",
                )
            passed = bool(
                trained_value >= minimum_success
                and trained_delta >= delta_threshold
            )
            return finish({
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reference_value": reference_value,
                "comparison_values": comparison_values,
                "delta": deltas,
                "effect_size": dict(deltas),
                "reference_uncertainty": reference_uncertainty,
                "comparison_uncertainty": comparison_uncertainty,
                "delta_uncertainty": delta_uncertainty,
                "effect_size_uncertainty": effect_size_uncertainty,
                "cohens_d": cohens_d_values,
                "effect_magnitude": effect_magnitudes,
                "primary_metric": spec.primary_metric,
                "scenarios_evaluated": list(spec.scenarios),
                "notes": [spec.success_criterion],
            })

        return skip(
            f"Unsupported learning-evidence claim test {spec.name!r}.",
        )

    if source == "ablation":
        payload = payloads.get("ablation")
        reference_payload, reason = claim_registry_entry(
            payload,
            registry_key="variants",
            entry_name=spec.reference_condition,
        )
        if reference_payload is None:
            return skip(str(reason))

        if spec.name == "memory_improves_shelter_return":
            reference_value, reason = claim_subset_scenario_success_rate(
                reference_payload,
                scenarios=spec.scenarios,
            )
            if reference_value is None:
                return skip(str(reason))
            reference_seed_values = claim_subset_seed_values(
                reference_payload,
                scenarios=spec.scenarios,
                fallback_value=reference_value,
            )
            reference_uncertainty = aggregate_with_uncertainty(
                [
                    SeedLevelResult(
                        metric_name=spec.primary_metric,
                        seed=seed,
                        value=value,
                        condition=spec.reference_condition,
                    )
                    for seed, value in reference_seed_values
                ]
            )
            comparison_values: Dict[str, float] = {}
            comparison_uncertainty: Dict[str, object] = {}
            deltas: Dict[str, float] = {}
            delta_uncertainty: Dict[str, object] = {}
            effect_size_uncertainty: Dict[str, object] = {}
            cohens_d_values: Dict[str, float] = {}
            effect_magnitudes: Dict[str, str] = {}
            for comparison_name in spec.comparison_conditions:
                comparison_payload, reason = claim_registry_entry(
                    payload,
                    registry_key="variants",
                    entry_name=comparison_name,
                )
                if comparison_payload is None:
                    return skip(str(reason))
                comparison_value, reason = claim_subset_scenario_success_rate(
                    comparison_payload,
                    scenarios=spec.scenarios,
                )
                if comparison_value is None:
                    return skip(str(reason))
                comparison_values[comparison_name] = comparison_value
                deltas[comparison_name] = round(comparison_value - reference_value, 6)
                comparison_seed_values = claim_subset_seed_values(
                    comparison_payload,
                    scenarios=spec.scenarios,
                    fallback_value=comparison_value,
                )
                comparison_uncertainty[comparison_name] = (
                    aggregate_with_uncertainty(
                        [
                            SeedLevelResult(
                                metric_name=spec.primary_metric,
                                seed=seed,
                                value=value,
                                condition=comparison_name,
                            )
                            for seed, value in comparison_seed_values
                        ]
                    )
                )
                stats = claim_comparison_statistics(
                    reference_seed_values=reference_seed_values,
                    comparison_seed_values=comparison_seed_values,
                    comparison_name=comparison_name,
                    fallback_delta=deltas[comparison_name],
                )
                delta_uncertainty[comparison_name] = stats["delta_uncertainty"]
                effect_size_uncertainty[comparison_name] = stats[
                    "effect_size_uncertainty"
                ]
                cohens_d_values[comparison_name] = float(stats["cohens_d"])
                effect_magnitudes[comparison_name] = str(stats["effect_magnitude"])
            delta_threshold = claim_threshold_from_phrase(
                spec.success_criterion,
                "by at least",
            )
            if delta_threshold is None:
                return skip(
                    f"Could not parse success criterion {spec.success_criterion!r}.",
                )
            comparison_name = spec.comparison_conditions[0]
            passed = bool(deltas.get(comparison_name, 0.0) >= delta_threshold)
            return finish({
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reference_value": reference_value,
                "comparison_values": comparison_values,
                "delta": deltas,
                "effect_size": dict(deltas),
                "reference_uncertainty": reference_uncertainty,
                "comparison_uncertainty": comparison_uncertainty,
                "delta_uncertainty": delta_uncertainty,
                "effect_size_uncertainty": effect_size_uncertainty,
                "cohens_d": cohens_d_values,
                "effect_magnitude": effect_magnitudes,
                "primary_metric": spec.primary_metric,
                "scenarios_evaluated": list(spec.scenarios),
                "notes": [spec.success_criterion],
            })

        if spec.name == "specialization_emerges_with_multiple_predators":
            comparison_summary = compare_predator_type_ablation_performance(
                payload or {},
                variant_names=(spec.reference_condition, *spec.comparison_conditions),
            )
            comparisons = comparison_summary.get("comparisons", {})
            if not isinstance(comparisons, dict):
                return skip(
                    "Predator-type ablation comparison did not return comparison rows.",
                )
            reference_comparison = comparisons.get(spec.reference_condition)
            if not isinstance(reference_comparison, dict):
                return skip(
                    f"Predator-type comparison is missing reference variant {spec.reference_condition!r}.",
                )
            reference_value = reference_comparison.get(
                "visual_minus_olfactory_success_rate"
            )
            if reference_value is None:
                return skip(
                    f"Reference variant {spec.reference_condition!r} is missing "
                    "visual_minus_olfactory_success_rate.",
                )
            reference_value = round(float(reference_value), 6)
            reference_rows = visual_minus_olfactory_seed_rows(
                reference_payload,
                condition=spec.reference_condition,
                metric_name=spec.primary_metric,
                fallback_value=reference_value,
            )
            reference_seed_values = [
                (row.seed, row.value) for row in reference_rows
            ]
            reference_uncertainty = aggregate_with_uncertainty(
                reference_rows
            )
            comparison_values: Dict[str, float] = {}
            comparison_uncertainty: Dict[str, object] = {}
            deltas: Dict[str, float] = {}
            delta_uncertainty: Dict[str, object] = {}
            effect_sizes: Dict[str, float | None] = {}
            effect_size_uncertainty: Dict[str, object] = {}
            cohens_d_values: Dict[str, float] = {}
            effect_magnitudes: Dict[str, str] = {}
            for comparison_name in spec.comparison_conditions:
                comparison_payload = comparisons.get(comparison_name)
                if not isinstance(comparison_payload, dict):
                    return skip(
                        f"Predator-type comparison is missing {comparison_name!r}.",
                    )
                raw_value = comparison_payload.get("visual_minus_olfactory_success_rate")
                if raw_value is None:
                    return skip(
                        f"Comparison {comparison_name!r} is missing "
                        "visual_minus_olfactory_success_rate.",
                    )
                comparison_value = round(float(raw_value), 6)
                comparison_values[comparison_name] = comparison_value
                deltas[comparison_name] = round(
                    comparison_value - reference_value,
                    6,
                )
                raw_effect_size = comparison_payload.get(
                    "visual_minus_olfactory_success_rate_delta"
                )
                effect_sizes[comparison_name] = (
                    round(float(raw_effect_size), 6)
                    if raw_effect_size is not None
                    else None
                )
                variant_payload, variant_reason = claim_registry_entry(
                    payload,
                    registry_key="variants",
                    entry_name=comparison_name,
                )
                if variant_payload is None:
                    return skip(str(variant_reason))
                comparison_rows = visual_minus_olfactory_seed_rows(
                    variant_payload,
                    condition=comparison_name,
                    metric_name=spec.primary_metric,
                    fallback_value=comparison_value,
                )
                comparison_seed_values = [
                    (row.seed, row.value) for row in comparison_rows
                ]
                comparison_uncertainty[comparison_name] = (
                    aggregate_with_uncertainty(comparison_rows)
                )
                stats = claim_comparison_statistics(
                    reference_seed_values=reference_seed_values,
                    comparison_seed_values=comparison_seed_values,
                    comparison_name=comparison_name,
                    fallback_delta=effect_sizes[comparison_name],
                )
                delta_uncertainty[comparison_name] = stats["delta_uncertainty"]
                effect_size_uncertainty[comparison_name] = stats[
                    "effect_size_uncertainty"
                ]
                cohens_d_values[comparison_name] = float(stats["cohens_d"])
                effect_magnitudes[comparison_name] = str(stats["effect_magnitude"])
            engagement_threshold = claim_count_threshold(spec.success_criterion)
            negative_threshold = claim_threshold_from_operator(
                spec.success_criterion,
                "<=",
            )
            positive_threshold = claim_threshold_from_operator(
                spec.success_criterion,
                ">=",
            )
            representation_threshold = claim_threshold_from_phrase(
                spec.success_criterion,
                "representation_specialization_score >=",
            )
            if (
                engagement_threshold is None
                or negative_threshold is None
                or positive_threshold is None
                or representation_threshold is None
            ):
                return skip(
                    f"Could not parse success criterion {spec.success_criterion!r}.",
                )
            engagement_count, engagement_pass_rates, reason = (
                claim_specialization_engagement(
                    payload,
                    variant_name=spec.reference_condition,
                    scenarios=spec.scenarios,
                )
            )
            if engagement_count is None or engagement_pass_rates is None:
                return skip(str(reason))
            visual_drop = comparison_values.get("drop_visual_cortex")
            sensory_drop = comparison_values.get("drop_sensory_cortex")
            if visual_drop is None or sensory_drop is None:
                return skip(
                    "Missing drop_visual_cortex or drop_sensory_cortex comparison data.",
                )
            representation_metrics = representation_specialization_from_payload(
                reference_payload,
            )
            if not bool(representation_metrics.get("available")):
                return skip(
                    f"Reference variant {spec.reference_condition!r} is missing "
                    "representation specialization evidence.",
                )
            representation_score = round(
                safe_float(
                    representation_metrics.get(
                        "representation_specialization_score"
                    )
                ),
                6,
            )
            behavior_tier_passed = bool(
                visual_drop <= negative_threshold
                and sensory_drop >= positive_threshold
            )
            engagement_tier_passed = bool(
                engagement_count >= engagement_threshold
            )
            representation_tier_passed = bool(
                representation_score >= representation_threshold
            )
            passed = bool(
                behavior_tier_passed
                and engagement_tier_passed
                and representation_tier_passed
            )
            notes = [spec.success_criterion]
            if behavior_tier_passed and engagement_tier_passed and not representation_tier_passed:
                notes.append(
                    "Behavioral specialization passed while representation separation stayed below the emerging threshold; interpret this as possible downstream gating or scenario asymmetry rather than clean proposer separation."
                )
            elif (
                representation_tier_passed
                and not (behavior_tier_passed and engagement_tier_passed)
            ):
                notes.append(
                    "Representation separation cleared the emerging threshold without full behavioral specialization; interpret this as early internal differentiation that has not yet stabilized into policy outcomes."
                )
            return finish({
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reference_value": {
                    "visual_minus_olfactory_success_rate": reference_value,
                    "type_specific_cortex_engagement_count": engagement_count,
                    "type_specific_cortex_engagement_pass_rates": engagement_pass_rates,
                    "proposer_divergence_by_module": dict(
                        representation_metrics.get(
                            "proposer_divergence_by_module",
                            {},
                        )
                    ),
                    "action_center_gate_differential": dict(
                        representation_metrics.get(
                            "action_center_gate_differential",
                            {},
                        )
                    ),
                    "action_center_contribution_differential": dict(
                        representation_metrics.get(
                            "action_center_contribution_differential",
                            {},
                        )
                    ),
                    "representation_specialization_score": representation_score,
                    "behavior_tier_passed": behavior_tier_passed,
                    "engagement_tier_passed": engagement_tier_passed,
                    "representation_tier_passed": representation_tier_passed,
                },
                "comparison_values": comparison_values,
                "delta": deltas,
                "effect_size": effect_sizes,
                "reference_uncertainty": reference_uncertainty,
                "comparison_uncertainty": comparison_uncertainty,
                "delta_uncertainty": delta_uncertainty,
                "effect_size_uncertainty": effect_size_uncertainty,
                "cohens_d": cohens_d_values,
                "effect_magnitude": effect_magnitudes,
                "primary_metric": spec.primary_metric,
                "scenarios_evaluated": list(spec.scenarios),
                "notes": notes,
            })

        return skip(
            f"Unsupported ablation-backed claim test {spec.name!r}.",
        )

    if source == "noise_robustness":
        payload = payloads.get("noise_robustness")
        diagonal_score, off_diagonal_score, reason = claim_noise_subset_scores(
            payload,
            scenarios=spec.scenarios,
        )
        if diagonal_score is None or off_diagonal_score is None:
            return skip(str(reason))
        minimum_off_diagonal = claim_threshold_from_operator(
            spec.success_criterion,
            ">=",
        )
        maximum_gap = claim_threshold_from_operator(
            spec.success_criterion,
            "<=",
        )
        if minimum_off_diagonal is None or maximum_gap is None:
            return skip(
                f"Could not parse success criterion {spec.success_criterion!r}.",
            )
        effect_size = round(diagonal_score - off_diagonal_score, 6)
        passed = bool(
            off_diagonal_score >= minimum_off_diagonal
            and effect_size <= maximum_gap
        )
        diagonal_seed_values, off_diagonal_seed_values = (
            claim_noise_subset_seed_values(
                payload,
                scenarios=spec.scenarios,
                diagonal_fallback=diagonal_score,
                off_diagonal_fallback=off_diagonal_score,
            )
        )
        reference_uncertainty = aggregate_with_uncertainty(
            [
                SeedLevelResult(
                    metric_name="diagonal_score",
                    seed=seed,
                    value=value,
                    condition=spec.reference_condition,
                )
                for seed, value in diagonal_seed_values
            ]
        )
        comparison_uncertainty = {
            "off_diagonal": aggregate_with_uncertainty(
                [
                    SeedLevelResult(
                        metric_name="off_diagonal_score",
                        seed=seed,
                        value=value,
                        condition="off_diagonal",
                    )
                    for seed, value in off_diagonal_seed_values
                ]
            )
        }
        noise_delta_stats = claim_comparison_statistics(
            reference_seed_values=diagonal_seed_values,
            comparison_seed_values=off_diagonal_seed_values,
            comparison_name="off_diagonal",
            fallback_delta=round(off_diagonal_score - diagonal_score, 6),
        )
        noise_effect_stats = claim_comparison_statistics(
            reference_seed_values=off_diagonal_seed_values,
            comparison_seed_values=diagonal_seed_values,
            comparison_name="diagonal_minus_off_diagonal",
            fallback_delta=effect_size,
        )
        return finish({
            "status": "passed" if passed else "failed",
            "passed": passed,
            "reference_value": diagonal_score,
            "comparison_values": {"off_diagonal": off_diagonal_score},
            "delta": {"off_diagonal": round(off_diagonal_score - diagonal_score, 6)},
            "effect_size": effect_size,
            "reference_uncertainty": reference_uncertainty,
            "comparison_uncertainty": comparison_uncertainty,
            "delta_uncertainty": {
                "off_diagonal": noise_delta_stats["delta_uncertainty"],
            },
            "effect_size_uncertainty": {
                "off_diagonal": noise_effect_stats["effect_size_uncertainty"],
            },
            "cohens_d": {"off_diagonal": noise_effect_stats["cohens_d"]},
            "effect_magnitude": {
                "off_diagonal": noise_effect_stats["effect_magnitude"],
            },
            "primary_metric": spec.primary_metric,
            "scenarios_evaluated": list(spec.scenarios),
            "notes": [spec.success_criterion],
        })

    return skip(
        f"Unsupported claim-test source {source!r}.",
    )


def build_claim_test_summary(
    claim_results: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    Summarizes claim-test results into pass/fail counts and scaffold-tier metadata.

    Parameters:
        claim_results (Dict[str, Dict[str, object]]): Mapping from claim-test name to its result record.
            Each result record is expected to include a `status` field (commonly `"passed"`, `"failed"`, or `"skipped"`)
            and may include scaffold metadata used for benchmark-of-record summaries.

    Returns:
        Dict[str, object]: Summary dictionary with the following keys:
            - `claim_count`: total number of claim tests processed.
            - `claims_passed`: number of tests whose `status` equals `"passed"`.
            - `claims_failed`: number of tests whose `status` equals `"failed"`.
            - `claims_skipped`: number of tests whose `status` equals `"skipped"`.
            - `claims_at_minimal_manual`: number of tests classified at the minimal-manual scaffold level.
            - `claims_at_standard_constrained`: number of tests classified at the standard-constrained level.
            - `claims_at_scaffolded_runtime`: number of tests classified at the scaffolded-runtime level.
            - `benchmark_of_record_claims`: number of passing tests marked benchmark-of-record eligible.
            - `all_primary_claims_passed`: `true` if every executed primary claim has a truthy `passed` value in
              `claim_results`, `false` otherwise.
            - `all_primary_claims_benchmark_eligible`: `true` if every passed primary claim is classified as
              minimal manual and benchmark-of-record eligible, `false` otherwise.
            - `primary_claims`: list of canonical primary claim names derived from the claim registry.
    """
    claims_passed = sum(
        1
        for result in claim_results.values()
        if str(result.get("status")) == "passed"
    )
    claims_failed = sum(
        1
        for result in claim_results.values()
        if str(result.get("status")) == "failed"
    )
    claims_skipped = sum(
        1
        for result in claim_results.values()
        if str(result.get("status")) == "skipped"
    )
    claims_at_minimal_manual = sum(
        1
        for result in claim_results.values()
        if str(result.get("scaffold_support_level")) == "minimal_manual"
    )
    claims_at_standard_constrained = sum(
        1
        for result in claim_results.values()
        if str(result.get("scaffold_support_level")) == "standard_constrained"
    )
    claims_at_scaffolded_runtime = sum(
        1
        for result in claim_results.values()
        if str(result.get("scaffold_support_level")) == "scaffolded_runtime"
    )
    benchmark_of_record_claims = sum(
        1
        for result in claim_results.values()
        if bool(result.get("passed", False))
        and bool(result.get("benchmark_of_record_eligible", False))
    )
    primary_claims = primary_claim_test_names()
    executed_primary_claims = [
        name for name in primary_claims if name in claim_results
    ]
    all_primary_claims_passed = bool(executed_primary_claims) and all(
        bool(claim_results[name].get("passed", False))
        for name in executed_primary_claims
    )
    passed_primary_claims = [
        name
        for name in executed_primary_claims
        if bool(claim_results[name].get("passed", False))
    ]
    all_primary_claims_benchmark_eligible = bool(passed_primary_claims) and all(
        str(claim_results[name].get("scaffold_support_level"))
        == "minimal_manual"
        and bool(
            claim_results[name].get("benchmark_of_record_eligible", False)
        )
        for name in passed_primary_claims
    )
    return {
        "claim_count": len(claim_results),
        "claims_passed": claims_passed,
        "claims_failed": claims_failed,
        "claims_skipped": claims_skipped,
        "claims_at_minimal_manual": claims_at_minimal_manual,
        "claims_at_standard_constrained": claims_at_standard_constrained,
        "claims_at_scaffolded_runtime": claims_at_scaffolded_runtime,
        "benchmark_of_record_claims": benchmark_of_record_claims,
        "all_primary_claims_passed": bool(all_primary_claims_passed),
        "all_primary_claims_benchmark_eligible": bool(
            all_primary_claims_benchmark_eligible
        ),
        "primary_claims": list(primary_claims),
    }


def condense_claim_test_summary(
    claim_test_payload: object,
) -> dict[str, object]:
    """Construct a condensed summary of a claim-test payload for CLI display."""
    payload = claim_test_payload if isinstance(claim_test_payload, dict) else {}
    claims = payload.get("claims", {})
    summary = payload.get("summary", {})
    claim_rows = claims if isinstance(claims, dict) else {}
    summary_row = summary if isinstance(summary, dict) else {}
    condensed_claims: dict[str, dict[str, bool]] = {}
    for name, data in claim_rows.items():
        if not isinstance(data, dict):
            continue
        skipped = bool(data.get("skipped")) or str(data.get("status")) == "skipped"
        condensed_claims[str(name)] = {
            "passed": bool(data.get("passed", False)) and not skipped,
            "skipped": skipped,
        }

    def summary_count(key: str) -> int:
        try:
            return int(summary_row.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "claims": condensed_claims,
        "claims_passed": summary_count("claims_passed"),
        "claims_failed": summary_count("claims_failed"),
        "claims_skipped": summary_count("claims_skipped"),
        "all_primary_claims_passed": bool(
            summary_row.get("all_primary_claims_passed", False)
        ),
    }


def run_claim_test_suite(
    *,
    claim_tests: Sequence[str] | None = None,
    width: int = 12,
    height: int = 12,
    food_count: int = 4,
    day_length: int = 18,
    night_length: int = 12,
    max_steps: int | None = None,
    episodes: int | None = None,
    evaluation_episodes: int | None = None,
    gamma: float = 0.96,
    module_lr: float = 0.010,
    motor_lr: float = 0.012,
    module_dropout: float = 0.05,
    reward_profile: str = "classic",
    map_template: str = "central_burrow",
    brain_config: BrainAblationConfig | None = None,
    operational_profile: str | OperationalProfile | None = None,
    noise_profile: str | NoiseConfig | None = None,
    budget_profile: str | BudgetProfile | None = None,
    long_budget_profile: str | BudgetProfile | None = "report",
    seeds: Sequence[int] | None = None,
    episodes_per_scenario: int | None = None,
    robustness_matrix: RobustnessMatrixSpec | None = None,
    checkpoint_selection: str = "none",
    checkpoint_metric: str = "scenario_success_rate",
    checkpoint_override_penalty: float = 0.0,
    checkpoint_dominance_penalty: float = 0.0,
    checkpoint_penalty_mode: CheckpointPenaltyMode | str = (
        CheckpointPenaltyMode.TIEBREAKER
    ),
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | Path | None = None,
    ablation_payload: Dict[str, object] | None = None,
    learning_evidence_payload: Dict[str, object] | None = None,
    noise_robustness_payload: Dict[str, object] | None = None,
    austere_survival_payload: Dict[str, object] | None = None,
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Run selected claim tests by synthesizing or reusing primitive comparison payloads.

    This method resolves requested claim-test specs, ensures required primitive comparison data (learning evidence, ablation, noise robustness, and austere survival gates for primary claims) are available by reusing provided payloads or invoking the corresponding comparison routines, evaluates each claim test to produce structured pass/skip/fail results, and returns a combined claims payload plus CSV-like row records suitable for export.

    Parameters:
        claim_tests: Optional sequence of claim-test identifiers or specs to run; if None all canonical claim tests are evaluated.
        width, height, food_count, day_length, night_length, max_steps:
            Environment layout and step-limit overrides for any generated comparison runs.
        episodes, evaluation_episodes, episodes_per_scenario:
            Training / evaluation budget overrides used for generated primitive comparisons.
        gamma, module_lr, motor_lr, module_dropout:
            Learning hyperparameter overrides applied when running training comparisons.
        reward_profile, map_template, brain_config, operational_profile, noise_profile:
            Configuration overrides used when generating comparison payloads.
        budget_profile, long_budget_profile:
            Budget profile names for base and long-form comparisons where applicable.
        seeds:
            Sequence of RNG seeds to use for generated comparisons; when omitted comparison helpers choose defaults.
        robustness_matrix:
            Optional robustness-matrix spec to use for noise-robustness comparisons.
        checkpoint_selection, checkpoint_metric, checkpoint_interval, checkpoint_dir:
            Checkpointing controls passed through to comparison runs that support candidate selection.
        ablation_payload, learning_evidence_payload, noise_robustness_payload, austere_survival_payload:
            Optional precomputed primitive payloads to reuse; when provided the method will not regenerate that source.

    Returns:
        tuple:
            - claims_payload (dict): A mapping with keys "claims" (per-claim result dicts), "summary" (aggregate pass/skip/fail counts and primary-claims gating), and "metadata" (requested tests, required sources, per-source metadata, seeds, noise profile mapping, and leakage-audit summary).
            - rows (list[dict]): CSV-ready row dictionaries, one per evaluated claim test, containing serialized reference/comparison values, deltas, effect sizes, evaluated scenarios, status, reason, and notes.
    """
    resolved_claim_tests = resolve_claim_tests(claim_tests)
    required_sources = {
        source
        for spec in resolved_claim_tests
        if (source := claim_test_source(spec)) is not None
    }
    austere_survival_required = any(
        spec.austere_survival_required for spec in resolved_claim_tests
    )
    austere_scenarios = list(
        dict.fromkeys(
            scenario_name
            for spec in resolved_claim_tests
            if spec.austere_survival_required
            for scenario_name in spec.scenarios
            if SCENARIO_AUSTERE_REQUIREMENTS.get(
                scenario_name,
                {},
            ).get("requirement_level") == "gate"
        )
    )
    if austere_survival_required:
        required_sources.add("austere_survival")
    learning_scenarios = list(
        dict.fromkeys(
            scenario_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "learning_evidence"
            for scenario_name in spec.scenarios
        )
    )
    learning_conditions = list(
        dict.fromkeys(
            condition_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "learning_evidence"
            for condition_name in (spec.reference_condition, *spec.comparison_conditions)
        )
    )
    ablation_scenarios = list(
        dict.fromkeys(
            scenario_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "ablation"
            for scenario_name in spec.scenarios
        )
    )
    ablation_variants = list(
        dict.fromkeys(
            variant_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "ablation"
            for variant_name in (spec.reference_condition, *spec.comparison_conditions)
        )
    )
    noise_scenarios = list(
        dict.fromkeys(
            scenario_name
            for spec in resolved_claim_tests
            if claim_test_source(spec) == "noise_robustness"
            for scenario_name in spec.scenarios
        )
    )

    payloads: Dict[str, Dict[str, object]] = {}
    source_reused = {
        "ablation": ablation_payload is not None,
        "austere_survival": austere_survival_payload is not None,
        "learning_evidence": learning_evidence_payload is not None,
        "noise_robustness": noise_robustness_payload is not None,
    }
    if "learning_evidence" in required_sources:
        if learning_evidence_payload is None:
            learning_evidence_payload, _ = compare_learning_evidence(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=max_steps,
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                reward_profile=reward_profile,
                map_template=map_template,
                brain_config=brain_config,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                budget_profile=budget_profile,
                long_budget_profile=long_budget_profile,
                seeds=seeds,
                names=learning_scenarios or None,
                condition_names=learning_conditions or None,
                episodes_per_scenario=episodes_per_scenario,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
            )
        payloads["learning_evidence"] = learning_evidence_payload
    if "ablation" in required_sources:
        if ablation_payload is None:
            ablation_payload, _ = compare_ablation_suite(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=max_steps,
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                reward_profile=reward_profile,
                map_template=map_template,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                budget_profile=budget_profile,
                seeds=seeds,
                names=ablation_scenarios or None,
                variant_names=ablation_variants or None,
                episodes_per_scenario=episodes_per_scenario,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
            )
        payloads["ablation"] = ablation_payload
    if "noise_robustness" in required_sources:
        if noise_robustness_payload is None:
            noise_robustness_payload, _ = compare_noise_robustness(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=max_steps,
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                reward_profile=reward_profile,
                map_template=map_template,
                operational_profile=operational_profile,
                budget_profile=budget_profile,
                seeds=seeds,
                names=noise_scenarios or None,
                episodes_per_scenario=episodes_per_scenario,
                robustness_matrix=robustness_matrix,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
            )
        payloads["noise_robustness"] = noise_robustness_payload
    if "austere_survival" in required_sources:
        if austere_survival_payload is None:
            comparison_profiles = tuple(
                dict.fromkeys((str(reward_profile), "austere"))
            )
            austere_survival_payload, _ = compare_behavior_suite(
                width=width,
                height=height,
                food_count=food_count,
                day_length=day_length,
                night_length=night_length,
                max_steps=max_steps,
                episodes=episodes,
                evaluation_episodes=evaluation_episodes,
                gamma=gamma,
                module_lr=module_lr,
                motor_lr=motor_lr,
                module_dropout=module_dropout,
                operational_profile=operational_profile,
                noise_profile=noise_profile,
                budget_profile=budget_profile,
                reward_profiles=comparison_profiles,
                map_templates=(map_template,),
                seeds=seeds,
                names=austere_scenarios or None,
                episodes_per_scenario=episodes_per_scenario,
                checkpoint_selection=checkpoint_selection,
                checkpoint_metric=checkpoint_metric,
                checkpoint_override_penalty=checkpoint_override_penalty,
                checkpoint_dominance_penalty=checkpoint_dominance_penalty,
                checkpoint_penalty_mode=checkpoint_penalty_mode,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
            )
        payloads["austere_survival"] = austere_survival_payload

    claim_results: Dict[str, Dict[str, object]] = {
        spec.name: evaluate_claim_test(spec, payloads)
        for spec in resolved_claim_tests
    }
    rows: List[Dict[str, object]] = []
    for spec in resolved_claim_tests:
        result = claim_results[spec.name]
        austere_gate = result.get("austere_survival_gate")
        if not isinstance(austere_gate, dict):
            austere_gate = {}
        row = {
            "claim_test": spec.name,
            "claim_test_status": result.get("status"),
            "claim_test_passed": bool(result.get("passed", False)),
            "claim_test_austere_survival_required": bool(
                result.get("austere_survival_required", False)
            ),
            "claim_test_austere_survival_passed": bool(
                austere_gate.get("passed", False)
            ),
            "claim_test_austere_survival_gate": json.dumps(
                austere_gate,
                sort_keys=True,
            ),
            "claim_test_scaffold_support_level": result.get(
                "scaffold_support_level"
            ),
            "claim_test_scaffold_findings": json.dumps(
                result.get("scaffold_findings", []),
                sort_keys=True,
            ),
            "claim_test_benchmark_of_record_eligible": bool(
                result.get("benchmark_of_record_eligible", False)
            ),
            "claim_test_severity": result.get("claim_severity"),
            "claim_test_primary_metric": result.get("primary_metric"),
            "claim_test_reference_condition": spec.reference_condition,
            "claim_test_comparison_conditions": json.dumps(
                list(spec.comparison_conditions),
                sort_keys=True,
            ),
            "claim_test_reference_value": json.dumps(
                result.get("reference_value"),
                sort_keys=True,
            ),
            "claim_test_comparison_values": json.dumps(
                result.get("comparison_values", {}),
                sort_keys=True,
            ),
            "claim_test_delta": json.dumps(
                result.get("delta", {}),
                sort_keys=True,
            ),
            "claim_test_effect_size": json.dumps(
                result.get("effect_size"),
                sort_keys=True,
            ),
            "claim_test_reference_uncertainty": json.dumps(
                result.get("reference_uncertainty"),
                sort_keys=True,
            ),
            "claim_test_comparison_uncertainty": json.dumps(
                result.get("comparison_uncertainty", {}),
                sort_keys=True,
            ),
            "claim_test_delta_uncertainty": json.dumps(
                result.get("delta_uncertainty", {}),
                sort_keys=True,
            ),
            "claim_test_effect_size_uncertainty": json.dumps(
                result.get("effect_size_uncertainty", {}),
                sort_keys=True,
            ),
            "claim_test_cohens_d": json.dumps(
                result.get("cohens_d", {}),
                sort_keys=True,
            ),
            "claim_test_effect_magnitude": json.dumps(
                result.get("effect_magnitude", {}),
                sort_keys=True,
            ),
            "claim_test_scenarios": json.dumps(
                result.get("scenarios_evaluated", []),
                sort_keys=True,
            ),
            "claim_test_reason": str(result.get("reason", "")),
            "claim_test_notes": json.dumps(result.get("notes", []), sort_keys=True),
        }
        rows.append(row)

    def _metadata_sequence_or_empty(value: object) -> list[object]:
        """
        Return a list copy of the input if it is a list or tuple, otherwise return an empty list.
        
        Parameters:
            value (object): The value to coerce into a list.
        
        Returns:
            list[object]: A new list containing the elements of `value` when it is a list or tuple, or an empty list otherwise.
        """
        if isinstance(value, (list, tuple)):
            return list(value)
        return []

    def _metadata_payload_or_empty(value: object) -> dict[str, object]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    source_metadata: Dict[str, object] = {}
    metadata_seeds: set[int] = set()
    noise_profiles: Dict[str, object] = {}
    for source_name, raw_payload in payloads.items():
        payload = _metadata_payload_or_empty(raw_payload)
        source_metadata[source_name] = {
            "reused": bool(source_reused.get(source_name, False)),
            "budget_profile": payload.get("budget_profile"),
            "benchmark_strength": payload.get("benchmark_strength"),
            "noise_profile": payload.get("noise_profile"),
            "seeds": _metadata_sequence_or_empty(payload.get("seeds")),
            "scenario_names": _metadata_sequence_or_empty(
                payload.get("scenario_names")
            ),
            "episodes_per_scenario": payload.get("episodes_per_scenario"),
        }
        for seed in _metadata_sequence_or_empty(payload.get("seeds")):
            if isinstance(seed, int):
                metadata_seeds.add(int(seed))
        noise_profile = payload.get("noise_profile")
        if noise_profile is not None:
            noise_profiles[source_name] = noise_profile
    metadata = {
        "requested_claim_tests": [spec.name for spec in resolved_claim_tests],
        "required_sources": sorted(required_sources),
        "sources": source_metadata,
        "seeds": sorted(metadata_seeds),
        "noise_profiles": noise_profiles,
        "leakage_audit": claim_leakage_audit_summary(),
    }
    return {
        "claims": claim_results,
        "summary": build_claim_test_summary(claim_results),
        "metadata": metadata,
    }, rows
