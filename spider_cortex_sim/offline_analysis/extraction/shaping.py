from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

from ...ablations import compare_predator_type_ablation_performance
from ..constants import (
    CANONICAL_NOISE_CONDITIONS,
    DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    DEFAULT_MODULE_NAMES,
    REFLEX_DOMINANCE_WARNING_THRESHOLD,
    REFLEX_OVERRIDE_WARNING_THRESHOLD,
    SHAPING_DEPENDENCE_WARNING_THRESHOLD,
)
from ..uncertainty import _payload_has_zero_reflex_scale, _payload_uncertainty
from ..utils import (
    _coerce_bool,
    _coerce_float,
    _coerce_optional_float,
    _dominant_module_by_score,
    _mapping_or_empty,
    _mean,
    _safe_divide,
)

def _profile_disposition_weights(
    reward_profiles: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    """
    Builds a mapping from reward profile names to disposition weight proxies rounded to six decimals.
    
    Parameters:
        reward_profiles (Mapping[str, object]): Mapping from profile name to a payload that is expected to contain a `disposition_summary` mapping whose values expose `total_weight_proxy`.
    
    Returns:
        dict[str, dict[str, float]]: A nested mapping where outer keys are profile names and inner keys are disposition names; inner values are the `total_weight_proxy` coerced to a finite float and rounded to six decimal places. Profiles or disposition entries that are not mappings are skipped.
    """
    weights: dict[str, dict[str, float]] = {}
    for profile_name, profile_payload in sorted(reward_profiles.items()):
        if not isinstance(profile_payload, Mapping):
            continue
        disposition_summary = profile_payload.get("disposition_summary", {})
        if not isinstance(disposition_summary, Mapping):
            continue
        weights[str(profile_name)] = {
            str(disposition): round(
                _coerce_float(
                    payload.get("total_weight_proxy")
                    if isinstance(payload, Mapping)
                    else 0.0
                ),
                6,
            )
            for disposition, payload in sorted(disposition_summary.items())
        }
    return weights

def _component_disposition_rows(
    reward_components: Mapping[str, object],
) -> list[dict[str, object]]:
    """
    Normalize reward component metadata into rows describing each component's shaping disposition.
    
    Parameters:
        reward_components (Mapping[str, object]): Mapping from component name to a metadata mapping; entries whose metadata is not a mapping are ignored.
    
    Returns:
        list[dict[str, object]]: List of rows with keys:
            - component: component name as a string
            - category: component category or empty string
            - risk: shaping risk or empty string
            - disposition: shaping disposition or empty string
            - rationale: disposition rationale or empty string
    """
    rows: list[dict[str, object]] = []
    for component_name, metadata in sorted(reward_components.items()):
        if not isinstance(metadata, Mapping):
            continue
        rows.append(
            {
                "component": str(component_name),
                "category": str(metadata.get("category") or ""),
                "risk": str(metadata.get("shaping_risk") or ""),
                "disposition": str(metadata.get("shaping_disposition") or ""),
                "rationale": str(metadata.get("disposition_rationale") or ""),
            }
        )
    return rows

def _component_disposition_summary(
    component_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Builds a disposition-centric summary of reward components.
    
    Parameters:
        component_rows (Sequence[Mapping[str, object]]): Iterable of rows where each row is expected to contain
            `"component"` (component name) and `"disposition"` (classification) keys.
    
    Returns:
        disposition_summary (dict[str, object]): Mapping from disposition name to a summary dict with:
            - `components` (list[str]): Alphabetically sorted list of component names for that disposition.
            - `component_count` (int): Number of components in the list.
    """
    summary: dict[str, list[str]] = {}
    for row in component_rows:
        disposition = str(row.get("disposition") or "")
        component = str(row.get("component") or "")
        if not disposition or not component:
            continue
        summary.setdefault(disposition, []).append(component)
    return {
        disposition: {
            "components": sorted(components),
            "component_count": len(components),
        }
        for disposition, components in sorted(summary.items())
    }

def _normalize_behavior_survival(
    behavior_survival: Mapping[str, object],
) -> dict[str, object]:
    """
    Normalize a behavior_survival payload into a consistent summary of per-scenario survival metrics.
    
    Parameters:
        behavior_survival (Mapping[str, object]): Raw `behavior_survival` payload which may contain `scenarios` as a mapping or list and optional aggregate fields.
    
    Returns:
        dict[str, object]: A normalized mapping with the following keys:
            available (bool): True when there is at least one scenario row and the payload does not explicitly mark availability false.
            minimal_profile (str): The `minimal_profile` identifier (empty string if missing).
            survival_threshold (float): The survival threshold (falls back to DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD).
            scenario_count (int): Number of scenarios (uses provided `scenario_count` or the count of parsed scenarios).
            surviving_scenario_count (int): Number of scenarios marked as surviving (uses provided `surviving_scenario_count` or computed value).
            survival_rate (float): Overall survival rate rounded to 6 decimals (uses provided `survival_rate` or computed surviving/scenario ratio).
            scenarios (list[dict[str, object]]): Sorted list of per-scenario rows, each containing:
                scenario (str): Scenario identifier.
                austere_success_rate (float): Rounded austere success rate (6 decimal places).
                survives (bool): Whether the scenario is considered to survive.
                episodes (int): Number of episodes for the scenario.
    """
    raw_scenarios = behavior_survival.get("scenarios", {})
    scenario_rows: list[dict[str, object]] = []
    if isinstance(raw_scenarios, Mapping):
        iterable = raw_scenarios.items()
    elif isinstance(raw_scenarios, list):
        iterable = (
            (
                str(item.get("scenario") or ""),
                item,
            )
            for item in raw_scenarios
            if isinstance(item, Mapping)
        )
    else:
        iterable = ()
    default_survival_threshold = _coerce_float(
        behavior_survival.get("survival_threshold"),
        DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    )
    for scenario_name, payload in iterable:
        if not isinstance(payload, Mapping):
            continue
        scenario = str(scenario_name or payload.get("scenario") or "")
        if not scenario:
            continue
        success_rate = payload.get("austere_success_rate")
        if success_rate is None:
            success_rate = payload.get("success_rate")
        success_rate_value = _coerce_float(success_rate)
        survival_threshold = _coerce_float(
            payload.get("survival_threshold"),
            default_survival_threshold,
        )
        survives_value = (
            payload.get("survives")
            if "survives" in payload
            else success_rate_value >= survival_threshold
        )
        scenario_rows.append(
            {
                "scenario": scenario,
                "austere_success_rate": round(success_rate_value, 6),
                "survives": _coerce_bool(survives_value),
                "episodes": int(_coerce_float(payload.get("episodes"), 0.0)),
            }
        )
    scenario_rows = sorted(scenario_rows, key=lambda item: str(item["scenario"]))
    scenario_count = int(
        _coerce_float(behavior_survival.get("scenario_count"), len(scenario_rows))
    )
    if scenario_count <= 0:
        scenario_count = len(scenario_rows)
    surviving_count = int(
        _coerce_float(
            behavior_survival.get("surviving_scenario_count"),
            sum(1 for item in scenario_rows if bool(item["survives"])),
        )
    )
    survival_rate = _coerce_float(
        behavior_survival.get("survival_rate"),
        _safe_divide(float(surviving_count), float(scenario_count)),
    )
    raw_available = behavior_survival.get("available")
    available = (
        bool(scenario_rows)
        if raw_available is None
        else bool(raw_available) and bool(scenario_rows)
    )
    return {
        "available": available,
        "minimal_profile": str(behavior_survival.get("minimal_profile") or ""),
        "survival_threshold": default_survival_threshold,
        "scenario_count": scenario_count,
        "surviving_scenario_count": surviving_count,
        "survival_rate": round(survival_rate, 6),
        "scenarios": scenario_rows,
    }

def extract_shaping_audit(summary: Mapping[str, object]) -> dict[str, object]:
    """
    Extract dense-vs-minimal shaping diagnostics from a simulation `summary`.
    
    Reads `summary["reward_audit"]` and produces diagnostics comparing a dense (e.g., "classic")
    reward profile to a minimal (e.g., "austere") profile, including gap metrics,
    per-profile disposition weight proxies, component disposition rows, normalized
    minimal-profile behavior survival, interpreted flags (e.g., shaping dependence),
    and human-readable interpretation and limitations.
    
    Parameters:
        summary (Mapping[str, object]): Parsed simulation summary JSON; may omit
            `reward_audit` in which case the result is marked unavailable.
    
    Returns:
        dict[str, object]: A report-ready mapping containing:
            - available (bool): Whether any shaping diagnostics were produced.
            - source (str): Data source identifier (typically "summary.reward_audit" or "none").
            - dense_profile (str), minimal_profile (str): Selected profile names.
            - thresholds (dict): Warning thresholds used (shaping_dependence, behavior_survival).
            - gap_metrics (dict): Numeric deltas for scenario/episode success and mean reward.
            - interpretive_flags (dict): Flags such as `gap_available` and `shaping_dependent`.
            - interpretation (str): Short human-facing interpretation of the gaps.
            - disposition_summary (dict): Aggregated lists/counts of components per disposition.
            - profile_weight_breakdown (dict): Per-profile disposition weight proxies.
            - removed_weight_gap (float): Difference in removed weight between dense and minimal.
            - component_classification (list[dict]): Rows classifying reward components by disposition.
            - behavior_survival (dict): Normalized minimal-profile survival diagnostics.
            - limitations (list[str]): Any observed limitations or missing data notes.
    """
    reward_audit = _mapping_or_empty(summary.get("reward_audit"))
    if not reward_audit:
        return {
            "available": False,
            "source": "none",
            "dense_profile": "classic",
            "minimal_profile": "austere",
            "thresholds": {
                "shaping_dependence": SHAPING_DEPENDENCE_WARNING_THRESHOLD,
                "behavior_survival": DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
            },
            "gap_metrics": {
                "scenario_success_rate_delta": 0.0,
                "episode_success_rate_delta": 0.0,
                "mean_reward_delta": 0.0,
            },
            "interpretive_flags": {
                "gap_available": False,
                "shaping_dependent": False,
            },
            "interpretation": "No reward_audit payload was available.",
            "disposition_summary": {},
            "profile_weight_breakdown": {},
            "removed_weight_gap": 0.0,
            "component_classification": [],
            "behavior_survival": _normalize_behavior_survival({}),
            "limitations": ["No reward_audit payload was available."],
        }

    comparison = _mapping_or_empty(reward_audit.get("comparison"))
    deltas_vs_minimal = _mapping_or_empty(comparison.get("deltas_vs_minimal"))
    minimal_profile = str(
        comparison.get("minimal_profile")
        or reward_audit.get("minimal_profile")
        or "austere"
    )
    dense_profile = "classic"
    if dense_profile not in deltas_vs_minimal:
        dense_profile = next(
            (
                str(profile_name)
                for profile_name in deltas_vs_minimal
                if str(profile_name) != minimal_profile
            ),
            "classic",
        )
    dense_delta_payload = _mapping_or_empty(deltas_vs_minimal.get(dense_profile))
    gap_available = bool(dense_delta_payload)
    gap_metrics = {
        "scenario_success_rate_delta": round(
            _coerce_float(dense_delta_payload.get("scenario_success_rate_delta")),
            6,
        ),
        "episode_success_rate_delta": round(
            _coerce_float(dense_delta_payload.get("episode_success_rate_delta")),
            6,
        ),
        "mean_reward_delta": round(
            _coerce_float(dense_delta_payload.get("mean_reward_delta")),
            6,
        ),
    }
    scenario_gap = gap_metrics["scenario_success_rate_delta"]
    shaping_dependent = bool(
        gap_available and scenario_gap > SHAPING_DEPENDENCE_WARNING_THRESHOLD
    )

    reward_profiles = _mapping_or_empty(reward_audit.get("reward_profiles"))
    profile_weight_breakdown = _profile_disposition_weights(reward_profiles)
    dense_removed_weight = _coerce_float(
        profile_weight_breakdown.get(dense_profile, {}).get("removed")
    )
    minimal_removed_weight = _coerce_float(
        profile_weight_breakdown.get(minimal_profile, {}).get("removed")
    )
    component_classification = _component_disposition_rows(
        _mapping_or_empty(reward_audit.get("reward_components"))
    )
    behavior_survival = _normalize_behavior_survival(
        _mapping_or_empty(comparison.get("behavior_survival"))
    )
    behavior_threshold = _coerce_float(
        behavior_survival.get("survival_threshold"),
        DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    )
    if shaping_dependent:
        interpretation = (
            f"High shaping dependence: {dense_profile} exceeds {minimal_profile} "
            f"by {scenario_gap:.2f} scenario-success points."
        )
    elif gap_available and scenario_gap > 0.0:
        interpretation = (
            f"{dense_profile} exceeds {minimal_profile}, but the gap is below "
            "the warning threshold."
        )
    elif gap_available:
        interpretation = (
            f"{minimal_profile} matches or exceeds {dense_profile} on "
            "scenario success."
        )
    else:
        interpretation = "No classic-vs-austere gap metrics were available."
    limitations: list[str] = []
    if not gap_available:
        limitations.append("No dense-vs-minimal delta was available.")
    if not component_classification:
        limitations.append("No reward component disposition table was available.")
    if not behavior_survival["available"]:
        limitations.append("No austere per-scenario behavior survival data was available.")
    disposition_summary = _component_disposition_summary(component_classification)
    for weights in profile_weight_breakdown.values():
        for disposition in weights:
            disposition_summary.setdefault(
                disposition,
                {"components": [], "component_count": 0},
            )
    return {
        "available": bool(
            gap_available
            or component_classification
            or profile_weight_breakdown
            or behavior_survival["available"]
        ),
        "source": "summary.reward_audit",
        "dense_profile": dense_profile,
        "minimal_profile": minimal_profile,
        "thresholds": {
            "shaping_dependence": SHAPING_DEPENDENCE_WARNING_THRESHOLD,
            "behavior_survival": behavior_threshold,
        },
        "gap_metrics": gap_metrics,
        "interpretive_flags": {
            "gap_available": gap_available,
            "shaping_dependent": shaping_dependent,
        },
        "interpretation": interpretation,
        "disposition_summary": disposition_summary,
        "profile_weight_breakdown": profile_weight_breakdown,
        "removed_weight_gap": round(dense_removed_weight - minimal_removed_weight, 6),
        "component_classification": component_classification,
        "behavior_survival": behavior_survival,
        "limitations": limitations,
    }

def _rows_with_minimal_reflex_support(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """
    Selects rows with the minimal `eval_reflex_scale` for each `ablation_variant`.
    
    Each input row should include an `ablation_variant` key and may include an `eval_reflex_scale` value. Missing or non-finite `eval_reflex_scale` values are treated as 0.0. For each distinct `ablation_variant`, all rows whose `eval_reflex_scale` equals the minimal scale observed for that variant are returned (ties preserved). If no rows are selected, the original input rows are returned.
     
    Parameters:
        rows: Iterable of row mappings; each mapping is expected to contain `ablation_variant` and optionally `eval_reflex_scale`.
    
    Returns:
        A list of row mappings containing, per variant, the rows with the minimal `eval_reflex_scale`, or the original rows if no selection was made.
    """
    by_variant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("ablation_variant") or "")].append(row)
    selected: list[Mapping[str, object]] = []
    for items in by_variant.values():
        if not items:
            continue
        min_scale = min(_coerce_float(item.get("eval_reflex_scale"), 0.0) for item in items)
        selected.extend(
            item
            for item in items
            if math.isclose(
                _coerce_float(item.get("eval_reflex_scale"), 0.0),
                min_scale,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    return selected or list(rows)

def _variant_with_minimal_reflex_support(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """
    Produce a variant payload adjusted to represent minimal (no) reflex support.
    
    Parameters:
        payload (Mapping[str, object]): An ablation variant payload, optionally containing a
            `without_reflex_support` mapping with alternative `config`, `summary`,
            `suite`, or `legacy_scenarios` entries.
    
    Returns:
        dict[str, object]: A shallow copy of `payload` modified so that `eval_reflex_scale`
        is set to `0.0`, `primary_evaluation` is `"without_reflex_support"`, and when
        `without_reflex_support` provides `config`, `summary`, `suite`,
        `legacy_scenarios`, `seed_level`, or `uncertainty`, those entries are promoted
        into the returned mapping.
    """
    result = dict(payload)
    without_reflex = payload.get("without_reflex_support")
    if not isinstance(without_reflex, Mapping):
        return result
    for key in (
        "config",
        "summary",
        "suite",
        "legacy_scenarios",
        "seed_level",
        "uncertainty",
    ):
        if key in without_reflex:
            value = without_reflex[key]
            result[key] = dict(value) if isinstance(value, Mapping) else value
    summary = result.get("summary")
    if isinstance(summary, Mapping):
        result["summary"] = {**summary, "eval_reflex_scale": 0.0}
    result["eval_reflex_scale"] = 0.0
    result["primary_evaluation"] = "without_reflex_support"
    return result

def _summary_delta(
    payload: Mapping[str, object],
    reference: Mapping[str, object],
    key: str,
) -> float:
    """
    Compute the numeric difference for a named summary metric between a payload and a reference.
    
    Parameters:
        payload (Mapping[str, object]): Mapping expected to contain a `summary` mapping with numeric metrics.
        reference (Mapping[str, object]): Mapping expected to contain a `summary` mapping to compare against.
        key (str): The metric name to compare inside each `summary` mapping.
    
    Returns:
        float: The value of `payload["summary"][key] - reference["summary"][key]`, coerced to floats, rounded to 6 decimal places; returns `0.0` if either `summary` is missing or not a mapping.
    """
    payload_summary = payload.get("summary", {})
    reference_summary = reference.get("summary", {})
    if not isinstance(payload_summary, Mapping) or not isinstance(reference_summary, Mapping):
        return 0.0
    return round(
        _coerce_float(payload_summary.get(key))
        - _coerce_float(reference_summary.get(key)),
        6,
    )

def _scenario_delta(
    payload: Mapping[str, object],
    reference: Mapping[str, object],
    scenario_name: str,
) -> float:
    """
    Compute the difference in `success_rate` for a named scenario between two suite payloads.
    
    Parameters:
        payload (Mapping[str, object]): Payload containing a `suite` mapping with per-scenario data.
        reference (Mapping[str, object]): Reference payload containing a `suite` mapping to compare against.
        scenario_name (str): Name of the scenario whose `success_rate` delta to compute.
    
    Returns:
        float: `payload.success_rate - reference.success_rate` rounded to 6 decimal places; `0.0` if the required suite or scenario entries are missing or not mappings.
    """
    payload_suite = payload.get("suite", {})
    reference_suite = reference.get("suite", {})
    if not isinstance(payload_suite, Mapping) or not isinstance(reference_suite, Mapping):
        return 0.0
    payload_scenario = payload_suite.get(scenario_name, {})
    reference_scenario = reference_suite.get(scenario_name, {})
    if not isinstance(payload_scenario, Mapping) or not isinstance(reference_scenario, Mapping):
        return 0.0
    return round(
        _coerce_float(payload_scenario.get("success_rate"))
        - _coerce_float(reference_scenario.get("success_rate")),
        6,
    )

def _ablation_deltas_vs_reference(
    variants: Mapping[str, object],
    *,
    reference_variant: str,
) -> dict[str, object]:
    """
    Compute per-variant deltas of summary metrics and per-scenario success rates against a reference variant.
    
    Parameters:
        variants (Mapping[str, object]): Mapping from variant name to variant payloads (expected to contain a `summary` mapping and an optional `suite` mapping of scenarios).
        reference_variant (str): Key of the variant within `variants` to use as the reference baseline.
    
    Returns:
        dict[str, object]: A mapping keyed by each variant name to a dictionary with:
            - "summary": a mapping with numeric deltas for
                - "scenario_success_rate_delta"
                - "episode_success_rate_delta"
                - "mean_reward_delta"
            - "scenarios": a mapping from scenario name to {"success_rate_delta": float}
        If the reference variant is missing or not a mapping, returns an empty dict.
    """
    reference = variants.get(reference_variant)
    if not isinstance(reference, Mapping):
        return {}
    scenario_names = sorted(
        {
            str(scenario_name)
            for payload in variants.values()
            if isinstance(payload, Mapping)
            for scenario_name in (
                payload.get("suite", {}).keys()
                if isinstance(payload.get("suite"), Mapping)
                else ()
            )
        }
    )
    deltas: dict[str, object] = {}
    for variant_name, payload in variants.items():
        if not isinstance(payload, Mapping):
            continue
        deltas[str(variant_name)] = {
            "summary": {
                "scenario_success_rate_delta": _summary_delta(
                    payload,
                    reference,
                    "scenario_success_rate",
                ),
                "episode_success_rate_delta": _summary_delta(
                    payload,
                    reference,
                    "episode_success_rate",
                ),
                "mean_reward_delta": _summary_delta(
                    payload,
                    reference,
                    "mean_reward",
                ),
            },
            "scenarios": {
                scenario_name: {
                    "success_rate_delta": _scenario_delta(
                        payload,
                        reference,
                        scenario_name,
                    )
                }
                for scenario_name in scenario_names
            },
        }
    return deltas
