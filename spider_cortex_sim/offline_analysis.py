from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ablations import compare_predator_type_ablation_performance
from .noise import canonical_noise_profile_names

DEFAULT_MODULE_NAMES: tuple[str, ...] = (
    "visual_cortex",
    "sensory_cortex",
    "hunger_center",
    "sleep_center",
    "alert_center",
    "monolithic_policy",
)

REFLEX_OVERRIDE_WARNING_THRESHOLD = 0.10
REFLEX_DOMINANCE_WARNING_THRESHOLD = 0.25
SHAPING_DEPENDENCE_WARNING_THRESHOLD = 0.20
DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD = 0.50
CANONICAL_NOISE_CONDITIONS: tuple[str, ...] = canonical_noise_profile_names()


def _coerce_float(value: object, default: float = 0.0) -> float:
    """
    Convert a value to a finite float, with a safe fallback.
    
    Attempts to coerce `value` to a `float`. Booleans are converted as numeric (`True` -> `1.0`, `False` -> `0.0`). Returns `default` when `value` is `None`, cannot be converted to a float, or converts to a non-finite value (`NaN`, `inf`).
    
    Parameters:
        value (object): The input to convert.
        default (float): The value to return when conversion is not possible or yields a non-finite result.
    
    Returns:
        float: The finite float representation of `value`, or `default` if conversion fails or yields a non-finite value.
    """
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _coerce_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _mean(values: Iterable[float]) -> float:
    """
    Compute the arithmetic mean of an iterable of numbers.
    
    Returns:
        The mean of the input values as a float, or 0.0 if the iterable is empty.
    """
    items = list(values)
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _dominant_module_by_score(modules: Mapping[str, float]) -> str:
    """
    Return the module name with the largest numeric value from a mapping of module -> score.

    Parameters:
        modules (Mapping[str, float]): Mapping from module name to a numeric score.

    Returns:
        str: The module name with the highest score, or an empty string if `modules` is empty.
    """
    if not modules:
        return ""
    return max(modules.items(), key=lambda item: _coerce_float(item[1]))[0]


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    """
    Normalize a value to a mapping, yielding an empty mapping when the input is not a mapping.

    Parameters:
        value (object): The value to normalize; if it is a Mapping it is returned unchanged.

    Returns:
        Mapping[str, object]: The original mapping when `value` is a Mapping, otherwise an empty mapping.
    """
    if isinstance(value, Mapping):
        return value
    return {}


def _format_optional_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def load_summary(path: str | Path | None) -> dict[str, object]:
    """
    Load and parse a JSON summary file into a dictionary.
    
    Parameters:
        path (str | Path | None): Path to a JSON file. If `None`, no file is read.
    
    Returns:
        dict[str, object]: Parsed JSON object from the file, or an empty dict when `path` is `None`.
    """
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_trace(path: str | Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    items: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                items.append(payload)
    return items


def load_behavior_csv(path: str | Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def normalize_behavior_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """
    Normalize and canonicalize behavior CSV rows for downstream analysis.
    
    Converts each input mapping into a dictionary with standardized string keys and coerced/typed fields suitable for the rest of the pipeline. Normalized fields include canonical map/profile names and numeric/bool conversions for common metrics.
    
    Parameters:
        rows (Sequence[Mapping[str, object]]): Iterable of raw behavior rows (e.g., csv.DictReader rows or similar mappings).
    
    Returns:
        list[dict[str, object]]: A list of normalized row dictionaries. Each row will contain canonical keys and coerced values, notably:
            - evaluation_map, map_template, scenario_map (str)
            - reward_profile, ablation_variant, ablation_architecture, operational_profile, budget_profile (str)
            - train_noise_profile, eval_noise_profile (str)
            - scenario (str)
            - success (bool)
            - failure_count, simulation_seed, episode_seed, episode (int)
            - reflex_scale, reflex_anneal_final_scale, eval_reflex_scale (float)
    """
    normalized: list[dict[str, object]] = []
    for row in rows:
        item = {str(key): value for key, value in dict(row).items()}
        evaluation_map = str(
            item.get("evaluation_map")
            or item.get("map_template")
            or item.get("scenario_map")
            or ""
        )
        scenario_map = str(item.get("scenario_map") or evaluation_map)
        reward_profile = str(item.get("reward_profile") or "")
        ablation_variant = str(item.get("ablation_variant") or "")
        ablation_architecture = str(item.get("ablation_architecture") or "")
        operational_profile = str(item.get("operational_profile") or "")
        budget_profile = str(item.get("budget_profile") or "")
        train_noise_profile = str(item.get("train_noise_profile") or "")
        eval_noise_profile = str(
            item.get("eval_noise_profile")
            or item.get("noise_profile")
            or ""
        )
        item["evaluation_map"] = evaluation_map
        item["map_template"] = evaluation_map
        item["scenario_map"] = scenario_map
        item["reward_profile"] = reward_profile
        item["ablation_variant"] = ablation_variant
        item["ablation_architecture"] = ablation_architecture
        item["operational_profile"] = operational_profile
        item["budget_profile"] = budget_profile
        item["train_noise_profile"] = train_noise_profile
        item["eval_noise_profile"] = eval_noise_profile
        item["scenario"] = str(item.get("scenario") or "")
        item["success"] = _coerce_bool(item.get("success"))
        item["failure_count"] = int(_coerce_float(item.get("failure_count"), 0.0))
        item["simulation_seed"] = int(_coerce_float(item.get("simulation_seed"), 0.0))
        item["episode_seed"] = int(_coerce_float(item.get("episode_seed"), 0.0))
        item["episode"] = int(_coerce_float(item.get("episode"), 0.0))
        item["reflex_scale"] = _coerce_float(item.get("reflex_scale"), 0.0)
        item["reflex_anneal_final_scale"] = _coerce_float(
            item.get("reflex_anneal_final_scale"),
            item["reflex_scale"],
        )
        item["eval_reflex_scale"] = _coerce_float(
            item.get("eval_reflex_scale"),
            item["reflex_scale"],
        )
        normalized.append(item)
    return normalized


def _extract_reward_from_episode(detail: Mapping[str, object]) -> float:
    if "total_reward" in detail:
        return _coerce_float(detail.get("total_reward"))
    if "reward" in detail:
        return _coerce_float(detail.get("reward"))
    return 0.0


def extract_training_eval_series(summary: Mapping[str, object]) -> dict[str, object]:
    training = summary.get("training", {})
    evaluation = summary.get("evaluation", {})
    training_points: list[dict[str, object]] = []
    evaluation_points: list[dict[str, object]] = []
    limitations: list[str] = []
    source = "none"

    if isinstance(training, Mapping):
        episodes_detail = training.get("episodes_detail")
        if isinstance(episodes_detail, list) and episodes_detail:
            source = "summary.training.episodes_detail"
            for index, detail in enumerate(episodes_detail, start=1):
                if not isinstance(detail, Mapping):
                    continue
                training_points.append(
                    {
                        "index": index,
                        "episode": int(_coerce_float(detail.get("episode"), index)),
                        "reward": _extract_reward_from_episode(detail),
                    }
                )
        else:
            history_tail = training.get("history_tail")
            if isinstance(history_tail, list) and history_tail:
                source = "summary.training.history_tail"
                for index, detail in enumerate(history_tail, start=1):
                    if not isinstance(detail, Mapping):
                        continue
                    training_points.append(
                        {
                            "index": index,
                            "episode": int(_coerce_float(detail.get("episode"), index)),
                            "reward": _extract_reward_from_episode(detail),
                        }
                    )
            else:
                aggregate_reward = None
                if "mean_reward" in training:
                    aggregate_reward = _coerce_float(training.get("mean_reward"))
                elif "mean_reward_all" in training:
                    aggregate_reward = _coerce_float(training.get("mean_reward_all"))
                if aggregate_reward is not None:
                    source = "summary.training.aggregate"
                    training_points.append(
                        {"index": 1, "episode": 1, "reward": aggregate_reward}
                    )
                limitations.append(
                    "Training detail is partial; the chart may reflect a tail or aggregate instead of the full run."
                )

    if isinstance(evaluation, Mapping):
        episodes_detail = evaluation.get("episodes_detail")
        if isinstance(episodes_detail, list) and episodes_detail:
            for index, detail in enumerate(episodes_detail, start=1):
                if not isinstance(detail, Mapping):
                    continue
                evaluation_points.append(
                    {
                        "index": index,
                        "episode": int(_coerce_float(detail.get("episode"), index)),
                        "reward": _extract_reward_from_episode(detail),
                    }
                )
        elif "mean_reward" in evaluation:
            evaluation_points.append(
                {
                    "index": 1,
                    "episode": 1,
                    "reward": _coerce_float(evaluation.get("mean_reward")),
                }
            )
            limitations.append(
                "Evaluation detail is partial; the chart includes aggregate reward only."
            )

    available = bool(training_points or evaluation_points)
    if not available:
        limitations.append("No training or evaluation reward series was available.")
    return {
        "available": available,
        "source": source,
        "training_points": training_points,
        "evaluation_points": evaluation_points,
        "limitations": limitations,
    }


def _suite_from_summary(summary: Mapping[str, object]) -> dict[str, object]:
    behavior_evaluation = summary.get("behavior_evaluation", {})
    if not isinstance(behavior_evaluation, Mapping):
        return {}
    suite = behavior_evaluation.get("suite", {})
    if not isinstance(suite, Mapping):
        return {}
    return dict(suite)


def _scenario_suite_from_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_scenario: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        scenario = str(row.get("scenario") or "")
        if scenario:
            by_scenario[scenario].append(row)
    suite: dict[str, object] = {}
    for scenario, items in sorted(by_scenario.items()):
        checks: dict[str, object] = {}
        check_names = sorted(
            {
                key[len("check_") : -len("_passed")]
                for row in items
                for key in row
                if key.startswith("check_") and key.endswith("_passed")
            }
        )
        for check_name in check_names:
            passed_values = [
                1.0 if _coerce_bool(row.get(f"check_{check_name}_passed")) else 0.0
                for row in items
            ]
            value_samples = [
                _coerce_float(row.get(f"check_{check_name}_value"))
                for row in items
                if row.get(f"check_{check_name}_value") not in {"", None}
            ]
            expected = next(
                (
                    str(row.get(f"check_{check_name}_expected") or "")
                    for row in items
                    if row.get(f"check_{check_name}_expected")
                ),
                "",
            )
            checks[check_name] = {
                "description": "Recovered from behavior_csv.",
                "expected": expected,
                "pass_rate": _mean(passed_values),
                "mean_value": _mean(value_samples) if value_samples else _mean(passed_values),
            }
        successes = [1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items]
        failures = sorted(
            {
                failure.strip()
                for row in items
                for failure in str(row.get("failures") or "").split(";")
                if failure.strip()
            }
        )
        suite[scenario] = {
            "scenario": scenario,
            "description": str(items[0].get("scenario_description") or ""),
            "objective": str(items[0].get("scenario_objective") or ""),
            "episodes": len(items),
            "success_rate": _mean(successes),
            "checks": checks,
            "behavior_metrics": {},
            "failures": failures,
            "legacy_metrics": {},
        }
    return suite


def extract_scenario_success(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Produce normalized per-scenario success data using the evaluation summary when available, otherwise reconstruct from behavior CSV rows.
    
    Parameters:
        summary (Mapping[str, object]): Parsed summary JSON (may be empty or missing `behavior_evaluation.suite`).
        behavior_rows (Sequence[Mapping[str, object]]): Rows from the behavior CSV used to reconstruct suite data if needed.
    
    Returns:
        dict[str, object]: A mapping with keys:
            - available (bool): `True` if any scenario entries were produced.
            - source (str): Either `"summary.behavior_evaluation.suite"` or `"behavior_csv"` indicating the data source.
            - scenarios (list[dict[str, object]]): List of per-scenario mappings, each containing:
                - scenario (str): Scenario identifier.
                - description (str): Human-readable description (empty string if missing).
                - objective (str): Scenario objective (empty string if missing).
                - success_rate (float): Success rate for the scenario (0.0 if missing/invalid).
                - episodes (int): Number of episodes (0 if missing/invalid).
                - failures (list): List of failure identifiers (empty list if none).
                - checks (dict): Mapping of check names to check payloads (empty dict if none).
                - legacy_metrics (dict): Legacy per-scenario metrics if present (empty dict if none).
            - limitations (list[str]): Explanatory messages about fallbacks or missing data.
    """
    suite = _suite_from_summary(summary)
    source = "summary.behavior_evaluation.suite"
    limitations: list[str] = []
    if not suite:
        suite = _scenario_suite_from_rows(behavior_rows)
        source = "behavior_csv"
        limitations.append(
            "Scenario checks were reconstructed from behavior_csv because summary.behavior_evaluation.suite was absent."
        )
    scenarios: list[dict[str, object]] = []
    for scenario, payload in sorted(suite.items()):
        if not isinstance(payload, Mapping):
            continue
        scenarios.append(
            {
                "scenario": scenario,
                "description": str(payload.get("description") or ""),
                "objective": str(payload.get("objective") or ""),
                "success_rate": _coerce_float(payload.get("success_rate")),
                "episodes": int(_coerce_float(payload.get("episodes"), 0.0)),
                "failures": list(payload.get("failures", []))
                if isinstance(payload.get("failures"), list)
                else [],
                "checks": dict(payload.get("checks", {}))
                if isinstance(payload.get("checks"), Mapping)
                else {},
                "legacy_metrics": dict(payload.get("legacy_metrics", {}))
                if isinstance(payload.get("legacy_metrics"), Mapping)
                else {},
            }
        )
    available = bool(scenarios)
    if not available:
        limitations.append("No scenario-level success data was available.")
    return {
        "available": available,
        "source": source,
        "scenarios": scenarios,
        "limitations": limitations,
    }


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
    Convert a reward-component mapping into normalized rows describing each component's disposition.
    
    Parameters:
        reward_components (Mapping[str, object]): Mapping from component name to a metadata mapping. Entries whose metadata is not a mapping are ignored.
    
    Returns:
        list[dict[str, object]]: A list of rows where each row contains the keys:
            - `component`: component name as string
            - `category`: component category or empty string
            - `risk`: shaping risk or empty string
            - `disposition`: shaping disposition or empty string
            - `rationale`: disposition rationale or empty string
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
    Selects one or more rows per ablation variant that have the minimal `eval_reflex_scale`.
    
    Parameters:
        rows (Sequence[Mapping[str, object]]): Iterable of behavior-CSV row mappings. Each row is expected to include an `ablation_variant` key and may include an `eval_reflex_scale` value.
    
    Returns:
        list[Mapping[str, object]]: A list containing, for each distinct `ablation_variant`, all rows whose `eval_reflex_scale` equals the minimal scale observed for that variant. Missing or non-finite `eval_reflex_scale` values are treated as `0.0`. If no rows are selected, returns the original list of input rows.
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
            `without_reflex_support` mapping with alternative `summary`, `suite`, or
            `legacy_scenarios` entries.
    
    Returns:
        dict[str, object]: A shallow copy of `payload` modified so that `eval_reflex_scale`
        is set to `0.0`, `primary_evaluation` is `"without_reflex_support"`, and when
        `without_reflex_support` provides `summary`, `suite`, or `legacy_scenarios` those
        entries are promoted into the returned mapping.
    """
    result = dict(payload)
    without_reflex = payload.get("without_reflex_support")
    if not isinstance(without_reflex, Mapping):
        return result
    for key in ("summary", "suite", "legacy_scenarios"):
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


def build_primary_benchmark(
    summary: Mapping[str, object],
    scenario_success: Mapping[str, object],
    ablations: Mapping[str, object],
) -> dict[str, object]:
    """
    Selects a primary "no-reflex" scenario success benchmark from available ablation, summary, or scenario data.
    
    Parameters:
    	summary (Mapping[str, object]): Parsed report summary (may contain evaluation blocks and fallbacks).
    	scenario_success (Mapping[str, object]): Scenario-level success data and metadata, used as a final fallback.
    	ablations (Mapping[str, object]): Ablation variants and metadata; preferred source for a no-reflex benchmark.
    
    Returns:
    	dict[str, object]: A payload describing the chosen primary benchmark with these keys:
    		available (bool): True when a benchmark value was found.
    		primary (bool): Always True for this payload.
    		label (str): Human-friendly label for the metric (always "No-reflex scenario_success_rate").
    		metric (str): Metric name ("scenario_success_rate").
    		scenario_success_rate (float): The benchmark score (0.0 when unavailable).
    		eval_reflex_scale (float|None): The eval reflex scale associated with the benchmark, or None when unknown.
    		reference_variant (str): Reference ablation variant name when sourced from ablations, else empty string.
    		source (str): Dot-path or descriptor indicating where the benchmark was sourced.
    		interpretation (str): Brief guidance about how to interpret the metric.
    """
    metric_name = "scenario_success_rate"
    label = "No-reflex scenario_success_rate"
    variants = ablations.get("variants", {})
    reference_variant = str(ablations.get("reference_variant") or "")
    if isinstance(variants, Mapping) and variants:
        if not reference_variant and "modular_full" in variants:
            reference_variant = "modular_full"
        reference_payload = variants.get(reference_variant)
        if isinstance(reference_payload, Mapping):
            summary_payload = reference_payload.get("summary", {})
            if isinstance(summary_payload, Mapping) and metric_name in summary_payload:
                return {
                    "available": True,
                    "primary": True,
                    "label": label,
                    "metric": metric_name,
                    "scenario_success_rate": _coerce_float(summary_payload.get(metric_name)),
                    "eval_reflex_scale": _coerce_float(
                        summary_payload.get(
                            "eval_reflex_scale",
                            reference_payload.get("eval_reflex_scale"),
                        ),
                        0.0,
                    ),
                    "reference_variant": reference_variant,
                    "source": (
                        f"{ablations.get('source')}.variants.{reference_variant}.summary"
                    ),
                    "interpretation": "Higher is better; this benchmark is evaluated with reflex support disabled.",
                }

    no_reflex_eval = summary.get("evaluation_without_reflex_support", {})
    if isinstance(no_reflex_eval, Mapping):
        no_reflex_summary = no_reflex_eval.get("summary", {})
        if isinstance(no_reflex_summary, Mapping) and metric_name in no_reflex_summary:
            return {
                "available": True,
                "primary": True,
                "label": label,
                "metric": metric_name,
                "scenario_success_rate": _coerce_float(no_reflex_summary.get(metric_name)),
                "eval_reflex_scale": _coerce_float(
                    no_reflex_summary.get(
                        "eval_reflex_scale",
                        no_reflex_eval.get("eval_reflex_scale"),
                    ),
                    0.0,
                ),
                "reference_variant": "",
                "source": "summary.evaluation_without_reflex_support.summary",
                "interpretation": "Higher is better; this benchmark is evaluated with reflex support disabled.",
            }

    evaluation = summary.get("evaluation", {})
    if isinstance(evaluation, Mapping) and metric_name in evaluation:
        eval_reflex_scale_raw = evaluation.get("eval_reflex_scale")
        if eval_reflex_scale_raw is not None:
            eval_reflex_scale = _coerce_float(eval_reflex_scale_raw, math.nan)
            if math.isfinite(eval_reflex_scale) and abs(eval_reflex_scale) <= 1e-6:
                return {
                    "available": True,
                    "primary": True,
                    "label": label,
                    "metric": metric_name,
                    "scenario_success_rate": _coerce_float(evaluation.get(metric_name)),
                    "eval_reflex_scale": eval_reflex_scale,
                    "reference_variant": "",
                    "source": "summary.evaluation",
                    "interpretation": "Higher is better; this benchmark is evaluated with reflex support disabled when eval_reflex_scale is 0.0.",
                }

    scenarios = scenario_success.get("scenarios", [])
    if isinstance(scenarios, list) and scenarios:
        values = [
            _coerce_float(item.get("success_rate"))
            for item in scenarios
            if isinstance(item, Mapping)
        ]
        if values:
            return {
                "available": True,
                "primary": True,
                "label": label,
                "metric": metric_name,
                "scenario_success_rate": _mean(values),
                "eval_reflex_scale": None,
                "reference_variant": "",
                "source": str(scenario_success.get("source") or ""),
                "interpretation": "Higher is better; eval_reflex_scale was unavailable for this fallback scenario aggregate.",
            }

    return {
        "available": False,
        "primary": True,
        "label": label,
        "metric": metric_name,
        "scenario_success_rate": 0.0,
        "eval_reflex_scale": None,
        "reference_variant": "",
        "source": "none",
        "interpretation": "No no-reflex scenario_success_rate benchmark was available.",
    }


def build_reflex_dependence_indicators(
    summary: Mapping[str, object],
    reflex_frequency: Mapping[str, object],
) -> dict[str, object]:
    """
    Compute reflex override and dominance indicators and their warning statuses using data from an evaluation summary and optional trace-derived reflex frequency.
    
    Parameters:
        summary (Mapping[str, object]): Report summary that may contain
            `evaluation_with_reflex_support` (with a `summary` mapping) or
            `evaluation` entries that include `mean_final_reflex_override_rate`
            and/or `mean_reflex_dominance`.
        reflex_frequency (Mapping[str, object]): Trace-derived reflex frequency
            payload (typically from `extract_reflex_frequency`) that may include a
            `modules` list with per-module `override_rate` and `mean_dominance`.
    
    Returns:
        dict[str, object]: A mapping with keys:
          - `available` (bool): `True` if any reflex indicator was found.
          - `source` (str): Source summary/trace fields used (e.g. `"summary.evaluation"`, `"trace.reflex_frequency"`, or `"none"`).
          - `failure_indicators` (dict): Contains two indicator entries:
              - `override_rate`: mapping with `value` (float), `warning_threshold` (float), and `status` (`"warning"` or `"ok"`).
              - `dominance`: mapping with `value` (float), `warning_threshold` (float), and `status` (`"warning"` or `"ok"`).
    """
    source_parts: list[str] = []
    override_rate = 0.0
    dominance = 0.0
    available = False

    reflex_eval = summary.get("evaluation_with_reflex_support", {})
    reflex_summary = (
        reflex_eval.get("summary", {})
        if isinstance(reflex_eval, Mapping)
        else {}
    )
    if isinstance(reflex_summary, Mapping) and (
        "mean_final_reflex_override_rate" in reflex_summary
        or "mean_reflex_dominance" in reflex_summary
    ):
        override_rate = _coerce_float(
            reflex_summary.get("mean_final_reflex_override_rate"),
            0.0,
        )
        dominance = _coerce_float(reflex_summary.get("mean_reflex_dominance"), 0.0)
        available = True
        source_parts.append("summary.evaluation_with_reflex_support.summary")
    else:
        evaluation = summary.get("evaluation", {})
        if isinstance(evaluation, Mapping) and (
            "mean_final_reflex_override_rate" in evaluation
            or "mean_reflex_dominance" in evaluation
        ):
            override_rate = _coerce_float(
                evaluation.get("mean_final_reflex_override_rate"),
                0.0,
            )
            dominance = _coerce_float(evaluation.get("mean_reflex_dominance"), 0.0)
            available = True
            source_parts.append("summary.evaluation")

    modules = reflex_frequency.get("modules", [])
    trace_has_debug_reflexes = bool(reflex_frequency.get("uses_debug_reflexes"))
    if trace_has_debug_reflexes and isinstance(modules, list) and modules:
        module_override_rates = [
            _coerce_float(module.get("override_rate"))
            for module in modules
            if isinstance(module, Mapping)
        ]
        module_dominance = [
            _coerce_float(module.get("mean_dominance"))
            for module in modules
            if isinstance(module, Mapping)
        ]
        if module_override_rates:
            override_rate = max(override_rate, max(module_override_rates))
            available = True
        if module_dominance:
            dominance = max(dominance, max(module_dominance))
            available = True
        source_parts.append("trace.reflex_frequency")

    override_status = (
        "warning"
        if override_rate >= REFLEX_OVERRIDE_WARNING_THRESHOLD
        else "ok"
    )
    dominance_status = (
        "warning"
        if dominance >= REFLEX_DOMINANCE_WARNING_THRESHOLD
        else "ok"
    )
    return {
        "available": available,
        "source": "+".join(source_parts) if source_parts else "none",
        "failure_indicators": {
            "override_rate": {
                "value": override_rate,
                "warning_threshold": REFLEX_OVERRIDE_WARNING_THRESHOLD,
                "status": override_status,
            },
            "dominance": {
                "value": dominance,
                "warning_threshold": REFLEX_DOMINANCE_WARNING_THRESHOLD,
                "status": dominance_status,
            },
        },
    }


def _format_failure_indicator(
    value: float,
    threshold: float,
    status: str,
) -> str:
    """
    Format a numeric failure indicator into a concise human-readable string.
    
    Parameters:
        value (float): Numeric indicator value to format.
        threshold (float): Threshold used in the warning message.
        status (str): Status label (e.g., "ok" or "warning").
    
    Returns:
        str: Formatted string like "0.123456 (warning; warning >= 0.10)" where `value` is shown with six decimal places and `threshold` with two decimal places.
    """
    return f"{value:.6f} ({status}; warning >= {threshold:.2f})"


def _fallback_group_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    key_name: str,
) -> dict[str, object]:
    """
    Aggregate per-group success statistics and counts from behavior CSV rows.
    
    Groups input rows by the string value of the given key name (skipping rows with no key or empty string). For each group computes:
    - `scenario_success_rate`: mean of per-row `success` (coerced to boolean then 1.0/0.0)
    - `episode_success_rate`: same mean computed over episodes
    - `scenario_count`: number of distinct non-empty `scenario` values in the group
    - `episode_count`: total number of rows in the group
    
    Parameters:
        rows (Sequence[Mapping[str, object]]): Iterable of row mappings (e.g., csv.DictReader rows).
        key_name (str): Column/key name to group rows by; its string value is used as the group key.
    
    Returns:
        dict[str, object]: Mapping from each group key to a dict containing a `summary` mapping with
        `scenario_success_rate`, `episode_success_rate`, `scenario_count`, and `episode_count`.
    """
    by_key: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(key_name) or "")
        if key:
            by_key[key].append(row)
    summary: dict[str, object] = {}
    for key, items in sorted(by_key.items()):
        success_values = [1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items]
        scenario_names = sorted({str(row.get("scenario") or "") for row in items if row.get("scenario")})
        summary[key] = {
            "summary": {
                "scenario_success_rate": _mean(success_values),
                "episode_success_rate": _mean(success_values),
                "scenario_count": len(scenario_names),
                "episode_count": len(items),
            }
        }
    return summary


def _ordered_noise_conditions(names: Iterable[str]) -> list[str]:
    """
    Return a stable ordering of noise condition names prioritizing canonical conditions.
    
    Parameters:
        names (Iterable[str]): Iterable of noise condition names; empty or falsy names are ignored.
    
    Returns:
        list[str]: Unique names present in `names`, with canonical conditions (as listed in CANONICAL_NOISE_CONDITIONS)
        kept in their canonical order first, followed by any remaining names sorted alphabetically.
    """
    unique_names = {str(name) for name in names if str(name)}
    ordered = [
        name for name in CANONICAL_NOISE_CONDITIONS if name in unique_names
    ]
    extras = sorted(name for name in unique_names if name not in CANONICAL_NOISE_CONDITIONS)
    return [*ordered, *extras]


def _canonicalize_inferred_noise_conditions(names: Iterable[str]) -> list[str]:
    """
    Expand inferred canonical noise axes to the full protocol ordering.

    When summary metadata omits explicit train/eval axes, treat subsets of the
    canonical noise names as incomplete protocol metadata and use the full
    canonical condition list. Non-canonical names are preserved as observed.
    """
    ordered = _ordered_noise_conditions(names)
    if not ordered:
        return list(CANONICAL_NOISE_CONDITIONS)
    if all(name in CANONICAL_NOISE_CONDITIONS for name in ordered):
        return list(CANONICAL_NOISE_CONDITIONS)
    return ordered


def _noise_robustness_cell_summary(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """
    Produce a normalized summary of a single robustness-matrix cell.
    
    Parameters:
        payload (Mapping[str, object]): Cell payload that may contain a `summary` mapping
            with numeric metrics and/or a `legacy_scenarios` mapping of per-scenario entries.
    
    Returns:
        dict[str, object]: A mapping with keys:
            - `scenario_success_rate`: float, coerced scenario-level success rate (0.0 if missing).
            - `episode_success_rate`: float, coerced episode-level success rate (0.0 if missing).
            - `mean_reward`: float, coerced mean reward for the cell; if `summary.mean_reward`
              is absent, the mean of `legacy_scenarios[*].mean_reward` is used (0.0 if none).
            - `scenario_count`: int, coerced scenario count (falls back to the number of
              `legacy_scenarios` entries when missing).
            - `episode_count`: int, coerced episode count (0 if missing).
    """
    summary = _mapping_or_empty(payload.get("summary"))
    legacy = _mapping_or_empty(payload.get("legacy_scenarios"))
    mean_reward_value = summary.get("mean_reward")
    mean_reward: float | None = (
        _coerce_float(mean_reward_value)
        if mean_reward_value is not None
        else None
    )
    if mean_reward is None:
        legacy_rewards = [
            _coerce_float(item.get("mean_reward"))
            for item in legacy.values()
            if isinstance(item, Mapping)
        ]
        mean_reward = _mean(legacy_rewards) if legacy_rewards else 0.0
    return {
        "scenario_success_rate": _coerce_float(summary.get("scenario_success_rate")),
        "episode_success_rate": _coerce_float(summary.get("episode_success_rate")),
        "mean_reward": _coerce_float(mean_reward),
        "scenario_count": int(_coerce_float(summary.get("scenario_count"), len(legacy))),
        "episode_count": int(_coerce_float(summary.get("episode_count"), 0.0)),
    }


def _normalize_noise_marginals(
    payload: object,
    *,
    conditions: Sequence[str],
    fallback: Mapping[str, object],
) -> dict[str, float]:
    """
    Builds a mapping of noise condition names to finite float marginals using values from `payload` with `fallback` as a secondary source.
    
    Parameters:
        payload (object): Mapping-like source of per-condition marginal values; non-mapping inputs are treated as empty.
        conditions (Sequence[str]): Ordered list of condition names to include in the result.
        fallback (Mapping[str, object]): Secondary mapping used when a condition is missing or not convertible in `payload`. Values from `fallback` are also coerced to finite floats.
    
    Returns:
        dict[str, float]: A dict mapping each condition name from `conditions` to a finite float. For each condition, the function uses the coerced value from `payload` when available; otherwise it uses the coerced value from `fallback`. If neither yields a finite number, the value defaults to 0.0.
    """
    raw = _mapping_or_empty(payload)
    return {
        condition: _coerce_float(raw.get(condition), _coerce_float(fallback.get(condition)))
        for condition in conditions
    }


def _noise_robustness_metrics(
    matrix: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    train_conditions: Sequence[str],
    eval_conditions: Sequence[str],
) -> dict[str, object]:
    """
    Compute marginal means and aggregate robustness scores for a train-by-eval robustness matrix.
    
    Parameters:
        matrix (Mapping[str, Mapping[str, Mapping[str, object]]]): Nested mapping keyed by train condition then eval condition to a cell dictionary containing at least `scenario_success_rate`.
        train_conditions (Sequence[str]): Ordered list of train condition names to include (rows).
        eval_conditions (Sequence[str]): Ordered list of eval condition names to include (columns).
    
    Returns:
        dict[str, object]: A summary containing:
            - `train_marginals` (dict[str, float]): Mean `scenario_success_rate` per train condition across available eval columns.
            - `eval_marginals` (dict[str, float]): Mean `scenario_success_rate` per eval condition across available train rows.
            - `robustness_score` (float): Mean of all available cell `scenario_success_rate` values.
            - `diagonal_score` (float): Mean of cell scores where train == eval.
            - `off_diagonal_score` (float): Mean of cell scores where train != eval.
            - `available_cell_count` (int): Number of matrix cells that contributed to the computed scores.
    """
    train_marginals: dict[str, float] = {}
    eval_marginals: dict[str, float] = {}
    all_scores: list[float] = []
    diagonal_scores: list[float] = []
    off_diagonal_scores: list[float] = []

    for train_condition in train_conditions:
        train_scores = [
            _coerce_float(
                _mapping_or_empty(matrix.get(train_condition, {}))
                .get(eval_condition, {})
                .get("scenario_success_rate")
            )
            for eval_condition in eval_conditions
            if eval_condition in _mapping_or_empty(matrix.get(train_condition, {}))
        ]
        train_marginals[train_condition] = _mean(train_scores)

    for eval_condition in eval_conditions:
        eval_scores = [
            _coerce_float(
                _mapping_or_empty(matrix.get(train_condition, {}))
                .get(eval_condition, {})
                .get("scenario_success_rate")
            )
            for train_condition in train_conditions
            if eval_condition in _mapping_or_empty(matrix.get(train_condition, {}))
        ]
        eval_marginals[eval_condition] = _mean(eval_scores)

    for train_condition in train_conditions:
        for eval_condition in eval_conditions:
            cell = _mapping_or_empty(matrix.get(train_condition, {})).get(eval_condition)
            if not isinstance(cell, Mapping):
                continue
            score = _coerce_float(cell.get("scenario_success_rate"))
            all_scores.append(score)
            if train_condition == eval_condition:
                diagonal_scores.append(score)
            else:
                off_diagonal_scores.append(score)

    return {
        "train_marginals": train_marginals,
        "eval_marginals": eval_marginals,
        "robustness_score": _mean(all_scores),
        "diagonal_score": _mean(diagonal_scores),
        "off_diagonal_score": _mean(off_diagonal_scores),
        "available_cell_count": len(all_scores),
    }


def _noise_robustness_rate(
    matrix: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    train_condition: str,
    eval_condition: str,
) -> float | None:
    cell = _mapping_or_empty(matrix.get(train_condition, {})).get(eval_condition)
    if not isinstance(cell, Mapping):
        return None
    return _coerce_optional_float(cell.get("scenario_success_rate"))


def _noise_robustness_marginal(
    marginals: Mapping[str, object],
    condition: str,
) -> float | None:
    if condition not in marginals:
        return None
    return _coerce_optional_float(marginals.get(condition))


def _noise_robustness_mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return _mean(present)


def _observed_noise_robustness_metrics(
    matrix: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    train_conditions: Sequence[str],
    eval_conditions: Sequence[str],
) -> dict[str, object]:
    train_marginals: dict[str, float] = {}
    eval_marginals: dict[str, float] = {}
    all_scores: list[float] = []
    diagonal_scores: list[float] = []
    off_diagonal_scores: list[float] = []

    for train_condition in train_conditions:
        train_score = _noise_robustness_mean(
            _noise_robustness_rate(
                matrix,
                train_condition=train_condition,
                eval_condition=eval_condition,
            )
            for eval_condition in eval_conditions
        )
        if train_score is not None:
            train_marginals[train_condition] = train_score

    for eval_condition in eval_conditions:
        eval_score = _noise_robustness_mean(
            _noise_robustness_rate(
                matrix,
                train_condition=train_condition,
                eval_condition=eval_condition,
            )
            for train_condition in train_conditions
        )
        if eval_score is not None:
            eval_marginals[eval_condition] = eval_score

    for train_condition in train_conditions:
        for eval_condition in eval_conditions:
            score = _noise_robustness_rate(
                matrix,
                train_condition=train_condition,
                eval_condition=eval_condition,
            )
            if score is None:
                continue
            all_scores.append(score)
            if train_condition == eval_condition:
                diagonal_scores.append(score)
            else:
                off_diagonal_scores.append(score)

    return {
        "train_marginals": train_marginals,
        "eval_marginals": eval_marginals,
        "robustness_score": _noise_robustness_mean(all_scores),
        "diagonal_score": _noise_robustness_mean(diagonal_scores),
        "off_diagonal_score": _noise_robustness_mean(off_diagonal_scores),
        "available_cell_count": len(all_scores),
    }


def extract_noise_robustness(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Compute a train-by-eval noise robustness matrix and summary metrics from available summary payload or behavior CSV rows.
    
    Parameters:
        summary (Mapping[str, object]): Parsed analysis summary which may contain a precomputed
            `behavior_evaluation.robustness_matrix` payload.
        behavior_rows (Sequence[Mapping[str, object]]): Normalized behavior CSV rows; used to
            reconstruct the robustness matrix when the summary payload is absent.
    
    Returns:
        dict[str, object]: A structured robustness report containing:
            - available (bool): Whether any matrix cells are present.
            - source (str): Either "summary.behavior_evaluation.robustness_matrix", "behavior_csv", or "none".
            - matrix_spec (dict): Keys `train_conditions`, `eval_conditions`, and `cell_count`.
            - matrix (dict): Mapping train_condition -> eval_condition -> cell summary with keys
              `scenario_success_rate`, `episode_success_rate`, `mean_reward`, `scenario_count`, `episode_count`.
            - train_marginals / eval_marginals (dict): Marginal scores per condition.
            - robustness_score, diagonal_score, off_diagonal_score (float): Aggregate metrics.
            - metadata (dict): `complete` (bool), `available_cell_count`, and `expected_cell_count`.
            - limitations (list[str]): Human-readable notes when payloads or cells were missing or reconstructed.
    """
    behavior_evaluation = summary.get("behavior_evaluation", {})
    robustness_payload = (
        behavior_evaluation.get("robustness_matrix", {})
        if isinstance(behavior_evaluation, Mapping)
        else {}
    )
    limitations: list[str] = []
    expected_train_conditions: list[str] = []
    expected_eval_conditions: list[str] = []
    summary_available_cell_count = 0
    summary_was_incomplete = False
    summary_incomplete_limitation: str | None = None
    partial_summary_matrix: dict[str, dict[str, dict[str, object]]] = {}
    partial_summary_train_marginals: dict[str, float] = {}
    partial_summary_eval_marginals: dict[str, float] = {}

    if isinstance(robustness_payload, Mapping) and robustness_payload:
        raw_matrix = _mapping_or_empty(robustness_payload.get("matrix"))
        matrix_spec = _mapping_or_empty(
            robustness_payload.get("matrix_spec")
            or robustness_payload.get("robustness_matrix")
        )
        _tc_raw = matrix_spec.get("train_conditions")
        _ec_raw = matrix_spec.get("eval_conditions")
        train_conditions = list(_tc_raw) if isinstance(_tc_raw, list) else []
        eval_conditions = list(_ec_raw) if isinstance(_ec_raw, list) else []
        if not train_conditions:
            train_conditions = _canonicalize_inferred_noise_conditions(
                raw_matrix.keys()
            )
        if not eval_conditions:
            eval_names = {
                str(eval_condition)
                for row in raw_matrix.values()
                if isinstance(row, Mapping)
                for eval_condition in row.keys()
            }
            eval_conditions = _canonicalize_inferred_noise_conditions(eval_names)
        expected_train_conditions = list(train_conditions)
        expected_eval_conditions = list(eval_conditions)

        matrix: dict[str, dict[str, dict[str, object]]] = {}
        for train_condition in train_conditions:
            raw_row = _mapping_or_empty(raw_matrix.get(train_condition))
            matrix[train_condition] = {}
            for eval_condition in eval_conditions:
                raw_cell = _mapping_or_empty(raw_row.get(eval_condition))
                if raw_cell:
                    matrix[train_condition][eval_condition] = (
                        _noise_robustness_cell_summary(raw_cell)
                    )

        metrics = _noise_robustness_metrics(
            matrix,
            train_conditions=train_conditions,
            eval_conditions=eval_conditions,
        )
        train_marginals = _normalize_noise_marginals(
            robustness_payload.get("train_marginals"),
            conditions=train_conditions,
            fallback=metrics["train_marginals"],
        )
        eval_marginals = _normalize_noise_marginals(
            robustness_payload.get("eval_marginals"),
            conditions=eval_conditions,
            fallback=metrics["eval_marginals"],
        )
        expected_cell_count = len(train_conditions) * len(eval_conditions)
        if expected_cell_count > 0 and metrics["available_cell_count"] == expected_cell_count:
            return {
                "available": True,
                "source": "summary.behavior_evaluation.robustness_matrix",
                "matrix_spec": {
                    "train_conditions": train_conditions,
                    "eval_conditions": eval_conditions,
                    "cell_count": expected_cell_count,
                },
                "matrix": matrix,
                "train_marginals": train_marginals,
                "eval_marginals": eval_marginals,
                "robustness_score": _coerce_float(
                    robustness_payload.get("robustness_score"),
                    metrics["robustness_score"],
                ),
                "diagonal_score": _coerce_float(
                    robustness_payload.get("diagonal_score"),
                    metrics["diagonal_score"],
                ),
                "off_diagonal_score": _coerce_float(
                    robustness_payload.get("off_diagonal_score"),
                    metrics["off_diagonal_score"],
                ),
                "metadata": {
                    "complete": True,
                    "available_cell_count": metrics["available_cell_count"],
                    "expected_cell_count": expected_cell_count,
                },
                "limitations": limitations,
            }
        if expected_cell_count == 0:
            summary_incomplete_limitation = (
                "Noise robustness matrix was present in the summary but did not define any train/eval conditions."
            )
        else:
            summary_incomplete_limitation = (
                "Noise robustness matrix was present in the summary but some train/eval cells were missing."
            )
        observed_metrics = _observed_noise_robustness_metrics(
            matrix,
            train_conditions=train_conditions,
            eval_conditions=eval_conditions,
        )
        summary_available_cell_count = metrics["available_cell_count"]
        summary_was_incomplete = True
        partial_summary_matrix = matrix
        partial_summary_train_marginals = observed_metrics["train_marginals"]
        partial_summary_eval_marginals = observed_metrics["eval_marginals"]

    by_cell: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    missing_noise_metadata = 0
    for row in behavior_rows:
        train_condition = str(row.get("train_noise_profile") or "")
        eval_condition = str(row.get("eval_noise_profile") or "")
        if not train_condition or not eval_condition:
            if row.get("noise_profile") or row.get("train_noise_profile") or row.get("eval_noise_profile"):
                missing_noise_metadata += 1
            continue
        by_cell[(train_condition, eval_condition)].append(row)

    if not by_cell:
        if summary_incomplete_limitation is not None:
            limitations.append(summary_incomplete_limitation)
        if summary_was_incomplete:
            limitations.append(
                "Behavior CSV did not provide enough train/eval noise metadata to reconstruct the missing robustness cells."
            )
        train_conditions = list(expected_train_conditions)
        eval_conditions = list(expected_eval_conditions)
        expected_cell_count = len(train_conditions) * len(eval_conditions)
        return {
            "available": False,
            "source": (
                "summary.behavior_evaluation.robustness_matrix"
                if summary_was_incomplete
                else "none"
            ),
            "matrix_spec": {
                "train_conditions": train_conditions,
                "eval_conditions": eval_conditions,
                "cell_count": expected_cell_count,
            },
            "matrix": partial_summary_matrix,
            "train_marginals": partial_summary_train_marginals,
            "eval_marginals": partial_summary_eval_marginals,
            "robustness_score": None,
            "diagonal_score": None,
            "off_diagonal_score": None,
            "metadata": {
                "complete": False,
                "available_cell_count": summary_available_cell_count,
                "expected_cell_count": expected_cell_count,
            },
            "limitations": limitations
            or [
                "No noise robustness matrix payload or train/eval noise rows were available."
            ],
        }

    derived_train_conditions = _ordered_noise_conditions(
        train_condition for train_condition, _ in by_cell
    )
    derived_eval_conditions = _ordered_noise_conditions(
        eval_condition for _, eval_condition in by_cell
    )
    train_conditions = [
        *expected_train_conditions,
        *[
            name
            for name in derived_train_conditions
            if name not in expected_train_conditions
        ],
    ] if expected_train_conditions else derived_train_conditions
    eval_conditions = [
        *expected_eval_conditions,
        *[
            name
            for name in derived_eval_conditions
            if name not in expected_eval_conditions
        ],
    ] if expected_eval_conditions else derived_eval_conditions
    matrix: dict[str, dict[str, dict[str, object]]] = {
        train_condition: {}
        for train_condition in train_conditions
    }
    for train_condition in train_conditions:
        partial_row = _mapping_or_empty(partial_summary_matrix.get(train_condition))
        for eval_condition in eval_conditions:
            partial_cell = _mapping_or_empty(partial_row.get(eval_condition))
            if partial_cell:
                matrix[train_condition][eval_condition] = dict(partial_cell)
    for (train_condition, eval_condition), items in sorted(by_cell.items()):
        success_values = [
            1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items
        ]
        scenario_names = {
            str(row.get("scenario") or "")
            for row in items
            if row.get("scenario")
        }
        matrix[train_condition][eval_condition] = {
            "scenario_success_rate": _mean(success_values),
            "episode_success_rate": _mean(success_values),
            "mean_reward": 0.0,
            "scenario_count": len(scenario_names),
            "episode_count": len(items),
        }

    metrics = _noise_robustness_metrics(
        matrix,
        train_conditions=train_conditions,
        eval_conditions=eval_conditions,
    )
    observed_metrics = _observed_noise_robustness_metrics(
        matrix,
        train_conditions=train_conditions,
        eval_conditions=eval_conditions,
    )
    expected_cell_count = len(train_conditions) * len(eval_conditions)
    if metrics["available_cell_count"] < expected_cell_count:
        if summary_incomplete_limitation is not None:
            limitations.append(summary_incomplete_limitation)
        limitations.append(
            "Noise robustness matrix was reconstructed from behavior_csv and some train/eval cells were missing."
        )
    else:
        limitations.append(
            (
                "Noise robustness matrix was reconstructed from behavior_csv because summary.behavior_evaluation.robustness_matrix was incomplete."
                if summary_was_incomplete
                else "Noise robustness matrix was reconstructed from behavior_csv because summary.behavior_evaluation.robustness_matrix was absent."
            )
        )
    if missing_noise_metadata:
        limitations.append(
            "Some behavior_csv rows were ignored because train_noise_profile or eval_noise_profile metadata was missing."
        )
    complete = bool(expected_cell_count) and metrics["available_cell_count"] == expected_cell_count
    return {
        "available": complete,
        "source": "behavior_csv",
        "matrix_spec": {
            "train_conditions": train_conditions,
            "eval_conditions": eval_conditions,
            "cell_count": expected_cell_count,
        },
        "matrix": matrix,
        "train_marginals": (
            metrics["train_marginals"]
            if complete
            else observed_metrics["train_marginals"]
        ),
        "eval_marginals": (
            metrics["eval_marginals"]
            if complete
            else observed_metrics["eval_marginals"]
        ),
        "robustness_score": metrics["robustness_score"] if complete else None,
        "diagonal_score": metrics["diagonal_score"] if complete else None,
        "off_diagonal_score": metrics["off_diagonal_score"] if complete else None,
        "metadata": {
            "complete": complete,
            "available_cell_count": metrics["available_cell_count"],
            "expected_cell_count": expected_cell_count,
        },
        "limitations": limitations,
    }


def extract_comparisons(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Produce reward-profile and map-template comparison summaries, preferring a precomputed comparisons payload in the provided summary and falling back to reconstruction from behavior CSV rows.
    
    When the summary contains a `behavior_evaluation.comparisons` mapping, that payload is returned (with `reward_profiles` and `map_templates` taken directly). Otherwise this function groups `behavior_rows` by `reward_profile` and `evaluation_map` to reconstruct comparable summaries and records a limitation note.
    
    Returns:
        dict: A mapping with the following keys:
            - available (bool): True when any comparison data is present.
            - source (str): `"summary.behavior_evaluation.comparisons"` when taken from the summary, otherwise `"behavior_csv"`.
            - reward_profiles (dict): Mapping of reward-profile identifiers to group summary objects (empty if none).
            - map_templates (dict): Mapping of map/template identifiers to group summary objects (empty if none).
            - limitations (list[str]): Human-readable notes describing reconstruction or missing-data limitations.
    """
    behavior_evaluation = summary.get("behavior_evaluation", {})
    comparisons = (
        behavior_evaluation.get("comparisons", {})
        if isinstance(behavior_evaluation, Mapping)
        else {}
    )
    limitations: list[str] = []
    if isinstance(comparisons, Mapping) and comparisons:
        reward_profiles = comparisons.get("reward_profiles", {})
        map_templates = comparisons.get("map_templates", {})
        return {
            "available": bool(reward_profiles or map_templates),
            "source": "summary.behavior_evaluation.comparisons",
            "reward_profiles": dict(reward_profiles) if isinstance(reward_profiles, Mapping) else {},
            "map_templates": dict(map_templates) if isinstance(map_templates, Mapping) else {},
            "limitations": [],
        }

    reward_profiles = _fallback_group_summary(behavior_rows, key_name="reward_profile")
    map_templates = _fallback_group_summary(behavior_rows, key_name="evaluation_map")
    if reward_profiles or map_templates:
        limitations.append(
            "Profile/map comparisons were reconstructed from behavior_csv and may not match the compact comparison payload exactly."
        )
    else:
        limitations.append("No profile or map comparison payload was available.")
    return {
        "available": bool(reward_profiles or map_templates),
        "source": "behavior_csv",
        "reward_profiles": reward_profiles,
        "map_templates": map_templates,
        "limitations": limitations,
    }


def _normalize_module_response_by_predator_type(
    value: object,
) -> dict[str, dict[str, float]]:
    """
    Normalize a nested mapping of predator-type to module responses into floats.
    
    Converts an input mapping of the form {predator_type: {module_name: value, ...}, ...}
    into a dictionary with stringified predator type and module keys and float-coerced
    values. Non-mapping inputs or non-mapping predator entries are ignored.
    
    Parameters:
        value (object): Expected to be a mapping from predator type to a mapping of
            module names to numeric-like values. Other shapes are tolerated and will
            be skipped.
    
    Returns:
        dict[str, dict[str, float]]: A mapping where each predator type (string) maps
        to a mapping of module name (string) to a finite float value (defaults to
        0.0 when coercion fails).
    """
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, dict[str, float]] = {}
    for predator_type, payload in value.items():
        if not isinstance(payload, Mapping):
            continue
        normalized[str(predator_type)] = {
            str(module_name): _coerce_float(module_value)
            for module_name, module_value in payload.items()
        }
    return normalized


def _aggregate_specialization_from_scenarios(
    scenarios: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    """
    Aggregate module response values across scenarios by predator type and compute per-module means.
    
    Parameters:
        scenarios (Mapping[str, object]): Mapping of scenario identifiers to payloads. Each payload may include
            `mean_module_response_by_predator_type` or `module_response_by_predator_type`, a mapping from
            predator type to module-to-value mappings. Non-mapping payloads are ignored.
    
    Returns:
        dict[str, dict[str, float]]: Mapping from predator type to a mapping of module name to the arithmetic
        mean of that module's response values across all scenarios (0.0 for modules present but with no finite values).
    """
    per_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for payload in scenarios.values():
        if not isinstance(payload, Mapping):
            continue
        response = _normalize_module_response_by_predator_type(
            payload.get("mean_module_response_by_predator_type")
            or payload.get("module_response_by_predator_type")
        )
        for predator_type, module_values in response.items():
            for module_name, module_value in module_values.items():
                per_type[predator_type][module_name].append(_coerce_float(module_value))
    return {
        predator_type: {
            module_name: _mean(values)
            for module_name, values in module_values.items()
        }
        for predator_type, module_values in per_type.items()
    }


def extract_predator_type_specialization(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
    trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Compute predator-type module specialization and related metrics from per-type module response data.
    
    When available, selects the first source that provides per-predator-type module response values and computes:
    - per-predator-type module distributions and dominant module,
    - differential activation for visual cortex (visual minus olfactory) and sensory cortex (olfactory minus visual),
    - a bounded specialization score in [0.0, 1.0],
    - a type-module correlation proxy and an interpretation bucket.
    
    Returns:
        dict: Analysis payload with keys:
            - available (bool): True when usable per-type metrics were found.
            - source (str): The chosen data source name or "none".
            - predator_types (dict): Per-type entries for "visual" and "olfactory", each containing:
                - module_distribution (dict[str, float]): Raw module response values.
                - visual_cortex_activation (float)
                - sensory_cortex_activation (float)
                - dominant_module (str)
            - differential_activation (dict): {
                "visual_cortex_visual_minus_olfactory": float,
                "sensory_cortex_olfactory_minus_visual": float
              }
            - type_module_correlation (float): Correlation proxy in [0.0, 1.0].
            - specialization_score (float): Bounded score in [0.0, 1.0].
            - interpretation (str): One of "high", "moderate", "low", "insufficient_data", or "unavailable".
            - limitations (list[str]): Notes about missing or reconstructed data when unavailable.
    """
    del behavior_rows
    del trace

    candidate_sources: list[tuple[str, dict[str, dict[str, float]]]] = []
    evaluation = _mapping_or_empty(summary.get("evaluation"))
    candidate_sources.append(
        (
            "summary.evaluation",
            _normalize_module_response_by_predator_type(
                evaluation.get("mean_module_response_by_predator_type")
                or evaluation.get("module_response_by_predator_type")
            ),
        )
    )
    evaluation_without_reflex = _mapping_or_empty(
        _mapping_or_empty(summary.get("evaluation_without_reflex_support")).get("summary")
    )
    candidate_sources.append(
        (
            "summary.evaluation_without_reflex_support.summary",
            _normalize_module_response_by_predator_type(
                evaluation_without_reflex.get("mean_module_response_by_predator_type")
                or evaluation_without_reflex.get("module_response_by_predator_type")
            ),
        )
    )
    behavior_evaluation = _mapping_or_empty(summary.get("behavior_evaluation"))
    candidate_sources.append(
        (
            "summary.behavior_evaluation.suite",
            _aggregate_specialization_from_scenarios(
                _mapping_or_empty(behavior_evaluation.get("suite"))
            ),
        )
    )
    candidate_sources.append(
        (
            "summary.behavior_evaluation.legacy_scenarios",
            _aggregate_specialization_from_scenarios(
                _mapping_or_empty(behavior_evaluation.get("legacy_scenarios"))
            ),
        )
    )

    source_name = "none"
    module_response_by_predator_type: dict[str, dict[str, float]] = {}
    fallback_source_name = "none"
    fallback_candidate: dict[str, dict[str, float]] = {}
    for candidate_name, candidate in candidate_sources:
        has_any_data = any(sum(module_values.values()) > 0.0 for module_values in candidate.values())
        if not has_any_data:
            continue
        visual_modules = candidate.get("visual", {})
        olfactory_modules = candidate.get("olfactory", {})
        has_visual_data = any(_coerce_float(value) > 0.0 for value in visual_modules.values())
        has_olfactory_data = any(
            _coerce_float(value) > 0.0 for value in olfactory_modules.values()
        )
        if has_visual_data and has_olfactory_data:
            source_name = candidate_name
            module_response_by_predator_type = candidate
            break
        if not fallback_candidate:
            fallback_source_name = candidate_name
            fallback_candidate = candidate

    if not module_response_by_predator_type and fallback_candidate:
        source_name = fallback_source_name
        module_response_by_predator_type = fallback_candidate

    if not module_response_by_predator_type:
        return {
            "available": False,
            "source": "none",
            "predator_types": {},
            "differential_activation": {},
            "type_module_correlation": 0.0,
            "specialization_score": 0.0,
            "interpretation": "unavailable",
            "limitations": [
                "No per-predator-type module-response metrics were available.",
            ],
        }

    visual_modules = module_response_by_predator_type.get("visual", {})
    olfactory_modules = module_response_by_predator_type.get("olfactory", {})
    visual_visual = _coerce_float(visual_modules.get("visual_cortex"))
    visual_sensory = _coerce_float(visual_modules.get("sensory_cortex"))
    olfactory_visual = _coerce_float(olfactory_modules.get("visual_cortex"))
    olfactory_sensory = _coerce_float(olfactory_modules.get("sensory_cortex"))
    limitations: list[str] = []
    has_visual_data = any(
        _coerce_float(value) > 0.0 for value in visual_modules.values()
    )
    has_olfactory_data = any(
        _coerce_float(value) > 0.0 for value in olfactory_modules.values()
    )

    if has_visual_data and has_olfactory_data:
        visual_dominant = _dominant_module_by_score(visual_modules)
        olfactory_dominant = _dominant_module_by_score(olfactory_modules)
        visual_cortex_differential = round(visual_visual - olfactory_visual, 6)
        sensory_cortex_differential = round(olfactory_sensory - visual_sensory, 6)
        specialization_score = max(
            0.0,
            min(
                1.0,
                (
                    max(0.0, visual_cortex_differential)
                    + max(0.0, sensory_cortex_differential)
                ) / 2.0,
            ),
        )
        dominant_alignment = _mean(
            [
                1.0 if visual_dominant == "visual_cortex" else 0.0,
                1.0 if olfactory_dominant == "sensory_cortex" else 0.0,
            ]
        )
        type_module_correlation = round(
            max(0.0, min(1.0, (dominant_alignment + specialization_score) / 2.0)),
            6,
        )
        if specialization_score >= 0.50:
            interpretation = "high"
        elif specialization_score >= 0.20:
            interpretation = "moderate"
        else:
            interpretation = "low"
    else:
        visual_dominant = _dominant_module_by_score(visual_modules) if has_visual_data else ""
        olfactory_dominant = (
            _dominant_module_by_score(olfactory_modules) if has_olfactory_data else ""
        )
        visual_cortex_differential = 0.0
        sensory_cortex_differential = 0.0
        specialization_score = 0.0
        type_module_correlation = 0.0
        interpretation = "insufficient_data"
        limitations.append(
            "Predator type specialization requires non-zero module-response data for both visual and olfactory predators."
        )

    return {
        "available": True,
        "source": source_name,
        "predator_types": {
            "visual": {
                "module_distribution": dict(visual_modules),
                "visual_cortex_activation": visual_visual,
                "sensory_cortex_activation": visual_sensory,
                "dominant_module": visual_dominant,
            },
            "olfactory": {
                "module_distribution": dict(olfactory_modules),
                "visual_cortex_activation": olfactory_visual,
                "sensory_cortex_activation": olfactory_sensory,
                "dominant_module": olfactory_dominant,
            },
        },
        "differential_activation": {
            "visual_cortex_visual_minus_olfactory": visual_cortex_differential,
            "sensory_cortex_olfactory_minus_visual": sensory_cortex_differential,
        },
        "type_module_correlation": type_module_correlation,
        "specialization_score": round(specialization_score, 6),
        "interpretation": interpretation,
        "limitations": limitations,
    }


def extract_ablations(
    summary: Mapping[str, object],
    behavior_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Extract ablation variant payloads and compute deltas versus a reference variant.
    
    When ablation data exists in summary["behavior_evaluation"]["ablations"], this function normalizes each variant to minimal-reflex-support form and computes deltas versus a chosen reference variant. When summary ablations are absent, it reconstructs per-variant summaries from behavior_csv rows by grouping on `ablation_variant` and using the rows with minimal `eval_reflex_scale` per variant.
    
    Parameters:
        summary (Mapping[str, object]): Parsed report summary that may contain `behavior_evaluation.ablations`.
        behavior_rows (Sequence[Mapping[str, object]]): Normalized rows from the behavior CSV used as a fallback when summary ablations are not present.
    
    Returns:
        dict[str, object]: A mapping with the following keys:
            available (bool): True when any variant data was extracted or reconstructed.
            source (str): `"summary.behavior_evaluation.ablations"` when sourced from the summary, otherwise `"behavior_csv"`.
            reference_variant (str): The chosen reference variant name (e.g., `"modular_full"`) or empty string if none.
            variants (dict[str, object]): Mapping of variant names to normalized payloads. Each payload contains at least `config` and `summary` with `eval_reflex_scale` and success-rate metrics.
            deltas_vs_reference (dict[str, object]): Per-variant and per-scenario deltas computed against the selected reference variant.
            limitations (list[str]): Human-readable notes describing fallbacks, missing fields, or reconstruction caveats.
    """
    behavior_evaluation = summary.get("behavior_evaluation", {})
    ablations = (
        behavior_evaluation.get("ablations", {})
        if isinstance(behavior_evaluation, Mapping)
        else {}
    )
    limitations: list[str] = []
    if isinstance(ablations, Mapping) and ablations:
        variants_payload = ablations.get("variants", {})
        variants = (
            {
                str(variant): _variant_with_minimal_reflex_support(payload)
                for variant, payload in variants_payload.items()
                if isinstance(payload, Mapping)
            }
            if isinstance(variants_payload, Mapping)
            else {}
        )
        reference_variant = (
            "modular_full"
            if "modular_full" in variants
            else str(ablations.get("reference_variant") or "")
        )
        if reference_variant and reference_variant in variants:
            reference_payload = variants[reference_variant]
            if not isinstance(reference_payload.get("without_reflex_support"), Mapping):
                limitations.append(
                    f"Reference variant {reference_variant!r} did not include a without_reflex_support block; using the available summary."
                )
        deltas = _ablation_deltas_vs_reference(
            variants,
            reference_variant=reference_variant,
        )
        predator_type_comparisons = compare_predator_type_ablation_performance(
            {
                "variants": variants,
                "deltas_vs_reference": deltas,
            }
        )
        return {
            "available": bool(variants),
            "source": "summary.behavior_evaluation.ablations",
            "reference_variant": reference_variant,
            "variants": variants,
            "deltas_vs_reference": deltas,
            "predator_type_comparisons": predator_type_comparisons,
            "limitations": limitations,
        }

    filtered_rows = _rows_with_minimal_reflex_support(behavior_rows)
    by_variant: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in filtered_rows:
        variant = str(row.get("ablation_variant") or "")
        if variant:
            by_variant[variant].append(row)
    variants: dict[str, object] = {}
    for variant, items in sorted(by_variant.items()):
        success_values = [1.0 if _coerce_bool(row.get("success")) else 0.0 for row in items]
        by_scenario: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in items:
            by_scenario[str(row.get("scenario") or "")].append(row)
        suite = {
            scenario_name: {
                "scenario": scenario_name,
                "episodes": len(scenario_items),
                "success_rate": _mean(
                    1.0 if _coerce_bool(row.get("success")) else 0.0
                    for row in scenario_items
                ),
            }
            for scenario_name, scenario_items in sorted(by_scenario.items())
            if scenario_name
        }
        variants[variant] = {
            "config": {
                "architecture": str(items[0].get("ablation_architecture") or ""),
            },
            "summary": {
                "scenario_success_rate": _mean(success_values),
                "episode_success_rate": _mean(success_values),
                "eval_reflex_scale": min(
                    _coerce_float(row.get("eval_reflex_scale"), 0.0)
                    for row in items
                ),
            },
            "suite": suite,
            "legacy_scenarios": dict(suite),
        }
    if variants:
        reference_variant = "modular_full" if "modular_full" in variants else ""
        limitations.append(
            "Ablation comparisons were reconstructed from behavior_csv using the lowest eval_reflex_scale rows per variant."
        )
    else:
        reference_variant = ""
        limitations.append("No ablation comparison data was available.")
    deltas = _ablation_deltas_vs_reference(
        variants,
        reference_variant=reference_variant,
    )
    return {
        "available": bool(variants),
        "source": "behavior_csv",
        "reference_variant": reference_variant,
        "variants": variants,
        "deltas_vs_reference": deltas,
        "predator_type_comparisons": compare_predator_type_ablation_performance(
            {
                "variants": variants,
                "deltas_vs_reference": deltas,
            }
        ),
        "limitations": limitations,
    }


def extract_reflex_frequency(trace: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not trace:
        return {
            "available": False,
            "tick_count": 0,
            "modules": [],
            "uses_debug_reflexes": False,
            "limitations": ["No trace file was provided."],
        }

    message_counts: Counter[str] = Counter()
    debug_counts: Counter[str] = Counter()
    debug_override_counts: Counter[str] = Counter()
    debug_dominance_sum: defaultdict[str, float] = defaultdict(float)
    module_names: set[str] = set(DEFAULT_MODULE_NAMES)
    uses_debug_reflexes = False

    for item in trace:
        for message in item.get("messages", []) if isinstance(item.get("messages"), list) else []:
            if not isinstance(message, Mapping):
                continue
            payload = message.get("payload")
            sender = str(message.get("sender") or "")
            if sender:
                module_names.add(sender)
            if (
                isinstance(payload, Mapping)
                and payload.get("reflex")
                and sender
                and str(message.get("topic") or "") == "action.proposal"
            ):
                message_counts[sender] += 1
        debug = item.get("debug")
        reflexes = debug.get("reflexes") if isinstance(debug, Mapping) else None
        if isinstance(reflexes, Mapping):
            uses_debug_reflexes = True
            for module_name, payload in reflexes.items():
                module_name = str(module_name)
                module_names.add(module_name)
                if not isinstance(payload, Mapping):
                    continue
                reflex = payload.get("reflex")
                if reflex:
                    debug_counts[module_name] += 1
                if _coerce_bool(payload.get("module_reflex_override")):
                    debug_override_counts[module_name] += 1
                debug_dominance_sum[module_name] += _coerce_float(
                    payload.get("module_reflex_dominance"),
                    0.0,
                )

    tick_count = len(trace)
    module_rows: list[dict[str, object]] = []
    for module_name in sorted(module_names):
        message_count = int(message_counts[module_name])
        debug_count = int(debug_counts[module_name])
        reflex_events = message_count if message_count > 0 else debug_count
        debug_denominator = debug_count if debug_count > 0 else reflex_events
        module_rows.append(
            {
                "module": module_name,
                "reflex_events": reflex_events,
                "events_per_tick": _safe_divide(reflex_events, float(tick_count)),
                "message_reflex_events": message_count,
                "debug_reflex_events": debug_count,
                "override_rate": _safe_divide(
                    float(debug_override_counts[module_name]),
                    float(debug_denominator),
                ),
                "mean_dominance": _safe_divide(
                    float(debug_dominance_sum[module_name]),
                    float(debug_denominator),
                ),
            }
        )

    limitations: list[str] = []
    if not uses_debug_reflexes:
        limitations.append(
            "Debug reflex payloads were not present; override_rate and mean_dominance may stay at zero even when reflexes occurred."
        )
    return {
        "available": bool(module_rows),
        "tick_count": tick_count,
        "modules": module_rows,
        "uses_debug_reflexes": uses_debug_reflexes,
        "limitations": limitations,
    }


def build_scenario_checks_rows(
    scenario_success: Mapping[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenario_success.get("scenarios", []) if isinstance(scenario_success.get("scenarios"), list) else []:
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
    rows: list[dict[str, object]] = []

    def add_components(source: str, scope: str, components: object) -> None:
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

    for scenario in scenario_success.get("scenarios", []) if isinstance(scenario_success.get("scenarios"), list) else []:
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
        weakest = min(
            (
                scenario
                for scenario in scenarios
                if isinstance(scenario, Mapping)
            ),
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
        sortable = []
        for variant_name, payload in variants.items():
            if not isinstance(payload, Mapping):
                continue
            minimal_payload = _variant_with_minimal_reflex_support(payload)
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

    modules = reflex_frequency.get("modules", [])
    if isinstance(modules, list) and modules:
        highest = max(
            (
                module for module in modules
                if isinstance(module, Mapping)
            ),
            key=lambda item: _coerce_float(item.get("reflex_events"), 0.0),
        )
        diagnostics.append(
            {
                "label": "Most frequent reflex source",
                "value": f"{highest.get('module')} ({int(_coerce_float(highest.get('reflex_events'), 0.0))} events)",
            }
        )
    return diagnostics


def _chart_bounds(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        pad = 1.0 if minimum == 0.0 else abs(minimum) * 0.15
        return minimum - pad, maximum + pad
    pad = (maximum - minimum) * 0.1
    return minimum - pad, maximum + pad


def render_placeholder_svg(title: str, message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="260" viewBox="0 0 900 260">'
        '<rect x="0" y="0" width="900" height="260" fill="#f8fafc" />'
        f'<text x="40" y="54" font-size="24" font-family="monospace" fill="#0f172a">{escape(title)}</text>'
        f'<text x="40" y="120" font-size="16" font-family="monospace" fill="#475569">{escape(message)}</text>'
        "</svg>"
    )


def render_line_chart(
    title: str,
    series: Sequence[dict[str, object]],
    *,
    x_key: str = "index",
    y_key: str = "reward",
) -> str:
    if not series:
        return render_placeholder_svg(title, "No data available.")
    width = 900
    height = 320
    left = 70
    right = 30
    top = 55
    bottom = 45
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_y = [_coerce_float(point.get(y_key)) for point in series]
    min_y, max_y = _chart_bounds(all_y)
    max_x = max(_coerce_float(point.get(x_key), 1.0) for point in series)
    max_x = max(max_x, 1.0)

    def x_coord(value: float) -> float:
        return left + (value / max_x) * plot_width

    def y_coord(value: float) -> float:
        if math.isclose(max_y, min_y):
            return top + plot_height / 2.0
        normalized = (value - min_y) / (max_y - min_y)
        return top + (1.0 - normalized) * plot_height

    points_attr = " ".join(
        f"{x_coord(_coerce_float(point.get(x_key), 0.0)):.2f},{y_coord(_coerce_float(point.get(y_key), 0.0)):.2f}"
        for point in series
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{left}" y="32" font-size="22" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>',
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{points_attr}" />',
    ]
    for point in series:
        px = x_coord(_coerce_float(point.get(x_key), 0.0))
        py = y_coord(_coerce_float(point.get(y_key), 0.0))
        lines.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.5" fill="#1d4ed8" />'
        )
    for idx, label in enumerate((min_y, (min_y + max_y) / 2.0, max_y)):
        y = y_coord(label)
        lines.append(
            f'<text x="12" y="{y + 5:.2f}" font-size="12" font-family="monospace" fill="#475569">{label:.2f}</text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
    lines.append("</svg>")
    return "".join(lines)


def render_bar_chart(
    title: str,
    items: Sequence[Mapping[str, object]],
    *,
    label_key: str,
    value_key: str,
) -> str:
    """
    Render a horizontal bar chart as an SVG string.
    
    Parameters:
        title (str): Chart title placed at the top-left of the SVG.
        items (Sequence[Mapping[str, object]]): Sequence of records for each bar. Each record is expected to contain keys specified by `label_key` and `value_key`.
        label_key (str): Key in each item mapping used for the bar label (displayed at left).
        value_key (str): Key in each item mapping used for the numeric value determining bar length; non-finite or missing values are treated as 0.
    
    Returns:
        str: An SVG document string containing a horizontal bar chart of the provided items. If `items` is empty, returns a placeholder SVG indicating "No data available."
    """
    if not items:
        return render_placeholder_svg(title, "No data available.")
    width = 960
    height = max(260, 70 + 34 * len(items))
    left = 250
    right = 30
    top = 55
    bar_height = 22
    gap = 10
    plot_width = width - left - right
    max_value = max(_coerce_float(item.get(value_key), 0.0) for item in items)
    max_value = max(max_value, 1.0)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="32" y="32" font-size="22" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
    ]
    for index, item in enumerate(items):
        value = _coerce_float(item.get(value_key), 0.0)
        label = str(item.get(label_key) or "")
        y = top + index * (bar_height + gap)
        bar_width = plot_width * _safe_divide(value, max_value)
        lines.append(
            f'<text x="32" y="{y + 16}" font-size="13" font-family="monospace" fill="#334155">{escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{plot_width}" height="{bar_height}" fill="#e2e8f0" rx="4" />'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" fill="#2563eb" rx="4" />'
        )
        lines.append(
            f'<text x="{left + plot_width + 8}" y="{y + 16}" font-size="12" font-family="monospace" fill="#475569">{value:.2f}</text>'
        )
    lines.append("</svg>")
    return "".join(lines)


def render_matrix_heatmap(
    title: str,
    *,
    train_conditions: Sequence[str],
    eval_conditions: Sequence[str],
    matrix: Mapping[str, Mapping[str, Mapping[str, object]]],
    train_marginals: Mapping[str, object],
    eval_marginals: Mapping[str, object],
) -> str:
    """
    Render a train-by-eval heatmap as an SVG string.
    
    Renders a rectangular grid where each cell shows a success-rate value (0.00-1.00 clamped and shown as a two-decimal percentage-like fraction) with a background color scaled from light to dark according to the value. Appends an extra "mean" row and column computed from the provided marginals.
    
    Parameters:
        title (str): Chart title rendered at the top-left of the SVG.
        train_conditions (Sequence[str]): Ordered list of training condition names used for matrix rows.
        eval_conditions (Sequence[str]): Ordered list of evaluation condition names used for matrix columns.
        matrix (Mapping[str, Mapping[str, Mapping[str, object]]]): Nested mapping of train_condition -> eval_condition -> cell payload; cell payload should expose `scenario_success_rate` (numeric) when available.
        train_marginals (Mapping[str, object]): Per-train-condition marginal values used for the "mean" column and the final mean row.
        eval_marginals (Mapping[str, object]): Per-eval-condition marginal values used for the "mean" row and the final mean column.
    
    Returns:
        str: An SVG document as a string. If either `train_conditions` or `eval_conditions` is empty, returns a placeholder SVG indicating no data.
    """
    if not train_conditions or not eval_conditions:
        return render_placeholder_svg(title, "No data available.")
    width = 980
    height = max(320, 130 + 52 * (len(train_conditions) + 1))
    left = 220
    top = 92
    cell_width = max(78, int((width - left - 60) / (len(eval_conditions) + 1)))
    cell_height = 42

    def _cell_color(value: float | None) -> str:
        if value is None:
            return "#e2e8f0"
        clamped = max(0.0, min(1.0, value))
        red = int(241 - (clamped * 204))
        green = int(245 - (clamped * 108))
        blue = int(249 - (clamped * 26))
        return f"rgb({red},{green},{blue})"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
        f'<text x="32" y="36" font-size="22" font-family="monospace" fill="#0f172a">{escape(title)}</text>',
        f'<text x="{left - 120}" y="{top - 18}" font-size="13" font-family="monospace" fill="#475569">train \\ eval</text>',
    ]

    headers = [*eval_conditions, "mean"]
    for index, label in enumerate(headers):
        x = left + index * cell_width
        lines.append(
            f'<text x="{x + 8}" y="{top - 18}" font-size="13" font-family="monospace" fill="#334155">{escape(str(label))}</text>'
        )

    for row_index, train_condition in enumerate([*train_conditions, "mean"]):
        y = top + row_index * cell_height
        lines.append(
            f'<text x="32" y="{y + 26}" font-size="13" font-family="monospace" fill="#334155">{escape(str(train_condition))}</text>'
        )
        for col_index, eval_condition in enumerate(headers):
            x = left + col_index * cell_width
            if train_condition == "mean" and eval_condition == "mean":
                value = _noise_robustness_mean(
                    _noise_robustness_marginal(train_marginals, name)
                    for name in train_conditions
                )
            elif train_condition == "mean":
                value = _noise_robustness_marginal(eval_marginals, eval_condition)
            elif eval_condition == "mean":
                value = _noise_robustness_marginal(train_marginals, train_condition)
            else:
                value = _noise_robustness_rate(
                    matrix,
                    train_condition=train_condition,
                    eval_condition=eval_condition,
                )
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 8}" height="{cell_height - 8}" fill="{_cell_color(value)}" stroke="#cbd5e1" stroke-width="1" rx="4" />'
            )
            lines.append(
                f'<text x="{x + 10}" y="{y + 24}" font-size="13" font-family="monospace" fill="#0f172a">{escape(_format_optional_metric(value))}</text>'
            )
    lines.append("</svg>")
    return "".join(lines)


def _write_svg(path: Path, svg: str) -> None:
    """
    Write SVG content to the given filesystem path.
    
    Writes the provided SVG text to `path` using UTF-8 encoding, overwriting any existing file.
    """
    path.write_text(svg, encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _markdown_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    if not rows:
        return "_No data available._"
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def build_report_data(
    *,
    summary: Mapping[str, object],
    trace: Sequence[Mapping[str, object]],
    behavior_rows: Sequence[Mapping[str, object]],
    summary_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    behavior_csv_path: str | Path | None = None,
) -> dict[str, object]:
    """
    Assembles analysis sections from provided inputs and returns the consolidated report payload.
    
    Parameters:
        summary: Parsed summary JSON mapping (may be empty) used as the primary source for evaluations and ablations.
        trace: Sequence of parsed trace records (JSONL entries); used to compute reflex frequency and trace-derived indicators.
        behavior_rows: Sequence of behavior CSV row mappings; normalized and used to reconstruct scenario/ablation/comparison data when summary is incomplete.
        summary_path: Optional original path for the summary input (used only for reporting).
        trace_path: Optional original path for the trace input (used only for reporting).
        behavior_csv_path: Optional original path for the behavior CSV input (used only for reporting).
    
    Returns:
        A dict containing assembled report sections and metadata:
        - inputs: dict with input paths and booleans indicating which inputs produced data.
        - training_eval: training/evaluation series payload.
        - scenario_success: normalized per-scenario success payload.
        - primary_benchmark: selected "no-reflex" benchmark payload and provenance.
        - shaping_program: reward-shaping minimization audit, gap metrics, and minimal-profile survival indicators.
        - comparisons: reward-profile and map-template comparison payloads.
        - ablations: ablation variants, reference selection, deltas vs reference, and limitations.
        - reflex_frequency: per-module reflex event metrics derived from trace.
        - reflex_dependence: override/dominance indicators, thresholds, and statuses.
        - scenario_checks: list of CSV-like rows describing scenario checks.
        - reward_components: list of rows for reward component values aggregated from summary, scenarios, and trace.
        - diagnostics: diagnostic metric entries aggregated from the various analyses.
        - limitations: list of textual limitation messages surfaced from the assembled sections.
    """
    normalized_rows = normalize_behavior_rows(behavior_rows)
    training_eval = extract_training_eval_series(summary)
    scenario_success = extract_scenario_success(summary, normalized_rows)
    comparisons = extract_comparisons(summary, normalized_rows)
    noise_robustness = extract_noise_robustness(summary, normalized_rows)
    ablations = extract_ablations(summary, normalized_rows)
    shaping_program = extract_shaping_audit(summary)
    reflex_frequency = extract_reflex_frequency(trace)
    predator_type_specialization = extract_predator_type_specialization(
        summary,
        normalized_rows,
        trace,
    )
    primary_benchmark = build_primary_benchmark(
        summary,
        scenario_success,
        ablations,
    )
    reflex_dependence = build_reflex_dependence_indicators(
        summary,
        reflex_frequency,
    )
    scenario_check_rows = build_scenario_checks_rows(scenario_success)
    reward_component_rows = build_reward_component_rows(
        summary,
        scenario_success,
        trace,
    )
    diagnostics = build_diagnostics(
        summary,
        scenario_success,
        ablations,
        reflex_frequency,
    )

    limitations: list[str] = []
    for section in (
        training_eval,
        scenario_success,
        comparisons,
        noise_robustness,
        ablations,
        shaping_program,
        reflex_frequency,
        predator_type_specialization,
    ):
        if isinstance(section, Mapping):
            limitations.extend(
                str(item)
                for item in section.get("limitations", [])
                if item
            )

    inputs = {
        "summary_path": str(summary_path) if summary_path is not None else None,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "behavior_csv_path": str(behavior_csv_path) if behavior_csv_path is not None else None,
        "summary_loaded": bool(summary),
        "trace_loaded": bool(trace),
        "behavior_csv_loaded": bool(normalized_rows),
    }
    return {
        "inputs": inputs,
        "training_eval": training_eval,
        "scenario_success": scenario_success,
        "primary_benchmark": primary_benchmark,
        "shaping_program": shaping_program,
        "comparisons": comparisons,
        "noise_robustness": noise_robustness,
        "ablations": ablations,
        "predator_type_specialization": predator_type_specialization,
        "reflex_frequency": reflex_frequency,
        "reflex_dependence": reflex_dependence,
        "scenario_checks": scenario_check_rows,
        "reward_components": reward_component_rows,
        "diagnostics": diagnostics,
        "limitations": limitations,
    }


def write_report(output_dir: str | Path, report: Mapping[str, object]) -> dict[str, str]:
    """
    Write filesystem artifacts (SVG, CSV, JSON, and Markdown) that represent the provided analysis report.
    
    Parameters:
        output_dir (str | Path): Target directory where artifact files will be created; created if missing.
        report (Mapping[str, object]): Report data used to populate charts, tables, and markdown. Expected keys include (but are not limited to) "training_eval", "scenario_success", "shaping_program", "ablations", "reflex_frequency", "scenario_checks", "reward_components", "diagnostics", "primary_benchmark", "limitations", and "inputs".
    
    Returns:
        dict[str, str]: Mapping of artifact identifiers to their filesystem paths. Keys include "report_md", "report_json", "training_eval_svg", "scenario_success_svg", "scenario_checks_csv", "reward_components_csv", "ablation_comparison_svg", and "reflex_frequency_svg".
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_eval = report.get("training_eval", {})
    scenario_success = report.get("scenario_success", {})
    primary_benchmark = report.get("primary_benchmark", {})
    shaping_program = report.get("shaping_program", {})
    noise_robustness = report.get("noise_robustness", {})
    ablations = report.get("ablations", {})
    predator_type_specialization = report.get("predator_type_specialization", {})
    reflex_frequency = report.get("reflex_frequency", {})

    training_series = []
    if isinstance(training_eval, Mapping):
        training_series.extend(
            training_eval.get("training_points", [])
            if isinstance(training_eval.get("training_points"), list)
            else []
        )
        evaluation_points = (
            training_eval.get("evaluation_points", [])
            if isinstance(training_eval.get("evaluation_points"), list)
            else []
        )
    else:
        evaluation_points = []

    training_svg_path = output_path / "training_eval.svg"
    if training_series:
        _write_svg(
            training_svg_path,
            render_line_chart("Training reward trajectory", training_series),
        )
    elif evaluation_points:
        _write_svg(
            training_svg_path,
            render_line_chart("Evaluation reward trajectory", evaluation_points),
        )
    else:
        _write_svg(
            training_svg_path,
            render_placeholder_svg("Training / evaluation", "No summary reward series was available."),
        )

    scenario_svg_path = output_path / "scenario_success.svg"
    scenario_items = (
        scenario_success.get("scenarios", [])
        if isinstance(scenario_success, Mapping)
        else []
    )
    scenario_chart_items = [
        {"scenario": item.get("scenario"), "success_rate": item.get("success_rate")}
        for item in scenario_items
        if isinstance(item, Mapping)
    ]
    _write_svg(
        scenario_svg_path,
        render_bar_chart(
            "Scenario success rate",
            scenario_chart_items,
            label_key="scenario",
            value_key="success_rate",
        ),
    )

    robustness_svg_path = output_path / "robustness_matrix.svg"
    _nr = _mapping_or_empty(noise_robustness)
    robustness_matrix_spec = _mapping_or_empty(_nr.get("matrix_spec"))
    _train_conds_raw = robustness_matrix_spec.get("train_conditions")
    _eval_conds_raw = robustness_matrix_spec.get("eval_conditions")
    robustness_train_conditions = list(_train_conds_raw) if isinstance(_train_conds_raw, list) else []
    robustness_eval_conditions = list(_eval_conds_raw) if isinstance(_eval_conds_raw, list) else []
    robustness_matrix = _mapping_or_empty(_nr.get("matrix"))
    robustness_train_marginals = _mapping_or_empty(_nr.get("train_marginals"))
    robustness_eval_marginals = _mapping_or_empty(_nr.get("eval_marginals"))
    _write_svg(
        robustness_svg_path,
        render_matrix_heatmap(
            "Noise robustness matrix",
            train_conditions=robustness_train_conditions,
            eval_conditions=robustness_eval_conditions,
            matrix=robustness_matrix,
            train_marginals=robustness_train_marginals,
            eval_marginals=robustness_eval_marginals,
        ),
    )

    ablation_svg_path = ""
    if isinstance(ablations, Mapping) and ablations.get("available"):
        variants = []
        variants_payload = ablations.get("variants", {})
        if isinstance(variants_payload, Mapping):
            for variant_name, payload in sorted(variants_payload.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_payload = payload.get("summary", {})
                variants.append(
                    {
                        "variant": variant_name,
                        "score": _coerce_float(
                            summary_payload.get("scenario_success_rate"),
                            0.0,
                        ),
                    }
                )
        if variants:
            ablation_svg_path = str(output_path / "ablation_comparison.svg")
            _write_svg(
                Path(ablation_svg_path),
                render_bar_chart(
                    "Ablation scenario success rate",
                    variants,
                    label_key="variant",
                    value_key="score",
                ),
            )

    reflex_svg_path = ""
    if isinstance(reflex_frequency, Mapping) and reflex_frequency.get("available"):
        modules = reflex_frequency.get("modules", [])
        if isinstance(modules, list) and modules:
            reflex_svg_path = str(output_path / "reflex_frequency.svg")
            _write_svg(
                Path(reflex_svg_path),
                render_bar_chart(
                    "Reflex events per module",
                    modules,
                    label_key="module",
                    value_key="reflex_events",
                ),
            )

    scenario_checks_path = output_path / "scenario_checks.csv"
    _write_csv(
        scenario_checks_path,
        report.get("scenario_checks", []) if isinstance(report.get("scenario_checks"), list) else [],
        ("scenario", "check_name", "pass_rate", "mean_value", "expected", "description", "source"),
    )

    reward_components_path = output_path / "reward_components.csv"
    _write_csv(
        reward_components_path,
        report.get("reward_components", []) if isinstance(report.get("reward_components"), list) else [],
        ("source", "scope", "component", "value"),
    )

    report_json_path = output_path / "report.json"
    report_json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    diagnostics = report.get("diagnostics", [])
    diagnostic_rows = [
        (
            str(item.get("label") or ""),
            str(item.get("value") or ""),
        )
        for item in diagnostics
        if isinstance(item, Mapping)
    ]
    primary_benchmark_rows = []
    if isinstance(primary_benchmark, Mapping) and primary_benchmark.get("available"):
        primary_benchmark_rows.append(
            (
                str(primary_benchmark.get("label") or ""),
                f"{_coerce_float(primary_benchmark.get('scenario_success_rate')):.2f}",
                (
                    "n/a"
                    if primary_benchmark.get("eval_reflex_scale") is None
                    else f"{_coerce_float(primary_benchmark.get('eval_reflex_scale')):.2f}"
                ),
                str(primary_benchmark.get("reference_variant") or ""),
                str(primary_benchmark.get("source") or ""),
            )
        )
    shaping_gap_rows = []
    shaping_component_rows = []
    shaping_profile_rows = []
    shaping_disposition_rows = []
    shaping_survival_rows = []
    shaping_summary_line = "_No shaping minimization audit was available._"
    shaping_removed_gap_line = "_No removed disposition weight gap was available._"
    shaping_warning_line = ""
    shaping_program = _mapping_or_empty(shaping_program)
    if shaping_program and bool(shaping_program.get("available")):
        dense_profile = str(shaping_program.get("dense_profile") or "classic")
        minimal_profile = str(shaping_program.get("minimal_profile") or "austere")
        gap_metrics = _mapping_or_empty(shaping_program.get("gap_metrics"))
        flags = _mapping_or_empty(shaping_program.get("interpretive_flags"))
        thresholds = _mapping_or_empty(shaping_program.get("thresholds"))
        if gap_metrics and bool(flags.get("gap_available")):
            scenario_delta = _coerce_float(
                gap_metrics.get("scenario_success_rate_delta")
            )
            episode_delta = _coerce_float(
                gap_metrics.get("episode_success_rate_delta")
            )
            reward_delta = _coerce_float(gap_metrics.get("mean_reward_delta"))
            shaping_gap_rows.append(
                (
                    f"{dense_profile} - {minimal_profile}",
                    f"{scenario_delta:.2f}",
                    f"{episode_delta:.2f}",
                    f"{reward_delta:.2f}",
                    str(shaping_program.get("interpretation") or ""),
                )
            )
        if bool(flags.get("shaping_dependent")):
            shaping_warning_line = (
                "> WARNING: High shaping dependence detected. Dense-profile "
                "scenario success exceeds the minimal-shaping warning threshold of "
                f"{_coerce_float(thresholds.get('shaping_dependence'), SHAPING_DEPENDENCE_WARNING_THRESHOLD):.2f}."
            )
        profile_weight_breakdown = _mapping_or_empty(
            shaping_program.get("profile_weight_breakdown")
        )
        for profile_name, weights in sorted(profile_weight_breakdown.items()):
            if not isinstance(weights, Mapping):
                continue
            for disposition, weight in sorted(weights.items()):
                shaping_profile_rows.append(
                    (
                        str(profile_name),
                        str(disposition),
                        f"{_coerce_float(weight):.2f}",
                    )
                )
        disposition_summary = _mapping_or_empty(
            shaping_program.get("disposition_summary")
        )
        for disposition, payload in sorted(disposition_summary.items()):
            if not isinstance(payload, Mapping):
                continue
            components = payload.get("components", [])
            if isinstance(components, list):
                component_names = ", ".join(str(component) for component in components)
                default_count = len(components)
            else:
                component_names = ""
                default_count = 0
            shaping_disposition_rows.append(
                (
                    str(disposition),
                    str(int(_coerce_float(payload.get("component_count"), default_count))),
                    component_names,
                )
            )
        shaping_removed_gap_line = (
            "Removed disposition weight gap "
            f"({dense_profile} - {minimal_profile}): "
            f"{_coerce_float(shaping_program.get('removed_weight_gap')):.2f}."
        )
        components = shaping_program.get("component_classification", [])
        if isinstance(components, list):
            shaping_component_rows = [
                (
                    str(item.get("component") or ""),
                    str(item.get("category") or ""),
                    str(item.get("risk") or ""),
                    str(item.get("disposition") or ""),
                    str(item.get("rationale") or ""),
                )
                for item in components
                if isinstance(item, Mapping)
            ]
        behavior_survival = _mapping_or_empty(shaping_program.get("behavior_survival"))
        survival_scenarios = behavior_survival.get("scenarios", [])
        if isinstance(survival_scenarios, list):
            shaping_survival_rows = [
                (
                    str(item.get("scenario") or ""),
                    f"{_coerce_float(item.get('austere_success_rate')):.2f}",
                    "yes" if bool(item.get("survives")) else "no",
                    str(int(_coerce_float(item.get("episodes"), 0.0))),
                )
                for item in survival_scenarios
                if isinstance(item, Mapping)
            ]
        if not bool(behavior_survival.get("available")):
            shaping_summary_line = "_No shaping survival data available._"
        else:
            shaping_summary_line = (
                f"{int(_coerce_float(behavior_survival.get('surviving_scenario_count'), 0.0))}/"
                f"{int(_coerce_float(behavior_survival.get('scenario_count'), 0.0))} "
                "scenarios survive minimal shaping "
                f"({_coerce_float(behavior_survival.get('survival_rate')):.2f})."
            )
    scenario_rows = [
        (
            str(item.get("scenario") or ""),
            f"{_coerce_float(item.get('success_rate')):.2f}",
            str(len(item.get("checks", {})) if isinstance(item.get("checks"), Mapping) else 0),
        )
        for item in scenario_items
        if isinstance(item, Mapping)
    ]
    ablation_rows = []
    ablation_predator_type_rows = []
    if isinstance(ablations, Mapping):
        variants_payload = ablations.get("variants", {})
        if isinstance(variants_payload, Mapping):
            for variant_name, payload in sorted(variants_payload.items()):
                if not isinstance(payload, Mapping):
                    continue
                summary_payload = payload.get("summary", {})
                config_payload = payload.get("config", {})
                if not isinstance(summary_payload, Mapping):
                    summary_payload = {}
                if not isinstance(config_payload, Mapping):
                    config_payload = {}
                ablation_rows.append(
                    (
                        str(variant_name),
                        str(config_payload.get("architecture") or ""),
                        f"{_coerce_float(summary_payload.get('eval_reflex_scale'), _coerce_float(payload.get('eval_reflex_scale'), 0.0)):.2f}",
                        f"{_coerce_float(summary_payload.get('scenario_success_rate')):.2f}",
                    )
                )
        predator_type_comparisons = ablations.get("predator_type_comparisons", {})
        if isinstance(predator_type_comparisons, Mapping):
            comparisons = predator_type_comparisons.get("comparisons", {})
            if isinstance(comparisons, Mapping):
                for variant_name, payload in sorted(comparisons.items()):
                    if not isinstance(payload, Mapping):
                        continue
                    visual_group = _mapping_or_empty(payload.get("visual_predator_scenarios"))
                    olfactory_group = _mapping_or_empty(payload.get("olfactory_predator_scenarios"))
                    ablation_predator_type_rows.append(
                        (
                            str(variant_name),
                            f"{_coerce_float(visual_group.get('mean_success_rate')):.2f}",
                            f"{_coerce_float(olfactory_group.get('mean_success_rate')):.2f}",
                            f"{_coerce_float(payload.get('visual_minus_olfactory_success_rate')):.2f}",
                        )
                    )

    specialization_rows = []
    specialization_summary_line = "_No predator-type specialization metrics were available._"
    specialization = _mapping_or_empty(predator_type_specialization)
    if specialization and bool(specialization.get("available")):
        predator_types = _mapping_or_empty(specialization.get("predator_types"))
        for predator_type in ("visual", "olfactory"):
            payload = _mapping_or_empty(predator_types.get(predator_type))
            specialization_rows.append(
                (
                    predator_type,
                    f"{_coerce_float(payload.get('visual_cortex_activation')):.2f}",
                    f"{_coerce_float(payload.get('sensory_cortex_activation')):.2f}",
                    str(payload.get("dominant_module") or ""),
                )
            )
        specialization_summary_line = (
            "Specialization score: "
            f"{_coerce_float(specialization.get('specialization_score')):.2f} "
            f"({str(specialization.get('interpretation') or 'unknown')}). "
            "Type-module correlation proxy: "
            f"{_coerce_float(specialization.get('type_module_correlation')):.2f}."
        )
    differential_activation = _mapping_or_empty(
        _mapping_or_empty(predator_type_specialization).get("differential_activation")
    )
    specialization_delta_rows = []
    if differential_activation:
        specialization_delta_rows = [
            (
                "visual_cortex (visual - olfactory)",
                f"{_coerce_float(differential_activation.get('visual_cortex_visual_minus_olfactory')):.2f}",
            ),
            (
                "sensory_cortex (olfactory - visual)",
                f"{_coerce_float(differential_activation.get('sensory_cortex_olfactory_minus_visual')):.2f}",
            ),
        ]

    robustness_rows = []
    if robustness_train_conditions and robustness_eval_conditions:
        for train_condition in robustness_train_conditions:
            row = [train_condition]
            for eval_condition in robustness_eval_conditions:
                row.append(
                    _format_optional_metric(
                        _noise_robustness_rate(
                            robustness_matrix,
                            train_condition=train_condition,
                            eval_condition=eval_condition,
                        )
                    )
                )
            row.append(
                _format_optional_metric(
                    _noise_robustness_marginal(
                        robustness_train_marginals,
                        train_condition,
                    )
                )
            )
            robustness_rows.append(tuple(row))
        marginal_row = ["mean"]
        for eval_condition in robustness_eval_conditions:
            marginal_row.append(
                _format_optional_metric(
                    _noise_robustness_marginal(
                        robustness_eval_marginals,
                        eval_condition,
                    )
                )
            )
        marginal_row.append(
            _format_optional_metric(
                _coerce_optional_float(noise_robustness.get("robustness_score"))
            )
        )
        robustness_rows.append(tuple(marginal_row))

    markdown_lines = [
        "# Offline Analysis Report",
        "",
        "## Inputs",
        "",
        f"- `summary`: {report['inputs']['summary_path']}",
        f"- `trace`: {report['inputs']['trace_path']}",
        f"- `behavior_csv`: {report['inputs']['behavior_csv_path']}",
        "",
        "## Diagnostics",
        "",
        _markdown_table(diagnostic_rows, ("metric", "value")),
        "",
        "## Primary Benchmark",
        "",
        _markdown_table(
            primary_benchmark_rows,
            ("metric", "scenario_success_rate", "eval_reflex_scale", "reference_variant", "source"),
        ),
        "",
        "## Shaping Minimization Program",
        "",
    ]
    if shaping_warning_line:
        markdown_lines.extend([shaping_warning_line, ""])
    markdown_lines.extend(
        [
            "### Dense vs Minimal Gap",
            "",
            _markdown_table(
                shaping_gap_rows,
                (
                    "comparison",
                    "scenario_success_rate_delta",
                    "episode_success_rate_delta",
                    "mean_reward_delta",
                    "interpretation",
                ),
            ),
            "",
            "### Profile-Level Summary",
            "",
            shaping_removed_gap_line,
            "",
            "Profile disposition weight proxies:",
            "",
            _markdown_table(
                shaping_profile_rows,
                ("profile", "disposition", "total_weight_proxy"),
            ),
            "",
            "Disposition component summary:",
            "",
            _markdown_table(
                shaping_disposition_rows,
                ("disposition", "component_count", "components"),
            ),
            "",
            "### Component Dispositions",
            "",
            _markdown_table(
                shaping_component_rows,
                ("Component", "Category", "Risk", "Disposition", "Rationale"),
            ),
            "",
            "### Behavior Survival",
            "",
            shaping_summary_line,
            "",
            _markdown_table(
                shaping_survival_rows,
                ("scenario", "austere_success_rate", "survives", "episodes"),
            ),
            "",
            "## Scenario Success",
            "",
            _markdown_table(scenario_rows, ("scenario", "success_rate", "check_count")),
            "",
            "## Ablations",
            "",
            _markdown_table(
                ablation_rows,
                ("variant", "architecture", "eval_reflex_scale", "scenario_success_rate"),
            ),
            "",
            "Ablation predator-type comparisons:",
            "",
            _markdown_table(
                ablation_predator_type_rows,
                ("variant", "visual_scenarios", "olfactory_scenarios", "visual_minus_olfactory"),
            ),
            "",
            "## Predator Type Specialization",
            "",
            specialization_summary_line,
            "",
            _markdown_table(
                specialization_rows,
                (
                    "predator_type",
                    "visual_cortex_activation",
                    "sensory_cortex_activation",
                    "dominant_module",
                ),
            ),
            "",
            _markdown_table(
                specialization_delta_rows,
                ("differential", "value"),
            ),
            "",
            "## Noise Robustness Matrix",
            "",
            (
                _markdown_table(
                    robustness_rows,
                    ("train \\ eval", *robustness_eval_conditions, "mean"),
                )
                if robustness_rows
                else "_No noise robustness matrix was available._"
            ),
            "",
            "Overall robustness score: "
            f"{_format_optional_metric(_coerce_optional_float(_mapping_or_empty(noise_robustness).get('robustness_score')))}.",
            "Diagonal score: "
            f"{_format_optional_metric(_coerce_optional_float(_mapping_or_empty(noise_robustness).get('diagonal_score')))}.",
            "Off-diagonal score: "
            f"{_format_optional_metric(_coerce_optional_float(_mapping_or_empty(noise_robustness).get('off_diagonal_score')))}.",
            "",
            "## Limitations",
            "",
        ]
    )
    limitations = report.get("limitations", [])
    if isinstance(limitations, list) and limitations:
        markdown_lines.extend(f"- {item}" for item in limitations)
    else:
        markdown_lines.append("- None.")
    markdown_lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `training_eval.svg`",
            "- `scenario_success.svg`",
            "- `robustness_matrix.svg`",
            "- `scenario_checks.csv`",
            "- `reward_components.csv`",
            "- `report.json`",
        ]
    )
    if ablation_svg_path:
        markdown_lines.append("- `ablation_comparison.svg`")
    if reflex_svg_path:
        markdown_lines.append("- `reflex_frequency.svg`")

    report_md_path = output_path / "report.md"
    report_md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    return {
        "report_md": str(report_md_path),
        "report_json": str(report_json_path),
        "training_eval_svg": str(training_svg_path),
        "scenario_success_svg": str(scenario_svg_path),
        "robustness_matrix_svg": str(robustness_svg_path),
        "scenario_checks_csv": str(scenario_checks_path),
        "reward_components_csv": str(reward_components_path),
        "ablation_comparison_svg": ablation_svg_path,
        "reflex_frequency_svg": reflex_svg_path,
    }


def run_offline_analysis(
    *,
    summary_path: str | Path | None = None,
    trace_path: str | Path | None = None,
    behavior_csv_path: str | Path | None = None,
    output_dir: str | Path,
) -> dict[str, object]:
    summary = load_summary(summary_path)
    trace = load_trace(trace_path)
    behavior_rows = load_behavior_csv(behavior_csv_path)
    report = build_report_data(
        summary=summary,
        trace=trace,
        behavior_rows=behavior_rows,
        summary_path=summary_path,
        trace_path=trace_path,
        behavior_csv_path=behavior_csv_path,
    )
    report["generated_files"] = write_report(output_dir, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line ArgumentParser for the offline analysis tool.
    
    The parser exposes the following options:
    - --summary: path to an optional summary.json input.
    - --trace: path to an optional trace.jsonl input.
    - --behavior-csv: path to an optional behavior CSV exported by the main CLI.
    - --output-dir: required path where the report and artifacts will be written.
    
    Returns:
        argparse.ArgumentParser: A configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Generate reproducible offline analysis from summary, trace, and behavior_csv.",
    )
    parser.add_argument("--summary", type=Path, default=None, help="Path to summary.json.")
    parser.add_argument("--trace", type=Path, default=None, help="Path to trace.jsonl.")
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=None,
        help="Path to behavior CSV exported by the main CLI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the report and artifacts will be generated.",
    )
    return parser


def main() -> None:
    """
    Parse command-line arguments, run the offline analysis pipeline with the provided inputs, write report artifacts to the output directory, and print the resulting report as JSON to stdout.
    
    If none of --summary, --trace, or --behavior-csv are supplied, the parser terminates with an error.
    """
    parser = build_parser()
    args = parser.parse_args()
    if args.summary is None and args.trace is None and args.behavior_csv is None:
        parser.error(
            "At least one of --summary, --trace, and --behavior-csv is required."
        )
    report = run_offline_analysis(
        summary_path=args.summary,
        trace_path=args.trace,
        behavior_csv_path=args.behavior_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
