from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from ..utils import (
    _coerce_bool,
    _coerce_float,
    _mean,
)

def _extract_reward_from_episode(detail: Mapping[str, object]) -> float:
    """
    Extracts the numeric reward value from an episode detail mapping.
    
    Prefers the `total_reward` field if present, otherwise uses `reward`; returns 0.0 when neither is available.
    
    Parameters:
        detail (Mapping[str, object]): Episode detail mapping that may contain `total_reward` or `reward`.
    
    Returns:
        float: The reward coerced to a float from `total_reward` or `reward`, or 0.0 if neither key exists.
    """
    if "total_reward" in detail:
        return _coerce_float(detail.get("total_reward"))
    if "reward" in detail:
        return _coerce_float(detail.get("reward"))
    return 0.0

def extract_training_eval_series(summary: Mapping[str, object]) -> dict[str, object]:
    """
    Extract normalized training and evaluation reward series and metadata from a parsed summary.
    
    Parameters:
        summary (Mapping[str, object]): Parsed summary expected to contain optional "training" and "evaluation" mappings. Each mapping may include:
            - "episodes_detail": list of per-episode mappings with "episode", "total_reward" or "reward".
            - "history_tail": alternative list of per-episode mappings for training.
            - "mean_reward" or "mean_reward_all": aggregate reward values used when per-episode detail is absent.
    
    Returns:
        dict[str, object]: A dictionary with the following keys:
            - available (bool): `true` if any training or evaluation points were extracted, `false` otherwise.
            - source (str): Identifier of the data source used (e.g., "summary.training.episodes_detail",
              "summary.training.history_tail", "summary.training.aggregate", "behavior_csv", or "none").
            - training_points (list[dict]): Normalized training points; each entry is a dict with keys:
              "index" (int), "episode" (int), and "reward" (float).
            - evaluation_points (list[dict]): Normalized evaluation points with the same structure as training_points.
            - limitations (list[str]): Human-readable notes describing partial or missing data when applicable.
    """
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
            if source == "none":
                source = "summary.evaluation.episodes_detail"
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
            if source == "none":
                source = "summary.evaluation.aggregate"
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
    """
    Extract the `suite` mapping from a parsed summary's `behavior_evaluation` section.
    
    Parameters:
        summary (Mapping[str, object]): Parsed summary dictionary possibly containing a
            `behavior_evaluation` mapping.
    
    Returns:
        dict[str, object]: A shallow copy of the `suite` mapping if present and valid;
        otherwise an empty dict.
    """
    behavior_evaluation = summary.get("behavior_evaluation", {})
    if not isinstance(behavior_evaluation, Mapping):
        return {}
    suite = behavior_evaluation.get("suite", {})
    if not isinstance(suite, Mapping):
        return {}
    return dict(suite)

def _scenario_suite_from_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """
    Constructs a per-scenario suite summary from a sequence of behavior-row mappings.
    
    Groups rows by their `scenario` field and, for each scenario, aggregates episode count, overall success rate, distinct failures (parsed from semicolon-separated text), and per-check statistics. For each detected check name (derived from keys matching `check_{name}_passed`) the function computes:
    - `pass_rate`: mean of boolean `check_{name}_passed` across rows (as a 0.0-1.0 float),
    - `mean_value`: mean of `check_{name}_value` samples when present, otherwise the `pass_rate`,
    - `expected`: the first non-empty `check_{name}_expected` string found.
    
    Parameters:
        rows (Sequence[Mapping[str, object]]): Sequence of row mappings (typically parsed from a behavior CSV). Rows may contain fields such as `scenario`, `scenario_description`, `scenario_objective`, `success`, `failures`, and check-related keys like `check_{name}_passed`, `check_{name}_value`, and `check_{name}_expected`.
    
    Returns:
        dict[str, object]: Mapping from scenario identifier to a payload with the following keys:
            - `scenario` (str): Scenario identifier.
            - `description` (str): Scenario description (first row's `scenario_description` or empty string).
            - `objective` (str): Scenario objective (first row's `scenario_objective` or empty string).
            - `episodes` (int): Number of rows/episodes for the scenario.
            - `success_rate` (float): Mean success rate across episodes (0.0-1.0).
            - `checks` (dict): Per-check payloads keyed by check name; each payload contains `description`, `expected`, `pass_rate`, and `mean_value`.
            - `behavior_metrics` (dict): Empty dict (reserved).
            - `failures` (list[str]): Sorted list of distinct failure identifiers.
            - `legacy_metrics` (dict): Empty dict (reserved).
    """
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
