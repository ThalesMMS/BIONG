from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from .curriculum import CURRICULUM_COLUMNS


def jsonify(value: Any) -> Any:
    """
    Convert a Python value into a JSON-serializable representation.
    
    Parameters:
        value (Any): The value to convert. Nested structures are processed recursively.
    
    Returns:
        Any: A JSON-serializable form of `value`:
          - If `value` has a `tolist()` method, returns the result of `value.tolist()`.
          - If `value` is a `dict`, returns a dict with keys coerced to `str` and values recursively converted.
          - If `value` is a `list` or `tuple`, returns a `list` with each element recursively converted.
          - If `value` is a `float`, `int`, `str`, `bool`, or `None`, returns it unchanged.
          - For all other types, returns `str(value)`.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def compact_aggregate(data: Dict[str, object]) -> Dict[str, object]:
    """
    Create a shallow copy of an aggregate metrics mapping with per-episode details removed.
    
    Parameters:
        data (Dict[str, object]): Aggregate metrics mapping that may contain an "episodes_detail" key.
    
    Returns:
        Dict[str, object]: A shallow copy of `data` with the "episodes_detail" entry removed if it existed.
    """
    compact = dict(data)
    compact.pop("episodes_detail", None)
    return compact


def compact_behavior_payload(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Produce a compacted behavior-suite payload by removing per-episode details.
    
    Creates a new dict with a shallow copy of `payload["summary"]` (or an empty dict),
    a `suite` mapping where each entry is a shallow copy of the original entry with
    its `"episodes_detail"` removed and its `"legacy_metrics"` replaced by
    `compact_aggregate(...)` when that value is a dict, and a `legacy_scenarios`
    mapping where each entry is compacted via `compact_aggregate(...)`.
    
    Parameters:
        payload (Dict[str, object]): Original behavior payload, expected to possibly
            contain `summary`, `suite`, and `legacy_scenarios` keys.
    
    Returns:
        Dict[str, object]: The compacted payload containing `summary`, `suite`, and
        `legacy_scenarios` with per-episode details removed.
    """
    compact = {
        "summary": dict(payload.get("summary", {})),
        "suite": {},
        "legacy_scenarios": {},
    }
    suite = payload.get("suite", {})
    if isinstance(suite, dict):
        for name, data in suite.items():
            item = dict(data)
            item.pop("episodes_detail", None)
            legacy_metrics = item.get("legacy_metrics", {})
            if isinstance(legacy_metrics, dict):
                item["legacy_metrics"] = compact_aggregate(legacy_metrics)
            compact["suite"][name] = item
    legacy_scenarios = payload.get("legacy_scenarios", {})
    if isinstance(legacy_scenarios, dict):
        for name, data in legacy_scenarios.items():
            compact["legacy_scenarios"][name] = compact_aggregate(dict(data))
    return compact


def save_summary(summary: Dict[str, object], path: str | Path) -> None:
    """Write the simulation summary to a file as pretty-printed JSON."""
    path = Path(path)
    path.write_text(
        json.dumps(jsonify(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_trace(trace: List[Dict[str, object]], path: str | Path) -> None:
    """Write a trace to a file as one JSON object per line."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for item in trace:
            fh.write(json.dumps(jsonify(item), ensure_ascii=False) + "\n")


def save_behavior_csv(rows: List[Dict[str, object]], path: str | Path) -> None:
    """
    Write flattened behavior rows to a CSV file using a preferred column ordering.
    
    The CSV will contain the fixed preferred columns (including all CURRICULUM_COLUMNS), with any additional keys present in the rows appended in sorted order. Each dictionary in `rows` is written as a single CSV row; missing keys are left blank.
    
    Parameters:
        rows (List[Dict[str, object]]): Sequence of flattened behavior records to write.
        path (str | Path): Destination file path for the CSV (written with UTF-8 encoding).
    """
    path = Path(path)
    preferred = [
        "reward_profile",
        "scenario_map",
        "evaluation_map",
        "ablation_variant",
        "ablation_architecture",
        "ablation_architecture_description",
        "capacity_profile",
        "capacity_profile_version",
        "capacity_scale_factor",
        "action_center_hidden_dim",
        "arbitration_hidden_dim",
        "motor_hidden_dim",
        "integration_hidden_dim",
        "monolithic_hidden_dim",
        "credit_strategy",
        "route_mask_enabled",
        "route_mask_threshold",
        "capacity_axis",
        *CURRICULUM_COLUMNS,
        "learning_evidence_condition",
        "learning_evidence_policy_mode",
        "learning_evidence_training_regime",
        "learning_evidence_train_episodes",
        "learning_evidence_frozen_after_episode",
        "learning_evidence_checkpoint_source",
        "learning_evidence_budget_profile",
        "learning_evidence_budget_benchmark_strength",
        "reflex_scale",
        "reflex_anneal_final_scale",
        "competence_type",
        "is_primary_benchmark",
        "is_capability_probe",
        "probe_type",
        "target_skill",
        "geometry_assumptions",
        "benchmark_tier",
        "acceptable_partial_progress",
        "eval_reflex_scale",
        "budget_profile",
        "benchmark_strength",
        "architecture_version",
        "architecture_fingerprint",
        "operational_profile",
        "operational_profile_version",
        "train_noise_profile",
        "train_noise_profile_config",
        "eval_noise_profile",
        "eval_noise_profile_config",
        "noise_profile",
        "noise_profile_config",
        "checkpoint_source",
        "simulation_seed",
        "episode_seed",
        "scenario",
        "scenario_description",
        "scenario_objective",
        "scenario_focus",
        "episode",
        "success",
        "failure_count",
        "failures",
    ]
    extra = sorted({key for row in rows for key in row if key not in preferred})
    fieldnames = preferred + extra
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
