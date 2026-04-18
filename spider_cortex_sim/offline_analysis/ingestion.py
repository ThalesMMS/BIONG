from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .utils import _coerce_bool, _coerce_float


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
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(
            "summary must be a top-level JSON object; "
            f"found {type(result).__name__}"
        )
    return result


def load_trace(path: str | Path | None) -> list[dict[str, object]]:
    """
    Load a newline-delimited JSON trace file into a list of dictionary payloads.
    
    Parameters:
        path (str | Path | None): Path to a UTF-8 encoded, newline-delimited JSON file. If `None`, no file is read.
    
    Returns:
        list[dict[str, object]]: A list containing each parsed JSON object from non-empty lines that are JSON objects; empty list if `path` is `None`.
    """
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
    """
    Load behavior CSV rows from a file into a list of dictionaries.
    
    Opens the CSV file at `path` using UTF-8 encoding and newline="" and returns a list of rows where each row is a mapping from column names to string values. If `path` is None, returns an empty list.
    
    Parameters:
        path (str | Path | None): Path to the CSV file to read, or `None` to indicate no file.
    
    Returns:
        list[dict[str, object]]: List of rows as dictionaries; empty list when `path` is `None`.
    """
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
