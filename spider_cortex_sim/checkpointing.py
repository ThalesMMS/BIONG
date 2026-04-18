from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Sequence

from .export import jsonify


CHECKPOINT_PENALTY_MODE_NAMES: tuple[str, ...] = ("tiebreaker", "direct")
CHECKPOINT_METRIC_ORDER: tuple[str, ...] = (
    "scenario_success_rate",
    "episode_success_rate",
    "mean_reward",
)


class CheckpointPenaltyMode(str, Enum):
    TIEBREAKER = "tiebreaker"
    DIRECT = "direct"


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    metric: str
    override_penalty_weight: float = 0.0
    dominance_penalty_weight: float = 0.0
    penalty_mode: CheckpointPenaltyMode | str = CheckpointPenaltyMode.TIEBREAKER

    def __post_init__(self) -> None:
        """
        Validate and normalize fields after dataclass initialization.
        
        Ensures `metric` is one of the supported metrics, parses `penalty_mode` into its enum form,
        and validates that `override_penalty_weight` and `dominance_penalty_weight` are finite and
        greater than or equal to zero. Normalized values replace the original attributes on the
        frozen dataclass instance.
        Raises:
            ValueError: If `metric` is not one of 'scenario_success_rate', 'episode_success_rate',
                or 'mean_reward'.
            ValueError: If `penalty_mode` cannot be parsed into a valid CheckpointPenaltyMode
                (message lists available modes).
            ValueError: If either penalty weight is not finite or is negative.
        """
        metric = str(self.metric)
        if metric not in CHECKPOINT_METRIC_ORDER:
            raise ValueError(
                "Invalid checkpoint_metric. Use 'scenario_success_rate', "
                "'episode_success_rate' or 'mean_reward'."
            )
        try:
            mode = CheckpointPenaltyMode(self.penalty_mode)
        except ValueError as exc:
            available = ", ".join(repr(item) for item in CHECKPOINT_PENALTY_MODE_NAMES)
            raise ValueError(
                f"Invalid checkpoint_penalty_mode. Available modes: {available}."
            ) from exc
        override_weight = self._finite_non_negative(
            self.override_penalty_weight,
            "override_penalty_weight",
        )
        dominance_weight = self._finite_non_negative(
            self.dominance_penalty_weight,
            "dominance_penalty_weight",
        )
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "penalty_mode", mode)
        object.__setattr__(self, "override_penalty_weight", override_weight)
        object.__setattr__(self, "dominance_penalty_weight", dominance_weight)

    @staticmethod
    def _finite_non_negative(value: float, field_name: str) -> float:
        """
        Validate and coerce a numeric input to a finite, non-negative float.
        
        Parameters:
            value: A value convertible to float.
            field_name: Identifier used in error messages.
        
        Returns:
            The input converted to a float that is finite and greater than or equal to 0.0.
        
        Raises:
            ValueError: If the value is not finite or is negative.
        """
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field_name} must be finite.")
        if numeric < 0.0:
            raise ValueError(f"{field_name} must be non-negative.")
        return numeric

    def to_summary(self) -> dict[str, object]:
        """
        Produce a JSON-serializable summary of this CheckpointSelectionConfig.
        
        Returns:
            summary (dict[str, object]): A mapping with keys:
                - "metric" (str): The selected metric name.
                - "override_penalty_weight" (float): The override penalty weight as a float.
                - "dominance_penalty_weight" (float): The dominance penalty weight as a float.
                - "penalty_mode" (str): The string value of the penalty mode enum.
        """
        return {
            "metric": self.metric,
            "override_penalty_weight": float(self.override_penalty_weight),
            "dominance_penalty_weight": float(self.dominance_penalty_weight),
            "penalty_mode": self.penalty_mode.value,
        }


def resolve_checkpoint_load_dir(
    checkpoint_dir: str | Path | None,
    *,
    checkpoint_selection: str,
) -> Path | None:
    """
    Selects the most appropriate existing checkpoint directory according to the requested preference.

    Parameters:
        checkpoint_dir (str | Path | None): Root path where checkpoint candidates may exist. If `None`, no directory is selected.
        checkpoint_selection (str): Selection preference; expected values are `"best"`, `"last"`, or `"none"`. `"best"` prefers the `best` subdirectory over `last` and the root; `"last"` prefers the `last` subdirectory over `best` and the root. `"none"` disables selection.

    Returns:
        Path | None: The first existing directory matching the preference that contains a `metadata.json` file, or `None` if no suitable directory is found or selection is disabled.
    """
    if checkpoint_selection not in {"best", "last", "none"}:
        raise ValueError(
            "Invalid checkpoint_selection "
            f"{checkpoint_selection!r}; expected one of 'best', 'last', or 'none'."
        )
    if checkpoint_dir is None or checkpoint_selection == "none":
        return None
    root = Path(checkpoint_dir)
    if checkpoint_selection == "best":
        candidate_dirs = [root / "best", root / "last", root]
    else:
        candidate_dirs = [root / "last", root / "best", root]
    for candidate_dir in candidate_dirs:
        if candidate_dir.is_dir() and (candidate_dir / "metadata.json").exists():
            return candidate_dir
    return None


def checkpoint_run_fingerprint(payload: Dict[str, object]) -> str:
    """
    Compute a stable 12-character fingerprint for checkpoint-compatible run settings.
    
    The fingerprint is produced from the JSON-stable serialization of `payload` after applying `jsonify`, then taking the SHA-256 digest and returning its first 12 hexadecimal characters.
    
    Parameters:
        payload (dict): A mapping of run settings that can be normalized by `jsonify` for stable JSON serialization.
    
    Returns:
        str: A 12-character hexadecimal fingerprint derived from the SHA-256 digest of the serialized payload.
    """
    stable_payload = jsonify(payload)
    serialized = json.dumps(
        stable_payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    """
    Compute the SHA-256 hex digest of a file's contents.
    
    Parameters:
        path (Path): Path to the file to hash.
    
    Returns:
        sha256_hex (str): Hexadecimal SHA-256 digest of the file's contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_preload_fingerprint(
    load_brain: str | Path | None,
    load_modules: Sequence[str] | None = None,
) -> Dict[str, object]:
    """
    Produce content identifiers for a preloaded brain artifact (file or directory) and its selected modules.
    
    Parameters:
        load_brain (str | Path | None): Path to a brain artifact file or directory, or None to indicate no preload.
        load_modules (Sequence[str] | None): Optional sequence of module names to include; when provided the list is deduplicated and sorted. If omitted and `load_brain` is a directory, module names are taken from the artifact's metadata.
    
    Returns:
        Dict[str, object]: A mapping with the following keys:
            - load_brain (str | None): String path of the provided `load_brain`, or `None` if `load_brain` was `None`.
            - metadata_sha256 (str | None): SHA-256 hex digest of `metadata.json` when `load_brain` is a directory; `None` for a single-file artifact or when `load_brain` is `None`.
            - artifact_sha256 (str | None): SHA-256 hex digest of the artifact file when `load_brain` is a file; `None` for directory artifacts or when `load_brain` is `None`.
            - load_modules (list[str] | None): Normalized (deduplicated, sorted) list of module names to load, or `None` when no explicit module list was provided for a file artifact or when `load_brain` is `None`.
            - module_sha256 (dict[str, str] | None): Mapping from module name to SHA-256 hex digest of the corresponding `<module>.npz` file when `load_brain` is a directory; `None` for file artifacts or when `load_brain` is `None`.
    """
    if load_brain is None:
        return {
            "load_brain": None,
            "metadata_sha256": None,
            "load_modules": None,
            "module_sha256": None,
        }

    root = Path(load_brain)
    if root.is_file():
        normalized_modules = (
            sorted({str(module_name) for module_name in load_modules})
            if load_modules is not None
            else None
        )
        return {
            "load_brain": str(root),
            "artifact_sha256": file_sha256(root),
            "load_modules": normalized_modules,
            "module_sha256": None,
        }

    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_modules = metadata.get("modules", {})
    if not isinstance(saved_modules, dict):
        saved_modules = {}
    normalized_modules = (
        sorted({str(module_name) for module_name in load_modules})
        if load_modules is not None
        else sorted(str(module_name) for module_name in saved_modules)
    )
    module_sha256 = {
        module_name: file_sha256(root / f"{module_name}.npz")
        for module_name in normalized_modules
    }
    return {
        "load_brain": str(root),
        "metadata_sha256": file_sha256(metadata_path),
        "load_modules": normalized_modules,
        "module_sha256": module_sha256,
    }


def jsonify_observation(observation: Dict[str, object]) -> Dict[str, object]:
    """
    Convert an observation mapping's values to JSON-safe representations.
    
    Parameters:
        observation (dict): Mapping of observation keys to arbitrary values.
    
    Returns:
        dict: A new mapping with the same keys where each value has been converted to a JSON-safe representation via `jsonify`.
    """
    return {key: jsonify(value) for key, value in observation.items()}


def mean_reward_from_behavior_payload(payload: Dict[str, object]) -> float:
    """
    Compute the arithmetic mean of `mean_reward` values from a payload's legacy scenarios.
    
    Extracts `payload["legacy_scenarios"]` (expected to be a mapping of scenario names to dicts),
    collects each entry's `mean_reward` (missing values treated as 0.0), and returns their average.
    If `legacy_scenarios` is missing, not a dict, empty, or contains no valid entries, returns 0.0.
    
    Parameters:
        payload (dict): A payload that may contain a `legacy_scenarios` mapping of scenario -> dict.
    
    Returns:
        float: The average of the collected `mean_reward` values, or 0.0 if none are available.
    """
    legacy = payload.get("legacy_scenarios", {})
    if not isinstance(legacy, dict) or not legacy:
        return 0.0
    values = [
        float(data.get("mean_reward", 0.0))
        for data in legacy.values()
        if isinstance(data, dict)
    ]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def checkpoint_candidate_sort_key(
    candidate: Dict[str, object],
    *,
    primary_metric: str | None = None,
    selection_config: CheckpointSelectionConfig | None = None,
) -> tuple[float, ...]:
    """
    Build a tuple sort key for ranking a checkpoint candidate according to the given selection configuration.
    
    The returned tuple is suitable for sorting candidates so that higher-priority checkpoints sort later (Python default ascending order can be inverted by the caller if needed).
    
    Parameters:
        candidate (Dict[str, object]): Candidate metadata; expected keys include the metrics named in CHECKPOINT_METRIC_ORDER, "evaluation_summary" (a dict possibly containing "mean_final_reflex_override_rate" and "mean_reflex_dominance"), and "episode".
        primary_metric (str | None): If provided, used to construct a default selection configuration when `selection_config` is not given.
        selection_config (CheckpointSelectionConfig | None): Selection configuration that specifies the primary metric and penalty behavior.
    
    Returns:
        tuple[float, ...]: If `selection_config.penalty_mode` is TIEBREAKER, returns a legacy tuple:
            (primary_metric_value, second_metric_value, third_metric_value, -override_rate, -dominance, episode)
        If `penalty_mode` is DIRECT, returns a tuple whose first element is a composite score that subtracts configured penalties from the primary metric, followed by the legacy tuple:
            (composite_score, primary_metric_value, second_metric_value, third_metric_value, -override_rate, -dominance, episode)
    
    Raises:
        ValueError: If neither `primary_metric` nor `selection_config` is provided, or if both are provided but their metrics disagree.
    """
    if selection_config is None:
        if primary_metric is None:
            raise ValueError("checkpoint selection requires a primary metric or config.")
        selection_config = CheckpointSelectionConfig(metric=primary_metric)
    elif primary_metric is not None and primary_metric != selection_config.metric:
        raise ValueError(
            "primary_metric must match selection_config.metric when both are provided."
        )
    ordered_metrics = [selection_config.metric] + [
        metric_name
        for metric_name in CHECKPOINT_METRIC_ORDER
        if metric_name != selection_config.metric
    ]
    evaluation_summary = candidate.get("evaluation_summary", {})
    if not isinstance(evaluation_summary, dict):
        evaluation_summary = {}
    override_rate = float(
        evaluation_summary.get("mean_final_reflex_override_rate", 0.0)
    )
    dominance = float(evaluation_summary.get("mean_reflex_dominance", 0.0))
    legacy_key = (
        float(candidate.get(ordered_metrics[0], 0.0)),
        float(candidate.get(ordered_metrics[1], 0.0)),
        float(candidate.get(ordered_metrics[2], 0.0)),
        -override_rate,
        -dominance,
        int(candidate.get("episode", 0)),
    )
    if selection_config.penalty_mode is CheckpointPenaltyMode.TIEBREAKER:
        return legacy_key
    composite_score = (
        legacy_key[0]
        - float(selection_config.override_penalty_weight) * override_rate
        - float(selection_config.dominance_penalty_weight) * dominance
    )
    return (float(composite_score), *legacy_key)


def checkpoint_candidate_composite_score(
    candidate: Dict[str, object],
    selection_config: CheckpointSelectionConfig,
) -> float:
    """
    Compute the composite score for a checkpoint candidate using the selection configuration's metric and penalty weights with the penalty mode forced to DIRECT.
    
    Parameters:
        candidate (Dict[str, object]): The checkpoint candidate data (expected to include an "evaluation_summary" and other ranking fields).
        selection_config (CheckpointSelectionConfig): Selection configuration whose metric and penalty weights are used; its `penalty_mode` will be ignored and treated as `DIRECT` for this computation.
    
    Returns:
        float: Composite score (higher values indicate better-ranked candidates).
    """
    direct_config = CheckpointSelectionConfig(
        metric=selection_config.metric,
        override_penalty_weight=selection_config.override_penalty_weight,
        dominance_penalty_weight=selection_config.dominance_penalty_weight,
        penalty_mode=CheckpointPenaltyMode.DIRECT,
    )
    return float(
        checkpoint_candidate_sort_key(
            candidate,
            selection_config=direct_config,
        )[0]
    )


def persist_checkpoint_pair(
    *,
    checkpoint_dir: str | Path | None,
    best_candidate: Dict[str, object],
    last_candidate: Dict[str, object],
) -> Dict[str, str]:
    """
    Persist two checkpoint candidate directories into a checkpoint directory as "best" and "last".
    
    Parameters:
        checkpoint_dir (str | Path | None): Destination root for persisted checkpoints. If `None`, nothing is persisted.
        best_candidate (Dict[str, object]): Candidate mapping containing a `"path"` key pointing to the source directory for the best checkpoint.
        last_candidate (Dict[str, object]): Candidate mapping containing a `"path"` key pointing to the source directory for the last checkpoint.
    
    Returns:
        Dict[str, str]: Mapping with keys `"best"` and `"last"` to the persisted directory paths as strings. Returns an empty dict if `checkpoint_dir` is `None`.
    """
    if checkpoint_dir is None:
        return {}
    destination_root = Path(checkpoint_dir)
    destination_root.mkdir(parents=True, exist_ok=True)
    persisted: Dict[str, str] = {}
    for label, candidate in (("best", best_candidate), ("last", last_candidate)):
        source = Path(candidate["path"])
        target = destination_root / label
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        persisted[label] = str(target)
    return persisted
