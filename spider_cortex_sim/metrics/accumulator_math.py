"""Episode metric accumulation and representation-specialization helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Sequence

from ..ablations import PROPOSAL_SOURCE_NAMES, REFLEX_MODULE_NAMES
from .types import (
    EpisodeStats,
    PREDATOR_RESPONSE_END_THRESHOLD,
    PREDATOR_TYPE_NAMES,
    PRIMARY_REPRESENTATION_READOUT_MODULES,
    PROPOSER_REPRESENTATION_LOGIT_FIELD,
    SHELTER_ROLES,
)

def _clamp_unit_interval(value: float) -> float:
    """
    Clamp a numeric value to the closed interval [0.0, 1.0].
    
    Returns:
        float: The input coerced to float and clamped to the range 0.0 through 1.0.
    """
    return float(max(0.0, min(1.0, float(value))))

def _softmax_probabilities(logits: Sequence[float]) -> List[float]:
    """
    Convert a sequence of logit scores to a probability distribution via softmax.
    
    Parameters:
        logits (Sequence[float]): Input logit scores.
    
    Returns:
        List[float]: A list of probabilities corresponding to `logits`. Returns an empty list if `logits` is empty; if the softmax normalization sum is not greater than 0, returns a list of zeros of the same length.
    """
    values = [float(value) for value in logits]
    if not values:
        return []
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return [0.0 for _ in logits]
    max_logit = max(finite_values)
    values = [value if math.isfinite(value) else -math.inf for value in values]
    exp_values = [math.exp(value - max_logit) for value in values]
    total = sum(exp_values)
    if not math.isfinite(total) or total <= 0.0:
        return [0.0 for _ in values]
    return [float(value / total) for value in exp_values]

def jensen_shannon_divergence(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """
    Compute the base-2 Jensen-Shannon divergence between two numeric vectors.
    
    Inputs may be raw counts or already-normalized probabilities; each input must be a sequence of non-negative numbers of equal length. The inputs are normalized to probability distributions before computing the divergence. If either input is empty or sums to a value <= 0.0, the function returns 0.0.
    
    Parameters:
        left: Sequence[float] — Non-negative numeric values for the first distribution.
        right: Sequence[float] — Non-negative numeric values for the second distribution (must have the same length as `left`).
    
    Returns:
        float: Divergence clamped to the interval [0.0, 1.0]; `0.0` when distributions are identical, `1.0` when they are maximally different on the shared support.
    
    Raises:
        ValueError: If the sequences have different lengths or contain negative values.
    """
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values):
        raise ValueError("Probability distributions must have the same length.")
    if any(not math.isfinite(value) for value in left_values + right_values):
        raise ValueError("Probability distributions cannot contain non-finite values.")
    if any(value < 0.0 for value in left_values + right_values):
        raise ValueError("Probability distributions cannot contain negative values.")
    if not left_values:
        return 0.0
    left_total = sum(left_values)
    right_total = sum(right_values)
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0
    left_probs = [value / left_total for value in left_values]
    right_probs = [value / right_total for value in right_values]
    midpoint = [
        0.5 * (left_value + right_value)
        for left_value, right_value in zip(left_probs, right_probs, strict=True)
    ]

    def _kl_divergence(source: Sequence[float], target: Sequence[float]) -> float:
        """
        Compute the Kullback-Leibler divergence D(source || target) between two discrete probability distributions using base-2 logarithms.
        
        Both inputs represent non-negative probability-like weights over the same finite support; zero elements in `source` or `target` are ignored in the summation. The function returns the divergence in bits.
        
        Parameters:
            source (Sequence[float]): Source distribution weights (must correspond elementwise to `target`).
            target (Sequence[float]): Target distribution weights (must correspond elementwise to `source`).
        
        Returns:
            float: The KL divergence D(source || target) measured in base-2 units (bits).
        
        Raises:
            ValueError: If `source` and `target` have different lengths.
        """
        if len(source) != len(target):
            raise ValueError("Probability distributions must have the same length.")
        total = 0.0
        for source_value, target_value in zip(source, target, strict=True):
            if source_value <= 0.0 or target_value <= 0.0:
                continue
            total += source_value * math.log2(source_value / target_value)
        return float(total)

    divergence = 0.5 * _kl_divergence(left_probs, midpoint) + 0.5 * _kl_divergence(
        right_probs,
        midpoint,
    )
    return _clamp_unit_interval(divergence)

def _normalize_counts(counts: Dict[str, int], *, total: int) -> Dict[str, float]:
    """
    Normalize integer counts into fractional proportions using the provided total.
    
    Parameters:
        counts (Dict[str, int]): Mapping of names to integer counts.
        total (int): Denominator used to normalize counts.
    
    Returns:
        Dict[str, float]: Mapping of the same names to their normalized fraction (count / total). If `total` is less than or equal to 0, returns all zeros.
    """
    if total <= 0:
        return {name: 0.0 for name in counts}
    return {
        name: float(value / total)
        for name, value in counts.items()
    }

def _normalize_distribution(values: Mapping[str, int]) -> Dict[str, float]:
    """
    Convert a mapping of counts into a normalized distribution that sums to 1.0.
    
    Counts are coerced to integers and keys are coerced to strings. If the sum of all counts is less than or equal to zero, every key is mapped to 0.0.
    
    Parameters:
        values (Mapping[str, int]): Mapping from keys to counts; counts will be converted to `int` before normalization.
    
    Returns:
        Dict[str, float]: Mapping from each key (as `str`) to its normalized fraction (`int(count) / total`), or `0.0` for every key if the total is <= 0.
    """
    total = sum(int(value) for value in values.values())
    if total <= 0:
        return {str(name): 0.0 for name in values}
    return {
        str(name): float(int(value) / total)
        for name, value in values.items()
    }

def _predator_type_threat(meta: Mapping[str, object], predator_type: str) -> float:
    """
    Get the threat value for a predator type from metadata.
    
    Parameters:
        meta (Mapping[str, object]): Metadata mapping that may contain a key named "<predator_type>_predator_threat".
        predator_type (str): Predator type label (e.g., "visual" or "olfactory") used to form the metadata key.
    
    Returns:
        float: Threat value coerced to float; `0.0` if the key is missing or the value is falsy.
    """
    return float(meta.get(f"{predator_type}_predator_threat", 0.0) or 0.0)

def _dominant_predator_type(meta: Mapping[str, object]) -> str:
    """
    Determine the dominant predator type label from diagnostic metadata.
    
    Examines `meta["dominant_predator_type_label"]` (case-insensitive) and returns it if it matches a known predator type. If no valid label is present, compares numeric threat values for "visual" and "olfactory" (read via keys like `"visual_predator_threat"` / `"olfactory_predator_threat"`) and returns the type with the larger threat. Returns an empty string when both threats are zero or absent.
    
    Parameters:
        meta (Mapping[str, object]): Diagnostic metadata that may include
            - "dominant_predator_type_label": a preferred label (string)
            - "<type>_predator_threat": numeric threat values for "visual" and "olfactory"
    
    Returns:
        The dominant predator type: "visual" or "olfactory", or an empty string if none is dominant.
    """
    label = str(meta.get("dominant_predator_type_label") or "").strip().lower()
    if label in PREDATOR_TYPE_NAMES:
        return label
    visual_threat = _predator_type_threat(meta, "visual")
    olfactory_threat = _predator_type_threat(meta, "olfactory")
    if visual_threat <= 0.0 and olfactory_threat <= 0.0:
        return ""
    return "olfactory" if olfactory_threat > visual_threat else "visual"

def _diagnostic_predator_distance(meta: Mapping[str, object]) -> int:
    """
    Extract the diagnostic predator distance from a metadata mapping.
    
    Reads the "diagnostic" mapping from `meta` and returns the value of
    "diagnostic_predator_dist" coerced to an integer. If the diagnostic
    mapping or the distance value is missing or not a number, returns 0.
    
    Parameters:
        meta (Mapping[str, object]): Metadata that may contain a "diagnostic" mapping.
    
    Returns:
        int: The diagnostic predator distance, or 0 when unavailable or invalid.
    """
    diagnostic = meta.get("diagnostic", {})
    if not isinstance(diagnostic, Mapping):
        return 0
    value = diagnostic.get("diagnostic_predator_dist", 0)
    if value in (None, ""):
        return 0
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(numeric_value):
        return 0
    return int(numeric_value)

def _diagnostic_predator_distance_for_type(
    meta: Mapping[str, object],
    predator_type: str,
    *,
    state: object,
) -> int:
    """
    Determine the nearest Manhattan distance from the agent to any predator of the given detection style.
    
    If the agent's integer grid coordinates (`state.x`, `state.y`) are missing or no matching predators with valid integer positions are found, falls back to the diagnostic predator distance stored in `meta`.
    
    Parameters:
        meta (Mapping[str, object]): Episode metadata; may contain a "predators" list of mappings and a fallback diagnostic distance.
        predator_type (str): Lowercase detection style to match (e.g., "visual" or "olfactory").
        state (object): Agent state object expected to expose numeric `x` and `y` attributes.
    
    Returns:
        int: The smallest Manhattan distance to any predator whose `profile.detection_style` matches `predator_type`, or the diagnostic fallback distance from `meta` when coordinates or matches are unavailable.
    """
    spider_x = getattr(state, "x", None)
    spider_y = getattr(state, "y", None)
    if spider_x is None or spider_y is None:
        return _diagnostic_predator_distance(meta)
    try:
        spider_x_int = int(spider_x)
        spider_y_int = int(spider_y)
    except (TypeError, ValueError):
        return _diagnostic_predator_distance(meta)
    distances: list[int] = []
    predators = meta.get("predators", [])
    if isinstance(predators, list):
        for predator in predators:
            if not isinstance(predator, Mapping):
                continue
            profile = predator.get("profile", {})
            if not isinstance(profile, Mapping):
                continue
            detection_style = str(profile.get("detection_style") or "").strip().lower()
            if detection_style != predator_type:
                continue
            try:
                predator_x = int(predator["x"])
                predator_y = int(predator["y"])
            except (KeyError, TypeError, ValueError):
                continue
            distances.append(abs(spider_x_int - predator_x) + abs(spider_y_int - predator_y))
    if distances:
        return min(distances)
    return _diagnostic_predator_distance(meta)

def _first_active_predator_type(active: Mapping[str, object]) -> str:
    """
    Selects the first predator type key from the active mapping that matches the known predator type order.
    
    Parameters:
        active (Mapping[str, object]): Mapping whose keys may include predator type names.
    
    Returns:
        str: The first predator type (one of PREDATOR_TYPE_NAMES) found in `active`, or an empty string if none are present.
    """
    for predator_type in PREDATOR_TYPE_NAMES:
        if predator_type in active:
            return predator_type
    return ""

def _coerce_grid_coordinate(value: object) -> int | None:
    """Return an integer grid coordinate, or None when the value is malformed."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _contact_predator_types(
    meta: Mapping[str, object],
    *,
    state: object,
) -> list[str]:
    """
    Identify predator detection styles present at the agent's integer grid location.
    
    Scans meta["predators"] for predators whose integer (x, y) coordinates match the agent state's x and y and returns the unique detection-style labels (normalized to lowercase) that are members of PREDATOR_TYPE_NAMES. If the state does not expose x/y, or no predators occupy the agent's cell, falls back to the dominant predator type from meta; returns a single-element list with that type when available, otherwise returns an empty list.
    
    Parameters:
        meta (Mapping[str, object]): Metadata potentially containing a "predators" list and fields used to determine the dominant predator type.
        state (object): Agent state expected to expose numeric attributes `x` and `y`.
    
    Returns:
        list[str]: List of predator type labels present at the agent's integer grid cell, or a single-element list containing the dominant predator type, or an empty list if no type can be determined.
    """
    spider_x = getattr(state, "x", None)
    spider_y = getattr(state, "y", None)
    spider_grid_x = _coerce_grid_coordinate(spider_x)
    spider_grid_y = _coerce_grid_coordinate(spider_y)
    if spider_grid_x is None or spider_grid_y is None:
        dominant_type = _dominant_predator_type(meta)
        return [dominant_type] if dominant_type else []
    contact_types: list[str] = []
    predators = meta.get("predators", [])
    if isinstance(predators, list):
        for predator in predators:
            if not isinstance(predator, Mapping):
                continue
            predator_x = _coerce_grid_coordinate(predator.get("x"))
            predator_y = _coerce_grid_coordinate(predator.get("y"))
            if predator_x is None or predator_y is None:
                continue
            if predator_x != spider_grid_x or predator_y != spider_grid_y:
                continue
            profile = predator.get("profile", {})
            if not isinstance(profile, Mapping):
                continue
            detection_style = str(profile.get("detection_style") or "").strip().lower()
            if detection_style in PREDATOR_TYPE_NAMES and detection_style not in contact_types:
                contact_types.append(detection_style)
    if contact_types:
        return contact_types
    dominant_type = _dominant_predator_type(meta)
    return [dominant_type] if dominant_type else []
