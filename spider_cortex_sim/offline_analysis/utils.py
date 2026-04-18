from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


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
    """
    Convert a value to a finite float or return None.
    
    Parameters:
        value (object): The input to coerce; common numeric types and numeric strings are accepted.
    
    Returns:
        float | None: A finite float parsed from `value`, or `None` if `value` is `None`, cannot be converted to a float, or results in a non-finite value (`NaN`, `inf`, `-inf`).
    """
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
    """
    Convert common truthy representations to a boolean.
    
    If `value` is already a `bool` it is returned unchanged. `None` yields `False`. Other inputs are converted to string, stripped, lowercased, and considered true when equal to one of: "1", "true", "yes", "y", "on".
    
    Parameters:
        value (object): Input to interpret as boolean; may be a bool, None, string, number, or other object.
    
    Returns:
        bool: `True` if `value` represents a truthy value ("1", "true", "yes", "y", "on"), `False` otherwise.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _safe_divide(numerator: float, denominator: float) -> float:
    """
    Divide numerator by denominator, returning 0.0 when the denominator is less than or equal to zero.
    
    Parameters:
        numerator (float): The dividend.
        denominator (float): The divisor; treated as invalid when less than or equal to zero.
    
    Returns:
        result (float): Quotient of numerator divided by denominator, or 0.0 if denominator is less than or equal to zero.
    """
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
    Selects the module name with the highest numeric score from the given mapping.
    
    Scores are coerced to finite floats before comparison; if `modules` is empty this returns an empty string.
    
    Returns:
        The module name with the highest coerced numeric score, or an empty string if `modules` is empty.
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
    """
    Format an optional numeric metric for display.
    
    Parameters:
        value (float | None): Metric value to format; `None` represents a missing value.
    
    Returns:
        str: `"—"` if `value` is `None`, otherwise the value formatted with two decimal places (e.g., `"1.23"`).
    """
    return "—" if value is None else f"{value:.2f}"
