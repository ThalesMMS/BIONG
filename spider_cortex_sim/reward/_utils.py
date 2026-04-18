"""Shared reward-policy utility helpers."""

from __future__ import annotations

import math


def policy_float(value: object, default: float = 0.0) -> float:
    """
    Convert an arbitrary value to a finite float suitable for policy fields.

    Parameters:
        value (object): Value to coerce to float.
        default (float): Fallback value returned when coercion fails or yields NaN/inf.

    Returns:
        float: The finite float result, or `default` if conversion fails or the value is not finite.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result
