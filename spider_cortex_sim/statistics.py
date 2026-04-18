"""Small, deterministic statistical helpers for benchmark reporting.

The benchmark-of-record path reports uncertainty over seed-level metric values.
This module deliberately uses only the Python standard library: bootstrap
intervals are percentile intervals from resampled seed means, and effect sizes
use the conventional pooled-sample-standard-deviation definition of Cohen's d.
All bootstrap resampling uses a local deterministic RNG so repeated reports are
reproducible and do not perturb callers that rely on the process-global RNG.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Any


_DEFAULT_CONFIDENCE_LEVEL = 0.95
_DEFAULT_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_RANDOM_SEED = 0


def bootstrap_confidence_interval(
    values: Sequence[float] | Iterable[float],
    confidence_level: float = _DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    random_seed: int = _BOOTSTRAP_RANDOM_SEED,
) -> tuple[float, float, float, float]:
    """
    Return mean, percentile bootstrap CI bounds, and bootstrap standard error.

    The confidence interval is computed by drawing ``n_resamples`` bootstrap
    samples with a deterministic local RNG seeded by ``random_seed``. Each
    sample has the same size as the original input, then the lower and upper
    percentile bounds are taken around the resampled means. The standard error
    is the sample standard deviation of those bootstrap means.
    """
    samples = _coerce_float_values(values, name="values")
    confidence = float(confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0.0 and 1.0.")
    resamples = int(n_resamples)
    if resamples < 1:
        raise ValueError("n_resamples must be at least 1.")

    estimate = _mean(samples)
    if len(samples) == 1:
        return estimate, estimate, estimate, 0.0

    rng = random.Random(random_seed)
    bootstrap_means = sorted(
        _mean(rng.choices(samples, k=len(samples))) for _ in range(resamples)
    )
    tail_probability = (1.0 - confidence) / 2.0
    ci_lower = _percentile(bootstrap_means, tail_probability)
    ci_upper = _percentile(bootstrap_means, 1.0 - tail_probability)
    std_error = _sample_std(bootstrap_means)
    return estimate, ci_lower, ci_upper, std_error


def cohens_d(
    group_a: Sequence[float] | Iterable[float],
    group_b: Sequence[float] | Iterable[float],
) -> tuple[float, str]:
    """
    Return Cohen's d for group_a minus group_b and a magnitude label.

    Labels use common absolute-value thresholds: negligible < 0.2, small < 0.5,
    medium < 0.8, and large otherwise. If both groups have no within-group
    variance and different means, the standardized effect is undefined; the
    function returns 0.0 with the explicit "undefined" label to stay JSON-safe.
    """
    first = _coerce_float_values(group_a, name="group_a")
    second = _coerce_float_values(group_b, name="group_b")
    delta = _mean(first) - _mean(second)
    pooled_std = _pooled_sample_std(first, second)
    if pooled_std == 0.0:
        if delta == 0.0:
            return 0.0, "negligible"
        return 0.0, "undefined"
    effect_size = delta / pooled_std
    return effect_size, _magnitude_label(effect_size)


def aggregate_seed_results(
    seed_value_pairs: Iterable[tuple[int, float]],
    confidence_level: float = _DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """
    Aggregate seed-level metric values into point and interval estimates.

    The returned seed_values list preserves the input seed association for
    downstream benchmark-of-record tables while keeping the shape JSON-safe.
    ``std`` is the sample standard deviation of the original seed values, while
    CI bounds come from the deterministic bootstrap over those same values.
    """
    pairs = tuple((int(seed), float(value)) for seed, value in seed_value_pairs)
    if not pairs:
        raise ValueError("seed_value_pairs must contain at least one pair.")

    values = tuple(value for _, value in pairs)
    mean, ci_lower, ci_upper, _std_error = bootstrap_confidence_interval(
        values,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )
    return {
        "mean": mean,
        "std": _sample_std(values),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_seeds": len(pairs),
        "seed_values": [
            {"seed": seed, "value": value}
            for seed, value in pairs
        ],
    }


def _coerce_float_values(
    values: Sequence[float] | Iterable[float],
    *,
    name: str,
) -> tuple[float, ...]:
    coerced = tuple(float(value) for value in values)
    if not coerced:
        raise ValueError(f"{name} must contain at least one value.")
    return coerced


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    estimate = _mean(values)
    variance = sum((value - estimate) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _pooled_sample_std(
    group_a: Sequence[float],
    group_b: Sequence[float],
) -> float:
    degrees_of_freedom = len(group_a) + len(group_b) - 2
    if degrees_of_freedom <= 0:
        return 0.0
    mean_a = _mean(group_a)
    mean_b = _mean(group_b)
    sum_squares = sum((value - mean_a) ** 2 for value in group_a)
    sum_squares += sum((value - mean_b) ** 2 for value in group_b)
    return math.sqrt(sum_squares / degrees_of_freedom)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0.0 and 1.0.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = quantile * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return (
        float(sorted_values[lower_index]) * (1.0 - fraction)
        + float(sorted_values[upper_index]) * fraction
    )


def _magnitude_label(effect_size: float) -> str:
    magnitude = abs(effect_size)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


__all__ = [
    "aggregate_seed_results",
    "bootstrap_confidence_interval",
    "cohens_d",
]
