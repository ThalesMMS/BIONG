"""Shared benchmark dataclasses and metric-classification registries.

The uncertainty registry separates scientific benchmark metrics from descriptive
run metadata. Metrics in ``UNCERTAINTY_REQUIRED_METRICS`` should be reported
with seed-level uncertainty when they appear in benchmark tables. Fields in
``DESCRIPTIVE_ONLY_FIELDS`` are provenance, inventory, or diagnostic payloads
that should be preserved verbatim but not interpreted as estimands requiring
confidence intervals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


# Metric names and canonical families whose point estimates need uncertainty
# estimates for benchmark-of-record reporting.
UNCERTAINTY_REQUIRED_METRICS: set[str] = {
    "scenario_success_rate",
    "no_reflex_scenario_success_rate",
    "per_scenario_success_rate",
    "episode_success_rate",
    "mean_reward",
    "ablation_delta",
    "ablation_success_rate_delta",
    "learning_evidence_delta",
    "learning_evidence_success_rate_delta",
    "claim_test_primary_metric",
    "claim_test_primary_metric_delta",
    "claim_test_delta",
    "effect_size",
    "effect_size_metric",
    "effect_size_delta",
    "scenario_success_rate_delta",
    "predator_response_scenario_success_rate",
    "predator_response_scenario_success_rate_delta",
    "shelter_return_scenario_success_rate",
    "shelter_return_scenario_success_rate_delta",
    "robustness_score",
    "diagonal_score",
    "off_diagonal_score",
    "diagonal_minus_off_diagonal_score",
    "threat_response_off_diagonal_score",
    "specialization_score",
    "predator_type_specialization_score",
    "predator_type_specialization_delta",
    "visual_minus_olfactory_delta",
}

# Provenance and diagnostic fields that document how a run was produced. These
# are intentionally excluded from automatic uncertainty requirements.
DESCRIPTIVE_ONLY_FIELDS: set[str] = {
    "config_fingerprint",
    "config_fingerprints",
    "resolved_config",
    "budget_profile",
    "checkpoint_id",
    "checkpoint_ids",
    "checkpoint_selection",
    "command_metadata",
    "exact_seed_list",
    "seed_list",
    "seeds",
    "static_audit_inventory",
    "audit_inventory",
    "raw_diagnostic_count",
    "raw_diagnostic_counts",
    "trace_derived_diagnostic",
    "trace_derived_diagnostics",
    "trace_derived_implementation_diagnostic",
    "trace_derived_implementation_diagnostics",
    "implementation_diagnostic",
    "implementation_diagnostics",
}

_UNCERTAINTY_REQUIRED_SUFFIXES: tuple[str, ...] = (
    "_delta",
    "_effect_size",
    "_success_rate",
)

UNCERTAINTY_REQUIRED_SCORE_FAMILIES: set[str] = {
    "diagonal",
    "diagonal_minus_off_diagonal",
    "off_diagonal",
    "predator_type_specialization",
    "robustness",
    "specialization",
    "threat_response_off_diagonal",
}


@dataclass(frozen=True)
class SeedLevelResult:
    metric_name: str
    seed: int
    value: float
    condition: str
    scenario: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_name", str(self.metric_name))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "condition", str(self.condition))
        if self.scenario is not None:
            object.__setattr__(self, "scenario", str(self.scenario))

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "seed": self.seed,
            "value": self.value,
            "condition": self.condition,
            "scenario": self.scenario,
        }


@dataclass(frozen=True)
class UncertaintyEstimate:
    mean: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_seeds: int
    confidence_level: float
    seed_values: Sequence[float]

    def __post_init__(self) -> None:
        mean = float(self.mean)
        ci_lower = float(self.ci_lower)
        ci_upper = float(self.ci_upper)
        std_error = float(self.std_error)
        n_seeds = int(self.n_seeds)
        confidence_level = float(self.confidence_level)
        seed_values = tuple(float(value) for value in self.seed_values)

        if n_seeds < 0:
            raise ValueError("n_seeds must be non-negative.")
        if len(seed_values) != n_seeds:
            raise ValueError("seed_values length must match n_seeds.")
        if not 0.0 < confidence_level <= 1.0:
            raise ValueError("confidence_level must be in the interval (0, 1].")
        if std_error < 0.0:
            raise ValueError("std_error must be non-negative.")
        if ci_lower > ci_upper:
            raise ValueError("ci_lower must be less than or equal to ci_upper.")
        if not ci_lower <= mean <= ci_upper:
            raise ValueError("mean must lie within the confidence interval.")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "ci_lower", ci_lower)
        object.__setattr__(self, "ci_upper", ci_upper)
        object.__setattr__(self, "std_error", std_error)
        object.__setattr__(self, "n_seeds", n_seeds)
        object.__setattr__(self, "confidence_level", confidence_level)
        object.__setattr__(self, "seed_values", seed_values)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "std_error": self.std_error,
            "n_seeds": self.n_seeds,
            "confidence_level": self.confidence_level,
            "seed_values": list(self.seed_values),
        }


@dataclass(frozen=True)
class EffectSizeResult:
    raw_delta: float
    cohens_d: float
    magnitude_label: str
    reference_condition: str
    comparison_condition: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_delta", float(self.raw_delta))
        object.__setattr__(self, "cohens_d", float(self.cohens_d))
        object.__setattr__(self, "magnitude_label", str(self.magnitude_label))
        object.__setattr__(
            self,
            "reference_condition",
            str(self.reference_condition),
        )
        object.__setattr__(
            self,
            "comparison_condition",
            str(self.comparison_condition),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_delta": self.raw_delta,
            "cohens_d": self.cohens_d,
            "magnitude_label": self.magnitude_label,
            "reference_condition": self.reference_condition,
            "comparison_condition": self.comparison_condition,
        }


def requires_uncertainty(metric_name: str) -> bool:
    """Return whether a downstream table should require uncertainty for a metric."""

    normalized = str(metric_name)
    if normalized in UNCERTAINTY_REQUIRED_METRICS:
        return True
    if normalized in DESCRIPTIVE_ONLY_FIELDS:
        return False
    if normalized.endswith("_score"):
        family = normalized[: -len("_score")]
        return family in UNCERTAINTY_REQUIRED_SCORE_FAMILIES
    return normalized.endswith(_UNCERTAINTY_REQUIRED_SUFFIXES)


def register_score_family(name: str) -> None:
    """Register a score metric family that should require uncertainty."""

    family = str(name).removesuffix("_score")
    if not family:
        raise ValueError("score family name must be non-empty.")
    UNCERTAINTY_REQUIRED_SCORE_FAMILIES.add(family)


__all__ = [  # noqa: RUF022
    "DESCRIPTIVE_ONLY_FIELDS",
    "EffectSizeResult",
    "SeedLevelResult",
    "UNCERTAINTY_REQUIRED_METRICS",
    "UNCERTAINTY_REQUIRED_SCORE_FAMILIES",
    "UncertaintyEstimate",
    "register_score_family",
    "requires_uncertainty",
]
