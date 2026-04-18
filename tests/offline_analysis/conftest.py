from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from spider_cortex_sim.offline_analysis.constants import (
    SHAPING_DEPENDENCE_WARNING_THRESHOLD,
)
from spider_cortex_sim.reward import (
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD as SURVIVAL_THRESHOLD,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKIN_SUMMARY = REPO_ROOT / "spider_summary.json"
CHECKIN_TRACE = REPO_ROOT / "spider_trace.jsonl"

SHAPING_GAP_EPSILON = 0.01
LARGE_SHAPING_GAP = SHAPING_DEPENDENCE_WARNING_THRESHOLD + SHAPING_GAP_EPSILON
SMALL_SHAPING_GAP = max(
    SHAPING_DEPENDENCE_WARNING_THRESHOLD - SHAPING_GAP_EPSILON,
    0.0,
)
EPISODE_SHAPING_GAP = SHAPING_DEPENDENCE_WARNING_THRESHOLD + (2 * SHAPING_GAP_EPSILON)
MEAN_REWARD_SHAPING_GAP = SHAPING_DEPENDENCE_WARNING_THRESHOLD + (3 * SHAPING_GAP_EPSILON)


def build_shaping_reward_audit_summary(
    scenario_success_rate_delta: float = LARGE_SHAPING_GAP,
) -> dict[str, object]:
    """
    Builds a structured reward-audit summary describing shaping-related changes between reward profiles.
    
    Parameters:
        scenario_success_rate_delta (float): The scenario success rate delta to use for the `classic` profile when compared to the minimal (`austere`) profile.
    
    Returns:
        dict[str, object]: A nested dictionary representing the reward audit, including:
            - `minimal_profile`: the selected minimal profile ("austere"),
            - `reward_components`: entries marking shaping risk and dispositions (e.g., `food_progress` removed),
            - `reward_profiles`: disposition summaries for `classic` and `austere`,
            - `comparison`: deltas versus the minimal profile and behavior survival metrics for scenarios (contains `deltas_vs_minimal` and `behavior_survival`).
    """
    return {
        "reward_audit": {
            "minimal_profile": "austere",
            "reward_components": {
                "food_progress": {
                    "category": "progress",
                    "shaping_risk": "high",
                    "shaping_disposition": "removed",
                    "disposition_rationale": "Zeroed in the austere profile.",
                }
            },
            "reward_profiles": {
                "classic": {
                    "disposition_summary": {
                        "removed": {"total_weight_proxy": 1.2}
                    }
                },
                "austere": {
                    "disposition_summary": {
                        "removed": {"total_weight_proxy": 0.0}
                    }
                },
            },
            "comparison": {
                "minimal_profile": "austere",
                "deltas_vs_minimal": {
                    "classic": {
                        "scenario_success_rate_delta": scenario_success_rate_delta,
                        "episode_success_rate_delta": EPISODE_SHAPING_GAP,
                        "mean_reward_delta": MEAN_REWARD_SHAPING_GAP,
                    },
                    "austere": {
                        "scenario_success_rate_delta": 0.0,
                        "episode_success_rate_delta": 0.0,
                        "mean_reward_delta": 0.0,
                    },
                },
                "behavior_survival": {
                    "minimal_profile": "austere",
                    "survival_threshold": SURVIVAL_THRESHOLD,
                    "scenario_count": 1,
                    "surviving_scenario_count": 1,
                    "survival_rate": 1.0,
                    "scenarios": {
                        "night_rest": {
                            "austere_success_rate": 1.0,
                            "survives": True,
                            "episodes": 1,
                        }
                    },
                },
            },
        }
    }


def uncertainty_payload(
    mean: float,
    lower: float,
    upper: float,
    seed_values: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """
    Builds an uncertainty payload describing a metric estimate, its confidence interval, and seed-level information.
    
    Parameters:
        mean (float): Estimated mean of the metric.
        lower (float): Confidence-interval lower bound.
        upper (float): Confidence-interval upper bound.
        seed_values (list[dict[str, object]] | None): Per-seed metric records to
            include. If omitted, defaults to seeds 0 and 1 using lower and upper.
    
    Returns:
        dict[str, object]: Mapping with keys:
            - `mean`: the provided mean,
            - `ci_lower`: the provided lower bound,
            - `ci_upper`: the provided upper bound,
            - `std_error`: standard error (fixed at 0.05),
            - `n_seeds`: number of seed values included,
            - `confidence_level`: confidence level (fixed at 0.95),
            - `seed_values`: list of seed records used.
    """
    raw_values = (
        seed_values
        if seed_values is not None
        else [{"seed": 0, "value": lower}, {"seed": 1, "value": upper}]
    )
    values: list[dict[str, object]] = []
    for index, item in enumerate(raw_values):
        if isinstance(item, Mapping):
            seed = item.get("seed", index)
            value = item.get("value")
        else:
            seed = index
            value = item
        values.append({"seed": int(seed), "value": float(value)})
    return {
        "mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "std_error": 0.05,
        "n_seeds": len(values),
        "confidence_level": 0.95,
        "seed_values": values,
    }


def uncertainty_condition(
    name: str,
    values: list[float],
    *,
    scenario: str = "night_rest",
) -> dict[str, object]:
    """
    Builds a structured condition payload containing per-seed records, aggregated summary metrics, and uncertainty payloads for a named evaluation condition.
    
    Parameters:
        name (str): Identifier for the condition (e.g., "trained", "random_init").
        values (list[float]): Per-seed scenario success rates (fractional values, e.g., between 0 and 1) used to compute summary metrics and seed-level entries.
        scenario (str): Scenario name to place under the `suite` mapping and to tag seed-level entries (default "night_rest").
    
    Returns:
        dict[str, object]: A mapping with the following keys:
            - "summary": Aggregated metrics with keys `scenario_success_rate`, `episode_success_rate` (both equal to the mean of `values`), and `mean_reward` (mean scaled by 10.0).
            - "suite": A mapping from `scenario` to a dict containing `success_rate` and an `uncertainty` payload for the scenario success rate.
            - "seed_level": A list of per-seed records documenting `metric_name`, `seed`, `value`, `condition`, and scenario-tagged variants.
            - "uncertainty": Uncertainty payloads for `scenario_success_rate`, `episode_success_rate`, and `mean_reward` (the latter scaled consistently by 10.0).
    """
    if not values:
        raise ValueError("uncertainty_condition requires at least one value")
    mean = sum(values) / len(values)
    seed_level = [
        {
            "metric_name": "scenario_success_rate",
            "seed": index + 1,
            "value": value,
            "condition": name,
        }
        for index, value in enumerate(values)
    ]
    seed_level.extend(
        {
            "metric_name": "scenario_success_rate",
            "seed": index + 1,
            "value": value,
            "condition": name,
            "scenario": scenario,
        }
        for index, value in enumerate(values)
    )
    return {
        "summary": {
            "scenario_success_rate": mean,
            "episode_success_rate": mean,
            "mean_reward": mean * 10.0,
        },
        "suite": {
            scenario: {
                "success_rate": mean,
                "uncertainty": {
                    "success_rate": uncertainty_payload(
                        mean,
                        min(values),
                        max(values),
                        values,
                    )
                },
            }
        },
        "seed_level": seed_level,
        "uncertainty": {
            "scenario_success_rate": uncertainty_payload(
                mean,
                min(values),
                max(values),
                values,
            ),
            "episode_success_rate": uncertainty_payload(
                mean,
                min(values),
                max(values),
                values,
            ),
            "mean_reward": uncertainty_payload(
                mean * 10.0,
                min(values) * 10.0,
                max(values) * 10.0,
                [value * 10.0 for value in values],
            ),
        },
    }


def build_uncertainty_summary() -> dict[str, object]:
    """
    Builds a comprehensive uncertainty summary describing evaluation, learning evidence, ablations, and claim tests for multiple training conditions.
    
    Returns:
        summary (dict[str, object]): Nested dictionary containing:
            - config: experiment budget/profile metadata.
            - checkpointing: selection metric and strategy.
            - behavior_evaluation:
                - suite: per-scenario suite results (copied from the trained condition).
                - summary: aggregated metrics for the trained condition.
                - learning_evidence: reference condition, condition payloads, point estimate deltas between conditions, and per-delta uncertainty payloads.
                - ablations: reference variant and variant payloads.
                - claim_tests: claim entries with status, reference/comparison values, deltas, effect sizes, cohen's d, qualitative effect magnitude, uncertainties, primary metric, and evaluated scenarios.
    """
    trained = uncertainty_condition("trained_without_reflex_support", [0.8, 1.0])
    random_init = uncertainty_condition("random_init", [0.2, 0.4])
    reflex_only = uncertainty_condition("reflex_only", [0.4, 0.6])
    modular_full = uncertainty_condition("modular_full", [0.7, 0.9])
    monolithic = uncertainty_condition("monolithic_policy", [0.3, 0.5])
    return {
        "config": {
            "budget": {
                "profile": "paper",
                "benchmark_strength": "paper",
                "resolved": {},
            }
        },
        "checkpointing": {"selection": "best", "metric": "scenario_success_rate"},
        "behavior_evaluation": {
            "suite": trained["suite"],
            "summary": trained["summary"],
            "learning_evidence": {
                "reference_condition": "trained_without_reflex_support",
                "conditions": {
                    "trained_without_reflex_support": trained,
                    "random_init": random_init,
                    "reflex_only": reflex_only,
                },
                "evidence_summary": {
                    "trained_vs_random_init": {
                        "scenario_success_rate_delta": 0.6,
                        "episode_success_rate_delta": 0.6,
                        "mean_reward_delta": 6.0,
                    },
                    "trained_vs_reflex_only": {
                        "scenario_success_rate_delta": 0.4,
                        "episode_success_rate_delta": 0.4,
                        "mean_reward_delta": 4.0,
                    },
                    "uncertainty": {
                        "trained_vs_random_init": {
                            "scenario_success_rate_delta": uncertainty_payload(
                                0.6,
                                0.4,
                                0.8,
                                [0.6, 0.6],
                            ),
                            "episode_success_rate_delta": uncertainty_payload(
                                0.6,
                                0.4,
                                0.8,
                                [0.6, 0.6],
                            ),
                            "mean_reward_delta": uncertainty_payload(
                                6.0,
                                4.0,
                                8.0,
                                [6.0, 6.0],
                            ),
                        },
                        "trained_vs_reflex_only": {
                            "scenario_success_rate_delta": uncertainty_payload(
                                0.4,
                                0.2,
                                0.6,
                                [0.4, 0.4],
                            ),
                            "episode_success_rate_delta": uncertainty_payload(
                                0.4,
                                0.2,
                                0.6,
                                [0.4, 0.4],
                            ),
                            "mean_reward_delta": uncertainty_payload(
                                4.0,
                                2.0,
                                6.0,
                                [4.0, 4.0],
                            ),
                        },
                    },
                },
            },
            "ablations": {
                "reference_variant": "modular_full",
                "variants": {
                    "modular_full": modular_full,
                    "monolithic_policy": monolithic,
                },
            },
            "claim_tests": {
                "claims": {
                    "learning_without_privileged_signals": {
                        "status": "passed",
                        "passed": True,
                        "reference_value": 0.3,
                        "comparison_values": {
                            "trained_without_reflex_support": 0.9,
                        },
                        "delta": {
                            "trained_without_reflex_support": 0.6,
                        },
                        "effect_size": {
                            "trained_without_reflex_support": 0.6,
                        },
                        "reference_uncertainty": uncertainty_payload(
                            0.3,
                            0.2,
                            0.4,
                            [0.2, 0.4],
                        ),
                        "comparison_uncertainty": {
                            "trained_without_reflex_support": uncertainty_payload(
                                0.9,
                                0.8,
                                1.0,
                                [0.8, 1.0],
                            )
                        },
                        "delta_uncertainty": {
                            "trained_without_reflex_support": uncertainty_payload(
                                0.6,
                                0.4,
                                0.8,
                                [0.6, 0.6],
                            )
                        },
                        "effect_size_uncertainty": {
                            "trained_without_reflex_support": uncertainty_payload(
                                0.6,
                                0.4,
                                0.8,
                                [0.6, 0.6],
                            )
                        },
                        "cohens_d": {
                            "trained_without_reflex_support": 4.0,
                        },
                        "effect_magnitude": {
                            "trained_without_reflex_support": "large",
                        },
                        "primary_metric": "scenario_success_rate",
                        "scenarios_evaluated": ["night_rest"],
                    }
                }
            },
        },
    }


def assert_uncertainty_fields(testcase: object, row: dict[str, object]) -> None:
    """
    Assert required uncertainty fields are present and valid in a metric row.
    
    Performs the following checks using the provided test-case assertion methods:
    - Each of `value`, `ci_lower`, `ci_upper`, `std_error`, `n_seeds`, and `confidence_level` exists in `row`.
    - `row["value"]` is not None.
    - `row["confidence_level"]` is exactly 0.95.
    
    Parameters:
        testcase (object): A test-case instance exposing assertion methods (e.g., `assertIn`, `assertIsNotNone`, `assertEqual`).
        row (dict[str, object]): Mapping representing an uncertainty record to validate.
    """
    for key in (
        "value",
        "ci_lower",
        "ci_upper",
        "std_error",
        "n_seeds",
        "confidence_level",
    ):
        testcase.assertIn(key, row)
    testcase.assertIsNotNone(row["value"])
    testcase.assertEqual(row["confidence_level"], 0.95)
