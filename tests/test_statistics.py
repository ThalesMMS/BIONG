from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

from spider_cortex_sim.benchmark_types import (
    DESCRIPTIVE_ONLY_FIELDS,
    EffectSizeResult,
    SeedLevelResult,
    UNCERTAINTY_REQUIRED_METRICS,
    UncertaintyEstimate,
    register_score_family,
    requires_uncertainty,
)
from spider_cortex_sim.statistics import (
    aggregate_seed_results,
    bootstrap_confidence_interval,
    cohens_d,
)


class BootstrapConfidenceIntervalTest(unittest.TestCase):
    def test_single_value_returns_degenerate_interval(self) -> None:
        mean, ci_lower, ci_upper, std_error = bootstrap_confidence_interval(
            [0.75],
            confidence_level=0.95,
            n_resamples=10,
        )
        self.assertEqual(mean, 0.75)
        self.assertEqual(ci_lower, 0.75)
        self.assertEqual(ci_upper, 0.75)
        self.assertEqual(std_error, 0.0)

    def test_identical_values_return_degenerate_interval(self) -> None:
        mean, ci_lower, ci_upper, std_error = bootstrap_confidence_interval(
            [0.5, 0.5, 0.5],
            confidence_level=0.95,
            n_resamples=50,
        )
        self.assertEqual(mean, 0.5)
        self.assertEqual(ci_lower, 0.5)
        self.assertEqual(ci_upper, 0.5)
        self.assertEqual(std_error, 0.0)

    def test_spread_values_return_interval_around_mean(self) -> None:
        mean, ci_lower, ci_upper, std_error = bootstrap_confidence_interval(
            [0.0, 0.5, 1.0],
            confidence_level=0.95,
            n_resamples=100,
        )
        self.assertAlmostEqual(mean, 0.5)
        self.assertLessEqual(ci_lower, mean)
        self.assertGreaterEqual(ci_upper, mean)
        self.assertGreater(std_error, 0.0)

    def test_synthetic_centered_data_ci_contains_known_center(self) -> None:
        mean, ci_lower, ci_upper, std_error = bootstrap_confidence_interval(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            confidence_level=0.95,
            n_resamples=500,
        )
        self.assertAlmostEqual(mean, 0.0)
        self.assertLess(ci_lower, 0.0)
        self.assertGreater(ci_upper, 0.0)
        self.assertGreater(std_error, 0.0)

    def test_empty_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_confidence_interval([], confidence_level=0.95, n_resamples=10)


class CohensDTest(unittest.TestCase):
    def test_equal_groups_are_negligible(self) -> None:
        effect_size, label = cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(effect_size, 0.0)
        self.assertEqual(label, "negligible")

    def test_separated_groups_report_signed_large_effect(self) -> None:
        effect_size, label = cohens_d([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(effect_size, 1.0)
        self.assertEqual(label, "large")

    def test_small_effect_label(self) -> None:
        effect_size, label = cohens_d([-0.7, 0.3, 1.3], [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(effect_size, 0.3)
        self.assertEqual(label, "small")

    def test_medium_effect_label(self) -> None:
        effect_size, label = cohens_d([-0.4, 0.6, 1.6], [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(effect_size, 0.6)
        self.assertEqual(label, "medium")

    def test_large_effect_label(self) -> None:
        effect_size, label = cohens_d([0.0, 1.0, 2.0], [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(effect_size, 1.0)
        self.assertEqual(label, "large")

    def test_zero_variance_difference_is_marked_undefined(self) -> None:
        effect_size, label = cohens_d([2.0, 2.0], [1.0, 1.0])
        self.assertEqual(effect_size, 0.0)
        self.assertEqual(label, "undefined")


class AggregateSeedResultsTest(unittest.TestCase):
    def test_returns_seed_preserving_summary(self) -> None:
        result = aggregate_seed_results(
            [(11, 0.25), (12, 0.5), (13, 0.75)],
            confidence_level=0.95,
            n_resamples=100,
        )
        self.assertEqual(
            set(result),
            {"mean", "std", "ci_lower", "ci_upper", "n_seeds", "seed_values"},
        )
        self.assertAlmostEqual(result["mean"], 0.5)
        self.assertAlmostEqual(result["std"], 0.25)
        self.assertLessEqual(result["ci_lower"], result["mean"])
        self.assertGreaterEqual(result["ci_upper"], result["mean"])
        self.assertEqual(result["n_seeds"], 3)
        self.assertEqual(
            result["seed_values"],
            [
                {"seed": 11, "value": 0.25},
                {"seed": 12, "value": 0.5},
                {"seed": 13, "value": 0.75},
            ],
        )
        json.dumps(result)


class BenchmarkTypesTest(unittest.TestCase):
    def test_seed_level_result_is_frozen_and_serializable(self) -> None:
        result = SeedLevelResult(
            metric_name="scenario_success_rate",
            seed="7",  # type: ignore[arg-type]
            value="0.8",  # type: ignore[arg-type]
            condition=3,  # type: ignore[arg-type]
            scenario=4,  # type: ignore[arg-type]
        )
        self.assertEqual(result.seed, 7)
        self.assertEqual(result.value, 0.8)
        self.assertEqual(result.condition, "3")
        self.assertEqual(result.scenario, "4")
        self.assertEqual(
            result.to_dict(),
            {
                "metric_name": "scenario_success_rate",
                "seed": 7,
                "value": 0.8,
                "condition": "3",
                "scenario": "4",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            result.value = 0.0  # type: ignore[misc]

    def test_uncertainty_estimate_serializes_seed_values_as_list(self) -> None:
        estimate = UncertaintyEstimate(
            mean="0.5",  # type: ignore[arg-type]
            ci_lower=0.25,
            ci_upper=0.75,
            std_error=0.1,
            n_seeds="3",  # type: ignore[arg-type]
            confidence_level="0.95",  # type: ignore[arg-type]
            seed_values=[0.25, "0.5", 0.75],  # type: ignore[list-item]
        )
        self.assertEqual(estimate.seed_values, (0.25, 0.5, 0.75))
        self.assertEqual(
            estimate.to_dict(),
            {
                "mean": 0.5,
                "ci_lower": 0.25,
                "ci_upper": 0.75,
                "std_error": 0.1,
                "n_seeds": 3,
                "confidence_level": 0.95,
                "seed_values": [0.25, 0.5, 0.75],
            },
        )
        json.dumps(estimate.to_dict())

    def test_effect_size_result_is_serializable(self) -> None:
        result = EffectSizeResult(
            raw_delta="0.2",  # type: ignore[arg-type]
            cohens_d="0.6",  # type: ignore[arg-type]
            magnitude_label=5,  # type: ignore[arg-type]
            reference_condition="random_init",
            comparison_condition="trained_without_reflex_support",
        )
        self.assertEqual(
            result.to_dict(),
            {
                "raw_delta": 0.2,
                "cohens_d": 0.6,
                "magnitude_label": "5",
                "reference_condition": "random_init",
                "comparison_condition": "trained_without_reflex_support",
            },
        )
        json.dumps(result.to_dict())

    def test_metric_registries_cover_required_and_descriptive_outputs(self) -> None:
        self.assertIn("scenario_success_rate", UNCERTAINTY_REQUIRED_METRICS)
        self.assertIn("robustness_score", UNCERTAINTY_REQUIRED_METRICS)
        self.assertIn("visual_minus_olfactory_delta", UNCERTAINTY_REQUIRED_METRICS)
        self.assertIn("config_fingerprint", DESCRIPTIVE_ONLY_FIELDS)
        self.assertIn("checkpoint_ids", DESCRIPTIVE_ONLY_FIELDS)
        self.assertTrue(requires_uncertainty("scenario_success_rate"))
        self.assertTrue(requires_uncertainty("custom_ablation_delta"))
        self.assertTrue(requires_uncertainty("custom_success_rate"))
        self.assertFalse(requires_uncertainty("unregistered_score"))
        register_score_family("registered_custom")
        self.assertTrue(requires_uncertainty("registered_custom_score"))
        self.assertFalse(requires_uncertainty("config_fingerprint"))
        self.assertFalse(requires_uncertainty("raw_diagnostic_count"))

    def test_uncertainty_estimate_rejects_invalid_invariants(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed_values length"):
            UncertaintyEstimate(
                mean=0.5,
                ci_lower=0.0,
                ci_upper=1.0,
                std_error=0.1,
                n_seeds=2,
                confidence_level=0.95,
                seed_values=[0.5],
            )
        with self.assertRaisesRegex(ValueError, "confidence_level"):
            UncertaintyEstimate(
                mean=0.5,
                ci_lower=0.0,
                ci_upper=1.0,
                std_error=0.1,
                n_seeds=1,
                confidence_level=1.5,
                seed_values=[0.5],
            )
        with self.assertRaisesRegex(ValueError, "mean"):
            UncertaintyEstimate(
                mean=1.5,
                ci_lower=0.0,
                ci_upper=1.0,
                std_error=0.1,
                n_seeds=1,
                confidence_level=0.95,
                seed_values=[1.5],
            )


if __name__ == "__main__":
    unittest.main()
