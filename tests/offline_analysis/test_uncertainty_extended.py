"""Extended tests for spider_cortex_sim.offline_analysis.uncertainty.

The basic tests in test_uncertainty.py cover _payload_metric_seed_values and
_delta_uncertainty_from_seed_values. This file covers the remaining helpers
extracted from the monolith into the new uncertainty.py module.
"""
from __future__ import annotations

import math
import unittest

from spider_cortex_sim.offline_analysis.uncertainty import (
    _bootstrap_distribution_fields,
    _ci_row_fields,
    _cohens_d_row,
    _is_zero_reflex_scale,
    _mean_or_none,
    _payload_has_zero_reflex_scale,
    _payload_metric_seed_items,
    _payload_uncertainty,
    _percentile,
    _primary_benchmark_scenario_success,
    _primary_benchmark_source_payload,
    _sample_std,
    _seed_key,
    _seed_value_items,
    _uncertainty_or_empty,
    _values_from_seed_items,
)


class SeedKeyTest(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(_seed_key(None))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_seed_key(""))

    def test_string_seed_returned_as_is(self) -> None:
        self.assertEqual(_seed_key("seed-42"), "seed-42")

    def test_integer_seed_converted_to_string(self) -> None:
        self.assertEqual(_seed_key(7), "7")

    def test_zero_integer_is_not_none(self) -> None:
        # 0 is not None or "" so should be "0"
        self.assertEqual(_seed_key(0), "0")


class SeedValueItemsTest(unittest.TestCase):
    def test_empty_list_returns_empty(self) -> None:
        self.assertEqual(_seed_value_items([]), [])

    def test_plain_float_creates_none_seed_item(self) -> None:
        items = _seed_value_items([0.5])
        self.assertEqual(len(items), 1)
        seed, value = items[0]
        self.assertIsNone(seed)
        self.assertAlmostEqual(value, 0.5)

    def test_mapping_with_seed_and_value(self) -> None:
        items = _seed_value_items([{"seed": "s1", "value": 0.8}])
        self.assertEqual(len(items), 1)
        seed, value = items[0]
        self.assertEqual(seed, "s1")
        self.assertAlmostEqual(value, 0.8)

    def test_nested_summary_mapping(self) -> None:
        items = _seed_value_items(
            [
                {"summary": {"seed": "s4", "value": 0.6}},
                {"summary": {"seed": "bad", "value": "not_a_float"}},
            ]
        )

        self.assertEqual(items, [("s4", 0.6)])

    def test_tuple_pair(self) -> None:
        items = _seed_value_items([("s2", 0.3)])
        self.assertEqual(len(items), 1)
        seed, value = items[0]
        self.assertEqual(seed, "s2")
        self.assertAlmostEqual(value, 0.3)

    def test_list_pair(self) -> None:
        items = _seed_value_items([["s3", 0.7]])
        seed, value = items[0]
        self.assertEqual(seed, "s3")
        self.assertAlmostEqual(value, 0.7)

    def test_non_numeric_value_skipped(self) -> None:
        items = _seed_value_items([{"seed": "s1", "value": "not_a_float"}])
        self.assertEqual(items, [])

    def test_none_value_skipped(self) -> None:
        items = _seed_value_items([{"seed": "s1", "value": None}])
        self.assertEqual(items, [])

    def test_nan_value_skipped(self) -> None:
        items = _seed_value_items([{"seed": "s1", "value": float("nan")}])
        self.assertEqual(items, [])

    def test_mixed_items(self) -> None:
        inputs = [
            {"seed": "s1", "value": 0.5},
            0.3,  # plain float
            {"seed": "s2", "value": "bad"},  # skipped
        ]
        items = _seed_value_items(inputs)
        self.assertEqual(len(items), 2)


class ValuesFromSeedItemsTest(unittest.TestCase):
    def test_extracts_values(self) -> None:
        items = [("s1", 0.5), ("s2", 0.8)]
        values = _values_from_seed_items(items)
        self.assertEqual(values, [0.5, 0.8])

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(_values_from_seed_items([]), [])

    def test_none_seed_value_included(self) -> None:
        items = [(None, 0.4)]
        values = _values_from_seed_items(items)
        self.assertAlmostEqual(values[0], 0.4)


class SampleStdTest(unittest.TestCase):
    def test_empty_returns_zero(self) -> None:
        self.assertEqual(_sample_std([]), 0.0)

    def test_single_value_returns_zero(self) -> None:
        self.assertEqual(_sample_std([5.0]), 0.0)

    def test_two_identical_values_returns_zero(self) -> None:
        self.assertAlmostEqual(_sample_std([3.0, 3.0]), 0.0)

    def test_known_std(self) -> None:
        # std([2, 4, 4, 4, 5, 5, 7, 9]) = 2.0 (population), sample std = 2.138...
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        expected_sample_std = math.sqrt(
            sum((x - (sum(values) / len(values))) ** 2 for x in values) / (len(values) - 1)
        )
        self.assertAlmostEqual(_sample_std(values), expected_sample_std, places=6)

    def test_non_negative_result(self) -> None:
        result = _sample_std([1.0, 2.0, 3.0])
        self.assertGreaterEqual(result, 0.0)


class PercentileTest(unittest.TestCase):
    def test_empty_values_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted_values must be non-empty"):
            _percentile([], 0.5)

    def test_quantile_below_zero_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "quantile.*between 0 and 1"):
            _percentile([1.0, 2.0, 3.0], -0.1)

    def test_quantile_above_one_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "quantile.*between 0 and 1"):
            _percentile([1.0, 2.0, 3.0], 1.1)

    def test_single_value_returns_that_value(self) -> None:
        self.assertAlmostEqual(_percentile([5.0], 0.5), 5.0)

    def test_median_of_three(self) -> None:
        self.assertAlmostEqual(_percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_zero_quantile_returns_minimum(self) -> None:
        self.assertAlmostEqual(_percentile([1.0, 5.0, 10.0], 0.0), 1.0)

    def test_one_quantile_returns_maximum(self) -> None:
        self.assertAlmostEqual(_percentile([1.0, 5.0, 10.0], 1.0), 10.0)

    def test_interpolation_between_values(self) -> None:
        # 0.25 quantile of [1, 3] -> position = 0.25*(2-1) = 0.25, lower=1, upper=3
        # result = 1*(1-0.25) + 3*0.25 = 0.75 + 0.75 = 1.5
        self.assertAlmostEqual(_percentile([1.0, 3.0], 0.25), 1.5)

    def test_exact_index_returns_value(self) -> None:
        # 0.5 with 3 values -> position=1.0, exact index 1 -> value[1] = 2
        self.assertAlmostEqual(_percentile([1.0, 2.0, 3.0], 0.5), 2.0)


class BootstrapDistributionFieldsTest(unittest.TestCase):
    def test_empty_returns_empty_dict(self) -> None:
        self.assertEqual(_bootstrap_distribution_fields([]), {})

    def test_returns_ci_lower_upper_std_error_keys(self) -> None:
        result = _bootstrap_distribution_fields([0.1, 0.5, 0.9])
        for key in ("ci_lower", "ci_upper", "std_error"):
            self.assertIn(key, result)

    def test_ci_lower_le_ci_upper(self) -> None:
        result = _bootstrap_distribution_fields([0.2, 0.4, 0.6, 0.8])
        self.assertLessEqual(result["ci_lower"], result["ci_upper"])

    def test_single_value_returns_same_lower_and_upper(self) -> None:
        result = _bootstrap_distribution_fields([0.7])
        self.assertAlmostEqual(result["ci_lower"], 0.7)
        self.assertAlmostEqual(result["ci_upper"], 0.7)

    def test_std_error_is_non_negative(self) -> None:
        result = _bootstrap_distribution_fields([0.1, 0.5])
        self.assertGreaterEqual(result["std_error"], 0.0)


class UncertaintyOrEmptyTest(unittest.TestCase):
    def test_mapping_returned_as_is(self) -> None:
        m = {"ci_lower": 0.1}
        self.assertIs(_uncertainty_or_empty(m), m)

    def test_none_returns_empty_dict(self) -> None:
        result = _uncertainty_or_empty(None)
        self.assertEqual(dict(result), {})

    def test_list_returns_empty_dict(self) -> None:
        result = _uncertainty_or_empty([1, 2])
        self.assertEqual(dict(result), {})

    def test_string_returns_empty_dict(self) -> None:
        result = _uncertainty_or_empty("not a mapping")
        self.assertEqual(dict(result), {})


class CiRowFieldsTest(unittest.TestCase):
    def test_empty_uncertainty_with_value(self) -> None:
        result = _ci_row_fields({}, value=0.7)
        self.assertAlmostEqual(result["value"], 0.7)
        self.assertIsNone(result["ci_lower"])
        self.assertIsNone(result["ci_upper"])
        self.assertIsNone(result["std_error"])
        self.assertEqual(result["n_seeds"], 0)
        self.assertIsNone(result["confidence_level"])

    def test_full_uncertainty_payload(self) -> None:
        uncertainty = {
            "mean": 0.8,
            "ci_lower": 0.7,
            "ci_upper": 0.9,
            "std_error": 0.05,
            "n_seeds": 4,
            "confidence_level": 0.95,
        }
        result = _ci_row_fields(uncertainty)
        self.assertAlmostEqual(result["value"], 0.8)
        self.assertAlmostEqual(result["ci_lower"], 0.7)
        self.assertAlmostEqual(result["ci_upper"], 0.9)
        self.assertAlmostEqual(result["std_error"], 0.05)
        self.assertEqual(result["n_seeds"], 4)
        self.assertAlmostEqual(result["confidence_level"], 0.95)

    def test_explicit_value_overrides_mean(self) -> None:
        uncertainty = {"mean": 0.5, "ci_lower": 0.4, "ci_upper": 0.6}
        result = _ci_row_fields(uncertainty, value=0.99)
        self.assertAlmostEqual(result["value"], 0.99)

    def test_none_uncertainty_uses_value_parameter(self) -> None:
        result = _ci_row_fields(None, value=0.3)
        self.assertAlmostEqual(result["value"], 0.3)


class PayloadUncertaintyTest(unittest.TestCase):
    def test_non_mapping_payload_returns_empty(self) -> None:
        result = _payload_uncertainty(None, "scenario_success_rate")
        self.assertEqual(dict(result), {})

    def test_returns_metric_key_from_uncertainty(self) -> None:
        ci = {"ci_lower": 0.4, "ci_upper": 0.8}
        payload = {"uncertainty": {"scenario_success_rate": ci}}
        result = _payload_uncertainty(payload, "scenario_success_rate")
        self.assertIs(result, ci)

    def test_falls_back_to_success_rate_key(self) -> None:
        ci = {"ci_lower": 0.3, "ci_upper": 0.7}
        payload = {"uncertainty": {"success_rate": ci}}
        result = _payload_uncertainty(payload, "scenario_success_rate")
        self.assertIs(result, ci)

    def test_scenario_level_lookup(self) -> None:
        ci = {"ci_lower": 0.2, "ci_upper": 0.6}
        payload = {
            "suite": {
                "night_rest": {
                    "uncertainty": {"success_rate": ci}
                }
            }
        }
        result = _payload_uncertainty(payload, "scenario_success_rate", scenario="night_rest")
        self.assertIs(result, ci)

    def test_missing_uncertainty_returns_empty(self) -> None:
        result = _payload_uncertainty({}, "some_metric")
        self.assertEqual(dict(result), {})

    def test_non_mapping_uncertainty_nested_value_ignored(self) -> None:
        # uncertainty key exists but value is not a Mapping
        payload = {"uncertainty": {"scenario_success_rate": "not_a_mapping"}}
        result = _payload_uncertainty(payload, "scenario_success_rate")
        self.assertEqual(dict(result), {})


class PayloadMetricSeedItemsTest(unittest.TestCase):
    def test_scenario_filter_keeps_items_without_redundant_scenario(self) -> None:
        payload = {
            "seed_level": [
                {
                    "metric_name": "scenario_success_rate",
                    "seed": 1,
                    "value": 0.5,
                }
            ]
        }

        result = _payload_metric_seed_items(
            payload,
            "scenario_success_rate",
            scenario="night_rest",
        )

        self.assertEqual(result, [("1", 0.5)])

    def test_scenario_seed_level_preferred_over_top_level(self) -> None:
        payload = {
            "seed_level": [
                {
                    "metric_name": "scenario_success_rate",
                    "seed": "top",
                    "value": 0.1,
                }
            ],
            "suite": {
                "night_rest": {
                    "seed_level": [
                        {
                            "metric_name": "scenario_success_rate",
                            "seed": "scenario",
                            "value": 0.9,
                        }
                    ]
                }
            },
        }

        result = _payload_metric_seed_items(
            payload,
            "scenario_success_rate",
            scenario="night_rest",
        )

        self.assertEqual(result, [("scenario", 0.9)])

    def test_scenario_seed_level_falls_back_to_top_level_when_no_metric_match(self) -> None:
        payload = {
            "seed_level": [
                {
                    "metric_name": "scenario_success_rate",
                    "seed": "top",
                    "value": 0.1,
                }
            ],
            "suite": {
                "night_rest": {
                    "seed_level": [
                        {
                            "metric_name": "other_metric",
                            "seed": "scenario",
                            "value": 0.9,
                        }
                    ]
                }
            },
        }

        result = _payload_metric_seed_items(
            payload,
            "scenario_success_rate",
            scenario="night_rest",
        )

        self.assertEqual(result, [("top", 0.1)])


class IsZeroReflexScaleTest(unittest.TestCase):
    def test_zero_is_zero_reflex_scale(self) -> None:
        self.assertTrue(_is_zero_reflex_scale(0.0))

    def test_very_small_is_zero_reflex_scale(self) -> None:
        self.assertTrue(_is_zero_reflex_scale(1e-9))

    def test_one_is_not_zero_reflex_scale(self) -> None:
        self.assertFalse(_is_zero_reflex_scale(1.0))

    def test_none_is_not_zero_reflex_scale(self) -> None:
        self.assertFalse(_is_zero_reflex_scale(None))

    def test_nan_is_not_zero_reflex_scale(self) -> None:
        self.assertFalse(_is_zero_reflex_scale(float("nan")))

    def test_string_zero_is_zero_reflex_scale(self) -> None:
        self.assertTrue(_is_zero_reflex_scale("0.0"))

    def test_threshold_boundary_1e_6_is_zero(self) -> None:
        self.assertTrue(_is_zero_reflex_scale(1e-6))

    def test_above_threshold_is_not_zero(self) -> None:
        self.assertFalse(_is_zero_reflex_scale(1.1e-6))


class PayloadHasZeroReflexScaleTest(unittest.TestCase):
    def test_summary_eval_reflex_scale_zero(self) -> None:
        payload = {"summary": {"eval_reflex_scale": 0.0}}
        self.assertTrue(_payload_has_zero_reflex_scale(payload))

    def test_summary_eval_reflex_scale_nonzero(self) -> None:
        payload = {"summary": {"eval_reflex_scale": 1.0}}
        self.assertFalse(_payload_has_zero_reflex_scale(payload))

    def test_falls_back_to_top_level_eval_reflex_scale(self) -> None:
        payload = {"eval_reflex_scale": 0.0}
        self.assertTrue(_payload_has_zero_reflex_scale(payload))

    def test_missing_scale_returns_false(self) -> None:
        self.assertFalse(_payload_has_zero_reflex_scale({}))

    def test_none_scale_returns_false(self) -> None:
        payload = {"summary": {"eval_reflex_scale": None}}
        self.assertFalse(_payload_has_zero_reflex_scale(payload))


class PrimaryBenchmarkSourcePayloadTest(unittest.TestCase):
    def test_missing_configured_reference_variant_falls_back_to_modular_full(self) -> None:
        source, reference_variant, payload = _primary_benchmark_source_payload(
            {
                "behavior_evaluation": {
                    "ablations": {
                        "reference_variant": "missing_variant",
                        "variants": {
                            "modular_full": {
                                "summary": {
                                    "scenario_success_rate": 0.7,
                                }
                            }
                        },
                    }
                }
            }
        )

        self.assertEqual(
            source,
            "summary.behavior_evaluation.ablations.variants.modular_full",
        )
        self.assertEqual(reference_variant, "modular_full")
        self.assertEqual(payload["summary"]["scenario_success_rate"], 0.7)

    def test_uses_nested_evaluation_summary_for_zero_reflex_scale(self) -> None:
        source, reference_variant, payload = _primary_benchmark_source_payload(
            {
                "evaluation": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                    },
                    "eval_reflex_scale": 0.0,
                }
            }
        )

        self.assertEqual(source, "summary.evaluation")
        self.assertEqual(reference_variant, "")
        self.assertEqual(payload["summary"]["scenario_success_rate"], 0.8)

    def test_evaluation_branch_returns_full_payload_not_summary(self) -> None:
        source, _reference_variant, payload = _primary_benchmark_source_payload(
            {
                "evaluation": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                    },
                    "eval_reflex_scale": 0.0,
                }
            }
        )

        self.assertEqual(source, "summary.evaluation")
        self.assertIn("summary", payload)
        self.assertEqual(payload["summary"]["scenario_success_rate"], 0.8)

    def test_scenario_success_uses_zero_reflex_evaluation_summary(self) -> None:
        result = _primary_benchmark_scenario_success(
            {
                "evaluation": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                    },
                    "eval_reflex_scale": 0.0,
                }
            }
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.evaluation")
        self.assertAlmostEqual(result["scenarios"][0]["success_rate"], 0.8)

    def test_scenario_success_rejects_nonzero_reflex_evaluation_summary(self) -> None:
        zero_reflex_result = _primary_benchmark_scenario_success(
            {
                "evaluation": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                    },
                    "eval_reflex_scale": 0.0,
                }
            }
        )
        nonzero_reflex_result = _primary_benchmark_scenario_success(
            {
                "evaluation": {
                    "summary": {
                        "scenario_success_rate": 0.8,
                    },
                    "eval_reflex_scale": 0.5,
                }
            }
        )

        self.assertNotEqual(nonzero_reflex_result, zero_reflex_result)
        self.assertFalse(nonzero_reflex_result["available"])
        self.assertNotEqual(nonzero_reflex_result["source"], "summary.evaluation")


class MeanOrNoneTest(unittest.TestCase):
    def test_empty_returns_none(self) -> None:
        self.assertIsNone(_mean_or_none([]))

    def test_nonempty_returns_mean(self) -> None:
        result = _mean_or_none([0.2, 0.8])
        self.assertAlmostEqual(result, 0.5)

    def test_single_value(self) -> None:
        result = _mean_or_none([0.7])
        self.assertAlmostEqual(result, 0.7)


class CohensRowTest(unittest.TestCase):
    """Tests for _cohens_d_row function."""

    def test_returns_none_when_no_data_and_no_delta(self) -> None:
        result = _cohens_d_row(
            domain="test",
            baseline="baseline",
            comparison="comparison",
            metric="score",
            baseline_values=[],
            comparison_values=[],
            raw_delta=None,
            source="source",
        )
        self.assertIsNone(result)

    def test_returns_row_when_raw_delta_provided_no_values(self) -> None:
        result = _cohens_d_row(
            domain="test",
            baseline="A",
            comparison="B",
            metric="score",
            baseline_values=[],
            comparison_values=[],
            raw_delta=0.3,
            source="src",
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["raw_delta"], 0.3)

    def test_returns_none_when_raw_delta_not_numeric_and_no_values(self) -> None:
        result = _cohens_d_row(
            domain="test",
            baseline="A",
            comparison="B",
            metric="score",
            baseline_values=[],
            comparison_values=[],
            raw_delta="not-a-number",
            source="src",
        )
        self.assertIsNone(result)

    def test_computes_cohens_d_from_values(self) -> None:
        baseline = [0.2, 0.3, 0.4]
        comparison = [0.7, 0.8, 0.9]
        result = _cohens_d_row(
            domain="learning",
            baseline="random_init",
            comparison="trained",
            metric="scenario_success_rate",
            baseline_values=baseline,
            comparison_values=comparison,
            raw_delta=None,
            source="test",
        )
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["cohens_d"])
        self.assertIsInstance(result["cohens_d"], float)
        self.assertGreater(result["cohens_d"], 0)

    def test_row_has_required_keys(self) -> None:
        result = _cohens_d_row(
            domain="d",
            baseline="b",
            comparison="c",
            metric="m",
            baseline_values=[0.1, 0.2],
            comparison_values=[0.7, 0.8],
            raw_delta=None,
            source="s",
        )
        self.assertIsNotNone(result)
        for key in (
            "domain", "baseline", "comparison", "metric",
            "raw_delta", "cohens_d", "magnitude_label", "source",
            "effect_size_ci_lower", "effect_size_ci_upper",
            "effect_size_n_seeds", "delta_ci_lower", "delta_ci_upper",
        ):
            self.assertIn(key, result)
        self.assertIsNotNone(result["delta_ci_lower"])
        self.assertIsNotNone(result["delta_ci_upper"])
        seed_count_key = (
            "delta_n_seeds" if "delta_n_seeds" in result else "effect_size_n_seeds"
        )
        self.assertIsNotNone(result[seed_count_key])

    def test_domain_baseline_comparison_metric_set_correctly(self) -> None:
        result = _cohens_d_row(
            domain="ablation",
            baseline="modular_full",
            comparison="monolithic_policy",
            metric="scenario_success_rate",
            baseline_values=[0.6, 0.7],
            comparison_values=[0.4, 0.5],
            raw_delta=None,
            source="test",
        )
        self.assertEqual(result["domain"], "ablation")
        self.assertEqual(result["baseline"], "modular_full")
        self.assertEqual(result["comparison"], "monolithic_policy")
        self.assertEqual(result["metric"], "scenario_success_rate")

    def test_positive_delta_computed_from_values(self) -> None:
        # comparison mean > baseline mean => positive delta
        result = _cohens_d_row(
            domain="d",
            baseline="b",
            comparison="c",
            metric="m",
            baseline_values=[0.3, 0.3],
            comparison_values=[0.7, 0.7],
            raw_delta=None,
            source="s",
        )
        self.assertAlmostEqual(result["raw_delta"], 0.4, places=5)

    def test_explicit_raw_delta_used_over_computed(self) -> None:
        result = _cohens_d_row(
            domain="d",
            baseline="b",
            comparison="c",
            metric="m",
            baseline_values=[0.3, 0.3],
            comparison_values=[0.7, 0.7],
            raw_delta=0.99,
            source="s",
        )
        self.assertAlmostEqual(result["raw_delta"], 0.99)
