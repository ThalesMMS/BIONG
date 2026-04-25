"""Tests for spider_cortex_sim.ablation.predator_metrics module.

Covers compare_predator_type_ablation_performance and its internal helpers,
introduced as a focused submodule in the PR refactor.
"""

from __future__ import annotations

import math
import unittest

from spider_cortex_sim.ablation.predator_metrics import (
    _finite_float_or_none,
    _group_metric_value,
    _mean,
    _safe_float,
    _scenario_success_rate,
    compare_predator_type_ablation_performance,
)


class SafeFloatTest(unittest.TestCase):
    """Tests for _safe_float helper."""

    def test_valid_float_passes_through(self) -> None:
        self.assertAlmostEqual(_safe_float(0.75), 0.75)

    def test_integer_is_coerced_to_float(self) -> None:
        result = _safe_float(1)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 1.0)

    def test_string_number_is_coerced(self) -> None:
        self.assertAlmostEqual(_safe_float("0.5"), 0.5)

    def test_invalid_string_returns_zero(self) -> None:
        self.assertEqual(_safe_float("not_a_number"), 0.0)

    def test_none_returns_zero(self) -> None:
        self.assertEqual(_safe_float(None), 0.0)

    def test_nan_returns_zero(self) -> None:
        self.assertEqual(_safe_float(float("nan")), 0.0)

    def test_positive_infinity_returns_zero(self) -> None:
        self.assertEqual(_safe_float(float("inf")), 0.0)

    def test_negative_infinity_returns_zero(self) -> None:
        self.assertEqual(_safe_float(float("-inf")), 0.0)

    def test_zero_is_returned_as_is(self) -> None:
        self.assertEqual(_safe_float(0.0), 0.0)

    def test_negative_value_passes_through(self) -> None:
        self.assertAlmostEqual(_safe_float(-0.3), -0.3)


class FiniteFloatOrNoneTest(unittest.TestCase):
    """Tests for _finite_float_or_none helper."""

    def test_valid_float_returns_float(self) -> None:
        result = _finite_float_or_none(0.5)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.5)

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_finite_float_or_none(None))

    def test_nan_returns_none(self) -> None:
        self.assertIsNone(_finite_float_or_none(float("nan")))

    def test_inf_returns_none(self) -> None:
        self.assertIsNone(_finite_float_or_none(float("inf")))

    def test_neg_inf_returns_none(self) -> None:
        self.assertIsNone(_finite_float_or_none(float("-inf")))

    def test_string_number_is_coerced(self) -> None:
        result = _finite_float_or_none("0.7")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.7)

    def test_invalid_string_returns_none(self) -> None:
        self.assertIsNone(_finite_float_or_none("not_a_float"))

    def test_integer_is_coerced(self) -> None:
        result = _finite_float_or_none(3)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 3.0)

    def test_zero_is_returned_as_zero(self) -> None:
        result = _finite_float_or_none(0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0)


class MeanTest(unittest.TestCase):
    """Tests for _mean helper."""

    def test_empty_sequence_returns_zero(self) -> None:
        self.assertEqual(_mean([]), 0.0)

    def test_single_value_returns_that_value(self) -> None:
        self.assertAlmostEqual(_mean([0.6]), 0.6)

    def test_multiple_values_returns_arithmetic_mean(self) -> None:
        self.assertAlmostEqual(_mean([0.0, 1.0]), 0.5)

    def test_returns_float(self) -> None:
        result = _mean([1, 2, 3])
        self.assertIsInstance(result, float)

    def test_all_zeros_returns_zero(self) -> None:
        self.assertAlmostEqual(_mean([0.0, 0.0, 0.0]), 0.0)

    def test_mean_of_three_values(self) -> None:
        self.assertAlmostEqual(_mean([0.2, 0.4, 0.6]), 0.4)


class ScenarioSuccessRateTest(unittest.TestCase):
    """Tests for _scenario_success_rate helper."""

    def _make_payload(self, **suite_entries) -> dict:
        return {
            "suite": {
                name: {"success_rate": rate}
                for name, rate in suite_entries.items()
            }
        }

    def test_returns_float_when_present_in_suite(self) -> None:
        payload = self._make_payload(night_rest=0.75)
        result = _scenario_success_rate(payload, "night_rest")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.75)

    def test_returns_none_when_scenario_absent(self) -> None:
        payload = self._make_payload(night_rest=0.5)
        self.assertIsNone(_scenario_success_rate(payload, "missing_scenario"))

    def test_returns_none_when_payload_empty(self) -> None:
        self.assertIsNone(_scenario_success_rate({}, "night_rest"))

    def test_reads_from_legacy_scenarios_key(self) -> None:
        payload = {
            "legacy_scenarios": {
                "night_rest": {"success_rate": 0.6}
            }
        }
        result = _scenario_success_rate(payload, "night_rest")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.6)

    def test_suite_takes_precedence_over_legacy(self) -> None:
        payload = {
            "suite": {"night_rest": {"success_rate": 0.8}},
            "legacy_scenarios": {"night_rest": {"success_rate": 0.3}},
        }
        result = _scenario_success_rate(payload, "night_rest")
        self.assertAlmostEqual(result, 0.8)

    def test_returns_none_when_success_rate_is_nan(self) -> None:
        payload = {"suite": {"night_rest": {"success_rate": float("nan")}}}
        self.assertIsNone(_scenario_success_rate(payload, "night_rest"))

    def test_returns_none_when_success_rate_key_absent_in_scenario(self) -> None:
        payload = {"suite": {"night_rest": {"other_key": 0.5}}}
        self.assertIsNone(_scenario_success_rate(payload, "night_rest"))


class GroupMetricValueTest(unittest.TestCase):
    """Tests for _group_metric_value helper."""

    def test_returns_metric_when_count_positive(self) -> None:
        group = {"mean_success_rate": 0.7, "scenario_count": 2}
        result = _group_metric_value(group, "mean_success_rate", count_key="scenario_count")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.7)

    def test_returns_none_when_count_zero(self) -> None:
        group = {"mean_success_rate": 0.7, "scenario_count": 0}
        result = _group_metric_value(group, "mean_success_rate", count_key="scenario_count")
        self.assertIsNone(result)

    def test_returns_none_when_group_not_mapping(self) -> None:
        result = _group_metric_value("not_a_mapping", "mean_success_rate", count_key="scenario_count")
        self.assertIsNone(result)

    def test_returns_none_when_metric_absent(self) -> None:
        group = {"scenario_count": 2}
        result = _group_metric_value(group, "mean_success_rate", count_key="scenario_count")
        self.assertIsNone(result)

    def test_returns_none_when_metric_is_none(self) -> None:
        group = {"mean_success_rate": None, "scenario_count": 2}
        result = _group_metric_value(group, "mean_success_rate", count_key="scenario_count")
        self.assertIsNone(result)


class ComparePredatorTypeAblationPerformanceTest(unittest.TestCase):
    """Tests for compare_predator_type_ablation_performance."""

    def _make_payload(self, visual_rate: float = 0.8, olfactory_rate: float = 0.4) -> dict:
        """Build a minimal ablations payload with variant data."""
        return {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": visual_rate},
                        "visual_hunter_open_field": {"success_rate": visual_rate},
                        "olfactory_ambush": {"success_rate": olfactory_rate},
                    }
                },
                "drop_sensory_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": olfactory_rate},
                        "visual_hunter_open_field": {"success_rate": olfactory_rate},
                        "olfactory_ambush": {"success_rate": visual_rate},
                    }
                },
            }
        }

    def test_returns_dict_with_expected_keys(self) -> None:
        payload = self._make_payload()
        result = compare_predator_type_ablation_performance(payload)
        self.assertIn("available", result)
        self.assertIn("scenario_groups", result)
        self.assertIn("comparisons", result)

    def test_available_is_true_when_variants_present(self) -> None:
        payload = self._make_payload()
        result = compare_predator_type_ablation_performance(payload)
        self.assertTrue(result["available"])

    def test_available_is_false_when_variants_missing(self) -> None:
        result = compare_predator_type_ablation_performance({})
        self.assertFalse(result["available"])

    def test_available_is_false_when_variants_not_mapping(self) -> None:
        result = compare_predator_type_ablation_performance({"variants": "not_a_dict"})
        self.assertFalse(result["available"])

    def test_comparisons_contains_drop_visual_cortex(self) -> None:
        payload = self._make_payload()
        result = compare_predator_type_ablation_performance(payload)
        self.assertIn("drop_visual_cortex", result["comparisons"])

    def test_comparisons_contains_drop_sensory_cortex(self) -> None:
        payload = self._make_payload()
        result = compare_predator_type_ablation_performance(payload)
        self.assertIn("drop_sensory_cortex", result["comparisons"])

    def test_visual_minus_olfactory_is_numeric_for_drop_visual(self) -> None:
        payload = self._make_payload(visual_rate=0.8, olfactory_rate=0.4)
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        diff = comparison.get("visual_minus_olfactory_success_rate")
        self.assertIsNotNone(diff)
        self.assertIsInstance(diff, float)

    def test_visual_minus_olfactory_correct_sign_for_drop_visual(self) -> None:
        """When dropping visual cortex, visual group should perform worse (lower) than olfactory."""
        payload = self._make_payload(visual_rate=0.3, olfactory_rate=0.7)
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        # visual group mean success < olfactory group mean success
        diff = comparison.get("visual_minus_olfactory_success_rate")
        self.assertIsNotNone(diff)
        self.assertLess(diff, 0)

    def test_scenario_groups_contains_expected_groups(self) -> None:
        payload = self._make_payload()
        result = compare_predator_type_ablation_performance(payload)
        groups = result["scenario_groups"]
        self.assertIn("multi_predator_ecology", groups)
        self.assertIn("visual_predator_scenarios", groups)
        self.assertIn("olfactory_predator_scenarios", groups)

    def test_empty_variants_mapping_returns_available_false(self) -> None:
        result = compare_predator_type_ablation_performance({"variants": {}})
        self.assertFalse(result["available"])

    def test_group_rows_contain_scenario_count(self) -> None:
        payload = self._make_payload()
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        visual_group = comparison.get("visual_predator_scenarios", {})
        self.assertIn("scenario_count", visual_group)

    def test_group_mean_success_rate_is_float(self) -> None:
        payload = self._make_payload()
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        visual_group = comparison.get("visual_predator_scenarios", {})
        self.assertIsInstance(visual_group.get("mean_success_rate"), float)

    def test_variant_not_in_payload_is_skipped(self) -> None:
        payload = {"variants": {"drop_visual_cortex": {
            "suite": {
                "visual_olfactory_pincer": {"success_rate": 0.5},
                "visual_hunter_open_field": {"success_rate": 0.5},
                "olfactory_ambush": {"success_rate": 0.5},
            }
        }}}
        result = compare_predator_type_ablation_performance(
            payload, variant_names=["drop_visual_cortex", "drop_sensory_cortex"]
        )
        self.assertIn("drop_visual_cortex", result["comparisons"])
        self.assertNotIn("drop_sensory_cortex", result["comparisons"])

    def test_custom_variant_names_are_used(self) -> None:
        payload = {
            "variants": {
                "custom_variant": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.6},
                        "visual_hunter_open_field": {"success_rate": 0.6},
                        "olfactory_ambush": {"success_rate": 0.6},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(
            payload, variant_names=["custom_variant"]
        )
        self.assertIn("custom_variant", result["comparisons"])

    def test_delta_comparison_uses_deltas_vs_reference(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.5},
                        "visual_hunter_open_field": {"success_rate": 0.5},
                        "olfactory_ambush": {"success_rate": 0.5},
                    }
                }
            },
            "deltas_vs_reference": {
                "drop_visual_cortex_vs_modular_full": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"scenario_success_rate_delta": -0.1},
                        "visual_hunter_open_field": {"scenario_success_rate_delta": -0.2},
                        "olfactory_ambush": {"scenario_success_rate_delta": 0.1},
                    }
                }
            },
        }
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        visual_group = comparison.get("visual_predator_scenarios", {})
        # Delta count should be 2 (for the two visual scenarios)
        self.assertEqual(visual_group.get("scenario_delta_count"), 2)

    def test_visual_minus_olfactory_success_rate_delta_is_computed(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.5},
                        "visual_hunter_open_field": {"success_rate": 0.5},
                        "olfactory_ambush": {"success_rate": 0.5},
                    }
                }
            },
            "deltas_vs_reference": {
                "drop_visual_cortex_vs_modular_full": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"scenario_success_rate_delta": -0.2},
                        "visual_hunter_open_field": {"scenario_success_rate_delta": -0.3},
                        "olfactory_ambush": {"scenario_success_rate_delta": 0.1},
                    }
                }
            },
        }
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        diff_delta = comparison.get("visual_minus_olfactory_success_rate_delta")
        self.assertIsNotNone(diff_delta)
        # visual mean delta = (-0.2 + -0.3) / 2 = -0.25
        # olfactory has visual_olfactory_pincer AND olfactory_ambush
        # olfactory mean delta = (-0.2 + 0.1) / 2 = -0.05
        # diff = -0.25 - (-0.05) = -0.20
        self.assertAlmostEqual(diff_delta, -0.25 - (-0.05), places=5)

    def test_result_is_rounded_to_6_decimals(self) -> None:
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 1 / 3},
                        "visual_hunter_open_field": {"success_rate": 1 / 3},
                        "olfactory_ambush": {"success_rate": 2 / 3},
                    }
                }
            }
        }
        result = compare_predator_type_ablation_performance(payload)
        comparison = result["comparisons"]["drop_visual_cortex"]
        diff = comparison.get("visual_minus_olfactory_success_rate")
        self.assertIsNotNone(diff)
        # Verify rounded to 6 places
        self.assertAlmostEqual(diff, round(diff, 6), places=10)


if __name__ == "__main__":
    unittest.main()
