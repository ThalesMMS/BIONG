"""Tests for type-coercion and formatting helpers in spider_cortex_sim.offline_analysis.utils.

These functions were extracted from the monolithic offline_analysis.py into utils.py
in this PR. All tests target the new module path.
"""
from __future__ import annotations

import math
import unittest

from spider_cortex_sim.offline_analysis.utils import (
    _coerce_bool,
    _coerce_float,
    _coerce_optional_float,
    _dominant_module_by_score,
    _format_optional_metric,
    _mean,
    _safe_divide,
)


class CoerceFloatTest(unittest.TestCase):
    # Basic numeric conversions
    def test_int_converts_to_float(self) -> None:
        self.assertEqual(_coerce_float(3), 3.0)

    def test_float_is_returned_unchanged(self) -> None:
        self.assertEqual(_coerce_float(1.5), 1.5)

    def test_string_float_converts(self) -> None:
        self.assertEqual(_coerce_float("2.5"), 2.5)

    def test_string_int_converts(self) -> None:
        self.assertEqual(_coerce_float("10"), 10.0)

    # Boolean special handling
    def test_true_converts_to_1(self) -> None:
        self.assertEqual(_coerce_float(True), 1.0)

    def test_false_converts_to_0(self) -> None:
        self.assertEqual(_coerce_float(False), 0.0)

    # None handling
    def test_none_returns_default(self) -> None:
        self.assertEqual(_coerce_float(None), 0.0)

    def test_none_returns_custom_default(self) -> None:
        self.assertEqual(_coerce_float(None, 99.0), 99.0)

    # Non-finite values
    def test_nan_returns_default(self) -> None:
        self.assertEqual(_coerce_float(float("nan")), 0.0)

    def test_inf_returns_default(self) -> None:
        self.assertEqual(_coerce_float(float("inf")), 0.0)

    def test_negative_inf_returns_default(self) -> None:
        self.assertEqual(_coerce_float(float("-inf")), 0.0)

    # String nan/inf
    def test_string_nan_returns_default(self) -> None:
        self.assertEqual(_coerce_float("nan"), 0.0)

    def test_string_inf_returns_default(self) -> None:
        self.assertEqual(_coerce_float("inf"), 0.0)

    # Unconvertable types
    def test_non_numeric_string_returns_default(self) -> None:
        self.assertEqual(_coerce_float("not_a_number"), 0.0)

    def test_list_returns_default(self) -> None:
        self.assertEqual(_coerce_float([1, 2, 3]), 0.0)

    def test_dict_returns_default(self) -> None:
        self.assertEqual(_coerce_float({"a": 1}), 0.0)

    # Custom default propagation
    def test_invalid_value_uses_custom_default(self) -> None:
        self.assertEqual(_coerce_float("bad", 7.5), 7.5)

    # Edge: zero
    def test_zero_converts_correctly(self) -> None:
        self.assertEqual(_coerce_float(0), 0.0)

    # Edge: negative value
    def test_negative_value_converts_correctly(self) -> None:
        self.assertEqual(_coerce_float(-3.14), -3.14)


class CoerceOptionalFloatTest(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(_coerce_optional_float(None))

    def test_valid_float_string_converts(self) -> None:
        result = _coerce_optional_float("0.75")
        self.assertAlmostEqual(result, 0.75)

    def test_valid_int_converts(self) -> None:
        result = _coerce_optional_float(5)
        self.assertAlmostEqual(result, 5.0)

    def test_nan_returns_none(self) -> None:
        self.assertIsNone(_coerce_optional_float(float("nan")))

    def test_inf_returns_none(self) -> None:
        self.assertIsNone(_coerce_optional_float(float("inf")))

    def test_non_numeric_string_returns_none(self) -> None:
        self.assertIsNone(_coerce_optional_float("bad"))

    def test_list_returns_none(self) -> None:
        self.assertIsNone(_coerce_optional_float([1, 2]))

    def test_zero_converts(self) -> None:
        result = _coerce_optional_float(0)
        self.assertAlmostEqual(result, 0.0)


class CoerceBoolTest(unittest.TestCase):
    # Passthrough for actual booleans
    def test_true_passthrough(self) -> None:
        self.assertTrue(_coerce_bool(True))

    def test_false_passthrough(self) -> None:
        self.assertFalse(_coerce_bool(False))

    # None handling
    def test_none_returns_false(self) -> None:
        self.assertFalse(_coerce_bool(None))

    # Truthy string values
    def test_string_1_is_true(self) -> None:
        self.assertTrue(_coerce_bool("1"))

    def test_string_true_is_true(self) -> None:
        self.assertTrue(_coerce_bool("true"))

    def test_string_True_case_insensitive(self) -> None:
        self.assertTrue(_coerce_bool("True"))

    def test_string_TRUE_is_true(self) -> None:
        self.assertTrue(_coerce_bool("TRUE"))

    def test_string_yes_is_true(self) -> None:
        self.assertTrue(_coerce_bool("yes"))

    def test_string_YES_is_true(self) -> None:
        self.assertTrue(_coerce_bool("YES"))

    def test_string_y_is_true(self) -> None:
        self.assertTrue(_coerce_bool("y"))

    def test_string_on_is_true(self) -> None:
        self.assertTrue(_coerce_bool("on"))

    # Falsy string values
    def test_string_0_is_false(self) -> None:
        self.assertFalse(_coerce_bool("0"))

    def test_string_false_is_false(self) -> None:
        self.assertFalse(_coerce_bool("false"))

    def test_string_no_is_false(self) -> None:
        self.assertFalse(_coerce_bool("no"))

    def test_string_off_is_false(self) -> None:
        self.assertFalse(_coerce_bool("off"))

    def test_empty_string_is_false(self) -> None:
        self.assertFalse(_coerce_bool(""))

    # Whitespace trimmed
    def test_whitespace_around_true_is_true(self) -> None:
        self.assertTrue(_coerce_bool("  true  "))

    def test_whitespace_around_false_is_false(self) -> None:
        self.assertFalse(_coerce_bool("  false  "))

    # Integer non-bool values are coerced through their string forms.
    def test_integer_1_is_true(self) -> None:
        # str(1) -> "1", which is in the truthy set.
        self.assertTrue(_coerce_bool(1))

    def test_integer_0_is_false(self) -> None:
        # str(0) -> "0", which is not in the truthy set.
        self.assertFalse(_coerce_bool(0))


class SafeDivideTest(unittest.TestCase):
    def test_positive_denominator(self) -> None:
        self.assertAlmostEqual(_safe_divide(10.0, 4.0), 2.5)

    def test_zero_denominator_returns_zero(self) -> None:
        self.assertEqual(_safe_divide(10.0, 0.0), 0.0)
        self.assertEqual(_safe_divide(100.0, 0.0), 0.0)

    def test_negative_denominator_returns_zero(self) -> None:
        self.assertEqual(_safe_divide(10.0, -1.0), 0.0)

    def test_zero_numerator(self) -> None:
        self.assertEqual(_safe_divide(0.0, 5.0), 0.0)

    def test_fraction_result(self) -> None:
        self.assertAlmostEqual(_safe_divide(1.0, 3.0), 1.0 / 3.0)

class MeanTest(unittest.TestCase):
    def test_empty_list_returns_zero(self) -> None:
        self.assertEqual(_mean([]), 0.0)

    def test_single_element(self) -> None:
        self.assertAlmostEqual(_mean([5.0]), 5.0)

    def test_equal_values(self) -> None:
        self.assertAlmostEqual(_mean([3.0, 3.0, 3.0]), 3.0)

    def test_mixed_values(self) -> None:
        self.assertAlmostEqual(_mean([1.0, 2.0, 3.0, 4.0]), 2.5)

    def test_negative_values(self) -> None:
        self.assertAlmostEqual(_mean([-1.0, 1.0]), 0.0)

    def test_generator_input(self) -> None:
        # _mean accepts any Iterable[float]
        self.assertAlmostEqual(_mean(x * 0.5 for x in range(1, 5)), 1.25)

    def test_returns_float_type(self) -> None:
        result = _mean([1, 2, 3])
        self.assertIsInstance(result, float)


class DominantModuleByScoreTest(unittest.TestCase):
    def test_empty_mapping_returns_empty_string(self) -> None:
        self.assertEqual(_dominant_module_by_score({}), "")

    def test_single_entry_returns_that_key(self) -> None:
        self.assertEqual(_dominant_module_by_score({"visual_cortex": 0.9}), "visual_cortex")

    def test_returns_key_with_highest_score(self) -> None:
        modules = {
            "visual_cortex": 0.3,
            "sensory_cortex": 0.8,
            "hunger_center": 0.1,
        }
        self.assertEqual(_dominant_module_by_score(modules), "sensory_cortex")

    def test_ties_return_one_of_them(self) -> None:
        modules = {"a": 1.0, "b": 1.0}
        result = _dominant_module_by_score(modules)
        self.assertIn(result, {"a", "b"})

    def test_coerces_non_float_values(self) -> None:
        # _coerce_float is used, so "0.5" should work
        modules = {"a": "0.5", "b": "0.9", "c": None}
        result = _dominant_module_by_score(modules)
        self.assertEqual(result, "b")

    def test_negative_scores_handled(self) -> None:
        modules = {"x": -1.0, "y": -0.5}
        self.assertEqual(_dominant_module_by_score(modules), "y")


class FormatOptionalMetricTest(unittest.TestCase):
    def test_none_returns_dash(self) -> None:
        self.assertEqual(_format_optional_metric(None), "—")

    def test_zero_formats_as_two_decimals(self) -> None:
        self.assertEqual(_format_optional_metric(0.0), "0.00")

    def test_positive_float_formatted(self) -> None:
        self.assertEqual(_format_optional_metric(0.75), "0.75")

    def test_rounds_to_two_decimals(self) -> None:
        # 0.123 rounds to 0.12
        self.assertEqual(_format_optional_metric(0.123), "0.12")

    def test_large_value_formatted(self) -> None:
        self.assertEqual(_format_optional_metric(100.0), "100.00")

    def test_negative_value_formatted(self) -> None:
        self.assertEqual(_format_optional_metric(-0.5), "-0.50")
