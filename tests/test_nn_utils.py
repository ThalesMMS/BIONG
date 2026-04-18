from __future__ import annotations

import unittest

import numpy as np

from spider_cortex_sim.nn_utils import (
    _clip_grad_logits,
    _coerce_state_array,
    _parameter_norm_of,
    _sigmoid,
    _weight_scale,
    one_hot,
    softmax,
)


class ClipGradLogitsTest(unittest.TestCase):
    """Helper tests for shared gradient sanitization and clipping."""

    def test_non_1d_grad_logits_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "1-D vector"):
            _clip_grad_logits(np.ones((3, 1)), 5.0)

    def test_zero_grad_clip_returns_zeros_with_same_shape(self) -> None:
        grad = np.array([np.nan, np.inf, -np.inf, 3.0], dtype=float)
        clipped = _clip_grad_logits(grad, 0.0)
        self.assertEqual(clipped.shape, grad.shape)
        self.assertEqual(clipped.dtype, float)
        np.testing.assert_allclose(clipped, np.zeros_like(grad, dtype=float))

    def test_negative_grad_clip_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "grad_clip"):
            _clip_grad_logits(np.ones(3), -1.0)

    def test_nonfinite_grad_clip_raises_value_error(self) -> None:
        for grad_clip in (np.nan, np.inf, -np.inf):
            with self.subTest(grad_clip=grad_clip):
                with self.assertRaisesRegex(ValueError, "grad_clip"):
                    _clip_grad_logits(np.ones(3), grad_clip)


class SigmoidFunctionTest(unittest.TestCase):
    """Tests for the _sigmoid helper added in this PR."""

    def test_zero_input_gives_half(self) -> None:
        result = _sigmoid(np.array([0.0]))
        self.assertAlmostEqual(float(result[0]), 0.5, places=10)

    def test_large_positive_approaches_one(self) -> None:
        result = _sigmoid(np.array([100.0]))
        self.assertGreater(float(result[0]), 0.999)
        self.assertLessEqual(float(result[0]), 1.0)

    def test_large_negative_approaches_zero(self) -> None:
        result = _sigmoid(np.array([-100.0]))
        self.assertLess(float(result[0]), 0.001)
        self.assertGreaterEqual(float(result[0]), 0.0)

    def test_output_always_in_open_unit_interval(self) -> None:
        x = np.linspace(-10.0, 10.0, 50)
        result = _sigmoid(x)
        self.assertTrue(np.all(result > 0.0))
        self.assertTrue(np.all(result < 1.0))

    def test_nan_input_treated_as_zero(self) -> None:
        result = _sigmoid(np.array([np.nan]))
        self.assertAlmostEqual(float(result[0]), 0.5, places=10)

    def test_posinf_input_clamped_to_near_one(self) -> None:
        result = _sigmoid(np.array([np.inf]))
        self.assertGreater(float(result[0]), 0.999)

    def test_neginf_input_clamped_to_near_zero(self) -> None:
        result = _sigmoid(np.array([-np.inf]))
        self.assertLess(float(result[0]), 0.001)

    def test_scalar_array_works(self) -> None:
        result = _sigmoid(np.array(1.0))
        self.assertTrue(np.isfinite(result))
        expected = 1.0 / (1.0 + np.exp(-1.0))
        self.assertAlmostEqual(float(result), expected, places=8)

    def test_output_is_finite_for_extreme_inputs(self) -> None:
        x = np.array([-1e9, 0.0, 1e9, np.nan, np.inf, -np.inf])
        result = _sigmoid(x)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_output_shape_matches_input(self) -> None:
        x = np.arange(12, dtype=float).reshape(3, 4)
        result = _sigmoid(x)
        self.assertEqual(result.shape, (3, 4))

    def test_monotonically_increasing(self) -> None:
        x = np.linspace(-5.0, 5.0, 20)
        result = _sigmoid(x)
        self.assertTrue(np.all(np.diff(result) > 0))


class CoerceStateArrayTest(unittest.TestCase):
    """Direct unit tests for the _coerce_state_array helper added in this PR."""

    def test_returns_float_array_for_integer_list(self) -> None:
        state = {"W": [[1, 2], [3, 4]]}
        result = _coerce_state_array(state, "W", (2, 2), name="net")
        self.assertEqual(result.dtype, float)
        self.assertEqual(result.shape, (2, 2))

    def test_returns_copy_not_original_reference(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        state = {"W": arr}
        result = _coerce_state_array(state, "W", (2, 2), name="net")
        result[0, 0] = 999.0
        self.assertAlmostEqual(arr[0, 0], 1.0, msg="Modifying result should not affect original array")

    def test_correct_shape_passes(self) -> None:
        state = {"b": np.zeros(8)}
        result = _coerce_state_array(state, "b", (8,), name="test_net")
        self.assertEqual(result.shape, (8,))

    def test_wrong_shape_raises_value_error(self) -> None:
        state = {"W1": np.zeros((10, 8))}
        with self.assertRaises(ValueError):
            _coerce_state_array(state, "W1", (10, 99), name="test_net")

    def test_wrong_shape_error_includes_network_name(self) -> None:
        state = {"W1": np.zeros((5, 3))}
        with self.assertRaisesRegex(ValueError, "my_network"):
            _coerce_state_array(state, "W1", (5, 10), name="my_network")

    def test_wrong_shape_error_includes_key_name(self) -> None:
        state = {"W2": np.zeros((4, 6))}
        with self.assertRaisesRegex(ValueError, "W2"):
            _coerce_state_array(state, "W2", (4, 99), name="net")

    def test_missing_key_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing_key"):
            _coerce_state_array({}, "missing_key", (1,), name="net")

    def test_accepts_numpy_array_input(self) -> None:
        arr = np.array([0.1, 0.2, 0.3, 0.4])
        state = {"b": arr}
        result = _coerce_state_array(state, "b", (4,), name="net")
        np.testing.assert_array_almost_equal(result, arr)

    def test_accepts_list_input(self) -> None:
        state = {"b": [0.5, 0.6, 0.7]}
        result = _coerce_state_array(state, "b", (3,), name="net")
        self.assertEqual(result.shape, (3,))
        self.assertAlmostEqual(result[1], 0.6)

    def test_2d_wrong_row_count_raises_value_error(self) -> None:
        state = {"W": np.zeros((3, 4))}
        with self.assertRaises(ValueError):
            _coerce_state_array(state, "W", (5, 4), name="net")

    def test_1d_wrong_size_raises_value_error(self) -> None:
        state = {"b": np.zeros(5)}
        with self.assertRaises(ValueError):
            _coerce_state_array(state, "b", (8,), name="net")

    def test_result_values_match_source(self) -> None:
        data = np.array([[1.5, 2.5], [3.5, 4.5]])
        state = {"W": data}
        result = _coerce_state_array(state, "W", (2, 2), name="net")
        np.testing.assert_array_almost_equal(result, data)

    def test_nan_values_preserved(self) -> None:
        """_coerce_state_array does not sanitize NaNs; the forward pass handles that."""
        state = {"b": np.array([1.0, float("nan"), 3.0])}
        result = _coerce_state_array(state, "b", (3,), name="net")
        self.assertTrue(np.isnan(result[1]))


class SoftmaxTest(unittest.TestCase):
    def test_returns_probability_distribution(self) -> None:
        result = softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(np.sum(result)), 1.0, places=10)
        self.assertEqual(result.shape, (3,))

    def test_handles_nonfinite_values(self) -> None:
        result = softmax(np.array([np.nan, np.inf, -np.inf]))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertAlmostEqual(float(np.sum(result)), 1.0, places=10)


class OneHotTest(unittest.TestCase):
    def test_sets_only_requested_index(self) -> None:
        result = one_hot(2, 5)
        np.testing.assert_array_equal(result, np.array([0.0, 0.0, 1.0, 0.0, 0.0]))

    def test_uses_float_dtype(self) -> None:
        self.assertEqual(one_hot(0, 2).dtype, float)


class WeightScaleTest(unittest.TestCase):
    def test_uses_inverse_sqrt_dimension(self) -> None:
        self.assertAlmostEqual(_weight_scale(4), 0.35 / np.sqrt(4))

    def test_nonpositive_dimension_uses_one(self) -> None:
        self.assertAlmostEqual(_weight_scale(0), 0.35)


class ParameterNormOfTest(unittest.TestCase):
    def test_combines_all_parameter_arrays(self) -> None:
        result = _parameter_norm_of(np.array([3.0, 4.0]), np.array([[12.0]]))
        self.assertAlmostEqual(result, 13.0)

    def test_empty_input_is_zero(self) -> None:
        self.assertEqual(_parameter_norm_of(), 0.0)


if __name__ == "__main__":
    unittest.main()
