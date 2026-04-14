"""Tests for the simplified single-hidden-layer neural networks.

This PR removed the intermediate W_mid/b_mid layer from both networks, reducing
the proposal and motor networks from two hidden layers to one. Tests here verify
those architectures plus the learned arbitration network.
"""

from __future__ import annotations

import unittest

import numpy as np

from spider_cortex_sim.nn import (
    ArbitrationCache,
    ArbitrationNetwork,
    MotorCache,
    MotorNetwork,
    ProposalCache,
    ProposalNetwork,
    RecurrentProposalCache,
    RecurrentProposalNetwork,
    _clip_grad_logits,
    _coerce_state_array,
)


class ProposalCacheFieldsTest(unittest.TestCase):
    """ProposalCache now stores only x and h (not h1 and h2)."""

    def test_proposal_cache_has_x_and_h_fields(self) -> None:
        x = np.array([1.0, 2.0])
        h = np.array([0.5, -0.5, 0.3])
        cache = ProposalCache(x=x, h=h)
        np.testing.assert_array_equal(cache.x, x)
        np.testing.assert_array_equal(cache.h, h)

    def test_proposal_cache_does_not_have_h1_h2_fields(self) -> None:
        x = np.zeros(4)
        h = np.zeros(8)
        cache = ProposalCache(x=x, h=h)
        self.assertFalse(hasattr(cache, "h1"), "ProposalCache should not have h1 after PR change")
        self.assertFalse(hasattr(cache, "h2"), "ProposalCache should not have h2 after PR change")


class RecurrentProposalCacheFieldsTest(unittest.TestCase):
    """RecurrentProposalCache stores one single-step recurrent transition."""

    def test_recurrent_cache_has_x_h_prev_and_h_new_fields(self) -> None:
        x = np.array([1.0, 2.0])
        h_prev = np.array([0.0, 0.1, 0.2])
        h_new = np.array([0.3, 0.4, 0.5])
        cache = RecurrentProposalCache(x=x, h_prev=h_prev, h_new=h_new)
        np.testing.assert_array_equal(cache.x, x)
        np.testing.assert_array_equal(cache.h_prev, h_prev)
        np.testing.assert_array_equal(cache.h_new, h_new)


class MotorCacheFieldsTest(unittest.TestCase):
    """MotorCache now stores only x and h (not h1 and h2)."""

    def test_motor_cache_has_x_and_h_fields(self) -> None:
        x = np.array([0.1, 0.2, 0.3])
        h = np.array([0.9, -0.8, 0.7, -0.6])
        cache = MotorCache(x=x, h=h)
        np.testing.assert_array_equal(cache.x, x)
        np.testing.assert_array_equal(cache.h, h)

    def test_motor_cache_does_not_have_h1_h2_fields(self) -> None:
        x = np.zeros(5)
        h = np.zeros(10)
        cache = MotorCache(x=x, h=h)
        self.assertFalse(hasattr(cache, "h1"), "MotorCache should not have h1 after PR change")
        self.assertFalse(hasattr(cache, "h2"), "MotorCache should not have h2 after PR change")


class ArbitrationCacheFieldsTest(unittest.TestCase):
    """ArbitrationCache stores the evidence input and shared hidden activation."""

    def test_arbitration_cache_has_x_and_h_fields(self) -> None:
        x = np.array([0.1, 0.2, 0.3])
        h = np.array([0.4, -0.5, 0.6, -0.7])
        cache = ArbitrationCache(x=x, h=h)
        np.testing.assert_array_equal(cache.x, x)
        np.testing.assert_array_equal(cache.h, h)

    def test_arbitration_cache_does_not_have_h1_h2_fields(self) -> None:
        cache = ArbitrationCache(x=np.zeros(ArbitrationNetwork.INPUT_DIM), h=np.zeros(8))
        self.assertFalse(hasattr(cache, "h1"))
        self.assertFalse(hasattr(cache, "h2"))


class ProposalNetworkArchitectureTest(unittest.TestCase):
    """ProposalNetwork has a single hidden layer: W1/b1 then W2/b2, no W_mid/b_mid."""

    def _make_net(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 5) -> ProposalNetwork:
        rng = np.random.default_rng(42)
        return ProposalNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            name="test_proposal",
        )

    def test_has_w1_b1_w2_b2(self) -> None:
        net = self._make_net()
        self.assertTrue(hasattr(net, "W1"))
        self.assertTrue(hasattr(net, "b1"))
        self.assertTrue(hasattr(net, "W2"))
        self.assertTrue(hasattr(net, "b2"))

    def test_does_not_have_w_mid_b_mid(self) -> None:
        net = self._make_net()
        self.assertFalse(hasattr(net, "W_mid"), "ProposalNetwork should not have W_mid after PR change")
        self.assertFalse(hasattr(net, "b_mid"), "ProposalNetwork should not have b_mid after PR change")

    def test_w1_shape(self) -> None:
        net = self._make_net(input_dim=8, hidden_dim=16, output_dim=5)
        self.assertEqual(net.W1.shape, (16, 8))
        self.assertEqual(net.b1.shape, (16,))

    def test_w2_shape(self) -> None:
        net = self._make_net(input_dim=8, hidden_dim=16, output_dim=5)
        self.assertEqual(net.W2.shape, (5, 16))
        self.assertEqual(net.b2.shape, (5,))

    def test_cache_is_none_before_forward(self) -> None:
        net = self._make_net()
        self.assertIsNone(net.cache)

    def test_forward_returns_logits_of_correct_shape(self) -> None:
        net = self._make_net(input_dim=8, hidden_dim=16, output_dim=5)
        x = np.random.default_rng(0).standard_normal(8)
        logits = net.forward(x)
        self.assertEqual(logits.shape, (5,))

    def test_forward_stores_cache_with_h_field(self) -> None:
        net = self._make_net(input_dim=8, hidden_dim=16, output_dim=5)
        x = np.random.default_rng(1).standard_normal(8)
        net.forward(x, store_cache=True)
        self.assertIsNotNone(net.cache)
        self.assertIsInstance(net.cache, ProposalCache)
        self.assertEqual(net.cache.h.shape, (16,))
        self.assertFalse(hasattr(net.cache, "h1"))
        self.assertFalse(hasattr(net.cache, "h2"))

    def test_forward_no_cache_does_not_store_cache(self) -> None:
        net = self._make_net()
        x = np.random.default_rng(2).standard_normal(8)
        net.forward(x, store_cache=False)
        self.assertIsNone(net.cache)

    def test_logits_are_clipped_to_valid_range(self) -> None:
        net = self._make_net(input_dim=4, hidden_dim=8, output_dim=5)
        x = np.full(4, 1e6)
        logits = net.forward(x)
        self.assertTrue(np.all(logits >= -20.0))
        self.assertTrue(np.all(logits <= 20.0))

    def test_forward_handles_nan_input(self) -> None:
        net = self._make_net(input_dim=4, hidden_dim=8, output_dim=5)
        x = np.array([np.nan, 1.0, -np.inf, np.inf])
        logits = net.forward(x)
        self.assertTrue(np.all(np.isfinite(logits)))

    def test_backward_updates_w1_b1_w2_b2(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=12, output_dim=5)
        x = np.random.default_rng(3).standard_normal(6)
        net.forward(x)
        w1_before = net.W1.copy()
        b1_before = net.b1.copy()
        w2_before = net.W2.copy()
        b2_before = net.b2.copy()
        grad = np.ones(5) * 0.1
        net.backward(grad, lr=0.01)
        self.assertFalse(np.allclose(net.W1, w1_before))
        self.assertFalse(np.allclose(net.b1, b1_before))
        self.assertFalse(np.allclose(net.W2, w2_before))
        self.assertFalse(np.allclose(net.b2, b2_before))

    def test_backward_raises_without_cache(self) -> None:
        net = self._make_net()
        with self.assertRaises(RuntimeError):
            net.backward(np.ones(5), lr=0.01)

    def test_backward_clips_large_gradients(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=12, output_dim=5)
        x = np.random.default_rng(4).standard_normal(6)
        net.forward(x)
        w2_before = net.W2.copy()
        # Very large gradient - should be clipped
        grad = np.full(5, 1e6)
        net.backward(grad, lr=0.001, grad_clip=5.0)
        # Parameters should have changed but not diverged
        self.assertTrue(np.all(np.isfinite(net.W2)))
        self.assertFalse(np.allclose(net.W2, w2_before))


class ProposalNetworkStateDictTest(unittest.TestCase):
    """state_dict and load_state_dict with the new single-layer architecture."""

    def _make_net(self, input_dim: int = 6, hidden_dim: int = 10, output_dim: int = 5) -> ProposalNetwork:
        rng = np.random.default_rng(99)
        return ProposalNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            name="test_net",
        )

    def test_state_dict_contains_expected_keys(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        expected_keys = {"name", "input_dim", "hidden_dim", "output_dim", "W1", "b1", "W2", "b2"}
        self.assertEqual(set(sd.keys()), expected_keys)

    def test_state_dict_does_not_contain_w_mid_b_mid(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        self.assertNotIn("W_mid", sd)
        self.assertNotIn("b_mid", sd)

    def test_state_dict_values_are_copies(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        # Modifying the state dict should not affect the network
        sd["W1"][:] = 999.0
        self.assertFalse(np.allclose(net.W1, 999.0))

    def test_load_state_dict_restores_weights(self) -> None:
        rng = np.random.default_rng(10)
        net1 = ProposalNetwork(input_dim=6, hidden_dim=10, output_dim=5, rng=rng, name="n1")
        net2 = ProposalNetwork(input_dim=6, hidden_dim=10, output_dim=5, rng=rng, name="n1")
        sd = net1.state_dict()
        net2.load_state_dict(sd)
        np.testing.assert_array_equal(net2.W1, net1.W1)
        np.testing.assert_array_equal(net2.W2, net1.W2)

    def test_load_state_dict_clears_cache(self) -> None:
        net = self._make_net()
        x = np.random.default_rng(11).standard_normal(6)
        net.forward(x)
        self.assertIsNotNone(net.cache)
        net.load_state_dict(net.state_dict())
        self.assertIsNone(net.cache)

    def test_load_state_dict_rejects_wrong_w1_shape(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        sd = net.state_dict()
        sd["W1"] = np.zeros((10, 99))  # wrong input_dim
        with self.assertRaises(ValueError):
            net.load_state_dict(sd)

    def test_load_state_dict_rejects_wrong_w2_shape(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        sd = net.state_dict()
        sd["W2"] = np.zeros((99, 10))  # wrong output_dim
        with self.assertRaises(ValueError):
            net.load_state_dict(sd)

    def test_load_state_dict_accepts_state_without_w_mid(self) -> None:
        """Current state dicts do not include the removed W_mid key."""
        net = self._make_net()
        sd = net.state_dict()
        self.assertNotIn("W_mid", sd)
        net.load_state_dict(sd)

    def test_load_state_dict_rejects_legacy_w_mid(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["W_mid"] = np.zeros((10, 10))
        sd["b_mid"] = np.zeros(10)
        with self.assertRaisesRegex(ValueError, "unexpected.*W_mid"):
            net.load_state_dict(sd)

    def test_load_state_dict_rejects_metadata_mismatch(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["input_dim"] = 99
        with self.assertRaisesRegex(ValueError, "input_dim"):
            net.load_state_dict(sd)

    def test_load_state_dict_rejects_name_mismatch(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["name"] = "other_net"
        with self.assertRaisesRegex(ValueError, "name"):
            net.load_state_dict(sd)

    def test_roundtrip_state_dict(self) -> None:
        rng = np.random.default_rng(12)
        net = ProposalNetwork(input_dim=6, hidden_dim=10, output_dim=5, rng=rng, name="roundtrip")
        x = rng.standard_normal(6)
        logits_before = net.forward(x, store_cache=False)
        sd = net.state_dict()
        rng2 = np.random.default_rng(99)  # different rng
        net2 = ProposalNetwork(input_dim=6, hidden_dim=10, output_dim=5, rng=rng2, name="roundtrip")
        net2.load_state_dict(sd)
        logits_after = net2.forward(x, store_cache=False)
        np.testing.assert_array_almost_equal(logits_before, logits_after)


class ProposalNetworkParameterNormTest(unittest.TestCase):
    """parameter_norm() should only include W1, b1, W2, b2."""

    def test_parameter_norm_is_finite_and_positive(self) -> None:
        rng = np.random.default_rng(20)
        net = ProposalNetwork(input_dim=8, hidden_dim=16, output_dim=5, rng=rng, name="norm_test")
        norm = net.parameter_norm()
        self.assertGreater(norm, 0.0)
        self.assertTrue(np.isfinite(norm))

    def test_parameter_norm_zero_weights(self) -> None:
        rng = np.random.default_rng(21)
        net = ProposalNetwork(input_dim=4, hidden_dim=8, output_dim=5, rng=rng, name="zero_net")
        net.W1[:] = 0.0
        net.b1[:] = 0.0
        net.W2[:] = 0.0
        net.b2[:] = 0.0
        self.assertEqual(net.parameter_norm(), 0.0)

    def test_parameter_norm_matches_manual_calculation(self) -> None:
        rng = np.random.default_rng(22)
        net = ProposalNetwork(input_dim=4, hidden_dim=8, output_dim=5, rng=rng, name="manual_norm")
        expected = float(np.sqrt(
            np.sum(net.W1 ** 2)
            + np.sum(net.b1 ** 2)
            + np.sum(net.W2 ** 2)
            + np.sum(net.b2 ** 2)
        ))
        self.assertAlmostEqual(net.parameter_norm(), expected, places=10)


class RecurrentProposalNetworkTest(unittest.TestCase):
    """RecurrentProposalNetwork exposes the same training surface as ProposalNetwork."""

    def _make_net(self, input_dim: int = 6, hidden_dim: int = 10, output_dim: int = 5) -> RecurrentProposalNetwork:
        rng = np.random.default_rng(123)
        return RecurrentProposalNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
            name="recurrent_test",
        )

    def test_initializes_recurrent_parameters_and_hidden_state(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        self.assertEqual(net.W_xh.shape, (10, 6))
        self.assertEqual(net.W_hh.shape, (10, 10))
        self.assertEqual(net.b_h.shape, (10,))
        self.assertEqual(net.W_out.shape, (5, 10))
        self.assertEqual(net.b_out.shape, (5,))
        np.testing.assert_allclose(net.hidden_state, np.zeros(10, dtype=float))

    def test_forward_advances_hidden_state_and_stores_cache(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        x = np.random.default_rng(1).standard_normal(6)
        logits = net.forward(x, store_cache=True)
        self.assertEqual(logits.shape, (5,))
        self.assertIsInstance(net.cache, RecurrentProposalCache)
        np.testing.assert_allclose(net.cache.h_prev, np.zeros(10, dtype=float))
        np.testing.assert_allclose(net.hidden_state, net.cache.h_new)
        self.assertGreater(float(np.linalg.norm(net.hidden_state)), 0.0)

    def test_reset_hidden_state_clears_state_and_cache(self) -> None:
        net = self._make_net()
        net.forward(np.ones(6), store_cache=True)
        self.assertIsNotNone(net.cache)
        net.reset_hidden_state()
        np.testing.assert_allclose(net.hidden_state, np.zeros(10, dtype=float))
        self.assertIsNone(net.cache)

    def test_backward_updates_recurrent_parameters(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        rng = np.random.default_rng(2)
        net.forward(rng.standard_normal(6), store_cache=True)
        net.forward(rng.standard_normal(6), store_cache=True)
        w_xh_before = net.W_xh.copy()
        w_hh_before = net.W_hh.copy()
        w_out_before = net.W_out.copy()
        b_out_before = net.b_out.copy()
        net.backward(np.ones(5) * 0.1, lr=0.01)
        self.assertFalse(np.allclose(net.W_xh, w_xh_before))
        self.assertFalse(np.allclose(net.W_hh, w_hh_before))
        self.assertFalse(np.allclose(net.W_out, w_out_before))
        self.assertFalse(np.allclose(net.b_out, b_out_before))

    def test_state_dict_roundtrip_resets_hidden_state(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        state = net.state_dict()
        self.assertEqual(
            set(state.keys()),
            {"name", "input_dim", "hidden_dim", "output_dim", "W_xh", "W_hh", "b_h", "W_out", "b_out"},
        )
        self.assertNotIn("hidden_state", state)
        net.forward(np.ones(6), store_cache=True)
        self.assertGreater(float(np.linalg.norm(net.hidden_state)), 0.0)
        net.load_state_dict(state)
        np.testing.assert_allclose(net.hidden_state, np.zeros(10, dtype=float))
        self.assertIsNone(net.cache)

    def test_parameter_norm_is_finite_and_positive(self) -> None:
        net = self._make_net()
        norm = net.parameter_norm()
        self.assertGreater(norm, 0.0)
        self.assertTrue(np.isfinite(norm))

    def test_backward_keeps_parameters_finite_under_large_gradient(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        x = np.random.default_rng(4).standard_normal(6)
        net.forward(x, store_cache=True)
        net.backward(np.full(5, 1e6), lr=0.001, grad_clip=5.0)
        self.assertTrue(np.all(np.isfinite(net.W_xh)))
        self.assertTrue(np.all(np.isfinite(net.W_hh)))
        self.assertTrue(np.all(np.isfinite(net.b_h)))
        self.assertTrue(np.all(np.isfinite(net.W_out)))
        self.assertTrue(np.all(np.isfinite(net.b_out)))

    def test_get_hidden_state_returns_copy(self) -> None:
        net = self._make_net()
        net.hidden_state[:] = 0.25
        snapshot = net.get_hidden_state()
        snapshot[:] = 0.9
        np.testing.assert_allclose(net.hidden_state, np.full(10, 0.25, dtype=float))

    def test_set_hidden_state_validates_shape_and_copies_input(self) -> None:
        net = self._make_net(hidden_dim=10)
        state = np.full(10, 0.4, dtype=float)
        net.set_hidden_state(state)
        state[:] = 0.8
        np.testing.assert_allclose(net.hidden_state, np.full(10, 0.4, dtype=float))
        with self.assertRaises(ValueError):
            net.set_hidden_state(np.zeros(9, dtype=float))

    def test_forward_with_store_cache_false_leaves_cache_unchanged(self) -> None:
        net = self._make_net()
        # Cache should remain None if store_cache=False from the start
        logits = net.forward(np.ones(6), store_cache=False)
        self.assertIsNone(net.cache)
        self.assertEqual(logits.shape, (5,))

    def test_forward_with_store_cache_false_after_cache_set_leaves_old_cache(self) -> None:
        net = self._make_net()
        rng = np.random.default_rng(55)
        net.forward(rng.standard_normal(6), store_cache=True)
        old_cache = net.cache
        self.assertIsNotNone(old_cache)
        net.forward(rng.standard_normal(6), store_cache=False)
        # Cache should not be updated
        self.assertIs(net.cache, old_cache)

    def test_h_prev_in_second_forward_matches_h_new_from_first_forward(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        rng = np.random.default_rng(66)
        net.forward(rng.standard_normal(6), store_cache=True)
        h_new_first = net.cache.h_new.copy()
        net.forward(rng.standard_normal(6), store_cache=True)
        np.testing.assert_allclose(net.cache.h_prev, h_new_first)

    def test_forward_handles_nan_inf_inputs(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        x = np.array([np.nan, np.inf, -np.inf, 0.5, -0.5, 0.0])
        logits = net.forward(x, store_cache=False)
        self.assertTrue(np.all(np.isfinite(logits)), "NaN/inf inputs should produce finite logits")

    def test_forward_rejects_wrong_vector_shape(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        with self.assertRaisesRegex(ValueError, r"x expected shape"):
            net.forward(np.zeros(5), store_cache=False)

    def test_forward_rejects_column_vector_input(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        with self.assertRaisesRegex(ValueError, r"x expected shape"):
            net.forward(np.zeros((6, 1)), store_cache=False)

    def test_forward_logits_are_clipped_to_valid_range(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        # Force weights very large to produce extreme pre-clip outputs
        net.W_out[:] = 1e6
        net.b_out[:] = 1e6
        x = np.ones(6)
        logits = net.forward(x, store_cache=False)
        self.assertTrue(np.all(logits >= -20.0))
        self.assertTrue(np.all(logits <= 20.0))

    def test_backward_raises_runtime_error_without_cache(self) -> None:
        net = self._make_net()
        with self.assertRaises(RuntimeError):
            net.backward(np.ones(5), lr=0.01)

    def test_backward_rejects_wrong_grad_logits_shape(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        net.forward(np.ones(6), store_cache=True)
        with self.assertRaisesRegex(ValueError, r"grad_logits expected shape"):
            net.backward(np.ones(4), lr=0.01)

    def test_backward_rejects_column_vector_grad_logits(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        net.forward(np.ones(6), store_cache=True)
        with self.assertRaisesRegex(ValueError, r"grad_logits"):
            net.backward(np.ones((5, 1)), lr=0.01)

    def test_load_state_dict_rejects_wrong_w_xh_shape(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        state = net.state_dict()
        state["W_xh"] = np.zeros((10, 99))  # wrong input_dim
        with self.assertRaises(ValueError):
            net.load_state_dict(state)

    def test_load_state_dict_rejects_wrong_w_hh_shape(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        state = net.state_dict()
        state["W_hh"] = np.zeros((10, 99))  # should be (hidden_dim, hidden_dim)
        with self.assertRaises(ValueError):
            net.load_state_dict(state)

    def test_load_state_dict_rejects_wrong_w_out_shape(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        state = net.state_dict()
        state["W_out"] = np.zeros((99, 10))  # wrong output_dim
        with self.assertRaises(ValueError):
            net.load_state_dict(state)

    def test_load_state_dict_failure_does_not_partially_mutate_parameters(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        original_W_xh = net.W_xh.copy()
        original_W_hh = net.W_hh.copy()
        original_b_h = net.b_h.copy()
        original_W_out = net.W_out.copy()
        original_b_out = net.b_out.copy()
        net.hidden_state[:] = 0.7
        net.forward(np.ones(6), store_cache=True)
        original_hidden_state = net.hidden_state.copy()
        original_cache = net.cache

        state = net.state_dict()
        state["W_xh"] = np.full((10, 6), 11.0, dtype=float)
        state["W_hh"] = np.full((10, 10), 12.0, dtype=float)
        state["b_h"] = np.full(10, 13.0, dtype=float)
        state["W_out"] = np.zeros((99, 10), dtype=float)

        with self.assertRaises(ValueError):
            net.load_state_dict(state)

        np.testing.assert_allclose(net.W_xh, original_W_xh)
        np.testing.assert_allclose(net.W_hh, original_W_hh)
        np.testing.assert_allclose(net.b_h, original_b_h)
        np.testing.assert_allclose(net.W_out, original_W_out)
        np.testing.assert_allclose(net.b_out, original_b_out)
        np.testing.assert_allclose(net.hidden_state, original_hidden_state)
        self.assertIs(net.cache, original_cache)

    def test_load_state_dict_rejects_metadata_mismatch(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        state = net.state_dict()
        state["input_dim"] = 99
        with self.assertRaisesRegex(ValueError, "input_dim"):
            net.load_state_dict(state)

    def test_load_state_dict_rejects_name_mismatch(self) -> None:
        net = self._make_net()
        state = net.state_dict()
        state["name"] = "wrong_name"
        with self.assertRaisesRegex(ValueError, "name"):
            net.load_state_dict(state)

    def test_load_state_dict_rejects_unexpected_keys(self) -> None:
        net = self._make_net()
        state = net.state_dict()
        state["extra_key"] = np.zeros(5)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            net.load_state_dict(state)

    def test_parameter_norm_matches_manual_calculation(self) -> None:
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        expected = float(np.sqrt(
            np.sum(net.W_xh ** 2)
            + np.sum(net.W_hh ** 2)
            + np.sum(net.b_h ** 2)
            + np.sum(net.W_out ** 2)
            + np.sum(net.b_out ** 2)
        ))
        self.assertAlmostEqual(net.parameter_norm(), expected, places=10)

    def test_state_dict_values_are_copies(self) -> None:
        net = self._make_net()
        state = net.state_dict()
        original_val = state["W_xh"][0, 0]
        state["W_xh"][0, 0] = 999.0
        self.assertAlmostEqual(net.W_xh[0, 0], original_val,
                               msg="Modifying state dict should not affect network parameters")

    def test_forward_is_deterministic_given_same_weights_and_hidden_state(self) -> None:
        net = self._make_net()
        x = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        # Reset to known hidden state
        net.reset_hidden_state()
        out1 = net.forward(x, store_cache=False)
        net.reset_hidden_state()
        out2 = net.forward(x, store_cache=False)
        np.testing.assert_array_equal(out1, out2)

    def test_load_state_dict_roundtrip_preserves_forward_outputs(self) -> None:
        rng = np.random.default_rng(77)
        net = self._make_net(input_dim=6, hidden_dim=10, output_dim=5)
        x = rng.standard_normal(6)
        net.reset_hidden_state()
        logits_before = net.forward(x, store_cache=False)
        state = net.state_dict()
        rng2 = np.random.default_rng(999)
        net2 = RecurrentProposalNetwork(input_dim=6, hidden_dim=10, output_dim=5, rng=rng2, name="recurrent_test")
        net2.load_state_dict(state)
        logits_after = net2.forward(x, store_cache=False)
        np.testing.assert_array_almost_equal(logits_before, logits_after)


class MotorNetworkArchitectureTest(unittest.TestCase):
    """MotorNetwork has a single hidden layer: W1/b1 then W2_policy/W2_value, no W_mid/b_mid."""

    def _make_net(self, input_dim: int = 10, hidden_dim: int = 16, output_dim: int = 5) -> MotorNetwork:
        rng = np.random.default_rng(50)
        return MotorNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            rng=rng,
        )

    def test_has_w1_b1_w2_policy_w2_value(self) -> None:
        net = self._make_net()
        self.assertTrue(hasattr(net, "W1"))
        self.assertTrue(hasattr(net, "b1"))
        self.assertTrue(hasattr(net, "W2_policy"))
        self.assertTrue(hasattr(net, "b2_policy"))
        self.assertTrue(hasattr(net, "W2_value"))
        self.assertTrue(hasattr(net, "b2_value"))

    def test_does_not_have_w_mid_b_mid(self) -> None:
        net = self._make_net()
        self.assertFalse(hasattr(net, "W_mid"), "MotorNetwork should not have W_mid after PR change")
        self.assertFalse(hasattr(net, "b_mid"), "MotorNetwork should not have b_mid after PR change")

    def test_w1_shape(self) -> None:
        net = self._make_net(input_dim=10, hidden_dim=16, output_dim=5)
        self.assertEqual(net.W1.shape, (16, 10))
        self.assertEqual(net.b1.shape, (16,))

    def test_w2_policy_shape(self) -> None:
        net = self._make_net(input_dim=10, hidden_dim=16, output_dim=5)
        self.assertEqual(net.W2_policy.shape, (5, 16))
        self.assertEqual(net.b2_policy.shape, (5,))

    def test_w2_value_shape(self) -> None:
        net = self._make_net(input_dim=10, hidden_dim=16, output_dim=5)
        self.assertEqual(net.W2_value.shape, (1, 16))
        self.assertEqual(net.b2_value.shape, (1,))

    def test_forward_returns_policy_logits_and_value(self) -> None:
        net = self._make_net(input_dim=10, hidden_dim=16, output_dim=5)
        x = np.random.default_rng(51).standard_normal(10)
        policy_logits, value = net.forward(x)
        self.assertEqual(policy_logits.shape, (5,))
        self.assertIsInstance(value, float)

    def test_forward_stores_motor_cache_with_h_field(self) -> None:
        net = self._make_net(input_dim=10, hidden_dim=16, output_dim=5)
        x = np.random.default_rng(52).standard_normal(10)
        net.forward(x, store_cache=True)
        self.assertIsNotNone(net.cache)
        self.assertIsInstance(net.cache, MotorCache)
        self.assertEqual(net.cache.h.shape, (16,))
        self.assertFalse(hasattr(net.cache, "h1"))
        self.assertFalse(hasattr(net.cache, "h2"))

    def test_forward_logits_are_clipped(self) -> None:
        net = self._make_net(input_dim=5, hidden_dim=8, output_dim=5)
        x = np.full(5, 1e6)
        logits, value = net.forward(x)
        self.assertTrue(np.all(logits >= -20.0))
        self.assertTrue(np.all(logits <= 20.0))
        self.assertTrue(np.isfinite(value))

    def test_forward_handles_nan_inf_input(self) -> None:
        net = self._make_net(input_dim=5, hidden_dim=8, output_dim=5)
        x = np.array([np.nan, np.inf, -np.inf, 0.5, -0.5])
        logits, value = net.forward(x)
        self.assertTrue(np.all(np.isfinite(logits)))
        self.assertTrue(np.isfinite(value))

    def test_backward_updates_all_parameters(self) -> None:
        net = self._make_net(input_dim=10, hidden_dim=16, output_dim=5)
        x = np.random.default_rng(53).standard_normal(10)
        net.forward(x)
        w1_before = net.W1.copy()
        w2p_before = net.W2_policy.copy()
        w2v_before = net.W2_value.copy()
        net.backward(
            grad_policy_logits=np.ones(5) * 0.1,
            grad_value=0.1,
            lr=0.01,
        )
        self.assertFalse(np.allclose(net.W1, w1_before))
        self.assertFalse(np.allclose(net.W2_policy, w2p_before))
        self.assertFalse(np.allclose(net.W2_value, w2v_before))

    def test_backward_raises_without_cache(self) -> None:
        net = self._make_net()
        with self.assertRaises(RuntimeError):
            net.backward(np.ones(5), 0.1, lr=0.01)

    def test_backward_returns_grad_x(self) -> None:
        net = self._make_net(input_dim=10, hidden_dim=16, output_dim=5)
        x = np.random.default_rng(54).standard_normal(10)
        net.forward(x)
        grad_x = net.backward(
            grad_policy_logits=np.ones(5) * 0.1,
            grad_value=0.1,
            lr=0.01,
        )
        self.assertEqual(grad_x.shape, (10,))
        self.assertTrue(np.all(np.isfinite(grad_x)))


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


class MotorNetworkStateDictTest(unittest.TestCase):
    """state_dict and load_state_dict with the new single-layer architecture."""

    def _make_net(self) -> MotorNetwork:
        rng = np.random.default_rng(70)
        return MotorNetwork(input_dim=10, hidden_dim=16, output_dim=5, rng=rng)

    def test_state_dict_contains_expected_keys(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        expected_keys = {
            "name", "input_dim", "hidden_dim", "output_dim",
            "W1", "b1",
            "W2_policy", "b2_policy",
            "W2_value", "b2_value",
        }
        self.assertEqual(set(sd.keys()), expected_keys)

    def test_state_dict_does_not_contain_w_mid_b_mid(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        self.assertNotIn("W_mid", sd)
        self.assertNotIn("b_mid", sd)

    def test_load_state_dict_restores_weights(self) -> None:
        rng = np.random.default_rng(71)
        net1 = MotorNetwork(input_dim=10, hidden_dim=16, output_dim=5, rng=rng)
        rng2 = np.random.default_rng(72)
        net2 = MotorNetwork(input_dim=10, hidden_dim=16, output_dim=5, rng=rng2)
        net2.load_state_dict(net1.state_dict())
        np.testing.assert_array_equal(net2.W1, net1.W1)
        np.testing.assert_array_equal(net2.W2_policy, net1.W2_policy)
        np.testing.assert_array_equal(net2.W2_value, net1.W2_value)

    def test_load_state_dict_clears_cache(self) -> None:
        net = self._make_net()
        x = np.random.default_rng(73).standard_normal(10)
        net.forward(x)
        self.assertIsNotNone(net.cache)
        net.load_state_dict(net.state_dict())
        self.assertIsNone(net.cache)

    def test_load_state_dict_rejects_wrong_w1_shape(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["W1"] = np.zeros((16, 99))  # wrong input_dim
        with self.assertRaises(ValueError):
            net.load_state_dict(sd)

    def test_load_state_dict_rejects_wrong_w2_policy_shape(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["W2_policy"] = np.zeros((99, 16))  # wrong output_dim
        with self.assertRaises(ValueError):
            net.load_state_dict(sd)

    def test_load_state_dict_accepts_state_without_w_mid(self) -> None:
        """Regression: load_state_dict must not expect W_mid after the PR change."""
        net = self._make_net()
        sd = net.state_dict()
        self.assertNotIn("W_mid", sd)
        net.load_state_dict(sd)

    def test_load_state_dict_rejects_legacy_w_mid(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["W_mid"] = np.zeros((16, 16))
        sd["b_mid"] = np.zeros(16)
        with self.assertRaisesRegex(ValueError, "unexpected.*W_mid"):
            net.load_state_dict(sd)

    def test_load_state_dict_rejects_metadata_mismatch(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["hidden_dim"] = 99
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            net.load_state_dict(sd)

    def test_roundtrip_state_dict(self) -> None:
        rng = np.random.default_rng(74)
        net = MotorNetwork(input_dim=10, hidden_dim=16, output_dim=5, rng=rng)
        x = rng.standard_normal(10)
        logits_before, value_before = net.forward(x, store_cache=False)
        rng2 = np.random.default_rng(999)
        net2 = MotorNetwork(input_dim=10, hidden_dim=16, output_dim=5, rng=rng2)
        net2.load_state_dict(net.state_dict())
        logits_after, value_after = net2.forward(x, store_cache=False)
        np.testing.assert_array_almost_equal(logits_before, logits_after)
        self.assertAlmostEqual(value_before, value_after, places=10)


class MotorNetworkParameterNormTest(unittest.TestCase):
    """parameter_norm() should only include W1, b1, W2_policy, b2_policy, W2_value, b2_value."""

    def test_parameter_norm_is_positive_and_finite(self) -> None:
        rng = np.random.default_rng(80)
        net = MotorNetwork(input_dim=10, hidden_dim=16, output_dim=5, rng=rng)
        norm = net.parameter_norm()
        self.assertGreater(norm, 0.0)
        self.assertTrue(np.isfinite(norm))

    def test_parameter_norm_zero_weights(self) -> None:
        rng = np.random.default_rng(81)
        net = MotorNetwork(input_dim=5, hidden_dim=8, output_dim=5, rng=rng)
        net.W1[:] = 0.0
        net.b1[:] = 0.0
        net.W2_policy[:] = 0.0
        net.b2_policy[:] = 0.0
        net.W2_value[:] = 0.0
        net.b2_value[:] = 0.0
        self.assertEqual(net.parameter_norm(), 0.0)

    def test_parameter_norm_matches_manual_calculation(self) -> None:
        rng = np.random.default_rng(82)
        net = MotorNetwork(input_dim=5, hidden_dim=8, output_dim=5, rng=rng)
        expected = float(np.sqrt(
            np.sum(net.W1 ** 2)
            + np.sum(net.b1 ** 2)
            + np.sum(net.W2_policy ** 2)
            + np.sum(net.b2_policy ** 2)
            + np.sum(net.W2_value ** 2)
            + np.sum(net.b2_value ** 2)
        ))
        self.assertAlmostEqual(net.parameter_norm(), expected, places=10)


class ArbitrationNetworkArchitectureTest(unittest.TestCase):
    """ArbitrationNetwork has a shared hidden layer with valence, gate, and value heads."""

    def _make_net(self, hidden_dim: int = 16) -> ArbitrationNetwork:
        """
        Constructs an ArbitrationNetwork configured for tests using a deterministic RNG.
        
        Parameters:
            hidden_dim (int): Size of the shared hidden layer.
        
        Returns:
            ArbitrationNetwork: An instance with input_dim set to ArbitrationNetwork.INPUT_DIM and the given hidden_dim; RNG is deterministic for repeatable tests.
        """
        rng = np.random.default_rng(90)
        return ArbitrationNetwork(
            input_dim=ArbitrationNetwork.INPUT_DIM,
            hidden_dim=hidden_dim,
            rng=rng,
        )

    def test_input_dim_matches_concatenated_evidence_vector(self) -> None:
        self.assertEqual(ArbitrationNetwork.INPUT_DIM, 24)
        self.assertEqual(len(ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES), 24)

    def test_rejects_wrong_input_dim(self) -> None:
        rng = np.random.default_rng(91)
        with self.assertRaisesRegex(ValueError, "input_dim"):
            ArbitrationNetwork(input_dim=23, hidden_dim=16, rng=rng)

    def test_has_shared_trunk_and_three_heads(self) -> None:
        net = self._make_net(hidden_dim=12)
        self.assertEqual(net.W1.shape, (12, ArbitrationNetwork.INPUT_DIM))
        self.assertEqual(net.b1.shape, (12,))
        self.assertEqual(net.W2_valence.shape, (4, 12))
        self.assertEqual(net.b2_valence.shape, (4,))
        self.assertEqual(net.W2_gate.shape, (6, 12))
        self.assertEqual(net.b2_gate.shape, (6,))
        self.assertEqual(net.W2_value.shape, (1, 12))
        self.assertEqual(net.b2_value.shape, (1,))

    def test_forward_returns_valence_logits_gate_adjustments_and_value(self) -> None:
        net = self._make_net()
        x = np.random.default_rng(92).standard_normal(ArbitrationNetwork.INPUT_DIM)
        valence_logits, gate_adjustments, value = net.forward(x)
        self.assertEqual(valence_logits.shape, (4,))
        self.assertEqual(gate_adjustments.shape, (6,))
        self.assertIsInstance(value, float)

    def test_forward_stores_arbitration_cache_with_h_field(self) -> None:
        net = self._make_net()
        x = np.random.default_rng(93).standard_normal(ArbitrationNetwork.INPUT_DIM)
        net.forward(x, store_cache=True)
        self.assertIsNotNone(net.cache)
        self.assertIsInstance(net.cache, ArbitrationCache)
        self.assertEqual(net.cache.x.shape, (ArbitrationNetwork.INPUT_DIM,))
        self.assertEqual(net.cache.h.shape, (16,))
        self.assertFalse(hasattr(net.cache, "h1"))
        self.assertFalse(hasattr(net.cache, "h2"))

    def test_forward_no_cache_does_not_store_cache(self) -> None:
        net = self._make_net()
        x = np.random.default_rng(94).standard_normal(ArbitrationNetwork.INPUT_DIM)
        net.forward(x, store_cache=False)
        self.assertIsNone(net.cache)

    def test_gate_adjustments_are_constrained(self) -> None:
        net = self._make_net()
        net.W2_gate[:] = 100.0
        net.b2_gate[:] = 100.0
        x = np.ones(ArbitrationNetwork.INPUT_DIM)
        _, high_adjustments, _ = net.forward(x, store_cache=False)
        self.assertTrue(np.all(high_adjustments >= ArbitrationNetwork.GATE_ADJUSTMENT_MIN))
        self.assertTrue(np.all(high_adjustments <= ArbitrationNetwork.GATE_ADJUSTMENT_MAX))

        net.W2_gate[:] = -100.0
        net.b2_gate[:] = -100.0
        _, low_adjustments, _ = net.forward(x, store_cache=False)
        self.assertTrue(np.all(low_adjustments >= ArbitrationNetwork.GATE_ADJUSTMENT_MIN))
        self.assertTrue(np.all(low_adjustments <= ArbitrationNetwork.GATE_ADJUSTMENT_MAX))

    def test_forward_handles_nan_inf_input(self) -> None:
        net = self._make_net()
        x = np.zeros(ArbitrationNetwork.INPUT_DIM)
        x[:4] = [np.nan, np.inf, -np.inf, 0.5]
        valence_logits, gate_adjustments, value = net.forward(x, store_cache=False)
        self.assertTrue(np.all(np.isfinite(valence_logits)))
        self.assertTrue(np.all(np.isfinite(gate_adjustments)))
        self.assertTrue(np.isfinite(value))

    def test_forward_rejects_wrong_vector_shape(self) -> None:
        net = self._make_net()
        with self.assertRaisesRegex(ValueError, "evidence_vector"):
            net.forward(np.zeros(ArbitrationNetwork.INPUT_DIM - 1))

    def test_backward_updates_all_heads_and_returns_grad_x(self) -> None:
        net = self._make_net()
        x = np.random.default_rng(95).standard_normal(ArbitrationNetwork.INPUT_DIM)
        net.forward(x)
        w1_before = net.W1.copy()
        w2v_before = net.W2_valence.copy()
        w2g_before = net.W2_gate.copy()
        w2value_before = net.W2_value.copy()
        grad_x = net.backward(
            grad_valence_logits=np.ones(4) * 0.1,
            grad_gate_adjustments=np.ones(6) * 0.1,
            grad_value=0.1,
            lr=0.01,
        )
        self.assertEqual(grad_x.shape, (ArbitrationNetwork.INPUT_DIM,))
        self.assertTrue(np.all(np.isfinite(grad_x)))
        self.assertFalse(np.allclose(net.W1, w1_before))
        self.assertFalse(np.allclose(net.W2_valence, w2v_before))
        self.assertFalse(np.allclose(net.W2_gate, w2g_before))
        self.assertFalse(np.allclose(net.W2_value, w2value_before))

    def test_backward_raises_without_cache(self) -> None:
        net = self._make_net()
        with self.assertRaises(RuntimeError):
            net.backward(np.ones(4), np.ones(6), 0.1, lr=0.01)


class ArbitrationNetworkStateDictTest(unittest.TestCase):
    """state_dict and load_state_dict for the arbitration network."""

    def _make_net(self) -> ArbitrationNetwork:
        """
        Constructs a deterministic ArbitrationNetwork instance for tests.
        
        Returns:
            ArbitrationNetwork: An instance with input_dim set to ArbitrationNetwork.INPUT_DIM,
            hidden_dim set to 16, and a fixed RNG seeded for deterministic behavior.
        """
        rng = np.random.default_rng(110)
        return ArbitrationNetwork(
            input_dim=ArbitrationNetwork.INPUT_DIM,
            hidden_dim=16,
            rng=rng,
        )

    def test_state_dict_contains_expected_keys(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        expected_keys = {
            "name", "input_dim", "hidden_dim", "valence_dim", "gate_dim",
            "gate_adjustment_min", "gate_adjustment_max",
            "W1", "b1",
            "W2_valence", "b2_valence",
            "W2_gate", "b2_gate",
            "W2_value", "b2_value",
        }
        self.assertEqual(set(sd.keys()), expected_keys)

    def test_load_state_dict_restores_weights(self) -> None:
        rng1 = np.random.default_rng(111)
        net1 = ArbitrationNetwork(input_dim=ArbitrationNetwork.INPUT_DIM, hidden_dim=16, rng=rng1)
        rng2 = np.random.default_rng(112)
        net2 = ArbitrationNetwork(input_dim=ArbitrationNetwork.INPUT_DIM, hidden_dim=16, rng=rng2)
        net2.load_state_dict(net1.state_dict())
        np.testing.assert_array_equal(net2.W1, net1.W1)
        np.testing.assert_array_equal(net2.W2_valence, net1.W2_valence)
        np.testing.assert_array_equal(net2.W2_gate, net1.W2_gate)
        np.testing.assert_array_equal(net2.W2_value, net1.W2_value)

    def test_load_state_dict_clears_cache(self) -> None:
        net = self._make_net()
        x = np.random.default_rng(113).standard_normal(ArbitrationNetwork.INPUT_DIM)
        net.forward(x)
        self.assertIsNotNone(net.cache)
        net.load_state_dict(net.state_dict())
        self.assertIsNone(net.cache)

    def test_load_state_dict_rejects_wrong_gate_shape(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["W2_gate"] = np.zeros((99, 16))
        with self.assertRaises(ValueError):
            net.load_state_dict(sd)

    def test_load_state_dict_rejects_metadata_mismatch(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["gate_dim"] = 99
        with self.assertRaisesRegex(ValueError, "gate_dim"):
            net.load_state_dict(sd)

    def test_load_state_dict_rejects_unexpected_keys(self) -> None:
        net = self._make_net()
        sd = net.state_dict()
        sd["W_mid"] = np.zeros((16, 16))
        with self.assertRaisesRegex(ValueError, "unexpected.*W_mid"):
            net.load_state_dict(sd)

    def test_roundtrip_state_dict(self) -> None:
        """
        Verifies that saving and loading a network's state_dict preserves forward outputs for the same input.
        
        Creates an ArbitrationNetwork, computes forward outputs for a fixed input, loads the saved state into a newly constructed network, and asserts that the valence logits and gate adjustments match elementwise and the scalar value matches within numerical tolerance.
        """
        rng = np.random.default_rng(114)
        net = ArbitrationNetwork(input_dim=ArbitrationNetwork.INPUT_DIM, hidden_dim=16, rng=rng)
        x = rng.standard_normal(ArbitrationNetwork.INPUT_DIM)
        before = net.forward(x, store_cache=False)
        net2 = ArbitrationNetwork(
            input_dim=ArbitrationNetwork.INPUT_DIM,
            hidden_dim=16,
            rng=np.random.default_rng(115),
        )
        net2.load_state_dict(net.state_dict())
        after = net2.forward(x, store_cache=False)
        np.testing.assert_array_almost_equal(before[0], after[0])
        np.testing.assert_array_almost_equal(before[1], after[1])
        self.assertAlmostEqual(before[2], after[2], places=10)


class ArbitrationNetworkParameterNormTest(unittest.TestCase):
    """parameter_norm() should include all arbitration heads and the shared trunk."""

    def test_parameter_norm_is_positive_and_finite(self) -> None:
        rng = np.random.default_rng(120)
        net = ArbitrationNetwork(input_dim=ArbitrationNetwork.INPUT_DIM, hidden_dim=16, rng=rng)
        norm = net.parameter_norm()
        self.assertGreater(norm, 0.0)
        self.assertTrue(np.isfinite(norm))

    def test_parameter_norm_zero_weights(self) -> None:
        rng = np.random.default_rng(121)
        net = ArbitrationNetwork(input_dim=ArbitrationNetwork.INPUT_DIM, hidden_dim=8, rng=rng)
        net.W1[:] = 0.0
        net.b1[:] = 0.0
        net.W2_valence[:] = 0.0
        net.b2_valence[:] = 0.0
        net.W2_gate[:] = 0.0
        net.b2_gate[:] = 0.0
        net.W2_value[:] = 0.0
        net.b2_value[:] = 0.0
        self.assertEqual(net.parameter_norm(), 0.0)

    def test_parameter_norm_matches_manual_calculation(self) -> None:
        rng = np.random.default_rng(122)
        net = ArbitrationNetwork(input_dim=ArbitrationNetwork.INPUT_DIM, hidden_dim=8, rng=rng)
        expected = float(np.sqrt(
            np.sum(net.W1 ** 2)
            + np.sum(net.b1 ** 2)
            + np.sum(net.W2_valence ** 2)
            + np.sum(net.b2_valence ** 2)
            + np.sum(net.W2_gate ** 2)
            + np.sum(net.b2_gate ** 2)
            + np.sum(net.W2_value ** 2)
            + np.sum(net.b2_value ** 2)
        ))
        self.assertAlmostEqual(net.parameter_norm(), expected, places=10)


class ProposalNetworkWeightInitTest(unittest.TestCase):
    """Weight initialization uses scale = 0.35 / sqrt(dim) for the single layer."""

    def test_w1_scale_inversely_proportional_to_input_dim(self) -> None:
        rng1 = np.random.default_rng(100)
        net_small = ProposalNetwork(input_dim=4, hidden_dim=8, output_dim=5, rng=rng1, name="small")
        rng2 = np.random.default_rng(100)
        net_large = ProposalNetwork(input_dim=100, hidden_dim=8, output_dim=5, rng=rng2, name="large")
        # Larger input_dim → smaller initial W1 spread
        self.assertGreater(np.std(net_small.W1), np.std(net_large.W1))

    def test_w2_scale_inversely_proportional_to_hidden_dim(self) -> None:
        rng1 = np.random.default_rng(101)
        net_small = ProposalNetwork(input_dim=8, hidden_dim=4, output_dim=5, rng=rng1, name="small_h")
        rng2 = np.random.default_rng(101)
        net_large = ProposalNetwork(input_dim=8, hidden_dim=100, output_dim=5, rng=rng2, name="large_h")
        self.assertGreater(np.std(net_small.W2), np.std(net_large.W2))


class ProposalNetworkForwardDeterminismTest(unittest.TestCase):
    """Forward pass should be deterministic given fixed weights."""

    def test_same_input_gives_same_output(self) -> None:
        rng = np.random.default_rng(200)
        net = ProposalNetwork(input_dim=8, hidden_dim=16, output_dim=5, rng=rng, name="det_test")
        x = np.array([0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8])
        out1 = net.forward(x, store_cache=False)
        out2 = net.forward(x, store_cache=False)
        np.testing.assert_array_equal(out1, out2)


class MotorNetworkValueOnlyTest(unittest.TestCase):
    """value_only() should agree with forward() and not store a cache."""

    def test_value_only_matches_forward_value(self) -> None:
        rng = np.random.default_rng(300)
        net = MotorNetwork(input_dim=10, hidden_dim=16, output_dim=5, rng=rng)
        x = rng.standard_normal(10)
        _, value_from_forward = net.forward(x, store_cache=False)
        value_only = net.value_only(x)
        self.assertAlmostEqual(value_from_forward, value_only, places=10)

    def test_value_only_does_not_store_cache(self) -> None:
        rng = np.random.default_rng(301)
        net = MotorNetwork(input_dim=10, hidden_dim=16, output_dim=5, rng=rng)
        net.value_only(rng.standard_normal(10))
        self.assertIsNone(net.cache)


class SigmoidFunctionTest(unittest.TestCase):
    """Tests for the _sigmoid helper added in this PR."""

    def setUp(self) -> None:
        from spider_cortex_sim.nn import _sigmoid
        self._sigmoid = _sigmoid

    def test_zero_input_gives_half(self) -> None:
        result = self._sigmoid(np.array([0.0]))
        self.assertAlmostEqual(float(result[0]), 0.5, places=10)

    def test_large_positive_approaches_one(self) -> None:
        result = self._sigmoid(np.array([100.0]))
        self.assertGreater(float(result[0]), 0.999)
        self.assertLessEqual(float(result[0]), 1.0)

    def test_large_negative_approaches_zero(self) -> None:
        result = self._sigmoid(np.array([-100.0]))
        self.assertLess(float(result[0]), 0.001)
        self.assertGreaterEqual(float(result[0]), 0.0)

    def test_output_always_in_open_unit_interval(self) -> None:
        x = np.linspace(-10.0, 10.0, 50)
        result = self._sigmoid(x)
        self.assertTrue(np.all(result > 0.0))
        self.assertTrue(np.all(result < 1.0))

    def test_nan_input_treated_as_zero(self) -> None:
        result = self._sigmoid(np.array([np.nan]))
        self.assertAlmostEqual(float(result[0]), 0.5, places=10)

    def test_posinf_input_clamped_to_near_one(self) -> None:
        result = self._sigmoid(np.array([np.inf]))
        self.assertGreater(float(result[0]), 0.999)

    def test_neginf_input_clamped_to_near_zero(self) -> None:
        result = self._sigmoid(np.array([-np.inf]))
        self.assertLess(float(result[0]), 0.001)

    def test_scalar_array_works(self) -> None:
        result = self._sigmoid(np.array(1.0))
        self.assertTrue(np.isfinite(result))
        expected = 1.0 / (1.0 + np.exp(-1.0))
        self.assertAlmostEqual(float(result), expected, places=8)

    def test_output_is_finite_for_extreme_inputs(self) -> None:
        x = np.array([-1e9, 0.0, 1e9, np.nan, np.inf, -np.inf])
        result = self._sigmoid(x)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_output_shape_matches_input(self) -> None:
        x = np.arange(12, dtype=float).reshape(3, 4)
        result = self._sigmoid(x)
        self.assertEqual(result.shape, (3, 4))

    def test_monotonically_increasing(self) -> None:
        x = np.linspace(-5.0, 5.0, 20)
        result = self._sigmoid(x)
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


if __name__ == "__main__":
    unittest.main()
