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
