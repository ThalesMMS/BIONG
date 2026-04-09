"""Tests for the simplified single-hidden-layer ProposalNetwork and MotorNetwork.

This PR removed the intermediate W_mid/b_mid layer from both networks, reducing
them from two hidden layers to one. Tests here verify the new architecture.
"""

from __future__ import annotations

import unittest

import numpy as np

from spider_cortex_sim.nn import (
    MotorCache,
    MotorNetwork,
    ProposalCache,
    ProposalNetwork,
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
        net2 = ProposalNetwork(input_dim=6, hidden_dim=10, output_dim=5, rng=rng, name="n2")
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

    def test_load_state_dict_does_not_check_w_mid(self) -> None:
        """load_state_dict should not raise for absence of W_mid (regression test)."""
        net = self._make_net()
        sd = net.state_dict()
        self.assertNotIn("W_mid", sd)
        # Should succeed without any W_mid key
        net.load_state_dict(sd)

    def test_roundtrip_state_dict(self) -> None:
        rng = np.random.default_rng(12)
        net = ProposalNetwork(input_dim=6, hidden_dim=10, output_dim=5, rng=rng, name="roundtrip")
        x = rng.standard_normal(6)
        logits_before = net.forward(x, store_cache=False)
        sd = net.state_dict()
        rng2 = np.random.default_rng(99)  # different rng
        net2 = ProposalNetwork(input_dim=6, hidden_dim=10, output_dim=5, rng=rng2, name="roundtrip2")
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

    def test_load_state_dict_does_not_require_w_mid(self) -> None:
        """Regression: load_state_dict must not expect W_mid after the PR change."""
        net = self._make_net()
        sd = net.state_dict()
        self.assertNotIn("W_mid", sd)
        # Should succeed with no W_mid key in the dict
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


if __name__ == "__main__":
    unittest.main()