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

    def test_count_parameters_matches_manual_calculation(self) -> None:
        net = MotorNetwork(
            input_dim=5,
            hidden_dim=8,
            output_dim=5,
            rng=np.random.default_rng(83),
        )
        expected = (
            net.W1.size
            + net.b1.size
            + net.W2_policy.size
            + net.b2_policy.size
            + net.W2_value.size
            + net.b2_value.size
        )
        self.assertEqual(net.count_parameters(), expected)

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
        self.assertEqual(net.W2_gate.shape, (ArbitrationNetwork.GATE_DIM, 12))
        self.assertEqual(net.b2_gate.shape, (ArbitrationNetwork.GATE_DIM,))
        self.assertEqual(net.W2_value.shape, (1, 12))
        self.assertEqual(net.b2_value.shape, (1,))

    def test_forward_returns_valence_logits_gate_adjustments_and_value(self) -> None:
        net = self._make_net()
        x = np.random.default_rng(92).standard_normal(ArbitrationNetwork.INPUT_DIM)
        valence_logits, gate_adjustments, value = net.forward(x)
        self.assertEqual(valence_logits.shape, (4,))
        self.assertEqual(gate_adjustments.shape, (ArbitrationNetwork.GATE_DIM,))
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
            grad_gate_adjustments=np.ones(ArbitrationNetwork.GATE_DIM) * 0.1,
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
            net.backward(
                np.ones(4),
                np.ones(ArbitrationNetwork.GATE_DIM),
                0.1,
                lr=0.01,
            )

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

    def test_count_parameters_matches_manual_calculation(self) -> None:
        net = ArbitrationNetwork(
            input_dim=ArbitrationNetwork.INPUT_DIM,
            hidden_dim=8,
            rng=np.random.default_rng(123),
        )
        expected = (
            net.W1.size
            + net.b1.size
            + net.W2_valence.size
            + net.b2_valence.size
            + net.W2_gate.size
            + net.b2_gate.size
            + net.W2_value.size
            + net.b2_value.size
        )
        self.assertEqual(net.count_parameters(), expected)

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
