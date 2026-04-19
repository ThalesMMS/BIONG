"""Tests for the learned arbitration network."""

import unittest

import numpy as np

from spider_cortex_sim.nn import ArbitrationNetwork


class ArbitrationNetworkTest(unittest.TestCase):
    def _network(
        self,
        *,
        seed: int = 123,
        hidden_dim: int = 16,
        gate_adjustment_min: float = ArbitrationNetwork.GATE_ADJUSTMENT_MIN,
        gate_adjustment_max: float = ArbitrationNetwork.GATE_ADJUSTMENT_MAX,
    ) -> ArbitrationNetwork:
        """
        Create a configured ArbitrationNetwork test instance.
        
        Parameters:
            seed (int): RNG seed for deterministic parameter initialization.
            hidden_dim (int): Dimension of the network's hidden layer.
            gate_adjustment_min (float): Minimum allowed gate adjustment produced by the network.
            gate_adjustment_max (float): Maximum allowed gate adjustment produced by the network.
        
        Returns:
            ArbitrationNetwork: Instance initialized with ArbitrationNetwork.INPUT_DIM, the given hidden_dim, an RNG from `seed`, and the specified gate adjustment bounds.
        """
        return ArbitrationNetwork(
            input_dim=ArbitrationNetwork.INPUT_DIM,
            hidden_dim=hidden_dim,
            rng=np.random.default_rng(seed),
            name="arbitration_network",
            gate_adjustment_min=gate_adjustment_min,
            gate_adjustment_max=gate_adjustment_max,
        )

    def _evidence_vector(self) -> np.ndarray:
        """
        Constructs a deterministic evidence vector spanning 0.0 to 1.0 across the network input dimension.
        
        Returns:
            np.ndarray: 1D float array of length ArbitrationNetwork.INPUT_DIM with values linearly spaced from 0.0 to 1.0.
        """
        return np.linspace(0.0, 1.0, ArbitrationNetwork.INPUT_DIM, dtype=float)

    def test_forward_output_shapes(self) -> None:
        network = self._network()

        valence_logits, gate_adjustments, value = network.forward(
            self._evidence_vector(),
            store_cache=True,
        )

        self.assertEqual(valence_logits.shape, (ArbitrationNetwork.VALENCE_DIM,))
        self.assertEqual(gate_adjustments.shape, (ArbitrationNetwork.GATE_DIM,))
        self.assertIsInstance(value, float)

    def test_backward_returns_finite_input_gradients(self) -> None:
        """
        Verifies that calling backward produces a finite input gradient and leaves all ndarray parameters finite.
        
        Runs a forward pass with caching, invokes `backward` with sample gradients (valence logits, gate adjustments, and scalar value gradient) and a learning rate, then asserts:
        - the returned input gradient has shape (ArbitrationNetwork.INPUT_DIM,),
        - every entry in the input gradient is finite,
        - every numpy array in `network.state_dict()` contains only finite values.
        """
        network = self._network()
        network.forward(self._evidence_vector(), store_cache=True)

        grad_input = network.backward(
            grad_valence_logits=np.array([0.1, -0.2, 0.05, 0.05], dtype=float),
            grad_gate_adjustments=np.linspace(-0.3, 0.2, network.gate_dim, dtype=float),
            grad_value=0.4,
            lr=0.01,
        )

        self.assertEqual(grad_input.shape, (ArbitrationNetwork.INPUT_DIM,))
        self.assertTrue(np.all(np.isfinite(grad_input)))
        for value in network.state_dict().values():
            if isinstance(value, np.ndarray):
                self.assertTrue(np.all(np.isfinite(value)))

    def test_backward_sanitizes_nonfinite_value_gradient(self) -> None:
        network = self._network()
        network.forward(self._evidence_vector(), store_cache=True)

        grad_input = network.backward(
            grad_valence_logits=np.zeros(4, dtype=float),
            grad_gate_adjustments=np.zeros(network.gate_dim, dtype=float),
            grad_value=np.nan,
            lr=0.01,
        )

        self.assertTrue(np.all(np.isfinite(grad_input)))
        for value in network.state_dict().values():
            if isinstance(value, np.ndarray):
                self.assertTrue(np.all(np.isfinite(value)))

    def test_state_dict_round_trips(self) -> None:
        source = self._network(seed=1)
        target = self._network(seed=2)
        evidence = self._evidence_vector()

        state = source.state_dict()
        target.load_state_dict(state)

        for key, expected in state.items():
            actual = target.state_dict()[key]
            if isinstance(expected, np.ndarray):
                np.testing.assert_allclose(actual, expected)
            else:
                self.assertEqual(actual, expected)
        source_outputs = source.forward(evidence, store_cache=False)
        target_outputs = target.forward(evidence, store_cache=False)
        self.assertEqual(len(target_outputs), len(source_outputs))
        for actual, expected in zip(target_outputs, source_outputs):
            np.testing.assert_allclose(actual, expected)

    def test_state_dict_includes_gate_adjustment_bounds(self) -> None:
        network = self._network(
            gate_adjustment_min=0.1,
            gate_adjustment_max=2.0,
        )

        state = network.state_dict()

        self.assertEqual(state["gate_adjustment_min"], 0.1)
        self.assertEqual(state["gate_adjustment_max"], 2.0)

    def test_load_state_dict_rejects_invalid_gate_adjustment_bounds_metadata(
        self,
    ) -> None:
        def _pop(key: str):
            def mutate(state: dict[str, object]) -> None:
                state.pop(key)

            return mutate

        def _noop(_: dict[str, object]) -> None:
            return None

        cases = (
            {
                "name": "missing_min",
                "source_kwargs": {"seed": 1},
                "target_kwargs": {"seed": 2},
                "mutate": _pop("gate_adjustment_min"),
                "expected": "gate_adjustment_min",
            },
            {
                "name": "mismatched_min",
                "source_kwargs": {
                    "seed": 1,
                    "gate_adjustment_min": 0.1,
                    "gate_adjustment_max": 1.5,
                },
                "target_kwargs": {
                    "seed": 2,
                    "gate_adjustment_min": 0.5,
                    "gate_adjustment_max": 1.5,
                },
                "mutate": _noop,
                "expected": "gate_adjustment_min",
            },
            {
                "name": "missing_max",
                "source_kwargs": {"seed": 1},
                "target_kwargs": {"seed": 2},
                "mutate": _pop("gate_adjustment_max"),
                "expected": "gate_adjustment_max",
            },
            {
                "name": "mismatched_max",
                "source_kwargs": {
                    "seed": 1,
                    "gate_adjustment_min": 0.5,
                    "gate_adjustment_max": 2.0,
                },
                "target_kwargs": {
                    "seed": 2,
                    "gate_adjustment_min": 0.5,
                    "gate_adjustment_max": 1.5,
                },
                "mutate": _noop,
                "expected": "gate_adjustment_max",
            },
        )

        for case in cases:
            with self.subTest(case=case["name"]):
                source = self._network(**case["source_kwargs"])
                target = self._network(**case["target_kwargs"])
                state = source.state_dict()
                case["mutate"](state)

                with self.assertRaisesRegex(ValueError, case["expected"]):
                    target.load_state_dict(state)

    def test_gate_adjustments_are_constrained(self) -> None:
        """
        Verify that gate adjustments remain within ArbitrationNetwork.GATE_ADJUSTMENT_MIN and ArbitrationNetwork.GATE_ADJUSTMENT_MAX when the gate layer parameters are set to extreme positive or negative values.
        """
        network = self._network()
        evidence = self._evidence_vector()
        self.assertAlmostEqual(
            network.gate_adjustment_min,
            ArbitrationNetwork.GATE_ADJUSTMENT_MIN,
        )
        self.assertAlmostEqual(
            network.gate_adjustment_max,
            ArbitrationNetwork.GATE_ADJUSTMENT_MAX,
        )

        network.W2_gate.fill(100.0)
        network.b2_gate.fill(100.0)
        _, high_adjustments, _ = network.forward(evidence, store_cache=False)
        self.assertTrue(np.all(high_adjustments >= ArbitrationNetwork.GATE_ADJUSTMENT_MIN))
        self.assertTrue(np.all(high_adjustments <= ArbitrationNetwork.GATE_ADJUSTMENT_MAX))

        network.W2_gate.fill(-100.0)
        network.b2_gate.fill(-100.0)
        _, low_adjustments, _ = network.forward(evidence, store_cache=False)
        self.assertTrue(np.all(low_adjustments >= ArbitrationNetwork.GATE_ADJUSTMENT_MIN))
        self.assertTrue(np.all(low_adjustments <= ArbitrationNetwork.GATE_ADJUSTMENT_MAX))

    def test_custom_gate_adjustment_bounds_are_used(self) -> None:
        network = self._network(
            gate_adjustment_min=0.1,
            gate_adjustment_max=2.0,
        )
        evidence = self._evidence_vector()

        network.W2_gate.fill(100.0)
        network.b2_gate.fill(100.0)
        _, high_adjustments, _ = network.forward(evidence, store_cache=False)
        self.assertTrue(np.all(high_adjustments >= 0.1))
        self.assertTrue(np.all(high_adjustments <= 2.0))
        self.assertAlmostEqual(float(np.max(high_adjustments)), 2.0, places=5)

        network.W2_gate.fill(-100.0)
        network.b2_gate.fill(-100.0)
        _, low_adjustments, _ = network.forward(evidence, store_cache=False)
        self.assertTrue(np.all(low_adjustments >= 0.1))
        self.assertTrue(np.all(low_adjustments <= 2.0))
        self.assertAlmostEqual(float(np.min(low_adjustments)), 0.1, places=5)

    def test_backward_gate_gradient_uses_non_default_bounds_exactly(self) -> None:
        network = self._network(
            gate_adjustment_min=0.1,
            gate_adjustment_max=2.0,
        )
        evidence = self._evidence_vector()
        network.W2_gate.fill(0.0)
        network.b2_gate.fill(0.0)
        network.forward(evidence, store_cache=True)
        b2_before = network.b2_gate.copy()
        grad_gate_adjustments = np.linspace(
            0.1,
            0.6,
            network.gate_dim,
            dtype=float,
        )
        lr = 0.03

        network.backward(
            grad_valence_logits=np.zeros(network.valence_dim, dtype=float),
            grad_gate_adjustments=grad_gate_adjustments,
            grad_value=0.0,
            lr=lr,
        )

        gate_range = 2.0 - 0.1
        sigmoid_derivative_at_zero = 0.25
        expected_raw_grad = (
            grad_gate_adjustments
            * gate_range
            * sigmoid_derivative_at_zero
        )
        np.testing.assert_allclose(
            network.b2_gate,
            b2_before - lr * expected_raw_grad,
        )

    def test_custom_gate_adjustment_bounds_affect_gate_gradients(self) -> None:
        narrow = self._network(gate_adjustment_min=0.5, gate_adjustment_max=1.5)
        wide = self._network(gate_adjustment_min=0.0, gate_adjustment_max=2.0)
        evidence = self._evidence_vector()
        narrow.forward(evidence, store_cache=True)
        wide.W1 = narrow.W1.copy()
        wide.b1 = narrow.b1.copy()
        wide.W2_valence = narrow.W2_valence.copy()
        wide.b2_valence = narrow.b2_valence.copy()
        wide.W2_gate = narrow.W2_gate.copy()
        wide.b2_gate = narrow.b2_gate.copy()
        wide.W2_value = narrow.W2_value.copy()
        wide.b2_value = narrow.b2_value.copy()
        wide.forward(evidence, store_cache=True)
        narrow_before = narrow.W2_gate.copy()
        wide_before = wide.W2_gate.copy()

        grad = np.ones(narrow.gate_dim, dtype=float)
        narrow.backward(
            grad_valence_logits=np.zeros(narrow.valence_dim, dtype=float),
            grad_gate_adjustments=grad,
            grad_value=0.0,
            lr=0.01,
        )
        wide.backward(
            grad_valence_logits=np.zeros(wide.valence_dim, dtype=float),
            grad_gate_adjustments=grad,
            grad_value=0.0,
            lr=0.01,
        )

        narrow_delta = np.linalg.norm(narrow.W2_gate - narrow_before)
        wide_delta = np.linalg.norm(wide.W2_gate - wide_before)
        self.assertGreater(wide_delta, narrow_delta)

    def test_rejects_invalid_gate_adjustment_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "gate adjustment bounds"):
            self._network(gate_adjustment_min=1.0, gate_adjustment_max=1.0)
        with self.assertRaisesRegex(ValueError, "gate adjustment bounds"):
            self._network(gate_adjustment_min=1.0, gate_adjustment_max=0.0)


if __name__ == "__main__":
    unittest.main()
