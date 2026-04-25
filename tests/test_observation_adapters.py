from __future__ import annotations

import unittest

import numpy as np

from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.observation_adapters import (
    adapt_observation_contracts,
    adapter_trace_summary,
    observation_vectors_from_adapters,
)
from spider_cortex_sim.world import SpiderWorld


def get_interface_by_name(name: str):
    interface = next(
        (interface for interface in MODULE_INTERFACES if interface.name == name),
        None,
    )
    if interface is None:
        available = ", ".join(interface.name for interface in MODULE_INTERFACES)
        raise AssertionError(f"Unknown interface {name!r}; available: {available}")
    return interface


class ObservationAdaptersTest(unittest.TestCase):
    def test_adapters_validate_and_preserve_vectors(self) -> None:
        """
        Verify that adapting a SpiderWorld observation produces vectors and trace entries matching MODULE_INTERFACES.
        
        Asserts that for every interface in MODULE_INTERFACES:
        - the interface's observation_key exists in the produced vectors with shape (interface.input_dim,)
        - the adapter trace contains the interface name with "tick" equal to 4 and "signal_names" equal to list(interface.signal_names)
        """
        world = SpiderWorld(seed=3)
        observation = world.reset(seed=3)

        adapters = adapt_observation_contracts(observation, tick=4)
        vectors = observation_vectors_from_adapters(adapters)
        trace = adapter_trace_summary(adapters)

        for interface in MODULE_INTERFACES:
            self.assertIn(interface.observation_key, vectors)
            self.assertEqual(
                vectors[interface.observation_key].shape,
                (interface.input_dim,),
            )
            expected = np.asarray(observation[interface.observation_key], dtype=float)
            np.testing.assert_array_equal(vectors[interface.observation_key], expected)
            self.assertIn(interface.name, trace)
            self.assertEqual(trace[interface.name]["tick"], 4)
            self.assertEqual(
                trace[interface.name]["signal_names"],
                list(interface.signal_names),
            )

    def test_adapter_normalizes_non_finite_values_and_reports_freshness(self) -> None:
        world = SpiderWorld(seed=7)
        observation = dict(world.reset(seed=7))
        visual = get_interface_by_name("visual_cortex")
        vector = np.asarray(observation[visual.observation_key], dtype=float).copy()
        names = list(visual.signal_names)
        vector[names.index("food_certainty")] = np.inf
        vector[names.index("foveal_scan_age")] = 0.25
        vector[names.index("food_trace_strength")] = np.nan
        vector[names.index("predator_trace_heading_dx")] = -np.inf
        observation[visual.observation_key] = vector

        adapters = adapt_observation_contracts(observation, tick=9)
        vectors = observation_vectors_from_adapters(adapters)
        trace = adapter_trace_summary(adapters)
        visual_vector = vectors[visual.observation_key]
        visual_trace = trace[visual.name]

        self.assertTrue(np.all(np.isfinite(visual_vector)))
        self.assertEqual(visual_vector[names.index("food_certainty")], 1.0)
        self.assertEqual(visual_vector[names.index("food_trace_strength")], 0.0)
        self.assertEqual(visual_vector[names.index("predator_trace_heading_dx")], -1.0)
        self.assertEqual(visual_trace["tick"], 9)
        self.assertEqual(visual_trace["freshness_by_signal"]["food_certainty"], 1.0)
        self.assertEqual(visual_trace["freshness_by_signal"]["foveal_scan_age"], 0.75)
        self.assertEqual(
            visual_trace["freshness_by_signal"]["food_trace_strength"],
            0.0,
        )

    def test_adapter_rejects_missing_required_key(self) -> None:
        world = SpiderWorld(seed=8)
        observation = dict(world.reset(seed=8))
        visual = get_interface_by_name("visual_cortex")
        del observation[visual.observation_key]

        with self.assertRaisesRegex(KeyError, "Observation missing key"):
            adapt_observation_contracts(observation, tick=0)

    def test_context_interfaces_are_adapted_and_traced(self) -> None:
        world = SpiderWorld(seed=9)
        observation = world.reset(seed=9)

        adapters = adapt_observation_contracts(observation, tick=11)
        vectors = observation_vectors_from_adapters(adapters)
        trace = adapter_trace_summary(adapters)

        for interface in (ACTION_CONTEXT_INTERFACE, MOTOR_CONTEXT_INTERFACE):
            expected = np.asarray(observation[interface.observation_key], dtype=float)
            self.assertIn(interface.observation_key, vectors)
            np.testing.assert_array_equal(vectors[interface.observation_key], expected)
            self.assertEqual(
                trace[interface.name]["signal_names"],
                list(interface.signal_names),
            )
            self.assertEqual(trace[interface.name]["tick"], 11)
            self.assertEqual(
                set(trace[interface.name]["freshness_by_signal"]),
                set(interface.signal_names),
            )

    def test_adapter_rejects_wrong_dimension_before_training(self) -> None:
        world = SpiderWorld(seed=5)
        observation = dict(world.reset(seed=5))
        visual = get_interface_by_name("visual_cortex")
        observation[visual.observation_key] = np.zeros(visual.input_dim + 1)

        with self.assertRaisesRegex(ValueError, "expected shape"):
            adapt_observation_contracts(observation, tick=0)


if __name__ == "__main__":
    unittest.main()
