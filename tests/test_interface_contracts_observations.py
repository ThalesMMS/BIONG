"""Tests for spider_cortex_sim.interface_contracts.observations module.

Covers SignalSpec, ObservationView, _validate_exact_keys, and all
concrete ObservationView subclasses introduced in the PR refactor.
"""

from __future__ import annotations

import unittest

from spider_cortex_sim.interface_contracts.observations import (
    ActionContextObservation,
    AlertObservation,
    HomeostasisObservation,
    HungerObservation,
    MotorContextObservation,
    ObservationView,
    PerceptionObservation,
    SensoryObservation,
    SignalSpec,
    SleepObservation,
    ThreatObservation,
    VisualObservation,
    _validate_exact_keys,
)


class SignalSpecTest(unittest.TestCase):
    """Tests for SignalSpec dataclass."""

    def test_creation_with_defaults(self) -> None:
        spec = SignalSpec(name="my_signal", description="test")
        self.assertEqual(spec.name, "my_signal")
        self.assertEqual(spec.description, "test")
        self.assertEqual(spec.minimum, -1.0)
        self.assertEqual(spec.maximum, 1.0)

    def test_creation_with_custom_range(self) -> None:
        spec = SignalSpec(name="age", description="age", minimum=0.0, maximum=100.0)
        self.assertEqual(spec.minimum, 0.0)
        self.assertEqual(spec.maximum, 100.0)

    def test_to_summary_returns_dict(self) -> None:
        spec = SignalSpec(name="my_signal", description="a signal", minimum=-1.0, maximum=1.0)
        summary = spec.to_summary()
        self.assertIsInstance(summary, dict)

    def test_to_summary_contains_all_keys(self) -> None:
        spec = SignalSpec(name="hunger", description="hunger level", minimum=0.0, maximum=1.0)
        summary = spec.to_summary()
        self.assertIn("name", summary)
        self.assertIn("description", summary)
        self.assertIn("minimum", summary)
        self.assertIn("maximum", summary)

    def test_to_summary_values_match(self) -> None:
        spec = SignalSpec(name="fatigue", description="fatigue level", minimum=0.0, maximum=1.0)
        summary = spec.to_summary()
        self.assertEqual(summary["name"], "fatigue")
        self.assertEqual(summary["description"], "fatigue level")
        self.assertAlmostEqual(summary["minimum"], 0.0)
        self.assertAlmostEqual(summary["maximum"], 1.0)

    def test_to_summary_minimum_maximum_are_floats(self) -> None:
        spec = SignalSpec(name="x", description="x")
        summary = spec.to_summary()
        self.assertIsInstance(summary["minimum"], float)
        self.assertIsInstance(summary["maximum"], float)

    def test_is_frozen(self) -> None:
        spec = SignalSpec(name="x", description="x")
        with self.assertRaises((AttributeError, TypeError)):
            spec.name = "changed"  # type: ignore[misc]


class ValidateExactKeysTest(unittest.TestCase):
    """Tests for _validate_exact_keys validation helper."""

    def test_exact_match_does_not_raise(self) -> None:
        _validate_exact_keys("TestLabel", ["a", "b"], {"a": 1.0, "b": 2.0})

    def test_missing_key_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("TestLabel", ["a", "b"], {"a": 1.0})
        self.assertIn("TestLabel", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))

    def test_extra_key_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("TestLabel", ["a"], {"a": 1.0, "extra": 2.0})
        self.assertIn("unexpected", str(ctx.exception))

    def test_empty_expected_and_empty_values_does_not_raise(self) -> None:
        _validate_exact_keys("TestLabel", [], {})

    def test_error_message_includes_label(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("MyObs", ["x"], {})
        self.assertIn("MyObs", str(ctx.exception))

    def test_both_missing_and_extra_mentioned_in_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("TestObs", ["a", "b"], {"c": 1.0})
        msg = str(ctx.exception)
        self.assertIn("missing", msg)
        self.assertIn("unexpected", msg)


class ObservationViewTest(unittest.TestCase):
    """Tests for ObservationView base class methods."""

    def test_visual_observation_field_names_non_empty(self) -> None:
        names = VisualObservation.field_names()
        self.assertIsInstance(names, tuple)
        self.assertGreater(len(names), 0)

    def test_sensory_observation_field_names_non_empty(self) -> None:
        names = SensoryObservation.field_names()
        self.assertGreater(len(names), 0)

    def test_hunger_observation_field_names_non_empty(self) -> None:
        names = HungerObservation.field_names()
        self.assertGreater(len(names), 0)

    def test_observation_key_is_set_for_all_views(self) -> None:
        views = [
            VisualObservation, SensoryObservation, HungerObservation,
            SleepObservation, AlertObservation, PerceptionObservation,
            HomeostasisObservation, ThreatObservation,
            ActionContextObservation, MotorContextObservation,
        ]
        for view_cls in views:
            self.assertTrue(
                bool(view_cls.observation_key),
                f"{view_cls.__name__} has empty observation_key",
            )

    def test_from_mapping_constructs_hunger_observation(self) -> None:
        names = HungerObservation.field_names()
        values = {name: 0.5 for name in names}
        obs = HungerObservation.from_mapping(values)
        self.assertIsInstance(obs, HungerObservation)

    def test_from_mapping_raises_for_missing_key(self) -> None:
        names = HungerObservation.field_names()
        values = {name: 0.5 for name in names}
        del values[names[0]]  # remove first key
        with self.assertRaises(ValueError):
            HungerObservation.from_mapping(values)

    def test_from_mapping_raises_for_extra_key(self) -> None:
        names = HungerObservation.field_names()
        values = {name: 0.5 for name in names}
        values["unexpected_extra_key"] = 0.0
        with self.assertRaises(ValueError):
            HungerObservation.from_mapping(values)

    def test_as_mapping_returns_dict_of_floats(self) -> None:
        names = SleepObservation.field_names()
        values = {name: 0.1 for name in names}
        obs = SleepObservation.from_mapping(values)
        mapping = obs.as_mapping()
        self.assertIsInstance(mapping, dict)
        for v in mapping.values():
            self.assertIsInstance(v, float)

    def test_as_mapping_keys_match_field_names(self) -> None:
        names = AlertObservation.field_names()
        values = {name: 0.0 for name in names}
        obs = AlertObservation.from_mapping(values)
        mapping = obs.as_mapping()
        self.assertEqual(set(mapping.keys()), set(names))

    def test_round_trip_from_and_as_mapping(self) -> None:
        names = SensoryObservation.field_names()
        values = {name: float(i) * 0.1 for i, name in enumerate(names)}
        obs = SensoryObservation.from_mapping(values)
        mapping = obs.as_mapping()
        for name, original_value in values.items():
            self.assertAlmostEqual(mapping[name], original_value)

    def test_from_mapping_coerces_values_to_float(self) -> None:
        names = MotorContextObservation.field_names()
        # Pass integers instead of floats
        values = {name: 1 for name in names}
        obs = MotorContextObservation.from_mapping(values)
        mapping = obs.as_mapping()
        for v in mapping.values():
            self.assertIsInstance(v, float)

    def test_visual_observation_key_is_visual(self) -> None:
        self.assertEqual(VisualObservation.observation_key, "visual")

    def test_sensory_observation_key_is_sensory(self) -> None:
        self.assertEqual(SensoryObservation.observation_key, "sensory")

    def test_hunger_observation_key_is_hunger(self) -> None:
        self.assertEqual(HungerObservation.observation_key, "hunger")

    def test_alert_observation_key_is_alert(self) -> None:
        self.assertEqual(AlertObservation.observation_key, "alert")

    def test_sleep_observation_key_is_sleep(self) -> None:
        self.assertEqual(SleepObservation.observation_key, "sleep")

    def test_action_context_observation_key_is_action_context(self) -> None:
        self.assertEqual(ActionContextObservation.observation_key, "action_context")

    def test_motor_context_observation_key_is_motor_context(self) -> None:
        self.assertEqual(MotorContextObservation.observation_key, "motor_context")

    def test_motor_context_has_momentum_default(self) -> None:
        """MotorContextObservation.momentum defaults to 0.0 if not provided."""
        names = MotorContextObservation.field_names()
        # Exclude momentum to test default
        values = {name: 0.1 for name in names if name != "momentum"}
        values["momentum"] = 0.5
        obs = MotorContextObservation.from_mapping(values)
        self.assertAlmostEqual(obs.momentum, 0.5)


if __name__ == "__main__":
    unittest.main()