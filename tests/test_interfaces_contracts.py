import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import spider_cortex_sim.interface_docs as interface_docs
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interface_docs import escape_markdown_cell, render_interfaces_markdown
from spider_cortex_sim.interfaces import (
    ALL_INTERFACES,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    ACTION_CONTEXT_INTERFACE,
    ALERT_CENTER_V1_INTERFACE,
    ALERT_CENTER_V2_INTERFACE,
    ALERT_CENTER_V3_INTERFACE,
    HUNGER_CENTER_V1_INTERFACE,
    HUNGER_CENTER_V2_INTERFACE,
    HUNGER_CENTER_V3_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    OBSERVATION_VIEW_BY_KEY,
    ORIENT_HEADINGS,
    SENSORY_CORTEX_V1_INTERFACE,
    SENSORY_CORTEX_V2_INTERFACE,
    SENSORY_CORTEX_V3_INTERFACE,
    SLEEP_CENTER_V1_INTERFACE,
    SLEEP_CENTER_V2_INTERFACE,
    SLEEP_CENTER_V3_INTERFACE,
    VISUAL_CORTEX_V1_INTERFACE,
    VISUAL_CORTEX_V2_INTERFACE,
    VISUAL_CORTEX_V3_INTERFACE,
    AlertObservation,
    ActionContextObservation,
    HungerObservation,
    HomeostasisObservation,
    INTERFACE_VARIANTS,
    ModuleInterface,
    MotorContextObservation,
    ObservationView,
    PerceptionObservation,
    SensoryObservation,
    SignalSpec,
    SleepObservation,
    ThreatObservation,
    VARIANT_MODULES,
    VisualObservation,
    _validate_exact_keys,
    architecture_signature,
    get_interface_variant,
    get_variant_levels,
    interface_registry,
    interface_registry_fingerprint,
    is_variant_interface,
    MODULE_INTERFACE_BY_NAME,
    validate_variant_interfaces,
)

class ObservationContractTest(unittest.TestCase):
    def _interfaces(self):
        """
        Return the ordered sequence of observation/module interfaces to include in contract tests.
        
        Returns:
            interfaces (list): A list containing all entries from MODULE_INTERFACES followed by ACTION_CONTEXT_INTERFACE and MOTOR_CONTEXT_INTERFACE.
        """
        return [*MODULE_INTERFACES, ACTION_CONTEXT_INTERFACE, MOTOR_CONTEXT_INTERFACE]

    def _sample_mapping(self, interface):
        """
        Create a deterministic mapping from an interface's signal names to scaled float values.
        
        Parameters:
            interface: Observation interface exposing `signal_names` (ordered iterable of str) and `input_dim` (int).
        
        Returns:
            dict[str, float]: Mapping where each signal name maps to (index + 1) / (input_dim + 1) as a float, with `index` being the signal's zero-based position in `interface.signal_names`.
        """
        denominator = interface.input_dim + 1
        return {
            name: float((idx + 1) / denominator)
            for idx, name in enumerate(interface.signal_names)
        }

    def test_round_trip_mapping_vector_and_typed_view_for_all_interfaces(self) -> None:
        """
        Verify that every observation interface round-trips a sample mapping through vector conversion and produces a matching typed view.
        
        For each interface returned by _interfaces(), constructs a deterministic sample mapping, converts it to a vector with vector_from_mapping, rebinds it with bind_values, and asserts that:
        - the rebound mapping's keys equal interface.signal_names in order,
        - each signal value in the rebound mapping matches the original sample (using approximate equality),
        - a typed observation view built from the rebound mapping yields the same mapping via as_mapping().
        """
        for interface in self._interfaces():
            with self.subTest(interface=interface.name):
                expected_mapping = self._sample_mapping(interface)

                vector = interface.vector_from_mapping(expected_mapping)
                rebound_mapping = interface.bind_values(vector)
                view_type = OBSERVATION_VIEW_BY_KEY[interface.observation_key]
                view = view_type.from_mapping(rebound_mapping)

                self.assertEqual(tuple(rebound_mapping.keys()), interface.signal_names)
                for signal_name in interface.signal_names:
                    self.assertAlmostEqual(rebound_mapping[signal_name], expected_mapping[signal_name])
                self.assertEqual(view.as_mapping(), rebound_mapping)

    def test_vector_from_mapping_rejects_missing_signal(self) -> None:
        """
        Verifies that an interface rejects mappings missing a required signal.
        
        Constructs a sample mapping for the first module interface, removes its first signal,
        and asserts that calling `vector_from_mapping` raises a `ValueError` whose message
        includes the interface's name.
        Raises:
            ValueError: if a required signal is missing (message contains the interface name).
        """
        interface = MODULE_INTERFACES[0]
        missing_signal = interface.signal_names[0]
        mapping = self._sample_mapping(interface)
        mapping.pop(missing_signal)

        with self.assertRaisesRegex(ValueError, interface.name):
            interface.vector_from_mapping(mapping)

    def test_vector_from_mapping_rejects_extra_signal(self) -> None:
        interface = MODULE_INTERFACES[0]
        mapping = self._sample_mapping(interface)
        mapping["unexpected_signal"] = 0.5

        with self.assertRaisesRegex(ValueError, "unexpected_signal"):
            interface.vector_from_mapping(mapping)

class SignalDescriptionRegressionTest(unittest.TestCase):
    """Regression tests for interface signal descriptions translated in this PR."""

    def test_all_signal_descriptions_are_non_empty(self) -> None:
        for interface in ALL_INTERFACES:
            for signal in interface.inputs:
                self.assertTrue(
                    signal.description.strip(),
                    f"Signal {signal.name!r} in {interface.name!r} has empty description",
                )

    def test_all_signal_descriptions_are_strings(self) -> None:
        for interface in ALL_INTERFACES:
            for signal in interface.inputs:
                self.assertIsInstance(
                    signal.description,
                    str,
                    f"Signal {signal.name!r} in {interface.name!r} description is not a string",
                )

    def test_visual_cortex_signal_descriptions_are_in_english(self) -> None:
        # Spot-check a few well-known signals to confirm the Portuguese descriptions
        # were replaced with English ones in this PR.
        visual_iface = next(i for i in ALL_INTERFACES if i.name == "visual_cortex")
        by_name = {s.name: s.description for s in visual_iface.inputs}
        # These descriptions must not contain Portuguese keywords from the old text
        for signal_name, description in by_name.items():
            self.assertNotIn(
                "se há",
                description.lower(),
                f"Signal {signal_name!r} still has Portuguese text: {description!r}",
            )
            self.assertNotIn(
                "durante o",
                description.lower(),
                f"Signal {signal_name!r} still has Portuguese text: {description!r}",
            )

    def test_sensory_cortex_signal_descriptions_are_in_english(self) -> None:
        sensory_iface = next(i for i in ALL_INTERFACES if i.name == "sensory_cortex")
        by_name = {s.name: s.description for s in sensory_iface.inputs}
        for signal_name, description in by_name.items():
            self.assertNotIn(
                "causada",
                description.lower(),
                f"Signal {signal_name!r} still has Portuguese text: {description!r}",
            )

    def test_signal_descriptions_do_not_contain_portuguese_common_words(self) -> None:
        portuguese_markers = [
            "está",
            "aranha",
            "predador",
            "abrigo",
            "comida",
            "direção",
            "distância",
            "visível",
            "intensidade",
            "memória",
        ]
        for interface in ALL_INTERFACES:
            for signal in interface.inputs:
                desc_lower = signal.description.lower()
                for marker in portuguese_markers:
                    self.assertNotIn(
                        marker,
                        desc_lower,
                        f"Signal {signal.name!r} in {interface.name!r} still has Portuguese "
                        f"word {marker!r}: {signal.description!r}",
                    )

    def test_fingerprint_matches_current_registry(self) -> None:
        # Regression: ensure the fingerprint in docs/interfaces.md matches the actual registry.
        # This catches future description changes that update the registry but not the docs.
        docs_path = Path(__file__).parent.parent / "docs" / "interfaces.md"
        if not docs_path.exists():
            self.skipTest("docs/interfaces.md not found")
        content = docs_path.read_text(encoding="utf-8")
        expected_fingerprint = interface_registry_fingerprint()
        self.assertIn(
            expected_fingerprint,
            content,
            "docs/interfaces.md fingerprint does not match the current registry fingerprint",
        )

class LocomotionActionCountTest(unittest.TestCase):
    def test_locomotion_actions_has_nine_entries(self) -> None:
        """LOCOMOTION_ACTIONS must include 5 original + 4 ORIENT actions = 9 total."""
        self.assertEqual(len(LOCOMOTION_ACTIONS), 9)

    def test_orient_actions_are_last_four(self) -> None:
        self.assertEqual(
            tuple(LOCOMOTION_ACTIONS[5:]),
            ("ORIENT_UP", "ORIENT_DOWN", "ORIENT_LEFT", "ORIENT_RIGHT"),
        )

    def test_action_to_index_has_nine_entries(self) -> None:
        self.assertEqual(len(ACTION_TO_INDEX), 9)

    def test_action_deltas_contains_movement_and_stay_only(self) -> None:
        self.assertEqual(
            tuple(ACTION_DELTAS),
            ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STAY"),
        )

    def test_orient_headings_has_four_entries(self) -> None:
        self.assertEqual(len(ORIENT_HEADINGS), 4)

    def test_all_orient_actions_in_locomotion_actions(self) -> None:
        for action in ORIENT_HEADINGS:
            self.assertIn(action, LOCOMOTION_ACTIONS)

    def test_move_actions_have_nonzero_deltas(self) -> None:
        for action_name in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"):
            dx, dy = ACTION_DELTAS[action_name]
            self.assertNotEqual((dx, dy), (0, 0), f"{action_name} should have nonzero delta")

    def test_stay_action_has_zero_delta(self) -> None:
        self.assertEqual(ACTION_DELTAS["STAY"], (0, 0))

    def test_orient_headings_are_cardinal_unit_vectors(self) -> None:
        for action_name, (hdx, hdy) in ORIENT_HEADINGS.items():
            magnitude = abs(hdx) + abs(hdy)
            self.assertEqual(magnitude, 1, f"{action_name} heading {(hdx, hdy)} is not a cardinal unit vector")
