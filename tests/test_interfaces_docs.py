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
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    OBSERVATION_VIEW_BY_KEY,
    ORIENT_HEADINGS,
    AlertObservation,
    ActionContextObservation,
    HungerObservation,
    ModuleInterface,
    MotorContextObservation,
    ObservationView,
    SensoryObservation,
    SignalSpec,
    SleepObservation,
    VisualObservation,
    _validate_exact_keys,
    architecture_signature,
    interface_registry,
    interface_registry_fingerprint,
    MODULE_INTERFACE_BY_NAME,
)

class RenderInterfacesMarkdownTest(unittest.TestCase):
    def test_markdown_contains_perception_and_active_sensing_section(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("## Perception And Active Sensing", md)

    def test_markdown_contains_foveal_and_peripheral_vision_section(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("### Foveal And Peripheral Vision", md)

    def test_markdown_contains_orient_actions_section(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("### ORIENT Actions", md)

    def test_markdown_contains_perceptual_delay_section(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("### Perceptual Delay", md)

    def test_markdown_contains_explicit_memory_signals_section(self) -> None:
        """
        Interface docs name memory signals and point ownership semantics to the audit.
        """
        md = render_interfaces_markdown()
        self.assertIn("### Explicit Memory Signals", md)
        self.assertIn("perception-grounded memory slots", md)
        self.assertIn("MEMORY_LEAKAGE_AUDIT", md)

    def test_markdown_contains_breaking_change_action_space(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("### Breaking Change: Action Space", md)
        self.assertIn("expanded from 5 to 9 actions", md)

    def test_markdown_contains_orient_action_indices(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("`ORIENT_UP`=5", md)
        self.assertIn("`ORIENT_DOWN`=6", md)
        self.assertIn("`ORIENT_LEFT`=7", md)
        self.assertIn("`ORIENT_RIGHT`=8", md)

    def test_markdown_handles_empty_orient_actions(self) -> None:
        with patch.object(interface_docs, "ORIENT_HEADINGS", {}):
            md = interface_docs.render_interfaces_markdown()

        self.assertIn("New active-sensing actions are none.", md)

    def test_markdown_handles_single_orient_action(self) -> None:
        orient_up_index = ACTION_TO_INDEX["ORIENT_UP"]
        with patch.object(interface_docs, "ORIENT_HEADINGS", {"ORIENT_UP": (0, -1)}):
            md = interface_docs.render_interfaces_markdown()

        self.assertIn(f"New active-sensing actions are `ORIENT_UP`={orient_up_index}.", md)

    def test_escape_markdown_cell_escapes_table_delimiters(self) -> None:
        self.assertEqual(
            escape_markdown_cell("alpha|beta\nnext"),
            "alpha&#124;beta<br>next",
        )

    def test_markdown_all_interfaces_output_nine_actions(self) -> None:
        md = render_interfaces_markdown()
        expected_outputs = "MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, STAY, ORIENT_UP, ORIENT_DOWN, ORIENT_LEFT, ORIENT_RIGHT"
        # Check each interface entry references 9 actions
        self.assertIn(expected_outputs, md)

    def test_markdown_contains_fov_half_angle_default(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("`fov_half_angle` defaults to `45.0`", md)

    def test_markdown_contains_peripheral_half_angle_default(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("`peripheral_half_angle` defaults to `70.0`", md)

    def test_markdown_contains_scan_recency_default(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("`max_scan_age` defaults to `10.0`", md)
        self.assertIn("`PerceptTrace.heading_dx` and `PerceptTrace.heading_dy`", md)
        self.assertIn("`food_trace_heading_*`, `shelter_trace_heading_*`, and `predator_trace_heading_*`", md)

    def test_markdown_contains_perception_categories_section(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("### Perception Categories", md)
        self.assertIn("Direct sight", md)
        self.assertIn("Trace means", md)
        self.assertIn("Delayed means", md)
        self.assertIn("Uncertain means", md)
        self.assertIn("not observation signals", md)

    def test_markdown_contains_peripheral_certainty_penalty_default(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("default `peripheral_certainty_penalty` of `0.35`", md)

    def test_markdown_contains_perceptual_delay_ticks_default(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("`perceptual_delay_ticks` defaults to `1.0`", md)

    def test_markdown_contains_perceptual_delay_noise_default(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("`perceptual_delay_noise` defaults to `0.5`", md)

    def test_markdown_orient_actions_do_not_trigger_motor_slip(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("do not trigger motor slip", md)

    def test_markdown_ends_with_single_newline(self) -> None:
        md = render_interfaces_markdown()
        self.assertTrue(md.endswith("\n"))
        self.assertFalse(md.endswith("\n\n"))

    def test_markdown_schema_version_present(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("Schema version:", md)

class AlertObservationNewFieldsTest(unittest.TestCase):
    """Tests for new fields added to AlertObservation in this PR."""

    def _make_alert_obs(self, **overrides) -> AlertObservation:
        defaults = dict(
            predator_visible=0.0,
            predator_certainty=0.0,
            predator_occluded=0.0,
            predator_dx=0.0,
            predator_dy=0.0,
            predator_smell_strength=0.0,
            predator_motion_salience=0.0,
            visual_predator_threat=0.0,
            olfactory_predator_threat=0.0,
            dominant_predator_none=1.0,
            dominant_predator_visual=0.0,
            dominant_predator_olfactory=0.0,
            recent_pain=0.0,
            recent_contact=0.0,
            on_shelter=0.0,
            night=0.0,
            predator_trace_dx=0.0,
            predator_trace_dy=0.0,
            predator_trace_strength=0.0,
            predator_trace_heading_dx=0.0,
            predator_trace_heading_dy=0.0,
            predator_memory_dx=0.0,
            predator_memory_dy=0.0,
            predator_memory_age=1.0,
            escape_memory_dx=0.0,
            escape_memory_dy=0.0,
            escape_memory_age=1.0,
        )
        defaults.update(overrides)
        return AlertObservation(**defaults)

    def test_alert_observation_has_visual_predator_threat_field(self) -> None:
        obs = self._make_alert_obs(visual_predator_threat=0.7)
        self.assertAlmostEqual(obs.visual_predator_threat, 0.7)

    def test_alert_observation_has_olfactory_predator_threat_field(self) -> None:
        obs = self._make_alert_obs(olfactory_predator_threat=0.5)
        self.assertAlmostEqual(obs.olfactory_predator_threat, 0.5)

    def test_alert_observation_has_dominant_predator_flags(self) -> None:
        obs = self._make_alert_obs(
            dominant_predator_none=0.0,
            dominant_predator_visual=1.0,
            dominant_predator_olfactory=0.0,
        )
        self.assertAlmostEqual(obs.dominant_predator_none, 0.0)
        self.assertAlmostEqual(obs.dominant_predator_visual, 1.0)
        self.assertAlmostEqual(obs.dominant_predator_olfactory, 0.0)

    def test_alert_observation_dominant_predator_default_is_none(self) -> None:
        obs = self._make_alert_obs()
        self.assertAlmostEqual(obs.dominant_predator_none, 1.0)
        self.assertAlmostEqual(obs.dominant_predator_visual, 0.0)
        self.assertAlmostEqual(obs.dominant_predator_olfactory, 0.0)

    def test_alert_observation_serializes_to_correct_shape(self) -> None:
        from spider_cortex_sim.interfaces import OBSERVATION_INTERFACE_BY_KEY
        obs = self._make_alert_obs(
            visual_predator_threat=0.3,
            olfactory_predator_threat=0.1,
            dominant_predator_none=0.0,
            dominant_predator_visual=1.0,
        )
        interface = OBSERVATION_INTERFACE_BY_KEY["alert"]
        vec = interface.vector_from_mapping(obs.as_mapping())
        self.assertEqual(vec.shape, (interface.input_dim,))

    def test_alert_observation_round_trips_through_vector(self) -> None:
        from spider_cortex_sim.interfaces import OBSERVATION_INTERFACE_BY_KEY
        obs = self._make_alert_obs(
            visual_predator_threat=0.4,
            olfactory_predator_threat=0.2,
            dominant_predator_none=0.0,
            dominant_predator_olfactory=1.0,
        )
        interface = OBSERVATION_INTERFACE_BY_KEY["alert"]
        vec = interface.vector_from_mapping(obs.as_mapping())
        rebound = interface.bind_values(vec)
        self.assertAlmostEqual(rebound["visual_predator_threat"], 0.4)
        self.assertAlmostEqual(rebound["olfactory_predator_threat"], 0.2)
        self.assertAlmostEqual(rebound["dominant_predator_none"], 0.0)
        self.assertAlmostEqual(rebound["dominant_predator_visual"], 0.0)
        self.assertAlmostEqual(rebound["dominant_predator_olfactory"], 1.0)

    def test_alert_interface_input_names_include_new_fields(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        input_names = [spec.name for spec in interface.inputs]
        self.assertIn("visual_predator_threat", input_names)
        self.assertIn("olfactory_predator_threat", input_names)
        self.assertIn("dominant_predator_none", input_names)
        self.assertIn("dominant_predator_visual", input_names)
        self.assertIn("dominant_predator_olfactory", input_names)

    def test_alert_interface_new_fields_have_correct_bounds(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        bounds = {spec.name: (spec.minimum, spec.maximum) for spec in interface.inputs}
        self.assertEqual(bounds["visual_predator_threat"], (0.0, 1.0))
        self.assertEqual(bounds["olfactory_predator_threat"], (0.0, 1.0))
        self.assertEqual(bounds["dominant_predator_none"], (0.0, 1.0))
        self.assertEqual(bounds["dominant_predator_visual"], (0.0, 1.0))
        self.assertEqual(bounds["dominant_predator_olfactory"], (0.0, 1.0))

class VisualObservationNewFieldsTest(unittest.TestCase):
    """Tests for new fields added to VisualObservation in this PR."""

    def _make_visual_obs(self, **overrides) -> VisualObservation:
        defaults = dict(
            food_visible=0.0,
            food_certainty=0.0,
            food_occluded=0.0,
            food_dx=0.0,
            food_dy=0.0,
            shelter_visible=0.0,
            shelter_certainty=0.0,
            shelter_occluded=0.0,
            shelter_dx=0.0,
            shelter_dy=0.0,
            predator_visible=0.0,
            predator_certainty=0.0,
            predator_occluded=0.0,
            predator_dx=0.0,
            predator_dy=0.0,
            heading_dx=1.0,
            heading_dy=0.0,
            foveal_scan_age=0.0,
            food_trace_strength=0.0,
            food_trace_heading_dx=0.0,
            food_trace_heading_dy=0.0,
            shelter_trace_strength=0.0,
            shelter_trace_heading_dx=0.0,
            shelter_trace_heading_dy=0.0,
            predator_trace_strength=0.0,
            predator_trace_heading_dx=0.0,
            predator_trace_heading_dy=0.0,
            predator_motion_salience=0.0,
            visual_predator_threat=0.0,
            olfactory_predator_threat=0.0,
            day=1.0,
            night=0.0,
        )
        defaults.update(overrides)
        return VisualObservation(**defaults)

    def test_visual_observation_has_visual_predator_threat_field(self) -> None:
        obs = self._make_visual_obs(visual_predator_threat=0.6)
        self.assertAlmostEqual(obs.visual_predator_threat, 0.6)

    def test_visual_observation_has_olfactory_predator_threat_field(self) -> None:
        obs = self._make_visual_obs(olfactory_predator_threat=0.3)
        self.assertAlmostEqual(obs.olfactory_predator_threat, 0.3)

    def test_visual_observation_round_trips_through_vector(self) -> None:
        from spider_cortex_sim.interfaces import OBSERVATION_INTERFACE_BY_KEY
        obs = self._make_visual_obs(
            visual_predator_threat=0.5,
            olfactory_predator_threat=0.1,
        )
        interface = OBSERVATION_INTERFACE_BY_KEY["visual"]
        vec = interface.vector_from_mapping(obs.as_mapping())
        rebound = interface.bind_values(vec)
        self.assertAlmostEqual(rebound["visual_predator_threat"], 0.5)
        self.assertAlmostEqual(rebound["olfactory_predator_threat"], 0.1)

    def test_visual_interface_input_names_include_new_fields(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["visual_cortex"]
        input_names = [spec.name for spec in interface.inputs]
        self.assertIn("foveal_scan_age", input_names)
        self.assertIn("food_trace_heading_dx", input_names)
        self.assertIn("food_trace_heading_dy", input_names)
        self.assertIn("shelter_trace_heading_dx", input_names)
        self.assertIn("shelter_trace_heading_dy", input_names)
        self.assertIn("predator_trace_heading_dx", input_names)
        self.assertIn("predator_trace_heading_dy", input_names)
        self.assertIn("visual_predator_threat", input_names)
        self.assertIn("olfactory_predator_threat", input_names)

    def test_visual_active_sensing_signal_positions_are_stable(self) -> None:
        interface = OBSERVATION_INTERFACE_BY_KEY["visual"]
        expected_positions = {
            "heading_dx": 15,
            "heading_dy": 16,
            "foveal_scan_age": 17,
            "food_trace_strength": 18,
            "food_trace_heading_dx": 19,
            "food_trace_heading_dy": 20,
            "shelter_trace_strength": 21,
            "shelter_trace_heading_dx": 22,
            "shelter_trace_heading_dy": 23,
            "predator_trace_strength": 24,
            "predator_trace_heading_dx": 25,
            "predator_trace_heading_dy": 26,
        }
        for signal_name, index in expected_positions.items():
            with self.subTest(signal=signal_name):
                self.assertEqual(interface.signal_names[index], signal_name)

    def test_trace_heading_consumer_signal_positions_are_stable(self) -> None:
        expected_positions = {
            "hunger_center": {
                "food_trace_heading_dx": 13,
                "food_trace_heading_dy": 14,
            },
            "sleep_center": {
                "shelter_trace_heading_dx": 13,
                "shelter_trace_heading_dy": 14,
            },
            "alert_center": {
                "predator_trace_heading_dx": 19,
                "predator_trace_heading_dy": 20,
            },
        }
        for interface_name, positions in expected_positions.items():
            interface = MODULE_INTERFACE_BY_NAME[interface_name]
            for signal_name, index in positions.items():
                with self.subTest(interface=interface_name, signal=signal_name):
                    self.assertEqual(interface.signal_names[index], signal_name)

    def test_visual_interface_new_fields_have_correct_bounds(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["visual_cortex"]
        bounds = {spec.name: (spec.minimum, spec.maximum) for spec in interface.inputs}
        self.assertEqual(bounds["foveal_scan_age"], (0.0, 1.0))
        self.assertEqual(bounds["visual_predator_threat"], (0.0, 1.0))
        self.assertEqual(bounds["olfactory_predator_threat"], (0.0, 1.0))

    def test_serialized_visual_observation_shape_exceeds_previous_contract(self) -> None:
        interface = OBSERVATION_INTERFACE_BY_KEY["visual"]
        self.assertEqual(interface.input_dim, len(VisualObservation.field_names()))
        self.assertGreaterEqual(interface.input_dim, 26)
        self.assertGreater(interface.input_dim, 25)

class InterfaceVersionBumpTest(unittest.TestCase):
    """Tests verifying the active-sensing interface version bumps."""

    def test_visual_cortex_version_is_7(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["visual_cortex"]
        self.assertEqual(interface.version, 7)

    def test_alert_center_version_is_8(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        self.assertEqual(interface.version, 8)

    def test_trace_heading_consumer_versions_are_bumped(self) -> None:
        expected_versions = {
            "visual_cortex": 7,
            "hunger_center": 4,
            "sleep_center": 5,
            "alert_center": 8,
        }
        for interface_name, expected_version in expected_versions.items():
            with self.subTest(interface=interface_name):
                self.assertEqual(MODULE_INTERFACE_BY_NAME[interface_name].version, expected_version)

    def test_visual_cortex_version_greater_than_three(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["visual_cortex"]
        self.assertGreater(interface.version, 3)

    def test_alert_center_version_greater_than_four(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        self.assertGreater(interface.version, 4)

    def test_markdown_contains_visual_cortex_version_7(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("Version: `7`", md)

    def test_markdown_contains_alert_center_version_8(self) -> None:
        md = render_interfaces_markdown()
        self.assertIn("Version: `8`", md)

    def test_visual_cortex_input_dim_matches_field_count(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["visual_cortex"]
        obs = VisualObservation(
            food_visible=0.0,
            food_certainty=0.0,
            food_occluded=0.0,
            food_dx=0.0,
            food_dy=0.0,
            shelter_visible=0.0,
            shelter_certainty=0.0,
            shelter_occluded=0.0,
            shelter_dx=0.0,
            shelter_dy=0.0,
            predator_visible=0.0,
            predator_certainty=0.0,
            predator_occluded=0.0,
            predator_dx=0.0,
            predator_dy=0.0,
            heading_dx=1.0,
            heading_dy=0.0,
            foveal_scan_age=0.0,
            food_trace_strength=0.0,
            food_trace_heading_dx=0.0,
            food_trace_heading_dy=0.0,
            shelter_trace_strength=0.0,
            shelter_trace_heading_dx=0.0,
            shelter_trace_heading_dy=0.0,
            predator_trace_strength=0.0,
            predator_trace_heading_dx=0.0,
            predator_trace_heading_dy=0.0,
            predator_motion_salience=0.0,
            visual_predator_threat=0.0,
            olfactory_predator_threat=0.0,
            day=1.0,
            night=0.0,
        )
        mapping = obs.as_mapping()
        self.assertEqual(len(mapping), interface.input_dim)

    def test_alert_center_input_dim_includes_three_new_fields(self) -> None:
        # alert_center had 20 fields before threat/context additions, now includes threat, dominant-type,
        # and trace-heading fields.
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        self.assertGreaterEqual(interface.input_dim, 27)

class InterfaceRegistryFingerprintChangedTest(unittest.TestCase):
    """Tests for the updated registry fingerprint in this PR."""

    def test_fingerprint_is_non_empty_string(self) -> None:
        fp = interface_registry_fingerprint()
        self.assertIsInstance(fp, str)
        self.assertTrue(len(fp) > 0)

    def test_fingerprint_matches_documented_value(self) -> None:
        # The new fingerprint from docs/interfaces.md after PR
        expected = "b5bff4986a57404d4c3c2d3c075b3f8e4c283cf5355c36591588197859807ef7"
        fp = interface_registry_fingerprint()
        self.assertEqual(fp, expected)
