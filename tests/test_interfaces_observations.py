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

class LocomotionActionsTest(unittest.TestCase):
    EXPECTED_ACTIONS = (
        "MOVE_UP",
        "MOVE_DOWN",
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "STAY",
        "ORIENT_UP",
        "ORIENT_DOWN",
        "ORIENT_LEFT",
        "ORIENT_RIGHT",
    )

    def test_locomotion_actions_match_expected_contract(self) -> None:
        self.assertEqual(tuple(LOCOMOTION_ACTIONS), self.EXPECTED_ACTIONS)
        self.assertEqual(len(LOCOMOTION_ACTIONS), len(self.EXPECTED_ACTIONS))
        self.assertEqual(
            ACTION_TO_INDEX,
            {action_name: index for index, action_name in enumerate(self.EXPECTED_ACTIONS)},
        )

    def test_existing_action_indices_are_preserved(self) -> None:
        self.assertEqual(
            tuple(LOCOMOTION_ACTIONS[:5]),
            ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "STAY"),
        )
        self.assertEqual(ACTION_TO_INDEX["MOVE_UP"], 0)
        self.assertEqual(ACTION_TO_INDEX["MOVE_DOWN"], 1)
        self.assertEqual(ACTION_TO_INDEX["MOVE_LEFT"], 2)
        self.assertEqual(ACTION_TO_INDEX["MOVE_RIGHT"], 3)
        self.assertEqual(ACTION_TO_INDEX["STAY"], 4)

    def test_orient_actions_have_indices_five_through_eight(self) -> None:
        self.assertEqual(ACTION_TO_INDEX["ORIENT_UP"], 5)
        self.assertEqual(ACTION_TO_INDEX["ORIENT_DOWN"], 6)
        self.assertEqual(ACTION_TO_INDEX["ORIENT_LEFT"], 7)
        self.assertEqual(ACTION_TO_INDEX["ORIENT_RIGHT"], 8)

    def test_orient_actions_do_not_displace(self) -> None:
        for action_name in ORIENT_HEADINGS:
            self.assertNotIn(action_name, ACTION_DELTAS)

    def test_orient_headings_match_contract(self) -> None:
        self.assertEqual(
            ORIENT_HEADINGS,
            {
                "ORIENT_UP": (0, -1),
                "ORIENT_DOWN": (0, 1),
                "ORIENT_LEFT": (-1, 0),
                "ORIENT_RIGHT": (1, 0),
            },
        )

class ValidateExactKeysTest(unittest.TestCase):
    def test_valid_exact_match_raises_nothing(self) -> None:
        _validate_exact_keys("test", ["a", "b", "c"], {"a": 1.0, "b": 2.0, "c": 3.0})

    def test_missing_key_raises_value_error_with_label(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("MyLabel", ["x", "y"], {"x": 0.0})
        self.assertIn("MyLabel", str(ctx.exception))
        self.assertIn("y", str(ctx.exception))

    def test_extra_key_raises_value_error_listing_extras(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("SomeInterface", ["x"], {"x": 0.0, "z": 1.0})
        self.assertIn("SomeInterface", str(ctx.exception))
        self.assertIn("z", str(ctx.exception))

    def test_both_missing_and_extra_keys_reported(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _validate_exact_keys("Combo", ["a", "b"], {"a": 1.0, "c": 2.0})
        msg = str(ctx.exception)
        self.assertIn("b", msg)
        self.assertIn("c", msg)

    def test_empty_expected_and_empty_values_is_valid(self) -> None:
        _validate_exact_keys("empty", [], {})

    def test_empty_expected_with_extra_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_exact_keys("empty", [], {"x": 1.0})

class ObservationViewTest(unittest.TestCase):
    def _make_visual(self) -> VisualObservation:
        """
        Create a representative VisualObservation instance populated with deterministic example values for use in tests.
        
        The returned observation models a scenario with visible food (with certainty and a nonzero displacement), no visible shelter or predator, a defined heading vector, nonzero food trace strength, and day/night indicators set to daytime. Other numeric fields are set to zero where no stimulus is present.
        
        Returns:
            VisualObservation: A test-ready VisualObservation populated with sample float values.
        """
        return VisualObservation(
            food_visible=1.0,
            food_certainty=0.8,
            food_occluded=0.0,
            food_dx=0.5,
            food_dy=-0.5,
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
            foveal_scan_age=0.2,
            food_trace_strength=0.3,
            food_trace_heading_dx=1.0,
            food_trace_heading_dy=0.0,
            shelter_trace_strength=0.0,
            shelter_trace_heading_dx=0.0,
            shelter_trace_heading_dy=0.0,
            predator_trace_strength=0.0,
            predator_trace_heading_dx=0.0,
            predator_trace_heading_dy=0.0,
            predator_motion_salience=0.0,
            visual_predator_threat=0.1,
            olfactory_predator_threat=0.2,
            day=1.0,
            night=0.0,
        )

    def test_field_names_returns_ordered_tuple(self) -> None:
        self.assertEqual(
            VisualObservation.field_names(),
            (
                "food_visible",
                "food_certainty",
                "food_occluded",
                "food_dx",
                "food_dy",
                "shelter_visible",
                "shelter_certainty",
                "shelter_occluded",
                "shelter_dx",
                "shelter_dy",
                "predator_visible",
                "predator_certainty",
                "predator_occluded",
                "predator_dx",
                "predator_dy",
                "heading_dx",
                "heading_dy",
                "foveal_scan_age",
                "food_trace_strength",
                "food_trace_heading_dx",
                "food_trace_heading_dy",
                "shelter_trace_strength",
                "shelter_trace_heading_dx",
                "shelter_trace_heading_dy",
                "predator_trace_strength",
                "predator_trace_heading_dx",
                "predator_trace_heading_dy",
                "predator_motion_salience",
                "visual_predator_threat",
                "olfactory_predator_threat",
                "day",
                "night",
            ),
        )

    def test_as_mapping_returns_float_values(self) -> None:
        """
        Verifies VisualObservation.as_mapping() produces a complete mapping of floats and preserves specific sample values.
        
        Asserts the result is a dict whose keys equal VisualObservation.field_names(), every value is a float, and selected fields match the expected float values for the deterministic test fixture (`food_visible == 1.0`, `food_dx == 0.5`, `day == 1.0`).
        """
        view = self._make_visual()
        mapping = view.as_mapping()
        self.assertIsInstance(mapping, dict)
        self.assertEqual(set(mapping.keys()), set(VisualObservation.field_names()))
        for v in mapping.values():
            self.assertIsInstance(v, float)
        self.assertAlmostEqual(mapping["food_visible"], 1.0)
        self.assertAlmostEqual(mapping["food_dx"], 0.5)
        self.assertAlmostEqual(mapping["day"], 1.0)

    def test_from_mapping_round_trips(self) -> None:
        original = self._make_visual()
        mapping = original.as_mapping()
        recovered = VisualObservation.from_mapping(mapping)
        self.assertEqual(original, recovered)

    def test_from_mapping_converts_to_float(self) -> None:
        mapping = {name: i for i, name in enumerate(VisualObservation.field_names())}
        view = VisualObservation.from_mapping(mapping)
        for v in view.as_mapping().values():
            self.assertIsInstance(v, float)

    def test_from_mapping_rejects_missing_field(self) -> None:
        mapping = self._make_visual().as_mapping()
        mapping.pop("food_visible")
        with self.assertRaisesRegex(ValueError, "VisualObservation"):
            VisualObservation.from_mapping(mapping)

    def test_from_mapping_rejects_extra_field(self) -> None:
        mapping = self._make_visual().as_mapping()
        mapping["extra_field"] = 0.0
        with self.assertRaisesRegex(ValueError, "extra_field"):
            VisualObservation.from_mapping(mapping)

    def test_sensory_observation_field_count(self) -> None:
        self.assertEqual(len(SensoryObservation.field_names()), 12)

    def test_hunger_observation_field_count(self) -> None:
        self.assertEqual(len(HungerObservation.field_names()), 18)

    def test_sleep_observation_field_names_match_contract(self) -> None:
        self.assertEqual(
            SleepObservation.field_names(),
            (
                "fatigue",
                "hunger",
                "on_shelter",
                "night",
                "health",
                "recent_pain",
                "sleep_phase_level",
                "rest_streak_norm",
                "sleep_debt",
                "shelter_role_level",
                "shelter_trace_dx",
                "shelter_trace_dy",
                "shelter_trace_strength",
                "shelter_trace_heading_dx",
                "shelter_trace_heading_dy",
                "shelter_memory_dx",
                "shelter_memory_dy",
                "shelter_memory_age",
            ),
        )

    def test_alert_observation_field_names_match_contract(self) -> None:
        self.assertEqual(
            AlertObservation.field_names(),
            (
                "predator_visible",
                "predator_certainty",
                "predator_occluded",
                "predator_dx",
                "predator_dy",
                "predator_smell_strength",
                "predator_motion_salience",
                "visual_predator_threat",
                "olfactory_predator_threat",
                "dominant_predator_none",
                "dominant_predator_visual",
                "dominant_predator_olfactory",
                "recent_pain",
                "recent_contact",
                "on_shelter",
                "night",
                "predator_trace_dx",
                "predator_trace_dy",
                "predator_trace_strength",
                "predator_trace_heading_dx",
                "predator_trace_heading_dy",
                "predator_memory_dx",
                "predator_memory_dy",
                "predator_memory_age",
                "escape_memory_dx",
                "escape_memory_dy",
                "escape_memory_age",
            ),
        )

    def test_action_context_observation_field_names_match_contract(self) -> None:
        self.assertEqual(
            ActionContextObservation.field_names(),
            (
                "hunger",
                "fatigue",
                "health",
                "recent_pain",
                "recent_contact",
                "on_food",
                "on_shelter",
                "predator_visible",
                "predator_certainty",
                "day",
                "night",
                "last_move_dx",
                "last_move_dy",
                "sleep_debt",
                "shelter_role_level",
            ),
        )

    def test_action_context_observation_round_trip(self) -> None:
        """
        Verifies that an ActionContextObservation can be converted to a mapping and reconstructed unchanged.
        
        Constructs an ActionContextObservation with representative values, converts it to a mapping via as_mapping(), recreates it with from_mapping(), and asserts the reconstructed object equals the original.
        """
        view = ActionContextObservation(
            hunger=0.3,
            fatigue=0.5,
            health=0.9,
            recent_pain=0.0,
            recent_contact=0.1,
            on_food=0.0,
            on_shelter=1.0,
            predator_visible=0.0,
            predator_certainty=0.0,
            day=0.0,
            night=1.0,
            last_move_dx=0.0,
            last_move_dy=-1.0,
            sleep_debt=0.4,
            shelter_role_level=0.75,
        )
        recovered = ActionContextObservation.from_mapping(view.as_mapping())
        self.assertEqual(view, recovered)

    def test_motor_context_observation_field_names_match_contract(self) -> None:
        self.assertEqual(
            MotorContextObservation.field_names(),
            (
                "on_food",
                "on_shelter",
                "predator_visible",
                "predator_certainty",
                "day",
                "night",
                "last_move_dx",
                "last_move_dy",
                "shelter_role_level",
                "heading_dx",
                "heading_dy",
                "terrain_difficulty",
                "fatigue",
                "momentum",
            ),
        )

    def test_motor_context_observation_round_trip(self) -> None:
        """
        Verify MotorContextObservation round-trips through mapping conversion.
        
        Constructs a MotorContextObservation, converts it to a mapping, reconstructs a view
        from that mapping, and asserts the reconstructed view equals the original.
        """
        view = MotorContextObservation(
            on_food=0.0,
            on_shelter=1.0,
            predator_visible=0.0,
            predator_certainty=0.0,
            day=0.0,
            night=1.0,
            last_move_dx=0.0,
            last_move_dy=-1.0,
            shelter_role_level=0.75,
            heading_dx=1.0,
            heading_dy=0.0,
            terrain_difficulty=0.4,
            fatigue=0.2,
            momentum=0.0,
        )
        recovered = MotorContextObservation.from_mapping(view.as_mapping())
        self.assertEqual(view, recovered)
