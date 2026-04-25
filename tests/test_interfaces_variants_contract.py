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

class InterfaceVariantsTest(unittest.TestCase):
    def test_variant_modules_match_expected_targets(self) -> None:
        self.assertEqual(
            VARIANT_MODULES,
            (
                "visual_cortex",
                "sensory_cortex",
                "hunger_center",
                "sleep_center",
                "alert_center",
            ),
        )

    def test_variant_levels_include_canonical_level_four(self) -> None:
        for module_name in VARIANT_MODULES:
            self.assertEqual(get_variant_levels(module_name), (1, 2, 3, 4))

    def test_get_interface_variant_returns_registered_subsets_and_canonical_v4(self) -> None:
        self.assertIs(get_interface_variant("visual_cortex", 1), VISUAL_CORTEX_V1_INTERFACE)
        self.assertIs(get_interface_variant("sensory_cortex", 2), SENSORY_CORTEX_V2_INTERFACE)
        self.assertIs(get_interface_variant("hunger_center", 1), HUNGER_CENTER_V1_INTERFACE)
        self.assertIs(get_interface_variant("sleep_center", 2), SLEEP_CENTER_V2_INTERFACE)
        self.assertIs(get_interface_variant("alert_center", 3), ALERT_CENTER_V3_INTERFACE)
        self.assertIs(get_interface_variant("alert_center", 4), MODULE_INTERFACE_BY_NAME["alert_center"])

    def test_variant_signal_names_match_expected_contracts(self) -> None:
        self.assertEqual(
            VISUAL_CORTEX_V1_INTERFACE.signal_names,
            ("food_visible", "food_dx", "food_dy", "heading_dx", "heading_dy"),
        )
        self.assertEqual(
            VISUAL_CORTEX_V2_INTERFACE.signal_names,
            (
                "food_visible",
                "food_dx",
                "food_dy",
                "shelter_visible",
                "shelter_dx",
                "shelter_dy",
                "predator_visible",
                "predator_dx",
                "predator_dy",
                "heading_dx",
                "heading_dy",
            ),
        )
        self.assertEqual(
            VISUAL_CORTEX_V3_INTERFACE.signal_names,
            (
                "food_visible",
                "food_dx",
                "food_dy",
                "shelter_visible",
                "shelter_dx",
                "shelter_dy",
                "predator_visible",
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
        self.assertEqual(
            SENSORY_CORTEX_V1_INTERFACE.signal_names,
            (
                "recent_pain",
                "recent_contact",
                "predator_smell_strength",
                "predator_smell_dx",
                "predator_smell_dy",
            ),
        )
        self.assertEqual(
            SENSORY_CORTEX_V2_INTERFACE.signal_names,
            (
                "recent_pain",
                "recent_contact",
                "food_smell_strength",
                "food_smell_dx",
                "food_smell_dy",
                "predator_smell_strength",
                "predator_smell_dx",
                "predator_smell_dy",
            ),
        )
        self.assertEqual(
            SENSORY_CORTEX_V3_INTERFACE.signal_names,
            (
                "recent_pain",
                "recent_contact",
                "health",
                "hunger",
                "fatigue",
                "food_smell_strength",
                "food_smell_dx",
                "food_smell_dy",
                "predator_smell_strength",
                "predator_smell_dx",
                "predator_smell_dy",
            ),
        )
        self.assertEqual(
            HUNGER_CENTER_V1_INTERFACE.signal_names,
            ("hunger", "on_food", "food_visible", "food_dx", "food_dy"),
        )
        self.assertEqual(
            HUNGER_CENTER_V2_INTERFACE.signal_names,
            (
                "hunger",
                "on_food",
                "food_visible",
                "food_dx",
                "food_dy",
                "food_smell_strength",
                "food_smell_dx",
                "food_smell_dy",
            ),
        )
        self.assertEqual(
            HUNGER_CENTER_V3_INTERFACE.signal_names,
            (
                "hunger",
                "on_food",
                "food_visible",
                "food_dx",
                "food_dy",
                "food_smell_strength",
                "food_smell_dx",
                "food_smell_dy",
                "food_trace_dx",
                "food_trace_dy",
                "food_trace_strength",
                "food_trace_heading_dx",
                "food_trace_heading_dy",
            ),
        )
        self.assertEqual(
            SLEEP_CENTER_V1_INTERFACE.signal_names,
            ("fatigue", "on_shelter", "night"),
        )
        self.assertEqual(
            SLEEP_CENTER_V2_INTERFACE.signal_names,
            ("fatigue", "on_shelter", "night", "shelter_role_level"),
        )
        self.assertEqual(
            SLEEP_CENTER_V3_INTERFACE.signal_names,
            (
                "fatigue",
                "on_shelter",
                "night",
                "shelter_role_level",
                "shelter_trace_dx",
                "shelter_trace_dy",
                "shelter_trace_strength",
                "shelter_trace_heading_dx",
                "shelter_trace_heading_dy",
            ),
        )
        self.assertEqual(
            ALERT_CENTER_V1_INTERFACE.signal_names,
            ("predator_visible", "predator_dx", "predator_dy"),
        )
        self.assertEqual(
            ALERT_CENTER_V2_INTERFACE.signal_names,
            (
                "predator_visible",
                "predator_certainty",
                "predator_occluded",
                "predator_dx",
                "predator_dy",
            ),
        )
        self.assertEqual(
            ALERT_CENTER_V3_INTERFACE.signal_names,
            (
                "predator_visible",
                "predator_certainty",
                "predator_occluded",
                "predator_dx",
                "predator_dy",
                "predator_smell_strength",
                "recent_pain",
                "recent_contact",
                "on_shelter",
            ),
        )

    def test_variant_interfaces_preserve_parent_observation_keys(self) -> None:
        for (module_name, _level), interface in INTERFACE_VARIANTS.items():
            self.assertEqual(
                interface.observation_key,
                MODULE_INTERFACE_BY_NAME[module_name].observation_key,
            )

    def test_is_variant_interface_distinguishes_variants_from_canonical(self) -> None:
        self.assertTrue(is_variant_interface(VISUAL_CORTEX_V1_INTERFACE))
        self.assertTrue(is_variant_interface(HUNGER_CENTER_V1_INTERFACE))
        self.assertFalse(is_variant_interface(MODULE_INTERFACE_BY_NAME["hunger_center"]))

    def test_validate_variant_interfaces_succeeds_for_current_registry(self) -> None:
        validate_variant_interfaces()

    def test_unknown_variant_module_or_level_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown interface variant module"):
            get_variant_levels("perception_center")
        with self.assertRaisesRegex(ValueError, "level 99"):
            get_interface_variant("hunger_center", 99)
