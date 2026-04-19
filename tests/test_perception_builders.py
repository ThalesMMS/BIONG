import unittest
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_TO_INDEX,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    AlertObservation,
    ActionContextObservation,
    HungerObservation,
    MotorContextObservation,
    SensoryObservation,
    SleepObservation,
    VisualObservation,
)
from spider_cortex_sim.maps import CLUTTER, NARROW
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.perception import (
    DOMINANT_PREDATOR_TYPE_OLFACTORY,
    DOMINANT_PREDATOR_TYPE_VISUAL,
    OBSERVATION_LEAKAGE_AUDIT,
    TERRAIN_DIFFICULTY,
    PerceivedTarget,
    _apply_visibility_zone_certainty,
    _compute_target_visibility_zone,
    _fov_thresholds,
    _percept_trace_decay,
    _percept_trace_ttl,
    _perception_category,
    advance_percept_trace,
    build_action_context_observation,
    build_alert_observation,
    build_hunger_observation,
    build_motor_context_observation,
    compute_per_type_threats,
    empty_percept_trace,
    observation_leakage_audit,
    build_sensory_observation,
    build_sleep_observation,
    build_visual_observation,
    has_line_of_sight,
    line_cells,
    lizard_detects_spider,
    predator_detects_spider,
    predator_visible_to_spider,
    predators_visible_to_spider,
    serialize_observation_view,
    smell_gradient,
    trace_strength,
    trace_view,
    visible_object,
    visibility_confidence,
    visible_range,
)
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.predator import (
    DEFAULT_LIZARD_PROFILE,
    OLFACTORY_HUNTER_PROFILE,
    VISUAL_HUNTER_PROFILE,
)
from spider_cortex_sim.world import SpiderWorld
from spider_cortex_sim.world_types import PerceptTrace

class PerceptionBuildersTest(unittest.TestCase):
    memory_vector_audit_keys: Tuple[str, ...] = (
        "predator_memory_vector",
        "shelter_memory_vector",
        "escape_memory_vector",
    )

    def setUp(self) -> None:
        """
        Initialize test fixtures for PerceptionBuildersTest.
        
        Creates a SpiderWorld seeded with 7 and resets it, and defines three PerceivedTarget fixtures:
        - `null_percept`: all visibility/certainty/occluded set to 0.0, dx/dy = 0.0, dist = 99.
        - `food_percept`: visible=1.0, certainty=0.9, occluded=0.0, dx=0.5, dy=0.0, dist=3, with a concrete position.
        - `predator_percept`: visible=1.0, certainty=0.8, occluded=0.0, dx=-0.3, dy=0.6, dist=4, with a concrete position.
        """
        self.world = SpiderWorld(seed=7)
        self.world.reset(seed=7)
        self.null_percept = PerceivedTarget(
            visible=0.0,
            certainty=0.0,
            occluded=0.0,
            dx=0.0,
            dy=0.0,
            dist=99,
        )
        self.food_percept = PerceivedTarget(
            visible=1.0,
            certainty=0.9,
            occluded=0.0,
            dx=0.5,
            dy=0.0,
            dist=3,
            position=(8, 5),
        )
        self.predator_percept = PerceivedTarget(
            visible=1.0,
            certainty=0.8,
            occluded=0.0,
            dx=-0.3,
            dy=0.6,
            dist=4,
            position=(3, 7),
        )

    def assert_memory_vector_audit_entries(
        self,
        audit: dict[str, dict[str, object]],
        *,
        expected_classification: str | None = None,
        expected_risk: str | None = None,
        expected_source_fragment: str | None = None,
        expected_modules: dict[str, str] | None = None,
    ) -> None:
        for key in self.memory_vector_audit_keys:
            with self.subTest(memory_vector=key):
                self.assertIn(key, audit)
                metadata = audit[key]
                if expected_classification is not None:
                    self.assertEqual(metadata["classification"], expected_classification)
                if expected_risk is not None:
                    self.assertEqual(metadata["risk"], expected_risk)
                if expected_source_fragment is not None:
                    self.assertIn(expected_source_fragment, metadata["source"])
                if expected_modules is not None:
                    self.assertIn(key, expected_modules)
                    expected_module = expected_modules[key]
                    self.assertIn(expected_module, metadata["modules"])

    def test_serialize_visual_observation_produces_correct_shape(self) -> None:
        """
        Verify that serializing a VisualObservation produces the expected flattened vector length.
        
        Asserts that a VisualObservation containing visual, scan recency, trace, heading, day/night and predator-threat fields serializes to a numpy array of shape (32,).
        """
        view = VisualObservation(
            food_visible=1.0,
            food_certainty=0.9,
            food_occluded=0.0,
            food_dx=0.5,
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
            food_trace_strength=0.1,
            food_trace_heading_dx=1.0,
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
        vector = serialize_observation_view("visual", view)
        self.assertEqual(vector.shape, (32,))

    def test_serialize_observation_view_values_match_as_mapping(self) -> None:
        view = VisualObservation(
            food_visible=0.0,
            food_certainty=0.0,
            food_occluded=0.0,
            food_dx=0.0,
            food_dy=0.0,
            shelter_visible=1.0,
            shelter_certainty=0.7,
            shelter_occluded=0.0,
            shelter_dx=-0.4,
            shelter_dy=0.2,
            predator_visible=0.0,
            predator_certainty=0.0,
            predator_occluded=0.0,
            predator_dx=0.0,
            predator_dy=0.0,
            heading_dx=0.0,
            heading_dy=1.0,
            foveal_scan_age=0.4,
            food_trace_strength=0.0,
            food_trace_heading_dx=0.0,
            food_trace_heading_dy=0.0,
            shelter_trace_strength=0.3,
            shelter_trace_heading_dx=0.0,
            shelter_trace_heading_dy=1.0,
            predator_trace_strength=0.0,
            predator_trace_heading_dx=0.0,
            predator_trace_heading_dy=0.0,
            predator_motion_salience=0.0,
            visual_predator_threat=0.1,
            olfactory_predator_threat=0.2,
            day=0.0,
            night=1.0,
        )
        vector = serialize_observation_view("visual", view)
        rebound = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(vector)
        for name, value in view.as_mapping().items():
            self.assertAlmostEqual(rebound[name], value)

    def test_serialize_action_context_produces_correct_shape(self) -> None:
        obs = self.world.observe()
        self.assertEqual(obs["action_context"].shape, (ACTION_CONTEXT_INTERFACE.input_dim,))

    def test_serialize_motor_context_produces_correct_shape(self) -> None:
        obs = self.world.observe()
        self.assertEqual(obs["motor_context"].shape, (MOTOR_CONTEXT_INTERFACE.input_dim,))

    def test_build_visual_observation_fields_match_percepts(self) -> None:
        visual = build_visual_observation(
            food_view=self.food_percept,
            shelter_view=self.null_percept,
            predator_view=self.predator_percept,
            heading_dx=1.0,
            heading_dy=0.0,
            foveal_scan_age=0.3,
            food_trace_strength=0.25,
            shelter_trace_strength=0.0,
            predator_trace_strength=0.5,
            predator_motion_salience_value=0.1,
            visual_predator_threat=0.3,
            olfactory_predator_threat=0.2,
            day=1.0,
            night=0.0,
        )
        self.assertIsInstance(visual, VisualObservation)
        self.assertAlmostEqual(visual.food_visible, self.food_percept.visible)
        self.assertAlmostEqual(visual.food_certainty, self.food_percept.certainty)
        self.assertAlmostEqual(visual.food_dx, self.food_percept.dx)
        self.assertAlmostEqual(visual.predator_visible, self.predator_percept.visible)
        self.assertAlmostEqual(visual.predator_dx, self.predator_percept.dx)
        self.assertAlmostEqual(visual.heading_dx, 1.0)
        self.assertAlmostEqual(visual.foveal_scan_age, 0.3)
        self.assertAlmostEqual(visual.predator_trace_strength, 0.5)
        self.assertAlmostEqual(visual.visual_predator_threat, 0.3)
        self.assertAlmostEqual(visual.olfactory_predator_threat, 0.2)

    def test_build_visual_observation_null_percept(self) -> None:
        visual = build_visual_observation(
            food_view=self.null_percept,
            shelter_view=self.null_percept,
            predator_view=self.null_percept,
            heading_dx=0.0,
            heading_dy=1.0,
            foveal_scan_age=1.0,
            food_trace_strength=0.0,
            shelter_trace_strength=0.0,
            predator_trace_strength=0.0,
            predator_motion_salience_value=0.0,
            visual_predator_threat=0.0,
            olfactory_predator_threat=0.0,
            day=0.5,
            night=0.5,
        )
        self.assertAlmostEqual(visual.food_visible, 0.0)
        self.assertAlmostEqual(visual.shelter_visible, 0.0)
        self.assertAlmostEqual(visual.predator_visible, 0.0)
        self.assertAlmostEqual(visual.visual_predator_threat, 0.0)
        self.assertAlmostEqual(visual.olfactory_predator_threat, 0.0)

    def test_build_visual_observation_derives_scan_age_from_world(self) -> None:
        self.world._move_spider_action("ORIENT_RIGHT")
        visual = build_visual_observation(
            food_view=self.null_percept,
            shelter_view=self.null_percept,
            predator_view=self.null_percept,
            world=self.world,
            heading_dx=1.0,
            heading_dy=0.0,
            food_trace_strength=0.0,
            shelter_trace_strength=0.0,
            predator_trace_strength=0.0,
            predator_motion_salience_value=0.0,
            visual_predator_threat=0.0,
            olfactory_predator_threat=0.0,
            day=1.0,
            night=0.0,
        )

        self.assertAlmostEqual(visual.foveal_scan_age, 0.0)

    def test_observe_exposes_foveal_scan_age_signal_and_meta(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["perception"]["perceptual_delay_ticks"] = 0.0
        summary["perception"]["max_scan_age"] = 10.0
        profile = OperationalProfile.from_summary(summary)
        world = SpiderWorld(seed=7, operational_profile=profile, lizard_move_interval=999999)
        world.reset(seed=7)
        world._move_spider_action("ORIENT_RIGHT")

        obs = world.observe()
        visual_mapping = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(obs["visual"])

        self.assertAlmostEqual(visual_mapping["foveal_scan_age"], 0.0)
        self.assertAlmostEqual(obs["meta"]["active_sensing"]["foveal_scan_age"], 0.0)
        self.assertEqual(obs["meta"]["active_sensing"]["raw_foveal_scan_age"], 0)

    def test_foveal_scan_age_increments_without_rescanning_heading(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["perception"]["perceptual_delay_ticks"] = 0.0
        summary["perception"]["max_scan_age"] = 10.0
        profile = OperationalProfile.from_summary(summary)
        world = SpiderWorld(seed=7, operational_profile=profile, lizard_move_interval=999999)
        world.reset(seed=7)
        world._move_spider_action("ORIENT_RIGHT")
        fresh_obs = world.observe()
        fresh_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(fresh_obs["visual"])

        one_tick_obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])
        two_tick_obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])
        one_tick_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(one_tick_obs["visual"])
        two_tick_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(two_tick_obs["visual"])

        self.assertAlmostEqual(fresh_visual["foveal_scan_age"], 0.0)
        self.assertAlmostEqual(one_tick_visual["foveal_scan_age"], 0.1)
        self.assertAlmostEqual(two_tick_visual["foveal_scan_age"], 0.2)

    def test_build_sensory_observation_maps_state_fields(self) -> None:
        self.world.state.recent_pain = 0.3
        self.world.state.recent_contact = 0.1
        self.world.state.health = 0.85
        self.world.state.hunger = 0.6
        self.world.state.fatigue = 0.4
        sensory = build_sensory_observation(
            self.world,
            food_smell_strength=0.7,
            food_smell_dx=0.5,
            food_smell_dy=-0.2,
            predator_smell_strength=0.1,
            predator_smell_dx=0.0,
            predator_smell_dy=0.0,
            light=0.9,
        )
        self.assertIsInstance(sensory, SensoryObservation)
        self.assertAlmostEqual(sensory.recent_pain, 0.3)
        self.assertAlmostEqual(sensory.health, 0.85)
        self.assertAlmostEqual(sensory.hunger, 0.6)
        self.assertAlmostEqual(sensory.food_smell_strength, 0.7)

    def test_build_sensory_observation_serializes_correctly(self) -> None:
        sensory = build_sensory_observation(
            self.world,
            food_smell_strength=0.0,
            food_smell_dx=0.0,
            food_smell_dy=0.0,
            predator_smell_strength=0.0,
            predator_smell_dx=0.0,
            predator_smell_dy=0.0,
            light=0.5,
        )
        self.assertEqual(serialize_observation_view("sensory", sensory).shape, (12,))

    def test_build_hunger_observation_populates_all_fields(self) -> None:
        self.world.state.hunger = 0.75
        hunger = build_hunger_observation(
            self.world,
            on_food=1.0,
            food_view=self.food_percept,
            food_smell_strength=0.5,
            food_smell_dx=0.3,
            food_smell_dy=-0.1,
            food_trace=(0.4, -0.2, 0.6, 1.0, 0.0),
            food_memory=(0.6, -0.2, 0.4),
        )
        self.assertIsInstance(hunger, HungerObservation)
        self.assertAlmostEqual(hunger.hunger, 0.75)
        self.assertAlmostEqual(hunger.food_trace_dx, 0.4)
        self.assertAlmostEqual(hunger.food_trace_heading_dx, 1.0)
        self.assertAlmostEqual(hunger.food_trace_heading_dy, 0.0)
        self.assertAlmostEqual(hunger.food_memory_dx, 0.6)
        self.assertAlmostEqual(hunger.food_memory_age, 0.4)

    def test_build_hunger_observation_zero_memory(self) -> None:
        hunger = build_hunger_observation(
            self.world,
            on_food=0.0,
            food_view=self.null_percept,
            food_smell_strength=0.0,
            food_smell_dx=0.0,
            food_smell_dy=0.0,
            food_trace=(0.0, 0.0, 0.0, 0.0, 0.0),
            food_memory=(0.0, 0.0, 1.0),
        )
        self.assertAlmostEqual(hunger.food_memory_age, 1.0)

    def test_observation_leakage_audit_marks_predator_dist_as_resolved(self) -> None:
        audit = observation_leakage_audit()
        self.assertEqual(audit["predator_dist"]["risk"], "resolved")
        self.assertEqual(audit["predator_dist"]["status"], "removed_from_observations")
        self.assertIn("diagnostic_predator_dist", audit["predator_dist"]["evidence"])
        self.assertEqual(set(audit.keys()), set(OBSERVATION_LEAKAGE_AUDIT.keys()))

    def test_observation_leakage_audit_returns_deep_copy_not_reference(self) -> None:
        audit1 = observation_leakage_audit()
        audit2 = observation_leakage_audit()
        audit1["predator_dist"]["risk"] = "mutated"
        self.assertEqual(audit2["predator_dist"]["risk"], "resolved")

    def test_observation_leakage_audit_home_vector_is_resolved_world_derived(self) -> None:
        audit = observation_leakage_audit()
        self.assertEqual(audit["home_vector"]["risk"], "resolved")
        self.assertEqual(audit["home_vector"]["status"], "removed_from_observations")
        self.assertEqual(audit["home_vector"]["classification"], "world_derived_navigation_hint")
        self.assertIn("diagnostic_home", audit["home_vector"]["evidence"])

    def test_observation_leakage_audit_low_risk_entries_are_agent_grounded(self) -> None:
        audit = observation_leakage_audit()
        low_risk = [name for name, data in audit.items() if data["risk"] == "low"]
        for name in low_risk:
            self.assertIn(
                audit[name]["classification"],
                {"direct_perception", "plausible_memory", "self_knowledge"},
                f"Low-risk entry {name!r} should be direct perception, plausible memory, or self-knowledge",
            )

    def test_observation_leakage_audit_all_entries_have_required_fields(self) -> None:
        audit = observation_leakage_audit()
        required_fields = {"classification", "risk", "modules", "source", "notes"}
        for key, metadata in audit.items():
            for field in required_fields:
                self.assertIn(
                    field,
                    metadata,
                    f"Observation leakage entry {key!r} missing field {field!r}",
                )

    def test_observation_leakage_audit_modules_are_lists(self) -> None:
        audit = observation_leakage_audit()
        for key, metadata in audit.items():
            self.assertIsInstance(
                metadata["modules"],
                list,
                f"Observation leakage entry {key!r} 'modules' should be a list",
            )
            self.assertGreater(
                len(metadata["modules"]),
                0,
                f"Observation leakage entry {key!r} should have at least one module",
            )

    def test_observation_leakage_audit_risk_values_are_valid(self) -> None:
        audit = observation_leakage_audit()
        valid_risk_levels = {"low", "medium", "high", "resolved"}
        for key, metadata in audit.items():
            self.assertIn(
                metadata["risk"],
                valid_risk_levels,
                f"Entry {key!r} has unexpected risk level {metadata['risk']!r}",
            )

    def test_observation_leakage_audit_scan_age_is_self_knowledge(self) -> None:
        audit = observation_leakage_audit()
        self.assertEqual(audit["foveal_scan_age"]["classification"], "self_knowledge")
        self.assertEqual(audit["foveal_scan_age"]["risk"], "low")
        self.assertIn("visual", audit["foveal_scan_age"]["modules"])

    def test_observation_leakage_audit_memory_vectors_are_plausible_low_risk(self) -> None:
        audit = observation_leakage_audit()
        self.assert_memory_vector_audit_entries(
            audit,
            expected_classification="plausible_memory",
            expected_risk="low",
        )

    def test_observation_leakage_audit_no_world_owned_memory_classification(self) -> None:
        """Reclassified memory-vector entries should not be classified as world_owned_memory."""
        audit = observation_leakage_audit()
        for key in self.memory_vector_audit_keys:
            with self.subTest(memory_vector=key):
                self.assertIn(key, audit)
                metadata = audit[key]
                self.assertNotEqual(
                    metadata["classification"],
                    "world_owned_memory",
                    f"Entry {key!r} still classified as 'world_owned_memory'",
                )

    def test_observation_leakage_audit_no_medium_or_high_risk(self) -> None:
        """Reclassified memory-vector entries must not be medium or high risk."""
        audit = observation_leakage_audit()
        for key in self.memory_vector_audit_keys:
            with self.subTest(memory_vector=key):
                self.assertIn(key, audit)
                metadata = audit[key]
                if metadata["risk"] == "resolved":
                    continue
                self.assertNotIn(
                    metadata["risk"],
                    {"medium", "high"},
                    f"Entry {key!r} has unexpected risk level {metadata['risk']!r}",
                )

    def test_observation_leakage_audit_shelter_memory_vector_notes_mention_perceived(self) -> None:
        """shelter_memory_vector notes must reference perceived shelter rather than world-selected target."""
        audit = observation_leakage_audit()
        notes = audit["shelter_memory_vector"]["notes"].casefold()
        self.assertIn("perceived", notes)

    def test_observation_leakage_audit_predator_memory_vector_notes_mention_visual_or_contact(self) -> None:
        """predator_memory_vector notes must mention visual perception and contact events."""
        audit = observation_leakage_audit()
        notes = audit["predator_memory_vector"]["notes"].casefold()
        self.assertIn("visual", notes)
        self.assertIn("contact", notes)

    def test_observation_leakage_audit_escape_memory_vector_notes_mention_movement(self) -> None:
        """escape_memory_vector notes must reference movement history, not walkability."""
        audit = observation_leakage_audit()
        notes = audit["escape_memory_vector"]["notes"].casefold()
        self.assertIn("movement", notes)

    def test_observation_leakage_audit_memory_vector_sources_reference_refresh_memory(self) -> None:
        """All three memory-vector entries should cite refresh_memory as part of their source."""
        audit = observation_leakage_audit()
        self.assert_memory_vector_audit_entries(audit, expected_source_fragment="refresh_memory")

    def test_observation_leakage_audit_memory_vectors_have_expected_modules(self) -> None:
        """Memory-vector entries should be associated with their consumer modules."""
        audit = observation_leakage_audit()
        self.assert_memory_vector_audit_entries(
            audit,
            expected_modules={
                "predator_memory_vector": "alert",
                "shelter_memory_vector": "sleep",
                "escape_memory_vector": "alert",
            },
        )

    def test_observation_leakage_audit_all_three_memory_vectors_present(self) -> None:
        """predator_memory_vector, shelter_memory_vector, escape_memory_vector are all in the audit."""
        audit = observation_leakage_audit()
        self.assert_memory_vector_audit_entries(audit)

    def test_build_sleep_observation_maps_state_and_args(self) -> None:
        self.world.state.fatigue = 0.65
        self.world.state.hunger = 0.2
        self.world.state.health = 0.9
        self.world.state.recent_pain = 0.0
        self.world.state.sleep_debt = 0.5
        sleep = build_sleep_observation(
            self.world,
            on_shelter=1.0,
            night=1.0,
            sleep_phase_level=0.7,
            rest_streak_norm=0.5,
            shelter_role_level=0.8,
            shelter_trace=(0.1, -0.2, 0.55, 0.0, -1.0),
            shelter_memory=(0.2, -0.1, 0.3),
        )
        self.assertIsInstance(sleep, SleepObservation)
        self.assertAlmostEqual(sleep.sleep_debt, 0.5)
        self.assertAlmostEqual(sleep.shelter_trace_strength, 0.55)
        self.assertAlmostEqual(sleep.shelter_trace_heading_dx, 0.0)
        self.assertAlmostEqual(sleep.shelter_trace_heading_dy, -1.0)
        self.assertAlmostEqual(sleep.shelter_memory_dx, 0.2)
        self.assertFalse(hasattr(sleep, "home_dx"))
        self.assertFalse(hasattr(sleep, "home_dy"))
        self.assertFalse(hasattr(sleep, "home_dist"))

    def test_build_sleep_observation_serializes_to_correct_shape(self) -> None:
        sleep = build_sleep_observation(
            self.world,
            on_shelter=0.0,
            night=0.0,
            sleep_phase_level=0.0,
            rest_streak_norm=0.0,
            shelter_role_level=0.0,
            shelter_trace=(0.0, 0.0, 0.0, 0.0, 0.0),
            shelter_memory=(0.0, 0.0, 1.0),
        )
        self.assertEqual(serialize_observation_view("sleep", sleep).shape, (18,))

    def test_sleep_observation_excludes_removed_home_fields(self) -> None:
        sleep = build_sleep_observation(
            self.world,
            on_shelter=0.0,
            night=0.0,
            sleep_phase_level=0.0,
            rest_streak_norm=0.0,
            shelter_role_level=0.0,
            shelter_trace=(0.0, 0.0, 0.0, 0.0, 0.0),
            shelter_memory=(0.0, 0.0, 1.0),
        )
        self.assertNotIn("home_dx", sleep.as_mapping())
        self.assertNotIn("home_dy", sleep.as_mapping())
        self.assertNotIn("home_dist", sleep.as_mapping())

    def test_build_alert_observation_maps_percept_and_state(self) -> None:
        self.world.state.recent_pain = 0.2
        self.world.state.recent_contact = 0.0
        alert = build_alert_observation(
            self.world,
            predator_view=self.predator_percept,
            predator_smell_strength=0.1,
            predator_motion_salience_value=0.1,
            visual_predator_threat=0.4,
            olfactory_predator_threat=0.2,
            dominant_predator_type=DOMINANT_PREDATOR_TYPE_VISUAL,
            on_shelter=0.0,
            night=0.0,
            predator_trace=(0.2, -0.4, 0.7, -1.0, 0.0),
            predator_memory=(0.7, -0.2, 0.5),
            escape_memory=(0.0, 0.0, 1.0),
        )
        self.assertIsInstance(alert, AlertObservation)
        self.assertAlmostEqual(alert.predator_trace_strength, 0.7)
        self.assertAlmostEqual(alert.predator_trace_heading_dx, -1.0)
        self.assertAlmostEqual(alert.predator_trace_heading_dy, 0.0)
        self.assertAlmostEqual(alert.predator_memory_dx, 0.7)
        self.assertAlmostEqual(alert.dominant_predator_none, 0.0)
        self.assertAlmostEqual(alert.dominant_predator_visual, 1.0)
        self.assertAlmostEqual(alert.dominant_predator_olfactory, 0.0)
        self.assertFalse(hasattr(alert, "predator_dist"))
        self.assertFalse(hasattr(alert, "home_dx"))
        self.assertFalse(hasattr(alert, "home_dy"))

    def test_alert_observation_excludes_removed_privileged_fields(self) -> None:
        alert = build_alert_observation(
            self.world,
            predator_view=self.null_percept,
            predator_smell_strength=0.0,
            predator_motion_salience_value=0.0,
            visual_predator_threat=0.0,
            olfactory_predator_threat=0.0,
            dominant_predator_type=0.0,
            on_shelter=0.0,
            night=0.0,
            predator_trace=(0.0, 0.0, 0.0, 0.0, 0.0),
            predator_memory=(0.0, 0.0, 1.0),
            escape_memory=(0.0, 0.0, 1.0),
        )
        self.assertNotIn("predator_dist", alert.as_mapping())
        self.assertNotIn("home_dx", alert.as_mapping())
        self.assertNotIn("home_dy", alert.as_mapping())
        self.assertNotIn("dominant_predator_type", alert.as_mapping())

    def test_build_alert_observation_serializes_to_correct_shape(self) -> None:
        alert = build_alert_observation(
            self.world,
            predator_view=self.null_percept,
            predator_smell_strength=0.0,
            predator_motion_salience_value=0.0,
            visual_predator_threat=0.0,
            olfactory_predator_threat=0.0,
            dominant_predator_type=0.0,
            on_shelter=0.0,
            night=0.0,
            predator_trace=(0.0, 0.0, 0.0, 0.0, 0.0),
            predator_memory=(0.0, 0.0, 1.0),
            escape_memory=(0.0, 0.0, 1.0),
        )
        self.assertEqual(serialize_observation_view("alert", alert).shape, (27,))

    def test_build_action_context_observation_maps_state_fields(self) -> None:
        self.world.state.hunger = 0.4
        self.world.state.fatigue = 0.3
        self.world.state.health = 0.95
        self.world.state.sleep_debt = 0.25
        self.world.state.last_move_dx = 1
        ac = build_action_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=1.0,
            predator_view=self.null_percept,
            day=1.0,
            night=0.0,
            shelter_role_level=0.6,
        )
        self.assertIsInstance(ac, ActionContextObservation)
        self.assertAlmostEqual(ac.last_move_dx, 1.0)
        self.assertAlmostEqual(ac.sleep_debt, 0.25)
        self.assertFalse(hasattr(ac, "predator_dist"))

    def test_context_observations_exclude_predator_dist(self) -> None:
        ac = build_action_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )
        mc = build_motor_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )
        action_mapping = ac.as_mapping()
        motor_mapping = mc.as_mapping()
        self.assertNotIn("predator_dist", action_mapping)
        self.assertNotIn("home_dx", action_mapping)
        self.assertNotIn("home_dy", action_mapping)
        self.assertNotIn("predator_dist", motor_mapping)
        self.assertNotIn("home_dx", motor_mapping)
        self.assertNotIn("home_dy", motor_mapping)

    def test_build_action_context_observation_serializes_to_correct_shape(self) -> None:
        ac = build_action_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )
        self.assertEqual(serialize_observation_view("action_context", ac).shape, (ACTION_CONTEXT_INTERFACE.input_dim,))

    def test_build_motor_context_observation_serializes_to_correct_shape(self) -> None:
        mc = build_motor_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )
        self.assertEqual(serialize_observation_view("motor_context", mc).shape, (MOTOR_CONTEXT_INTERFACE.input_dim,))

    def test_build_motor_context_observation_adds_embodiment_signals(self) -> None:
        self.world.state.heading_dx = 0
        self.world.state.heading_dy = -1
        self.world.state.fatigue = 0.37
        self.world.state.momentum = 0.58
        self.world.map_template.terrain[self.world.spider_pos()] = NARROW

        mc = build_motor_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )

        self.assertAlmostEqual(mc.heading_dx, 0.0)
        self.assertAlmostEqual(mc.heading_dy, -1.0)
        self.assertAlmostEqual(mc.terrain_difficulty, TERRAIN_DIFFICULTY[NARROW])
        self.assertAlmostEqual(mc.fatigue, 0.37)
        self.assertAlmostEqual(mc.momentum, 0.58)

    def test_build_motor_context_observation_clamps_momentum(self) -> None:
        self.world.state.momentum = 1.5
        high = build_motor_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )
        self.world.state.momentum = -0.5
        low = build_motor_context_observation(
            self.world,
            on_food=0.0,
            on_shelter=0.0,
            predator_view=self.null_percept,
            day=1.0,
            night=0.0,
            shelter_role_level=0.0,
        )

        self.assertAlmostEqual(high.momentum, 1.0)
        self.assertAlmostEqual(low.momentum, 0.0)

    def test_observe_all_modalities_match_interface_dims(self) -> None:
        obs = self.world.observe()
        for key in ("visual", "sensory", "hunger", "sleep", "alert", "action_context", "motor_context"):
            iface = OBSERVATION_INTERFACE_BY_KEY[key]
            self.assertEqual(obs[key].shape, (iface.input_dim,))

    def test_observe_world_uses_reduced_observation_dims(self) -> None:
        obs = self.world.observe()
        self.assertEqual(obs["visual"].shape, (32,))
        self.assertEqual(obs["sleep"].shape, (18,))
        self.assertEqual(obs["alert"].shape, (27,))
        self.assertEqual(obs["action_context"].shape, (15,))
        self.assertEqual(obs["motor_context"].shape, (14,))

    def test_observe_motor_extra_equals_motor_context(self) -> None:
        obs = self.world.observe()
        np.testing.assert_array_equal(obs["motor_extra"], obs["motor_context"])

    def test_observe_builder_values_consistent_with_state(self) -> None:
        """
        Verifies that the `action_context` observation reflects the world's `hunger` and `fatigue` state fields.
        
        Reconstructs an ActionContextObservation from the serialized `action_context` vector and asserts `hunger` and `fatigue` match the world state within 1e-5.
        """
        self.world.state.hunger = 0.77
        self.world.state.fatigue = 0.33
        obs = self.world.observe()
        ac = ActionContextObservation.from_mapping(
            ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        )
        self.assertAlmostEqual(ac.hunger, 0.77, places=5)
        self.assertAlmostEqual(ac.fatigue, 0.33, places=5)

    def test_observe_visual_day_night_consistent_with_meta(self) -> None:
        obs = self.world.observe()
        visual_mapping = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(obs["visual"])
        self.assertEqual(obs["meta"]["day"], bool(visual_mapping["day"]))
        self.assertEqual(obs["meta"]["night"], bool(visual_mapping["night"]))

    def test_observe_sleep_phase_level_in_sleep_observation(self) -> None:
        obs = self.world.observe()
        sleep_mapping = OBSERVATION_INTERFACE_BY_KEY["sleep"].bind_values(obs["sleep"])
        self.assertAlmostEqual(sleep_mapping["sleep_phase_level"], obs["meta"]["sleep_phase_level"], places=5)

    def test_observe_returns_numpy_arrays_for_all_vector_keys(self) -> None:
        obs = self.world.observe()
        for key in ("visual", "sensory", "hunger", "sleep", "alert", "action_context", "motor_context", "motor_extra"):
            self.assertIsInstance(obs[key], np.ndarray)

    def test_build_visual_observation_round_trips_through_serialization(self) -> None:
        visual = build_visual_observation(
            food_view=self.food_percept,
            shelter_view=self.null_percept,
            predator_view=self.null_percept,
            heading_dx=1.0,
            heading_dy=0.0,
            foveal_scan_age=0.0,
            food_trace_strength=0.3,
            shelter_trace_strength=0.0,
            predator_trace_strength=0.0,
            predator_motion_salience_value=0.0,
            visual_predator_threat=0.2,
            olfactory_predator_threat=0.1,
            day=1.0,
            night=0.0,
        )
        vector = serialize_observation_view("visual", visual)
        recovered_mapping = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(vector)
        recovered_view = VisualObservation.from_mapping(recovered_mapping)
        self.assertEqual(visual, recovered_view)

    def test_observe_world_meta_exposes_heading_and_percept_traces(self) -> None:
        obs = self.world.observe()
        self.assertIn("heading", obs["meta"])
        self.assertIn("dx", obs["meta"]["heading"])
        self.assertIn("dy", obs["meta"]["heading"])
        self.assertIn("percept_traces", obs["meta"])
        self.assertEqual(
            set(obs["meta"]["percept_traces"].keys()),
            {"food", "shelter", "predator"},
        )
        for key in ("food", "shelter", "predator"):
            trace = obs["meta"]["percept_traces"][key]
            self.assertEqual(
                set(trace.keys()),
                {
                    "target",
                    "age",
                    "certainty",
                    "strength",
                    "dx",
                    "dy",
                    "heading_dx",
                    "heading_dy",
                    "ttl",
                    "decay",
                },
            )
        self.assertIn("predator_motion_salience", obs["meta"])

    def test_observe_world_meta_contains_diagnostic_home_vector(self) -> None:
        """Privileged home-vector diagnostics live under meta['diagnostic'] only."""
        obs = self.world.observe()
        diagnostic = obs["meta"]["diagnostic"]
        self.assertIn("diagnostic_home_dx", diagnostic)
        self.assertIn("diagnostic_home_dy", diagnostic)
        self.assertIn("diagnostic_home_dist", diagnostic)
        self.assertNotIn("home_dx", obs["meta"])
        self.assertNotIn("home_dy", obs["meta"])
        self.assertNotIn("home_dist", obs["meta"])

    def test_observe_world_meta_home_dx_is_float(self) -> None:
        obs = self.world.observe()
        self.assertIsInstance(obs["meta"]["diagnostic"]["diagnostic_home_dx"], float)

    def test_observe_world_meta_home_dy_is_float(self) -> None:
        obs = self.world.observe()
        self.assertIsInstance(obs["meta"]["diagnostic"]["diagnostic_home_dy"], float)

    def test_observe_world_meta_home_dist_is_float(self) -> None:
        obs = self.world.observe()
        self.assertIsInstance(obs["meta"]["diagnostic"]["diagnostic_home_dist"], float)

    def test_observe_world_meta_home_dist_is_non_negative(self) -> None:
        obs = self.world.observe()
        self.assertGreaterEqual(obs["meta"]["diagnostic"]["diagnostic_home_dist"], 0.0)

    def test_observe_world_meta_home_dx_dy_are_unit_range(self) -> None:
        """home_dx and home_dy from _relative() should be in [-1, 1]."""
        obs = self.world.observe()
        diagnostic = obs["meta"]["diagnostic"]
        self.assertGreaterEqual(diagnostic["diagnostic_home_dx"], -1.0)
        self.assertLessEqual(diagnostic["diagnostic_home_dx"], 1.0)
        self.assertGreaterEqual(diagnostic["diagnostic_home_dy"], -1.0)
        self.assertLessEqual(diagnostic["diagnostic_home_dy"], 1.0)

    def test_observe_world_meta_home_vector_not_in_sleep_observation(self) -> None:
        """The diagnostic home vector must not appear in the sleep observation vector."""
        obs = self.world.observe()
        sleep_mapping = OBSERVATION_INTERFACE_BY_KEY["sleep"].bind_values(obs["sleep"])
        self.assertNotIn("home_dx", sleep_mapping)
        self.assertNotIn("home_dy", sleep_mapping)
        self.assertNotIn("home_dist", sleep_mapping)

    def test_observe_world_meta_home_vector_not_in_alert_observation(self) -> None:
        """The diagnostic home vector must not appear in the alert observation vector."""
        obs = self.world.observe()
        alert_mapping = OBSERVATION_INTERFACE_BY_KEY["alert"].bind_values(obs["alert"])
        self.assertNotIn("home_dx", alert_mapping)
        self.assertNotIn("home_dy", alert_mapping)

    def test_observe_world_meta_predator_dist_present(self) -> None:
        """Predator distance remains available only as diagnostic metadata."""
        obs = self.world.observe()
        self.assertIn("diagnostic_predator_dist", obs["meta"]["diagnostic"])
        self.assertNotIn("predator_dist", obs["meta"])

    def test_observe_world_meta_predator_dist_is_non_negative(self) -> None:
        obs = self.world.observe()
        self.assertGreaterEqual(obs["meta"]["diagnostic"]["diagnostic_predator_dist"], 0)
