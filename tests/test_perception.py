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


class PerceptTraceFunctionsTest(unittest.TestCase):
    def test_percept_trace_ttl_minimum_is_one(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.operational_profile.perception["percept_trace_ttl"] = 0.3

        self.assertEqual(_percept_trace_ttl(world), 1)

    def test_percept_trace_ttl_rounds_to_nearest_integer(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.operational_profile.perception["percept_trace_ttl"] = 3.6

        self.assertEqual(_percept_trace_ttl(world), 4)

    def test_percept_trace_decay_clips_to_unit_interval(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.operational_profile.perception["percept_trace_decay"] = 1.5
        self.assertAlmostEqual(_percept_trace_decay(world), 1.0)
        world.operational_profile.perception["percept_trace_decay"] = -0.2
        self.assertAlmostEqual(_percept_trace_decay(world), 0.0)

    def test_trace_strength_empty_trace_returns_zero(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)

        self.assertAlmostEqual(trace_strength(world, empty_percept_trace()), 0.0)

    def test_trace_strength_at_ttl_boundary_returns_zero(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        ttl = _percept_trace_ttl(world)
        trace = PerceptTrace(target=(5, 5), age=ttl, certainty=1.0)

        self.assertAlmostEqual(trace_strength(world, trace), 0.0)

    def test_trace_strength_age_zero_equals_certainty(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        trace = PerceptTrace(target=(5, 5), age=0, certainty=0.8)

        self.assertAlmostEqual(trace_strength(world, trace), 0.8)

    def test_trace_strength_decays_over_age(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        decay = _percept_trace_decay(world)
        trace_age0 = PerceptTrace(target=(5, 5), age=0, certainty=1.0)
        trace_age1 = PerceptTrace(target=(5, 5), age=1, certainty=1.0)
        if _percept_trace_ttl(world) > 1:
            expected_age1 = min(max(1.0 * (decay ** 1), 0.0), 1.0)
            self.assertAlmostEqual(trace_strength(world, trace_age1), expected_age1, places=6)
        self.assertGreaterEqual(
            trace_strength(world, trace_age0),
            trace_strength(world, trace_age1),
        )

    def test_trace_view_empty_trace_produces_zero_fields(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)

        view = trace_view(world, empty_percept_trace())

        self.assertIsNone(view["target"])
        self.assertEqual(view["age"], 0)
        self.assertAlmostEqual(view["strength"], 0.0)
        self.assertAlmostEqual(view["dx"], 0.0)
        self.assertAlmostEqual(view["dy"], 0.0)
        self.assertEqual(view["heading_dx"], 0)
        self.assertEqual(view["heading_dy"], 0)

    def test_trace_view_has_all_required_keys(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)

        view = trace_view(world, empty_percept_trace())

        for key in (
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
        ):
            self.assertIn(key, view)

    def test_trace_view_active_trace_has_nonzero_strength_and_direction(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        trace = PerceptTrace(target=(8, 5), age=0, certainty=1.0, heading_dx=1, heading_dy=0)

        view = trace_view(world, trace)

        self.assertEqual(view["target"], [8, 5])
        self.assertGreater(view["strength"], 0.0)
        self.assertNotEqual(view["dx"], 0.0)
        self.assertEqual(view["heading_dx"], 1)
        self.assertEqual(view["heading_dy"], 0)

    def test_trace_view_preserves_heading_from_capture_time(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        world.state.heading_dx = 0
        world.state.heading_dy = -1
        visible_percept = PerceivedTarget(
            visible=1.0,
            certainty=0.9,
            occluded=0.0,
            dx=0.0,
            dy=-0.25,
            dist=3,
            position=(5, 2),
        )

        updated = advance_percept_trace(
            world,
            PerceptTrace(target=None, age=0, certainty=0.0),
            visible_percept,
            [(5, 2)],
        )
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        view = trace_view(world, updated)

        self.assertEqual((updated.heading_dx, updated.heading_dy), (0, -1))
        self.assertEqual(view["heading_dx"], 0)
        self.assertEqual(view["heading_dy"], -1)

    def test_trace_view_expired_trace_has_zero_direction(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        ttl = _percept_trace_ttl(world)
        expired = PerceptTrace(target=(8, 5), age=ttl, certainty=1.0)

        view = trace_view(world, expired)

        self.assertIsNone(view["target"])
        self.assertAlmostEqual(view["strength"], 0.0)
        self.assertAlmostEqual(view["dx"], 0.0)
        self.assertAlmostEqual(view["dy"], 0.0)
        self.assertEqual(view["heading_dx"], 0)
        self.assertEqual(view["heading_dy"], 0)

    def test_advance_percept_trace_visible_target_resets_trace(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        old_trace = PerceptTrace(target=(3, 3), age=2, certainty=0.4)
        visible_percept = PerceivedTarget(
            visible=1.0,
            certainty=0.9,
            occluded=0.0,
            dx=1.0,
            dy=0.0,
            dist=3,
            position=(8, 5),
        )

        updated = advance_percept_trace(world, old_trace, visible_percept, [(8, 5)])

        self.assertEqual(updated.target, (8, 5))
        self.assertEqual(updated.age, 0)
        self.assertAlmostEqual(updated.certainty, 0.9)
        self.assertEqual((updated.heading_dx, updated.heading_dy), (1, 0))

    def test_advance_percept_trace_uses_same_target_selected_by_visible_object(self) -> None:
        world = SpiderWorld(seed=89, vision_range=8, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        positions = [(2, 5), (8, 5)]
        percept = visible_object(world, positions, radius=visible_range(world), apply_noise=False)

        updated = advance_percept_trace(
            world,
            PerceptTrace(target=None, age=0, certainty=0.0),
            percept,
            positions,
        )

        self.assertEqual(updated.target, (8, 5))

    def test_advance_percept_trace_heading_check_blocks_update_when_position_is_not_none(self) -> None:
        world = SpiderWorld(seed=89, vision_range=8, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        world.state.heading_dx = -1
        world.state.heading_dy = 0
        opposed_percept = PerceivedTarget(
            visible=1.0,
            certainty=0.9,
            occluded=0.0,
            dx=1.0,
            dy=0.0,
            dist=3,
            position=(8, 5),
        )
        old_trace = PerceptTrace(target=(3, 3), age=1, certainty=0.5)

        updated = advance_percept_trace(world, old_trace, opposed_percept, [(8, 5)])

        self.assertEqual(updated.age, 2)
        self.assertEqual(updated.target, (3, 3))

    def test_advance_percept_trace_refreshes_peripheral_target(self) -> None:
        world = SpiderWorld(seed=89, vision_range=8, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        peripheral_percept = PerceivedTarget(
            visible=1.0,
            certainty=0.7,
            occluded=0.0,
            dx=0.25,
            dy=0.5,
            dist=3,
            position=(6, 7),
        )
        self.assertEqual(
            _compute_target_visibility_zone(world, world.spider_pos(), (6, 7)),
            "peripheral",
        )

        updated = advance_percept_trace(
            world,
            PerceptTrace(target=None, age=0, certainty=0.0),
            peripheral_percept,
            [(6, 7)],
        )

        self.assertEqual(updated.target, (6, 7))
        self.assertEqual(updated.age, 0)
        self.assertAlmostEqual(updated.certainty, 0.7)

    def test_advance_percept_trace_positions_as_list_of_lists_is_handled(self) -> None:
        world = SpiderWorld(seed=89, vision_range=8, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        positions = [[8, 5]]
        percept = PerceivedTarget(
            visible=1.0,
            certainty=0.9,
            occluded=0.0,
            dx=1.0,
            dy=0.0,
            dist=3,
            position=(8, 5),
        )

        updated = advance_percept_trace(
            world,
            PerceptTrace(target=None, age=0, certainty=0.0),
            percept,
            positions,
        )

        self.assertEqual(updated.target, (8, 5))
        self.assertEqual(updated.age, 0)

    def test_advance_percept_trace_occluded_target_does_not_reset(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        old_trace = PerceptTrace(target=(3, 3), age=1, certainty=0.5)
        occluded_percept = PerceivedTarget(
            visible=0.0,
            certainty=0.5,
            occluded=1.0,
            dx=0.0,
            dy=0.0,
            dist=3,
        )

        updated = advance_percept_trace(world, old_trace, occluded_percept, [(3, 3)])

        self.assertEqual(updated.age, 2)
        self.assertEqual(updated.target, (3, 3))

    def test_advance_percept_trace_no_target_no_visible_returns_empty(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        invisible_percept = PerceivedTarget(
            visible=0.0,
            certainty=0.0,
            occluded=0.0,
            dx=0.0,
            dy=0.0,
            dist=0,
        )

        updated = advance_percept_trace(world, empty_percept_trace(), invisible_percept, [(5, 5)])

        self.assertIsNone(updated.target)
        self.assertEqual(updated.age, 0)

    def test_advance_percept_trace_ages_and_expires_at_ttl(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        ttl = _percept_trace_ttl(world)
        invisible = PerceivedTarget(
            visible=0.0,
            certainty=0.0,
            occluded=0.0,
            dx=0.0,
            dy=0.0,
            dist=0,
        )
        trace = PerceptTrace(target=(5, 5), age=0, certainty=1.0)

        for _ in range(ttl):
            trace = advance_percept_trace(world, trace, invisible, [(5, 5)])

        self.assertIsNone(trace.target)

    def test_visible_perceived_target_requires_position(self) -> None:
        with self.assertRaisesRegex(ValueError, "Visible PerceivedTarget requires position"):
            PerceivedTarget(
                visible=1.0,
                certainty=0.9,
                occluded=0.0,
                dx=1.0,
                dy=0.0,
                dist=3,
            )


class PerceptionCategoryTest(unittest.TestCase):
    def test_fresh_high_certainty_is_direct(self) -> None:
        self.assertEqual(
            _perception_category(
                certainty=0.8,
                trace_strength=0.0,
                scan_age=1,
                is_delayed=False,
            ),
            "direct",
        )

    def test_trace_takes_precedence_over_delayed(self) -> None:
        self.assertEqual(
            _perception_category(
                certainty=0.2,
                trace_strength=0.3,
                scan_age=5,
                is_delayed=True,
            ),
            "trace",
        )

    def test_delayed_when_not_direct_or_trace(self) -> None:
        self.assertEqual(
            _perception_category(
                certainty=0.2,
                trace_strength=0.0,
                scan_age=5,
                is_delayed=True,
            ),
            "delayed",
        )

    def test_uncertain_when_no_category_matches(self) -> None:
        self.assertEqual(
            _perception_category(
                certainty=0.2,
                trace_strength=0.0,
                scan_age=5,
                is_delayed=False,
            ),
            "uncertain",
        )


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


class HeadingAwarePerceptionTest(unittest.TestCase):
    def _heading_east_world(self) -> SpiderWorld:
        """
        Create a deterministic SpiderWorld positioned at (3, 3) with the agent's heading set to face east.
        
        Returns:
            SpiderWorld: A world seeded for reproducible tests with vision range 6, lizard movement effectively disabled, and the spider's heading set to (1, 0).
        """
        world = SpiderWorld(seed=101, vision_range=6, lizard_move_interval=999999)
        world.reset(seed=101)
        world.state.x, world.state.y = 3, 3
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        return world

    def test_compute_target_visibility_zone_uses_two_zone_fov(self) -> None:
        world = self._heading_east_world()

        self.assertEqual(_compute_target_visibility_zone(world, (3, 3), (5, 3)), "foveal")
        self.assertEqual(_compute_target_visibility_zone(world, (3, 3), (4, 5)), "peripheral")
        self.assertEqual(_compute_target_visibility_zone(world, (3, 3), (2, 5)), "outside")

    def test_default_visibility_zones_at_representative_angles(self) -> None:
        """
        Verify representative target angles relative to an east-facing heading are classified into the expected visibility zones.
        
        Asserts that targets in front and slightly off-center are `foveal`, targets at a lateral angle are `peripheral`, and targets behind or far to the side are `outside` for the world configured by `_heading_east_world()`.
        """
        world = self._heading_east_world()

        self.assertEqual(_compute_target_visibility_zone(world, (3, 3), (5, 3)), "foveal")
        self.assertEqual(_compute_target_visibility_zone(world, (3, 3), (5, 5)), "foveal")
        self.assertEqual(_compute_target_visibility_zone(world, (3, 3), (4, 5)), "peripheral")
        self.assertEqual(_compute_target_visibility_zone(world, (3, 3), (3, 5)), "outside")
        self.assertEqual(_compute_target_visibility_zone(world, (3, 3), (1, 5)), "outside")
        self.assertEqual(_compute_target_visibility_zone(world, (3, 3), (1, 3)), "outside")

    def test_visibility_confidence_penalizes_peripheral_zone(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "peripheral_penalty_test"
        summary["version"] = 101
        summary["perception"]["peripheral_certainty_penalty"] = 0.25
        world = SpiderWorld(
            seed=101,
            vision_range=6,
            lizard_move_interval=999999,
            operational_profile=OperationalProfile.from_summary(summary),
        )
        world.reset(seed=101)
        world.state.x, world.state.y = 3, 3

        foveal_conf = visibility_confidence(
            world,
            source=(3, 3),
            target=(6, 3),
            dist=3,
            radius=6,
            visibility_zone="foveal",
        )
        peripheral_conf = visibility_confidence(
            world,
            source=(3, 3),
            target=(6, 3),
            dist=3,
            radius=6,
            visibility_zone="peripheral",
        )
        outside_conf = visibility_confidence(
            world,
            source=(3, 3),
            target=(6, 3),
            dist=3,
            radius=6,
            visibility_zone="outside",
        )

        self.assertAlmostEqual(foveal_conf - peripheral_conf, 0.25)
        self.assertEqual(outside_conf, 0.0)

    def test_visible_object_respects_forward_fov(self) -> None:
        world = self._heading_east_world()

        front = visible_object(world, [(5, 3)], radius=visible_range(world), apply_noise=False)
        back = visible_object(world, [(1, 3)], radius=visible_range(world), apply_noise=False)

        self.assertGreater(front.visible, 0.0)
        self.assertEqual(front.occluded, 0.0)
        self.assertGreater(front.certainty, 0.0)
        self.assertEqual(back.certainty, 0.0)
        self.assertEqual(back.visible, 0.0)

    def test_visible_object_penalizes_peripheral_and_blocks_outside_targets(self) -> None:
        world = self._heading_east_world()

        foveal = visible_object(world, [(6, 3)], radius=visible_range(world), apply_noise=False)
        peripheral = visible_object(world, [(4, 5)], radius=visible_range(world), apply_noise=False)
        outside = visible_object(world, [(2, 5)], radius=visible_range(world), apply_noise=False)

        self.assertGreater(foveal.certainty, peripheral.certainty)
        self.assertGreater(peripheral.certainty, 0.0)
        self.assertEqual(outside.visible, 0.0)
        self.assertEqual(outside.certainty, 0.0)
        self.assertIsNone(outside.position)

    def test_smell_gradient_is_not_heading_gated(self) -> None:
        world = SpiderWorld(seed=101, vision_range=6, lizard_move_interval=999999)
        world.reset(seed=101)
        world.state.x, world.state.y = 3, 3
        food_position = [(1, 3)]

        world.state.heading_dx = 1
        world.state.heading_dy = 0
        facing_away = smell_gradient(
            world,
            food_position,
            radius=visible_range(world),
            apply_noise=False,
        )

        world.state.heading_dx = -1
        world.state.heading_dy = 0
        facing_toward = smell_gradient(
            world,
            food_position,
            radius=visible_range(world),
            apply_noise=False,
        )

        self.assertEqual(facing_away, facing_toward)

    def test_predator_motion_salience_is_explicit(self) -> None:
        world = SpiderWorld(seed=103, lizard_move_interval=999999)
        world.reset(seed=103)
        world.state.x, world.state.y = 3, 3
        world.state.heading_dx = 1
        world.state.heading_dy = 0

        world.lizard.x = 1
        world.lizard.y = 3
        world.lizard.mode = "PATROL"
        patrol_obs = world.observe()
        patrol_visual = VisualObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(patrol_obs["visual"])
        )
        patrol_alert = AlertObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["alert"].bind_values(patrol_obs["alert"])
        )
        self.assertAlmostEqual(patrol_visual.predator_motion_salience, 0.0)
        self.assertAlmostEqual(patrol_alert.predator_motion_salience, 0.0)

        world.lizard.x = 5
        world.lizard.y = 3
        world.lizard.mode = "CHASE"
        chase_obs = world.observe()
        chase_visual = VisualObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(chase_obs["visual"])
        )
        chase_alert = AlertObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["alert"].bind_values(chase_obs["alert"])
        )
        self.assertGreater(chase_visual.predator_motion_salience, 0.0)
        self.assertGreater(chase_alert.predator_motion_salience, 0.0)


class LineCellsTest(unittest.TestCase):
    def test_line_cells_same_point_returns_empty(self) -> None:
        cells = line_cells((3, 3), (3, 3))
        self.assertEqual(cells, [])

    def test_line_cells_horizontal_right(self) -> None:
        cells = line_cells((0, 0), (3, 0))
        self.assertEqual(cells, [(1, 0), (2, 0)])

    def test_line_cells_horizontal_left(self) -> None:
        cells = line_cells((3, 0), (0, 0))
        self.assertEqual(cells, [(2, 0), (1, 0)])

    def test_line_cells_vertical_down(self) -> None:
        cells = line_cells((0, 0), (0, 3))
        self.assertEqual(cells, [(0, 1), (0, 2)])

    def test_line_cells_adjacent_returns_empty(self) -> None:
        cells = line_cells((2, 2), (3, 2))
        self.assertEqual(cells, [])

    def test_line_cells_diagonal_excludes_endpoints(self) -> None:
        cells = line_cells((0, 0), (3, 3))
        self.assertNotIn((0, 0), cells)
        self.assertNotIn((3, 3), cells)
        self.assertGreater(len(cells), 0)

    def test_line_cells_two_apart_has_one_intermediate(self) -> None:
        cells = line_cells((0, 0), (2, 0))
        self.assertEqual(cells, [(1, 0)])


class VisibleRangeTest(unittest.TestCase):
    def _profile_with_perception_updates(self, **updates: float) -> OperationalProfile:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "perception_test_profile"
        summary["version"] = 11
        summary["perception"].update({name: float(value) for name, value in updates.items()})
        return OperationalProfile.from_summary(summary)

    def test_visible_range_full_during_day(self) -> None:
        """
        Verifies that a world reports its full vision range during daytime.
        
        Sets the world's tick to the start of day, asserts the world is not night, and checks that visible_range(world) equals the configured vision_range (4).
        """
        world = SpiderWorld(seed=1, vision_range=4, lizard_move_interval=999999)
        world.reset(seed=1)
        world.tick = 0
        self.assertFalse(world.is_night())
        self.assertEqual(visible_range(world), 4)

    def test_visible_range_reduced_at_night(self) -> None:
        world = SpiderWorld(seed=1, vision_range=4, day_length=5, night_length=10, lizard_move_interval=999999)
        world.reset(seed=1)
        world.tick = 6
        self.assertTrue(world.is_night())
        self.assertLess(visible_range(world), 4)

    def test_visible_range_minimum_two_at_night(self) -> None:
        world = SpiderWorld(seed=1, vision_range=2, day_length=5, night_length=10, lizard_move_interval=999999)
        world.reset(seed=1)
        world.tick = 6
        self.assertTrue(world.is_night())
        self.assertEqual(visible_range(world), 2)

    def test_visible_range_uses_operational_profile(self) -> None:
        world = SpiderWorld(
            seed=1,
            vision_range=4,
            day_length=5,
            night_length=10,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_perception_updates(
                night_vision_range_penalty=3.0,
                night_vision_min_range=1.0,
            ),
        )
        world.reset(seed=1)
        world.tick = 6
        self.assertEqual(visible_range(world), 1)


class HasLineOfSightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SpiderWorld(seed=1, lizard_move_interval=999999)
        self.world.reset(seed=1)

    def test_los_same_cell_is_clear(self) -> None:
        pos = self.world.spider_pos()
        self.assertTrue(has_line_of_sight(self.world, pos, pos))

    def test_los_adjacent_open_cells(self) -> None:
        self.world.state.x, self.world.state.y = 2, 2
        self.assertTrue(has_line_of_sight(self.world, (2, 2), (3, 2)))

    def test_los_blocked_by_blocked_cell(self) -> None:
        from spider_cortex_sim.maps import BLOCKED
        # Find two open cells with a blocked cell in between
        blocked = list(self.world.blocked_cells)
        if blocked:
            bx, by = blocked[0]
            for origin in [(bx - 2, by), (bx - 1, by)]:
                ox, oy = origin
                if (0 <= ox < self.world.width and 0 <= oy < self.world.height
                        and self.world.terrain_at(origin) != BLOCKED):
                    target = (bx + 2, by)
                    if (0 <= target[0] < self.world.width
                            and self.world.terrain_at(target) != BLOCKED):
                        result = has_line_of_sight(self.world, origin, target)
                        self.assertFalse(result)
                        return
        self.skipTest("no suitable blocked-cell geometry found")

    def test_los_outside_to_deep_shelter_is_false(self) -> None:
        """
        Verifies that an outside map cell does not have line of sight into a deep shelter cell.
        
        If the world has no deep shelter cells or no outside cells the test is skipped.
        """
        deep_cells = list(self.world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells")
        deep = deep_cells[0]
        # Find an outside cell
        outside = None
        for x in range(self.world.width):
            for y in range(self.world.height):
                if self.world.shelter_role_at((x, y)) == "outside":
                    outside = (x, y)
                    break
            if outside:
                break
        if outside is None:
            self.skipTest("No outside cells")
        self.assertFalse(has_line_of_sight(self.world, outside, deep))


class SmellGradientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SpiderWorld(seed=1, lizard_move_interval=999999)
        self.world.reset(seed=1)

    def test_smell_gradient_returns_zero_when_no_targets(self) -> None:
        self.world.state.x, self.world.state.y = 5, 5
        strength, gx, gy, dist = smell_gradient(self.world, [], radius=5)
        self.assertAlmostEqual(strength, 0.0)
        self.assertAlmostEqual(gx, 0.0)
        self.assertAlmostEqual(gy, 0.0)
        self.assertEqual(dist, 10**9)

    def test_smell_gradient_returns_zero_when_target_out_of_range(self) -> None:
        self.world.state.x, self.world.state.y = 0, 0
        far_target = [(self.world.width - 1, self.world.height - 1)]
        strength, _, _, _ = smell_gradient(self.world, far_target, radius=2)
        self.assertAlmostEqual(strength, 0.0)

    def test_smell_gradient_detects_nearby_target(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        nearby = [(4, 3)]
        strength, _, _, dist = smell_gradient(self.world, nearby, radius=5)
        self.assertGreater(strength, 0.0)
        self.assertEqual(dist, 1)

    def test_smell_gradient_direction_positive_x(self) -> None:
        self.world.state.x, self.world.state.y = 2, 2
        target = [(5, 2)]
        _, gx, gy, _ = smell_gradient(self.world, target, radius=5)
        self.assertGreater(gx, 0.0)
        self.assertAlmostEqual(gy, 0.0, places=5)

    def test_smell_gradient_at_same_position_returns_nonzero_strength(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        same_pos = [(3, 3)]
        strength, _, _, dist = smell_gradient(self.world, same_pos, radius=5)
        self.assertGreater(strength, 0.0)
        self.assertEqual(dist, 0)


class NoiseAwarePerceptionTest(unittest.TestCase):
    def _noise_profile(self) -> NoiseConfig:
        """
        Create a deterministic NoiseConfig tailored for perception tests.
        
        Returns:
            NoiseConfig: A noise profile named "perception_noise_test" with small, fixed jitter and zero-probability randomness:
                - visual: certainty_jitter=0.2, direction_jitter=0.1, dropout_prob=0.0
                - olfactory: strength_jitter=0.2, direction_jitter=0.1
                - motor: action_flip_prob=0.0
                - spawn: uniform_mix=0.0
                - predator: random_choice_prob=0.0
        """
        return NoiseConfig(
            name="perception_noise_test",
            visual={"certainty_jitter": 0.2, "direction_jitter": 0.1, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 0.2, "direction_jitter": 0.1},
            motor={"action_flip_prob": 0.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )

    def test_visual_noise_is_reproducible_for_same_seed(self) -> None:
        world_a = SpiderWorld(seed=5, vision_range=4, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        world_b = SpiderWorld(seed=5, vision_range=4, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        for world in (world_a, world_b):
            world.reset(seed=105)
            world.state.x, world.state.y = 2, 2
            world.lizard.x, world.lizard.y = 3, 2
            world.lizard.mode = "CHASE"

        percept_a = predator_visible_to_spider(world_a)
        percept_b = predator_visible_to_spider(world_b)

        self.assertAlmostEqual(percept_a.certainty, percept_b.certainty)
        self.assertAlmostEqual(percept_a.dx, percept_b.dx)
        self.assertAlmostEqual(percept_a.dy, percept_b.dy)
        self.assertEqual(percept_a.visible, percept_b.visible)

    def test_olfactory_noise_is_reproducible_for_same_seed(self) -> None:
        world_a = SpiderWorld(seed=7, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        world_b = SpiderWorld(seed=7, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        for world in (world_a, world_b):
            world.reset(seed=207)
            world.state.x, world.state.y = 3, 3

        smell_a = smell_gradient(world_a, [(5, 3)], radius=6)
        smell_b = smell_gradient(world_b, [(5, 3)], radius=6)

        self.assertEqual(smell_a[3], smell_b[3])
        self.assertAlmostEqual(smell_a[0], smell_b[0])
        self.assertAlmostEqual(smell_a[1], smell_b[1])
        self.assertAlmostEqual(smell_a[2], smell_b[2])

    def test_smell_field_does_not_consume_olfactory_rng(self) -> None:
        world_a = SpiderWorld(seed=8, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        world_b = SpiderWorld(seed=8, noise_profile=self._noise_profile(), lizard_move_interval=999999)
        for world in (world_a, world_b):
            world.reset(seed=208)
            world.state.x, world.state.y = 3, 3

        world_a.smell_field("food")
        smell_a = smell_gradient(world_a, [(5, 3)], radius=6)
        smell_b = smell_gradient(world_b, [(5, 3)], radius=6)

        self.assertEqual(smell_a, smell_b)

    def test_visual_noise_recomputes_visible_after_dropout(self) -> None:
        profile = NoiseConfig(
            name="visual_dropout_test",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.1, "dropout_prob": 1.0},
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={"action_flip_prob": 0.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )
        world = SpiderWorld(seed=11, vision_range=4, noise_profile=profile, lizard_move_interval=999999)
        world.reset(seed=211)
        world.state.x, world.state.y = 2, 2
        world.lizard.x, world.lizard.y = 4, 2
        world.lizard.mode = "PATROL"

        percept = predator_visible_to_spider(world)

        self.assertEqual(percept.visible, 0.0)
        self.assertEqual(percept.dx, 0.0)
        self.assertEqual(percept.dy, 0.0)

    def test_noise_profile_does_not_change_observation_shapes(self) -> None:
        world = SpiderWorld(seed=9, noise_profile="high", lizard_move_interval=999999)
        obs = world.reset(seed=9)

        self.assertEqual(obs["visual"].shape, (32,))
        self.assertEqual(obs["sensory"].shape, (12,))
        self.assertEqual(obs["hunger"].shape, (18,))
        self.assertEqual(obs["sleep"].shape, (18,))
        self.assertEqual(obs["alert"].shape, (27,))
        self.assertEqual(obs["action_context"].shape, (15,))
        self.assertEqual(obs["motor_context"].shape, (14,))


class VisibilityConfidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        """
        Prepare a deterministic SpiderWorld test fixture on self.world.
        
        Initializes self.world with seed=1, vision_range=4, and lizard_move_interval=999999, then calls reset(seed=1) so tests start from a reproducible state.
        """
        self.world = SpiderWorld(seed=1, vision_range=4, lizard_move_interval=999999)
        self.world.reset(seed=1)

    def test_visibility_confidence_at_zero_dist_is_near_one(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        conf = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(3, 3),
            dist=0,
            radius=4,
        )
        self.assertGreater(conf, 0.8)

    def test_visibility_confidence_at_max_range_is_low(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        conf = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(7, 3),
            dist=4,
            radius=4,
        )
        self.assertLess(conf, 0.5)

    def test_visibility_confidence_clipped_to_zero_one(self) -> None:
        # Very far away should not go below 0
        self.world.state.x, self.world.state.y = 3, 3
        conf = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(7, 3),
            dist=100,
            radius=2,
        )
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_visibility_confidence_motion_bonus_increases_value(self) -> None:
        self.world.state.x, self.world.state.y = 3, 3
        conf_no_bonus = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            motion_bonus=0.0,
        )
        conf_with_bonus = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            motion_bonus=0.20,
        )
        self.assertGreater(conf_with_bonus, conf_no_bonus)

    def test_visibility_confidence_uses_operational_profile_penalties(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "visibility_profile"
        summary["version"] = 12
        summary["perception"]["visibility_clutter_penalty"] = 0.0
        world = SpiderWorld(
            seed=1,
            vision_range=4,
            lizard_move_interval=999999,
            operational_profile=OperationalProfile.from_summary(summary),
        )
        world.reset(seed=1)

        clutter_target = None
        for x in range(world.width):
            for y in range(world.height):
                if world.terrain_at((x, y)) == CLUTTER:
                    clutter_target = (x, y)
                    break
            if clutter_target is not None:
                break
        if clutter_target is None:
            self.skipTest("No clutter cell found for visibility test")

        world.state.x, world.state.y = max(0, clutter_target[0] - 1), clutter_target[1]
        dist = world.manhattan(world.spider_pos(), clutter_target)
        no_penalty_conf = visibility_confidence(
            world,
            source=world.spider_pos(),
            target=clutter_target,
            dist=dist,
            radius=4,
        )

        baseline_world = SpiderWorld(seed=1, vision_range=4, lizard_move_interval=999999)
        baseline_world.reset(seed=1)
        baseline_world.state.x, baseline_world.state.y = world.spider_pos()
        baseline_conf = visibility_confidence(
            baseline_world,
            source=baseline_world.spider_pos(),
            target=clutter_target,
            dist=dist,
            radius=4,
        )

        self.assertGreater(no_penalty_conf, baseline_conf)


class LizardDetectsSpiderTest(unittest.TestCase):
    def test_deep_shelter_blocks_lizard_detection(self) -> None:
        world = SpiderWorld(seed=21, lizard_move_interval=999999)
        world.reset(seed=21)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells")
        deep = sorted(deep_cells)[len(deep_cells) // 2]
        entrance_cells = list(world.shelter_entrance_cells)
        if not entrance_cells:
            self.skipTest("No entrance cells")
        entrance = sorted(entrance_cells)[len(entrance_cells) // 2]
        world.state.x, world.state.y = deep
        world.lizard.x = max(0, entrance[0] - 1)
        world.lizard.y = entrance[1]
        self.assertFalse(lizard_detects_spider(world))

    def test_lizard_detects_exposed_spider(self) -> None:
        world = SpiderWorld(seed=21, lizard_move_interval=999999)
        world.reset(seed=21)
        entrance_cells = list(world.shelter_entrance_cells)
        if not entrance_cells:
            self.skipTest("No entrance cells")
        entrance = sorted(entrance_cells)[len(entrance_cells) // 2]
        world.state.x, world.state.y = entrance
        world.lizard.x = max(0, entrance[0] - 1)
        world.lizard.y = entrance[1]
        world.lizard_vision_range = 5
        self.assertTrue(lizard_detects_spider(world))
        self.assertTrue(predator_detects_spider(world, world.lizard))

    def test_predator_visible_to_spider_far_away(self) -> None:
        world = SpiderWorld(seed=5, vision_range=4, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 1, 1
        world.lizard.x = world.width - 1
        world.lizard.y = world.height - 1
        percept = predator_visible_to_spider(world)
        self.assertAlmostEqual(percept.visible, 0.0)

    def test_predator_visible_to_spider_very_close(self) -> None:
        world = SpiderWorld(seed=5, vision_range=4, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 2, 2
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        world.lizard.x = 3
        world.lizard.y = 2
        percept = predator_visible_to_spider(world)
        self.assertGreater(percept.certainty, 0.0)


class MultiPredatorPerceptionTest(unittest.TestCase):
    def test_predators_visible_to_spider_returns_per_type_views(self) -> None:
        world = SpiderWorld(seed=41, vision_range=6, lizard_move_interval=999999)
        world.reset(
            seed=41,
            predator_profiles=[VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )
        world.state.x, world.state.y = 3, 3
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        world.get_predator(0).x, world.get_predator(0).y = 4, 3
        world.get_predator(1).x, world.get_predator(1).y = 5, 3

        views = predators_visible_to_spider(world, apply_noise=False)

        self.assertEqual(set(views.keys()), {"visual", "olfactory"})
        self.assertEqual(views["visual"].position, (4, 3))
        self.assertEqual(views["olfactory"].position, (5, 3))

    def test_compute_per_type_threats_tracks_dominant_predator_type(self) -> None:
        world = SpiderWorld(seed=43, vision_range=6, lizard_move_interval=999999)
        world.reset(
            seed=43,
            predator_profiles=[VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )
        world.state.x, world.state.y = 3, 3
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        world.get_predator(0).x, world.get_predator(0).y = 7, 3
        world.get_predator(1).x, world.get_predator(1).y = 4, 3

        threats = compute_per_type_threats(world)

        self.assertIn("visual_predator_threat", threats)
        self.assertIn("olfactory_predator_threat", threats)
        self.assertGreater(threats["olfactory_predator_threat"], threats["visual_predator_threat"])
        self.assertEqual(
            threats["dominant_predator_type"],
            DOMINANT_PREDATOR_TYPE_OLFACTORY,
        )

    def test_predator_visible_to_spider_picks_most_threatening_visible_predator(self) -> None:
        world = SpiderWorld(seed=47, vision_range=6, lizard_move_interval=999999)
        world.reset(
            seed=47,
            predator_profiles=[VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )
        world.state.x, world.state.y = 3, 3
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        world.get_predator(0).x, world.get_predator(0).y = 6, 3
        world.get_predator(1).x, world.get_predator(1).y = 4, 3

        percept = predator_visible_to_spider(world, apply_noise=False)

        self.assertEqual(percept.position, (4, 3))

    def test_predator_visible_to_spider_scores_candidates_with_noisy_view_when_requested(self) -> None:
        world = SpiderWorld(seed=48, vision_range=6, lizard_move_interval=999999)
        world.reset(seed=48)
        predator_a = SimpleNamespace(x=1, y=1)
        predator_b = SimpleNamespace(x=2, y=2)

        def fake_visual_view(_: SpiderWorld, predator: object, *, apply_noise: bool) -> PerceivedTarget:
            if predator is predator_a:
                if apply_noise:
                    return PerceivedTarget(
                        visible=1.0,
                        certainty=0.1,
                        occluded=0.0,
                        dx=1.0,
                        dy=0.0,
                        dist=4,
                        position=(1, 1),
                    )
                return PerceivedTarget(
                    visible=1.0,
                    certainty=0.95,
                    occluded=0.0,
                    dx=1.0,
                    dy=0.0,
                    dist=1,
                    position=(1, 1),
                )
            if apply_noise:
                return PerceivedTarget(
                    visible=1.0,
                    certainty=0.9,
                    occluded=0.0,
                    dx=1.0,
                    dy=0.0,
                    dist=1,
                    position=(2, 2),
                )
            return PerceivedTarget(
                visible=1.0,
                certainty=0.4,
                occluded=0.0,
                dx=1.0,
                dy=0.0,
                dist=2,
                position=(2, 2),
            )

        with patch("spider_cortex_sim.perception._predator_visual_view", side_effect=fake_visual_view):
            percept = predator_visible_to_spider(
                world,
                predators=[predator_a, predator_b],
                apply_noise=True,
            )

        self.assertEqual(percept.position, (2, 2))

    def test_explicit_default_profile_remains_authoritative_for_detection_ranges(self) -> None:
        world = SpiderWorld(
            seed=49,
            map_template="exposed_feeding_ground",
            lizard_vision_range=1,
            predator_smell_range=1,
            lizard_move_interval=999999,
        )
        world.reset(seed=49, predator_profiles=[DEFAULT_LIZARD_PROFILE])
        world.state.x, world.state.y = 6, 6
        world.state.heading_dx = -1
        world.state.heading_dy = 0
        world.get_predator(0).x, world.get_predator(0).y = 8, 6

        distance = world.manhattan(
            world.spider_pos(),
            (world.get_predator(0).x, world.get_predator(0).y),
        )
        self.assertGreater(distance, world.lizard_vision_range)
        self.assertGreater(distance, world.predator_smell_range)
        self.assertTrue(predator_detects_spider(world, world.get_predator(0)))

    def test_visual_hunter_detection_uses_profile_vision_range(self) -> None:
        world = SpiderWorld(
            seed=50,
            map_template="exposed_feeding_ground",
            lizard_vision_range=1,
            lizard_move_interval=999999,
        )
        world.reset(seed=50, predator_profiles=[VISUAL_HUNTER_PROFILE])
        world.state.x, world.state.y = 6, 6
        world.state.last_move_dx = 0
        world.state.last_move_dy = 0
        predator = world.lizard
        predator.x, predator.y = 8, 6

        self.assertGreater(world.manhattan(world.spider_pos(), world.lizard_pos()), world.lizard_vision_range)
        self.assertTrue(has_line_of_sight(world, world.lizard_pos(), world.spider_pos()))
        self.assertTrue(predator_detects_spider(world, predator))

    def test_olfactory_hunter_detection_uses_profile_smell_range(self) -> None:
        world = SpiderWorld(
            seed=54,
            map_template="central_burrow",
            predator_smell_range=1,
            lizard_move_interval=999999,
        )
        world.reset(seed=54, predator_profiles=[OLFACTORY_HUNTER_PROFILE])
        world.state.x, world.state.y = 6, 6
        world.state.last_move_dx = 0
        world.state.last_move_dy = 0
        predator = world.lizard
        predator.x, predator.y = 8, 6

        self.assertGreater(world.manhattan(world.spider_pos(), world.lizard_pos()), world.predator_smell_range)
        self.assertFalse(has_line_of_sight(world, world.lizard_pos(), world.spider_pos()))
        self.assertTrue(predator_detects_spider(world, predator))


class ObserveWorldMetaTest(unittest.TestCase):
    def test_observe_world_meta_has_required_fields(self) -> None:
        """
        Verify that world.observe() returns a meta mapping containing all required top-level observation fields.
        
        Asserts presence of the following keys in obs["meta"]: "food_dist", "shelter_dist", "diagnostic", "night", "day", "on_shelter", "on_food", "predator_visible", "lizard_x", "lizard_y", "lizard_mode", "sleep_phase", "sleep_debt", "shelter_role", "terrain", "map_template", "reward_profile", "vision", "memory_vectors", "predators", "visual_predator_threat", "olfactory_predator_threat", "dominant_predator_type", and "dominant_predator_type_label".
        """
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        obs = world.observe()
        meta = obs["meta"]
        required = [
            "food_dist", "shelter_dist", "diagnostic",
            "night", "day", "on_shelter", "on_food",
            "predator_visible", "lizard_x", "lizard_y", "lizard_mode",
            "sleep_phase", "sleep_debt", "shelter_role",
            "terrain", "map_template", "reward_profile",
            "vision", "memory_vectors", "predators",
            "visual_predator_threat", "olfactory_predator_threat",
            "dominant_predator_type", "dominant_predator_type_label",
        ]
        for field in required:
            self.assertIn(field, meta, f"Missing meta field: {field}")

    def test_observe_world_reuses_sampled_predator_views(self) -> None:
        world = SpiderWorld(seed=22)
        world.reset(
            seed=22,
            predator_profiles=[VISUAL_HUNTER_PROFILE, VISUAL_HUNTER_PROFILE],
        )
        sampled_positions: list[tuple[int, int]] = []

        def fake_visual_view(
            _: SpiderWorld,
            predator: object,
            *,
            apply_noise: bool,
        ) -> PerceivedTarget:
            self.assertTrue(apply_noise)
            position = (int(predator.x), int(predator.y))
            sampled_positions.append(position)
            dist = world.manhattan(world.spider_pos(), position)
            return PerceivedTarget(
                visible=1.0,
                certainty=0.8,
                occluded=0.0,
                dx=1.0,
                dy=0.0,
                dist=dist,
                position=position,
            )

        with patch(
            "spider_cortex_sim.perception._predator_visual_view",
            side_effect=fake_visual_view,
        ) as sampled_view:
            obs = world.observe()

        self.assertEqual(sampled_view.call_count, world.predator_count)
        self.assertCountEqual(sampled_positions, world.predator_positions())
        self.assertGreater(obs["meta"]["visual_predator_threat"], 0.0)

    def test_observe_world_meta_retains_diagnostic_privileged_values(self) -> None:
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        obs = world.observe()
        meta = obs["meta"]
        diagnostic = meta["diagnostic"]
        self.assertIsInstance(diagnostic["diagnostic_predator_dist"], int)
        self.assertIsInstance(diagnostic["diagnostic_home_dx"], float)
        self.assertIsInstance(diagnostic["diagnostic_home_dy"], float)
        self.assertIsInstance(diagnostic["diagnostic_home_dist"], float)
        self.assertNotIn("predator_dist", meta)
        self.assertNotIn("home_dx", meta)
        self.assertNotIn("home_dy", meta)
        self.assertNotIn("home_dist", meta)

    def test_observe_world_memory_vectors_all_four_keys(self) -> None:
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        obs = world.observe()
        memory_vectors = obs["meta"]["memory_vectors"]
        self.assertIn("food", memory_vectors)
        self.assertIn("predator", memory_vectors)
        self.assertIn("shelter", memory_vectors)
        self.assertIn("escape", memory_vectors)

    def test_observe_world_vision_has_food_shelter_predator(self) -> None:
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        obs = world.observe()
        vision = obs["meta"]["vision"]
        self.assertIn("food", vision)
        self.assertIn("shelter", vision)
        self.assertIn("predator", vision)
        self.assertIn("predators_by_type", vision)

    def test_observe_world_meta_predator_dump_includes_profiles(self) -> None:
        world = SpiderWorld(seed=53, lizard_move_interval=999999)
        world.reset(
            seed=53,
            predator_profiles=[VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )

        meta = world.observe()["meta"]

        self.assertEqual(len(meta["predators"]), 2)
        self.assertIn("profile", meta["predators"][0])
        self.assertEqual(meta["predators"][0]["profile"]["detection_style"], "visual")
        self.assertEqual(meta["predators"][1]["profile"]["detection_style"], "olfactory")


class OperationalProfilePerceptionIntegrationTest(unittest.TestCase):
    """Additional tests verifying that perception functions use the operational profile correctly."""

    def _make_profile(self, **perception_updates: float) -> OperationalProfile:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "perception_integration_test"
        summary["version"] = 77
        summary["perception"].update({k: float(v) for k, v in perception_updates.items()})
        return OperationalProfile.from_summary(summary)

    def test_world_stores_operational_profile(self) -> None:
        profile = self._make_profile(night_vision_range_penalty=2.0)
        world = SpiderWorld(seed=1, lizard_move_interval=999999, operational_profile=profile)
        self.assertIs(world.operational_profile, profile)

    def test_visible_range_night_penalty_scales_correctly(self) -> None:
        # With penalty=2, vision_range=5, min=1 → night range = max(1, 5-2) = 3
        world = SpiderWorld(
            seed=1,
            vision_range=5,
            day_length=5,
            night_length=10,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(
                night_vision_range_penalty=2.0,
                night_vision_min_range=1.0,
            ),
        )
        world.reset(seed=1)
        world.tick = 6
        self.assertTrue(world.is_night())
        self.assertEqual(visible_range(world), 3)

    def test_visible_range_min_range_enforced(self) -> None:
        # With large penalty, min_range clamps result
        world = SpiderWorld(
            seed=1,
            vision_range=3,
            day_length=5,
            night_length=10,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(
                night_vision_range_penalty=10.0,
                night_vision_min_range=2.0,
            ),
        )
        world.reset(seed=1)
        world.tick = 6
        self.assertTrue(world.is_night())
        self.assertEqual(visible_range(world), 2)

    def test_visibility_night_penalty_from_profile(self) -> None:
        # Zero night penalty should increase confidence compared to default
        world_no_penalty = SpiderWorld(
            seed=1,
            vision_range=4,
            day_length=5,
            night_length=10,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(visibility_night_penalty=0.0),
        )
        world_no_penalty.reset(seed=1)
        world_no_penalty.tick = 6
        world_no_penalty.state.x, world_no_penalty.state.y = 3, 3

        world_default = SpiderWorld(seed=1, vision_range=4, day_length=5, night_length=10, lizard_move_interval=999999)
        world_default.reset(seed=1)
        world_default.tick = 6
        world_default.state.x, world_default.state.y = 3, 3

        conf_no_penalty = visibility_confidence(
            world_no_penalty,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
        )
        conf_default = visibility_confidence(
            world_default,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
        )
        self.assertGreater(conf_no_penalty, conf_default)

    def test_lizard_detection_threshold_from_profile_blocks_detection(self) -> None:
        # Very high detection threshold → lizard can't detect spider even when close
        world = SpiderWorld(
            seed=21,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(lizard_detection_threshold=1.1),
        )
        world.reset(seed=21)
        entrance_cells = list(world.shelter_entrance_cells)
        if not entrance_cells:
            self.skipTest("No entrance cells")
        entrance = sorted(entrance_cells)[len(entrance_cells) // 2]
        world.state.x, world.state.y = entrance
        world.lizard.x = max(0, entrance[0] - 1)
        world.lizard.y = entrance[1]
        world.lizard_vision_range = 5
        world.lizard.profile = None
        self.assertFalse(lizard_detects_spider(world))

    def test_predator_motion_bonus_from_profile_increases_exported_salience(self) -> None:
        # A higher predator_motion_bonus should yield a stronger exported salience channel when the predator is seen.
        world_high = SpiderWorld(
            seed=5,
            vision_range=4,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(predator_motion_bonus=0.50),
        )
        world_low = SpiderWorld(
            seed=5,
            vision_range=4,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(predator_motion_bonus=0.0),
        )
        for world in (world_high, world_low):
            world.reset(seed=5)
            world.state.x, world.state.y = 2, 2
            world.lizard.x = 3
            world.lizard.y = 2
            world.lizard.mode = "CHASE"
            world.state.heading_dx = 1
            world.state.heading_dy = 0

        high_obs = world_high.observe()
        low_obs = world_low.observe()
        high_visual = VisualObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(high_obs["visual"])
        )
        low_visual = VisualObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(low_obs["visual"])
        )
        high_alert = AlertObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["alert"].bind_values(high_obs["alert"])
        )
        low_alert = AlertObservation.from_mapping(
            OBSERVATION_INTERFACE_BY_KEY["alert"].bind_values(low_obs["alert"])
        )
        self.assertGreater(high_visual.predator_motion_salience, low_visual.predator_motion_salience)
        self.assertGreater(high_alert.predator_motion_salience, low_alert.predator_motion_salience)

    def test_occluded_certainty_base_from_profile(self) -> None:
        # With a very high occluded_certainty_base, certainty of occluded target should be higher
        world_high = SpiderWorld(
            seed=1,
            vision_range=8,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(
                occluded_certainty_base=0.90,
                occluded_certainty_min=0.10,
                occluded_certainty_decay_per_step=0.01,
            ),
        )
        world_low = SpiderWorld(
            seed=1,
            vision_range=8,
            lizard_move_interval=999999,
            operational_profile=self._make_profile(
                occluded_certainty_base=0.10,
                occluded_certainty_min=0.05,
                occluded_certainty_decay_per_step=0.01,
            ),
        )
        # Find an occluded food item by moving spider far away and measuring perceived certainty
        for world in (world_high, world_low):
            world.reset(seed=1)
            world.state.x, world.state.y = 0, 3

        # Use a known occluded line on the current map so the branch is always exercised.
        for world in (world_high, world_low):
            world.food_positions = [(5, 6)]
            world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
            world.state.heading_dx = 1
            world.state.heading_dy = 0

        from spider_cortex_sim.perception import visible_object
        high_percept = visible_object(
            world_high,
            positions=world_high.food_positions,
            origin=world_high.spider_pos(),
            radius=visible_range(world_high),
        )
        low_percept = visible_object(
            world_low,
            positions=world_low.food_positions,
            origin=world_low.spider_pos(),
            radius=visible_range(world_low),
        )
        self.assertIsNotNone(high_percept)
        self.assertIsNotNone(low_percept)
        self.assertEqual(high_percept.visible, 0.0)
        self.assertEqual(low_percept.visible, 0.0)
        self.assertGreater(high_percept.occluded, 0.0)
        self.assertGreater(low_percept.occluded, 0.0)
        self.assertGreater(high_percept.certainty, low_percept.certainty)


class FovThresholdsTest(unittest.TestCase):
    def _make_world(self, **perception_overrides: float) -> SpiderWorld:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["perception"].update(perception_overrides)
        profile = OperationalProfile.from_summary(summary)
        world = SpiderWorld(seed=77, vision_range=6, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=77)
        return world

    def _zone_for_angle(self, angle_degrees: float, **perception_overrides: float) -> str:
        world = self._make_world(**perception_overrides)
        world.state.x = 0
        world.state.y = 0
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        radians = np.deg2rad(angle_degrees)
        target = (float(np.cos(radians)), float(np.sin(radians)))
        return _compute_target_visibility_zone(world, world.spider_pos(), target)

    def test_default_thresholds_are_forty_five_and_seventy(self) -> None:
        world = self._make_world()
        foveal, peripheral = _fov_thresholds(world)
        self.assertAlmostEqual(foveal, 45.0)
        self.assertAlmostEqual(peripheral, 70.0)

    def test_default_fov_half_angle_is_forty_five(self) -> None:
        world = self._make_world()
        foveal, _ = _fov_thresholds(world)
        self.assertAlmostEqual(foveal, 45.0)

    def test_default_peripheral_half_angle_is_seventy(self) -> None:
        world = self._make_world()
        _, peripheral = _fov_thresholds(world)
        self.assertAlmostEqual(peripheral, 70.0)

    def test_angle_between_new_and_old_foveal_limits_is_now_peripheral(self) -> None:
        self.assertEqual(
            self._zone_for_angle(50.0, fov_half_angle=60.0, peripheral_half_angle=90.0),
            "foveal",
        )
        self.assertEqual(self._zone_for_angle(50.0), "peripheral")

    def test_angle_between_new_and_old_peripheral_limits_is_now_outside(self) -> None:
        self.assertEqual(
            self._zone_for_angle(80.0, fov_half_angle=60.0, peripheral_half_angle=90.0),
            "peripheral",
        )
        self.assertEqual(self._zone_for_angle(80.0), "outside")

    def test_exact_foveal_boundary_is_foveal(self) -> None:
        self.assertEqual(self._zone_for_angle(45.0), "foveal")

    def test_exact_peripheral_boundary_is_peripheral(self) -> None:
        self.assertEqual(self._zone_for_angle(70.0), "peripheral")

    def test_just_outside_peripheral_boundary_is_outside(self) -> None:
        self.assertEqual(self._zone_for_angle(70.1), "outside")

    def test_custom_fov_half_angle_is_returned(self) -> None:
        world = self._make_world(fov_half_angle=45.0)
        foveal, _ = _fov_thresholds(world)
        self.assertAlmostEqual(foveal, 45.0)

    def test_custom_peripheral_half_angle_is_returned(self) -> None:
        world = self._make_world(peripheral_half_angle=120.0)
        _, peripheral = _fov_thresholds(world)
        self.assertAlmostEqual(peripheral, 120.0)

    def test_foveal_cannot_exceed_peripheral(self) -> None:
        world = self._make_world(fov_half_angle=100.0, peripheral_half_angle=60.0)
        foveal, peripheral = _fov_thresholds(world)
        self.assertLessEqual(foveal, peripheral)

    def test_fov_half_angle_clipped_to_180(self) -> None:
        world = self._make_world(fov_half_angle=200.0)
        foveal, _ = _fov_thresholds(world)
        self.assertAlmostEqual(foveal, 180.0)

    def test_fov_half_angle_clipped_to_zero(self) -> None:
        world = self._make_world(fov_half_angle=-10.0)
        foveal, _ = _fov_thresholds(world)
        self.assertAlmostEqual(foveal, 0.0)

    def test_peripheral_angle_clipped_to_zero(self) -> None:
        world = self._make_world(fov_half_angle=0.0, peripheral_half_angle=-5.0)
        _, peripheral = _fov_thresholds(world)
        self.assertGreaterEqual(peripheral, 0.0)

    def test_returns_tuple_of_two_floats(self) -> None:
        world = self._make_world()
        result = _fov_thresholds(world)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        for val in result:
            self.assertIsInstance(val, float)


class ApplyVisibilityZoneCertaintyTest(unittest.TestCase):
    def _make_world(self, penalty: float = 0.35) -> SpiderWorld:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["perception"]["peripheral_certainty_penalty"] = penalty
        profile = OperationalProfile.from_summary(summary)
        world = SpiderWorld(seed=55, vision_range=6, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=55)
        return world

    def test_foveal_zone_returns_certainty_unchanged(self) -> None:
        world = self._make_world()
        result = _apply_visibility_zone_certainty(world, 0.8, "foveal")
        self.assertAlmostEqual(result, 0.8)

    def test_outside_zone_returns_zero(self) -> None:
        world = self._make_world()
        result = _apply_visibility_zone_certainty(world, 0.9, "outside")
        self.assertAlmostEqual(result, 0.0)

    def test_outside_zone_always_returns_zero_regardless_of_certainty(self) -> None:
        world = self._make_world()
        for certainty in (0.0, 0.5, 1.0):
            with self.subTest(certainty=certainty):
                result = _apply_visibility_zone_certainty(world, certainty, "outside")
                self.assertAlmostEqual(result, 0.0)

    def test_peripheral_zone_applies_configured_penalty(self) -> None:
        world = self._make_world(penalty=0.25)
        result = _apply_visibility_zone_certainty(world, 0.8, "peripheral")
        self.assertAlmostEqual(result, 0.55)

    def test_peripheral_zone_default_penalty_is_0_35(self) -> None:
        world = self._make_world(penalty=0.35)
        result = _apply_visibility_zone_certainty(world, 0.7, "peripheral")
        self.assertAlmostEqual(result, 0.35)

    def test_peripheral_certainty_clipped_to_zero_when_penalty_exceeds_certainty(self) -> None:
        world = self._make_world(penalty=0.5)
        result = _apply_visibility_zone_certainty(world, 0.3, "peripheral")
        self.assertAlmostEqual(result, 0.0)

    def test_foveal_certainty_clipped_to_unit_interval(self) -> None:
        world = self._make_world()
        result = _apply_visibility_zone_certainty(world, 1.5, "foveal")
        self.assertAlmostEqual(result, 1.0)

    def test_result_always_in_unit_interval(self) -> None:
        world = self._make_world()
        for zone in ("foveal", "peripheral", "outside"):
            for certainty in (-0.5, 0.0, 0.5, 1.0, 1.5):
                result = _apply_visibility_zone_certainty(world, certainty, zone)  # type: ignore[arg-type]
                self.assertGreaterEqual(result, 0.0)
                self.assertLessEqual(result, 1.0)


class VisibilityConfidenceZoneAliasTest(unittest.TestCase):
    """Test the backward-compatible `zone` alias parameter for visibility_confidence."""

    def setUp(self) -> None:
        self.world = SpiderWorld(seed=1, vision_range=4, lizard_move_interval=999999)
        self.world.reset(seed=1)
        self.world.state.x, self.world.state.y = 3, 3

    def test_zone_alias_produces_same_result_as_visibility_zone_for_foveal(self) -> None:
        conf_visibility_zone = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            visibility_zone="foveal",
        )
        conf_zone = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            zone="foveal",
        )
        self.assertAlmostEqual(conf_visibility_zone, conf_zone)

    def test_zone_alias_produces_same_result_as_visibility_zone_for_peripheral(self) -> None:
        conf_visibility_zone = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            visibility_zone="peripheral",
        )
        conf_zone = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            zone="peripheral",
        )
        self.assertAlmostEqual(conf_visibility_zone, conf_zone)

    def test_zone_alias_produces_zero_for_outside(self) -> None:
        conf = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            zone="outside",
        )
        self.assertAlmostEqual(conf, 0.0)

    def test_zone_alias_overrides_visibility_zone_when_both_provided(self) -> None:
        """When both zone and visibility_zone are provided, zone alias takes precedence."""
        conf_via_zone_foveal = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            visibility_zone="outside",
            zone="foveal",
        )
        # zone="foveal" overrides visibility_zone="outside" — result should be positive
        self.assertGreater(conf_via_zone_foveal, 0.0)

    def test_default_visibility_zone_is_foveal(self) -> None:
        """Default visibility_zone is foveal, so certainty should not be penalized."""
        conf_explicit_foveal = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
            visibility_zone="foveal",
        )
        conf_default = visibility_confidence(
            self.world,
            source=(3, 3),
            target=(5, 3),
            dist=2,
            radius=4,
        )
        self.assertAlmostEqual(conf_explicit_foveal, conf_default)
