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
