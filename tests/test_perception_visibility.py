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
