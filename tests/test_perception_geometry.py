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
