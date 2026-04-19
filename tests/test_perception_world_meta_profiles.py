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
