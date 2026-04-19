import unittest
from collections import deque
from collections.abc import Mapping, Sequence
from unittest.mock import patch

import numpy as np

from spider_cortex_sim import stages as tick_stages
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_DELTAS,
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    ORIENT_HEADINGS,
    ActionContextObservation,
    MotorContextObservation,
)
from spider_cortex_sim.maps import MAP_TEMPLATE_NAMES, build_map_template
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.predator import (
    OLFACTORY_HUNTER_PROFILE,
    VISUAL_HUNTER_PROFILE,
    LizardState,
)
from spider_cortex_sim.world_types import PerceptTrace, TickContext
from spider_cortex_sim.world import (
    ACTION_TO_INDEX,
    MOVE_DELTAS,
    REWARD_COMPONENT_NAMES,
    PerceptualBuffer,
    SpiderWorld,
    _copy_observation_payload,
    _is_temporal_direction_field,
    _refresh_perception_for_active_scan,
    _scan_age_for_heading,
)

from tests.fixtures.world import SpiderWorldTestBase

from tests.fixtures.world import _profile_with_perception

class SpiderWorldMemoryMapsScenariosTest(SpiderWorldTestBase):
    def test_predator_escape_bonus_only_emits_once_per_threat_episode(self) -> None:
        world = SpiderWorld(seed=77, lizard_move_interval=999999)
        world.reset(seed=77)
        world.state.x, world.state.y = self._deep_cell(world)
        self._move_lizard_to_safe_corner(world)
        world.state.recent_contact = 1.0

        _, _, _, info1 = world.step(ACTION_TO_INDEX["STAY"])
        _, _, _, info2 = world.step(ACTION_TO_INDEX["STAY"])
        _, _, _, info3 = world.step(ACTION_TO_INDEX["STAY"])

        self.assertTrue(info1["predator_escape"])
        self.assertFalse(info2["predator_escape"])
        self.assertFalse(info3["predator_escape"])
        self.assertEqual(world.state.predator_escapes, 1)

    def test_memory_persists_after_visibility_loss(self) -> None:
        """
        Verify food and predator memories persist and age after leaving view.
        """
        world = SpiderWorld(seed=31, lizard_move_interval=999999)
        world.reset(seed=31)
        world.state.x, world.state.y = 2, 2
        world.state.heading_dx = 1
        world.state.heading_dy = 1
        world.food_positions = [(2, 3)]
        world.lizard.x, world.lizard.y = 4, 2
        world.step(ACTION_TO_INDEX["STAY"])

        self.assertEqual(world.state.food_memory.target, (2, 3))
        self.assertEqual(world.state.predator_memory.target, (4, 2))

        world.food_positions = [(world.width - 1, world.height - 1)]
        world.lizard.x, world.lizard.y = 0, world.height - 1
        world.step(ACTION_TO_INDEX["STAY"])

        self.assertEqual(world.state.food_memory.age, 1)
        self.assertEqual(world.state.predator_memory.age, 1)
        self.assertIsNotNone(world.state.food_memory.target)
        self.assertIsNotNone(world.state.predator_memory.target)

    def test_memory_guides_retreat_after_predator_leaves_view(self) -> None:
        profile = _profile_with_perception(perceptual_delay_ticks=0.0)
        world = SpiderWorld(seed=67, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=67)
        world.state.x, world.state.y = 3, 3
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        world.food_positions = [(world.width - 1, world.height - 1)]
        world.lizard.x, world.lizard.y = 5, 3
        world.step(ACTION_TO_INDEX["STAY"])

        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        brain = self._reflex_brain(seed=11)
        decision = brain.act(world.observe(), sample=False)
        self.assertEqual(world.state.predator_memory.target, (5, 3))
        self.assertEqual(world.state.predator_memory.age, 0)
        self.assertEqual(decision.action_idx, ACTION_TO_INDEX["MOVE_LEFT"])

    def test_memory_guides_food_approach_after_food_leaves_view(self) -> None:
        profile = _profile_with_perception(perceptual_delay_ticks=0.0)
        world = SpiderWorld(seed=71, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=71)
        world.state.x, world.state.y = 2, 2
        world.state.hunger = 0.92
        world.food_positions = [(4, 2)]
        world.state.heading_dx, world.state.heading_dy = world._heading_toward(
            world.food_positions[0],
            origin=world.spider_pos(),
        )
        self._move_lizard_to_safe_corner(world)
        world.step(ACTION_TO_INDEX["STAY"])

        world.food_positions = [(world.width - 1, world.height - 1)]
        brain = self._reflex_brain(seed=13)
        decision = brain.act(world.observe(), sample=False)
        self.assertEqual(world.state.food_memory.target, (4, 2))
        self.assertEqual(world.state.food_memory.age, 0)
        self.assertEqual(decision.action_idx, ACTION_TO_INDEX["MOVE_RIGHT"])

    def test_map_templates_keep_reachability(self) -> None:
        for name in MAP_TEMPLATE_NAMES:
            world = SpiderWorld(seed=41, map_template=name)
            world.reset(seed=41)
            start = world.map_template.spider_start
            food_target = world.map_template.food_spawn_cells[0]
            lizard_target = world.map_template.lizard_spawn_cells[0]
            self.assertTrue(self._reachable(world, start, food_target))
            self.assertTrue(self._reachable(world, start, lizard_target))

    def test_map_templates_start_spider_in_safe_shelter(self) -> None:
        for name in MAP_TEMPLATE_NAMES:
            world = SpiderWorld(seed=73, map_template=name)
            world.reset(seed=73)
            start = world.map_template.spider_start
            self.assertIn(start, world.shelter_interior_cells | world.shelter_deep_cells)
            self.assertTrue(world.on_shelter())
            self.assertNotIn(world.lizard_pos(), world.shelter_cells)

    def test_two_shelters_rejects_too_small_width(self) -> None:
        """
        Verifies that creating a "two_shelters" map template rejects widths smaller than 8.
        
        Calls build_map_template("two_shelters", width=7, height=12) and asserts a ValueError is raised with a message containing "two_shelters requires width >= 8".
        """
        with self.assertRaisesRegex(ValueError, "two_shelters requires width >= 8"):
            build_map_template("two_shelters", width=7, height=12)

    def test_refresh_memory_public_method_initializes_memory(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 2, 2
        world.food_positions = [(2, 3)]
        world.lizard.x, world.lizard.y = 4, 2
        world.state.food_memory.target = None
        world.state.food_memory.age = 0
        world.refresh_memory(initial=True)
        self.assertEqual(world.state.food_memory.target, (2, 3))

    def test_refresh_memory_public_method_ages_slots(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 2, 2
        world.state.food_memory.target = (5, 5)
        world.state.food_memory.age = 0
        world.food_positions = [(world.width - 1, world.height - 1)]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.refresh_memory(initial=False)
        self.assertEqual(world.state.food_memory.age, 1)

    def test_refresh_memory_public_updates_predator_memory_with_escape(self) -> None:
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        world.state.x, world.state.y = 2, 2
        world.state.last_move_dx = 1
        world.state.last_move_dy = 0
        world.state.escape_memory.target = None
        world.refresh_memory(predator_escape=True, initial=True)
        self.assertIsNotNone(world.state.escape_memory.target)

    def test_scenario_night_rest_sets_correct_state(self) -> None:
        """
        Set up the "night_rest" scenario and verify the world is in night and the spider is positioned for rest.
        
        Asserts that:
        - world.is_night() is True
        - world.state.fatigue > 0.5
        - the spider's shelter role at its current position is one of "inside", "deep", or "entrance"
        """
        from spider_cortex_sim.scenarios import SCENARIOS
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        scenario = SCENARIOS["night_rest"]
        scenario.setup(world)
        self.assertTrue(world.is_night())
        self.assertGreater(world.state.fatigue, 0.5)
        self.assertIn(world.shelter_role_at(world.spider_pos()), {"inside", "deep", "entrance"})

    def test_scenario_predator_edge_sets_correct_state(self) -> None:
        from spider_cortex_sim.scenarios import SCENARIOS
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        scenario = SCENARIOS["predator_edge"]
        predator_memory_before = world.state.predator_memory.target
        scenario.setup(world)
        self.assertNotEqual(world.state.predator_memory.target, predator_memory_before)
        self.assertEqual(world.state.predator_memory.target, world.lizard_pos())
        self.assertEqual(world.state.predator_memory.age, 0)

    def test_all_scenarios_initialize_without_error(self) -> None:
        """
        Verify that every configured scenario can initialize a SpiderWorld and produce a valid observation containing "visual" and "meta" keys.
        
        For each scenario name, this test constructs a SpiderWorld using the scenario's map template, resets and runs the scenario setup, obtains an observation via world.observe(), and asserts that the observation includes the "visual" and "meta" entries.
        """
        from spider_cortex_sim.scenarios import SCENARIOS, SCENARIO_NAMES
        for name in SCENARIO_NAMES:
            spec = SCENARIOS[name]
            world = SpiderWorld(seed=42, lizard_move_interval=999999, map_template=spec.map_template)
            world.reset(seed=42)
            spec.setup(world)
            obs = world.observe()
            self.assertIn("visual", obs)
            self.assertIn("meta", obs)
