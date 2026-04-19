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

class SpiderWorldCoreTest(SpiderWorldTestBase):
    def test_default_operational_profile_is_copied_for_world_runtime(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)

        self.assertIsNot(world.operational_profile, DEFAULT_OPERATIONAL_PROFILE)
        self.assertEqual(
            world.operational_profile.to_summary(),
            DEFAULT_OPERATIONAL_PROFILE.to_summary(),
        )

        world.operational_profile.perception["percept_trace_decay"] = -0.2
        self.assertAlmostEqual(
            DEFAULT_OPERATIONAL_PROFILE.perception["percept_trace_decay"],
            0.65,
        )

    def test_observation_shapes_and_metadata(self) -> None:
        """
        Verify that a SpiderWorld observation contains correctly shaped vectors and consistent metadata.
        
        Asserts exact lengths for each observation vector (visual, sensory, hunger, sleep, alert, action_context, motor_context), that meta fields `map_template` and `reward_profile` have expected values, and that required metadata keys (`sleep_debt`, `shelter_role`, `terrain`, `vision`, `memory_vectors`) are present. Confirms `vision` includes `food` and `predator`, that `motor_extra` equals `motor_context`, and that bound observation interfaces expose the expected signal names. Validates that `meta` booleans for day/night and on_food/on_shelter are consistent with values parsed from the bound action and motor context observations.
        """
        world = SpiderWorld(seed=3)
        obs = world.reset(seed=3)
        self.assertEqual(obs["visual"].shape, (32,))
        self.assertEqual(obs["sensory"].shape, (12,))
        self.assertEqual(obs["hunger"].shape, (18,))
        self.assertEqual(obs["sleep"].shape, (18,))
        self.assertEqual(obs["alert"].shape, (27,))
        self.assertEqual(obs["action_context"].shape, (15,))
        self.assertEqual(obs["motor_context"].shape, (14,))
        self.assertEqual(obs["meta"]["map_template"], "central_burrow")
        self.assertEqual(obs["meta"]["reward_profile"], "classic")
        self.assertIn("sleep_debt", obs["meta"])
        self.assertIn("shelter_role", obs["meta"])
        self.assertIn("terrain", obs["meta"])
        self.assertIn("vision", obs["meta"])
        self.assertIn("memory_vectors", obs["meta"])
        self.assertIn("heading", obs["meta"])
        self.assertIn("percept_traces", obs["meta"])
        self.assertIn("food", obs["meta"]["vision"])
        self.assertIn("predator", obs["meta"]["vision"])
        np.testing.assert_array_equal(obs["motor_extra"], obs["motor_context"])

        visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(obs["visual"])
        action_context = ActionContextObservation.from_mapping(
            ACTION_CONTEXT_INTERFACE.bind_values(obs["action_context"])
        )
        motor_context = MotorContextObservation.from_mapping(
            MOTOR_CONTEXT_INTERFACE.bind_values(obs["motor_context"])
        )

        self.assertEqual(set(visual.keys()), set(OBSERVATION_INTERFACE_BY_KEY["visual"].signal_names))
        self.assertEqual(obs["meta"]["day"], bool(visual["day"]))
        self.assertEqual(obs["meta"]["night"], bool(visual["night"]))
        self.assertEqual(obs["meta"]["on_food"], bool(action_context.on_food))
        self.assertEqual(obs["meta"]["on_shelter"], bool(action_context.on_shelter))
        self.assertEqual(obs["meta"]["on_shelter"], bool(motor_context.on_shelter))

    def test_reset_spawns_multiple_predators_and_preserves_lizard_alias(self) -> None:
        world = SpiderWorld(seed=31, lizard_move_interval=999999)
        world.reset(
            seed=31,
            predator_profiles=[VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )

        self.assertEqual(world.predator_count, 2)
        self.assertEqual(
            [predator.profile for predator in world.predators],
            [VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )
        self.assertEqual(
            world.predator_positions(),
            [(predator.x, predator.y) for predator in world.predators],
        )
        self.assertEqual(len(set(world.predator_positions())), 2)
        self.assertIs(world.lizard, world.get_predator(0))
        self.assertEqual(world.lizard_pos(), world.predator_positions()[0])
        self.assertIs(world.predator_controller, world.predator_controllers[0])
        self.assertEqual(world.predator_controller.predator_index, 0)
        self.assertIs(world.predator_controller._predator(world), world.lizard)
        for index, controller in enumerate(world.predator_controllers):
            self.assertEqual(controller.predator_index, index)
            predator = world.get_predator(controller.predator_index)
            self.assertIs(controller._predator(world), predator)
            self.assertEqual((predator.x, predator.y), world.predator_positions()[index])

    def test_random_spawn_cell_fallback_respects_min_spider_distance(self) -> None:
        world = SpiderWorld(
            seed=37,
            map_template="exposed_feeding_ground",
            lizard_move_interval=999999,
        )
        world.reset(seed=37)
        world.state.x, world.state.y = 2, 2

        class FixedIndexRng:
            def integers(self, low: int, high: int | None = None) -> int:
                del low, high
                return 0

        world.spawn_rng = FixedIndexRng()
        cell = world._random_spawn_cell(
            [world.spider_pos()],
            min_spider_distance=5,
            avoid_lizard=False,
        )

        self.assertGreaterEqual(world.manhattan(cell, world.spider_pos()), 5)

    def test_reset_initializes_heading_toward_nearest_entrance(self) -> None:
        world = SpiderWorld(seed=71, lizard_move_interval=999999)
        obs = world.reset(seed=71)
        entrance = world.nearest_shelter_entrance(origin=world.spider_pos())
        expected = world._heading_toward(entrance, origin=world.spider_pos())
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), expected)
        self.assertEqual(obs["meta"]["heading"]["dx"], expected[0])
        self.assertEqual(obs["meta"]["heading"]["dy"], expected[1])

    def test_successful_move_updates_heading_and_blocked_move_preserves_it(self) -> None:
        world = SpiderWorld(seed=73, lizard_move_interval=999999)
        world.reset(seed=73)
        start = None
        action_name = None
        for x in range(world.width):
            for y in range(world.height):
                for candidate_action, (dx, dy) in ACTION_DELTAS.items():
                    if candidate_action == "STAY":
                        continue
                    next_pos = (x + dx, y + dy)
                    if not (0 <= next_pos[0] < world.width and 0 <= next_pos[1] < world.height):
                        continue
                    if world.is_walkable((x, y)) and world.is_walkable(next_pos):
                        start = (x, y)
                        action_name = candidate_action
                        break
                if start is not None:
                    break
            if start is not None:
                break
        self.assertIsNotNone(start)
        self.assertIsNotNone(action_name)

        world.state.x, world.state.y = start
        world.state.heading_dx = 0
        world.state.heading_dy = 1
        world.step(ACTION_TO_INDEX[action_name])
        expected_dx, expected_dy = ACTION_DELTAS[action_name]
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), (expected_dx, expected_dy))

        world.state.heading_dx = -1
        world.state.heading_dy = 0
        world.state.x, world.state.y = 0, 0
        world.step(ACTION_TO_INDEX["MOVE_LEFT"])
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), (-1, 0))

    def test_percept_traces_refresh_decay_and_observe_is_read_only(self) -> None:
        profile = _profile_with_perception(perceptual_delay_ticks=0.0, perceptual_delay_noise=0.0)
        world = SpiderWorld(
            seed=79,
            vision_range=6,
            lizard_move_interval=999999,
            operational_profile=profile,
        )
        world.reset(seed=79)
        pair = None
        for x in range(world.width - 1):
            for y in range(world.height):
                if world.is_walkable((x, y)) and world.is_walkable((x + 1, y)):
                    pair = ((x, y), (x + 1, y))
                    break
            if pair is not None:
                break
        self.assertIsNotNone(pair)

        current, front = pair
        world.state.x, world.state.y = current
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        world.food_positions = [front]
        self._move_lizard_to_safe_corner(world)

        obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])
        initial_strength = float(obs["meta"]["percept_traces"]["food"]["strength"])
        self.assertEqual(world.state.food_trace.target, front)
        self.assertGreater(initial_strength, 0.0)

        age_before_observe = world.state.food_trace.age
        world.observe()
        self.assertEqual(world.state.food_trace.age, age_before_observe)

        world.food_positions = []
        obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])
        decayed_strength = float(obs["meta"]["percept_traces"]["food"]["strength"])
        self.assertLess(decayed_strength, initial_strength)

        configured_ttl = max(1, round(world.operational_profile.perception["percept_trace_ttl"]))
        for _ in range(configured_ttl):
            world.step(ACTION_TO_INDEX["STAY"])
        self.assertIsNone(world.state.food_trace.target)

    def test_noise_profile_low_is_reproducible_for_initial_state_and_spawns(self) -> None:
        """Identical seeds with noise_profile='low' must reproduce episode seed, spawns, and initial physiology."""
        world_a = SpiderWorld(seed=41, noise_profile="low", lizard_move_interval=999999)
        world_b = SpiderWorld(seed=41, noise_profile="low", lizard_move_interval=999999)
        world_a.reset(seed=123)
        world_b.reset(seed=123)

        self.assertEqual(world_a.episode_seed, world_b.episode_seed)
        self.assertEqual(world_a.food_positions, world_b.food_positions)
        self.assertEqual(world_a.lizard_pos(), world_b.lizard_pos())
        self.assertAlmostEqual(world_a.state.hunger, world_b.state.hunger)
        self.assertAlmostEqual(world_a.state.fatigue, world_b.state.fatigue)

    def test_heading_components_from_delta_all_sign_combinations(self) -> None:
        """
        _heading_components_from_delta must map any (dx, dy) to their independent signs in {-1, 0, 1}.
        """
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        cases = [
            ((3, 5), (1, 1)),
            ((-2, 0), (-1, 0)),
            ((0, -7), (0, -1)),
            ((0, 0), (0, 0)),
            ((-100, 100), (-1, 1)),
            ((1, -1), (1, -1)),
        ]
        for delta, expected in cases:
            with self.subTest(delta=delta):
                self.assertEqual(world._heading_components_from_delta(*delta), expected)

    def test_heading_toward_none_target_returns_zero(self) -> None:
        world = SpiderWorld(seed=91, lizard_move_interval=999999)
        world.reset(seed=91)
        self.assertEqual(world._heading_toward(None), (0, 0))

    def test_heading_toward_with_explicit_origin(self) -> None:
        world = SpiderWorld(seed=93, lizard_move_interval=999999)
        world.reset(seed=93)
        result = world._heading_toward((10, 5), origin=(7, 3))
        self.assertEqual(result, (1, 1))

    def test_heading_toward_same_as_origin_returns_zero(self) -> None:
        world = SpiderWorld(seed=95, lizard_move_interval=999999)
        world.reset(seed=95)
        pos = world.spider_pos()
        self.assertEqual(world._heading_toward(pos), (0, 0))

    def test_heading_toward_uses_spider_pos_when_origin_omitted(self) -> None:
        world = SpiderWorld(seed=97, lizard_move_interval=999999)
        world.reset(seed=97)
        world.state.x, world.state.y = 5, 5
        target = (8, 5)
        result = world._heading_toward(target)
        self.assertEqual(result, (1, 0))

    def test_motor_noise_populates_intended_and_executed_action_fields(self) -> None:
        profile = NoiseConfig(
            name="motor_flip_test",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={"action_flip_prob": 1.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )
        world = SpiderWorld(seed=47, noise_profile=profile, lizard_move_interval=999999)
        world.reset(seed=47)

        _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_UP"])

        self.assertEqual(info["intended_action"], "MOVE_UP")
        self.assertNotEqual(info["executed_action"], "MOVE_UP")
        self.assertEqual(info["action"], info["intended_action"])
        self.assertTrue(info["motor_noise_applied"])

    def test_food_and_shelter_trigger_autonomic_behaviors(self) -> None:
        """
        Verify that encountering food triggers eating and being in deep shelter triggers sleep with appropriate state and rewards.
        
        The test places the spider on a food tile with high hunger and asserts that the world registers an eating event, grants a substantial positive reward, reduces hunger, and reports consistent reward components. It then places the spider in a deep shelter during the night with high fatigue and sleep debt (and moves the predator away), steps the world multiple times, and asserts that sleep occurs (ending in `DEEP_SLEEP`), rewards are positive and accounted for in reward components, and fatigue and sleep debt decrease.
        """
        world = SpiderWorld(seed=5)
        world.reset(seed=5)

        world.state.x, world.state.y = world.food_positions[0]
        world.state.hunger = 0.9
        _, reward_eat, _, info_eat = world.step(ACTION_TO_INDEX["STAY"])
        self.assertTrue(info_eat["ate"])
        self.assertGreater(reward_eat, 1.0)
        self.assertLess(world.state.hunger, 0.5)
        self._assert_reward_components(reward_eat, info_eat)

        world.state.x, world.state.y = self._deep_cell(world)
        world.tick = world.day_length + 1
        world.state.fatigue = 0.9
        world.state.sleep_debt = 0.8
        world.state.hunger = 0.3
        self._move_lizard_to_safe_corner(world)
        reward_sleep = 0.0
        info_sleep = None
        for _ in range(3):
            _, reward_sleep, _, info_sleep = world.step(ACTION_TO_INDEX["STAY"])
        self.assertIsNotNone(info_sleep)
        self.assertTrue(info_sleep["slept"])
        self.assertEqual(info_sleep["state"]["sleep_phase"], "DEEP_SLEEP")
        self.assertGreater(reward_sleep, 0.0)
        self.assertLess(world.state.fatigue, 0.75)
        self.assertLess(world.state.sleep_debt, 0.8)
        self._assert_reward_components(reward_sleep, info_sleep)

    def test_sleep_phase_progression_requires_deep_shelter_and_reduces_sleep_debt(self) -> None:
        world = SpiderWorld(seed=13, lizard_move_interval=999999)
        world.reset(seed=13)
        world.state.x, world.state.y = self._deep_cell(world)
        world.tick = world.day_length + 1
        world.state.hunger = 0.05
        world.state.fatigue = 0.75
        world.state.sleep_debt = 0.7
        self._move_lizard_to_safe_corner(world)

        phases = []
        debts = []
        for _ in range(3):
            _, reward, _, info = world.step(ACTION_TO_INDEX["STAY"])
            phases.append(info["state"]["sleep_phase"])
            debts.append(info["state"]["sleep_debt"])
            self._assert_reward_components(reward, info)

        self.assertEqual(phases, ["SETTLING", "RESTING", "DEEP_SLEEP"])
        self.assertLess(debts[1], debts[0])
        self.assertLess(debts[2], debts[1])

    def test_daytime_rest_is_lower_quality_than_night_rest(self) -> None:
        def run_rest(tick: int) -> tuple[list[str], float, float]:
            world = SpiderWorld(seed=61, lizard_move_interval=999999)
            world.reset(seed=61)
            world.state.x, world.state.y = self._deep_cell(world)
            world.tick = tick
            world.state.hunger = 0.05
            world.state.fatigue = 0.76
            world.state.sleep_debt = 0.72
            self._move_lizard_to_safe_corner(world)
            phases = []
            reward = 0.0
            for _ in range(3):
                _, reward, _, info = world.step(ACTION_TO_INDEX["STAY"])
                phases.append(info["state"]["sleep_phase"])
            return phases, reward, world.state.sleep_debt

        night_phases, night_reward, night_debt = run_rest(37)
        day_phases, day_reward, day_debt = run_rest(2)
        self.assertEqual(night_phases[-1], "DEEP_SLEEP")
        self.assertEqual(day_phases[-1], "RESTING")
        self.assertGreater(night_reward, day_reward)
        self.assertLess(night_debt, day_debt)

    def test_sleep_interrupts_on_movement_and_predator_threat(self) -> None:
        move_world = SpiderWorld(seed=17, lizard_move_interval=999999)
        move_world.reset(seed=17)
        move_world.state.x, move_world.state.y = self._deep_cell(move_world)
        move_world.tick = move_world.day_length + 1
        move_world.state.hunger = 0.05
        move_world.state.fatigue = 0.65
        move_world.state.sleep_debt = 0.6
        self._move_lizard_to_safe_corner(move_world)
        move_world.step(ACTION_TO_INDEX["STAY"])
        _, move_reward, _, move_info = move_world.step(ACTION_TO_INDEX["MOVE_UP"])

        self.assertEqual(move_info["state"]["sleep_phase"], "AWAKE")
        self.assertEqual(move_info["state"]["rest_streak"], 0)
        self.assertFalse(move_info["slept"])
        self.assertGreater(move_info["state"]["sleep_debt"], 0.6)
        self._assert_reward_components(move_reward, move_info)

        threat_world = SpiderWorld(seed=19, lizard_move_interval=999999)
        threat_world.reset(seed=19)
        deep = self._deep_cell(threat_world)
        threat_world.state.x, threat_world.state.y = deep
        threat_world.tick = threat_world.day_length + 1
        threat_world.state.hunger = 0.05
        threat_world.state.fatigue = 0.65
        threat_world.state.sleep_debt = 0.6
        threat_world.step(ACTION_TO_INDEX["STAY"])
        entrance = self._entrance_cell(threat_world)
        threat_world.lizard.x, threat_world.lizard.y = max(0, entrance[0] - 1), entrance[1]
        _, threat_reward, _, threat_info = threat_world.step(ACTION_TO_INDEX["STAY"])

        self.assertEqual(threat_info["state"]["sleep_phase"], "AWAKE")
        self.assertEqual(threat_info["state"]["rest_streak"], 0)
        self.assertFalse(threat_info["slept"])
        self._assert_reward_components(threat_reward, threat_info)

    def test_lizard_contact_hurts_and_respects_shelter_geometry(self) -> None:
        world = SpiderWorld(seed=9, lizard_move_interval=1)
        world.reset(seed=9)

        world.state.x, world.state.y = 0, 0
        world.lizard.x, world.lizard.y = 0, 0
        health_before = world.state.health
        _, reward_contact, _, info_contact = world.step(ACTION_TO_INDEX["STAY"])
        predator_contact_events = [
            item
            for item in info_contact["event_log"]
            if item["stage"] == "predator_contact" and item["name"] == "predator_contact"
        ]
        self.assertTrue(info_contact["predator_contact"])
        self.assertEqual(len(predator_contact_events), 1)
        self.assertLess(world.state.health, health_before)
        self._assert_reward_components(reward_contact, info_contact)

        deep = self._deep_cell(world)
        entrance = self._entrance_cell(world)
        world.state.x, world.state.y = deep
        world.lizard.x, world.lizard.y = max(0, entrance[0] - 1), entrance[1]
        world.lizard.profile = None
        for _ in range(5):
            world.step(ACTION_TO_INDEX["STAY"])
            self.assertNotIn(world.lizard_pos(), world.shelter_cells)

    def test_predator_contact_is_applied_only_once_per_tick(self) -> None:
        world = SpiderWorld(seed=10, lizard_move_interval=999999)
        world.reset(seed=10)
        world.state.x, world.state.y = 0, 0
        world.lizard.x, world.lizard.y = 0, 0

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        predator_contact_events = [
            item
            for item in info["event_log"]
            if item["stage"] == "predator_contact" and item["name"] == "predator_contact"
        ]
        self.assertTrue(info.get("predator_contact"))
        self.assertEqual(len(predator_contact_events), 1)

    def test_deep_shelter_blocks_lizard_detection(self) -> None:
        """
        Verify that a lizard cannot detect the spider while the spider is inside a deep shelter but can detect the spider when it is at the shelter entrance.
        
        Creates a world with frequent lizard movement, places the spider in a deep shelter cell and the lizard just outside the entrance and asserts detection is false, then moves the spider to the entrance cell and asserts detection becomes true.
        """
        world = SpiderWorld(seed=21, lizard_move_interval=1)
        world.reset(seed=21)
        entrance = self._entrance_cell(world)
        deep = self._deep_cell(world)

        world.state.x, world.state.y = deep
        world.lizard.x, world.lizard.y = max(0, entrance[0] - 1), entrance[1]
        self.assertFalse(world.lizard_detects_spider())

        world.state.x, world.state.y = entrance
        self.assertTrue(world.lizard_detects_spider())

    def test_predator_waits_near_entrance_after_losing_spider_in_shelter(self) -> None:
        """
        Verify that when the spider moves from the entrance into deep shelter, the predator transitions from PATROL→ORIENT, then to INVESTIGATE/WAIT, and ultimately settles into WAIT targeting the entrance after a failed chase.
        
        Asserts:
        - A PATROL lizard placed outside the entrance transitions to ORIENT and reports {"from": "PATROL", "to": "ORIENT"}.
        - After the spider retreats into deep shelter, the lizard enters either INVESTIGATE or WAIT and reports a non-None predator transition.
        - When forced into CHASE with appropriate internal counters and last-known spider position set to the deep shelter, the lizard transitions to WAIT, sets its wait_target to the entrance, and reports {"from": "CHASE", "to": "WAIT"}.
        """
        world = SpiderWorld(seed=27, lizard_move_interval=1, map_template="entrance_funnel")
        world.reset(seed=27)
        entrance = self._entrance_cell(world)
        deep = self._deep_cell(world)
        ambush = self._outside_entrance_cell(world)

        world.state.x, world.state.y = entrance
        world.lizard = LizardState(x=ambush[0], y=ambush[1], mode="PATROL")
        _, _, _, orient_info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertEqual(world.lizard.mode, "ORIENT")
        self.assertEqual(orient_info["predator_transition"], {"from": "PATROL", "to": "ORIENT"})

        world.state.x, world.state.y = deep
        _, _, _, investigate_info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertIn(world.lizard.mode, {"INVESTIGATE", "WAIT"})
        self.assertIsNotNone(investigate_info["predator_transition"])

        world.lizard.mode = "CHASE"
        world.lizard.mode_ticks = 4
        world.lizard.last_known_spider = deep
        world.lizard.chase_streak = 4
        _, _, _, wait_info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertEqual(world.lizard.mode, "WAIT")
        self.assertEqual(world.lizard.wait_target, entrance)
        self.assertEqual(wait_info["predator_transition"], {"from": "CHASE", "to": "WAIT"})

    def test_predator_wait_state_can_decay_into_patrol(self) -> None:
        """
        Verifies that a lizard in the WAIT state can transition back to PATROL within a few steps.
        
        Sets up a world using the "entrance_funnel" template with a lizard positioned at an outside-entrance ambush in the WAIT mode, advances the world up to four steps, collects any reported predator state transitions, and asserts that a transition {"from": "WAIT", "to": "PATROL"} occurs.
        """
        world = SpiderWorld(seed=33, lizard_move_interval=1, map_template="entrance_funnel")
        world.reset(seed=33)
        entrance = self._entrance_cell(world)
        ambush = self._outside_entrance_cell(world)
        world.lizard = LizardState(x=ambush[0], y=ambush[1], mode="WAIT", wait_target=entrance)

        transitions = []
        for _ in range(4):
            _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
            if info["predator_transition"] is not None:
                transitions.append(info["predator_transition"])

        self.assertIn({"from": "WAIT", "to": "PATROL"}, transitions)

    def test_predator_failed_chase_flows_into_recover(self) -> None:
        world = SpiderWorld(seed=35, lizard_move_interval=1, map_template="entrance_funnel")
        world.reset(seed=35)
        entrance = self._entrance_cell(world)
        deep = self._deep_cell(world)
        ambush = self._outside_entrance_cell(world)
        world.state.x, world.state.y = deep
        world.lizard = LizardState(
            x=ambush[0],
            y=ambush[1],
            mode="CHASE",
            mode_ticks=4,
            last_known_spider=deep,
            wait_target=entrance,
            chase_streak=4,
        )

        transitions = []
        for _ in range(6):
            _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
            if info["predator_transition"] is not None:
                transitions.append(info["predator_transition"])

        self.assertIn({"from": "CHASE", "to": "WAIT"}, transitions)
        self.assertIn({"from": "WAIT", "to": "RECOVER"}, transitions)

    def test_step_reports_distance_deltas(self) -> None:
        world = SpiderWorld(seed=39, lizard_move_interval=999999)
        world.reset(seed=39)
        world.state.x, world.state.y = 1, 1
        world.food_positions = [(3, 1)]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1

        _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_RIGHT"])

        self.assertEqual(set(info["distance_deltas"].keys()), {"food", "shelter", "predator"})
        self.assertEqual(info["distance_deltas"]["food"], 1)

    def test_step_always_emits_event_log(self) -> None:
        world = SpiderWorld(seed=45, lizard_move_interval=999999)
        world.reset(seed=45)

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        self.assertIn("event_log", info)
        self.assertIsInstance(info["event_log"], list)
        self.assertTrue(info["event_log"])
        self.assertEqual({"stage", "name", "payload"}, set(info["event_log"][0].keys()))

    def test_step_accepts_numpy_integer_action_idx(self) -> None:
        world = SpiderWorld(seed=47, lizard_move_interval=999999)
        world.reset(seed=47)

        _, _, _, info = world.step(np.int64(ACTION_TO_INDEX["STAY"]))

        self.assertEqual(info["intended_action"], "STAY")
