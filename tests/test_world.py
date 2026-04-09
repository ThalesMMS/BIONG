import unittest
from collections import deque

import numpy as np

from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_DELTAS,
    ACTION_CONTEXT_INTERFACE,
    MOTOR_CONTEXT_INTERFACE,
    OBSERVATION_INTERFACE_BY_KEY,
    ActionContextObservation,
    MotorContextObservation,
)
from spider_cortex_sim.maps import MAP_TEMPLATE_NAMES, build_map_template
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.predator import LizardState
from spider_cortex_sim.world import ACTION_TO_INDEX, REWARD_COMPONENT_NAMES, SpiderWorld


class SpiderWorldTest(unittest.TestCase):
    def _deep_cell(self, world: SpiderWorld) -> tuple[int, int]:
        return sorted(world.shelter_deep_cells)[len(world.shelter_deep_cells) // 2]

    def _interior_cell(self, world: SpiderWorld) -> tuple[int, int]:
        return sorted(world.shelter_interior_cells)[len(world.shelter_interior_cells) // 2]

    def _entrance_cell(self, world: SpiderWorld) -> tuple[int, int]:
        """
        Return the median cell (by sorted order) from the world's shelter entrance cells.
        
        Returns:
            (int, int): The (x, y) coordinate tuple of the median entrance cell.
        """
        return sorted(world.shelter_entrance_cells)[len(world.shelter_entrance_cells) // 2]

    def _outside_entrance_cell(self, world: SpiderWorld) -> tuple[int, int]:
        """
        Selects a lizard-walkable cell adjacent to the shelter entrance, falling back to a spawn cell if none are suitable.
        
        Parameters:
            world (SpiderWorld): World instance whose map, dimensions, and lizard walkability are consulted.
        
        Returns:
            tuple[int, int]: An (x, y) coordinate that is adjacent to the shelter entrance and lizard-walkable, or the first entry from `world.map_template.lizard_spawn_cells` if no adjacent walkable cell is found.
        """
        entrance = self._entrance_cell(world)
        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1)):
            candidate = (entrance[0] + dx, entrance[1] + dy)
            if not (0 <= candidate[0] < world.width and 0 <= candidate[1] < world.height):
                continue
            if world.is_lizard_walkable(candidate):
                return candidate
        return sorted(world.map_template.lizard_spawn_cells)[0]

    def _move_lizard_to_safe_corner(self, world: SpiderWorld) -> None:
        """
        Move the lizard to the spawn cell that is farthest from the spider.
        
        Parameters:
            world (SpiderWorld): World instance whose `map_template.lizard_spawn_cells` and spider position are used to choose the destination.
        """
        candidates = sorted(
            world.map_template.lizard_spawn_cells,
            key=lambda cell: -world.manhattan(cell, world.spider_pos()),
        )
        world.lizard.x, world.lizard.y = candidates[0]

    def _assert_reward_components(self, reward: float, info: dict[str, object]) -> None:
        reward_components = info["reward_components"]
        self.assertEqual(set(reward_components.keys()), set(REWARD_COMPONENT_NAMES))
        self.assertAlmostEqual(sum(reward_components.values()), reward)

    def _reflex_brain(self, seed: int = 0) -> SpiderBrain:
        """
        Create a SpiderBrain configured as a deterministic "reflex" brain with all action and motor network parameters zeroed.
        
        Parameters:
            seed (int): Random seed used to initialize the brain.
        
        Returns:
            SpiderBrain: A brain whose action_center and motor_cortex parameters are set to zero so outputs depend only on fixed biases (deterministic reflex behavior).
        """
        brain = SpiderBrain(seed=seed, module_dropout=0.0)
        brain.action_center.W1.fill(0.0)
        brain.action_center.b1.fill(0.0)
        brain.action_center.W2_policy.fill(0.0)
        brain.action_center.b2_policy.fill(0.0)
        brain.action_center.W2_value.fill(0.0)
        brain.action_center.b2_value.fill(0.0)
        brain.motor_cortex.W1.fill(0.0)
        brain.motor_cortex.b1.fill(0.0)
        brain.motor_cortex.W2.fill(0.0)
        brain.motor_cortex.b2.fill(0.0)
        if brain.module_bank is not None:
            for network in brain.module_bank.modules.values():
                network.W1.fill(0.0)
                network.b1.fill(0.0)
                network.W2.fill(0.0)
                network.b2.fill(0.0)
        return brain

    def test_observation_shapes_and_metadata(self) -> None:
        """
        Verify that a SpiderWorld observation contains correctly shaped vectors and consistent metadata.
        
        Asserts exact lengths for each observation vector (visual, sensory, hunger, sleep, alert, action_context, motor_context), that meta fields `map_template` and `reward_profile` have expected values, and that required metadata keys (`sleep_debt`, `shelter_role`, `terrain`, `vision`, `memory_vectors`) are present. Confirms `vision` includes `food` and `predator`, that `motor_extra` equals `motor_context`, and that bound observation interfaces expose the expected signal names. Validates that `meta` booleans for day/night and on_food/on_shelter are consistent with values parsed from the bound action and motor context observations.
        """
        world = SpiderWorld(seed=3)
        obs = world.reset(seed=3)
        self.assertEqual(obs["visual"].shape, (23,))
        self.assertEqual(obs["sensory"].shape, (12,))
        self.assertEqual(obs["hunger"].shape, (16,))
        self.assertEqual(obs["sleep"].shape, (19,))
        self.assertEqual(obs["alert"].shape, (23,))
        self.assertEqual(obs["action_context"].shape, (16,))
        self.assertEqual(obs["motor_context"].shape, (10,))
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
        world = SpiderWorld(seed=79, vision_range=6, lizard_move_interval=999999)
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

        world.step(ACTION_TO_INDEX["STAY"])
        initial_strength = world._trace_strength(world.state.food_trace)
        self.assertEqual(world.state.food_trace.target, front)
        self.assertGreater(initial_strength, 0.0)

        age_before_observe = world.state.food_trace.age
        world.observe()
        self.assertEqual(world.state.food_trace.age, age_before_observe)

        world.food_positions = []
        world.step(ACTION_TO_INDEX["STAY"])
        self.assertLess(world._trace_strength(world.state.food_trace), initial_strength)

        for _ in range(world._percept_trace_ttl()):
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

    def test_percept_trace_ttl_minimum_is_one(self) -> None:
        """TTL is clipped to at least 1 even if the config value rounds to 0."""
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.operational_profile.perception["percept_trace_ttl"] = 0.3
        self.assertEqual(world._percept_trace_ttl(), 1)

    def test_percept_trace_ttl_rounds_to_nearest_integer(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.operational_profile.perception["percept_trace_ttl"] = 3.6
        self.assertEqual(world._percept_trace_ttl(), 4)

    def test_percept_trace_decay_clips_to_unit_interval(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.operational_profile.perception["percept_trace_decay"] = 1.5
        self.assertAlmostEqual(world._percept_trace_decay(), 1.0)
        world.operational_profile.perception["percept_trace_decay"] = -0.2
        self.assertAlmostEqual(world._percept_trace_decay(), 0.0)

    def test_trace_strength_empty_trace_returns_zero(self) -> None:
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        empty = world._empty_percept_trace()
        self.assertAlmostEqual(world._trace_strength(empty), 0.0)

    def test_trace_strength_at_ttl_boundary_returns_zero(self) -> None:
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        ttl = world._percept_trace_ttl()
        trace = PerceptTrace(target=(5, 5), age=ttl, certainty=1.0)
        self.assertAlmostEqual(world._trace_strength(trace), 0.0)

    def test_trace_strength_age_zero_equals_certainty(self) -> None:
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        trace = PerceptTrace(target=(5, 5), age=0, certainty=0.8)
        self.assertAlmostEqual(world._trace_strength(trace), 0.8)

    def test_trace_strength_decays_over_age(self) -> None:
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        decay = world._percept_trace_decay()
        trace_age0 = PerceptTrace(target=(5, 5), age=0, certainty=1.0)
        trace_age1 = PerceptTrace(target=(5, 5), age=1, certainty=1.0)
        if world._percept_trace_ttl() > 1:
            expected_age1 = min(max(1.0 * (decay ** 1), 0.0), 1.0)
            self.assertAlmostEqual(world._trace_strength(trace_age1), expected_age1, places=6)
        self.assertGreaterEqual(world._trace_strength(trace_age0), world._trace_strength(trace_age1))

    def test_trace_view_empty_trace_produces_zero_fields(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        view = world._trace_view(world._empty_percept_trace())
        self.assertIsNone(view["target"])
        self.assertEqual(view["age"], 0)
        self.assertAlmostEqual(view["strength"], 0.0)
        self.assertAlmostEqual(view["dx"], 0.0)
        self.assertAlmostEqual(view["dy"], 0.0)

    def test_trace_view_has_all_required_keys(self) -> None:
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        view = world._trace_view(world._empty_percept_trace())
        for key in ("target", "age", "certainty", "strength", "dx", "dy", "ttl", "decay"):
            self.assertIn(key, view)

    def test_trace_view_active_trace_has_nonzero_strength_and_direction(self) -> None:
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        trace = PerceptTrace(target=(8, 5), age=0, certainty=1.0)
        view = world._trace_view(trace)
        self.assertGreater(view["strength"], 0.0)
        self.assertNotEqual(view["dx"], 0.0)

    def test_trace_view_expired_trace_has_zero_direction(self) -> None:
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        ttl = world._percept_trace_ttl()
        expired = PerceptTrace(target=(8, 5), age=ttl, certainty=1.0)
        view = world._trace_view(expired)
        self.assertAlmostEqual(view["strength"], 0.0)
        self.assertAlmostEqual(view["dx"], 0.0)
        self.assertAlmostEqual(view["dy"], 0.0)

    def test_advance_percept_trace_visible_target_resets_trace(self) -> None:
        from spider_cortex_sim.perception import PerceivedTarget
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
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
        positions = [(8, 5)]
        updated = world._advance_percept_trace(old_trace, visible_percept, positions)
        self.assertEqual(updated.target, (8, 5))
        self.assertEqual(updated.age, 0)
        self.assertAlmostEqual(updated.certainty, 0.9)

    def test_visible_perceived_target_requires_position(self) -> None:
        from spider_cortex_sim.perception import PerceivedTarget

        with self.assertRaisesRegex(ValueError, "Visible PerceivedTarget requires position"):
            PerceivedTarget(
                visible=1.0,
                certainty=0.9,
                occluded=0.0,
                dx=1.0,
                dy=0.0,
                dist=3,
            )

    def test_advance_percept_trace_uses_same_target_selected_by_visible_object(self) -> None:
        from spider_cortex_sim.perception import visible_object, visible_range
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, vision_range=8, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        positions = [(2, 5), (8, 5)]
        percept = visible_object(world, positions, radius=visible_range(world), apply_noise=False)
        updated = world._advance_percept_trace(
            PerceptTrace(target=None, age=0, certainty=0.0),
            percept,
            positions,
        )
        self.assertEqual(updated.target, (8, 5))

    def test_advance_percept_trace_heading_check_blocks_update_when_position_is_not_none(self) -> None:
        """Regression: when percept.position is provided, an opposed heading prevents
        the trace from being updated — the existing heading-gating logic remains active."""
        from spider_cortex_sim.perception import PerceivedTarget
        from spider_cortex_sim.world_types import PerceptTrace

        world = SpiderWorld(seed=89, vision_range=8, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        # Heading points LEFT — opposed to target at (8, 5)
        world.state.heading_dx = -1
        world.state.heading_dy = 0

        positions = [(8, 5)]
        # percept.position is explicitly set → heading check is active
        opposed_percept = PerceivedTarget(
            visible=1.0,
            certainty=0.9,
            occluded=0.0,
            dx=1.0,
            dy=0.0,
            dist=3,
            position=(8, 5),  # position set: heading check applies
        )

        old_trace = PerceptTrace(target=(3, 3), age=1, certainty=0.5)
        updated = world._advance_percept_trace(old_trace, opposed_percept, positions)

        # Opposed heading with explicit position → trace should age, not reset
        self.assertEqual(updated.age, 2)
        self.assertEqual(updated.target, (3, 3))

    def test_advance_percept_trace_positions_as_list_of_lists_is_handled(self) -> None:
        """candidate_positions conversion handles positions given as list-of-lists."""
        from spider_cortex_sim.perception import PerceivedTarget
        from spider_cortex_sim.world_types import PerceptTrace

        world = SpiderWorld(seed=89, vision_range=8, lizard_move_interval=999999)
        world.reset(seed=89)
        world.state.x, world.state.y = 5, 5
        world.state.heading_dx = 1
        world.state.heading_dy = 0

        # Pass positions as list of lists (not tuples)
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

        updated = world._advance_percept_trace(
            PerceptTrace(target=None, age=0, certainty=0.0),
            percept,
            positions,
        )

        self.assertEqual(updated.target, (8, 5))
        self.assertEqual(updated.age, 0)

    def test_advance_percept_trace_occluded_target_does_not_reset(self) -> None:
        from spider_cortex_sim.perception import PerceivedTarget
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        old_trace = PerceptTrace(target=(3, 3), age=1, certainty=0.5)
        occluded_percept = PerceivedTarget(visible=0.0, certainty=0.5, occluded=1.0, dx=0.0, dy=0.0, dist=3)
        updated = world._advance_percept_trace(old_trace, occluded_percept, [(3, 3)])
        self.assertEqual(updated.age, 2)
        self.assertEqual(updated.target, (3, 3))

    def test_advance_percept_trace_no_target_no_visible_returns_empty(self) -> None:
        from spider_cortex_sim.perception import PerceivedTarget
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        empty_trace = world._empty_percept_trace()
        invisible_percept = PerceivedTarget(visible=0.0, certainty=0.0, occluded=0.0, dx=0.0, dy=0.0, dist=0)
        updated = world._advance_percept_trace(empty_trace, invisible_percept, [(5, 5)])
        self.assertIsNone(updated.target)
        self.assertEqual(updated.age, 0)

    def test_advance_percept_trace_ages_and_expires_at_ttl(self) -> None:
        from spider_cortex_sim.perception import PerceivedTarget
        from spider_cortex_sim.world_types import PerceptTrace
        world = SpiderWorld(seed=89, lizard_move_interval=999999)
        world.reset(seed=89)
        ttl = world._percept_trace_ttl()
        invisible = PerceivedTarget(visible=0.0, certainty=0.0, occluded=0.0, dx=0.0, dy=0.0, dist=0)
        trace = PerceptTrace(target=(5, 5), age=0, certainty=1.0)
        for _ in range(ttl):
            trace = world._advance_percept_trace(trace, invisible, [(5, 5)])
        self.assertIsNone(trace.target)

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
        self.assertEqual(info["action"], info["executed_action"])
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

        night_phases, night_reward, night_debt = run_rest(19)
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

    def test_event_log_records_pipeline_stages_in_order(self) -> None:
        world = SpiderWorld(seed=49, lizard_move_interval=999999)
        world.reset(seed=49)

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        stages = [item["stage"] for item in info["event_log"]]
        first_index = {stage: stages.index(stage) for stage in {
            "pre_tick",
            "action",
            "terrain_and_wakefulness",
            "predator_contact",
            "autonomic",
            "predator_update",
            "reward",
            "postprocess",
            "memory",
            "finalize",
        }}
        self.assertLess(first_index["pre_tick"], first_index["action"])
        self.assertLess(first_index["action"], first_index["terrain_and_wakefulness"])
        self.assertLess(first_index["terrain_and_wakefulness"], first_index["predator_contact"])
        self.assertLess(first_index["predator_contact"], first_index["autonomic"])
        self.assertLess(first_index["autonomic"], first_index["predator_update"])
        self.assertLess(first_index["predator_update"], first_index["reward"])
        self.assertLess(first_index["reward"], first_index["postprocess"])
        self.assertLess(first_index["postprocess"], first_index["memory"])
        self.assertLess(first_index["memory"], first_index["finalize"])

    def test_event_log_memory_stage_runs_after_tick_increment(self) -> None:
        world = SpiderWorld(seed=51, lizard_move_interval=999999)
        world.reset(seed=51)
        tick_before = world.tick

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        memory_event = next(item for item in info["event_log"] if item["stage"] == "memory")
        finalize_event = next(item for item in info["event_log"] if item["stage"] == "finalize")
        memory_index = next(i for i, item in enumerate(info["event_log"]) if item["stage"] == "memory")
        finalize_index = next(i for i, item in enumerate(info["event_log"]) if item["stage"] == "finalize")
        self.assertEqual(world.tick, tick_before + 1)
        self.assertEqual(memory_event["payload"]["tick"], tick_before + 1)
        self.assertEqual(finalize_event["payload"]["tick"], tick_before + 1)
        self.assertEqual(memory_event["name"], "memory_refreshed")
        self.assertLess(memory_index, finalize_index)

    def test_event_log_pre_tick_snapshot_has_required_fields(self) -> None:
        world = SpiderWorld(seed=53, lizard_move_interval=999999)
        world.reset(seed=53)

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        pre_tick_event = next(item for item in info["event_log"] if item["stage"] == "pre_tick")
        payload = pre_tick_event["payload"]
        expected_keys = {
            "tick", "spider_pos", "lizard_pos", "was_on_shelter",
            "prev_shelter_role", "prev_food_dist", "prev_shelter_dist",
            "prev_predator_dist", "prev_predator_visible", "night", "rest_streak",
        }
        self.assertEqual(set(payload.keys()), expected_keys)

    def test_event_log_pre_tick_snapshot_captures_positions(self) -> None:
        world = SpiderWorld(seed=55, lizard_move_interval=999999)
        world.reset(seed=55)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = 7, 7

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        pre_tick_event = next(item for item in info["event_log"] if item["stage"] == "pre_tick")
        payload = pre_tick_event["payload"]
        self.assertEqual(payload["spider_pos"], [3, 3])
        self.assertEqual(payload["lizard_pos"], [7, 7])

    def test_event_log_action_stage_records_movement(self) -> None:
        world = SpiderWorld(seed=57, lizard_move_interval=999999)
        world.reset(seed=57)

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        action_events = [item for item in info["event_log"] if item["stage"] == "action"]
        self.assertTrue(len(action_events) >= 2)
        names = [e["name"] for e in action_events]
        self.assertIn("action_resolved", names)
        self.assertIn("movement_applied", names)

    def test_event_log_action_resolved_has_action_fields(self) -> None:
        world = SpiderWorld(seed=59, lizard_move_interval=999999)
        world.reset(seed=59)

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        action_resolved = next(
            item for item in info["event_log"]
            if item["stage"] == "action" and item["name"] == "action_resolved"
        )
        payload = action_resolved["payload"]
        self.assertIn("action_index", payload)
        self.assertIn("intended_action", payload)
        self.assertIn("executed_action", payload)
        self.assertIn("motor_noise_applied", payload)
        self.assertEqual(payload["intended_action"], "STAY")
        self.assertEqual(payload["executed_action"], "STAY")

    def test_event_log_terrain_stage_has_required_fields(self) -> None:
        world = SpiderWorld(seed=61, lizard_move_interval=999999)
        world.reset(seed=61)

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        terrain_event = next(
            item for item in info["event_log"]
            if item["stage"] == "terrain_and_wakefulness"
        )
        payload = terrain_event["payload"]
        self.assertIn("terrain", payload)
        self.assertIn("predator_threat", payload)
        self.assertIn("interrupted_rest", payload)
        self.assertIn("exposed_at_night", payload)
        self.assertIn("sleep_debt", payload)
        self.assertIn("fatigue", payload)

    def test_event_log_predator_contact_stage_no_contact(self) -> None:
        world = SpiderWorld(seed=63, lizard_move_interval=999999)
        world.reset(seed=63)
        # Move lizard far away
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        predator_events = [item for item in info["event_log"] if item["stage"] == "predator_contact"]
        self.assertTrue(len(predator_events) >= 1)
        # When no contact, should have a contact_check event
        contact_check = next(
            (item for item in predator_events if item["name"] == "contact_check"), None
        )
        self.assertIsNotNone(contact_check)
        self.assertFalse(contact_check["payload"]["predator_contact"])

    def test_event_log_postprocess_has_reward_and_done(self) -> None:
        world = SpiderWorld(seed=65, lizard_move_interval=999999)
        world.reset(seed=65)

        _, reward, done, info = world.step(ACTION_TO_INDEX["STAY"])

        postprocess_event = next(
            item for item in info["event_log"] if item["stage"] == "postprocess"
        )
        payload = postprocess_event["payload"]
        self.assertIn("reward", payload)
        self.assertIn("done", payload)
        self.assertIn("health", payload)
        self.assertIn("tick", payload)
        self.assertAlmostEqual(payload["reward"], round(reward, 6), places=5)
        self.assertEqual(done, payload["done"])

    def test_event_log_reward_stage_has_distance_deltas(self) -> None:
        world = SpiderWorld(seed=67, lizard_move_interval=999999)
        world.reset(seed=67)

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        reward_events = [item for item in info["event_log"] if item["stage"] == "reward"]
        reward_names = [e["name"] for e in reward_events]
        self.assertIn("distance_deltas", reward_names)
        delta_event = next(e for e in reward_events if e["name"] == "distance_deltas")
        self.assertIn("food", delta_event["payload"])
        self.assertIn("shelter", delta_event["payload"])
        self.assertIn("predator", delta_event["payload"])

    def test_event_log_memory_stage_has_memory_targets(self) -> None:
        world = SpiderWorld(seed=69, lizard_move_interval=999999)
        world.reset(seed=69)

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        memory_event = next(item for item in info["event_log"] if item["stage"] == "memory")
        payload = memory_event["payload"]
        self.assertIn("predator_escape", payload)
        self.assertIn("food_memory_target", payload)
        self.assertIn("predator_memory_target", payload)
        self.assertIn("shelter_memory_target", payload)
        self.assertIn("escape_memory_target", payload)

    def test_event_log_is_independent_between_steps(self) -> None:
        world = SpiderWorld(seed=71, lizard_move_interval=999999)
        world.reset(seed=71)

        _, _, _, info1 = world.step(ACTION_TO_INDEX["STAY"])
        _, _, _, info2 = world.step(ACTION_TO_INDEX["STAY"])

        # Each step should have its own event_log
        self.assertIsNot(info1["event_log"], info2["event_log"])
        # Pre-tick tick field should differ between steps
        pre_tick1 = next(item for item in info1["event_log"] if item["stage"] == "pre_tick")
        pre_tick2 = next(item for item in info2["event_log"] if item["stage"] == "pre_tick")
        self.assertNotEqual(pre_tick1["payload"]["tick"], pre_tick2["payload"]["tick"])

    def test_event_log_predator_contact_event_recorded_when_contact_occurs(self) -> None:
        world = SpiderWorld(seed=73, lizard_move_interval=999999)
        world.reset(seed=73)
        # Place spider and lizard at the same position to force contact
        world.state.x = world.lizard.x
        world.state.y = world.lizard.y

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        predator_contact_events = [
            item for item in info["event_log"]
            if item["stage"] == "predator_contact" and item["name"] == "predator_contact"
        ]
        self.assertTrue(
            info.get("predator_contact") and len(predator_contact_events) >= 1,
            "Expected predator contact when spider and lizard share position",
        )

    def test_event_log_finalize_tick_complete_matches_world_tick(self) -> None:
        world = SpiderWorld(seed=75, lizard_move_interval=999999)
        world.reset(seed=75)

        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])

        finalize_event = next(item for item in info["event_log"] if item["stage"] == "finalize")
        self.assertEqual(finalize_event["name"], "tick_complete")
        self.assertEqual(finalize_event["payload"]["tick"], world.tick)

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
        world = SpiderWorld(seed=31, lizard_move_interval=999999)
        world.reset(seed=31)
        world.state.x, world.state.y = 2, 2
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
        world = SpiderWorld(seed=67, lizard_move_interval=999999)
        world.reset(seed=67)
        world.state.x, world.state.y = 3, 3
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
        world = SpiderWorld(seed=71, lizard_move_interval=999999)
        world.reset(seed=71)
        world.state.x, world.state.y = 2, 2
        world.state.hunger = 0.92
        world.food_positions = [(4, 2)]
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

    def _reachable(self, world: SpiderWorld, start: tuple[int, int], goal: tuple[int, int]) -> bool:
        """
        Determine whether the goal cell can be reached from the start cell within the world's walkable area.
        
        Parameters:
        	world (SpiderWorld): The world providing dimensions, walkability checks, and movement deltas.
        	start (tuple[int, int]): Starting (x, y) coordinates.
        	goal (tuple[int, int]): Target (x, y) coordinates.
        
        Returns:
        	`true` if the goal cell can be reached from start by traversing walkable cells, `false` otherwise.
        """
        queue = deque([start])
        seen = {start}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                return True
            for dx, dy in world.move_deltas:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nxt[0] < world.width and 0 <= nxt[1] < world.height):
                    continue
                if nxt in seen or not world.is_walkable(nxt):
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return False


class NoiseChannelIsolationTest(unittest.TestCase):
    """Tests for the _reset_rngs channel-splitting and episode_seed tracking added in this PR."""

    def test_episode_seed_stored_after_reset_with_explicit_seed(self) -> None:
        world = SpiderWorld(seed=11, lizard_move_interval=999999)
        world.reset(seed=999)
        self.assertEqual(world.episode_seed, 999)

    def test_episode_seed_stored_after_reset_with_none_seed(self) -> None:
        world = SpiderWorld(seed=17, lizard_move_interval=999999)
        world.reset(seed=None)
        # When no seed is given, reset falls back to self.seed.
        self.assertEqual(world.episode_seed, world.seed)

    def test_rng_is_alias_for_predator_rng(self) -> None:
        world = SpiderWorld(seed=23, lizard_move_interval=999999)
        world.reset(seed=23)
        self.assertIs(world.rng, world.predator_rng)

    def test_five_independent_rng_channels_exist(self) -> None:
        world = SpiderWorld(seed=29, lizard_move_interval=999999)
        world.reset(seed=29)
        channels = [world.spawn_rng, world.predator_rng, world.visual_rng, world.olfactory_rng, world.motor_rng]
        # Each channel should be a distinct object.
        ids = [id(ch) for ch in channels]
        self.assertEqual(len(ids), len(set(ids)))

    def test_consuming_spawn_rng_does_not_perturb_other_channels(self) -> None:
        world_a = SpiderWorld(seed=31, lizard_move_interval=999999)
        world_b = SpiderWorld(seed=31, lizard_move_interval=999999)
        world_a.reset(seed=31)
        world_b.reset(seed=31)

        for _ in range(3):
            world_a.spawn_rng.random()

        self.assertAlmostEqual(float(world_a.predator_rng.random()), float(world_b.predator_rng.random()))
        self.assertAlmostEqual(float(world_a.visual_rng.random()), float(world_b.visual_rng.random()))
        self.assertAlmostEqual(float(world_a.olfactory_rng.random()), float(world_b.olfactory_rng.random()))
        self.assertAlmostEqual(float(world_a.motor_rng.random()), float(world_b.motor_rng.random()))

    def test_reset_with_different_seed_changes_episode_seed(self) -> None:
        world = SpiderWorld(seed=37, lizard_move_interval=999999)
        world.reset(seed=100)
        self.assertEqual(world.episode_seed, 100)
        world.reset(seed=200)
        self.assertEqual(world.episode_seed, 200)

    def test_reset_with_different_seed_changes_spawn_rng_output(self) -> None:
        world = SpiderWorld(seed=41, lizard_move_interval=999999)
        world.reset(seed=10)
        val_a = float(world.spawn_rng.random())
        world.reset(seed=20)
        val_b = float(world.spawn_rng.random())
        self.assertNotAlmostEqual(val_a, val_b)

    def test_state_meta_includes_noise_profile_name(self) -> None:
        world = SpiderWorld(seed=53, noise_profile="medium", lizard_move_interval=999999)
        world.reset(seed=53)
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        state_meta = info["state"]
        self.assertIn("noise_profile", state_meta)
        self.assertEqual(state_meta["noise_profile"], "medium")

    def test_state_meta_includes_episode_seed(self) -> None:
        world = SpiderWorld(seed=59, lizard_move_interval=999999)
        world.reset(seed=777)
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        state_meta = info["state"]
        self.assertIn("episode_seed", state_meta)
        self.assertEqual(state_meta["episode_seed"], 777)


class MotorNoiseTest(unittest.TestCase):
    """Tests for SpiderWorld._apply_motor_noise added in this PR."""

    def _flip_all_profile(self) -> NoiseConfig:
        return NoiseConfig(
            name="always_flip",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={"action_flip_prob": 1.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )

    def _no_flip_profile(self) -> NoiseConfig:
        return NoiseConfig(
            name="never_flip",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={"action_flip_prob": 0.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )

    def test_motor_noise_zero_never_changes_action(self) -> None:
        world = SpiderWorld(seed=61, noise_profile=self._no_flip_profile(), lizard_move_interval=999999)
        world.reset(seed=61)
        for _ in range(10):
            _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
            self.assertEqual(info["intended_action"], "STAY")
            self.assertEqual(info["executed_action"], "STAY")
            self.assertFalse(info["motor_noise_applied"])

    def test_motor_noise_one_always_changes_action(self) -> None:
        world = SpiderWorld(seed=67, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=67)
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertNotEqual(info["executed_action"], "STAY")
        self.assertTrue(info["motor_noise_applied"])

    def test_motor_noise_executed_is_never_same_as_intended_when_flip_prob_is_one(self) -> None:
        world = SpiderWorld(seed=71, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=71)
        for action_name in ["STAY", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]:
            world.reset(seed=71)
            _, _, _, info = world.step(ACTION_TO_INDEX[action_name])
            self.assertNotEqual(
                info["executed_action"],
                action_name,
                f"Expected noise to change {action_name!r}, but got same action",
            )

    def test_motor_noise_info_action_equals_executed_action(self) -> None:
        world = SpiderWorld(seed=73, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=73)
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertEqual(info["action"], info["executed_action"])

    def test_motor_noise_none_profile_motor_noise_applied_false(self) -> None:
        world = SpiderWorld(seed=79, noise_profile="none", lizard_move_interval=999999)
        world.reset(seed=79)
        _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_UP"])
        self.assertFalse(info["motor_noise_applied"])
        self.assertEqual(info["intended_action"], info["executed_action"])

    def test_motor_noise_reproducible_for_same_seed(self) -> None:
        world_a = SpiderWorld(seed=83, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world_b = SpiderWorld(seed=83, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world_a.reset(seed=83)
        world_b.reset(seed=83)

        _, _, _, info_a = world_a.step(ACTION_TO_INDEX["MOVE_UP"])
        _, _, _, info_b = world_b.step(ACTION_TO_INDEX["MOVE_UP"])

        self.assertEqual(info_a["executed_action"], info_b["executed_action"])


class PredatorDeterministicTiebreakingTest(unittest.TestCase):
    """Tests that predator movement is deterministic with noise_profile='none'."""

    def test_predator_moves_deterministically_with_none_noise(self) -> None:
        world_a = SpiderWorld(seed=89, noise_profile="none", lizard_move_interval=1)
        world_b = SpiderWorld(seed=97, noise_profile="none", lizard_move_interval=1)
        world_a.reset(seed=89)
        world_b.reset(seed=97)
        for world in (world_a, world_b):
            world.lizard.x, world.lizard.y = 0, 0

        for _ in range(5):
            world_a.predator_rng.random()

        moved_a = world_a.predator_controller._step_towards(world_a, (1, 1))
        moved_b = world_b.predator_controller._step_towards(world_b, (1, 1))

        self.assertTrue(moved_a)
        self.assertTrue(moved_b)
        self.assertEqual(world_a.lizard_pos(), world_b.lizard_pos())

    def test_predator_same_seed_gives_same_patrol_path(self) -> None:
        world_a = SpiderWorld(seed=97, noise_profile="none", lizard_move_interval=1)
        world_b = SpiderWorld(seed=101, noise_profile="none", lizard_move_interval=1)
        world_a.reset(seed=97)
        world_b.reset(seed=101)
        for world in (world_a, world_b):
            world.state.x, world.state.y = world.width // 2, world.height // 2
            world.lizard.x, world.lizard.y = 0, 0

        for _ in range(7):
            world_a.predator_rng.random()

        target_a = world_a.predator_controller._pick_patrol_target(world_a)
        target_b = world_b.predator_controller._pick_patrol_target(world_b)

        self.assertIsNotNone(target_a)
        self.assertEqual(target_a, target_b)


if __name__ == "__main__":
    unittest.main()
