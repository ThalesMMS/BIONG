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
    ActionContextObservation,
    MotorContextObservation,
)
from spider_cortex_sim.maps import CLUTTER, NARROW, OPEN, MAP_TEMPLATE_NAMES, build_map_template
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.predator import (
    OLFACTORY_HUNTER_PROFILE,
    VISUAL_HUNTER_PROFILE,
    LizardState,
)
from spider_cortex_sim.world_types import PerceptTrace
from spider_cortex_sim.world import (
    ACTION_TO_INDEX,
    MOVE_DELTAS,
    REWARD_COMPONENT_NAMES,
    PerceptualBuffer,
    SpiderWorld,
    _copy_observation_payload,
    _is_temporal_direction_field,
    compute_execution_difficulty,
)


def _terrain_with_cleanup(
    test_case: unittest.TestCase,
    world: SpiderWorld,
) -> tuple[dict[tuple[int, int], str], tuple[int, int]]:
    """
    Save terrain at the spider position and register cleanup to restore it.
    Returns the terrain dict and saved position tuple.
    """
    pos = world.spider_pos()
    terrain = world.map_template.terrain
    original_present = pos in terrain
    original_terrain = terrain.get(pos)

    def restore_terrain() -> None:
        if original_present:
            terrain[pos] = original_terrain
        else:
            terrain.pop(pos, None)

    test_case.addCleanup(restore_terrain)
    return terrain, pos


def _profile_with_perception(**perception_overrides: float) -> OperationalProfile:
    """
    Return the default OperationalProfile with validated perception overrides.
    Raises ValueError for unknown perception keys.
    """
    summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
    perception = summary["perception"]
    allowed_keys = set(perception.keys())
    unknown_keys = sorted(set(perception_overrides) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown perception override keys: {unknown_keys}")
    perception.update(perception_overrides)
    return OperationalProfile.from_summary(summary)


def _compute_slip_and_difficulty(
    world: SpiderWorld,
    terrain: dict[tuple[int, int], str],
    pos: tuple[int, int],
    terrain_type: str,
    heading: tuple[float, float],
    fatigue: float,
) -> dict[str, object]:
    """
    Configure a MOVE_RIGHT motor-noise case and return its diagnostics.
    Mutates terrain, heading, and fatigue on the provided world.
    """
    terrain[pos] = terrain_type
    world.state.heading_dx = float(heading[0])
    world.state.heading_dy = float(heading[1])
    world.state.fatigue = float(fatigue)
    return world._apply_motor_noise("MOVE_RIGHT")


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
        Return a lizard-walkable cell next to the entrance.
        Falls back to the first lizard spawn cell if none is adjacent.
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
        Move the lizard to the spawn cell farthest from the spider.
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
        Return a deterministic reflex brain with learned weights neutralized.
        Uses the supplied seed for reproducible initialization.
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
        self.assertEqual(obs["visual"].shape, (25,))
        self.assertEqual(obs["sensory"].shape, (12,))
        self.assertEqual(obs["hunger"].shape, (16,))
        self.assertEqual(obs["sleep"].shape, (16,))
        self.assertEqual(obs["alert"].shape, (25,))
        self.assertEqual(obs["action_context"].shape, (15,))
        self.assertEqual(obs["motor_context"].shape, (13,))
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

    def test_advance_percept_trace_refreshes_peripheral_target(self) -> None:
        from spider_cortex_sim.perception import PerceivedTarget, _compute_target_visibility_zone
        from spider_cortex_sim.world_types import PerceptTrace

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

        updated = world._advance_percept_trace(
            PerceptTrace(target=None, age=0, certainty=0.0),
            peripheral_percept,
            [(6, 7)],
        )

        self.assertEqual(updated.target, (6, 7))
        self.assertEqual(updated.age, 0)
        self.assertAlmostEqual(updated.certainty, 0.7)

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

    def test_event_log_records_pipeline_stages_in_order(self) -> None:
        """
        Verifies the event log contains pipeline stages in the expected execution order.
        
        Asserts that the first occurrence of each major pipeline stage appears in the sequence:
        pre_tick -> action -> terrain_and_wakefulness -> predator_contact -> autonomic ->
        predator_update -> reward -> postprocess -> memory -> finalize.
        """
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

    def test_step_uses_live_stages_module_build_tick_context(self) -> None:
        world = SpiderWorld(seed=50, lizard_move_interval=999999)
        world.reset(seed=50)

        class SentinelError(RuntimeError):
            pass

        with patch.object(
            tick_stages,
            "build_tick_context",
            side_effect=SentinelError("patched build_tick_context should be used"),
        ):
            with self.assertRaises(SentinelError):
                world.step(ACTION_TO_INDEX["STAY"])

    def test_step_adds_stage_name_to_stage_failures(self) -> None:
        world = SpiderWorld(seed=50, lizard_move_interval=999999)
        world.reset(seed=50)

        class SentinelError(RuntimeError):
            pass

        def fail_stage(world: SpiderWorld, context: object) -> None:
            raise SentinelError("boom")

        with patch.object(
            tick_stages,
            "TICK_STAGES",
            (tick_stages.StageDescriptor(name="sentinel", run=fail_stage, mutates=("context.event_log",)),),
        ):
            with self.assertRaises(SentinelError) as ctx:
                world.step(ACTION_TO_INDEX["STAY"])

        self.assertIn("Tick stage 'sentinel' failed.", getattr(ctx.exception, "__notes__", []))

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
        self.assertIn("slip_reason", payload)
        self.assertIn("slip_probability", payload)
        self.assertIn("execution_difficulty", payload)
        self.assertIn("orientation_alignment", payload)
        self.assertIn("terrain_difficulty", payload)
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

    def test_six_independent_rng_channels_exist(self) -> None:
        world = SpiderWorld(seed=29, lizard_move_interval=999999)
        world.reset(seed=29)
        channels = [
            world.spawn_rng,
            world.predator_rng,
            world.visual_rng,
            world.olfactory_rng,
            world.motor_rng,
            world.delay_rng,
        ]
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
        self.assertAlmostEqual(float(world_a.delay_rng.random()), float(world_b.delay_rng.random()))

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


class PerceptualBufferTest(unittest.TestCase):
    def test_buffer_returns_requested_delayed_observation(self) -> None:
        buffer = PerceptualBuffer(max_delay=2)
        buffer.push(0, {"visual": np.array([1.0]), "meta": {"tick": 0}})
        buffer.push(1, {"visual": np.array([2.0]), "meta": {"tick": 1}})
        buffer.push(2, {"visual": np.array([3.0]), "meta": {"tick": 2}})

        observation, effective_delay = buffer.get(2)

        self.assertEqual(effective_delay, 2)
        np.testing.assert_allclose(observation["visual"], np.array([1.0]))
        self.assertEqual(observation["meta"]["tick"], 0)

    def test_buffer_replaces_same_tick_entry(self) -> None:
        buffer = PerceptualBuffer(max_delay=1)
        buffer.push(0, {"visual": np.array([1.0]), "meta": {"tick": 0}})
        buffer.push(0, {"visual": np.array([4.0]), "meta": {"tick": 0}})

        observation, effective_delay = buffer.get(1)

        self.assertEqual(effective_delay, 0)
        np.testing.assert_allclose(observation["visual"], np.array([4.0]))

    def test_buffer_stores_defensive_copies(self) -> None:
        vector = np.array([1.0])
        payload = {"visual": vector, "meta": {"tick": 0}}
        buffer = PerceptualBuffer(max_delay=1)
        buffer.push(0, payload)
        vector[0] = 99.0
        payload["meta"]["tick"] = 99

        observation, _ = buffer.get(0)

        np.testing.assert_allclose(observation["visual"], np.array([1.0]))
        self.assertEqual(observation["meta"]["tick"], 0)


class PerceptualDelayObservationTest(unittest.TestCase):
    def _no_delay_noise_profile(self, *, decay: float = 0.0, jitter: float = 0.0) -> NoiseConfig:
        """Return a delay_test NoiseConfig with only delayed-percept certainty_decay_per_tick and direction_jitter_per_tick enabled."""
        return NoiseConfig(
            name="delay_test",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={"action_flip_prob": 0.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
            delay={
                "certainty_decay_per_tick": decay,
                "direction_jitter_per_tick": jitter,
            },
        )

    def _world(self, *, delay_ticks: float = 1.0, delay_noise: float = 0.0, noise: NoiseConfig | None = None) -> SpiderWorld:
        """
        Return a test world with configured perceptual delay.
        Uses the supplied NoiseConfig or the no-delay profile helper.
        """
        profile = _profile_with_perception(
            perceptual_delay_ticks=delay_ticks,
            perceptual_delay_noise=delay_noise,
        )
        return SpiderWorld(
            seed=101,
            lizard_move_interval=999999,
            operational_profile=profile,
            noise_profile=noise if noise is not None else self._no_delay_noise_profile(),
        )

    def _place_visible_predator(self, world: SpiderWorld) -> None:
        """
        Place the spider and lizard so the predator is visible.
        Mutates position and heading on the supplied world.
        """
        world.state.x = 5
        world.state.y = 7
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        world.lizard.x = 7
        world.lizard.y = 7
        world.lizard.mode = "PATROL"

    def test_step_returns_previous_tick_observation_when_delay_is_one(self) -> None:
        world = self._world(delay_ticks=1.0, delay_noise=0.0)
        world.reset(seed=101)
        self._place_visible_predator(world)
        current_obs = world.observe()
        current_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(current_obs["visual"])
        self.assertGreater(current_visual["predator_visible"], 0.5)

        world.lizard.x = 5
        world.lizard.y = 0
        delayed_obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])

        delayed_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(delayed_obs["visual"])
        raw_current = world._raw_observation()
        raw_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(raw_current["visual"])
        self.assertGreater(delayed_visual["predator_visible"], 0.5)
        self.assertEqual(tuple(delayed_obs["meta"]["vision"]["predator"]["position"]), (7, 7))
        self.assertEqual(raw_visual["predator_visible"], 0.0)

    def test_delay_zero_returns_current_observation(self) -> None:
        world = self._world(delay_ticks=0.0, delay_noise=0.0)
        world.reset(seed=103)
        self._place_visible_predator(world)
        world.observe()
        world.lizard.x = 5
        world.lizard.y = 0

        observation, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])

        visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(observation["visual"])
        self.assertEqual(visual["predator_visible"], 0.0)
        self.assertNotIn("perceptual_delay", observation["meta"])

    def test_delayed_certainty_decays_by_delay_noise_config(self) -> None:
        noise = self._no_delay_noise_profile(decay=0.25, jitter=0.0)
        world = self._world(delay_ticks=1.0, delay_noise=1.0, noise=noise)
        world.reset(seed=107)
        self._place_visible_predator(world)
        current_obs = world.observe()
        current_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(current_obs["visual"])

        world.lizard.x = 5
        world.lizard.y = 0
        delayed_obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])

        delayed_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(delayed_obs["visual"])
        self.assertAlmostEqual(
            delayed_visual["predator_certainty"],
            current_visual["predator_certainty"] * 0.75,
        )
        self.assertEqual(delayed_obs["meta"]["perceptual_delay"]["effective_ticks"], 1)

    def test_delayed_certainty_decay_updates_visibility_flags(self) -> None:
        noise = self._no_delay_noise_profile(decay=1.0, jitter=0.0)
        world = self._world(delay_ticks=1.0, delay_noise=1.0, noise=noise)
        world.reset(seed=108)
        self._place_visible_predator(world)
        current_obs = world.observe()
        current_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(current_obs["visual"])
        self.assertGreater(current_visual["predator_visible"], 0.5)
        self.assertTrue(current_obs["meta"]["predator_visible"])

        world.lizard.x = 5
        world.lizard.y = 0
        delayed_obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])

        for key in ("visual", "alert", "action_context", "motor_context"):
            with self.subTest(view=key):
                values = OBSERVATION_INTERFACE_BY_KEY[key].bind_values(delayed_obs[key])
                self.assertAlmostEqual(values["predator_certainty"], 0.0)
                self.assertAlmostEqual(values["predator_visible"], 0.0)
                if "predator_dx" in values:
                    self.assertAlmostEqual(values["predator_dx"], 0.0)
                if "predator_dy" in values:
                    self.assertAlmostEqual(values["predator_dy"], 0.0)
        predator_meta = delayed_obs["meta"]["vision"]["predator"]
        self.assertAlmostEqual(predator_meta["certainty"], 0.0)
        self.assertAlmostEqual(predator_meta["visible"], 0.0)
        self.assertAlmostEqual(predator_meta["dx"], 0.0)
        self.assertAlmostEqual(predator_meta["dy"], 0.0)
        self.assertFalse(delayed_obs["meta"]["predator_visible"])

    def test_delay_jitter_does_not_create_zero_strength_directions(self) -> None:
        noise = self._no_delay_noise_profile(decay=0.0, jitter=1.0)
        world = self._world(delay_ticks=1.0, delay_noise=1.0, noise=noise)
        world.reset(seed=108)
        hunger_iface = OBSERVATION_INTERFACE_BY_KEY["hunger"]
        hunger_mapping = {name: 0.0 for name in hunger_iface.signal_names}
        hunger_mapping.update(
            {
                "food_smell_dx": 1.0,
                "food_smell_dy": -1.0,
                "food_smell_strength": 0.0,
                "food_trace_dx": 1.0,
                "food_trace_dy": -1.0,
                "food_trace_strength": 0.0,
            }
        )

        delayed = world._apply_temporal_noise_to_vector(
            "hunger",
            hunger_iface.vector_from_mapping(hunger_mapping),
            decay_factor=1.0,
            direction_jitter=1.0,
        )
        values = hunger_iface.bind_values(delayed)

        self.assertAlmostEqual(values["food_smell_dx"], 0.0)
        self.assertAlmostEqual(values["food_smell_dy"], 0.0)
        self.assertAlmostEqual(values["food_trace_dx"], 0.0)
        self.assertAlmostEqual(values["food_trace_dy"], 0.0)

    def test_delay_jitter_does_not_create_zero_strength_meta_directions(self) -> None:
        noise = self._no_delay_noise_profile(decay=0.0, jitter=1.0)
        world = self._world(delay_ticks=1.0, delay_noise=1.0, noise=noise)
        world.reset(seed=108)
        meta = {
            "vision": {},
            "percept_traces": {
                "food": {
                    "certainty": 1.0,
                    "strength": 0.0,
                    "dx": 1.0,
                    "dy": -1.0,
                }
            },
        }

        world._apply_temporal_noise_to_meta(
            meta,
            configured_delay=1,
            effective_delay=1,
            decay_factor=1.0,
            direction_jitter=1.0,
        )

        self.assertAlmostEqual(meta["percept_traces"]["food"]["dx"], 0.0)
        self.assertAlmostEqual(meta["percept_traces"]["food"]["dy"], 0.0)

    def test_trace_signals_are_read_from_delayed_percept_state(self) -> None:
        """
        Verify that when perceptual delay is enabled, the observation's hunger `food_trace_strength`
        reflects the delayed perceptual state rather than the current raw state.
        
        This test sets a nonzero food trace in the world's state, captures the current observation,
        then clears the immediate trace and advances one tick with a perceptual delay of 1.0.
        It asserts that:
        - the current observation shows the original trace strength,
        - the delayed observation returned by `step` also shows that trace strength,
        - the raw (undelayed) observation after the step shows the cleared trace strength.
        """
        world = self._world(delay_ticks=1.0, delay_noise=0.0)
        world.reset(seed=109)
        world.food_positions = []
        world.state.x = 5
        world.state.y = 7
        world.state.food_trace = PerceptTrace(target=(8, 7), age=0, certainty=0.8)
        current_obs = world.observe()
        current_hunger = OBSERVATION_INTERFACE_BY_KEY["hunger"].bind_values(current_obs["hunger"])

        world.state.food_trace = PerceptTrace(target=None, age=0, certainty=0.0)
        delayed_obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])

        delayed_hunger = OBSERVATION_INTERFACE_BY_KEY["hunger"].bind_values(delayed_obs["hunger"])
        raw_hunger = OBSERVATION_INTERFACE_BY_KEY["hunger"].bind_values(
            world._raw_observation()["hunger"]
        )
        self.assertAlmostEqual(current_hunger["food_trace_strength"], 0.8)
        self.assertAlmostEqual(delayed_hunger["food_trace_strength"], 0.8)
        self.assertAlmostEqual(raw_hunger["food_trace_strength"], 0.0)

    def test_predator_dist_absent_from_non_meta_observation_keys(self) -> None:
        """
        Asserts that predator distance signals are present only in the observation meta diagnostics and not in any non-meta observation keys.
        
        Creates a world with no perceptual delay or delay noise, calls observe(), and verifies:
        - No top-level observation key other than "meta" contains "predator_dist" in its name.
        - The corresponding observation interface signal names for each non-meta view do not include "predator_dist" or "diagnostic_predator_dist".
        - The only occurrence of "diagnostic_predator_dist" in the nested observation dict is at ("meta", "diagnostic", "diagnostic_predator_dist").
        """
        world = self._world(delay_ticks=0.0, delay_noise=0.0)
        world.reset(seed=111)

        observation = world.observe()

        for key in observation:
            if key == "meta":
                continue
            self.assertNotIn("predator_dist", key)
            view_key = "motor_context" if key == "motor_extra" else key
            signal_names = OBSERVATION_INTERFACE_BY_KEY[view_key].signal_names
            self.assertNotIn("predator_dist", signal_names)
            self.assertNotIn("diagnostic_predator_dist", signal_names)

        diagnostic_paths: list[tuple[str, ...]] = []

        def collect_diagnostic_paths(value: object, path: tuple[str, ...]) -> None:
            """Record all nested diagnostic_predator_dist key paths."""
            if isinstance(value, Mapping):
                for child_key, child_value in value.items():
                    child_path = (*path, str(child_key))
                    if child_key == "diagnostic_predator_dist":
                        diagnostic_paths.append(child_path)
                    collect_diagnostic_paths(child_value, child_path)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for index, child_value in enumerate(value):
                    collect_diagnostic_paths(child_value, (*path, str(index)))

        collect_diagnostic_paths(observation, ())
        self.assertEqual(
            diagnostic_paths,
            [("meta", "diagnostic", "diagnostic_predator_dist")],
        )


class ExecutionDifficultyTest(unittest.TestCase):
    def test_open_aligned_movement_has_zero_difficulty(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(1.0, 0.0),
            terrain=OPEN,
            fatigue=0.0,
        )
        self.assertAlmostEqual(difficulty, 0.0)
        self.assertAlmostEqual(components["orientation_alignment"], 1.0)
        self.assertAlmostEqual(components["terrain_difficulty"], 0.0)

    def test_opposed_narrow_fatigued_movement_is_hard(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(-1.0, 0.0),
            terrain=NARROW,
            fatigue=1.0,
        )
        self.assertAlmostEqual(components["orientation_alignment"], 0.0)
        self.assertAlmostEqual(components["terrain_difficulty"], 0.7)
        self.assertAlmostEqual(components["raw_difficulty"], 1.4)
        self.assertAlmostEqual(difficulty, 1.0)

    def test_stay_has_no_orientation_mismatch(self) -> None:
        difficulty, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(0.0, 0.0),
            terrain=CLUTTER,
            fatigue=1.0,
        )
        self.assertAlmostEqual(components["orientation_alignment"], 1.0)
        self.assertAlmostEqual(difficulty, 0.0)


class MotorNoiseTest(unittest.TestCase):
    """Tests for embodiment-aware motor slip."""

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
        _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_UP"])
        self.assertNotEqual(info["executed_action"], "MOVE_UP")
        self.assertTrue(info["motor_noise_applied"])

    def test_motor_noise_does_not_create_motion_from_stay(self) -> None:
        world = SpiderWorld(seed=71, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=71)
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertEqual(info["executed_action"], "STAY")
        self.assertFalse(info["motor_noise_applied"])

    def test_orient_action_updates_heading_without_moving(self) -> None:
        world = SpiderWorld(seed=72, noise_profile=self._no_flip_profile(), lizard_move_interval=999999)
        world.reset(seed=72)
        start_pos = world.spider_pos()
        world.state.heading_dx = 1
        world.state.heading_dy = 0

        _, _, _, info = world.step(ACTION_TO_INDEX["ORIENT_LEFT"])

        self.assertEqual(world.spider_pos(), start_pos)
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), (-1, 0))
        self.assertEqual((world.state.last_move_dx, world.state.last_move_dy), (0, 0))
        self.assertEqual(info["executed_action"], "ORIENT_LEFT")

    def test_orient_action_does_not_trigger_motor_slip(self) -> None:
        """
        Verifies that an ORIENT action updates the spider's heading without changing position and does not apply motor slip.
        
        Asserts the spider's position remains unchanged, the executed action equals the orient action, `motor_noise_applied` is false, `motor_slip["occurred"]` is false, `motor_slip["slip_probability"]` is 0.0, and the world heading is updated to the expected orient vector.
        """
        world = SpiderWorld(seed=74, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=74)
        start_pos = world.spider_pos()

        _, _, _, info = world.step(ACTION_TO_INDEX["ORIENT_UP"])

        self.assertEqual(world.spider_pos(), start_pos)
        self.assertEqual(info["executed_action"], "ORIENT_UP")
        self.assertFalse(info["motor_noise_applied"])
        self.assertFalse(info["motor_slip"]["occurred"])
        self.assertAlmostEqual(info["motor_slip"]["slip_probability"], 0.0)
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), (0, -1))

    def test_orient_action_advances_idle_physiology(self) -> None:
        world = SpiderWorld(seed=76, noise_profile=self._no_flip_profile(), lizard_move_interval=999999)
        world.reset(seed=76)
        hunger_before = world.state.hunger
        fatigue_before = world.state.fatigue

        world.step(ACTION_TO_INDEX["ORIENT_RIGHT"])

        self.assertGreater(world.state.hunger, hunger_before)
        self.assertGreater(world.state.fatigue, fatigue_before)
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), (1, 0))

    def test_motor_noise_executed_is_never_same_as_intended_when_flip_prob_is_one_for_movement(self) -> None:
        world = SpiderWorld(seed=71, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=71)
        for action_name in ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"]:
            world.reset(seed=71)
            _, _, _, info = world.step(ACTION_TO_INDEX[action_name])
            self.assertNotEqual(
                info["executed_action"],
                action_name,
                f"Expected noise to change {action_name!r}, but got same action",
            )

    def test_motor_noise_info_action_remains_intended_action(self) -> None:
        world = SpiderWorld(seed=73, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=73)
        _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_UP"])
        self.assertEqual(info["action"], info["intended_action"])
        self.assertEqual(info["action"], "MOVE_UP")
        self.assertNotEqual(info["executed_action"], info["action"])

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

    def test_motor_slip_info_contains_execution_diagnostics(self) -> None:
        world = SpiderWorld(seed=87, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=87)
        _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_UP"])
        slip = info["motor_slip"]
        self.assertTrue(slip["occurred"])
        self.assertEqual(slip["original_action"], "MOVE_UP")
        self.assertEqual(slip["executed_action"], info["executed_action"])
        self.assertIn(slip["reason"], {"base", "terrain", "orientation", "fatigue"})
        self.assertIn("orientation_alignment", slip["components"])
        self.assertIn("terrain_difficulty", slip["components"])
        self.assertIn("fatigue_factor", slip["components"])
        self.assertIn("motor_execution_difficulty", info)

    def test_slip_uses_stay_or_adjacent_action_not_reverse(self) -> None:
        world = SpiderWorld(seed=91, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=91)
        _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_UP"])
        self.assertIn(info["executed_action"], {"STAY", "MOVE_LEFT", "MOVE_RIGHT"})
        self.assertNotEqual(info["executed_action"], "MOVE_DOWN")

    def test_slip_probability_increases_with_execution_difficulty(self) -> None:
        profile = NoiseConfig(
            name="difficulty_slip_test",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={
                "action_flip_prob": 0.0,
                "orientation_slip_factor": 0.2,
                "terrain_slip_factor": 0.4,
                "fatigue_slip_factor": 0.2,
            },
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )
        world = SpiderWorld(seed=97, noise_profile=profile, lizard_move_interval=999999)
        world.reset(seed=97)
        terrain, pos = _terrain_with_cleanup(self, world)

        easy = _compute_slip_and_difficulty(
            world,
            terrain,
            pos,
            OPEN,
            heading=(1.0, 0.0),
            fatigue=0.0,
        )
        hard = _compute_slip_and_difficulty(
            world,
            terrain,
            pos,
            NARROW,
            heading=(-1.0, 0.0),
            fatigue=1.0,
        )

        self.assertLess(easy["slip_probability"], hard["slip_probability"])
        self.assertLess(easy["execution_difficulty"], hard["execution_difficulty"])


class MotorExecutionSlipMechanismTest(unittest.TestCase):
    def _slip_profile(self, *, base: float = 0.0) -> NoiseConfig:
        return NoiseConfig(
            name="action_center_slip_test",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={
                "action_flip_prob": base,
                "orientation_slip_factor": 0.2,
                "terrain_slip_factor": 0.4,
                "fatigue_slip_factor": 0.2,
            },
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )

    def test_slip_probability_increases_with_execution_difficulty(self) -> None:
        world = SpiderWorld(
            seed=211,
            noise_profile=self._slip_profile(),
            lizard_move_interval=999999,
        )
        world.reset(seed=211)
        terrain, pos = _terrain_with_cleanup(self, world)

        easy = _compute_slip_and_difficulty(
            world,
            terrain,
            pos,
            OPEN,
            heading=(1.0, 0.0),
            fatigue=0.0,
        )
        hard = _compute_slip_and_difficulty(
            world,
            terrain,
            pos,
            NARROW,
            heading=(-1.0, 0.0),
            fatigue=1.0,
        )

        self.assertLess(easy["slip_probability"], hard["slip_probability"])
        self.assertLess(easy["execution_difficulty"], hard["execution_difficulty"])

    def test_slip_sampler_biases_toward_stay(self) -> None:
        class FakeMotorRng:
            def __init__(self) -> None:
                self.weights: list[float] = []

            def choice(self, count: int, p: Sequence[float]) -> int:
                self.weights = [float(value) for value in p]
                return 0

        world = SpiderWorld(seed=223, lizard_move_interval=999999)
        fake_rng = FakeMotorRng()
        world.motor_rng = fake_rng

        self.assertEqual(world._sample_slip_action("MOVE_UP"), "STAY")
        self.assertGreater(fake_rng.weights[0], fake_rng.weights[1])
        self.assertGreater(fake_rng.weights[0], fake_rng.weights[2])
        self.assertAlmostEqual(sum(fake_rng.weights), 1.0)

    def test_slip_sampler_deviates_to_adjacent_not_reverse(self) -> None:
        class FakeMotorRng:
            def __init__(self, index: int) -> None:
                self.index = index

            def choice(self, count: int, p: Sequence[float]) -> int:
                return self.index

        world = SpiderWorld(seed=227, lizard_move_interval=999999)
        world.motor_rng = FakeMotorRng(1)
        self.assertEqual(world._sample_slip_action("MOVE_RIGHT"), "MOVE_UP")
        world.motor_rng = FakeMotorRng(2)
        self.assertEqual(world._sample_slip_action("MOVE_RIGHT"), "MOVE_DOWN")
        self.assertNotEqual(world._sample_slip_action("MOVE_RIGHT"), "MOVE_LEFT")

    def test_end_to_end_motor_selection_slips_to_final_execution_action(self) -> None:
        world = SpiderWorld(
            seed=229,
            noise_profile=self._slip_profile(base=1.0),
            lizard_move_interval=999999,
        )
        brain = SpiderBrain(seed=229, module_dropout=0.0)
        observation = world.reset(seed=229)

        def force_move_up(_inputs: np.ndarray, *, store_cache: bool = False) -> np.ndarray:
            logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
            logits[ACTION_TO_INDEX["MOVE_UP"]] = 100.0
            return logits

        with patch.object(brain.motor_cortex, "forward", side_effect=force_move_up):
            decision = brain.act(observation, bus=None, sample=False)

        self.assertEqual(decision.motor_action_idx, ACTION_TO_INDEX["MOVE_UP"])
        self.assertEqual(decision.action_idx, ACTION_TO_INDEX["MOVE_UP"])

        _, _, _, info = world.step(decision.action_idx)
        self.assertEqual(info["intended_action"], "MOVE_UP")
        self.assertEqual(info["motor_slip"]["original_action"], "MOVE_UP")
        self.assertTrue(info["motor_noise_applied"])
        self.assertNotEqual(info["executed_action"], info["intended_action"])
        self.assertIn(info["executed_action"], {"STAY", "MOVE_LEFT", "MOVE_RIGHT"})


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


class MoveDeltasTest(unittest.TestCase):
    """Tests for the updated MOVE_DELTAS constant (excludes all zero-delta actions)."""

    def test_move_deltas_does_not_contain_zero_delta(self) -> None:
        for delta in MOVE_DELTAS:
            self.assertNotEqual(delta, (0, 0), f"MOVE_DELTAS contains zero delta {delta!r}")

    def test_move_deltas_contains_four_cardinal_directions(self) -> None:
        self.assertEqual(len(MOVE_DELTAS), 4)

    def test_move_deltas_contains_all_cardinal_deltas(self) -> None:
        expected = {(0, -1), (0, 1), (-1, 0), (1, 0)}
        self.assertEqual(set(MOVE_DELTAS), expected)

    def test_stay_not_in_move_deltas(self) -> None:
        from spider_cortex_sim.interfaces import ACTION_DELTAS
        self.assertNotIn(ACTION_DELTAS["STAY"], MOVE_DELTAS)

    def test_orient_deltas_not_in_move_deltas(self) -> None:
        from spider_cortex_sim.interfaces import ACTION_DELTAS, ORIENT_HEADINGS
        for action_name in ORIENT_HEADINGS:
            self.assertNotIn(action_name, ACTION_DELTAS)
        self.assertEqual(
            tuple(delta for delta in ACTION_DELTAS.values() if delta != (0, 0)),
            MOVE_DELTAS,
        )


class CopyObservationPayloadTest(unittest.TestCase):
    def test_numpy_arrays_are_copied_not_aliased(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        obs = {"visual": arr, "meta": {"tick": 0}}
        copied = _copy_observation_payload(obs)
        arr[0] = 99.0
        np.testing.assert_allclose(copied["visual"], np.array([1.0, 2.0, 3.0]))

    def test_dict_values_are_deep_copied(self) -> None:
        meta = {"tick": 5, "info": {"nested": True}}
        obs = {"meta": meta}
        copied = _copy_observation_payload(obs)
        meta["tick"] = 99
        meta["info"]["nested"] = False
        self.assertEqual(copied["meta"]["tick"], 5)
        self.assertTrue(copied["meta"]["info"]["nested"])

    def test_returns_new_dict_not_same_object(self) -> None:
        obs = {"visual": np.array([0.0]), "meta": {}}
        copied = _copy_observation_payload(obs)
        self.assertIsNot(copied, obs)

    def test_all_keys_preserved(self) -> None:
        obs = {
            "visual": np.array([1.0]),
            "sensory": np.array([2.0]),
            "meta": {"tick": 0},
        }
        copied = _copy_observation_payload(obs)
        self.assertEqual(set(copied.keys()), set(obs.keys()))

    def test_copied_numpy_array_is_independent(self) -> None:
        arr = np.zeros(3)
        obs = {"data": arr}
        copied = _copy_observation_payload(obs)
        arr[1] = 7.0
        np.testing.assert_allclose(copied["data"], np.zeros(3))

    def test_scalar_values_are_preserved(self) -> None:
        obs = {"count": 42, "score": 0.5}
        copied = _copy_observation_payload(obs)
        self.assertEqual(copied["count"], 42)
        self.assertAlmostEqual(copied["score"], 0.5)

    def test_empty_observation_returns_empty_dict(self) -> None:
        copied = _copy_observation_payload({})
        self.assertEqual(copied, {})


class IsTemporalDirectionFieldTest(unittest.TestCase):
    def test_food_dx_is_temporal_direction(self) -> None:
        self.assertTrue(_is_temporal_direction_field("food_dx"))

    def test_food_dy_is_temporal_direction(self) -> None:
        self.assertTrue(_is_temporal_direction_field("food_dy"))

    def test_predator_dx_is_temporal_direction(self) -> None:
        self.assertTrue(_is_temporal_direction_field("predator_dx"))

    def test_shelter_dy_is_temporal_direction(self) -> None:
        self.assertTrue(_is_temporal_direction_field("shelter_dy"))

    def test_heading_dx_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("heading_dx"))

    def test_heading_dy_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("heading_dy"))

    def test_last_move_dx_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("last_move_dx"))

    def test_last_move_dy_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("last_move_dy"))

    def test_memory_dx_field_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("food_memory_dx"))

    def test_memory_dy_field_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("shelter_memory_dy"))

    def test_certainty_field_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("food_certainty"))

    def test_strength_field_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("food_smell_strength"))

    def test_plain_string_without_suffix_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("hunger"))

    def test_trace_dx_field_is_temporal_direction(self) -> None:
        self.assertTrue(_is_temporal_direction_field("food_trace_dx"))

    def test_trace_dy_field_is_temporal_direction(self) -> None:
        self.assertTrue(_is_temporal_direction_field("predator_trace_dy"))


class PerceptualBufferEdgeCasesTest(unittest.TestCase):
    def test_capacity_equals_max_delay_plus_one(self) -> None:
        for max_delay in (0, 1, 2, 5):
            with self.subTest(max_delay=max_delay):
                buf = PerceptualBuffer(max_delay=max_delay)
                self.assertEqual(buf.capacity, max_delay + 1)

    def test_negative_max_delay_treated_as_zero(self) -> None:
        buf = PerceptualBuffer(max_delay=-3)
        self.assertEqual(buf.max_delay, 0)
        self.assertEqual(buf.capacity, 1)

    def test_clear_empties_buffer(self) -> None:
        buf = PerceptualBuffer(max_delay=3)
        buf.push(0, {"v": np.array([1.0])})
        buf.push(1, {"v": np.array([2.0])})
        buf.clear()
        with self.assertRaises(ValueError):
            buf.get(0)

    def test_clear_then_push_works_normally(self) -> None:
        buf = PerceptualBuffer(max_delay=2)
        buf.push(0, {"v": np.array([1.0])})
        buf.clear()
        buf.push(5, {"v": np.array([5.0])})
        obs, _ = buf.get(0)
        np.testing.assert_allclose(obs["v"], np.array([5.0]))

    def test_get_empty_buffer_raises_value_error(self) -> None:
        buf = PerceptualBuffer(max_delay=2)
        with self.assertRaises(ValueError):
            buf.get(0)

    def test_get_with_delay_larger_than_buffer_returns_oldest(self) -> None:
        buf = PerceptualBuffer(max_delay=5)
        buf.push(10, {"v": np.array([10.0])})
        buf.push(11, {"v": np.array([11.0])})
        # Request delay of 10 but only 2 entries → should return oldest
        obs, effective_delay = buf.get(10)
        np.testing.assert_allclose(obs["v"], np.array([10.0]))
        self.assertEqual(effective_delay, 1)

    def test_get_delay_zero_returns_newest(self) -> None:
        buf = PerceptualBuffer(max_delay=3)
        buf.push(0, {"v": np.array([1.0])})
        buf.push(1, {"v": np.array([2.0])})
        buf.push(2, {"v": np.array([3.0])})
        obs, effective_delay = buf.get(0)
        np.testing.assert_allclose(obs["v"], np.array([3.0]))
        self.assertEqual(effective_delay, 0)

    def test_get_negative_delay_treated_as_zero(self) -> None:
        buf = PerceptualBuffer(max_delay=2)
        buf.push(0, {"v": np.array([9.0])})
        obs, effective_delay = buf.get(-5)
        np.testing.assert_allclose(obs["v"], np.array([9.0]))
        self.assertEqual(effective_delay, 0)

    def test_buffer_prunes_old_entries_to_maintain_capacity(self) -> None:
        buf = PerceptualBuffer(max_delay=2)
        for tick in range(10):
            buf.push(tick, {"tick": tick})
        self.assertLessEqual(len(buf._entries), buf.capacity)

    def test_get_returns_independent_copy(self) -> None:
        """Mutations to returned observation should not affect buffer contents."""
        buf = PerceptualBuffer(max_delay=1)
        buf.push(0, {"visual": np.array([1.0, 2.0])})
        obs1, _ = buf.get(0)
        obs1["visual"][0] = 99.0
        obs2, _ = buf.get(0)
        np.testing.assert_allclose(obs2["visual"], np.array([1.0, 2.0]))


class PerceptualDelayTicksTest(unittest.TestCase):
    def test_default_perceptual_delay_ticks_rounds_to_one(self) -> None:
        world = SpiderWorld(seed=33, lizard_move_interval=999999)
        world.reset(seed=33)
        self.assertEqual(world._perceptual_delay_ticks(), 1)

    def test_perceptual_delay_ticks_respects_configured_value(self) -> None:
        profile = _profile_with_perception(perceptual_delay_ticks=2.0)
        world = SpiderWorld(seed=33, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=33)
        self.assertEqual(world._perceptual_delay_ticks(), 2)

    def test_perceptual_delay_ticks_rounds_correctly(self) -> None:
        profile = _profile_with_perception(perceptual_delay_ticks=1.6)
        world = SpiderWorld(seed=33, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=33)
        self.assertEqual(world._perceptual_delay_ticks(), 2)

    def test_perceptual_delay_ticks_clips_negative_to_zero(self) -> None:
        profile = _profile_with_perception(perceptual_delay_ticks=-3.0)
        world = SpiderWorld(seed=33, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=33)
        self.assertEqual(world._perceptual_delay_ticks(), 0)

    def test_perceptual_delay_noise_scale_default_is_positive(self) -> None:
        world = SpiderWorld(seed=33, lizard_move_interval=999999)
        world.reset(seed=33)
        self.assertGreater(world._perceptual_delay_noise_scale(), 0.0)

    def test_perceptual_delay_noise_scale_respects_configured_value(self) -> None:
        profile = _profile_with_perception(perceptual_delay_noise=0.75)
        world = SpiderWorld(seed=33, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=33)
        self.assertAlmostEqual(world._perceptual_delay_noise_scale(), 0.75)

    def test_perceptual_delay_noise_scale_clips_negative_to_zero(self) -> None:
        profile = _profile_with_perception(perceptual_delay_noise=-1.0)
        world = SpiderWorld(seed=33, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=33)
        self.assertAlmostEqual(world._perceptual_delay_noise_scale(), 0.0)


class DelayRngChannelTest(unittest.TestCase):
    def test_delay_rng_exists_after_reset(self) -> None:
        world = SpiderWorld(seed=55, lizard_move_interval=999999)
        world.reset(seed=55)
        self.assertTrue(hasattr(world, "delay_rng"))

    def test_delay_rng_is_distinct_from_other_channels(self) -> None:
        world = SpiderWorld(seed=55, lizard_move_interval=999999)
        world.reset(seed=55)
        channels = [world.spawn_rng, world.predator_rng, world.visual_rng, world.olfactory_rng, world.motor_rng, world.delay_rng]
        ids = [id(ch) for ch in channels]
        self.assertEqual(len(ids), len(set(ids)), "RNG channels are not all distinct objects")

    def test_delay_rng_produces_reproducible_results(self) -> None:
        world_a = SpiderWorld(seed=57, lizard_move_interval=999999)
        world_b = SpiderWorld(seed=57, lizard_move_interval=999999)
        world_a.reset(seed=57)
        world_b.reset(seed=57)
        self.assertAlmostEqual(float(world_a.delay_rng.random()), float(world_b.delay_rng.random()))

    def test_delay_rng_differs_with_different_seeds(self) -> None:
        world_a = SpiderWorld(seed=59, lizard_move_interval=999999)
        world_b = SpiderWorld(seed=61, lizard_move_interval=999999)
        world_a.reset(seed=59)
        world_b.reset(seed=61)
        val_a = float(world_a.delay_rng.random())
        val_b = float(world_b.delay_rng.random())
        self.assertNotAlmostEqual(val_a, val_b, places=6)


class OrientActionHeadingTest(unittest.TestCase):
    """Regression/boundary tests for ORIENT action heading updates."""

    def _make_world(self) -> SpiderWorld:
        profile = _profile_with_perception(perceptual_delay_ticks=0.0)
        world = SpiderWorld(seed=81, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=81)
        return world

    def test_all_orient_actions_update_heading_correctly(self) -> None:
        from spider_cortex_sim.interfaces import ORIENT_HEADINGS
        expected = {
            "ORIENT_UP": (0, -1),
            "ORIENT_DOWN": (0, 1),
            "ORIENT_LEFT": (-1, 0),
            "ORIENT_RIGHT": (1, 0),
        }
        for action_name, (expected_dx, expected_dy) in expected.items():
            with self.subTest(action=action_name):
                world = self._make_world()
                world.state.heading_dx = 0
                world.state.heading_dy = 0
                world.step(ACTION_TO_INDEX[action_name])
                self.assertEqual(world.state.heading_dx, expected_dx)
                self.assertEqual(world.state.heading_dy, expected_dy)

    def test_orient_action_clears_last_move_deltas(self) -> None:
        world = self._make_world()
        world.state.last_move_dx = 1
        world.state.last_move_dy = -1
        world.step(ACTION_TO_INDEX["ORIENT_DOWN"])
        self.assertEqual(world.state.last_move_dx, 0)
        self.assertEqual(world.state.last_move_dy, 0)

    def test_orient_action_motor_slip_result_has_zero_probability(self) -> None:
        world = self._make_world()
        for action_name in ("ORIENT_UP", "ORIENT_DOWN", "ORIENT_LEFT", "ORIENT_RIGHT"):
            with self.subTest(action=action_name):
                result = world._apply_motor_noise(action_name)
                self.assertAlmostEqual(result["slip_probability"], 0.0)
                self.assertFalse(result["occurred"])
                self.assertEqual(result["original_action"], action_name)
                self.assertEqual(result["executed_action"], action_name)

    def test_orient_action_motor_noise_result_reason_is_none(self) -> None:
        world = self._make_world()
        result = world._apply_motor_noise("ORIENT_LEFT")
        self.assertEqual(result["reason"], "none")


if __name__ == "__main__":
    unittest.main()
