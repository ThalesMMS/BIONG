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

from tests.fixtures.world import _profile_with_perception

class MomentumDynamicsTest(unittest.TestCase):
    """Tests for the stateful execution-momentum transition."""

    def _world(self) -> SpiderWorld:
        world = SpiderWorld(seed=97, noise_profile="none", lizard_move_interval=999999)
        world.reset(seed=97)
        return world

    def _place_for_action(self, world: SpiderWorld, action_name: str) -> None:
        dx, dy = ACTION_DELTAS[action_name]
        for y in range(world.height):
            for x in range(world.width):
                target = (x + dx, y + dy)
                if not (0 <= target[0] < world.width and 0 <= target[1] < world.height):
                    continue
                if world.is_walkable((x, y)) and world.is_walkable(target):
                    world.state.x, world.state.y = x, y
                    return
        self.fail(f"No walkable source found for {action_name}.")

    def _place_for_sequence(self, world: SpiderWorld, action_names: Sequence[str]) -> None:
        for y in range(world.height):
            for x in range(world.width):
                current = (x, y)
                if not world.is_walkable(current):
                    continue
                valid = True
                for action_name in action_names:
                    dx, dy = ACTION_DELTAS.get(action_name, (0, 0))
                    target = (current[0] + dx, current[1] + dy)
                    if not (0 <= target[0] < world.width and 0 <= target[1] < world.height):
                        valid = False
                        break
                    if not world.is_walkable(target):
                        valid = False
                        break
                    current = target
                if valid:
                    world.state.x, world.state.y = x, y
                    return
        self.fail(f"No walkable path found for {action_names}.")

    def _run_action_stage(self, world: SpiderWorld, action_name: str) -> TickContext:
        context = tick_stages.build_tick_context(world, ACTION_TO_INDEX[action_name])
        tick_stages.run_action_stage(world, context)
        return context

    def _step(
        self,
        action_name: str,
        *,
        heading: tuple[int, int] = (1, 0),
        momentum: float = 0.6,
    ) -> tuple[SpiderWorld, dict[str, object]]:
        world = self._world()
        if action_name in ACTION_DELTAS and action_name != "STAY":
            self._place_for_action(world, action_name)
        world.state.heading_dx, world.state.heading_dy = heading
        world.state.momentum = momentum
        _, _, _, info = world.step(ACTION_TO_INDEX[action_name])
        return world, info

    def test_aligned_moves_build_momentum(self) -> None:
        world = self._world()
        self._place_for_sequence(world, ["MOVE_RIGHT"] * 5)
        world.state.heading_dx, world.state.heading_dy = (1, 0)
        world.state.momentum = 0.0

        observed: list[float] = []
        for _ in range(5):
            _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_RIGHT"])
            observed.append(float(world.state.momentum))

        self.assertEqual(observed, sorted(observed))
        self.assertTrue(all(later > earlier for earlier, later in zip(observed, observed[1:])))
        self.assertAlmostEqual(observed[-1], 0.75)
        self.assertAlmostEqual(info["momentum_after"], observed[-1])

    def test_perpendicular_turn_halves_momentum(self) -> None:
        world = self._world()
        self._place_for_sequence(world, ["MOVE_RIGHT", "MOVE_UP"])
        world.state.heading_dx, world.state.heading_dy = (1, 0)
        world.state.momentum = 0.6

        world.step(ACTION_TO_INDEX["MOVE_RIGHT"])
        momentum_before_turn = float(world.state.momentum)
        world.step(ACTION_TO_INDEX["MOVE_UP"])

        self.assertAlmostEqual(world.state.momentum, momentum_before_turn * 0.5)

    def test_reverse_move_resets_momentum(self) -> None:
        world = self._world()
        self._place_for_sequence(world, ["MOVE_RIGHT", "MOVE_LEFT"])
        world.state.heading_dx, world.state.heading_dy = (1, 0)
        world.state.momentum = 0.6

        world.step(ACTION_TO_INDEX["MOVE_RIGHT"])
        world.step(ACTION_TO_INDEX["MOVE_LEFT"])

        self.assertAlmostEqual(world.state.momentum, 0.0)

    def test_stay_decays_momentum_gradually(self) -> None:
        world, _ = self._step("STAY", heading=(1, 0), momentum=0.5)

        self.assertAlmostEqual(world.state.momentum, 0.4)
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertAlmostEqual(world.state.momentum, 0.32)
        self.assertAlmostEqual(info["momentum_after"], 0.32)

    def test_orient_action_resets_momentum(self) -> None:
        for action_name in ORIENT_HEADINGS:
            with self.subTest(action_name=action_name):
                world, _ = self._step(action_name, heading=(1, 0), momentum=0.5)

                self.assertAlmostEqual(world.state.momentum, 0.0)

    def test_momentum_clamped_to_one(self) -> None:
        world, _ = self._step("MOVE_RIGHT", heading=(1, 0), momentum=0.95)

        self.assertAlmostEqual(world.state.momentum, 1.0)

    def test_pre_tick_snapshot_captures_momentum_before_action(self) -> None:
        _, info = self._step("STAY", heading=(1, 0), momentum=0.5)

        pre_tick = next(item for item in info["event_log"] if item["stage"] == "pre_tick")
        self.assertAlmostEqual(pre_tick["payload"]["momentum"], 0.5)

    def test_event_log_action_resolved_includes_momentum(self) -> None:
        _, info = self._step("STAY", heading=(1, 0), momentum=0.5)

        action_resolved = next(
            item for item in info["event_log"]
            if item["stage"] == "action" and item["name"] == "action_resolved"
        )
        self.assertAlmostEqual(action_resolved["payload"]["momentum_before"], 0.5)
        self.assertAlmostEqual(action_resolved["payload"]["momentum_after"], 0.4)
        self.assertIn("turn_angle", action_resolved["payload"])
        self.assertIn("turn_fatigue_applied", action_resolved["payload"])

    def test_perpendicular_turn_adds_turn_fatigue(self) -> None:
        world = self._world()
        self._place_for_action(world, "MOVE_UP")
        world.state.heading_dx, world.state.heading_dy = (1, 0)
        world.state.fatigue = 0.2

        context = self._run_action_stage(world, "MOVE_UP")

        self.assertAlmostEqual(world.state.fatigue, 0.202)
        self.assertAlmostEqual(context.info["turn_angle"], 90.0)
        self.assertAlmostEqual(context.info["turn_fatigue_applied"], 0.002)

    def test_reverse_turn_adds_higher_fatigue(self) -> None:
        world = self._world()
        self._place_for_action(world, "MOVE_LEFT")
        world.state.heading_dx, world.state.heading_dy = (1, 0)
        world.state.fatigue = 0.2

        context = self._run_action_stage(world, "MOVE_LEFT")

        self.assertAlmostEqual(world.state.fatigue, 0.204)
        self.assertAlmostEqual(context.info["turn_angle"], 180.0)
        self.assertAlmostEqual(context.info["turn_fatigue_applied"], 0.004)

    def test_aligned_move_no_turn_fatigue(self) -> None:
        world = self._world()
        self._place_for_action(world, "MOVE_RIGHT")
        world.state.heading_dx, world.state.heading_dy = (1, 0)
        world.state.fatigue = 0.2

        context = self._run_action_stage(world, "MOVE_RIGHT")

        self.assertAlmostEqual(world.state.fatigue, 0.2)
        self.assertAlmostEqual(context.info["turn_angle"], 0.0)
        self.assertAlmostEqual(context.info["turn_fatigue_applied"], 0.0)

    def test_orient_action_turn_fatigue(self) -> None:
        world = self._world()
        world.state.heading_dx, world.state.heading_dy = (1, 0)
        world.state.fatigue = 0.2

        context = self._run_action_stage(world, "ORIENT_LEFT")

        self.assertAlmostEqual(world.state.fatigue, 0.204)
        self.assertAlmostEqual(context.info["turn_angle"], 180.0)
        self.assertAlmostEqual(context.info["turn_fatigue_applied"], 0.004)

    def test_reverse_heading_change_applies_turn_fatigue(self) -> None:
        world, info = self._step("MOVE_LEFT", heading=(1, 0), momentum=0.8)

        action_resolved = next(
            item for item in info["event_log"]
            if item["stage"] == "action" and item["name"] == "action_resolved"
        )
        self.assertAlmostEqual(info["turn_angle"], 180.0)
        self.assertAlmostEqual(info["turn_fatigue_applied"], 0.004)
        self.assertAlmostEqual(action_resolved["payload"]["turn_fatigue_applied"], 0.004)
        self.assertGreaterEqual(world.state.fatigue, 0.004)

    def test_info_dict_includes_momentum(self) -> None:
        _, info = self._step("MOVE_LEFT", heading=(1, 0), momentum=0.8)

        self.assertIn("momentum", info)
        self.assertIn("momentum_before", info)
        self.assertIn("momentum_after", info)
        self.assertIn("turn_angle", info)
        self.assertIn("turn_fatigue_applied", info)
        self.assertAlmostEqual(info["momentum"], info["state"]["momentum"])
        self.assertAlmostEqual(info["momentum_after"], 0.0)
        self.assertAlmostEqual(info["turn_angle"], 180.0)
        self.assertAlmostEqual(info["turn_fatigue_applied"], 0.004)

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

    def test_trace_heading_dx_field_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("food_trace_heading_dx"))

    def test_trace_heading_dy_field_is_not_temporal_direction(self) -> None:
        self.assertFalse(_is_temporal_direction_field("predator_trace_heading_dy"))

class OrientActionIntegrationTest(unittest.TestCase):
    """Integration checks for ORIENT actions through `SpiderWorld.step()`."""

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

    def test_orient_action_refreshes_current_tick_perception_buffer(self) -> None:
        profile = _profile_with_perception(
            perceptual_delay_ticks=1.0,
            perceptual_delay_noise=0.0,
        )
        world = SpiderWorld(seed=81, lizard_move_interval=999999, operational_profile=profile)
        world.reset(seed=81)
        world.state.x = 2
        world.state.y = 2
        world.state.heading_dx = -1
        world.state.heading_dy = 0
        world.food_positions = [(4, 2)]
        world.lizard.x = 10
        world.lizard.y = 10

        stale_obs = world.observe()
        stale_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(stale_obs["visual"])
        self.assertAlmostEqual(stale_visual["food_visible"], 0.0)

        refreshed_obs, _, _, _ = world.step(ACTION_TO_INDEX["ORIENT_RIGHT"])
        refreshed_visual = OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(refreshed_obs["visual"])

        self.assertAlmostEqual(refreshed_visual["food_visible"], 1.0)
        self.assertAlmostEqual(refreshed_visual["foveal_scan_age"], 0.0)
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), (1, 0))

    def test_active_scan_refresh_does_not_advance_tick(self) -> None:
        world = self._make_world()
        start_tick = world.tick

        _refresh_perception_for_active_scan(world)

        self.assertEqual(world.tick, start_tick)
