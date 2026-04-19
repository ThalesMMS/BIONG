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

    def test_orient_action_records_scan_tick_for_new_heading(self) -> None:
        world = SpiderWorld(seed=72, noise_profile=self._no_flip_profile(), lizard_move_interval=999999)
        world.reset(seed=72)

        world._move_spider_action("ORIENT_RIGHT")

        self.assertEqual(world.state.last_scan_tick_right, world.tick)
        self.assertEqual(_scan_age_for_heading(world, 1, 0), 0)

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

    def test_successful_locomotion_records_scan_tick_for_move_heading(self) -> None:
        world = SpiderWorld(seed=76, noise_profile=self._no_flip_profile(), lizard_move_interval=999999)
        world.reset(seed=76)
        source = None
        for y in range(world.height):
            for x in range(world.width - 1):
                if world.is_walkable((x, y)) and world.is_walkable((x + 1, y)):
                    source = (x, y)
                    break
            if source is not None:
                break
        self.assertIsNotNone(source)
        start_x, start_y = source
        world.state.x, world.state.y = start_x, start_y
        world.state.last_scan_tick_right = -1

        moved = world._move_spider_action("MOVE_RIGHT")

        self.assertTrue(moved)
        self.assertEqual((world.state.x, world.state.y), (start_x + 1, start_y))
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), (1, 0))
        self.assertEqual(world.state.last_scan_tick_right, world.tick)
        self.assertEqual(_scan_age_for_heading(world, 1, 0), 0)

    def test_scan_age_tracks_diagonal_heading(self) -> None:
        world = SpiderWorld(seed=76, noise_profile=self._no_flip_profile(), lizard_move_interval=999999)
        world.reset(seed=76)

        world._record_scan_for_heading(1, -1)

        self.assertEqual(world.state.last_scan_tick_up_right, world.tick)
        self.assertEqual(_scan_age_for_heading(world, 1, -1), 0)

    def test_unscanned_heading_returns_stale_scan_age_sentinel(self) -> None:
        world = SpiderWorld(seed=76, noise_profile=self._no_flip_profile(), lizard_move_interval=999999)
        world.reset(seed=76)
        world.state.last_scan_tick_left = -1

        self.assertGreaterEqual(
            _scan_age_for_heading(world, -1, 0),
            world.operational_profile.perception["max_scan_age"],
        )

    def test_unscanned_heading_sentinel_respects_configured_scan_horizon(self) -> None:
        profile = _profile_with_perception(max_scan_age=150.0)
        world = SpiderWorld(
            seed=76,
            noise_profile=self._no_flip_profile(),
            lizard_move_interval=999999,
            operational_profile=profile,
        )
        world.reset(seed=76)
        world.state.last_scan_tick_left = -1

        self.assertEqual(_scan_age_for_heading(world, -1, 0), 150)
        self.assertEqual(_scan_age_for_heading(world, 0, 0), 150)

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
        self.assertIn("momentum", slip["components"])
        self.assertIn("motor_execution_difficulty", info)

    def test_slip_uses_stay_or_adjacent_action_not_reverse(self) -> None:
        world = SpiderWorld(seed=91, noise_profile=self._flip_all_profile(), lizard_move_interval=999999)
        world.reset(seed=91)
        _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_UP"])
        self.assertIn(info["executed_action"], {"STAY", "MOVE_LEFT", "MOVE_RIGHT"})
        self.assertNotEqual(info["executed_action"], "MOVE_DOWN")

class MotorExecutionIntegrationTest(unittest.TestCase):
    """End-to-end motor integration through brain selection and `SpiderWorld.step()`."""

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
