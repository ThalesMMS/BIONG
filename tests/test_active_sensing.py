import unittest

import numpy as np

from spider_cortex_sim.interfaces import ACTION_TO_INDEX, OBSERVATION_INTERFACE_BY_KEY
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.world import SpiderWorld


def _profile_with_perception(**perception_overrides: float) -> OperationalProfile:
    summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
    summary["perception"].update(perception_overrides)
    return OperationalProfile.from_summary(summary)


def _always_slip_noise_profile() -> NoiseConfig:
    return NoiseConfig(
        name="always_slip_for_locomotion",
        visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
        olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
        motor={"action_flip_prob": 1.0},
        spawn={"uniform_mix": 0.0},
        predator={"random_choice_prob": 0.0},
    )


class ActiveSensingLoopTest(unittest.TestCase):
    def _make_world(self, *, delay_ticks: float = 0.0) -> SpiderWorld:
        profile = _profile_with_perception(
            perceptual_delay_ticks=delay_ticks,
            perceptual_delay_noise=0.0,
            max_scan_age=10.0,
        )
        world = SpiderWorld(
            seed=91,
            vision_range=6,
            lizard_move_interval=999999,
            operational_profile=profile,
        )
        world.reset(seed=91)
        return world

    def _place_food_east_of_west_facing_spider(self, world: SpiderWorld) -> None:
        world.state.x = 2
        world.state.y = 2
        world.state.heading_dx = -1
        world.state.heading_dy = 0
        world.food_positions = [(4, 2)]
        world.lizard.x = 10
        world.lizard.y = 10

    def _visual(self, obs: dict[str, object]) -> dict[str, float]:
        return OBSERVATION_INTERFACE_BY_KEY["visual"].bind_values(obs["visual"])

    def _sensory(self, obs: dict[str, object]) -> dict[str, float]:
        return OBSERVATION_INTERFACE_BY_KEY["sensory"].bind_values(obs["sensory"])

    def _hunger(self, obs: dict[str, object]) -> dict[str, float]:
        return OBSERVATION_INTERFACE_BY_KEY["hunger"].bind_values(obs["hunger"])

    def test_orient_action_reveals_target_in_same_returned_tick(self) -> None:
        world = self._make_world(delay_ticks=1.0)
        self._place_food_east_of_west_facing_spider(world)
        stale_visual = self._visual(world.observe())

        refreshed_obs, _, _, info = world.step(ACTION_TO_INDEX["ORIENT_RIGHT"])
        refreshed_visual = self._visual(refreshed_obs)

        self.assertAlmostEqual(stale_visual["food_visible"], 0.0)
        self.assertAlmostEqual(refreshed_visual["food_visible"], 1.0)
        self.assertAlmostEqual(refreshed_visual["foveal_scan_age"], 0.0)
        self.assertEqual(info["executed_action"], "ORIENT_RIGHT")
        self.assertEqual(world.tick, 1)

    def test_orient_action_replaces_perceptual_buffer_current_tick(self) -> None:
        world = self._make_world(delay_ticks=1.0)
        self._place_food_east_of_west_facing_spider(world)
        stale_visual = self._visual(world.observe())

        world._move_spider_action("ORIENT_RIGHT")
        buffered_obs, effective_delay = world._perceptual_buffer.get(0)
        buffered_visual = self._visual(buffered_obs)

        self.assertAlmostEqual(stale_visual["food_visible"], 0.0)
        self.assertEqual(effective_delay, 0)
        self.assertAlmostEqual(buffered_visual["food_visible"], 1.0)
        self.assertAlmostEqual(buffered_visual["foveal_scan_age"], 0.0)

    def test_orient_action_does_not_apply_slip_or_move(self) -> None:
        world = SpiderWorld(
            seed=92,
            lizard_move_interval=999999,
            noise_profile=_always_slip_noise_profile(),
        )
        world.reset(seed=92)
        start_pos = world.spider_pos()

        _, _, _, info = world.step(ACTION_TO_INDEX["ORIENT_UP"])

        self.assertEqual(world.spider_pos(), start_pos)
        self.assertEqual(info["executed_action"], "ORIENT_UP")
        self.assertFalse(info["motor_noise_applied"])
        self.assertFalse(info["motor_slip"]["occurred"])
        self.assertAlmostEqual(info["motor_slip"]["slip_probability"], 0.0)
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), (0, -1))

    def test_scan_recency_increments_and_resets_in_active_loop(self) -> None:
        world = self._make_world(delay_ticks=1.0)
        self._place_food_east_of_west_facing_spider(world)

        fresh_obs, _, _, _ = world.step(ACTION_TO_INDEX["ORIENT_RIGHT"])
        one_tick_obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])
        two_tick_obs, _, _, _ = world.step(ACTION_TO_INDEX["STAY"])
        rescanned_obs, _, _, _ = world.step(ACTION_TO_INDEX["ORIENT_RIGHT"])

        self.assertAlmostEqual(self._visual(fresh_obs)["foveal_scan_age"], 0.0)
        self.assertAlmostEqual(self._visual(one_tick_obs)["foveal_scan_age"], 0.1)
        self.assertAlmostEqual(self._visual(two_tick_obs)["foveal_scan_age"], 0.2)
        self.assertAlmostEqual(self._visual(rescanned_obs)["foveal_scan_age"], 0.0)

    def test_trace_heading_fields_match_scan_heading_after_capture(self) -> None:
        world = self._make_world(delay_ticks=0.0)
        self._place_food_east_of_west_facing_spider(world)

        obs, _, _, _ = world.step(ACTION_TO_INDEX["ORIENT_RIGHT"])
        visual = self._visual(obs)
        hunger = self._hunger(obs)

        self.assertEqual(world.state.food_trace.target, (4, 2))
        self.assertEqual((world.state.food_trace.heading_dx, world.state.food_trace.heading_dy), (1, 0))
        self.assertAlmostEqual(visual["food_trace_heading_dx"], 1.0)
        self.assertAlmostEqual(visual["food_trace_heading_dy"], 0.0)
        self.assertAlmostEqual(hunger["food_trace_heading_dx"], 1.0)
        self.assertAlmostEqual(hunger["food_trace_heading_dy"], 0.0)

    def test_olfactory_gradients_are_unchanged_by_heading_changes(self) -> None:
        world = self._make_world(delay_ticks=0.0)
        world.state.x = 3
        world.state.y = 3
        world.state.heading_dx = -1
        world.state.heading_dy = 0
        world.food_positions = [(5, 3)]
        world.lizard.x = 3
        world.lizard.y = 6

        left_heading_obs = world.observe()
        left_heading_sensory = self._sensory(left_heading_obs)
        left_heading_vector = left_heading_obs["sensory"].copy()
        world._move_spider_action("ORIENT_RIGHT")
        right_heading_obs = world.observe()
        right_heading_sensory = self._sensory(right_heading_obs)

        for field_name in (
            "food_smell_strength",
            "food_smell_dx",
            "food_smell_dy",
            "predator_smell_strength",
            "predator_smell_dx",
            "predator_smell_dy",
        ):
            self.assertAlmostEqual(left_heading_sensory[field_name], right_heading_sensory[field_name])
        np.testing.assert_allclose(left_heading_vector, right_heading_obs["sensory"])


if __name__ == "__main__":
    unittest.main()
