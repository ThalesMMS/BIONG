import unittest

from spider_cortex_sim.physiology import (
    SLEEP_PHASE_LEVELS,
    SLEEP_PHASES,
    apply_homeostasis_penalties,
    apply_predator_contact,
    apply_restoration,
    apply_wakefulness,
    clip_state,
    reset_sleep_state,
    resolve_autonomic_behaviors,
    rest_streak_norm,
    set_sleep_state,
    sleep_phase_from_streak,
    sleep_phase_level,
)
from spider_cortex_sim.reward import REWARD_COMPONENT_NAMES
from spider_cortex_sim.world import SpiderWorld


class PhysiologyModuleTest(unittest.TestCase):
    def test_set_and_reset_sleep_state(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        set_sleep_state(world, "RESTING", 2)
        self.assertEqual(world.state.sleep_phase, "RESTING")
        self.assertEqual(world.state.rest_streak, 2)
        reset_sleep_state(world)
        self.assertEqual(world.state.sleep_phase, "AWAKE")
        self.assertEqual(world.state.rest_streak, 0)

    def test_sleep_phase_from_streak_matches_shelter_rules(self) -> None:
        self.assertEqual(sleep_phase_from_streak(1, night=True, shelter_role="inside"), "SETTLING")
        self.assertEqual(sleep_phase_from_streak(2, night=True, shelter_role="inside"), "RESTING")
        self.assertEqual(sleep_phase_from_streak(3, night=True, shelter_role="deep"), "DEEP_SLEEP")

    def test_apply_restoration_reduces_fatigue_and_sleep_debt(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.fatigue = 0.8
        world.state.sleep_debt = 0.7
        apply_restoration(world, "DEEP_SLEEP", night=True, shelter_role="deep")
        self.assertLess(world.state.fatigue, 0.8)
        self.assertLess(world.state.sleep_debt, 0.7)

    def test_resolve_autonomic_behaviors_feeds_when_on_food(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = world.food_positions[0]
        world.state.hunger = 0.9
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info = {"ate": False, "slept": False}
        resolve_autonomic_behaviors(
            world,
            action_name="STAY",
            predator_threat=False,
            night=False,
            reward_components=reward_components,
            info=info,
        )
        self.assertTrue(info["ate"])
        self.assertLess(world.state.hunger, 0.9)

    def test_apply_homeostasis_penalties_reduces_health_above_thresholds(self) -> None:
        world = SpiderWorld(seed=9, lizard_move_interval=999999)
        world.reset(seed=9)
        world.state.hunger = 0.9
        world.state.fatigue = 0.9
        world.state.sleep_debt = 0.9
        health_before = world.state.health
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        apply_homeostasis_penalties(world, reward_components)
        self.assertLess(world.state.health, health_before)
        self.assertLess(reward_components["homeostasis_penalty"], 0.0)

    def test_sleep_phase_levels_cover_all_phases(self) -> None:
        self.assertIn("AWAKE", SLEEP_PHASE_LEVELS)
        self.assertIn("SETTLING", SLEEP_PHASE_LEVELS)
        self.assertIn("RESTING", SLEEP_PHASE_LEVELS)
        self.assertIn("DEEP_SLEEP", SLEEP_PHASE_LEVELS)
        self.assertAlmostEqual(SLEEP_PHASE_LEVELS["AWAKE"], 0.0)
        self.assertAlmostEqual(SLEEP_PHASE_LEVELS["DEEP_SLEEP"], 1.0)

    def test_sleep_phases_tuple_has_four_items(self) -> None:
        self.assertEqual(len(SLEEP_PHASES), 4)
        self.assertEqual(SLEEP_PHASES[0], "AWAKE")
        self.assertEqual(SLEEP_PHASES[-1], "DEEP_SLEEP")

    def test_sleep_phase_level_returns_correct_values(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        for phase, expected in SLEEP_PHASE_LEVELS.items():
            world.state.sleep_phase = phase
            self.assertAlmostEqual(sleep_phase_level(world), expected)

    def test_sleep_phase_level_unknown_phase_returns_zero(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.sleep_phase = "UNKNOWN_PHASE"
        self.assertAlmostEqual(sleep_phase_level(world), 0.0)

    def test_rest_streak_norm_zero_streak(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.rest_streak = 0
        self.assertAlmostEqual(rest_streak_norm(world), 0.0)

    def test_rest_streak_norm_three_streak(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.rest_streak = 3
        self.assertAlmostEqual(rest_streak_norm(world), 1.0)

    def test_rest_streak_norm_exceeds_three_clamps(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.rest_streak = 10
        self.assertAlmostEqual(rest_streak_norm(world), 1.0)

    def test_rest_streak_norm_two_streak(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.rest_streak = 2
        self.assertAlmostEqual(rest_streak_norm(world), 2.0 / 3.0)

    def test_apply_wakefulness_night_adds_more_debt_than_day(self) -> None:
        world_night = SpiderWorld(seed=1, lizard_move_interval=999999)
        world_night.reset(seed=1)
        world_night.state.sleep_debt = 0.0
        apply_wakefulness(world_night, night=True, exposed=False, interrupted_rest=False)
        debt_night = world_night.state.sleep_debt

        world_day = SpiderWorld(seed=1, lizard_move_interval=999999)
        world_day.reset(seed=1)
        world_day.state.sleep_debt = 0.0
        apply_wakefulness(world_day, night=False, exposed=False, interrupted_rest=False)
        debt_day = world_day.state.sleep_debt

        self.assertGreater(debt_night, debt_day)

    def test_apply_wakefulness_exposed_increases_debt(self) -> None:
        world_exposed = SpiderWorld(seed=1, lizard_move_interval=999999)
        world_exposed.reset(seed=1)
        world_exposed.state.sleep_debt = 0.0
        apply_wakefulness(world_exposed, night=True, exposed=True, interrupted_rest=False)
        debt_exposed = world_exposed.state.sleep_debt

        world_safe = SpiderWorld(seed=1, lizard_move_interval=999999)
        world_safe.reset(seed=1)
        world_safe.state.sleep_debt = 0.0
        apply_wakefulness(world_safe, night=True, exposed=False, interrupted_rest=False)
        debt_safe = world_safe.state.sleep_debt

        self.assertGreater(debt_exposed, debt_safe)

    def test_apply_wakefulness_interrupted_rest_adds_large_debt(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.sleep_debt = 0.0
        apply_wakefulness(world, night=False, exposed=False, interrupted_rest=True)
        self.assertGreater(world.state.sleep_debt, 0.0)

    def test_apply_predator_contact_decreases_health(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.health = 1.0
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info: dict = {}
        apply_predator_contact(world, reward_components, info)
        self.assertLess(world.state.health, 1.0)
        self.assertGreaterEqual(world.state.health, 0.0)

    def test_apply_predator_contact_sets_pain_and_contact(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.recent_pain = 0.0
        world.state.recent_contact = 0.0
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info: dict = {}
        apply_predator_contact(world, reward_components, info)
        self.assertGreater(world.state.recent_pain, 0.0)
        self.assertAlmostEqual(world.state.recent_contact, 1.0)
        self.assertTrue(info.get("pain"))
        self.assertTrue(info.get("predator_contact"))

    def test_apply_predator_contact_penalizes_reward(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info: dict = {}
        apply_predator_contact(world, reward_components, info)
        self.assertLess(reward_components["predator_contact"], 0.0)

    def test_apply_predator_contact_increments_counters(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.alert_events = 0
        world.state.predator_contacts = 0
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info: dict = {}
        apply_predator_contact(world, reward_components, info)
        self.assertEqual(world.state.alert_events, 1)
        self.assertEqual(world.state.predator_contacts, 1)

    def test_apply_restoration_settling_reduces_fatigue(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.fatigue = 0.5
        world.state.sleep_debt = 0.5
        fatigue_before = world.state.fatigue
        apply_restoration(world, "SETTLING", night=True, shelter_role="inside")
        self.assertLess(world.state.fatigue, fatigue_before)

    def test_apply_restoration_settling_night_increases_health(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.health = 0.8
        apply_restoration(world, "SETTLING", night=True, shelter_role="inside")
        self.assertGreater(world.state.health, 0.8)

    def test_apply_restoration_settling_day_no_health_gain(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.health = 0.8
        apply_restoration(world, "SETTLING", night=False, shelter_role="inside")
        self.assertAlmostEqual(world.state.health, 0.8)

    def test_apply_restoration_resting_entrance_less_effective(self) -> None:
        world_entrance = SpiderWorld(seed=3, lizard_move_interval=999999)
        world_entrance.reset(seed=3)
        world_entrance.state.fatigue = 0.6
        world_entrance.state.sleep_debt = 0.6
        apply_restoration(world_entrance, "RESTING", night=True, shelter_role="entrance")

        world_deep = SpiderWorld(seed=3, lizard_move_interval=999999)
        world_deep.reset(seed=3)
        world_deep.state.fatigue = 0.6
        world_deep.state.sleep_debt = 0.6
        apply_restoration(world_deep, "RESTING", night=True, shelter_role="deep")

        self.assertGreater(world_entrance.state.fatigue, world_deep.state.fatigue)
        self.assertGreater(world_entrance.state.sleep_debt, world_deep.state.sleep_debt)

    def test_clip_state_clamps_values_above_one(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.hunger = 2.5
        world.state.fatigue = 1.8
        world.state.sleep_debt = 3.0
        world.state.health = 1.5
        world.state.recent_pain = 2.0
        world.state.recent_contact = 5.0
        clip_state(world)
        self.assertAlmostEqual(world.state.hunger, 1.0)
        self.assertAlmostEqual(world.state.fatigue, 1.0)
        self.assertAlmostEqual(world.state.sleep_debt, 1.0)
        self.assertAlmostEqual(world.state.health, 1.0)
        self.assertAlmostEqual(world.state.recent_pain, 1.0)
        self.assertAlmostEqual(world.state.recent_contact, 1.0)

    def test_clip_state_clamps_values_below_zero(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.hunger = -0.5
        world.state.fatigue = -1.0
        world.state.sleep_debt = -0.2
        world.state.health = -0.1
        clip_state(world)
        self.assertAlmostEqual(world.state.hunger, 0.0)
        self.assertAlmostEqual(world.state.fatigue, 0.0)
        self.assertAlmostEqual(world.state.sleep_debt, 0.0)
        self.assertAlmostEqual(world.state.health, 0.0)

    def test_sleep_phase_from_streak_zero_returns_awake(self) -> None:
        self.assertEqual(sleep_phase_from_streak(0, night=True, shelter_role="deep"), "AWAKE")

    def test_sleep_phase_from_streak_outside_returns_awake(self) -> None:
        self.assertEqual(sleep_phase_from_streak(2, night=True, shelter_role="outside"), "AWAKE")

    def test_sleep_phase_from_streak_deep_day_streak_3_returns_resting(self) -> None:
        self.assertEqual(sleep_phase_from_streak(3, night=False, shelter_role="deep"), "RESTING")

    def test_sleep_phase_from_streak_deep_streak_2_returns_resting(self) -> None:
        self.assertEqual(sleep_phase_from_streak(2, night=True, shelter_role="deep"), "RESTING")

    def test_set_sleep_state_rejects_invalid_phase(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        with self.assertRaises(ValueError):
            set_sleep_state(world, "INVALID_PHASE", 0)

    def test_set_sleep_state_clamps_rest_streak_to_zero(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        set_sleep_state(world, "AWAKE", -5)
        self.assertEqual(world.state.rest_streak, 0)