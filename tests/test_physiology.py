import unittest

from spider_cortex_sim.physiology import (
    SLEEP_PHASE_LEVELS,
    SLEEP_PHASES,
    apply_homeostasis_penalties,
    apply_predator_contact,
    apply_restoration,
    apply_turn_fatigue,
    apply_wakefulness,
    clip_state,
    heading_change_angle,
    reset_sleep_state,
    resolve_autonomic_behaviors,
    rest_streak_norm,
    set_sleep_state,
    sleep_phase_from_streak,
    sleep_phase_level,
)
from spider_cortex_sim.reward import REWARD_COMPONENT_NAMES
from spider_cortex_sim.world import SpiderWorld
from spider_cortex_sim.world_types import TickContext, TickSnapshot


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

    def test_heading_change_angle_reports_right_angle(self) -> None:
        self.assertAlmostEqual(heading_change_angle((1, 0), (0, -1)), 90.0)

    def test_heading_change_angle_reports_reverse(self) -> None:
        self.assertAlmostEqual(heading_change_angle((1, 0), (-1, 0)), 180.0)

    def test_apply_turn_fatigue_charges_right_angle_turn(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        initial_fatigue = 0.2
        expected_cost = float(world.reward_config["turn_fatigue_cost"])
        world.state.fatigue = initial_fatigue

        diagnostics = apply_turn_fatigue(world, (1, 0), (0, -1))

        self.assertAlmostEqual(world.state.fatigue, initial_fatigue + expected_cost)
        self.assertAlmostEqual(diagnostics["turn_angle"], 90.0)
        self.assertAlmostEqual(diagnostics["turn_fatigue_applied"], expected_cost)

    def test_apply_turn_fatigue_charges_sharp_turn(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        initial_fatigue = 0.2
        expected_cost = float(world.reward_config["turn_fatigue_cost"])
        world.state.fatigue = initial_fatigue

        diagnostics = apply_turn_fatigue(world, (1, 1), (-1, 0))

        self.assertAlmostEqual(world.state.fatigue, initial_fatigue + expected_cost)
        self.assertGreater(diagnostics["turn_angle"], 90.0)
        self.assertAlmostEqual(diagnostics["turn_fatigue_applied"], expected_cost)

    def test_apply_turn_fatigue_charges_reverse_cost(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        initial_fatigue = 0.2
        expected_cost = float(world.reward_config["reverse_fatigue_cost"])
        world.state.fatigue = initial_fatigue

        diagnostics = apply_turn_fatigue(world, (1, 0), (-1, 0))

        self.assertAlmostEqual(world.state.fatigue, initial_fatigue + expected_cost)
        self.assertAlmostEqual(diagnostics["turn_angle"], 180.0)
        self.assertAlmostEqual(diagnostics["turn_fatigue_applied"], expected_cost)

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

    def test_apply_restoration_settling_resets_momentum(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.momentum = 0.7

        diagnostics = apply_restoration(world, "SETTLING", night=True, shelter_role="inside")

        self.assertAlmostEqual(world.state.momentum, 0.0)
        self.assertAlmostEqual(diagnostics["momentum_before"], 0.7)
        self.assertAlmostEqual(diagnostics["momentum_after"], 0.0)
        self.assertTrue(diagnostics["momentum_reset"])

    def test_apply_restoration_resting_keeps_momentum_zero(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.momentum = 0.4

        diagnostics = apply_restoration(world, "RESTING", night=True, shelter_role="deep")

        self.assertAlmostEqual(world.state.momentum, 0.0)
        self.assertTrue(diagnostics["momentum_reset"])

    def test_apply_restoration_deep_sleep_keeps_momentum_zero(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.momentum = 0.4

        diagnostics = apply_restoration(world, "DEEP_SLEEP", night=True, shelter_role="deep")

        self.assertAlmostEqual(world.state.momentum, 0.0)
        self.assertTrue(diagnostics["momentum_reset"])

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
        world.state.momentum = 2.0
        clip_state(world)
        self.assertAlmostEqual(world.state.hunger, 1.0)
        self.assertAlmostEqual(world.state.fatigue, 1.0)
        self.assertAlmostEqual(world.state.sleep_debt, 1.0)
        self.assertAlmostEqual(world.state.health, 1.0)
        self.assertAlmostEqual(world.state.recent_pain, 1.0)
        self.assertAlmostEqual(world.state.recent_contact, 1.0)
        self.assertAlmostEqual(world.state.momentum, 1.0)

    def test_clip_state_clamps_values_below_zero(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.hunger = -0.5
        world.state.fatigue = -1.0
        world.state.sleep_debt = -0.2
        world.state.health = -0.1
        world.state.momentum = -0.1
        clip_state(world)
        self.assertAlmostEqual(world.state.hunger, 0.0)
        self.assertAlmostEqual(world.state.fatigue, 0.0)
        self.assertAlmostEqual(world.state.sleep_debt, 0.0)
        self.assertAlmostEqual(world.state.health, 0.0)
        self.assertAlmostEqual(world.state.momentum, 0.0)

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

    # --- tick_context parameter tests ---

    def _make_tick_context(self, world: SpiderWorld) -> TickContext:
        return TickContext(
            action_idx=0,
            intended_action="STAY",
            executed_action="STAY",
            motor_noise_applied=False,
            snapshot=TickSnapshot(
                tick=int(world.tick),
                spider_pos=world.spider_pos(),
                lizard_pos=world.lizard_pos(),
                was_on_shelter=False,
                prev_shelter_role="outside",
                prev_food_dist=5,
                prev_shelter_dist=5,
                prev_predator_dist=10,
                prev_predator_visible=False,
                night=False,
                rest_streak=0,
            ),
            reward_components={name: 0.0 for name in REWARD_COMPONENT_NAMES},
            info={"ate": False, "slept": False},
        )

    def test_apply_predator_contact_without_tick_context_no_event_logged(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info: dict = {}
        # Should not raise and should work without tick_context
        apply_predator_contact(world, reward_components, info)
        self.assertTrue(info.get("predator_contact"))

    def test_apply_predator_contact_with_tick_context_records_event(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info: dict = {}
        ctx = self._make_tick_context(world)
        apply_predator_contact(world, reward_components, info, tick_context=ctx)
        event_names = [e.name for e in ctx.event_log]
        self.assertIn("predator_contact", event_names)

    def test_apply_predator_contact_tick_context_event_has_required_fields(self) -> None:
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        world.state.health = 1.0
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info: dict = {}
        ctx = self._make_tick_context(world)
        apply_predator_contact(world, reward_components, info, tick_context=ctx)
        event = next(e for e in ctx.event_log if e.name == "predator_contact")
        self.assertIn("damage", event.payload)
        self.assertIn("health", event.payload)
        self.assertIn("recent_pain", event.payload)
        self.assertIn("recent_contact", event.payload)
        self.assertGreater(event.payload["damage"], 0.0)
        self.assertLess(event.payload["health"], 1.0)
        self.assertEqual(event.stage, "predator_contact")

    def test_apply_predator_contact_tick_context_damage_reflects_world_state(self) -> None:
        world = SpiderWorld(seed=11, lizard_move_interval=999999)
        world.reset(seed=11)
        world.state.health = 1.0
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info: dict = {}
        ctx = self._make_tick_context(world)
        apply_predator_contact(world, reward_components, info, tick_context=ctx)
        event = next(e for e in ctx.event_log if e.name == "predator_contact")
        self.assertAlmostEqual(event.payload["health"], round(float(world.state.health), 6))

    def test_resolve_autonomic_behaviors_feeding_records_event(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = world.food_positions[0]
        world.state.hunger = 0.9
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info = {"ate": False, "slept": False}
        ctx = self._make_tick_context(world)
        resolve_autonomic_behaviors(
            world,
            action_name="STAY",
            predator_threat=False,
            night=False,
            reward_components=reward_components,
            info=info,
            tick_context=ctx,
        )
        event_names = [e.name for e in ctx.event_log]
        self.assertIn("feeding", event_names)
        event = next(e for e in ctx.event_log if e.name == "feeding")
        self.assertIn("hunger_before", event.payload)
        self.assertIn("hunger_after", event.payload)
        self.assertIn("action", event.payload)
        self.assertAlmostEqual(event.payload["hunger_before"], 0.9, places=4)
        self.assertLess(event.payload["hunger_after"], 0.9)

    def test_resolve_autonomic_behaviors_off_shelter_records_sleep_reset_event(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        # Place spider in open area (not on shelter, not on food)
        for x in range(world.width):
            for y in range(world.height):
                if (
                    world.shelter_role_at((x, y)) == "outside"
                    and (x, y) not in world.food_positions
                ):
                    world.state.x, world.state.y = x, y
                    break
            else:
                continue
            break
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info = {"ate": False, "slept": False}
        ctx = self._make_tick_context(world)
        resolve_autonomic_behaviors(
            world,
            action_name="STAY",
            predator_threat=False,
            night=False,
            reward_components=reward_components,
            info=info,
            tick_context=ctx,
        )
        event_names = [e.name for e in ctx.event_log]
        self.assertIn("sleep_reset_off_shelter", event_names)
        event = next(e for e in ctx.event_log if e.name == "sleep_reset_off_shelter")
        self.assertIn("action", event.payload)
        self.assertEqual(event.stage, "autonomic")

    def test_resolve_autonomic_behaviors_rest_phase_records_event(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells available")
        world.state.x, world.state.y = sorted(deep_cells)[0]
        world.state.fatigue = 0.6
        world.state.sleep_debt = 0.5
        world.state.hunger = 0.1
        world.state.rest_streak = 0
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info = {"ate": False, "slept": False}
        ctx = self._make_tick_context(world)
        resolve_autonomic_behaviors(
            world,
            action_name="STAY",
            predator_threat=False,
            night=True,
            reward_components=reward_components,
            info=info,
            tick_context=ctx,
        )
        event_names = [e.name for e in ctx.event_log]
        self.assertIn("rest_phase", event_names)
        event = next(e for e in ctx.event_log if e.name == "rest_phase")
        self.assertIn("phase", event.payload)
        self.assertIn("rest_streak", event.payload)
        self.assertIn("sleep_drive", event.payload)
        self.assertIn("shelter_role", event.payload)
        self.assertIn("momentum_before", event.payload)
        self.assertIn("momentum_after", event.payload)
        self.assertIn("momentum_reset", event.payload)
        self.assertEqual(event.stage, "autonomic")

    def test_resolve_autonomic_behaviors_rest_phase_reports_momentum_reset(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells available")
        world.state.x, world.state.y = sorted(deep_cells)[0]
        world.state.fatigue = 0.6
        world.state.sleep_debt = 0.5
        world.state.hunger = 0.1
        world.state.momentum = 0.7
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info = {"ate": False, "slept": False}
        ctx = self._make_tick_context(world)

        resolve_autonomic_behaviors(
            world,
            action_name="STAY",
            predator_threat=False,
            night=True,
            reward_components=reward_components,
            info=info,
            tick_context=ctx,
        )

        event = next(e for e in ctx.event_log if e.name == "rest_phase")
        self.assertAlmostEqual(world.state.momentum, 0.0)
        self.assertAlmostEqual(event.payload["momentum_before"], 0.7)
        self.assertAlmostEqual(event.payload["momentum_after"], 0.0)
        self.assertTrue(event.payload["momentum_reset"])
        self.assertTrue(info["restoration"]["momentum_reset"])

    def test_resolve_autonomic_behaviors_rest_blocked_records_event(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells available")
        world.state.x, world.state.y = sorted(deep_cells)[0]
        world.state.fatigue = 0.1
        world.state.sleep_debt = 0.0
        world.state.hunger = 0.7  # Above hunger threshold for rest
        world.state.rest_streak = 0
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info = {"ate": False, "slept": False}
        ctx = self._make_tick_context(world)
        resolve_autonomic_behaviors(
            world,
            action_name="STAY",
            predator_threat=False,
            night=False,
            reward_components=reward_components,
            info=info,
            tick_context=ctx,
        )
        event_names = [e.name for e in ctx.event_log]
        self.assertIn("rest_blocked", event_names)
        event = next(e for e in ctx.event_log if e.name == "rest_blocked")
        self.assertIn("action", event.payload)
        self.assertIn("predator_threat", event.payload)
        self.assertIn("shelter_role", event.payload)
        self.assertIn("fatigue_before", event.payload)
        self.assertIn("sleep_debt", event.payload)
        self.assertIn("hunger", event.payload)
        self.assertEqual(event.stage, "autonomic")

    def test_resolve_autonomic_behaviors_without_tick_context_still_works(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = world.food_positions[0]
        world.state.hunger = 0.9
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info = {"ate": False, "slept": False}
        # No tick_context - should work exactly as before
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

    def test_resolve_autonomic_behaviors_predator_threat_blocks_rest(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells available")
        world.state.x, world.state.y = sorted(deep_cells)[0]
        world.state.fatigue = 0.6
        world.state.sleep_debt = 0.5
        world.state.hunger = 0.1
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        info = {"ate": False, "slept": False}
        ctx = self._make_tick_context(world)
        resolve_autonomic_behaviors(
            world,
            action_name="STAY",
            predator_threat=True,  # Predator threat prevents rest
            night=True,
            reward_components=reward_components,
            info=info,
            tick_context=ctx,
        )
        event_names = [e.name for e in ctx.event_log]
        # With predator threat, rest_phase should NOT be recorded, rest_blocked should be
        self.assertNotIn("rest_phase", event_names)
        self.assertIn("rest_blocked", event_names)
        blocked_event = next(e for e in ctx.event_log if e.name == "rest_blocked")
        self.assertTrue(blocked_event.payload["predator_threat"])
