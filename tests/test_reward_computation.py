"""Focused reward tests grouped by reward subpackage responsibility."""

from __future__ import annotations

import math
import unittest

from spider_cortex_sim.claim_tests import canonical_claim_tests
from spider_cortex_sim.maps import NARROW
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.reward.audit import (
    REWARD_COMPONENT_AUDIT,
    _roadmap_status_for_profile,
    reward_component_audit,
    reward_profile_audit,
    shaping_disposition_summary,
)
from spider_cortex_sim.reward.computation import (
    apply_action_and_terrain_effects,
    apply_pressure_penalties,
    apply_progress_and_event_rewards,
    compute_predator_threat,
    copy_reward_components,
    empty_reward_components,
    reward_total,
)
from spider_cortex_sim.reward.profiles import (
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    REWARD_COMPONENT_NAMES,
    REWARD_PROFILES,
)
from spider_cortex_sim.reward.shaping import (
    DISPOSITION_EVIDENCE_CRITERIA,
    SCENARIO_AUSTERE_REQUIREMENTS,
    SHAPING_DISPOSITIONS,
    SHAPING_GAP_POLICY,
    SHAPING_REDUCTION_ROADMAP,
    shaping_reduction_roadmap,
    validate_gap_policy,
)
from spider_cortex_sim.scenarios import CAPABILITY_PROBE_SCENARIOS, SCENARIOS
from spider_cortex_sim.world import SpiderWorld
from spider_cortex_sim.world_types import TickContext, TickSnapshot

from tests.fixtures.reward_computation import RewardComputationModuleTestBase

class RewardComputationCoreTest(RewardComputationModuleTestBase):
    def test_empty_reward_components_matches_declared_names(self) -> None:
        reward_components = empty_reward_components()
        self.assertEqual(set(reward_components.keys()), set(REWARD_COMPONENT_NAMES))
        self.assertTrue(all(value == 0.0 for value in reward_components.values()))

    def test_reward_total_and_copy_round_trip(self) -> None:
        reward_components = empty_reward_components()
        reward_components["feeding"] = 1.2
        reward_components["action_cost"] = -0.2
        copied = copy_reward_components(reward_components)
        self.assertEqual(copied, reward_components)
        self.assertAlmostEqual(reward_total(reward_components), 1.0)

    def test_apply_action_and_terrain_effects_updates_state_and_costs(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999, map_template="entrance_funnel")
        world.reset(seed=1)
        world.state.x, world.state.y = 5, world.height // 2
        reward_components = empty_reward_components()
        hunger_before = world.state.hunger
        fatigue_before = world.state.fatigue
        terrain_now = apply_action_and_terrain_effects(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            reward_components=reward_components,
        )
        self.assertEqual(terrain_now, world.terrain_at(world.spider_pos()))
        self.assertLess(reward_components["action_cost"], 0.0)
        self.assertGreater(world.state.hunger, hunger_before)
        self.assertGreater(world.state.fatigue, fatigue_before)

    def test_compute_predator_threat_detects_visible_predator(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.x, world.state.y = 2, 2
        world.lizard.x, world.lizard.y = 4, 2
        self.assertTrue(
            compute_predator_threat(
                world,
                prev_predator_visible=True,
                prev_predator_dist=2,
            )
        )

    def test_apply_pressure_penalties_uses_state_levels(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.hunger = 0.8
        world.state.fatigue = 0.6
        world.state.sleep_debt = 0.4
        reward_components = empty_reward_components()
        apply_pressure_penalties(world, reward_components)
        self.assertLess(reward_components["hunger_pressure"], 0.0)
        self.assertLess(reward_components["fatigue_pressure"], 0.0)
        self.assertLess(reward_components["sleep_debt_pressure"], 0.0)

    def test_profile_with_reward_updates_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self._profile_with_reward_updates(not_a_real_reward_key=1.0)
        self.assertIn("not_a_real_reward_key", str(ctx.exception))

    def test_copy_reward_components_is_independent(self) -> None:
        original = empty_reward_components()
        original["feeding"] = 2.5
        copied = copy_reward_components(original)
        self.assertAlmostEqual(copied["feeding"], 2.5)
        copied["feeding"] = 9.9
        self.assertAlmostEqual(original["feeding"], 2.5)

    def test_copy_reward_components_fills_missing_keys_with_zero(self) -> None:
        copied = copy_reward_components({"feeding": 2.5})

        self.assertEqual(set(copied), set(REWARD_COMPONENT_NAMES))
        self.assertAlmostEqual(copied["feeding"], 2.5)
        self.assertAlmostEqual(copied["action_cost"], 0.0)

    def test_reward_total_handles_negative_values(self) -> None:
        components = empty_reward_components()
        components["action_cost"] = -0.5
        components["predator_contact"] = -2.0
        components["feeding"] = 3.0
        total = reward_total(components)
        self.assertAlmostEqual(total, 0.5)

    def test_reward_total_all_zeros(self) -> None:
        components = empty_reward_components()
        self.assertAlmostEqual(reward_total(components), 0.0)

    def test_apply_action_and_terrain_effects_stay_cost(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        reward_components = empty_reward_components()
        apply_action_and_terrain_effects(
            world,
            action_name="STAY",
            moved=False,
            reward_components=reward_components,
        )
        expected_cost = -world.reward_config["action_cost_stay"]
        self.assertAlmostEqual(reward_components["action_cost"], expected_cost)

    def test_apply_action_and_terrain_effects_move_costs_more_than_stay(self) -> None:
        world_stay = SpiderWorld(seed=1, lizard_move_interval=999999)
        world_stay.reset(seed=1)
        rc_stay = empty_reward_components()
        apply_action_and_terrain_effects(world_stay, action_name="STAY", moved=False, reward_components=rc_stay)

        world_move = SpiderWorld(seed=1, lizard_move_interval=999999)
        world_move.reset(seed=1)
        rc_move = empty_reward_components()
        apply_action_and_terrain_effects(world_move, action_name="MOVE_RIGHT", moved=True, reward_components=rc_move)

        self.assertLess(rc_move["action_cost"], rc_stay["action_cost"])

    def test_compute_predator_threat_false_when_safe(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 1, 1
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.recent_contact = 0.0
        world.state.recent_pain = 0.0
        result = compute_predator_threat(
            world,
            prev_predator_visible=False,
            prev_predator_dist=10,
        )
        self.assertFalse(result)

    def test_compute_predator_threat_uses_operational_profile_thresholds(self) -> None:
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(predator_threat_smell_threshold=0.0),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 1, 1
        world.lizard.x, world.lizard.y = 4, 1
        world.state.recent_contact = 0.0
        world.state.recent_pain = 0.0
        result = compute_predator_threat(
            world,
            prev_predator_visible=False,
            prev_predator_dist=10,
        )
        self.assertTrue(result)

    def test_compute_predator_threat_uses_deterministic_smell_signal(self) -> None:
        noise_profile = NoiseConfig(
            name="olfactory_drop_test",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 1.0, "direction_jitter": 1.0},
            motor={"action_flip_prob": 0.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            noise_profile=noise_profile,
            operational_profile=self._profile_with_reward_updates(predator_threat_smell_threshold=0.0),
        )
        for episode_seed in (1, 7, 29, 101):
            world.reset(seed=episode_seed)
            world.state.x, world.state.y = 1, 1
            world.lizard.x, world.lizard.y = 4, 1
            world.state.recent_contact = 0.0
            world.state.recent_pain = 0.0

            result = compute_predator_threat(
                world,
                prev_predator_visible=False,
                prev_predator_dist=10,
            )

            self.assertTrue(result, f"Threat detection changed under episode_seed={episode_seed}")

    def test_apply_progress_and_event_rewards_food_approach(self) -> None:
        """
        Tests that moving the spider closer to a food target increases the `food_progress` reward and records distance changes.
        
        Sets up a world with a food item, moves the spider one step nearer, invokes apply_progress_and_event_rewards, and asserts that `info` contains a "distance_deltas" entry and that `reward_components["food_progress"]` is greater than 0.0.
        """
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.food_positions = [(5, 3)]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.hunger = 0.7

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        prev_food_dist = world.manhattan(world.spider_pos(), world.food_positions[0])
        world.state.x = 4
        context = self._tick_context(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=prev_food_dist,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertIn("distance_deltas", context.info)
        self.assertGreater(context.reward_components["food_progress"], 0.0)

    def test_apply_progress_food_gate_uses_operational_profile_threshold(self) -> None:
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(food_progress_hunger_threshold=0.95),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.food_positions = [(5, 3)]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.hunger = 0.9

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        prev_food_dist = world.manhattan(world.spider_pos(), world.food_positions[0])
        world.state.x = 4
        context = self._tick_context(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=prev_food_dist,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertEqual(context.reward_components["food_progress"], 0.0)

    def test_apply_progress_and_event_rewards_shelter_entry_bonus(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        entrance_cells = list(world.shelter_entrance_cells)
        if not entrance_cells:
            self.skipTest("No entrance cells")
        world.state.x, world.state.y = sorted(entrance_cells)[0]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.shelter_entries = 0
        terrain_now = world.terrain_at(world.spider_pos())

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=3,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertGreater(context.reward_components["shelter_entry"], 0.0)
        self.assertEqual(world.state.shelter_entries, 1)

    def test_apply_progress_and_event_rewards_day_exploration_reward(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.state.hunger = 0.5
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.tick = 0

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertGreaterEqual(context.reward_components["day_exploration"], 0.0)

    def test_apply_progress_and_event_rewards_escape_bonus(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells")
        world.state.x, world.state.y = sorted(deep_cells)[0]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.recent_contact = 0.0
        terrain_now = world.terrain_at(world.spider_pos())

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=3,
            prev_predator_visible=True,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertGreater(context.reward_components["predator_escape"], 0.0)
        self.assertTrue(context.predator_escape)

    def test_apply_progress_and_event_rewards_updates_predator_visible_now(self) -> None:
        """
        Verifies that apply_progress_and_event_rewards sets the TickContext.predator_visible_now flag according to current predator proximity.
        
        Sets up two scenarios—one with the predator nearby and one with the predator far away—and asserts that predator_visible_now is True in the nearby case and False in the far case.
        """
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 2, 2
        world.state.heading_dx = 1
        world.state.heading_dy = 0
        world.lizard.x, world.lizard.y = 3, 2
        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        prev_predator_dist = world.manhattan(prev_spider_pos, prev_lizard_pos)

        visible_context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=world.terrain_at(world.spider_pos()),
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=prev_predator_dist,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=visible_context)
        self.assertTrue(visible_context.predator_visible_now)

        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        far_lizard_pos = world.lizard_pos()
        far_context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=world.terrain_at(world.spider_pos()),
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=world.manhattan(prev_spider_pos, far_lizard_pos),
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=far_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=far_context)
        self.assertFalse(far_context.predator_visible_now)

    def test_narrow_predator_risk_is_clamped_when_predator_overlaps_spider(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        for y in range(world.height):
            for x in range(world.width):
                cell = (x, y)
                if world.is_walkable(cell) and cell not in world.shelter_cells:
                    world.state.x, world.state.y = cell
                    world.lizard.x, world.lizard.y = cell
                    break
            else:
                continue
            break
        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=NARROW,
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=0,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )

        apply_progress_and_event_rewards(world, tick_context=context)

        self.assertAlmostEqual(
            context.reward_components["terrain_cost"],
            -world.reward_config["narrow_predator_risk"],
        )
        event = next(
            event for event in context.event_log if event.name == "narrow_predator_risk"
        )
        self.assertEqual(event.payload["risk"], 1.0)

    def test_narrow_predator_risk_zero_radius_does_not_apply_penalty(self) -> None:
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(
                narrow_predator_risk_max_distance=0.0,
            ),
        )
        world.reset(seed=1)
        for y in range(world.height):
            for x in range(world.width):
                cell = (x, y)
                if world.is_walkable(cell) and cell not in world.shelter_cells:
                    world.state.x, world.state.y = cell
                    world.lizard.x, world.lizard.y = cell
                    break
            else:
                continue
            break
        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=NARROW,
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=0,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )

        apply_progress_and_event_rewards(world, tick_context=context)

        self.assertAlmostEqual(context.reward_components["terrain_cost"], 0.0)
        event = next(
            event for event in context.event_log if event.name == "narrow_predator_risk"
        )
        self.assertEqual(event.payload["risk"], 0.0)

    def test_shelter_progress_gate_uses_operational_profile_fatigue_threshold(self) -> None:
        # High fatigue threshold in profile → shelter progress not awarded for moderate fatigue
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(shelter_progress_fatigue_threshold=0.99),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.fatigue = 0.5
        world.state.sleep_debt = 0.0

        shelter_cells = list(world.shelter_deep_cells or world.shelter_cells)
        if not shelter_cells:
            self.skipTest("No shelter cells")
        original_pos = world.spider_pos()
        _, prev_shelter_dist = world.nearest(shelter_cells)
        candidate_moves = {
            "MOVE_LEFT": (-1, 0),
            "MOVE_RIGHT": (1, 0),
            "MOVE_UP": (0, -1),
            "MOVE_DOWN": (0, 1),
        }
        chosen_action = None
        new_shelter_dist = prev_shelter_dist
        for action_name, (dx, dy) in candidate_moves.items():
            new_x = original_pos[0] + dx
            new_y = original_pos[1] + dy
            if not (0 <= new_x < world.width and 0 <= new_y < world.height):
                continue
            world.state.x, world.state.y = new_x, new_y
            _, candidate_shelter_dist = world.nearest(shelter_cells)
            if candidate_shelter_dist < prev_shelter_dist:
                chosen_action = action_name
                new_shelter_dist = candidate_shelter_dist
                break
        if chosen_action is None:
            self.fail("Expected at least one one-step move that reduces shelter distance.")
        self.assertGreater(prev_shelter_dist, new_shelter_dist)

        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name=chosen_action,
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=prev_shelter_dist,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=original_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        self.assertEqual(context.reward_components["shelter_progress"], 0.0)

    def test_day_exploration_hunger_threshold_uses_operational_profile(self) -> None:
        # Raise hunger threshold so that moderate hunger is treated as "calm"
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(day_exploration_hunger_threshold=0.90),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.state.hunger = 0.6  # above default 0.35, below custom 0.90
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.tick = 0

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context_hungry = self._tick_context(
            world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context_hungry)
        # At hunger=0.6 with threshold=0.90, should get calm exploration (not hungry bonus)
        default_world = SpiderWorld(seed=1, lizard_move_interval=999999)
        default_world.reset(seed=1)
        default_world.state.x, default_world.state.y = 3, 3
        default_world.state.hunger = 0.6
        default_world.lizard.x, default_world.lizard.y = default_world.width - 1, default_world.height - 1
        default_world.tick = 0

        default_prev_spider_pos = default_world.spider_pos()
        default_prev_lizard_pos = default_world.lizard_pos()
        context_default = self._tick_context(
            default_world,
            action_name="MOVE_RIGHT",
            moved=True,
            night=False,
            terrain_now="open",
            was_on_shelter=False,
            prev_food_dist=5,
            prev_shelter_dist=5,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=default_prev_spider_pos,
            prev_lizard_pos=default_prev_lizard_pos,
        )
        apply_progress_and_event_rewards(default_world, tick_context=context_default)
        # Custom world has raised threshold so gets calm bonus; default world gets hungry bonus
        self.assertGreater(
            context_default.reward_components["day_exploration"],
            context_hungry.reward_components["day_exploration"],
        )
