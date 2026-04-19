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

class RewardComputationEventsTest(RewardComputationModuleTestBase):
    def test_predator_threat_distance_threshold_uses_operational_profile(self) -> None:
        # Raise distance threshold so threat is detected from further away
        world = SpiderWorld(
            seed=1,
            lizard_move_interval=999999,
            operational_profile=self._profile_with_reward_updates(predator_threat_distance_threshold=10.0),
        )
        world.reset(seed=1)
        world.state.x, world.state.y = 1, 1
        world.lizard.x, world.lizard.y = world.width - 2, world.height - 2
        world.state.recent_contact = 0.0
        world.state.recent_pain = 0.0

        result = compute_predator_threat(
            world,
            prev_predator_visible=False,
            prev_predator_dist=10,
        )
        self.assertTrue(result)

    def test_world_stores_operational_profile(self) -> None:
        profile = self._profile_with_reward_updates(predator_threat_smell_threshold=0.99)
        world = SpiderWorld(seed=1, lizard_move_interval=999999, operational_profile=profile)
        self.assertIs(world.operational_profile, profile)

    def test_apply_progress_and_event_rewards_records_distance_deltas_event(self) -> None:
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
        event_names = [e.name for e in context.event_log]
        self.assertIn("distance_deltas", event_names)
        event = next(e for e in context.event_log if e.name == "distance_deltas")
        self.assertIn("food", event.payload)
        self.assertIn("shelter", event.payload)
        self.assertIn("predator", event.payload)
        self.assertIsInstance(event.payload["food"], int)
        self.assertIsInstance(event.payload["shelter"], int)
        self.assertIsInstance(event.payload["predator"], int)

    def test_apply_progress_and_event_rewards_records_distance_deltas_in_stage_reward(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
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
        # distance_deltas is the primary event recorded by this function directly
        event_stages = [e.stage for e in context.event_log]
        self.assertTrue(all(s == "reward" for s in event_stages), "All events from apply_progress_and_event_rewards should be in 'reward' stage")
        event_names = [e.name for e in context.event_log]
        self.assertIn("distance_deltas", event_names)

    def test_apply_progress_and_event_rewards_sets_predator_visible_now_in_context(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
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
        self.assertFalse(context.predator_visible_now)
        apply_progress_and_event_rewards(world, tick_context=context)
        # predator_visible_now should be a bool
        self.assertIsInstance(context.predator_visible_now, bool)

    def test_apply_progress_and_event_rewards_shelter_entry_records_event(self) -> None:
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
            was_on_shelter=False,  # Coming from outside, entering shelter
            prev_food_dist=5,
            prev_shelter_dist=3,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=context)
        if context.reward_components["shelter_entry"] > 0.0:
            event_names = [e.name for e in context.event_log]
            self.assertIn("shelter_entry", event_names)
            entry_event = next(e for e in context.event_log if e.name == "shelter_entry")
            self.assertIn("shelter_role", entry_event.payload)

    def test_apply_progress_and_event_rewards_predator_escape_sets_context_flag(self) -> None:
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
        # context.predator_escape should be set by the function
        self.assertIsInstance(context.predator_escape, bool)
        if context.predator_escape:
            event_names = [e.name for e in context.event_log]
            self.assertIn("predator_escape", event_names)

    def test_apply_progress_and_event_rewards_predator_escape_bonus_is_one_shot_per_threat_episode(self) -> None:
        """
        Verify that a predator escape reward and event are granted only once per predator-threat episode.
        
        Sets up a scenario where the spider is in a deep shelter with recent predator contact, invokes progress/event reward processing twice with identical tick contexts, and asserts that the first invocation awards `predator_escape` (and logs a `predator_escape` event) while the second does not, and that `world.state.predator_escapes` equals 1.
        """
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        deep_cells = list(world.shelter_deep_cells)
        if not deep_cells:
            self.skipTest("No deep shelter cells")
        world.state.x, world.state.y = sorted(deep_cells)[0]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.recent_contact = 1.0
        terrain_now = world.terrain_at(world.spider_pos())

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        first_context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=True,
            prev_food_dist=5,
            prev_shelter_dist=0,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=first_context)

        second_context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
            night=False,
            terrain_now=terrain_now,
            was_on_shelter=True,
            prev_food_dist=5,
            prev_shelter_dist=0,
            prev_predator_dist=10,
            prev_predator_visible=False,
            prev_spider_pos=prev_spider_pos,
            prev_lizard_pos=prev_lizard_pos,
        )
        apply_progress_and_event_rewards(world, tick_context=second_context)

        self.assertTrue(first_context.predator_escape)
        self.assertFalse(second_context.predator_escape)
        self.assertEqual(world.state.predator_escapes, 1)
        self.assertIn(
            "predator_escape",
            [event.name for event in first_context.event_log],
        )
        self.assertNotIn(
            "predator_escape",
            [event.name for event in second_context.event_log],
        )

    def test_apply_progress_and_event_rewards_info_distance_deltas_populated(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1

        prev_spider_pos = world.spider_pos()
        prev_lizard_pos = world.lizard_pos()
        context = self._tick_context(
            world,
            action_name="STAY",
            moved=False,
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
        self.assertIn("distance_deltas", context.info)
        deltas = context.info["distance_deltas"]
        self.assertIn("food", deltas)
        self.assertIn("shelter", deltas)
        self.assertIn("predator", deltas)
