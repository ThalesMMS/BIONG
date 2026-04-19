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

class SpiderWorldEventLogTest(SpiderWorldTestBase):
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
            "momentum",
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
        self.assertIn("momentum_before", payload)
        self.assertIn("momentum_after", payload)
        self.assertIn("turn_angle", payload)
        self.assertIn("turn_fatigue_applied", payload)
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
