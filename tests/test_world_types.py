import unittest

from spider_cortex_sim.world_types import MemorySlot, PerceptTrace, SpiderState, TickContext, TickEvent, TickSnapshot


class MemorySlotTest(unittest.TestCase):
    def test_memory_slot_default_values(self) -> None:
        slot = MemorySlot(target=None, age=0)
        self.assertIsNone(slot.target)
        self.assertEqual(slot.age, 0)

    def test_memory_slot_with_target(self) -> None:
        slot = MemorySlot(target=(3, 5), age=4)
        self.assertEqual(slot.target, (3, 5))
        self.assertEqual(slot.age, 4)

    def test_memory_slot_is_mutable(self) -> None:
        slot = MemorySlot(target=None, age=0)
        slot.target = (7, 2)
        slot.age = 10
        self.assertEqual(slot.target, (7, 2))
        self.assertEqual(slot.age, 10)

    def test_memory_slot_equality(self) -> None:
        a = MemorySlot(target=(1, 2), age=3)
        b = MemorySlot(target=(1, 2), age=3)
        self.assertEqual(a, b)

    def test_memory_slot_inequality_by_target(self) -> None:
        """
        Asserts that two MemorySlot instances with the same age but different targets compare as not equal.
        
        Creates two MemorySlot objects differing only by `target` and verifies inequality.
        """
        a = MemorySlot(target=(1, 2), age=3)
        b = MemorySlot(target=(4, 5), age=3)
        self.assertNotEqual(a, b)

    def test_memory_slot_inequality_by_age(self) -> None:
        a = MemorySlot(target=(1, 2), age=3)
        b = MemorySlot(target=(1, 2), age=7)
        self.assertNotEqual(a, b)

    def test_memory_slot_clear_target(self) -> None:
        slot = MemorySlot(target=(5, 5), age=2)
        slot.target = None
        slot.age = 0
        self.assertIsNone(slot.target)
        self.assertEqual(slot.age, 0)


class PerceptTraceTest(unittest.TestCase):
    def test_percept_trace_stores_fields(self) -> None:
        trace = PerceptTrace(target=(2, 5), age=3, certainty=0.75, heading_dx=1, heading_dy=0)
        self.assertEqual(trace.target, (2, 5))
        self.assertEqual(trace.age, 3)
        self.assertAlmostEqual(trace.certainty, 0.75)
        self.assertEqual(trace.heading_dx, 1)
        self.assertEqual(trace.heading_dy, 0)

    def test_percept_trace_none_target(self) -> None:
        trace = PerceptTrace(target=None, age=0, certainty=0.0)
        self.assertIsNone(trace.target)
        self.assertEqual(trace.age, 0)
        self.assertAlmostEqual(trace.certainty, 0.0)
        self.assertEqual(trace.heading_dx, 0)
        self.assertEqual(trace.heading_dy, 0)

    def test_percept_trace_is_mutable(self) -> None:
        trace = PerceptTrace(target=None, age=0, certainty=0.0)
        trace.target = (4, 7)
        trace.age = 2
        trace.certainty = 0.9
        trace.heading_dx = -1
        trace.heading_dy = 0
        self.assertEqual(trace.target, (4, 7))
        self.assertEqual(trace.age, 2)
        self.assertAlmostEqual(trace.certainty, 0.9)
        self.assertEqual(trace.heading_dx, -1)
        self.assertEqual(trace.heading_dy, 0)

    def test_percept_trace_equality(self) -> None:
        a = PerceptTrace(target=(1, 2), age=1, certainty=0.5)
        b = PerceptTrace(target=(1, 2), age=1, certainty=0.5)
        self.assertEqual(a, b)

    def test_percept_trace_inequality_by_age(self) -> None:
        a = PerceptTrace(target=(1, 2), age=1, certainty=0.5)
        b = PerceptTrace(target=(1, 2), age=2, certainty=0.5)
        self.assertNotEqual(a, b)

    def test_percept_trace_inequality_by_target(self) -> None:
        a = PerceptTrace(target=(1, 2), age=0, certainty=0.5)
        b = PerceptTrace(target=(3, 4), age=0, certainty=0.5)
        self.assertNotEqual(a, b)

    def test_percept_trace_inequality_by_certainty(self) -> None:
        a = PerceptTrace(target=(1, 2), age=0, certainty=0.5)
        b = PerceptTrace(target=(1, 2), age=0, certainty=0.8)
        self.assertNotEqual(a, b)

    def test_percept_trace_inequality_by_heading(self) -> None:
        a = PerceptTrace(target=(1, 2), age=0, certainty=0.5, heading_dx=1, heading_dy=0)
        b = PerceptTrace(target=(1, 2), age=0, certainty=0.5, heading_dx=0, heading_dy=1)
        self.assertNotEqual(a, b)

    def test_percept_trace_inequality_none_vs_coordinate(self) -> None:
        a = PerceptTrace(target=None, age=0, certainty=0.0)
        b = PerceptTrace(target=(0, 0), age=0, certainty=0.0)
        self.assertNotEqual(a, b)

    def test_percept_trace_target_coordinates_are_integers(self) -> None:
        trace = PerceptTrace(target=(10, 20), age=0, certainty=1.0)
        self.assertIsInstance(trace.target[0], int)
        self.assertIsInstance(trace.target[1], int)


class SpiderStateTest(unittest.TestCase):
    def _empty_trace(self) -> PerceptTrace:
        return PerceptTrace(target=None, age=0, certainty=0.0)

    def _make_state(self) -> SpiderState:
        """
        Constructs a SpiderState populated with a fixed set of values used by the tests.
        
        The returned state has position x=3, y=4; physiological metrics (hunger=0.5, fatigue=0.2, sleep_debt=0.1, health=1.0, recent_pain=0.0, recent_contact=0.0); sleep_phase set to "AWAKE" with rest_streak 0; last_action "STAY" with last_move_dx and last_move_dy set to 0; reward fields last_reward and total_reward set to 0.0; all counters (food_eaten, sleep_events, shelter_entries, alert_events, predator_contacts, predator_sightings, predator_escapes, steps_alive) set to 0; and four independent MemorySlot instances (food_memory, predator_memory, shelter_memory, escape_memory) each initialized with target=None and age=0.
        
        Returns:
            SpiderState: A SpiderState instance initialized with the values described above.
        """
        return SpiderState(
            x=3,
            y=4,
            hunger=0.5,
            fatigue=0.2,
            sleep_debt=0.1,
            health=1.0,
            recent_pain=0.0,
            recent_contact=0.0,
            sleep_phase="AWAKE",
            rest_streak=0,
            last_reward=0.0,
            total_reward=0.0,
            food_eaten=0,
            sleep_events=0,
            shelter_entries=0,
            alert_events=0,
            predator_contacts=0,
            predator_sightings=0,
            predator_escapes=0,
            steps_alive=0,
            last_action="STAY",
            last_move_dx=0,
            last_move_dy=0,
            heading_dx=1,
            heading_dy=0,
            food_memory=MemorySlot(target=None, age=0),
            predator_memory=MemorySlot(target=None, age=0),
            shelter_memory=MemorySlot(target=None, age=0),
            escape_memory=MemorySlot(target=None, age=0),
            food_trace=self._empty_trace(),
            shelter_trace=self._empty_trace(),
            predator_trace=self._empty_trace(),
        )

    def test_spider_state_initial_position(self) -> None:
        """
        Verify that a SpiderState constructed by _make_state initializes its x and y coordinates to (3, 4).
        
        Asserts that state.x equals 3 and state.y equals 4.
        """
        state = self._make_state()
        self.assertEqual(state.x, 3)
        self.assertEqual(state.y, 4)

    def test_spider_state_initial_physiology(self) -> None:
        state = self._make_state()
        self.assertAlmostEqual(state.hunger, 0.5)
        self.assertAlmostEqual(state.fatigue, 0.2)
        self.assertAlmostEqual(state.health, 1.0)
        self.assertAlmostEqual(state.recent_pain, 0.0)

    def test_spider_state_counters_start_at_zero(self) -> None:
        state = self._make_state()
        self.assertEqual(state.food_eaten, 0)
        self.assertEqual(state.sleep_events, 0)
        self.assertEqual(state.shelter_entries, 0)
        self.assertEqual(state.alert_events, 0)
        self.assertEqual(state.predator_contacts, 0)
        self.assertEqual(state.predator_sightings, 0)
        self.assertEqual(state.predator_escapes, 0)
        self.assertEqual(state.steps_alive, 0)

    def test_spider_state_sleep_phase_awake(self) -> None:
        state = self._make_state()
        self.assertEqual(state.sleep_phase, "AWAKE")
        self.assertEqual(state.rest_streak, 0)

    def test_spider_state_last_action_and_movement(self) -> None:
        state = self._make_state()
        self.assertEqual(state.last_action, "STAY")
        self.assertEqual(state.last_move_dx, 0)
        self.assertEqual(state.last_move_dy, 0)
        self.assertEqual(state.heading_dx, 1)
        self.assertEqual(state.heading_dy, 0)

    def test_spider_state_memory_slots_cleared(self) -> None:
        state = self._make_state()
        self.assertIsNone(state.food_memory.target)
        self.assertIsNone(state.predator_memory.target)
        self.assertIsNone(state.shelter_memory.target)
        self.assertIsNone(state.escape_memory.target)
        self.assertIsNone(state.food_trace.target)
        self.assertIsNone(state.shelter_trace.target)
        self.assertIsNone(state.predator_trace.target)

    def test_spider_state_is_mutable(self) -> None:
        state = self._make_state()
        state.hunger = 0.9
        state.food_eaten = 3
        self.assertAlmostEqual(state.hunger, 0.9)
        self.assertEqual(state.food_eaten, 3)

    def test_spider_state_memory_slot_independence(self) -> None:
        state = self._make_state()
        state.food_memory.target = (2, 2)
        state.food_memory.age = 5
        self.assertIsNone(state.predator_memory.target)
        self.assertEqual(state.predator_memory.age, 0)

    def test_percept_trace_equality(self) -> None:
        self.assertEqual(
            PerceptTrace(target=(1, 2), age=2, certainty=0.5),
            PerceptTrace(target=(1, 2), age=2, certainty=0.5),
        )

    def test_heading_fields_stored_and_readable(self) -> None:
        state = self._make_state()
        self.assertEqual(state.heading_dx, 1)
        self.assertEqual(state.heading_dy, 0)

    def test_momentum_defaults_to_zero(self) -> None:
        state = self._make_state()
        self.assertAlmostEqual(state.momentum, 0.0)

    def test_momentum_field_is_mutable(self) -> None:
        state = self._make_state()
        state.momentum = 0.75
        self.assertAlmostEqual(state.momentum, 0.75)

    def test_heading_fields_are_mutable(self) -> None:
        state = self._make_state()
        state.heading_dx = -1
        state.heading_dy = 1
        self.assertEqual(state.heading_dx, -1)
        self.assertEqual(state.heading_dy, 1)

    def test_percept_traces_independence(self) -> None:
        """Mutating one trace does not affect the others."""
        state = self._make_state()
        state.food_trace.target = (3, 3)
        state.food_trace.age = 1
        self.assertIsNone(state.shelter_trace.target)
        self.assertIsNone(state.predator_trace.target)

    def test_percept_traces_accept_coordinate_targets(self) -> None:
        state = self._make_state()
        state.food_trace = PerceptTrace(target=(5, 6), age=0, certainty=0.8)
        state.shelter_trace = PerceptTrace(target=(1, 1), age=2, certainty=0.3)
        state.predator_trace = PerceptTrace(target=(9, 9), age=1, certainty=0.6)
        self.assertEqual(state.food_trace.target, (5, 6))
        self.assertEqual(state.shelter_trace.certainty, 0.3)
        self.assertEqual(state.predator_trace.age, 1)


class TickSnapshotTest(unittest.TestCase):
    def _make_snapshot(self, **overrides: object) -> TickSnapshot:
        defaults = {
            "tick": 5,
            "spider_pos": (3, 4),
            "lizard_pos": (10, 10),
            "was_on_shelter": False,
            "prev_shelter_role": "outside",
            "prev_food_dist": 6,
            "prev_shelter_dist": 8,
            "prev_predator_dist": 12,
            "prev_predator_visible": False,
            "night": False,
            "rest_streak": 0,
            "momentum": 0.25,
        }
        defaults.update(overrides)
        return TickSnapshot(**defaults)

    def test_tick_snapshot_fields_stored_correctly(self) -> None:
        snap = self._make_snapshot()
        self.assertEqual(snap.tick, 5)
        self.assertEqual(snap.spider_pos, (3, 4))
        self.assertEqual(snap.lizard_pos, (10, 10))
        self.assertFalse(snap.was_on_shelter)
        self.assertEqual(snap.prev_shelter_role, "outside")
        self.assertEqual(snap.prev_food_dist, 6)
        self.assertEqual(snap.prev_shelter_dist, 8)
        self.assertEqual(snap.prev_predator_dist, 12)
        self.assertFalse(snap.prev_predator_visible)
        self.assertFalse(snap.night)
        self.assertEqual(snap.rest_streak, 0)
        self.assertAlmostEqual(snap.momentum, 0.25)

    def test_tick_snapshot_is_frozen(self) -> None:
        snap = self._make_snapshot()
        with self.assertRaises((AttributeError, TypeError)):
            snap.tick = 99  # type: ignore[misc]

    def test_to_payload_returns_all_expected_keys(self) -> None:
        snap = self._make_snapshot()
        payload = snap.to_payload()
        expected_keys = {
            "tick", "spider_pos", "lizard_pos", "was_on_shelter",
            "prev_shelter_role", "prev_food_dist", "prev_shelter_dist",
            "prev_predator_dist", "prev_predator_visible", "night", "rest_streak",
            "momentum",
        }
        self.assertEqual(set(payload.keys()), expected_keys)

    def test_to_payload_spider_pos_is_list(self) -> None:
        snap = self._make_snapshot(spider_pos=(3, 7))
        payload = snap.to_payload()
        self.assertIsInstance(payload["spider_pos"], list)
        self.assertEqual(payload["spider_pos"], [3, 7])

    def test_to_payload_lizard_pos_is_list(self) -> None:
        snap = self._make_snapshot(lizard_pos=(11, 2))
        payload = snap.to_payload()
        self.assertIsInstance(payload["lizard_pos"], list)
        self.assertEqual(payload["lizard_pos"], [11, 2])

    def test_to_payload_tick_is_int(self) -> None:
        snap = self._make_snapshot(tick=42)
        payload = snap.to_payload()
        self.assertIsInstance(payload["tick"], int)
        self.assertEqual(payload["tick"], 42)

    def test_to_payload_bool_fields_are_bool(self) -> None:
        snap = self._make_snapshot(was_on_shelter=True, prev_predator_visible=True, night=True)
        payload = snap.to_payload()
        self.assertIsInstance(payload["was_on_shelter"], bool)
        self.assertTrue(payload["was_on_shelter"])
        self.assertIsInstance(payload["prev_predator_visible"], bool)
        self.assertTrue(payload["prev_predator_visible"])
        self.assertIsInstance(payload["night"], bool)
        self.assertTrue(payload["night"])

    def test_to_payload_distance_fields_are_int(self) -> None:
        snap = self._make_snapshot(prev_food_dist=3, prev_shelter_dist=7, prev_predator_dist=15)
        payload = snap.to_payload()
        self.assertIsInstance(payload["prev_food_dist"], int)
        self.assertIsInstance(payload["prev_shelter_dist"], int)
        self.assertIsInstance(payload["prev_predator_dist"], int)
        self.assertEqual(payload["prev_food_dist"], 3)
        self.assertEqual(payload["prev_shelter_dist"], 7)
        self.assertEqual(payload["prev_predator_dist"], 15)

    def test_to_payload_preserves_shelter_role_string(self) -> None:
        snap = self._make_snapshot(prev_shelter_role="deep")
        payload = snap.to_payload()
        self.assertEqual(payload["prev_shelter_role"], "deep")

    def test_to_payload_rest_streak_is_int(self) -> None:
        snap = self._make_snapshot(rest_streak=3)
        payload = snap.to_payload()
        self.assertIsInstance(payload["rest_streak"], int)
        self.assertEqual(payload["rest_streak"], 3)

    def test_to_payload_momentum_is_float(self) -> None:
        snap = self._make_snapshot(momentum=0.625)
        payload = snap.to_payload()
        self.assertIsInstance(payload["momentum"], float)
        self.assertAlmostEqual(payload["momentum"], 0.625)

    def test_to_payload_momentum_is_clamped(self) -> None:
        high = self._make_snapshot(momentum=1.5).to_payload()
        low = self._make_snapshot(momentum=-0.5).to_payload()

        self.assertAlmostEqual(high["momentum"], 1.0)
        self.assertAlmostEqual(low["momentum"], 0.0)


class TickEventTest(unittest.TestCase):
    def test_tick_event_fields_stored_correctly(self) -> None:
        event = TickEvent(stage="action", name="movement_applied", payload={"moved": True})
        self.assertEqual(event.stage, "action")
        self.assertEqual(event.name, "movement_applied")
        self.assertEqual(event.payload, {"moved": True})

    def test_tick_event_default_payload_is_empty_dict(self) -> None:
        event = TickEvent(stage="pre_tick", name="snapshot")
        self.assertEqual(event.payload, {})

    def test_tick_event_is_frozen(self) -> None:
        event = TickEvent(stage="action", name="test_event")
        with self.assertRaises((AttributeError, TypeError)):
            event.stage = "other"  # type: ignore[misc]

    def test_to_payload_returns_stage_name_and_payload(self) -> None:
        event = TickEvent(stage="reward", name="distance_deltas", payload={"food": 1, "shelter": -1})
        result = event.to_payload()
        self.assertEqual(result["stage"], "reward")
        self.assertEqual(result["name"], "distance_deltas")
        self.assertIn("payload", result)
        self.assertEqual(result["payload"]["food"], 1)
        self.assertEqual(result["payload"]["shelter"], -1)

    def test_to_payload_returns_copy_of_payload(self) -> None:
        original_payload = {"x": 10}
        event = TickEvent(stage="test", name="test", payload=original_payload)
        result = event.to_payload()
        result["payload"]["x"] = 99
        self.assertEqual(original_payload["x"], 10)

    def test_to_payload_keys_are_stage_name_payload(self) -> None:
        event = TickEvent(stage="memory", name="memory_refreshed")
        result = event.to_payload()
        self.assertEqual(set(result.keys()), {"stage", "name", "payload"})

    def test_tick_event_with_nested_payload(self) -> None:
        event = TickEvent(stage="pre_tick", name="snapshot", payload={"spider_pos": [3, 4], "night": False})
        result = event.to_payload()
        self.assertEqual(result["payload"]["spider_pos"], [3, 4])
        self.assertFalse(result["payload"]["night"])


class TickContextTest(unittest.TestCase):
    def _make_snapshot(self) -> TickSnapshot:
        return TickSnapshot(
            tick=0,
            spider_pos=(2, 2),
            lizard_pos=(9, 9),
            was_on_shelter=False,
            prev_shelter_role="outside",
            prev_food_dist=5,
            prev_shelter_dist=7,
            prev_predator_dist=11,
            prev_predator_visible=False,
            night=False,
            rest_streak=0,
        )

    def _make_context(self, **overrides: object) -> TickContext:
        defaults: dict = {
            "action_idx": 0,
            "intended_action": "STAY",
            "executed_action": "STAY",
            "motor_noise_applied": False,
            "snapshot": self._make_snapshot(),
            "reward_components": {"food_progress": 0.0, "feeding": 0.0},
        }
        defaults.update(overrides)
        return TickContext(**defaults)

    def test_tick_context_fields_stored_correctly(self) -> None:
        ctx = self._make_context(action_idx=2, intended_action="MOVE_UP", executed_action="MOVE_UP")
        self.assertEqual(ctx.action_idx, 2)
        self.assertEqual(ctx.intended_action, "MOVE_UP")
        self.assertEqual(ctx.executed_action, "MOVE_UP")
        self.assertFalse(ctx.motor_noise_applied)
        self.assertAlmostEqual(ctx.execution_difficulty, 0.0)
        self.assertEqual(ctx.execution_components, {})
        self.assertEqual(ctx.motor_slip_info, {})

    def test_tick_context_defaults(self) -> None:
        ctx = self._make_context()
        self.assertEqual(ctx.event_log, [])
        self.assertFalse(ctx.moved)
        self.assertEqual(ctx.terrain_now, "")
        self.assertFalse(ctx.predator_threat)
        self.assertFalse(ctx.interrupted_rest)
        self.assertFalse(ctx.exposed_at_night)
        self.assertFalse(ctx.predator_moved)
        self.assertFalse(ctx.predator_escape)
        self.assertFalse(ctx.predator_visible_now)
        self.assertFalse(ctx.done)
        self.assertAlmostEqual(ctx.reward, 0.0)

    def test_tick_context_info_defaults_to_empty_dict(self) -> None:
        ctx = self._make_context()
        self.assertIsInstance(ctx.info, dict)
        self.assertEqual(ctx.info, {})

    def test_tick_context_event_log_defaults_to_empty_list(self) -> None:
        ctx = self._make_context()
        self.assertIsInstance(ctx.event_log, list)
        self.assertEqual(len(ctx.event_log), 0)

    def test_record_event_appends_tick_event(self) -> None:
        ctx = self._make_context()
        ctx.record_event("action", "movement_applied", moved=True)
        self.assertEqual(len(ctx.event_log), 1)
        event = ctx.event_log[0]
        self.assertIsInstance(event, TickEvent)
        self.assertEqual(event.stage, "action")
        self.assertEqual(event.name, "movement_applied")
        self.assertEqual(event.payload["moved"], True)

    def test_record_event_multiple_appends_in_order(self) -> None:
        ctx = self._make_context()
        ctx.record_event("pre_tick", "snapshot", tick=0)
        ctx.record_event("action", "action_resolved", action_index=0)
        ctx.record_event("reward", "distance_deltas", food=1)
        self.assertEqual(len(ctx.event_log), 3)
        self.assertEqual(ctx.event_log[0].stage, "pre_tick")
        self.assertEqual(ctx.event_log[1].stage, "action")
        self.assertEqual(ctx.event_log[2].stage, "reward")

    def test_record_event_with_no_kwargs_has_empty_payload(self) -> None:
        ctx = self._make_context()
        ctx.record_event("predator_contact", "contact_check")
        self.assertEqual(ctx.event_log[0].payload, {})

    def test_record_event_kwargs_stored_correctly(self) -> None:
        ctx = self._make_context()
        ctx.record_event("autonomic", "feeding", hunger_before=0.7, hunger_after=0.14, action="STAY")
        event = ctx.event_log[0]
        self.assertAlmostEqual(event.payload["hunger_before"], 0.7)
        self.assertAlmostEqual(event.payload["hunger_after"], 0.14)
        self.assertEqual(event.payload["action"], "STAY")

    def test_serialized_event_log_returns_list_of_dicts(self) -> None:
        ctx = self._make_context()
        ctx.record_event("action", "movement_applied", moved=False)
        ctx.record_event("reward", "distance_deltas", food=0, shelter=1)
        result = ctx.serialized_event_log()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for item in result:
            self.assertIsInstance(item, dict)
            self.assertIn("stage", item)
            self.assertIn("name", item)
            self.assertIn("payload", item)

    def test_serialized_event_log_preserves_order(self) -> None:
        ctx = self._make_context()
        ctx.record_event("pre_tick", "snapshot")
        ctx.record_event("action", "movement_applied")
        ctx.record_event("memory", "memory_refreshed")
        result = ctx.serialized_event_log()
        self.assertEqual(result[0]["stage"], "pre_tick")
        self.assertEqual(result[1]["stage"], "action")
        self.assertEqual(result[2]["stage"], "memory")

    def test_serialized_event_log_empty_when_no_events(self) -> None:
        ctx = self._make_context()
        self.assertEqual(ctx.serialized_event_log(), [])

    def test_serialized_event_log_payload_content(self) -> None:
        ctx = self._make_context()
        ctx.record_event("predator_contact", "predator_contact", damage=0.12, health=0.88)
        result = ctx.serialized_event_log()
        self.assertAlmostEqual(result[0]["payload"]["damage"], 0.12)
        self.assertAlmostEqual(result[0]["payload"]["health"], 0.88)

    def test_tick_context_is_mutable(self) -> None:
        ctx = self._make_context()
        ctx.moved = True
        ctx.terrain_now = "open"
        ctx.reward = 0.5
        ctx.done = True
        self.assertTrue(ctx.moved)
        self.assertEqual(ctx.terrain_now, "open")
        self.assertAlmostEqual(ctx.reward, 0.5)
        self.assertTrue(ctx.done)

    def test_tick_context_snapshot_reference(self) -> None:
        snap = self._make_snapshot()
        ctx = self._make_context(snapshot=snap)
        self.assertIs(ctx.snapshot, snap)
        self.assertEqual(ctx.snapshot.tick, 0)
        self.assertEqual(ctx.snapshot.spider_pos, (2, 2))

    def test_separate_contexts_have_independent_event_logs(self) -> None:
        ctx1 = self._make_context()
        ctx2 = self._make_context()
        ctx1.record_event("action", "movement_applied")
        self.assertEqual(len(ctx1.event_log), 1)
        self.assertEqual(len(ctx2.event_log), 0)


if __name__ == "__main__":
    unittest.main()
