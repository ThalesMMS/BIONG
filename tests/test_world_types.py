import unittest

from spider_cortex_sim.world_types import MemorySlot, SpiderState


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


class SpiderStateTest(unittest.TestCase):
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
            food_memory=MemorySlot(target=None, age=0),
            predator_memory=MemorySlot(target=None, age=0),
            shelter_memory=MemorySlot(target=None, age=0),
            escape_memory=MemorySlot(target=None, age=0),
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

    def test_spider_state_memory_slots_cleared(self) -> None:
        state = self._make_state()
        self.assertIsNone(state.food_memory.target)
        self.assertIsNone(state.predator_memory.target)
        self.assertIsNone(state.shelter_memory.target)
        self.assertIsNone(state.escape_memory.target)

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


if __name__ == "__main__":
    unittest.main()