import unittest

from spider_cortex_sim.memory import (
    MEMORY_TTLS,
    age_or_clear_memory,
    empty_memory_slot,
    escape_memory_target,
    memory_vector,
    refresh_memory,
    set_memory,
)
from spider_cortex_sim.world import SpiderWorld
from spider_cortex_sim.world_types import MemorySlot


class MemoryModuleTest(unittest.TestCase):
    def test_empty_memory_slot_starts_cleared(self) -> None:
        slot = empty_memory_slot()
        self.assertIsNone(slot.target)
        self.assertEqual(slot.age, 0)

    def test_memory_vector_encodes_direction_and_age(self) -> None:
        world = SpiderWorld(seed=1, lizard_move_interval=999999)
        world.reset(seed=1)
        world.state.x, world.state.y = 2, 2
        dx, dy, age = memory_vector(world, MemorySlot(target=(4, 3), age=6), ttl_name="food")
        self.assertGreater(dx, 0.0)
        self.assertGreater(dy, 0.0)
        self.assertAlmostEqual(age, 6 / MEMORY_TTLS["food"])

    def test_age_or_clear_memory_expires_slots(self) -> None:
        slot = MemorySlot(target=(1, 1), age=MEMORY_TTLS["escape"])
        age_or_clear_memory(slot, ttl_name="escape")
        self.assertIsNone(slot.target)
        self.assertEqual(slot.age, 0)

    def test_escape_memory_target_uses_last_motion(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.x, world.state.y = 2, 2
        world.state.last_move_dx = 1
        world.state.last_move_dy = 0
        self.assertEqual(escape_memory_target(world), (3, 2))

    def test_refresh_memory_records_visible_targets(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 2, 2
        world.food_positions = [(2, 3)]
        world.lizard.x, world.lizard.y = 4, 2
        refresh_memory(world, initial=True)
        self.assertEqual(world.state.food_memory.target, (2, 3))
        self.assertEqual(world.state.predator_memory.target, (4, 2))

    def test_refresh_memory_records_escape_after_flag(self) -> None:
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        world.state.x, world.state.y = 2, 2
        world.state.last_move_dx = 1
        world.state.last_move_dy = 0
        refresh_memory(world, predator_escape=True, initial=True)
        self.assertEqual(world.state.escape_memory.target, (3, 2))

    def test_set_memory_sets_target_and_resets_age(self) -> None:
        slot = MemorySlot(target=None, age=5)
        set_memory(slot, (4, 7))
        self.assertEqual(slot.target, (4, 7))
        self.assertEqual(slot.age, 0)

    def test_set_memory_ignores_none_target(self) -> None:
        slot = MemorySlot(target=(2, 3), age=2)
        set_memory(slot, None)
        self.assertEqual(slot.target, (2, 3))
        self.assertEqual(slot.age, 2)

    def test_memory_vector_returns_default_when_slot_is_none(self) -> None:
        world = SpiderWorld(seed=11, lizard_move_interval=999999)
        world.reset(seed=11)
        slot = MemorySlot(target=None, age=0)
        dx, dy, age = memory_vector(world, slot, ttl_name="food")
        self.assertAlmostEqual(dx, 0.0)
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(age, 1.0)

    def test_memory_vector_returns_default_when_age_exceeds_ttl(self) -> None:
        world = SpiderWorld(seed=11, lizard_move_interval=999999)
        world.reset(seed=11)
        ttl = MEMORY_TTLS["food"]
        slot = MemorySlot(target=(2, 2), age=ttl + 1)
        dx, dy, age = memory_vector(world, slot, ttl_name="food")
        self.assertAlmostEqual(dx, 0.0)
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(age, 1.0)

    def test_age_or_clear_memory_increments_age_when_not_expired(self) -> None:
        slot = MemorySlot(target=(1, 1), age=2)
        ttl = MEMORY_TTLS["food"]
        age_or_clear_memory(slot, ttl_name="food")
        self.assertEqual(slot.age, 3)
        self.assertIsNotNone(slot.target)

    def test_age_or_clear_memory_no_op_for_none_target(self) -> None:
        slot = MemorySlot(target=None, age=0)
        age_or_clear_memory(slot, ttl_name="predator")
        self.assertIsNone(slot.target)
        self.assertEqual(slot.age, 0)

    def test_memory_ttls_contain_expected_keys(self) -> None:
        expected_keys = {"food", "predator", "shelter", "escape"}
        self.assertEqual(set(MEMORY_TTLS.keys()), expected_keys)
        for key, ttl in MEMORY_TTLS.items():
            self.assertGreater(ttl, 0, f"TTL for {key} should be positive")

    def test_escape_memory_target_no_movement_returns_current_pos(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.x, world.state.y = 2, 2
        world.state.last_move_dx = 0
        world.state.last_move_dy = 0
        result = escape_memory_target(world)
        self.assertEqual(result, (2, 2))

    def test_escape_memory_target_clamps_to_grid_bounds(self) -> None:
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.x, world.state.y = 0, 0
        world.state.last_move_dx = -1
        world.state.last_move_dy = -1
        result = escape_memory_target(world)
        self.assertGreaterEqual(result[0], 0)
        self.assertGreaterEqual(result[1], 0)

    def test_refresh_memory_ages_slots_on_non_initial_call(self) -> None:
        world = SpiderWorld(seed=9, lizard_move_interval=999999)
        world.reset(seed=9)
        world.state.x, world.state.y = 2, 2
        world.state.food_memory.target = (5, 5)
        world.state.food_memory.age = 3
        world.food_positions = [(world.width - 1, world.height - 1)]
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        refresh_memory(world, initial=False)
        self.assertEqual(world.state.food_memory.age, 4)

    def test_refresh_memory_sets_predator_memory_on_recent_contact(self) -> None:
        """
        Verifies that calling refresh_memory with recent contact updates the predator memory target to the lizard's position.
        
        Sets up a world where the agent has recent contact and confirms that refresh_memory(initial=True) records the predator at the lizard's coordinates.
        """
        world = SpiderWorld(seed=11, lizard_move_interval=999999)
        world.reset(seed=11)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = 8, 8
        world.state.recent_contact = 1.0
        refresh_memory(world, initial=True)
        self.assertEqual(world.state.predator_memory.target, (8, 8))