import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spider_cortex_sim.memory import (
    MEMORY_TTLS,
    age_or_clear_memory,
    empty_memory_slot,
    escape_memory_target,
    memory_leakage_audit,
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

    def test_escape_memory_target_does_not_check_walkability(self) -> None:
        cases = (
            {
                "name": "positive_dx",
                "seed": 3,
                "world_kwargs": {"lizard_move_interval": 999999},
                "spider_pos": (2, 2),
                "last_move": (1, 0),
                "expected": (3, 2),
            },
            {
                "name": "negative_dx",
                "seed": 11,
                "world_kwargs": {"lizard_move_interval": 999999},
                "spider_pos": (3, 3),
                "last_move": (-1, 0),
                "expected": (2, 3),
            },
        )
        for case in cases:
            with self.subTest(case=case["name"]):
                world = SpiderWorld(seed=case["seed"], **case["world_kwargs"])
                world.reset(seed=case["seed"])
                world.state.x, world.state.y = case["spider_pos"]
                world.state.last_move_dx, world.state.last_move_dy = case["last_move"]
                world.is_walkable = lambda _target: False
                self.assertEqual(escape_memory_target(world), case["expected"])

    def test_refresh_memory_records_visible_targets(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 2, 2
        world.state.heading_dx = 1
        world.state.heading_dy = 1
        world.food_positions = [(2, 3)]
        world.lizard.x, world.lizard.y = 4, 2
        refresh_memory(world, initial=True)
        self.assertEqual(world.state.food_memory.target, (2, 3))
        self.assertEqual(world.state.predator_memory.target, (4, 2))

    def test_refresh_memory_uses_perception_derived_predator_position(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 2, 2
        world.lizard.x, world.lizard.y = 4, 2
        world.state.predator_memory.target = None
        perceived_position = (6, 2)
        predator_view = SimpleNamespace(visible=1.0, position=perceived_position)
        with patch("spider_cortex_sim.memory.predator_visible_to_spider", return_value=predator_view):
            refresh_memory(world, initial=True)
        self.assertEqual(world.state.predator_memory.target, perceived_position)
        self.assertNotEqual(world.state.predator_memory.target, world.lizard_pos())

    def test_refresh_memory_records_visible_shelter_cell(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 2, 2
        visible_shelter_cell = (world.state.x + 1, world.state.y)
        far_edge_cell = (world.width - 1, world.height - 1)
        world.lizard.x, world.lizard.y = far_edge_cell
        world.shelter_entrance_cells = {visible_shelter_cell}
        world.shelter_interior_cells = set()
        world.shelter_deep_cells = {far_edge_cell}
        world.shelter_cells = {visible_shelter_cell, far_edge_cell}
        refresh_memory(world, initial=True)
        self.assertEqual(world.state.shelter_memory.target, visible_shelter_cell)

    def test_refresh_memory_leaves_shelter_memory_when_no_shelter_perceived(self) -> None:
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 0, 0
        far_edge_cell = (world.width - 1, world.height - 1)
        world.lizard.x, world.lizard.y = far_edge_cell
        world.state.shelter_memory.target = (3, 2)
        world.state.shelter_memory.age = 4
        world.shelter_entrance_cells = set()
        world.shelter_interior_cells = set()
        world.shelter_deep_cells = {far_edge_cell}
        world.shelter_cells = {far_edge_cell}
        refresh_memory(world, initial=True)
        self.assertEqual(world.state.shelter_memory.target, (3, 2))
        self.assertEqual(world.state.shelter_memory.age, 4)

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
        Verifies that recent contact records predator proximity at the spider's own position.
        
        Sets up a world where the predator is not visible and confirms that refresh_memory(initial=True)
        records the contact event locally rather than writing the lizard's exact coordinates.
        """
        world = SpiderWorld(seed=11, lizard_move_interval=999999)
        world.reset(seed=11)
        world.state.x, world.state.y = 3, 3
        world.lizard.x, world.lizard.y = 8, 8
        world.state.recent_contact = 1.0
        refresh_memory(world, initial=True)
        self.assertEqual(world.state.predator_memory.target, (3, 3))

    def test_memory_leakage_audit_marks_shelter_memory_as_low_risk(self) -> None:
        audit = memory_leakage_audit()
        self.assertEqual(audit["shelter_memory"]["risk"], "low")
        self.assertEqual(audit["food_memory"]["classification"], "plausible_memory")

    def test_memory_leakage_audit_all_memory_slots_are_plausible_low_risk(self) -> None:
        """
        All explicit slots remain plausible memory under the perception-grounded model.
        """
        audit = memory_leakage_audit()
        for slot_name in ("food_memory", "predator_memory", "shelter_memory", "escape_memory"):
            self.assertEqual(audit[slot_name]["classification"], "plausible_memory")
            self.assertEqual(audit[slot_name]["risk"], "low")

    def test_memory_leakage_audit_returns_all_four_expected_keys(self) -> None:
        audit = memory_leakage_audit()
        expected_keys = {"food_memory", "predator_memory", "shelter_memory", "escape_memory"}
        self.assertEqual(set(audit.keys()), expected_keys)

    def test_memory_leakage_audit_returns_deep_copy_not_reference(self) -> None:
        audit1 = memory_leakage_audit()
        audit2 = memory_leakage_audit()
        audit1["shelter_memory"]["risk"] = "mutated"
        self.assertEqual(audit2["shelter_memory"]["risk"], "low")

    def test_memory_leakage_audit_each_entry_has_required_fields(self) -> None:
        audit = memory_leakage_audit()
        required_fields = {
            "classification",
            "risk",
            "source",
            "update_rule",
            "allowed_sources",
            "rationale",
            "notes",
        }
        for slot_name, metadata in audit.items():
            for field in required_fields:
                self.assertIn(
                    field,
                    metadata,
                    f"Memory slot {slot_name!r} missing field {field!r}",
                )

    def test_memory_leakage_audit_allowed_sources_and_rationales_are_documented(self) -> None:
        audit = memory_leakage_audit()
        for slot_name, metadata in audit.items():
            allowed_sources = metadata["allowed_sources"]
            rationale = metadata["rationale"]
            self.assertIsInstance(allowed_sources, tuple)
            self.assertGreater(
                len(allowed_sources),
                0,
                f"Memory slot {slot_name!r} must document at least one allowed source",
            )
            self.assertIsInstance(rationale, str)
            self.assertGreater(
                len(rationale.strip()),
                0,
                f"Memory slot {slot_name!r} must document why it is plausible memory",
            )

    def test_memory_leakage_audit_predator_memory_is_plausible_low_risk(self) -> None:
        audit = memory_leakage_audit()
        self.assertEqual(audit["predator_memory"]["classification"], "plausible_memory")
        self.assertEqual(audit["predator_memory"]["risk"], "low")

    def test_memory_leakage_audit_escape_memory_is_plausible_low_risk(self) -> None:
        audit = memory_leakage_audit()
        self.assertEqual(audit["escape_memory"]["classification"], "plausible_memory")
        self.assertEqual(audit["escape_memory"]["risk"], "low")

    def test_memory_leakage_audit_risk_values_are_valid(self) -> None:
        audit = memory_leakage_audit()
        valid_risk_levels = {"low", "medium", "high"}
        for slot_name, metadata in audit.items():
            self.assertIn(
                metadata["risk"],
                valid_risk_levels,
                f"Memory slot {slot_name!r} has unexpected risk level {metadata['risk']!r}",
            )

    def test_memory_leakage_audit_world_owned_slots_count(self) -> None:
        """
        Perception-grounded explicit memory should leave no world-owned slots.
        """
        audit = memory_leakage_audit()
        world_owned = [
            name for name, data in audit.items()
            if data["classification"] == "world_owned_memory"
        ]
        self.assertEqual(len(world_owned), 0)

    # --- Additional tests for PR changes ---

    def test_escape_memory_target_negative_last_move(self) -> None:
        """escape_memory_target moves in the negative direction when last_move is negative."""
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.x, world.state.y = 5, 5
        world.state.last_move_dx = -1
        world.state.last_move_dy = -1
        result = escape_memory_target(world)
        self.assertEqual(result, (4, 4))

    def test_escape_memory_target_large_magnitude_dx_normalized(self) -> None:
        """escape_memory_target normalizes any non-zero last_move to ±1."""
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.x, world.state.y = 2, 2
        world.state.last_move_dx = 5
        world.state.last_move_dy = -3
        result = escape_memory_target(world)
        self.assertEqual(result, (3, 1))

    def test_escape_memory_target_clamps_at_right_bottom_boundary(self) -> None:
        """escape_memory_target clamps to (width-1, height-1) at the bottom-right corner."""
        world = SpiderWorld(seed=3, lizard_move_interval=999999)
        world.reset(seed=3)
        world.state.x = world.width - 1
        world.state.y = world.height - 1
        world.state.last_move_dx = 1
        world.state.last_move_dy = 1
        result = escape_memory_target(world)
        self.assertEqual(result[0], world.width - 1)
        self.assertEqual(result[1], world.height - 1)

    def test_refresh_memory_does_not_set_predator_memory_when_not_visible_and_no_contact(self) -> None:
        """refresh_memory leaves predator_memory unchanged when predator is not visible and no contact."""
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 0, 0
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.state.recent_contact = 0.0
        world.state.predator_memory.target = None
        refresh_memory(world, initial=True)
        self.assertIsNone(world.state.predator_memory.target)

    def test_refresh_memory_visible_branch_takes_priority_over_contact(self) -> None:
        """When predator is visible (>0.5) AND recent_contact>0, the visual position is stored, not spider pos."""
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 2, 2
        world.lizard.x, world.lizard.y = 4, 2
        world.state.recent_contact = 1.0
        perceived_position = (4, 2)
        predator_view = SimpleNamespace(visible=1.0, position=perceived_position)
        with patch("spider_cortex_sim.memory.predator_visible_to_spider", return_value=predator_view):
            refresh_memory(world, initial=True)
        # Should use perceived_position, not spider pos (2, 2)
        self.assertEqual(world.state.predator_memory.target, perceived_position)
        self.assertNotEqual(world.state.predator_memory.target, world.spider_pos())

    def test_refresh_memory_predator_visible_exactly_at_boundary_does_not_use_visual_branch(self) -> None:
        """predator_view.visible == 0.5 is NOT > 0.5, so contact branch is used when recent_contact > 0."""
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 3, 3
        perceived_position = (world.width - 1, world.height - 1)
        world.lizard.x, world.lizard.y = perceived_position
        world.state.recent_contact = 1.0
        predator_view = SimpleNamespace(visible=0.5, position=perceived_position)
        with patch("spider_cortex_sim.memory.predator_visible_to_spider", return_value=predator_view):
            refresh_memory(world, initial=True)
        # visible == 0.5 is not > 0.5, so falls through to contact branch → spider pos
        self.assertEqual(world.state.predator_memory.target, (3, 3))

    def test_refresh_memory_shelter_memory_not_set_when_visible_object_returns_none_position(self) -> None:
        """When shelter_view.position is None and visible <= 0.5, shelter_memory is not updated."""
        world = SpiderWorld(seed=5, lizard_move_interval=999999)
        world.reset(seed=5)
        world.state.x, world.state.y = 0, 0
        world.state.shelter_memory.target = (5, 5)
        world.state.shelter_memory.age = 3
        # Move all shelter cells to the far corner, out of range
        world.shelter_entrance_cells = set()
        world.shelter_interior_cells = set()
        world.shelter_deep_cells = set()
        world.shelter_cells = set()
        refresh_memory(world, initial=True)
        # Shelter memory unchanged because no shelter was visible
        self.assertEqual(world.state.shelter_memory.target, (5, 5))
        self.assertEqual(world.state.shelter_memory.age, 3)

    def test_refresh_memory_escape_not_set_without_predator_escape_flag(self) -> None:
        """escape_memory is not written when predator_escape=False (default)."""
        world = SpiderWorld(seed=7, lizard_move_interval=999999)
        world.reset(seed=7)
        world.state.x, world.state.y = 2, 2
        world.state.last_move_dx = 1
        world.state.last_move_dy = 0
        world.state.escape_memory.target = None
        refresh_memory(world, initial=True)
        self.assertIsNone(world.state.escape_memory.target)

    def test_memory_leakage_audit_predator_update_rule_references_visual_perception(self) -> None:
        """predator_memory update_rule mentions visual perception as source."""
        audit = memory_leakage_audit()
        rule = audit["predator_memory"]["update_rule"]
        self.assertIn("predator_visible_to_spider", rule)

    def test_memory_leakage_audit_shelter_update_rule_references_visible_object(self) -> None:
        """shelter_memory update_rule mentions visible_object as source."""
        audit = memory_leakage_audit()
        rule = audit["shelter_memory"]["update_rule"]
        self.assertIn("visible_object", rule)

    def test_memory_leakage_audit_escape_notes_no_walkability(self) -> None:
        """escape_memory notes must mention absence of walkability check."""
        audit = memory_leakage_audit()
        notes = audit["escape_memory"]["notes"]
        self.assertIn("walkability", notes)

    def test_memory_leakage_audit_predator_notes_mentions_contact(self) -> None:
        """predator_memory notes must mention contact events."""
        audit = memory_leakage_audit()
        notes = audit["predator_memory"]["notes"]
        self.assertIn("contact", notes)

    def test_memory_leakage_audit_shelter_notes_mentions_perception(self) -> None:
        """shelter_memory notes must distinguish local perception from world-selected target."""
        audit = memory_leakage_audit()
        notes = audit["shelter_memory"]["notes"]
        self.assertIn("perception", notes)
