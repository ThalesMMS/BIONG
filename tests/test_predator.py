import unittest
from unittest.mock import patch

from spider_cortex_sim.predator import (
    DEFAULT_LIZARD_PROFILE,
    OLFACTORY_HUNTER_PROFILE,
    VISUAL_HUNTER_PROFILE,
    LizardState,
    PredatorProfile,
)
from spider_cortex_sim.world import SpiderWorld


class PredatorProfileTest(unittest.TestCase):
    def _outside_pair(
        self,
        world: SpiderWorld,
        *,
        distance: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Finds a pair of distinct outside walkable cells (spider position, lizard position) at a given Manhattan distance.
        
        Searches the map for a spider position on any cell whose shelter role is "outside" and a lizard position on any outside cell that is lizard-walkable, returning the first pair whose Manhattan distance equals `distance`.
        
        Parameters:
            world (SpiderWorld): The test world to search.
            distance (int): Desired Manhattan distance between the spider and lizard positions.
        
        Returns:
            tuple[tuple[int, int], tuple[int, int]]: (spider_pos, lizard_pos) coordinates meeting the distance requirement.
        
        Raises:
            AssertionError: If no matching pair is found (the test is failed).
        """
        outside_walkable = [
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
            if world.is_walkable((x, y)) and world.shelter_role_at((x, y)) == "outside"
        ]
        lizard_walkable = [
            pos for pos in outside_walkable if world.is_lizard_walkable(pos)
        ]
        for spider_pos in outside_walkable:
            for lizard_pos in lizard_walkable:
                if world.manhattan(spider_pos, lizard_pos) == distance:
                    return spider_pos, lizard_pos
        self.fail(f"Could not find outside spider/lizard pair at distance {distance}.")

    def _far_shelter_anchor(self, world: SpiderWorld) -> tuple[int, int]:
        """
        Finds a shelter cell that lies more than one Manhattan unit away from any lizard-walkable cell and from any shelter entrance.
        
        Parameters:
            world (SpiderWorld): World instance to search for shelter cells.
        
        Returns:
            tuple[int, int]: Coordinates of the first shelter cell that is at least distance 2 from every lizard-walkable cell and every shelter entrance.
        """
        lizard_walkable = [
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
            if world.is_lizard_walkable((x, y))
        ]
        entrances = sorted(world.shelter_entrance_cells)
        candidates = [
            pos
            for pos in sorted(world.shelter_cells)
            if min(world.manhattan(pos, probe) for probe in lizard_walkable) > 1
            and min(world.manhattan(pos, entrance) for entrance in entrances) > 1
        ]
        if not candidates:
            self.fail("Could not find a shelter cell outside the short profile target radius.")
        return candidates[0]

    def test_default_lizard_state_profile_falls_back_to_world_move_interval(self) -> None:
        world = SpiderWorld(seed=11, lizard_move_interval=7)
        world.reset(seed=11)
        world.lizard = LizardState(x=world.lizard.x, y=world.lizard.y)

        self.assertIsNone(world.lizard.profile)
        world.tick = 6
        self.assertFalse(world.predator_controller._can_move_this_tick(world))
        world.tick = 7
        self.assertTrue(world.predator_controller._can_move_this_tick(world))

    def test_reset_default_profile_uses_builtin_move_interval(self) -> None:
        world = SpiderWorld(seed=12, lizard_move_interval=7)
        world.reset(seed=12)

        self.assertEqual(world.lizard.profile, DEFAULT_LIZARD_PROFILE)
        world.tick = 1
        self.assertFalse(world.predator_controller._can_move_this_tick(world))
        world.tick = 2
        self.assertTrue(world.predator_controller._can_move_this_tick(world))

    def test_none_profile_falls_back_to_world_move_interval(self) -> None:
        """
        Verify that when the lizard's profile is `None`, the predator controller uses the world's `lizard_move_interval` to determine movement timing.
        
        The test creates a world with lizard_move_interval set to 5, clears the lizard profile, and asserts that movement is not allowed at tick 4 but is allowed at tick 5.
        """
        world = SpiderWorld(seed=13, lizard_move_interval=5)
        world.reset(seed=13)
        world.lizard.profile = None

        world.tick = 4
        self.assertFalse(world.predator_controller._can_move_this_tick(world))
        world.tick = 5
        self.assertTrue(world.predator_controller._can_move_this_tick(world))

    def test_explicit_profile_overrides_world_move_interval(self) -> None:
        world = SpiderWorld(seed=17, lizard_move_interval=999999)
        world.reset(seed=17)
        world.lizard.profile = VISUAL_HUNTER_PROFILE

        world.tick = 1
        self.assertFalse(world.predator_controller._can_move_this_tick(world))
        world.tick = 2
        self.assertTrue(world.predator_controller._can_move_this_tick(world))

    def test_update_uses_olfactory_profile_when_visual_line_of_sight_fails(self) -> None:
        world = SpiderWorld(seed=19)
        world.reset(seed=19)
        spider_pos, lizard_pos = self._outside_pair(world, distance=2)
        world.state.x, world.state.y = spider_pos
        world.lizard = LizardState(
            x=lizard_pos[0],
            y=lizard_pos[1],
            mode="PATROL",
            profile=OLFACTORY_HUNTER_PROFILE,
        )

        with patch("spider_cortex_sim.perception.has_line_of_sight", return_value=False):
            world.predator_controller.update(world)

        self.assertEqual(world.lizard.mode, "ORIENT")

    def test_update_keeps_visual_profile_patrolling_without_line_of_sight(self) -> None:
        """
        Verifies that a lizard using the visual-hunter profile remains in PATROL mode when line-of-sight to the spider is blocked.
        
        Sets up a world with the spider and lizard two Manhattan tiles apart, assigns the lizard the VISUAL_HUNTER_PROFILE in PATROL mode, forces perception.has_line_of_sight to return False, runs the predator controller update, and asserts the lizard's mode is still "PATROL".
        """
        world = SpiderWorld(seed=23)
        world.reset(seed=23)
        spider_pos, lizard_pos = self._outside_pair(world, distance=2)
        world.state.x, world.state.y = spider_pos
        world.lizard = LizardState(
            x=lizard_pos[0],
            y=lizard_pos[1],
            mode="PATROL",
            profile=VISUAL_HUNTER_PROFILE,
        )

        with patch("spider_cortex_sim.perception.has_line_of_sight", return_value=False):
            world.predator_controller.update(world)

        self.assertEqual(world.lizard.mode, "PATROL")

    def test_prime_targets_respects_explicit_profile_range(self) -> None:
        """
        Verify that _prime_targets applies an explicitly configured short-range PredatorProfile by setting the lizard's last known spider to the provided anchor and leaving both wait and investigate targets unset.
        """
        world = SpiderWorld(seed=29, map_template="central_burrow")
        world.reset(seed=29)
        anchor = self._far_shelter_anchor(world)
        world.lizard.profile = PredatorProfile(
            name="short_probe",
            vision_range=1,
            smell_range=1,
            detection_style="visual",
            move_interval=2,
            detection_threshold=0.45,
        )

        world.predator_controller._prime_targets(world, anchor)

        self.assertEqual(world.lizard.last_known_spider, anchor)
        self.assertIsNone(world.lizard.wait_target)
        self.assertIsNone(world.lizard.investigate_target)

    def test_investigate_fallback_respects_explicit_profile_range(self) -> None:
        """
        Verify that the INVESTIGATE fallback reuses the explicit profile radius when reseeding its probe target.
        """
        world = SpiderWorld(seed=30, map_template="central_burrow")
        world.reset(seed=30)
        anchor = self._far_shelter_anchor(world)
        world.tick = 2
        world.lizard.profile = PredatorProfile(
            name="short_probe",
            vision_range=1,
            smell_range=1,
            detection_style="visual",
            move_interval=2,
            detection_threshold=0.45,
        )
        world.lizard.mode = "INVESTIGATE"
        world.lizard.last_known_spider = anchor
        world.lizard.investigate_target = None
        world.lizard.wait_target = None
        world.lizard.recover_ticks = 0

        moved = world.predator_controller.update(world)

        self.assertFalse(moved)
        self.assertEqual(world.lizard.last_known_spider, anchor)
        self.assertIsNone(world.lizard.investigate_target)
        self.assertEqual(world.lizard.mode, "PATROL")

    def test_reset_with_multiple_predators_spawns_distinct_profiles_and_controllers(self) -> None:
        """
        Verify that resetting the world with multiple predator profiles creates distinct predators and controllers with correct indices, binds each predator to the provided profile object, and places predators at unique positions separated by at least the minimum required Manhattan distance from the spider and from each other.
        
        The test resets a SpiderWorld with two explicit predator profiles and asserts:
        - world.predator_count equals 2 and two predator controllers exist.
        - predator controllers are assigned consecutive predator_index values (0 and 1).
        - each predator's .profile is the exact profile object passed to reset.
        - predator spawn positions are unique.
        - each predator is at least min_dist Manhattan distance from the spider and from the other predator.
        """
        world = SpiderWorld(seed=31, map_template="central_burrow")
        world.reset(
            seed=31,
            predator_profiles=[VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )

        min_dist = max(3, min(world.width, world.height) // 3)
        positions = world.predator_positions()

        self.assertEqual(world.predator_count, 2)
        self.assertEqual(len(world.predator_controllers), 2)
        self.assertEqual(world.predator_controller.predator_index, 0)
        self.assertEqual(world.predator_controllers[1].predator_index, 1)
        self.assertIs(world.get_predator(0).profile, VISUAL_HUNTER_PROFILE)
        self.assertIs(world.get_predator(1).profile, OLFACTORY_HUNTER_PROFILE)
        self.assertEqual(len(positions), len(set(positions)))
        for pos in positions:
            self.assertGreaterEqual(world.manhattan(world.spider_pos(), pos), min_dist)
        self.assertGreaterEqual(world.manhattan(positions[0], positions[1]), min_dist)
