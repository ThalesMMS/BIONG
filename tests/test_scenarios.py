import unittest
from collections import deque

from spider_cortex_sim.scenarios import SCENARIO_NAMES, get_scenario
from spider_cortex_sim.world import ACTION_TO_INDEX, SpiderWorld


class ScenarioRegressionTest(unittest.TestCase):
    def _setup_world(self, name: str) -> SpiderWorld:
        """
        Create and return a SpiderWorld configured for the given scenario with deterministic initialization.
        
        Parameters:
            name (str): Identifier of the scenario to load.
        
        Returns:
            SpiderWorld: A world initialized with seed 101, the scenario's map template, and with the scenario's setup applied.
        """
        scenario = get_scenario(name)
        world = SpiderWorld(seed=101, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=101)
        scenario.setup(world)
        return world

    def _move_towards(self, world: SpiderWorld, target: tuple[int, int]) -> str:
        """
        Determine the first move action that guides the spider along a shortest walkable path to the given target cell.
        
        Parameters:
            world (SpiderWorld): The simulation world containing the spider, map dimensions, and walkability info.
            target (tuple[int, int]): Destination cell as (x, y) coordinates.
        
        Returns:
            str: One of "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT" indicating the first step toward the target, or "STAY" if the target is unreachable or is the spider's current position.
        """
        start = world.spider_pos()
        queue = deque([start])
        parent = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == target:
                break
            for action_name, (dx, dy) in (
                ("MOVE_UP", (0, -1)),
                ("MOVE_DOWN", (0, 1)),
                ("MOVE_LEFT", (-1, 0)),
                ("MOVE_RIGHT", (1, 0)),
            ):
                nxt = (cell[0] + dx, cell[1] + dy)
                if not (0 <= nxt[0] < world.width and 0 <= nxt[1] < world.height):
                    continue
                if nxt in parent or not world.is_walkable(nxt):
                    continue
                parent[nxt] = (cell, action_name)
                queue.append(nxt)
        if target not in parent:
            return "STAY"
        step = target
        action_name = "STAY"
        while parent[step] is not None:
            prev, action_name = parent[step]
            if prev == start:
                return action_name
            step = prev
        return "STAY"

    def test_scenario_registry_exposes_expected_names(self) -> None:
        self.assertEqual(
            set(SCENARIO_NAMES),
            {
                "night_rest",
                "predator_edge",
                "entrance_ambush",
                "open_field_foraging",
                "shelter_blockade",
                "recover_after_failed_chase",
                "corridor_gauntlet",
                "two_shelter_tradeoff",
                "exposed_day_foraging",
                "food_deprivation",
                "food_vs_predator_conflict",
                "sleep_vs_exploration_conflict",
            },
        )

    def test_night_rest_reaches_deep_sleep(self) -> None:
        world = self._setup_world("night_rest")
        phases = []
        for _ in range(3):
            _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
            phases.append(info["state"]["sleep_phase"])
        self.assertEqual(phases[-1], "DEEP_SLEEP")
        self.assertLess(world.state.sleep_debt, 0.60)

    def test_predator_edge_triggers_alert_memory(self) -> None:
        world = self._setup_world("predator_edge")
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertTrue(info["state"]["predator_memory"]["target"] is not None)
        self.assertIn(world.lizard.mode, {"ORIENT", "CHASE", "INVESTIGATE"})

    def test_non_facing_scenarios_reset_heading_after_teleport(self) -> None:
        for name in SCENARIO_NAMES:
            if name in {"predator_edge", "food_vs_predator_conflict"}:
                continue
            with self.subTest(scenario=name):
                world = self._setup_world(name)
                expected = world._heading_toward(
                    world.nearest_shelter_entrance(origin=world.spider_pos()),
                    origin=world.spider_pos(),
                )
                self.assertEqual((world.state.heading_dx, world.state.heading_dy), expected)

    def test_predator_facing_scenarios_keep_predator_oriented_heading(self) -> None:
        for name in ("predator_edge", "food_vs_predator_conflict"):
            with self.subTest(scenario=name):
                world = self._setup_world(name)
                expected = world._heading_toward(world.lizard_pos())
                self.assertEqual((world.state.heading_dx, world.state.heading_dy), expected)

    def test_entrance_ambush_keeps_lizard_waiting_outside(self) -> None:
        world = self._setup_world("entrance_ambush")
        entrance = next(iter(world.shelter_entrance_cells))
        for _ in range(3):
            world.step(ACTION_TO_INDEX["STAY"])
        self.assertEqual(world.lizard.mode, "WAIT")
        self.assertEqual(world.lizard.wait_target, entrance)
        self.assertNotIn(world.lizard_pos(), world.shelter_cells)

    def test_open_field_foraging_reduces_food_distance(self) -> None:
        world = self._setup_world("open_field_foraging")
        self.assertEqual(world.map_template_name, "exposed_feeding_ground")
        initial_food_dist = world.observe()["meta"]["food_dist"]
        target = world.food_positions[0]
        for _ in range(4):
            world.step(ACTION_TO_INDEX[self._move_towards(world, target)])
        self.assertLess(world.observe()["meta"]["food_dist"], initial_food_dist)
        self.assertFalse(world.on_shelter())

    def test_shelter_blockade_preserves_shelter_boundary(self) -> None:
        world = self._setup_world("shelter_blockade")
        for _ in range(4):
            world.step(ACTION_TO_INDEX["STAY"])
        self.assertEqual(world.lizard.mode, "WAIT")
        self.assertNotIn(world.lizard_pos(), world.shelter_cells)
        self.assertTrue(world.inside_shelter() or world.deep_shelter())

    def test_recover_after_failed_chase_enters_recover_state(self) -> None:
        world = self._setup_world("recover_after_failed_chase")
        modes = []
        for _ in range(6):
            world.step(ACTION_TO_INDEX["STAY"])
            modes.append(world.lizard.mode)
        self.assertIn("WAIT", modes)
        self.assertIn("RECOVER", modes)

    def test_corridor_gauntlet_uses_corridor_escape_map(self) -> None:
        world = self._setup_world("corridor_gauntlet")
        self.assertEqual(world.map_template_name, "corridor_escape")
        self.assertEqual(world.terrain_at(world.food_positions[0]), "OPEN")
        self.assertEqual(world.terrain_at((5, world.spider_pos()[1])), "NARROW")

    def test_two_shelter_tradeoff_starts_in_left_shelter_with_food_elsewhere(self) -> None:
        world = self._setup_world("two_shelter_tradeoff")
        self.assertEqual(world.map_template_name, "two_shelters")
        self.assertLess(world.spider_pos()[0], world.food_positions[0][0])
        self.assertGreater(len(world.shelter_deep_cells), 2)

    def test_exposed_day_foraging_starts_with_nearby_patrol_pressure(self) -> None:
        """
        Checks that the "exposed_day_foraging" scenario starts with daytime patrol pressure near food.
        
        Initializes the scenario and asserts that the world uses the "exposed_feeding_ground" map template, the environment reports daytime, and the lizard's Manhattan distance to the first food position is less than or equal to 3.
        """
        world = self._setup_world("exposed_day_foraging")
        self.assertEqual(world.map_template_name, "exposed_feeding_ground")
        self.assertTrue(world.observe()["meta"]["day"])
        self.assertLessEqual(
            world.manhattan(world.lizard_pos(), world.food_positions[0]),
            3,
        )

    def test_food_deprivation_starts_hungry_with_distant_food(self) -> None:
        """
        Verify the "food_deprivation" scenario initializes a hungry spider located far from the nearest food.
        
        Asserts that the scenario uses the "central_burrow" map template, the spider's hunger is greater than 0.9, and the Manhattan distance from the spider to the first food position is greater than 3.
        """
        world = self._setup_world("food_deprivation")
        self.assertEqual(world.map_template_name, "central_burrow")
        self.assertGreater(world.state.hunger, 0.9)
        self.assertGreater(world.manhattan(world.spider_pos(), world.food_positions[0]), 3)

    def test_food_vs_predator_conflict_starts_with_visible_pressure_and_near_food(self) -> None:
        """
        Verify the "food_vs_predator_conflict" scenario initializes on the exposed feeding ground with a hungry spider positioned near both food and the lizard.
        
        Asserts:
        - the world uses the "exposed_feeding_ground" map template
        - the spider's hunger is greater than 0.9
        - the Manhattan distance from the spider to the first food position is <= 3
        - the Manhattan distance from the spider to the lizard is <= 2
        """
        world = self._setup_world("food_vs_predator_conflict")
        observation = world.observe()
        self.assertEqual(world.map_template_name, "exposed_feeding_ground")
        self.assertGreater(world.state.hunger, 0.9)
        self.assertLessEqual(world.manhattan(world.spider_pos(), world.food_positions[0]), 3)
        self.assertLessEqual(world.manhattan(world.spider_pos(), world.lizard_pos()), 2)
        self.assertTrue(observation["meta"]["predator_visible"])
        self.assertGreater(observation["meta"]["vision"]["predator"]["certainty"], 0.0)

    def test_sleep_vs_exploration_conflict_starts_sleepy_and_safe(self) -> None:
        world = self._setup_world("sleep_vs_exploration_conflict")
        observation = world.observe()
        self.assertEqual(world.map_template_name, "central_burrow")
        self.assertTrue(observation["meta"]["night"])
        self.assertGreater(world.state.sleep_debt, 0.9)
        self.assertIn(world.spider_pos(), world.shelter_cells)
        self.assertGreater(observation["meta"]["food_dist"], 0)
        self.assertFalse(observation["meta"]["on_food"])

    def test_weak_scenarios_expose_diagnostic_metadata(self) -> None:
        """
        Validate that selected scenarios expose required diagnostic metadata.
        
        Asserts that for each of the scenarios "open_field_foraging", "corridor_gauntlet", "exposed_day_foraging", and "food_deprivation" the attributes `diagnostic_focus`, `success_interpretation`, `failure_interpretation`, and `budget_note` are truthy.
        """
        for name in (
            "open_field_foraging",
            "corridor_gauntlet",
            "exposed_day_foraging",
            "food_deprivation",
        ):
            scenario = get_scenario(name)
            self.assertTrue(scenario.diagnostic_focus)
            self.assertTrue(scenario.success_interpretation)
            self.assertTrue(scenario.failure_interpretation)
            self.assertTrue(scenario.budget_note)


class ConflictScenarioMetadataTest(unittest.TestCase):
    """Tests for the new conflict scenarios added in this PR."""

    def test_scenario_metadata_and_behavior_checks(self) -> None:
        cases = [
            ("food_vs_predator_conflict", "exposed_feeding_ground"),
            ("sleep_vs_exploration_conflict", "central_burrow"),
        ]
        for name, expected_map in cases:
            with self.subTest(scenario=name):
                scenario = get_scenario(name)
                self.assertTrue(scenario.diagnostic_focus)
                self.assertTrue(scenario.success_interpretation)
                self.assertTrue(scenario.failure_interpretation)
                self.assertTrue(scenario.budget_note)
                self.assertEqual(scenario.map_template, expected_map)
                self.assertGreater(scenario.max_steps, 0)
                self.assertLessEqual(scenario.max_steps, 24)
                self.assertEqual(len(scenario.behavior_checks), 3)


class ConflictScenarioSetupTest(unittest.TestCase):
    """Tests verifying initial state constants for the new conflict scenarios."""

    def _setup_world(self, name: str) -> SpiderWorld:
        scenario = get_scenario(name)
        world = SpiderWorld(seed=7, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=7)
        scenario.setup(world)
        return world

    def test_food_vs_predator_conflict_initializes_high_hunger(self) -> None:
        world = self._setup_world("food_vs_predator_conflict")
        self.assertGreater(world.state.hunger, 0.85)

    def test_food_vs_predator_conflict_initializes_low_fatigue(self) -> None:
        world = self._setup_world("food_vs_predator_conflict")
        self.assertLess(world.state.fatigue, 0.50)

    def test_food_vs_predator_conflict_daytime(self) -> None:
        world = self._setup_world("food_vs_predator_conflict")
        obs = world.observe()
        self.assertTrue(obs["meta"]["day"])

    def test_food_vs_predator_conflict_lizard_in_patrol_mode(self) -> None:
        world = self._setup_world("food_vs_predator_conflict")
        self.assertEqual(world.lizard.mode, "PATROL")

    def test_food_vs_predator_conflict_spider_not_in_shelter(self) -> None:
        world = self._setup_world("food_vs_predator_conflict")
        self.assertNotIn(world.spider_pos(), world.shelter_cells)

    def test_sleep_vs_exploration_conflict_initializes_high_sleep_debt(self) -> None:
        from spider_cortex_sim.scenarios import SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT
        world = self._setup_world("sleep_vs_exploration_conflict")
        self.assertAlmostEqual(world.state.sleep_debt, SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT, places=3)

    def test_sleep_vs_exploration_conflict_initializes_nighttime(self) -> None:
        world = self._setup_world("sleep_vs_exploration_conflict")
        obs = world.observe()
        self.assertTrue(obs["meta"]["night"])

    def test_sleep_vs_exploration_conflict_initializes_low_hunger(self) -> None:
        world = self._setup_world("sleep_vs_exploration_conflict")
        self.assertLess(world.state.hunger, 0.30)

    def test_sleep_vs_exploration_conflict_initializes_high_fatigue(self) -> None:
        world = self._setup_world("sleep_vs_exploration_conflict")
        self.assertGreater(world.state.fatigue, 0.80)

    def test_sleep_vs_exploration_conflict_spider_starts_in_shelter(self) -> None:
        world = self._setup_world("sleep_vs_exploration_conflict")
        self.assertIn(world.spider_pos(), world.shelter_cells)

    def test_sleep_vs_exploration_conflict_food_is_far_from_spider(self) -> None:
        world = self._setup_world("sleep_vs_exploration_conflict")
        food_pos = world.food_positions[0]
        self.assertGreater(world.manhattan(world.spider_pos(), food_pos), 1)

    def test_food_vs_predator_conflict_lizard_close_to_spider(self) -> None:
        world = self._setup_world("food_vs_predator_conflict")
        dist = world.manhattan(world.spider_pos(), world.lizard_pos())
        self.assertLessEqual(dist, 3)

    def test_food_vs_predator_conflict_heading_faces_lizard(self) -> None:
        world = self._setup_world("food_vs_predator_conflict")
        expected = world._heading_toward(world.lizard_pos())
        self.assertEqual(
            (world.state.heading_dx, world.state.heading_dy),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
