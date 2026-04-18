from __future__ import annotations

import unittest
from dataclasses import replace

from scenario_test_utils import ScenarioWorldHelpers
from spider_cortex_sim.interfaces import OBSERVATION_INTERFACE_BY_KEY
from spider_cortex_sim.maps import MapTemplate
from spider_cortex_sim.predator import (
    OLFACTORY_HUNTER_PROFILE,
    VISUAL_HUNTER_PROFILE,
)
from spider_cortex_sim.scenarios import (
    FAST_VISUAL_HUNTER_PROFILE,
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    SCENARIO_NAMES,
    get_scenario,
)
from spider_cortex_sim.scenarios.setup import (
    _first_food_visible_frontier,
    _safe_distinct_lizard_cell,
    _safe_lizard_cell,
    _set_predators,
)
from spider_cortex_sim.world import ACTION_TO_INDEX, SpiderWorld

class ScenarioSetupRegressionTest(ScenarioWorldHelpers, unittest.TestCase):

    def test_non_facing_scenarios_reset_heading_after_teleport(self) -> None:
        """
        Verify that scenarios not intended to spawn facing a predator reset the agent's heading to point toward the nearest shelter entrance after scenario setup.

        For each scenario name in SCENARIO_NAMES except predator-facing or special-case scenarios (predator_edge, food_vs_predator_conflict, visual_olfactory_pincer, olfactory_ambush, visual_hunter_open_field), this test constructs the world, computes the heading from the spider position toward the nearest shelter entrance, and asserts the world's initial heading equals that computed heading.
        """
        for name in SCENARIO_NAMES:
            if name in {
                "predator_edge",
                "food_vs_predator_conflict",
                "visual_olfactory_pincer",
                "olfactory_ambush",
                "visual_hunter_open_field",
            }:
                continue
            with self.subTest(scenario=name):
                world = self._setup_world(name)
                expected = world._heading_toward(
                    world.nearest_shelter_entrance(origin=world.spider_pos()),
                    origin=world.spider_pos(),
                )
                self.assertEqual((world.state.heading_dx, world.state.heading_dy), expected)

    def test_predator_facing_scenarios_keep_predator_oriented_heading(self) -> None:
        for name in (
            "predator_edge",
            "food_vs_predator_conflict",
            "visual_olfactory_pincer",
            "visual_hunter_open_field",
        ):
            with self.subTest(scenario=name):
                world = self._setup_world(name)
                expected = world._heading_toward(world.lizard_pos())
                self.assertEqual((world.state.heading_dx, world.state.heading_dy), expected)

    def test_olfactory_ambush_keeps_inward_heading(self) -> None:
        world = self._setup_world("olfactory_ambush")
        self.assertEqual((world.state.heading_dx, world.state.heading_dy), (-1, 0))

    def test_open_field_foraging_uses_calibrated_default_food(self) -> None:
        """
        Verify the "open_field_foraging" scenario initializes with the calibrated default food position.

        Asserts that the world's `food_positions` equals [(6, 2)] after scenario setup.
        """
        world = self._setup_world("open_field_foraging")
        self.assertEqual(world.food_positions, [(6, 2)])

    def test_open_field_foraging_initial_food_distance_is_calibrated(self) -> None:
        """
        Verify the initial observed food distance for the "open_field_foraging" scenario is calibrated between 4 and 6 inclusive.

        Asserts that the observation's `meta["food_dist"]` is >= 4 and <= 6.
        """
        world = self._setup_world("open_field_foraging")
        initial_food_dist = world.observe()["meta"]["food_dist"]
        self.assertGreaterEqual(initial_food_dist, 4)
        self.assertLessEqual(initial_food_dist, 6)

    def test_open_field_foraging_initial_food_signal_is_positive(self) -> None:
        world = self._setup_world("open_field_foraging")
        observation = world.observe()
        hunger = OBSERVATION_INTERFACE_BY_KEY["hunger"].bind_values(observation["hunger"])
        food_memory_signal = (
            1.0 - hunger["food_memory_age"]
            if abs(hunger["food_memory_dx"]) + abs(hunger["food_memory_dy"]) > 0.0
            else 0.0
        )
        food_signal = max(
            hunger["food_visible"],
            hunger["food_certainty"],
            hunger["food_smell_strength"],
            hunger["food_trace_strength"],
            food_memory_signal,
        )
        self.assertGreater(food_signal, 0.0)

    def test_open_field_foraging_first_route_avoids_blocked_east_wall(self) -> None:
        """
        Verifies the first move toward the configured food avoids a blocked eastern wall and produces measurable progress toward the food.

        Asserts that the computed first action is "MOVE_UP", that one step following that action increases the recorded food progress (info["distance_deltas"]["food"] > 0), that the spider is not positioned in any blocked cell, that the spider is not on shelter, and that the public observed food distance decreases relative to the initial observation.
        """
        world = self._setup_world("open_field_foraging", perceptual_delay_ticks=0.0)
        initial_food_dist = world.observe()["meta"]["food_dist"]
        target = world.food_positions[0]
        first_action = self._move_towards(world, target)

        self.assertEqual(first_action, "MOVE_UP")
        _, _, _, info = world.step(ACTION_TO_INDEX[first_action])

        self.assertTrue(info["distance_deltas"]["food"] > 0)
        self.assertNotIn(world.spider_pos(), world.blocked_cells)
        self.assertFalse(world.on_shelter())
        post_step_observation = world.observe()
        self.assertLess(post_step_observation["meta"]["food_dist"], initial_food_dist)

    def test_visual_olfactory_pincer_spawns_two_specialized_predators(self) -> None:
        """
        Verifies that the "visual_olfactory_pincer" scenario spawns two specialized predators and exposes matching threat signals.

        Asserts that:
        - two predators are present and the spider starts at (6, 6);
        - predator positions equal [(8, 6), (5, 8)];
        - predator profiles are [VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE] in that order;
        - predator detection styles are ["visual", "olfactory"];
        - observation metadata reports a positive visual predator threat and a positive olfactory predator threat.
        """
        world = self._setup_world("visual_olfactory_pincer")
        self.assertEqual(world.predator_count, 2)
        self.assertEqual(world.spider_pos(), (6, 6))
        self.assertEqual(world.predator_positions(), [(8, 6), (5, 8)])
        self.assertEqual(
            [predator.profile for predator in world.predators],
            [VISUAL_HUNTER_PROFILE, OLFACTORY_HUNTER_PROFILE],
        )
        self.assertEqual(
            [predator.profile.detection_style for predator in world.predators],
            ["visual", "olfactory"],
        )
        meta = world.observe()["meta"]
        self.assertGreater(meta["visual_predator_threat"], 0.0)
        self.assertGreater(meta["olfactory_predator_threat"], 0.0)

    def test_olfactory_ambush_uses_olfactory_profile_and_hidden_position(self) -> None:
        world = self._setup_world("olfactory_ambush")
        self.assertEqual(world.predator_count, 1)
        self.assertEqual(world.spider_pos(), (3, 5))
        self.assertEqual(world.predator_positions(), [(5, 6)])
        self.assertEqual(world.lizard.profile, OLFACTORY_HUNTER_PROFILE)
        self.assertEqual(world.lizard.mode, "WAIT")
        self.assertEqual(world.lizard.wait_target, (4, 6))
        meta = world.observe()["meta"]
        self.assertFalse(meta["predator_visible"])
        self.assertEqual(meta["visual_predator_threat"], 0.0)
        self.assertGreater(meta["olfactory_predator_threat"], 0.0)
        self.assertEqual(meta["dominant_predator_type_label"], "olfactory")

    def test_visual_hunter_open_field_uses_fast_visual_profile_and_open_position(self) -> None:
        world = self._setup_world("visual_hunter_open_field")
        self.assertEqual(world.predator_count, 1)
        self.assertEqual(world.spider_pos(), (6, 6))
        self.assertEqual(world.predator_positions(), [(8, 6)])
        self.assertEqual(world.lizard.profile, FAST_VISUAL_HUNTER_PROFILE)
        self.assertEqual(world.lizard.profile.detection_style, "visual")
        self.assertEqual(world.lizard.profile.move_interval, 1)
        meta = world.observe()["meta"]
        self.assertTrue(meta["predator_visible"])
        self.assertGreater(meta["visual_predator_threat"], 0.0)
        self.assertEqual(meta["olfactory_predator_threat"], 0.0)
        self.assertEqual(meta["dominant_predator_type_label"], "visual")

    def test_set_predators_rejects_empty_roster(self) -> None:
        world = self._setup_world("visual_hunter_open_field")
        with self.assertRaises(ValueError):
            _set_predators(world, [])

    def test_corridor_gauntlet_uses_corridor_escape_map(self) -> None:
        world = self._setup_world("corridor_gauntlet")
        self.assertEqual(world.map_template_name, "corridor_escape")
        self.assertEqual(world.terrain_at(world.food_positions[0]), "OPEN")
        self.assertEqual(world.terrain_at((5, world.spider_pos()[1])), "NARROW")

    def test_corridor_gauntlet_lizard_is_between_spider_and_food(self) -> None:
        world = self._setup_world("corridor_gauntlet")
        spider = world.spider_pos()
        lizard = world.lizard_pos()
        food = world.food_positions[0]

        self.assertEqual(lizard[1], spider[1])
        self.assertGreater(lizard[0], spider[0])
        self.assertLess(lizard[0], food[0])
        self.assertEqual(
            world.manhattan(spider, lizard) + world.manhattan(lizard, food),
            world.manhattan(spider, food),
        )

    def test_corridor_gauntlet_food_is_far_from_spider(self) -> None:
        world = self._setup_world("corridor_gauntlet")
        spider = world.spider_pos()
        food = world.food_positions[0]
        spawn_candidates = world.map_template.food_spawn_cells
        max_dist = max(
            world.manhattan(spider, pos)
            for pos in spawn_candidates
        )

        self.assertIn(food, spawn_candidates)
        self.assertGreaterEqual(
            world.manhattan(spider, food),
            world.width - 2,
        )
        self.assertEqual(world.manhattan(spider, food), max_dist)

    def test_corridor_gauntlet_has_narrow_terrain_on_spider_row(self) -> None:
        """Verify narrow corridor terrain exists on the spider row."""
        world = self._setup_world("corridor_gauntlet")
        spider = world.spider_pos()
        lizard = world.lizard_pos()
        narrow_xs = [
            x
            for x in range(world.width)
            if world.terrain_at((x, spider[1])) == "NARROW"
        ]

        self.assertGreaterEqual(len(narrow_xs), 4)
        self.assertGreater(min(narrow_xs), spider[0])
        self.assertTrue(0 <= lizard[0] < world.width)
        self.assertTrue(0 <= lizard[1] < world.height)

    def test_two_shelter_tradeoff_starts_in_left_shelter_with_food_elsewhere(self) -> None:
        """
        Verify the 'two_shelter_tradeoff' scenario initializes the spider in the left shelter while food is placed to the right.

        Asserts that the scenario uses the "two_shelters" map template, that the spider's x-coordinate is less than the first food position's x-coordinate, and that the map contains more than two deep shelter cells.
        """
        world = self._setup_world("two_shelter_tradeoff")
        self.assertEqual(world.map_template_name, "two_shelters")
        self.assertLess(world.spider_pos()[0], world.food_positions[0][0])
        self.assertGreater(len(world.shelter_deep_cells), 2)

    def test_exposed_day_foraging_starts_with_nearby_patrol_pressure(self) -> None:
        """
        Verify the "exposed_day_foraging" scenario starts in daytime on the "exposed_feeding_ground" map with the lizard positioned three Manhattan steps from the first food.

        Asserts that:
        - the map template is "exposed_feeding_ground",
        - observation metadata reports daytime, and
        - the lizard-to-first-food Manhattan distance is exactly 3.
        """
        world = self._setup_world("exposed_day_foraging")
        self.assertEqual(world.map_template_name, "exposed_feeding_ground")
        self.assertTrue(world.observe()["meta"]["day"])
        self.assertTrue(world.is_lizard_walkable(world.lizard_pos()))
        food_distance = world.manhattan(world.lizard_pos(), world.food_positions[0])
        # this is the regression guard for the geometry fix
        self.assertEqual(food_distance, 3)
        frontier = _first_food_visible_frontier(
            world,
            world.spider_pos(),
            world.food_positions[0],
        )
        self.assertTrue(frontier)
        for frontier_cell in frontier:
            with self.subTest(frontier_cell=frontier_cell):
                self.assertGreaterEqual(world.manhattan(world.lizard_pos(), frontier_cell), 3)

    def test_exposed_day_foraging_falls_back_when_lizard_spawns_unwalkable(self) -> None:
        scenario = get_scenario("exposed_day_foraging")
        world = SpiderWorld(seed=101, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=101)
        world.map_template = replace(
            world.map_template,
            lizard_spawn_cells=tuple(sorted(world.shelter_cells)),
        )
        self.assertFalse(
            any(
                world.is_lizard_walkable(cell)
                for cell in world.map_template.lizard_spawn_cells
            )
        )

        scenario.setup(world)

        self.assertTrue(world.is_lizard_walkable(world.lizard_pos()))
        self.assertGreaterEqual(world.manhattan(world.lizard_pos(), world.food_positions[0]), 3)

    def test_exposed_day_foraging_lizard_separated_from_food_multiple_seeds(self) -> None:
        """
        Regression guard: lizard must be at least 3 Manhattan steps from food for
        several seeds, verifying the geometry fix is seed-independent.
        """
        scenario = get_scenario("exposed_day_foraging")
        for seed in (101, 202, 303, 404):
            with self.subTest(seed=seed):
                world = SpiderWorld(seed=seed, lizard_move_interval=1, map_template=scenario.map_template)
                world.reset(seed=seed)
                scenario.setup(world)
                dist = world.manhattan(world.lizard_pos(), world.food_positions[0])
                self.assertGreaterEqual(
                    dist,
                    3,
                    msg=f"seed={seed}: lizard too close to food (dist={dist})",
                )

    def test_exposed_day_foraging_food_is_farthest_from_spider(self) -> None:
        """
        Verify that the food spawn selected by setup is the cell farthest from the
        spider's start position among all food_spawn_cells in the map template.
        """
        world = self._setup_world("exposed_day_foraging")
        spider_pos = world.spider_pos()
        farthest = max(
            world.map_template.food_spawn_cells,
            key=lambda cell: world.manhattan(cell, spider_pos),
        )
        self.assertEqual(
            world.food_positions[0],
            farthest,
            msg="Food should be placed at the spawn cell farthest from the spider",
        )

    def test_exposed_day_foraging_lizard_in_patrol_mode(self) -> None:
        """
        Verify the lizard is placed in PATROL mode by the exposed_day_foraging setup.
        """
        world = self._setup_world("exposed_day_foraging")
        self.assertEqual(world.lizard.mode, "PATROL")

    def test_exposed_day_foraging_initial_state_values(self) -> None:
        """
        Verify the scenario sets the expected initial hunger, fatigue, sleep_debt, and tick.
        """
        world = self._setup_world("exposed_day_foraging")
        self.assertAlmostEqual(world.state.hunger, 0.94, places=5)
        self.assertAlmostEqual(world.state.fatigue, 0.16, places=5)
        self.assertAlmostEqual(world.state.sleep_debt, 0.18, places=5)
        self.assertEqual(world.tick, 1)

    def test_food_deprivation_starts_hungry_with_calibrated_food(self) -> None:
        """
        Check that the "food_deprivation" scenario starts with high hunger and places the selected food at a calibrated distance.

        Asserts that the scenario uses the "central_burrow" map template, the spider's hunger is greater than 0.9, the Manhattan distance from the spider to the first food is between 4 and 6 steps inclusive, and that this chosen distance is less than the farthest food-spawn distance defined by the map template.
        """
        world = self._setup_world("food_deprivation")
        self.assertEqual(world.map_template_name, "central_burrow")
        self.assertGreater(world.state.hunger, 0.9)
        distance = world.manhattan(world.spider_pos(), world.food_positions[0])
        self.assertGreaterEqual(distance, 4)
        self.assertLessEqual(distance, 6)
        farthest_spawn_distance = max(
            world.manhattan(world.spider_pos(), cell)
            for cell in world.map_template.food_spawn_cells
        )
        self.assertLess(distance, farthest_spawn_distance)

    def test_food_deprivation_fallback_uses_closest_spawn_at_least_four_steps(self) -> None:
        """
        Verify that the "food_deprivation" scenario falls back to the closest food spawn that is at least four Manhattan steps away when no spawn falls within the calibrated [4, 6] range.

        Sets up a custom map with food spawn cells at distances 3, 7, and 9 from the spider start, runs the scenario setup, and asserts:
        - no food spawn distance is within 4..6,
        - the selected world.food_positions contains the closest eligible fallback (distance >= 4),
        - the chosen spawn has Manhattan distance 7.
        """
        scenario = get_scenario("food_deprivation")
        world = SpiderWorld(seed=101, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=101)
        entrance = ((2, 0),)
        interior = ((2, 1),)
        deep = ((2, 2),)
        food_spawns = (
            (5, 2),   # distance 3, below fallback floor
            (9, 2),   # distance 7, closest eligible fallback
            (11, 2),  # distance 9
        )
        custom_template = MapTemplate(
            name="central_burrow",
            width=world.width,
            height=world.height,
            terrain={},
            shelter_entrance=entrance,
            shelter_interior=interior,
            shelter_deep=deep,
            blocked_cells=(),
            food_spawn_cells=food_spawns,
            lizard_spawn_cells=((11, 11),),
            spider_start=deep[0],
        )
        world.map_template = custom_template
        world.shelter_entrance_cells = set(entrance)
        world.shelter_interior_cells = set(interior)
        world.shelter_deep_cells = set(deep)
        world.shelter_cells = custom_template.shelter_cells
        world.blocked_cells = set()
        world.state = world._initial_spider_state(*custom_template.spider_start)

        scenario.setup(world)

        start = world.spider_pos()
        distances = {
            cell: world.manhattan(start, cell)
            for cell in custom_template.food_spawn_cells
        }
        self.assertFalse(any(4 <= distance <= 6 for distance in distances.values()))
        eligible_fallbacks = [
            cell
            for cell, distance in distances.items()
            if distance >= 4
        ]
        expected_food = min(
            eligible_fallbacks,
            key=lambda cell: (distances[cell], cell[0], cell[1]),
        )
        self.assertEqual(world.food_positions, [expected_food])
        self.assertEqual(distances[expected_food], 7)

    def test_food_deprivation_handles_empty_food_spawns(self) -> None:
        scenario = get_scenario("food_deprivation")
        world = SpiderWorld(seed=101, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=101)
        custom_template = MapTemplate(
            name="central_burrow",
            width=world.width,
            height=world.height,
            terrain={},
            shelter_entrance=((2, 0),),
            shelter_interior=((2, 1),),
            shelter_deep=((2, 2),),
            blocked_cells=(),
            food_spawn_cells=(),
            lizard_spawn_cells=((11, 11),),
            spider_start=(2, 2),
        )
        world.map_template = custom_template
        world.shelter_entrance_cells = set(custom_template.shelter_entrance)
        world.shelter_interior_cells = set(custom_template.shelter_interior)
        world.shelter_deep_cells = set(custom_template.shelter_deep)
        world.shelter_cells = custom_template.shelter_cells
        world.blocked_cells = set()
        world.state = world._initial_spider_state(*custom_template.spider_start)

        with self.assertRaisesRegex(
            ValueError,
            "food_deprivation has no food spawn cells in map_template",
        ):
            scenario.setup(world)

    def test_food_deprivation_initial_hunger_remains_0_96(self) -> None:
        world = self._setup_world("food_deprivation")
        self.assertAlmostEqual(
            world.state.hunger,
            FOOD_DEPRIVATION_INITIAL_HUNGER,
            places=3,
        )

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

    def test_food_vs_predator_conflict_raises_without_open_spider_cell(self) -> None:
        scenario = get_scenario("food_vs_predator_conflict")
        world = SpiderWorld(seed=7, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=7)
        food = (5, 2)
        shelter_cells = tuple(
            sorted(
                (x, y)
                for x in range(world.width)
                for y in range(world.height)
                if (x, y) != food
            )
        )
        custom_template = MapTemplate(
            name="exposed_feeding_ground",
            width=world.width,
            height=world.height,
            terrain={},
            shelter_entrance=shelter_cells,
            shelter_interior=(),
            shelter_deep=(),
            blocked_cells=(),
            food_spawn_cells=(food,),
            lizard_spawn_cells=((2, 1),),
            spider_start=shelter_cells[0],
        )
        world.map_template = custom_template
        world.shelter_entrance_cells = set(shelter_cells)
        world.shelter_interior_cells = set()
        world.shelter_deep_cells = set()
        world.shelter_cells = custom_template.shelter_cells
        world.blocked_cells = set()

        with self.assertRaisesRegex(
            ValueError,
            r"food_vs_predator_conflict requires an open traversable spider cell.*\(5, 2\)",
        ):
            scenario.setup(world)

class OpenFieldForagingSetupTest(ScenarioWorldHelpers, unittest.TestCase):
    """Tests for the _open_field_foraging scenario setup and _open_field_foraging_food_cell."""

    def _setup_world(self, seed: int = 101) -> "SpiderWorld":
        scenario = get_scenario("open_field_foraging")
        world = SpiderWorld(seed=seed, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=seed)
        scenario.setup(world)
        return world

    def test_physiology_on_setup(self) -> None:
        """Verify hunger, fatigue, and sleep_debt are configured to expected values."""
        world = self._setup_world()
        self.assertAlmostEqual(world.state.hunger, 0.88)
        self.assertAlmostEqual(world.state.fatigue, 0.22)
        self.assertAlmostEqual(world.state.sleep_debt, 0.20)

    def test_tick_is_2_on_setup(self) -> None:
        """Verify world tick is set to 2 (early-foraging phase)."""
        world = self._setup_world()
        self.assertEqual(world.tick, 2)

    def test_lizard_starts_in_patrol(self) -> None:
        """Verify the predator is in PATROL mode after scenario setup."""
        world = self._setup_world()
        self.assertEqual(world.lizard.mode, "PATROL")

    def test_spider_starts_in_deep_shelter(self) -> None:
        """Verify the spider starts in a deep/interior shelter cell after setup."""
        world = self._setup_world()
        shelter_cells = world.shelter_deep_cells | world.shelter_interior_cells | world.shelter_entrance_cells
        self.assertIn(world.spider_pos(), shelter_cells)

    def test_single_food_position(self) -> None:
        """Verify exactly one food position is configured."""
        world = self._setup_world()
        self.assertEqual(len(world.food_positions), 1)

    def test_food_is_reachable_on_multiple_seeds(self) -> None:
        """
        Verify the food cell is walkable and reachable for several seeds,
        exercising _open_field_foraging_food_cell's main and fallback paths.
        """
        scenario = get_scenario("open_field_foraging")
        for seed in (42, 77, 99, 200, 303):
            with self.subTest(seed=seed):
                world = SpiderWorld(seed=seed, lizard_move_interval=1, map_template=scenario.map_template)
                world.reset(seed=seed)
                scenario.setup(world)
                self.assertEqual(len(world.food_positions), 1)
                food = world.food_positions[0]
                self.assertTrue(world.is_walkable(food), msg=f"seed={seed}: food {food} not walkable")
                action = self._move_towards(world, food)
                self.assertNotEqual(action, "STAY", msg=f"seed={seed}: food unreachable from {world.spider_pos()}")

    def test_food_distance_at_least_4_on_multiple_seeds(self) -> None:
        """
        Verify food is at least 4 manhattan steps from the spider for several seeds.
        Distances > 6 are allowed when the fallback (farthest cell) is used.
        """
        scenario = get_scenario("open_field_foraging")
        for seed in (101, 202, 303):
            with self.subTest(seed=seed):
                world = SpiderWorld(seed=seed, lizard_move_interval=1, map_template=scenario.map_template)
                world.reset(seed=seed)
                scenario.setup(world)
                dist = world.manhattan(world.spider_pos(), world.food_positions[0])
                self.assertGreaterEqual(dist, 4, msg=f"seed={seed}: food too close (dist={dist})")

class FoodDeprivationSetupTest(unittest.TestCase):
    """Tests for the _food_deprivation scenario setup function."""

    def _setup_world(self) -> SpiderWorld:
        scenario = get_scenario("food_deprivation")
        world = SpiderWorld(seed=101, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=101)
        scenario.setup(world)
        return world

    def test_spider_starts_in_deep_shelter(self) -> None:
        world = self._setup_world()
        self.assertIn(world.spider_pos(), world.shelter_deep_cells)

    def test_hunger_is_high(self) -> None:
        world = self._setup_world()
        self.assertGreater(world.state.hunger, 0.9)

    def test_hunger_is_exactly_0_96(self) -> None:
        world = self._setup_world()
        self.assertAlmostEqual(world.state.hunger, FOOD_DEPRIVATION_INITIAL_HUNGER)

    def test_fatigue_is_set(self) -> None:
        world = self._setup_world()
        self.assertAlmostEqual(world.state.fatigue, 0.22)

    def test_sleep_debt_is_set(self) -> None:
        world = self._setup_world()
        self.assertAlmostEqual(world.state.sleep_debt, 0.18)

    def test_tick_is_4(self) -> None:
        world = self._setup_world()
        self.assertEqual(world.tick, 4)

    def test_single_food_position(self) -> None:
        world = self._setup_world()
        self.assertEqual(len(world.food_positions), 1)

    def test_food_is_far_from_spider(self) -> None:
        world = self._setup_world()
        dist = world.manhattan(world.spider_pos(), world.food_positions[0])
        self.assertGreaterEqual(dist, 4)
        self.assertLessEqual(dist, 6)

    def test_lizard_starts_in_patrol_mode(self) -> None:
        world = self._setup_world()
        self.assertEqual(world.lizard.mode, "PATROL")

    def test_uses_central_burrow_map(self) -> None:
        scenario = get_scenario("food_deprivation")
        self.assertEqual(scenario.map_template, "central_burrow")

    def test_max_steps_is_22(self) -> None:
        scenario = get_scenario("food_deprivation")
        self.assertEqual(scenario.max_steps, 22)


class SafeLizardCellTest(unittest.TestCase):
    """Tests for _safe_lizard_cell behavior in the new setup.py module."""

    def _make_world(self, name: str = "central_burrow") -> SpiderWorld:
        scenario = get_scenario("night_rest")
        world = SpiderWorld(seed=42, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=42)
        return world

    def test_returns_walkable_cell(self) -> None:
        """_safe_lizard_cell must return a lizard-walkable cell."""
        world = self._make_world()
        cell = _safe_lizard_cell(world)
        self.assertTrue(world.is_lizard_walkable(cell))

    def test_returns_tuple_of_two_ints(self) -> None:
        world = self._make_world()
        cell = _safe_lizard_cell(world)
        self.assertIsInstance(cell, tuple)
        self.assertEqual(len(cell), 2)
        self.assertIsInstance(cell[0], int)
        self.assertIsInstance(cell[1], int)

    def test_cell_is_within_bounds(self) -> None:
        world = self._make_world()
        cell = _safe_lizard_cell(world)
        self.assertGreaterEqual(cell[0], 0)
        self.assertLess(cell[0], world.width)
        self.assertGreaterEqual(cell[1], 0)
        self.assertLess(cell[1], world.height)

    def test_raises_when_no_walkable_cell_exists(self) -> None:
        """_safe_lizard_cell raises ValueError when no lizard-walkable cell is available."""
        scenario = get_scenario("night_rest")
        world = SpiderWorld(seed=42, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=42)
        # Make shelter_cells include all traversable cells so is_lizard_walkable returns False everywhere
        world.shelter_cells = set(world.map_template.traversable_cells)
        # Also override spawn cells so none are outside shelter
        custom_template = replace(
            world.map_template,
            lizard_spawn_cells=tuple(sorted(world.shelter_cells)),
        )
        world.map_template = custom_template
        with self.assertRaises(ValueError):
            _safe_lizard_cell(world)

    def test_prefers_spawn_cells_over_full_grid_search(self) -> None:
        """When spawn cells contain a walkable cell, it is returned without full-grid scan."""
        world = self._make_world()
        # Verify spawn cells exist and are walked
        spawn_cells = world.map_template.lizard_spawn_cells
        walkable_spawns = [c for c in spawn_cells if world.is_lizard_walkable(c)]
        self.assertTrue(walkable_spawns)
        cell = _safe_lizard_cell(world)
        # Cell should be from spawn cells
        self.assertIn(cell, spawn_cells)

    def test_maximizes_distance_from_spider(self) -> None:
        """Returned cell should be farthest from current spider position."""
        world = self._make_world()
        cell = _safe_lizard_cell(world)
        spawn_cells = world.map_template.lizard_spawn_cells
        walkable_spawns = [c for c in spawn_cells if world.is_lizard_walkable(c)]
        self.assertTrue(walkable_spawns)
        spider = world.spider_pos()
        max_dist = max(world.manhattan(c, spider) for c in walkable_spawns)
        self.assertEqual(world.manhattan(cell, spider), max_dist)


class SafeDistinctLizardCellTest(unittest.TestCase):
    """Tests for the new _safe_distinct_lizard_cell function in setup.py."""

    def _make_world(self) -> SpiderWorld:
        scenario = get_scenario("night_rest")
        world = SpiderWorld(seed=42, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=42)
        return world

    def test_returns_walkable_cell(self) -> None:
        world = self._make_world()
        cell = _safe_distinct_lizard_cell(world, excluded=set())
        self.assertTrue(world.is_lizard_walkable(cell))

    def test_returns_tuple_of_two_ints(self) -> None:
        world = self._make_world()
        cell = _safe_distinct_lizard_cell(world, excluded=set())
        self.assertIsInstance(cell, tuple)
        self.assertEqual(len(cell), 2)
        self.assertIsInstance(cell[0], int)
        self.assertIsInstance(cell[1], int)

    def test_excluded_empty_set_behaves_like_safe_lizard_cell(self) -> None:
        """With no exclusions, result equals _safe_lizard_cell."""
        world = self._make_world()
        cell = _safe_distinct_lizard_cell(world, excluded=set())
        expected = _safe_lizard_cell(world)
        self.assertEqual(cell, expected)

    def test_excluded_cell_not_returned(self) -> None:
        """The returned cell must not be in the excluded set."""
        world = self._make_world()
        preferred = _safe_lizard_cell(world)
        excluded = {preferred}
        cell = _safe_distinct_lizard_cell(world, excluded=excluded)
        self.assertNotIn(cell, excluded)
        self.assertTrue(world.is_lizard_walkable(cell))

    def test_raises_when_all_cells_excluded(self) -> None:
        """Raises ValueError when every lizard-walkable cell is excluded."""
        world = self._make_world()
        all_walkable = {
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
            if world.is_lizard_walkable((x, y))
        }
        with self.assertRaises(ValueError):
            _safe_distinct_lizard_cell(world, excluded=all_walkable)

    def test_cell_within_bounds(self) -> None:
        world = self._make_world()
        cell = _safe_distinct_lizard_cell(world, excluded=set())
        self.assertGreaterEqual(cell[0], 0)
        self.assertLess(cell[0], world.width)
        self.assertGreaterEqual(cell[1], 0)
        self.assertLess(cell[1], world.height)

    def test_multiple_exclusions_all_avoided(self) -> None:
        """All cells in the excluded set should not appear in the result."""
        world = self._make_world()
        all_walkable = sorted(
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
            if world.is_lizard_walkable((x, y))
        )
        if len(all_walkable) < 2:
            self.skipTest("Not enough walkable cells")
        excluded = set(all_walkable[:-1])  # exclude all but last
        cell = _safe_distinct_lizard_cell(world, excluded=excluded)
        self.assertNotIn(cell, excluded)

    def test_maximizes_distance_from_spider_when_preferred_excluded(self) -> None:
        """When preferred is excluded, a maximally-distant valid cell is chosen."""
        world = self._make_world()
        preferred = _safe_lizard_cell(world)
        excluded = {preferred}
        cell = _safe_distinct_lizard_cell(world, excluded=excluded)
        valid_candidates = [
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
            if (x, y) not in excluded and world.is_lizard_walkable((x, y))
        ]
        spider = world.spider_pos()
        max_dist = max(
            world.manhattan(candidate, spider) for candidate in valid_candidates
        )
        self.assertEqual(world.manhattan(cell, spider), max_dist)
