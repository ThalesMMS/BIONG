import unittest
from collections import deque
from dataclasses import replace

from spider_cortex_sim.interfaces import OBSERVATION_INTERFACE_BY_KEY
from spider_cortex_sim.maps import MapTemplate
from spider_cortex_sim.metrics import EpisodeStats
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.predator import (
    OLFACTORY_HUNTER_PROFILE,
    PREDATOR_STATES,
    VISUAL_HUNTER_PROFILE,
)
from spider_cortex_sim.scenarios import (
    FAST_VISUAL_HUNTER_PROFILE,
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    SCENARIO_NAMES,
    _set_predators,
    _classify_corridor_gauntlet_failure,
    _first_food_visible_frontier,
    _score_corridor_gauntlet,
    _trace_corridor_metrics,
    get_scenario,
)
from spider_cortex_sim.world import ACTION_TO_INDEX, REWARD_COMPONENT_NAMES, SpiderWorld


def _make_corridor_episode_stats(**overrides: object) -> EpisodeStats:
    """Return EpisodeStats pre-populated with corridor_gauntlet defaults, accepting overrides."""
    defaults = dict(
        episode=0,
        seed=42,
        training=False,
        scenario="corridor_gauntlet",
        total_reward=0.0,
        steps=3,
        food_eaten=0,
        sleep_events=0,
        shelter_entries=0,
        alert_events=0,
        predator_contacts=0,
        predator_sightings=0,
        predator_escapes=0,
        night_ticks=0,
        night_shelter_ticks=0,
        night_still_ticks=0,
        night_role_ticks={"outside": 0, "entrance": 0, "inside": 0, "deep": 0},
        night_shelter_occupancy_rate=0.0,
        night_stillness_rate=0.0,
        night_role_distribution={"outside": 0.0, "entrance": 0.0, "inside": 0.0, "deep": 0.0},
        predator_response_events=0,
        mean_predator_response_latency=0.0,
        mean_sleep_debt=0.0,
        food_distance_delta=0.0,
        shelter_distance_delta=0.0,
        final_hunger=0.9,
        final_fatigue=0.2,
        final_sleep_debt=0.2,
        final_health=1.0,
        alive=True,
        reward_component_totals={name: 0.0 for name in REWARD_COMPONENT_NAMES},
        predator_state_ticks={state: 0 for state in PREDATOR_STATES},
        predator_mode_transitions=0,
        dominant_predator_state="PATROL",
    )
    defaults.update(overrides)
    return EpisodeStats(**defaults)


def _make_corridor_trace_item(
    *,
    tick: int,
    pos: tuple[int, int],
    health: float,
    food_dist: int,
    predator_visible: bool = False,
) -> dict[str, object]:
    """Return a corridor trace item for one diagnostic tick."""
    return {
        "tick": tick,
        "state": {
            "x": pos[0],
            "y": pos[1],
            "health": health,
            "map_template": "corridor_escape",
        },
        "messages": [
            {
                "sender": "environment",
                "topic": "observation",
                "payload": {
                    "meta": {
                        "food_dist": food_dist,
                        "predator_visible": predator_visible,
                    },
                },
            }
        ],
    }


class ScenarioRegressionTest(unittest.TestCase):
    def _setup_world(
        self,
        name: str,
        *,
        perceptual_delay_ticks: float | None = None,
    ) -> SpiderWorld:
        """
        Create and return a SpiderWorld configured for the given scenario with deterministic initialization.
        
        Parameters:
            name (str): Identifier of the scenario to load.
        
        Returns:
            SpiderWorld: A world initialized with seed 101, the scenario's map template, and with the scenario's setup applied.
        """
        scenario = get_scenario(name)
        kwargs: dict[str, object] = {}
        if perceptual_delay_ticks is not None:
            summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
            summary["perception"]["perceptual_delay_ticks"] = perceptual_delay_ticks
            kwargs["operational_profile"] = OperationalProfile.from_summary(summary)
        world = SpiderWorld(
            seed=101,
            lizard_move_interval=1,
            map_template=scenario.map_template,
            **kwargs,
        )
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
                "visual_olfactory_pincer",
                "olfactory_ambush",
                "visual_hunter_open_field",
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

    def test_entrance_ambush_keeps_lizard_waiting_outside(self) -> None:
        world = self._setup_world("entrance_ambush")
        entrance = next(iter(world.shelter_entrance_cells))
        for _ in range(3):
            world.step(ACTION_TO_INDEX["STAY"])
        self.assertEqual(world.lizard.mode, "WAIT")
        self.assertEqual(world.lizard.wait_target, entrance)
        self.assertNotIn(world.lizard_pos(), world.shelter_cells)

    def test_open_field_foraging_reduces_food_distance(self) -> None:
        """
        Verify that in the "open_field_foraging" scenario the spider reduces its observed distance to the nearest food within four move steps and is not on shelter afterward.
        
        The test checks the scenario uses the "exposed_feeding_ground" map, records the initial observed `meta["food_dist"]`, advances the world up to four steps moving toward the first configured food position, and then asserts the observed food distance is smaller than the initial value and that the spider is not on a shelter cell.
        """
        world = self._setup_world("open_field_foraging")
        self.assertEqual(world.map_template_name, "exposed_feeding_ground")
        initial_food_dist = world.observe()["meta"]["food_dist"]
        target = world.food_positions[0]
        for _ in range(4):
            world.step(ACTION_TO_INDEX[self._move_towards(world, target)])
        self.assertLess(world.observe()["meta"]["food_dist"], initial_food_dist)
        self.assertFalse(world.on_shelter())

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

    def test_multi_predator_scenario_scores_preserve_per_type_metrics(self) -> None:
        spec = get_scenario("visual_olfactory_pincer")
        stats = _make_corridor_episode_stats(
            scenario="visual_olfactory_pincer",
            predator_contacts_by_type={"visual": 1, "olfactory": 0},
            predator_response_latency_by_type={"visual": 2.0, "olfactory": 4.0},
            module_response_by_predator_type={
                "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
                "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
            },
        )
        trace = [
            {
                "observation": {
                    "meta": {
                        "visual_predator_threat": 0.8,
                        "olfactory_predator_threat": 0.6,
                        "dominant_predator_type_label": "visual",
                    }
                }
            },
            {
                "observation": {
                    "meta": {
                        "visual_predator_threat": 0.4,
                        "olfactory_predator_threat": 0.9,
                        "dominant_predator_type_label": "olfactory",
                    }
                }
            },
        ]

        score = spec.score_episode(stats, trace)

        self.assertEqual(
            score.behavior_metrics["predator_contacts_by_type"],
            {"visual": 1, "olfactory": 0},
        )
        self.assertEqual(
            score.behavior_metrics["predator_response_latency_by_type"],
            {"visual": 2.0, "olfactory": 4.0},
        )
        self.assertEqual(
            score.behavior_metrics["dominant_predator_types_seen"],
            ["olfactory", "visual"],
        )
        self.assertEqual(
            score.behavior_metrics["module_response_by_predator_type"],
            {
                "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
                "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
            },
        )

    def test_olfactory_ambush_scoring_uses_initial_hidden_window(self) -> None:
        spec = get_scenario("olfactory_ambush")
        stats = _make_corridor_episode_stats(
            scenario="olfactory_ambush",
            predator_contacts=0,
            alive=True,
            module_response_by_predator_type={
                "olfactory": {"sensory_cortex": 0.8, "visual_cortex": 0.2},
            },
        )
        trace = [
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.6,
                "visual_predator_threat": 0.0,
                "predator_visible": False,
            }}},
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.5,
                "visual_predator_threat": 0.0,
                "predator_visible": False,
            }}},
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.3,
                "visual_predator_threat": 0.0,
                "predator_visible": False,
            }}},
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.2,
                "visual_predator_threat": 0.0,
                "predator_visible": False,
            }}},
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.3,
                "visual_predator_threat": 0.4,
                "predator_visible": True,
            }}},
        ]

        score = spec.score_episode(stats, trace)

        self.assertTrue(score.checks["olfactory_threat_detected"].passed)
        self.assertTrue(score.checks["sensory_cortex_engaged"].passed)
        self.assertEqual(score.behavior_metrics["predator_visible_ticks_initial"], 0)
        self.assertGreater(score.behavior_metrics["olfactory_predator_threat_peak_initial"], 0.0)

    def test_olfactory_ambush_detection_ignores_late_olfactory_peak(self) -> None:
        spec = get_scenario("olfactory_ambush")
        stats = _make_corridor_episode_stats(
            scenario="olfactory_ambush",
            predator_contacts=0,
            alive=True,
            module_response_by_predator_type={
                "olfactory": {"sensory_cortex": 0.8, "visual_cortex": 0.2},
            },
        )
        trace = [
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.0,
                "visual_predator_threat": 0.0,
                "predator_visible": False,
            }}},
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.0,
                "visual_predator_threat": 0.0,
                "predator_visible": False,
            }}},
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.0,
                "visual_predator_threat": 0.0,
                "predator_visible": False,
            }}},
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.0,
                "visual_predator_threat": 0.0,
                "predator_visible": False,
            }}},
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.7,
                "visual_predator_threat": 0.2,
                "predator_visible": True,
            }}},
        ]

        score = spec.score_episode(stats, trace)

        self.assertFalse(score.checks["olfactory_threat_detected"].passed)
        self.assertEqual(score.behavior_metrics["olfactory_predator_threat_peak_initial"], 0.0)

    def test_visual_olfactory_pincer_detection_requires_exported_threat_peaks(self) -> None:
        spec = get_scenario("visual_olfactory_pincer")
        stats = _make_corridor_episode_stats(
            scenario="visual_olfactory_pincer",
            module_response_by_predator_type={
                "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
                "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
            },
        )
        trace = [
            {"observation": {"meta": {
                "visual_predator_threat": 0.0,
                "olfactory_predator_threat": 0.0,
                "dominant_predator_type_label": "",
            }}}
        ]

        score = spec.score_episode(stats, trace)

        self.assertFalse(score.checks["dual_threat_detected"].passed)

    def test_olfactory_ambush_detection_requires_exported_threat_peak(self) -> None:
        spec = get_scenario("olfactory_ambush")
        stats = _make_corridor_episode_stats(
            scenario="olfactory_ambush",
            module_response_by_predator_type={
                "olfactory": {"sensory_cortex": 0.8, "visual_cortex": 0.2},
            },
        )
        trace = [
            {"observation": {"meta": {
                "olfactory_predator_threat": 0.0,
                "visual_predator_threat": 0.0,
                "predator_visible": False,
            }}}
        ]

        score = spec.score_episode(stats, trace)

        self.assertFalse(score.checks["olfactory_threat_detected"].passed)

    def test_visual_hunter_open_field_detection_requires_exported_threat_peak(self) -> None:
        spec = get_scenario("visual_hunter_open_field")
        stats = _make_corridor_episode_stats(
            scenario="visual_hunter_open_field",
            module_response_by_predator_type={
                "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
            },
        )
        trace = [
            {"observation": {"meta": {
                "visual_predator_threat": 0.0,
                "olfactory_predator_threat": 0.0,
                "predator_visible": True,
            }}}
        ]

        score = spec.score_episode(stats, trace)

        self.assertFalse(score.checks["visual_threat_detected"].passed)

    def test_set_predators_rejects_empty_roster(self) -> None:
        world = self._setup_world("visual_hunter_open_field")
        with self.assertRaises(ValueError):
            _set_predators(world, [])

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

        scenario.setup(world)

        self.assertEqual(world.food_positions, [])

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

    def test_exposed_day_foraging_metadata_documents_geometry_hypothesis(self) -> None:
        scenario = get_scenario("exposed_day_foraging")
        self.assertIn("Hypothesis: previous geometry placed lizard on food", scenario.diagnostic_focus)
        self.assertIn("failure_mode", scenario.failure_interpretation)
        self.assertIn("heuristic lizard-food/frontier spacing", scenario.failure_interpretation)
        self.assertIn("Geometry fix", scenario.budget_note)
        self.assertIn("threat-radius heuristic", scenario.budget_note)
        self.assertIn("first food-visible frontier", scenario.budget_note)
        self.assertIn("arbitration or training limits", scenario.budget_note)

    def test_corridor_gauntlet_metadata_documents_failure_modes(self) -> None:
        scenario = get_scenario("corridor_gauntlet")
        self.assertIn("failure_mode classification", scenario.diagnostic_focus)
        self.assertIn("positive food-distance progress", scenario.success_interpretation)
        self.assertIn("zero predator contacts", scenario.success_interpretation)
        self.assertIn("survival to episode end", scenario.success_interpretation)
        for failure_mode in (
            '"success"',
            '"frozen_in_shelter"',
            '"contact_failure_died"',
            '"contact_failure_survived"',
            '"survived_no_progress"',
            '"progress_then_died"',
            '"scoring_mismatch"',
        ):
            with self.subTest(failure_mode=failure_mode):
                self.assertIn(failure_mode, scenario.failure_interpretation)

    def test_food_deprivation_metadata_documents_timing_hypothesis(self) -> None:
        """
        Assert that the `food_deprivation` scenario's metadata documents the expected timing hypothesis and related diagnostic phrases.
        
        Verifies that:
        - `diagnostic_focus` exactly matches the timing hypothesis string about hunger-driven foraging versus a homeostasis death timer and calibrated 4-6 Manhattan-distance food (with farther >=4 as fallback).
        - `failure_interpretation` contains the substring "failure_mode".
        - `budget_note` contains the substring "geometric race condition".
        """
        scenario = get_scenario("food_deprivation")
        self.assertEqual(
            scenario.diagnostic_focus,
            "Primary hypothesis: spider commits to foraging but homeostasis death timer "
            "(~12-15 ticks at hunger 0.96) must beat calibrated 4-6 Manhattan-distance food, "
            "with farther >=4 spawns only as fallback",
        )
        self.assertIn("failure_mode", scenario.failure_interpretation)
        self.assertIn("geometric race condition", scenario.budget_note)


class CorridorGauntletClassifierTest(unittest.TestCase):
    def test_classifies_each_failure_mode_branch(self) -> None:
        """
        Run classifier over a set of representative corridor-gauntlet fixtures and assert each maps to the expected failure-mode label.
        
        Constructs multiple test cases with controlled EpisodeStats, trace metrics, and `full_success` flags (covers success, frozen_in_shelter, contact failures, progress-related deaths, and scoring mismatch) and verifies `_classify_corridor_gauntlet_failure` returns the expected string for each case.
        """
        trace_metrics = {
            "left_shelter": True,
            "shelter_exit_tick": 1,
            "predator_visible_ticks": 0,
            "peak_food_progress": 0.0,
            "death_tick": None,
        }
        cases = {
            "success": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=2.0,
                    alive=True,
                    predator_contacts=0,
                ),
                "trace_metrics": {**trace_metrics, "left_shelter": False},
                "full_success": True,
            },
            "frozen_in_shelter": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=0.0,
                    alive=True,
                    predator_contacts=0,
                ),
                "trace_metrics": {**trace_metrics, "left_shelter": False},
                "full_success": False,
            },
            "contact_failure_died": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=1.0,
                    alive=False,
                    final_health=0.0,
                    predator_contacts=1,
                ),
                "trace_metrics": trace_metrics,
                "full_success": False,
            },
            "contact_failure_survived": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=1.0,
                    alive=True,
                    predator_contacts=1,
                ),
                "trace_metrics": trace_metrics,
                "full_success": False,
            },
            "survived_no_progress": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=0.0,
                    alive=True,
                    predator_contacts=0,
                ),
                "trace_metrics": trace_metrics,
                "full_success": False,
            },
            "progress_then_died": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=1.0,
                    alive=False,
                    final_health=0.0,
                    predator_contacts=0,
                ),
                "trace_metrics": trace_metrics,
                "full_success": False,
            },
            "scoring_mismatch": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=0.0,
                    alive=False,
                    final_health=0.0,
                    predator_contacts=0,
                ),
                "trace_metrics": trace_metrics,
                "full_success": False,
            },
        }
        for expected, fixture in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    _classify_corridor_gauntlet_failure(
                        fixture["stats"],
                        fixture["trace_metrics"],
                        fixture["full_success"],
                    ),
                    expected,
                )


class CorridorGauntletScorerTest(unittest.TestCase):
    EXPECTED_DIAGNOSTIC_KEYS: frozenset[str] = frozenset({
        "failure_mode",
        "left_shelter",
        "shelter_exit_tick",
        "predator_visible_ticks",
        "peak_food_progress",
        "death_tick",
    })

    def _shelter_trace(self) -> list[dict[str, object]]:
        """
        Builds a minimal two-tick corridor-gauntlet trace with the spider remaining at a fixed shelter position.
        
        Returns:
            list[dict[str, object]]: Two trace items (ticks 0 and 1) at shelter position (1, 6), each containing full state and observation fields with `health=1.0` and `food_dist=9`.
        """
        shelter_pos = (1, 6)
        return [
            _make_corridor_trace_item(tick=0, pos=shelter_pos, health=1.0, food_dist=9),
            _make_corridor_trace_item(tick=1, pos=shelter_pos, health=1.0, food_dist=9),
        ]

    def _left_shelter_trace(
        self,
        *,
        final_health: float = 1.0,
        predator_visible: bool = False,
        final_food_dist: int = 9,
    ) -> list[dict[str, object]]:
        """
        Create a minimal two-tick trace representing the spider leaving shelter.
        
        Parameters:
            final_health (float): Health value recorded at tick 1.
            predator_visible (bool): Whether the predator is visible in the tick-1 observation.
            final_food_dist (int): Reported food distance in the tick-1 observation.
        
        Returns:
            list[dict[str, object]]: Two trace entries (tick 0 and tick 1) suitable for corridor-gauntlet scoring tests.
        """
        return [
            _make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=9),
            _make_corridor_trace_item(
                tick=1,
                pos=(4, 6),
                health=final_health,
                food_dist=final_food_dist,
                predator_visible=predator_visible,
            ),
        ]

    def test_score_failure_mode_outputs_from_minimal_traces(self) -> None:
        cases = {
            "frozen_in_shelter": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=0.0,
                    alive=True,
                    predator_contacts=0,
                ),
                "trace": self._shelter_trace(),
            },
            "contact_failure_died": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=1.0,
                    alive=False,
                    final_health=0.0,
                    predator_contacts=1,
                ),
                "trace": self._left_shelter_trace(
                    final_health=0.0,
                    predator_visible=True,
                    final_food_dist=8,
                ),
            },
            "survived_no_progress": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=0.0,
                    alive=True,
                    predator_contacts=0,
                ),
                "trace": self._left_shelter_trace(final_food_dist=9),
            },
            "success": {
                "stats": _make_corridor_episode_stats(
                    food_distance_delta=2.0,
                    alive=True,
                    predator_contacts=0,
                ),
                "trace": self._left_shelter_trace(final_food_dist=7),
            },
        }
        for expected, fixture in cases.items():
            with self.subTest(expected=expected):
                score = _score_corridor_gauntlet(fixture["stats"], fixture["trace"])

                self.assertEqual(score.behavior_metrics["failure_mode"], expected)
                self.assertTrue(
                    self.EXPECTED_DIAGNOSTIC_KEYS.issubset(score.behavior_metrics)
                )


class TraceCorridorMetricsTest(unittest.TestCase):
    """Unit tests for _trace_corridor_metrics added in this PR."""

    def test_empty_trace_returns_default_values(self) -> None:
        """Empty trace should return safe zero/None defaults for every key."""
        metrics = _trace_corridor_metrics([])
        self.assertFalse(metrics["left_shelter"])
        self.assertIsNone(metrics["shelter_exit_tick"])
        self.assertEqual(metrics["predator_visible_ticks"], 0)
        self.assertEqual(metrics["peak_food_progress"], 0.0)
        self.assertIsNone(metrics["death_tick"])

    def test_returns_all_required_keys(self) -> None:
        """Result must contain exactly the five documented keys."""
        metrics = _trace_corridor_metrics([])
        expected_keys = {
            "left_shelter",
            "shelter_exit_tick",
            "predator_visible_ticks",
            "peak_food_progress",
            "death_tick",
        }
        self.assertEqual(expected_keys, set(metrics.keys()))

    def test_single_shelter_item_no_exit(self) -> None:
        """A single shelter-position item produces left_shelter=False."""
        trace = [_make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=9)]
        metrics = _trace_corridor_metrics(trace)
        self.assertFalse(metrics["left_shelter"])
        self.assertIsNone(metrics["shelter_exit_tick"])

    def test_no_predator_visible_ticks_when_all_false(self) -> None:
        """All items with predator_visible=False should yield zero predator_visible_ticks."""
        trace = [
            _make_corridor_trace_item(tick=i, pos=(1, 6), health=1.0, food_dist=9, predator_visible=False)
            for i in range(5)
        ]
        metrics = _trace_corridor_metrics(trace)
        self.assertEqual(metrics["predator_visible_ticks"], 0)

    def test_all_predator_visible_ticks_when_all_true(self) -> None:
        """All items with predator_visible=True should yield count == len(trace)."""
        trace = [
            _make_corridor_trace_item(tick=i, pos=(4, 6), health=1.0, food_dist=8, predator_visible=True)
            for i in range(4)
        ]
        metrics = _trace_corridor_metrics(trace)
        self.assertEqual(metrics["predator_visible_ticks"], 4)

    def test_mixed_predator_visible_ticks_counted_correctly(self) -> None:
        """Only items where predator is visible contribute to predator_visible_ticks."""
        trace = [
            _make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=9, predator_visible=False),
            _make_corridor_trace_item(tick=1, pos=(4, 6), health=1.0, food_dist=7, predator_visible=True),
            _make_corridor_trace_item(tick=2, pos=(5, 6), health=1.0, food_dist=6, predator_visible=False),
            _make_corridor_trace_item(tick=3, pos=(6, 6), health=1.0, food_dist=5, predator_visible=True),
        ]
        metrics = _trace_corridor_metrics(trace)
        self.assertEqual(metrics["predator_visible_ticks"], 2)

    def test_peak_food_progress_clamped_at_zero_when_food_distance_increases(self) -> None:
        """When food distance only increases over trace, peak_food_progress should be 0.0."""
        # food_dist increases: 5 -> 7 -> 9 (regressing)
        trace = [
            _make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=5),
            _make_corridor_trace_item(tick=1, pos=(1, 6), health=1.0, food_dist=7),
            _make_corridor_trace_item(tick=2, pos=(1, 6), health=1.0, food_dist=9),
        ]
        metrics = _trace_corridor_metrics(trace)
        self.assertEqual(metrics["peak_food_progress"], 0.0)

    def test_peak_food_progress_computed_from_best_reduction(self) -> None:
        """peak_food_progress equals max reduction from initial food distance across all ticks."""
        # initial dist=10, then 8, 6, 7 -> best reduction is 4
        trace = [
            _make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=10),
            _make_corridor_trace_item(tick=1, pos=(3, 6), health=1.0, food_dist=8),
            _make_corridor_trace_item(tick=2, pos=(5, 6), health=1.0, food_dist=6),
            _make_corridor_trace_item(tick=3, pos=(4, 6), health=1.0, food_dist=7),
        ]
        metrics = _trace_corridor_metrics(trace)
        self.assertEqual(metrics["peak_food_progress"], 4.0)

    def test_death_tick_is_none_when_health_stays_positive(self) -> None:
        """No death should be detected when health remains above zero throughout."""
        trace = [
            _make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=9),
            _make_corridor_trace_item(tick=1, pos=(4, 6), health=0.5, food_dist=7),
            _make_corridor_trace_item(tick=2, pos=(5, 6), health=0.3, food_dist=6),
        ]
        metrics = _trace_corridor_metrics(trace)
        self.assertIsNone(metrics["death_tick"])

    def test_death_tick_detected_when_health_drops_to_zero(self) -> None:
        """death_tick should be the tick index when health first reaches zero."""
        trace = [
            _make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=9),
            _make_corridor_trace_item(tick=1, pos=(4, 6), health=0.5, food_dist=7),
            _make_corridor_trace_item(tick=2, pos=(5, 6), health=0.0, food_dist=6),
        ]
        metrics = _trace_corridor_metrics(trace)
        self.assertEqual(metrics["death_tick"], 2)

    def test_left_shelter_is_bool_type(self) -> None:
        """left_shelter must always be a plain Python bool."""
        for trace in ([], [_make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=9)]):
            with self.subTest(trace_len=len(trace)):
                metrics = _trace_corridor_metrics(trace)
                self.assertIs(type(metrics["left_shelter"]), bool)

    def test_peak_food_progress_is_float_type(self) -> None:
        """peak_food_progress must always be a Python float."""
        metrics = _trace_corridor_metrics([])
        self.assertIsInstance(metrics["peak_food_progress"], float)


class CorridorGauntletClassifierBoundaryTest(unittest.TestCase):
    """Boundary and edge-case tests for _classify_corridor_gauntlet_failure."""

    def _base_trace_metrics(self, *, left_shelter: bool = True) -> dict[str, object]:
        return {
            "left_shelter": left_shelter,
            "shelter_exit_tick": 1 if left_shelter else None,
            "predator_visible_ticks": 0,
            "peak_food_progress": 0.0,
            "death_tick": None,
        }

    def test_full_success_overrides_all_other_conditions(self) -> None:
        """full_success=True must always return 'success' regardless of stats."""
        stats = _make_corridor_episode_stats(
            food_distance_delta=-5.0,
            alive=False,
            predator_contacts=99,
        )
        result = _classify_corridor_gauntlet_failure(
            stats, self._base_trace_metrics(left_shelter=False), full_success=True
        )
        self.assertEqual(result, "success")

    def test_frozen_in_shelter_when_left_shelter_is_false_value(self) -> None:
        """Falsy trace left_shelter (False) must yield frozen_in_shelter."""
        stats = _make_corridor_episode_stats(food_distance_delta=0.0, alive=True, predator_contacts=0)
        for falsy in (False, 0, None, ""):
            with self.subTest(left_shelter=falsy):
                tm = {**self._base_trace_metrics(), "left_shelter": falsy}
                result = _classify_corridor_gauntlet_failure(stats, tm, full_success=False)
                self.assertEqual(result, "frozen_in_shelter")

    def test_survived_no_progress_when_food_delta_exactly_zero(self) -> None:
        """Boundary: food_distance_delta==0.0 with alive and no contacts → survived_no_progress."""
        stats = _make_corridor_episode_stats(food_distance_delta=0.0, alive=True, predator_contacts=0)
        result = _classify_corridor_gauntlet_failure(
            stats, self._base_trace_metrics(), full_success=False
        )
        self.assertEqual(result, "survived_no_progress")

    def test_survived_no_progress_when_food_delta_negative(self) -> None:
        """Negative food_distance_delta with alive and no contacts → survived_no_progress."""
        stats = _make_corridor_episode_stats(food_distance_delta=-3.0, alive=True, predator_contacts=0)
        result = _classify_corridor_gauntlet_failure(
            stats, self._base_trace_metrics(), full_success=False
        )
        self.assertEqual(result, "survived_no_progress")

    def test_contact_failure_died_with_multiple_contacts(self) -> None:
        """Multiple predator contacts with dead spider → contact_failure_died."""
        stats = _make_corridor_episode_stats(
            food_distance_delta=1.0, alive=False, predator_contacts=5
        )
        result = _classify_corridor_gauntlet_failure(
            stats, self._base_trace_metrics(), full_success=False
        )
        self.assertEqual(result, "contact_failure_died")

    def test_contact_failure_survived_with_food_delta_zero(self) -> None:
        """Contact with alive spider and zero food progress → contact_failure_survived."""
        stats = _make_corridor_episode_stats(
            food_distance_delta=0.0, alive=True, predator_contacts=1
        )
        result = _classify_corridor_gauntlet_failure(
            stats, self._base_trace_metrics(), full_success=False
        )
        self.assertEqual(result, "contact_failure_survived")

    def test_progress_then_died_requires_positive_delta_no_contacts(self) -> None:
        """Minimal positive delta, dead, no contacts → progress_then_died."""
        stats = _make_corridor_episode_stats(
            food_distance_delta=0.001, alive=False, predator_contacts=0
        )
        result = _classify_corridor_gauntlet_failure(
            stats, self._base_trace_metrics(), full_success=False
        )
        self.assertEqual(result, "progress_then_died")

    def test_scoring_mismatch_when_dead_no_progress_no_contacts_left_shelter(self) -> None:
        """Dead, no progress, no contacts, left shelter → scoring_mismatch."""
        stats = _make_corridor_episode_stats(
            food_distance_delta=0.0, alive=False, predator_contacts=0
        )
        result = _classify_corridor_gauntlet_failure(
            stats, self._base_trace_metrics(), full_success=False
        )
        self.assertEqual(result, "scoring_mismatch")

    def test_left_shelter_truthy_non_bool_triggers_correct_branch(self) -> None:
        """Truthy non-bool left_shelter (1, 'yes') should pass the shelter-exit check."""
        stats = _make_corridor_episode_stats(food_distance_delta=0.0, alive=True, predator_contacts=0)
        for truthy in (1, "yes", [True], (1,)):
            with self.subTest(left_shelter=truthy):
                tm = {**self._base_trace_metrics(), "left_shelter": truthy}
                result = _classify_corridor_gauntlet_failure(stats, tm, full_success=False)
                self.assertEqual(result, "survived_no_progress")


class CorridorGauntletScorerLegacyKeysTest(unittest.TestCase):
    """Tests that _score_corridor_gauntlet emits all expected behavior_metric keys."""

    LEGACY_KEYS: frozenset[str] = frozenset({
        "food_distance_delta",
        "predator_contacts",
        "alive",
        "predator_mode_transitions",
    })
    TRACE_KEYS: frozenset[str] = frozenset({
        "failure_mode",
        "left_shelter",
        "shelter_exit_tick",
        "predator_visible_ticks",
        "peak_food_progress",
        "death_tick",
    })

    def _minimal_trace(self) -> list[dict[str, object]]:
        return [_make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=9)]

    def test_legacy_keys_present_in_behavior_metrics(self) -> None:
        """All pre-existing metric keys must still appear in the updated scorer output."""
        stats = _make_corridor_episode_stats(food_distance_delta=0.0, alive=True, predator_contacts=0)
        score = _score_corridor_gauntlet(stats, self._minimal_trace())
        for key in self.LEGACY_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, score.behavior_metrics)

    def test_trace_keys_present_in_behavior_metrics(self) -> None:
        """All new trace-derived keys must appear in the updated scorer output."""
        stats = _make_corridor_episode_stats(food_distance_delta=0.0, alive=True, predator_contacts=0)
        score = _score_corridor_gauntlet(stats, self._minimal_trace())
        for key in self.TRACE_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, score.behavior_metrics)

    def test_empty_trace_does_not_raise(self) -> None:
        """_score_corridor_gauntlet must handle an empty trace without error."""
        stats = _make_corridor_episode_stats(food_distance_delta=0.0, alive=True, predator_contacts=0)
        score = _score_corridor_gauntlet(stats, [])
        self.assertIn("failure_mode", score.behavior_metrics)

    def test_full_success_sets_failure_mode_to_success(self) -> None:
        """When all three checks pass, failure_mode must be 'success'."""
        stats = _make_corridor_episode_stats(
            food_distance_delta=3.0,
            alive=True,
            predator_contacts=0,
        )
        trace = [
            _make_corridor_trace_item(tick=0, pos=(1, 6), health=1.0, food_dist=9),
            _make_corridor_trace_item(tick=1, pos=(5, 6), health=1.0, food_dist=6),
        ]
        score = _score_corridor_gauntlet(stats, trace)
        self.assertTrue(score.success)
        self.assertEqual(score.behavior_metrics["failure_mode"], "success")

    def test_behavior_metrics_food_distance_delta_matches_stats(self) -> None:
        """food_distance_delta in behavior_metrics must match stats value."""
        stats = _make_corridor_episode_stats(food_distance_delta=5.5, alive=True, predator_contacts=0)
        score = _score_corridor_gauntlet(stats, self._minimal_trace())
        self.assertAlmostEqual(score.behavior_metrics["food_distance_delta"], 5.5)

    def test_behavior_metrics_predator_contacts_matches_stats(self) -> None:
        """predator_contacts in behavior_metrics must match stats value."""
        stats = _make_corridor_episode_stats(food_distance_delta=0.0, alive=False, predator_contacts=3)
        score = _score_corridor_gauntlet(stats, self._minimal_trace())
        self.assertEqual(score.behavior_metrics["predator_contacts"], 3)

    def test_three_checks_are_emitted(self) -> None:
        """Scorer must produce exactly three behavior checks."""
        stats = _make_corridor_episode_stats(food_distance_delta=0.0, alive=True, predator_contacts=0)
        score = _score_corridor_gauntlet(stats, self._minimal_trace())
        self.assertEqual(len(score.checks), 3)

    def test_checks_reflect_stats_correctness(self) -> None:
        """When stats satisfy all pass conditions, all checks must be marked passed."""
        stats = _make_corridor_episode_stats(
            food_distance_delta=2.0,
            alive=True,
            predator_contacts=0,
        )
        score = _score_corridor_gauntlet(stats, self._minimal_trace())
        for name, check in score.checks.items():
            with self.subTest(check=name):
                self.assertTrue(check.passed)

    def test_scenario_name_matches_corridor_gauntlet(self) -> None:
        """score.scenario must be 'corridor_gauntlet'."""
        stats = _make_corridor_episode_stats(food_distance_delta=0.0, alive=True, predator_contacts=0)
        score = _score_corridor_gauntlet(stats, self._minimal_trace())
        self.assertEqual(score.scenario, "corridor_gauntlet")


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


class MultiPredatorScenarioMetadataTest(unittest.TestCase):
    """Tests for the new predator-specialization scenarios."""

    def test_scenario_metadata_and_behavior_checks(self) -> None:
        cases = [
            ("visual_olfactory_pincer", "exposed_feeding_ground"),
            ("olfactory_ambush", "entrance_funnel"),
            ("visual_hunter_open_field", "exposed_feeding_ground"),
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


# ---------------------------------------------------------------------------
# open_field_foraging setup: robustness and physiology tests
# ---------------------------------------------------------------------------

class OpenFieldForagingSetupTest(unittest.TestCase):
    """Tests for the _open_field_foraging scenario setup and _open_field_foraging_food_cell."""

    def _setup_world(self, seed: int = 101) -> "SpiderWorld":
        scenario = get_scenario("open_field_foraging")
        world = SpiderWorld(seed=seed, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=seed)
        scenario.setup(world)
        return world

    def _move_towards(self, world: "SpiderWorld", target: tuple[int, int]) -> str:
        from collections import deque
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


if __name__ == "__main__":
    unittest.main()
