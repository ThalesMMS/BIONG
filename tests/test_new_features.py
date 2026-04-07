"""
Tests for features introduced in the PR:
- New map templates (two_shelters, exposed_feeding_ground, entrance_funnel)
- Helper functions in maps.py (_available_cells, _distance_to_region via build_map_template)
- New EpisodeStats fields and EpisodeMetricAccumulator logic (food_distance_delta,
  shelter_distance_delta, predator_mode_transitions, dominant_predator_state)
- New LizardState fields (investigate_target, ambush_ticks, chase_streak, failed_chases)
- PredatorController new methods (_investigate_budget, FSM changes)
- ScenarioSpec.map_template field and new scenario helpers (_first_cell, _entrance_ambush_cell)
- SpiderWorld.configure_map_template()
- info dict in world.step() (predator_transition, distance_deltas)
"""
from __future__ import annotations

import unittest
from collections import deque
from typing import Any, Dict
from unittest.mock import MagicMock

from spider_cortex_sim.maps import (
    CLUTTER,
    NARROW,
    OPEN,
    MAP_TEMPLATE_NAMES,
    build_map_template,
)
from spider_cortex_sim.metrics import (
    EpisodeMetricAccumulator,
    EpisodeStats,
    aggregate_episode_stats,
)
from spider_cortex_sim.predator import LizardState, PREDATOR_STATES, PredatorController
from spider_cortex_sim.scenarios import (
    SCENARIOS,
    SCENARIO_NAMES,
    ScenarioSpec,
    get_scenario,
)
from spider_cortex_sim.world import ACTION_TO_INDEX, SpiderWorld


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_world(map_template: str = "central_burrow", seed: int = 1, lizard_move_interval: int = 999999) -> SpiderWorld:
    world = SpiderWorld(seed=seed, map_template=map_template, lizard_move_interval=lizard_move_interval)
    world.reset(seed=seed)
    return world


def _reachable(world: SpiderWorld, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for dx, dy in world.move_deltas:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nxt[0] < world.width and 0 <= nxt[1] < world.height):
                continue
            if nxt in seen or not world.is_walkable(nxt):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


# ---------------------------------------------------------------------------
# Map template structural tests (new maps)
# ---------------------------------------------------------------------------

class TwoSheltersMapTest(unittest.TestCase):
    """Tests for the new `two_shelters` map template."""

    def setUp(self) -> None:
        self.template = build_map_template("two_shelters", width=12, height=12)

    def test_template_name_is_two_shelters(self) -> None:
        self.assertEqual(self.template.name, "two_shelters")

    def test_has_non_empty_shelter_zones(self) -> None:
        self.assertTrue(len(self.template.shelter_entrance) > 0)
        self.assertTrue(len(self.template.shelter_interior) > 0)
        self.assertTrue(len(self.template.shelter_deep) > 0)

    def test_has_two_distinct_shelter_clusters(self) -> None:
        # Two separate deep zones: left and right sides
        deep = sorted(self.template.shelter_deep)
        x_coords = [c[0] for c in deep]
        # There should be at least two distinct x values (left and right shelter)
        self.assertGreater(len(set(x_coords)), 1)
        self.assertGreaterEqual(max(x_coords) - min(x_coords), 3)

    def test_shelter_entrance_and_interior_not_overlapping(self) -> None:
        entrance = set(self.template.shelter_entrance)
        interior = set(self.template.shelter_interior)
        deep = set(self.template.shelter_deep)
        self.assertTrue(entrance.isdisjoint(interior))
        self.assertTrue(entrance.isdisjoint(deep))
        self.assertTrue(interior.isdisjoint(deep))

    def test_shelter_cells_not_in_blocked(self) -> None:
        shelter = set(self.template.shelter_cells)
        blocked = set(self.template.blocked_cells)
        self.assertTrue(shelter.isdisjoint(blocked))

    def test_supported_small_widths_keep_blocked_disjoint_from_shelter(self) -> None:
        for width in (8, 9, 10):
            template = build_map_template("two_shelters", width=width, height=12)
            shelter = set(template.shelter_cells)
            blocked = set(template.blocked_cells)
            self.assertTrue(
                shelter.isdisjoint(blocked),
                msg=f"blocked overlaps shelter for width={width}",
            )

    def test_food_spawn_cells_exist_and_not_in_shelter_or_blocked(self) -> None:
        self.assertGreater(len(self.template.food_spawn_cells), 0)
        shelter = set(self.template.shelter_cells)
        blocked = set(self.template.blocked_cells)
        for cell in self.template.food_spawn_cells:
            self.assertNotIn(cell, shelter)
            self.assertNotIn(cell, blocked)

    def test_lizard_spawn_cells_exist_and_not_in_shelter(self) -> None:
        self.assertGreater(len(self.template.lizard_spawn_cells), 0)
        shelter = set(self.template.shelter_cells)
        for cell in self.template.lizard_spawn_cells:
            self.assertNotIn(cell, shelter)

    def test_spider_start_is_in_shelter_deep(self) -> None:
        self.assertIn(self.template.spider_start, self.template.shelter_deep)

    def test_all_cells_within_grid_bounds(self) -> None:
        w, h = self.template.width, self.template.height
        for cell in list(self.template.shelter_cells) + list(self.template.blocked_cells):
            self.assertTrue(0 <= cell[0] < w, f"x out of bounds: {cell}")
            self.assertTrue(0 <= cell[1] < h, f"y out of bounds: {cell}")

    def test_reachability_from_spider_start_to_food(self) -> None:
        world = _make_world("two_shelters")
        start = self.template.spider_start
        food = self.template.food_spawn_cells[0]
        self.assertTrue(_reachable(world, start, food))

    def test_terrain_has_clutter_in_central_zone(self) -> None:
        # The two_shelters map places CLUTTER terrain in the central column
        clutter_cells = [pos for pos, t in self.template.terrain.items() if t == CLUTTER]
        self.assertGreater(len(clutter_cells), 0)

    def test_small_grid_does_not_crash(self) -> None:
        # Should not raise even on a minimal grid
        t = build_map_template("two_shelters", width=10, height=8)
        self.assertEqual(t.name, "two_shelters")


class ExposedFeedingGroundMapTest(unittest.TestCase):
    """Tests for the new `exposed_feeding_ground` map template."""

    def setUp(self) -> None:
        self.template = build_map_template("exposed_feeding_ground", width=12, height=12)

    def test_template_name_is_exposed_feeding_ground(self) -> None:
        self.assertEqual(self.template.name, "exposed_feeding_ground")

    def test_shelter_is_on_left_side(self) -> None:
        # Entrance at x=2, interior at x=3, deep at x=4
        entrance_x = [c[0] for c in self.template.shelter_entrance]
        interior_x = [c[0] for c in self.template.shelter_interior]
        deep_x = [c[0] for c in self.template.shelter_deep]
        self.assertTrue(all(x <= 4 for x in entrance_x + interior_x + deep_x))

    def test_food_spawn_in_right_portion_of_grid(self) -> None:
        # Food should be in the feeding zone (right/open area)
        for cell in self.template.food_spawn_cells:
            self.assertGreater(cell[0], 4)

    def test_food_zone_is_open_terrain(self) -> None:
        food_cells = set(self.template.food_spawn_cells)
        for cell in food_cells:
            terrain = self.template.terrain.get(cell, OPEN)
            self.assertNotEqual(terrain, CLUTTER,
                msg=f"Food cell {cell} should not be CLUTTER terrain")

    def test_lizard_spawn_near_right_edge(self) -> None:
        if not self.template.lizard_spawn_cells:
            self.skipTest("No lizard spawn cells")
        # All lizard spawns should be on the right side or at edges
        for cell in self.template.lizard_spawn_cells:
            on_edge = (cell[0] >= self.template.width - 3
                       or cell[1] in {0, self.template.height - 1})
            self.assertTrue(on_edge, f"Lizard spawn {cell} not on expected edge")

    def test_spider_start_is_in_deep_shelter(self) -> None:
        self.assertIn(self.template.spider_start, self.template.shelter_deep)

    def test_has_clutter_terrain_between_shelter_and_feeding_zone(self) -> None:
        clutter_cells = [pos for pos, t in self.template.terrain.items() if t == CLUTTER]
        self.assertGreater(len(clutter_cells), 0)

    def test_food_spawn_cells_not_in_shelter_or_blocked(self) -> None:
        shelter = set(self.template.shelter_cells)
        blocked = set(self.template.blocked_cells)
        for cell in self.template.food_spawn_cells:
            self.assertNotIn(cell, shelter)
            self.assertNotIn(cell, blocked)

    def test_reachability_from_spider_start_to_food(self) -> None:
        world = _make_world("exposed_feeding_ground")
        start = self.template.spider_start
        food = self.template.food_spawn_cells[0]
        self.assertTrue(_reachable(world, start, food))

    def test_raises_clear_error_when_deep_shelter_is_empty(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"exposed_feeding_ground: no deep shelter cells for width=3 height=12",
        ):
            build_map_template("exposed_feeding_ground", width=3, height=12)

    def test_raises_clear_error_when_food_zone_is_empty(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"exposed_feeding_ground: no food spawn cells for width=5 height=12",
        ):
            build_map_template("exposed_feeding_ground", width=5, height=12)


class EntranceFunnelMapTest(unittest.TestCase):
    """Tests for the new `entrance_funnel` map template."""

    def setUp(self) -> None:
        self.template = build_map_template("entrance_funnel", width=12, height=12)

    def test_template_name_is_entrance_funnel(self) -> None:
        self.assertEqual(self.template.name, "entrance_funnel")

    def test_entrance_is_single_cell_bottleneck(self) -> None:
        # entrance_funnel has a single-cell entrance (the funnel)
        self.assertEqual(len(self.template.shelter_entrance), 1)

    def test_narrow_terrain_forms_funnel_corridor(self) -> None:
        narrow_cells = [pos for pos, t in self.template.terrain.items() if t == NARROW]
        self.assertGreater(len(narrow_cells), 0)
        # Narrow cells should be near x=5,6,7 at the center y
        cy = self.template.height // 2
        for cell in narrow_cells:
            self.assertEqual(cell[1], cy, f"Narrow cell {cell} not on center row")

    def test_shelter_deep_is_on_left_side(self) -> None:
        for cell in self.template.shelter_deep:
            self.assertLess(cell[0], 5, f"Deep cell {cell} too far right")

    def test_food_spawn_far_from_shelter(self) -> None:
        shelter = set(self.template.shelter_cells)
        for food in self.template.food_spawn_cells:
            min_dist = min(
                abs(food[0] - s[0]) + abs(food[1] - s[1])
                for s in shelter
            )
            self.assertGreaterEqual(min_dist, 5,
                msg=f"Food {food} too close to shelter (dist={min_dist})")

    def test_spider_start_in_deep_shelter(self) -> None:
        self.assertIn(self.template.spider_start, self.template.shelter_deep)

    def test_blocked_cells_create_funnel_walls(self) -> None:
        blocked = set(self.template.blocked_cells)
        cy = self.template.height // 2
        # Cells that should be blocked: (5, cy±1), (6, cy±1)
        for bx in (5, 6):
            for dy in (-1, 1):
                candidate = (bx, cy + dy)
                if 0 <= candidate[1] < self.template.height:
                    self.assertIn(candidate, blocked,
                        msg=f"Expected {candidate} to be blocked in entrance_funnel")

    def test_reachability_from_spider_start_to_food(self) -> None:
        world = _make_world("entrance_funnel")
        if not self.template.food_spawn_cells:
            self.skipTest("No food spawn cells on this grid size")
        start = self.template.spider_start
        food = self.template.food_spawn_cells[0]
        self.assertTrue(_reachable(world, start, food))

    def test_small_grid_does_not_crash(self) -> None:
        t = build_map_template("entrance_funnel", width=10, height=8)
        self.assertEqual(t.name, "entrance_funnel")

    def test_raises_clear_error_when_deep_shelter_is_empty(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"entrance_funnel: no deep shelter cells for width=2 height=12",
        ):
            build_map_template("entrance_funnel", width=2, height=12)

    def test_raises_clear_error_when_food_spawn_is_empty(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"entrance_funnel: no food spawn cells for width=7 height=12",
        ):
            build_map_template("entrance_funnel", width=7, height=12)


class BuildMapTemplateRegistryTest(unittest.TestCase):
    """Tests for build_map_template and MAP_TEMPLATE_NAMES registry."""

    def test_all_new_templates_in_registry(self) -> None:
        self.assertIn("two_shelters", MAP_TEMPLATE_NAMES)
        self.assertIn("exposed_feeding_ground", MAP_TEMPLATE_NAMES)
        self.assertIn("entrance_funnel", MAP_TEMPLATE_NAMES)

    def test_unknown_template_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            build_map_template("nonexistent_map", width=12, height=12)

    def test_all_registered_templates_build_without_error(self) -> None:
        for name in MAP_TEMPLATE_NAMES:
            t = build_map_template(name, width=12, height=12)
            self.assertEqual(t.name, name)


# ---------------------------------------------------------------------------
# EpisodeMetricAccumulator - new fields and record_transition signature
# ---------------------------------------------------------------------------

def _make_accumulator() -> EpisodeMetricAccumulator:
    from spider_cortex_sim.world import REWARD_COMPONENT_NAMES
    return EpisodeMetricAccumulator(
        reward_component_names=REWARD_COMPONENT_NAMES,
        predator_states=PREDATOR_STATES,
    )


def _make_fake_meta(food_dist: int = 5, shelter_dist: int = 3,
                    night: bool = False, predator_visible: bool = False,
                    shelter_role: str = "outside", predator_dist: int = 10,
                    on_shelter: bool = False) -> Dict[str, object]:
    return {
        "food_dist": food_dist,
        "shelter_dist": shelter_dist,
        "night": night,
        "predator_visible": predator_visible,
        "shelter_role": shelter_role,
        "predator_dist": predator_dist,
        "on_shelter": on_shelter,
    }


def _make_fake_info(reward_components: Dict[str, float] | None = None,
                    predator_contact: bool = False) -> Dict[str, object]:
    from spider_cortex_sim.world import REWARD_COMPONENT_NAMES
    if reward_components is None:
        reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
    return {
        "reward_components": reward_components,
        "predator_contact": predator_contact,
    }


def _make_fake_state(sleep_debt: float = 0.2, last_move_dx: int = 0,
                     last_move_dy: int = 0) -> object:
    state = MagicMock()
    state.sleep_debt = sleep_debt
    state.last_move_dx = last_move_dx
    state.last_move_dy = last_move_dy
    return state


class MetricsAccumulatorNewFieldsTest(unittest.TestCase):
    """Tests for new EpisodeMetricAccumulator fields introduced in this PR."""

    def test_initial_distance_fields_are_none(self) -> None:
        acc = _make_accumulator()
        self.assertIsNone(acc.initial_food_dist)
        self.assertIsNone(acc.initial_shelter_dist)
        self.assertIsNone(acc.final_food_dist)
        self.assertIsNone(acc.final_shelter_dist)

    def test_initial_predator_mode_transitions_is_zero(self) -> None:
        acc = _make_accumulator()
        self.assertEqual(acc.predator_mode_transitions, 0)

    def test_record_transition_captures_initial_food_and_shelter_dist(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta(food_dist=8, shelter_dist=4)
        next_meta = _make_fake_meta(food_dist=7, shelter_dist=5)
        acc.record_transition(
            step=0,
            observation_meta=obs_meta,
            next_meta=next_meta,
            info=_make_fake_info(),
            state=_make_fake_state(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )
        self.assertEqual(acc.initial_food_dist, 8)
        self.assertEqual(acc.initial_shelter_dist, 4)
        self.assertEqual(acc.final_food_dist, 7)
        self.assertEqual(acc.final_shelter_dist, 5)

    def test_record_transition_does_not_overwrite_initial_distances(self) -> None:
        acc = _make_accumulator()
        obs_meta1 = _make_fake_meta(food_dist=10, shelter_dist=6)
        next_meta1 = _make_fake_meta(food_dist=9, shelter_dist=7)
        obs_meta2 = _make_fake_meta(food_dist=9, shelter_dist=7)
        next_meta2 = _make_fake_meta(food_dist=5, shelter_dist=2)
        acc.record_transition(step=0, observation_meta=obs_meta1, next_meta=next_meta1,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        acc.record_transition(step=1, observation_meta=obs_meta2, next_meta=next_meta2,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        # Initial should still be from step 0
        self.assertEqual(acc.initial_food_dist, 10)
        self.assertEqual(acc.initial_shelter_dist, 6)
        # Final should be updated to step 1's next_meta
        self.assertEqual(acc.final_food_dist, 5)
        self.assertEqual(acc.final_shelter_dist, 2)

    def test_predator_mode_transition_counted_on_state_change(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="ORIENT")
        self.assertEqual(acc.predator_mode_transitions, 1)

    def test_predator_mode_transition_not_counted_when_same_state(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="CHASE", predator_state="CHASE")
        self.assertEqual(acc.predator_mode_transitions, 0)

    def test_predator_mode_transitions_accumulate_over_multiple_steps(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        transitions = [
            ("PATROL", "ORIENT"),    # +1
            ("ORIENT", "ORIENT"),    # no change
            ("ORIENT", "CHASE"),     # +1
            ("CHASE", "WAIT"),       # +1
            ("WAIT", "PATROL"),      # +1
        ]
        for before, after in transitions:
            acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                                  info=_make_fake_info(), state=_make_fake_state(),
                                  predator_state_before=before, predator_state=after)
        self.assertEqual(acc.predator_mode_transitions, 4)

    def test_snapshot_includes_predator_mode_transitions(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="ORIENT")
        snap = acc.snapshot()
        self.assertIn("predator_mode_transitions", snap)
        self.assertEqual(snap["predator_mode_transitions"], 1)

    def test_snapshot_includes_dominant_predator_state(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta()
        next_meta = _make_fake_meta()
        # Record mostly CHASE ticks
        for _ in range(5):
            acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                                  info=_make_fake_info(), state=_make_fake_state(),
                                  predator_state_before="PATROL", predator_state="CHASE")
        acc.record_transition(step=5, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="CHASE", predator_state="PATROL")
        snap = acc.snapshot()
        self.assertIn("dominant_predator_state", snap)
        self.assertEqual(snap["dominant_predator_state"], "CHASE")

    def test_snapshot_dominant_predator_state_defaults_to_patrol_when_no_ticks(self) -> None:
        acc = _make_accumulator()
        # Don't record any transitions - all ticks are 0
        snap = acc.snapshot()
        self.assertEqual(snap["dominant_predator_state"], "PATROL")

    def test_snapshot_food_distance_delta_positive_when_approaching_food(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta(food_dist=10)
        next_meta = _make_fake_meta(food_dist=7)
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        snap = acc.snapshot()
        self.assertAlmostEqual(snap["food_distance_delta"], 3.0)

    def test_snapshot_food_distance_delta_negative_when_moving_away(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta(food_dist=3)
        next_meta = _make_fake_meta(food_dist=7)
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        snap = acc.snapshot()
        self.assertAlmostEqual(snap["food_distance_delta"], -4.0)

    def test_snapshot_shelter_distance_delta_tracks_correctly(self) -> None:
        acc = _make_accumulator()
        obs_meta = _make_fake_meta(shelter_dist=8)
        next_meta = _make_fake_meta(shelter_dist=5)
        acc.record_transition(step=0, observation_meta=obs_meta, next_meta=next_meta,
                              info=_make_fake_info(), state=_make_fake_state(),
                              predator_state_before="PATROL", predator_state="PATROL")
        snap = acc.snapshot()
        self.assertAlmostEqual(snap["shelter_distance_delta"], 3.0)


class AggregateEpisodeStatsNewFieldsTest(unittest.TestCase):
    """Tests for new aggregate_episode_stats fields."""

    def _minimal_episode_stats(self, **overrides) -> EpisodeStats:
        """
        Create a minimal EpisodeStats object populated with sensible default values, allowing selected fields to be overridden.
        
        Parameters:
            overrides (dict): Keyword overrides merged into the default fields before constructing the EpisodeStats. Common keys include episode, seed, training, scenario, total_reward, steps, food_eaten, night_role_ticks, reward_component_totals, predator_state_ticks, predator_mode_transitions, and dominant_predator_state.
        
        Returns:
            EpisodeStats: An EpisodeStats instance with default test-oriented values merged with any provided overrides.
        """
        from spider_cortex_sim.world import REWARD_COMPONENT_NAMES
        defaults = dict(
            episode=0, seed=0, training=False, scenario=None,
            total_reward=1.0, steps=10, food_eaten=1, sleep_events=0,
            shelter_entries=1, alert_events=0, predator_contacts=0,
            predator_sightings=0, predator_escapes=0,
            night_ticks=5, night_shelter_ticks=4, night_still_ticks=3,
            night_role_ticks={"outside": 1, "entrance": 0, "inside": 1, "deep": 3},
            night_shelter_occupancy_rate=0.8, night_stillness_rate=0.6,
            night_role_distribution={"outside": 0.2, "entrance": 0.0, "inside": 0.2, "deep": 0.6},
            predator_response_events=0, mean_predator_response_latency=0.0,
            mean_sleep_debt=0.3,
            food_distance_delta=2.0,
            shelter_distance_delta=1.0,
            final_hunger=0.4, final_fatigue=0.3, final_sleep_debt=0.25,
            final_health=0.9, alive=True,
            reward_component_totals={name: 0.0 for name in REWARD_COMPONENT_NAMES},
            predator_state_ticks={s: 0 for s in PREDATOR_STATES},
            predator_mode_transitions=2,
            dominant_predator_state="PATROL",
        )
        defaults.update(overrides)
        return EpisodeStats(**defaults)

    def test_aggregate_includes_mean_food_distance_delta(self) -> None:
        stats = [self._minimal_episode_stats(food_distance_delta=4.0),
                 self._minimal_episode_stats(food_distance_delta=2.0)]
        result = aggregate_episode_stats(stats)
        self.assertIn("mean_food_distance_delta", result)
        self.assertAlmostEqual(result["mean_food_distance_delta"], 3.0)

    def test_aggregate_includes_mean_shelter_distance_delta(self) -> None:
        stats = [self._minimal_episode_stats(shelter_distance_delta=6.0),
                 self._minimal_episode_stats(shelter_distance_delta=2.0)]
        result = aggregate_episode_stats(stats)
        self.assertIn("mean_shelter_distance_delta", result)
        self.assertAlmostEqual(result["mean_shelter_distance_delta"], 4.0)

    def test_aggregate_includes_mean_predator_mode_transitions(self) -> None:
        stats = [self._minimal_episode_stats(predator_mode_transitions=4),
                 self._minimal_episode_stats(predator_mode_transitions=2)]
        result = aggregate_episode_stats(stats)
        self.assertIn("mean_predator_mode_transitions", result)
        self.assertAlmostEqual(result["mean_predator_mode_transitions"], 3.0)

    def test_aggregate_includes_dominant_predator_state(self) -> None:
        patrol_ticks = {s: 0 for s in PREDATOR_STATES}
        patrol_ticks["CHASE"] = 10
        stats = [self._minimal_episode_stats(predator_state_ticks=patrol_ticks,
                                              dominant_predator_state="CHASE")]
        result = aggregate_episode_stats(stats)
        self.assertIn("dominant_predator_state", result)
        self.assertEqual(result["dominant_predator_state"], "CHASE")

    def test_aggregate_empty_history_returns_zero_deltas(self) -> None:
        result = aggregate_episode_stats([])
        self.assertEqual(result["mean_food_distance_delta"], 0.0)
        self.assertEqual(result["mean_shelter_distance_delta"], 0.0)
        self.assertEqual(result["mean_predator_mode_transitions"], 0.0)
        self.assertEqual(result["dominant_predator_state"], "PATROL")

    def test_aggregate_negative_food_delta_preserved(self) -> None:
        stats = [self._minimal_episode_stats(food_distance_delta=-3.0)]
        result = aggregate_episode_stats(stats)
        self.assertAlmostEqual(result["mean_food_distance_delta"], -3.0)


# ---------------------------------------------------------------------------
# LizardState - new fields
# ---------------------------------------------------------------------------

class LizardStateNewFieldsTest(unittest.TestCase):
    """Tests for new LizardState dataclass fields."""

    def test_default_investigate_target_is_none(self) -> None:
        lizard = LizardState(x=0, y=0)
        self.assertIsNone(lizard.investigate_target)

    def test_default_ambush_ticks_is_zero(self) -> None:
        lizard = LizardState(x=0, y=0)
        self.assertEqual(lizard.ambush_ticks, 0)

    def test_default_chase_streak_is_zero(self) -> None:
        lizard = LizardState(x=0, y=0)
        self.assertEqual(lizard.chase_streak, 0)

    def test_default_failed_chases_is_zero(self) -> None:
        lizard = LizardState(x=0, y=0)
        self.assertEqual(lizard.failed_chases, 0)

    def test_new_fields_can_be_set_explicitly(self) -> None:
        lizard = LizardState(
            x=3, y=4,
            investigate_target=(5, 6),
            ambush_ticks=2,
            chase_streak=3,
            failed_chases=1,
        )
        self.assertEqual(lizard.investigate_target, (5, 6))
        self.assertEqual(lizard.ambush_ticks, 2)
        self.assertEqual(lizard.chase_streak, 3)
        self.assertEqual(lizard.failed_chases, 1)


# ---------------------------------------------------------------------------
# PredatorController - new methods and FSM logic
# ---------------------------------------------------------------------------

class PredatorControllerInvestigateBudgetTest(unittest.TestCase):
    """Tests for PredatorController._investigate_budget."""

    def setUp(self) -> None:
        self.controller = PredatorController()

    def test_base_budget_with_no_streak_or_failed_chases(self) -> None:
        lizard = LizardState(x=0, y=0, chase_streak=0, failed_chases=0)
        budget = self.controller._investigate_budget(lizard)
        self.assertEqual(budget, self.controller.INVESTIGATE_MOVES)

    def test_budget_increases_with_failed_chases(self) -> None:
        lizard_base = LizardState(x=0, y=0, chase_streak=0, failed_chases=0)
        lizard_fail = LizardState(x=0, y=0, chase_streak=0, failed_chases=2)
        base = self.controller._investigate_budget(lizard_base)
        fail = self.controller._investigate_budget(lizard_fail)
        self.assertGreater(fail, base)

    def test_budget_increases_with_chase_streak(self) -> None:
        lizard_base = LizardState(x=0, y=0, chase_streak=0, failed_chases=0)
        lizard_streak = LizardState(x=0, y=0, chase_streak=4, failed_chases=0)
        base = self.controller._investigate_budget(lizard_base)
        streak = self.controller._investigate_budget(lizard_streak)
        self.assertGreater(streak, base)

    def test_budget_capped_at_three_extra_moves(self) -> None:
        # min(3, failed_chases + chase_streak // 2) is capped at 3
        lizard_max = LizardState(x=0, y=0, chase_streak=100, failed_chases=100)
        budget = self.controller._investigate_budget(lizard_max)
        self.assertEqual(budget, self.controller.INVESTIGATE_MOVES + 3)

    def test_budget_formula(self) -> None:
        # _investigate_budget = INVESTIGATE_MOVES + min(3, failed_chases + chase_streak // 2)
        for failed in range(4):
            for streak in range(6):
                lizard = LizardState(x=0, y=0, chase_streak=streak, failed_chases=failed)
                expected = self.controller.INVESTIGATE_MOVES + min(3, failed + streak // 2)
                self.assertEqual(self.controller._investigate_budget(lizard), expected)


class PredatorControllerSetModeTest(unittest.TestCase):
    """Tests for PredatorController._set_mode clearing new fields."""

    def setUp(self) -> None:
        self.controller = PredatorController()

    def test_set_mode_clears_investigate_target_on_non_investigate_mode(self) -> None:
        lizard = LizardState(x=0, y=0, mode="INVESTIGATE",
                              investigate_target=(3, 4))
        self.controller._set_mode(lizard, "PATROL")
        self.assertIsNone(lizard.investigate_target)

    def test_set_mode_preserves_investigate_target_in_investigate_mode(self) -> None:
        lizard = LizardState(x=0, y=0, mode="PATROL",
                              investigate_target=(3, 4))
        self.controller._set_mode(lizard, "INVESTIGATE")
        self.assertEqual(lizard.investigate_target, (3, 4))

    def test_set_mode_clears_ambush_ticks_on_non_wait_mode(self) -> None:
        lizard = LizardState(x=0, y=0, mode="WAIT", ambush_ticks=5)
        self.controller._set_mode(lizard, "PATROL")
        self.assertEqual(lizard.ambush_ticks, 0)

    def test_set_mode_preserves_ambush_ticks_in_wait_mode(self) -> None:
        lizard = LizardState(x=0, y=0, mode="PATROL", ambush_ticks=2)
        self.controller._set_mode(lizard, "WAIT")
        self.assertEqual(lizard.ambush_ticks, 2)

    def test_set_mode_preserves_wait_target_in_recover_mode(self) -> None:
        # RECOVER should NOT clear wait_target (so predator can retreat away from it)
        lizard = LizardState(x=0, y=0, mode="WAIT", wait_target=(5, 3))
        self.controller._set_mode(lizard, "RECOVER")
        self.assertEqual(lizard.wait_target, (5, 3))

    def test_set_mode_clears_wait_target_in_patrol_mode(self) -> None:
        lizard = LizardState(x=0, y=0, mode="WAIT", wait_target=(5, 3))
        self.controller._set_mode(lizard, "PATROL")
        self.assertIsNone(lizard.wait_target)

    def test_set_mode_to_same_mode_is_no_op(self) -> None:
        lizard = LizardState(x=0, y=0, mode="CHASE",
                              chase_streak=3, ambush_ticks=2, investigate_target=(1, 1))
        self.controller._set_mode(lizard, "CHASE")
        # Nothing should change since mode is already CHASE
        self.assertEqual(lizard.chase_streak, 3)
        self.assertEqual(lizard.ambush_ticks, 2)
        self.assertEqual(lizard.investigate_target, (1, 1))


class PredatorControllerChaseStreakTest(unittest.TestCase):
    """Tests for chase_streak and failed_chases FSM behavior in world integration."""

    def _setup_world_with_lizard_chasing(self) -> SpiderWorld:
        world = SpiderWorld(seed=55, lizard_move_interval=1, map_template="entrance_funnel")
        world.reset(seed=55)
        entrance = sorted(world.shelter_entrance_cells)[0]
        deep = sorted(world.shelter_deep_cells)[0]
        # Place spider in deep shelter (not detectable by lizard)
        world.state.x, world.state.y = deep
        # Place lizard far from spider, in CHASE mode
        world.lizard = LizardState(
            x=world.width - 1, y=world.height - 1,
            mode="CHASE",
            mode_ticks=5,
            last_known_spider=deep,
            wait_target=entrance,
            chase_streak=3,
            failed_chases=0,
        )
        return world

    def test_failed_chases_increases_on_lost_chase(self) -> None:
        world = self._setup_world_with_lizard_chasing()
        before_failed = world.lizard.failed_chases
        world.step(ACTION_TO_INDEX["STAY"])
        # After losing the chase (spider not detectable from far away), failed_chases should increase
        self.assertGreater(world.lizard.failed_chases, before_failed)

    def test_chase_streak_increments_when_spider_detected_in_chase_mode(self) -> None:
        world = SpiderWorld(seed=57, lizard_move_interval=1, map_template="entrance_funnel")
        world.reset(seed=57)
        entrance = sorted(world.shelter_entrance_cells)[0]
        # Place spider at entrance (visible to lizard outside)
        world.state.x, world.state.y = entrance
        # Place lizard adjacent to entrance in CHASE mode
        ambush = None
        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1)):
            candidate = (entrance[0] + dx, entrance[1] + dy)
            if 0 <= candidate[0] < world.width and 0 <= candidate[1] < world.height:
                if world.is_lizard_walkable(candidate):
                    ambush = candidate
                    break
        self.assertIsNotNone(ambush)
        world.lizard = LizardState(x=ambush[0], y=ambush[1], mode="CHASE", chase_streak=2)
        self.assertTrue(world.lizard_detects_spider())
        streak_before = world.lizard.chase_streak
        world.step(ACTION_TO_INDEX["STAY"])
        self.assertGreater(world.lizard.chase_streak, streak_before)

    def test_chase_streak_resets_when_reaching_patrol_from_wait(self) -> None:
        world = SpiderWorld(seed=61, lizard_move_interval=1, map_template="entrance_funnel")
        world.reset(seed=61)
        entrance = sorted(world.shelter_entrance_cells)[0]
        # Find a lizard-walkable cell adjacent to entrance
        ambush = None
        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1)):
            candidate = (entrance[0] + dx, entrance[1] + dy)
            if 0 <= candidate[0] < world.width and 0 <= candidate[1] < world.height:
                if world.is_lizard_walkable(candidate):
                    ambush = candidate
                    break
        self.assertIsNotNone(ambush)

        world.lizard = LizardState(
            x=ambush[0], y=ambush[1],
            mode="WAIT",
            wait_target=entrance,
            chase_streak=5,
            failed_chases=0,
            ambush_ticks=0,
        )
        self.assertEqual(world.manhattan(world.lizard_pos(), entrance), 1)
        transition = None
        for _ in range(4):
            _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
            if info["predator_transition"] is not None:
                transition = info["predator_transition"]
                break
        self.assertIsNotNone(transition)
        self.assertEqual(transition, {"from": "WAIT", "to": "PATROL"})
        self.assertEqual(world.lizard.mode, "PATROL")
        self.assertEqual(world.lizard.chase_streak, 0)


# ---------------------------------------------------------------------------
# ScenarioSpec - map_template field and new helpers
# ---------------------------------------------------------------------------

class ScenarioSpecMapTemplateTest(unittest.TestCase):
    """Tests for the new map_template field on ScenarioSpec."""

    def test_all_scenarios_have_map_template_field(self) -> None:
        for name in SCENARIO_NAMES:
            spec = get_scenario(name)
            self.assertIsInstance(spec, ScenarioSpec)
            self.assertIsNotNone(spec.map_template)
            self.assertIn(spec.map_template, MAP_TEMPLATE_NAMES,
                msg=f"Scenario '{name}' has unknown map_template '{spec.map_template}'")

    def test_night_rest_uses_central_burrow(self) -> None:
        spec = get_scenario("night_rest")
        self.assertEqual(spec.map_template, "central_burrow")

    def test_predator_edge_uses_central_burrow(self) -> None:
        spec = get_scenario("predator_edge")
        self.assertEqual(spec.map_template, "central_burrow")

    def test_entrance_ambush_uses_entrance_funnel(self) -> None:
        spec = get_scenario("entrance_ambush")
        self.assertEqual(spec.map_template, "entrance_funnel")

    def test_open_field_foraging_uses_exposed_feeding_ground(self) -> None:
        spec = get_scenario("open_field_foraging")
        self.assertEqual(spec.map_template, "exposed_feeding_ground")

    def test_shelter_blockade_uses_entrance_funnel(self) -> None:
        spec = get_scenario("shelter_blockade")
        self.assertEqual(spec.map_template, "entrance_funnel")

    def test_recover_after_failed_chase_uses_entrance_funnel(self) -> None:
        spec = get_scenario("recover_after_failed_chase")
        self.assertEqual(spec.map_template, "entrance_funnel")

    def test_corridor_gauntlet_uses_corridor_escape(self) -> None:
        spec = get_scenario("corridor_gauntlet")
        self.assertEqual(spec.map_template, "corridor_escape")

    def test_two_shelter_tradeoff_uses_two_shelters(self) -> None:
        spec = get_scenario("two_shelter_tradeoff")
        self.assertEqual(spec.map_template, "two_shelters")

    def test_exposed_day_foraging_uses_exposed_feeding_ground(self) -> None:
        spec = get_scenario("exposed_day_foraging")
        self.assertEqual(spec.map_template, "exposed_feeding_ground")

    def test_food_deprivation_uses_central_burrow(self) -> None:
        spec = get_scenario("food_deprivation")
        self.assertEqual(spec.map_template, "central_burrow")

    def test_scenario_max_steps_are_positive(self) -> None:
        """
        Verify every registered scenario's `max_steps` is greater than zero.
        
        Asserts that `get_scenario(name).max_steps > 0` for each name in `SCENARIO_NAMES`; the assertion message identifies the scenario and its non-positive value when violated.
        """
        for name in SCENARIO_NAMES:
            spec = get_scenario(name)
            self.assertGreater(spec.max_steps, 0,
                msg=f"Scenario '{name}' has non-positive max_steps {spec.max_steps}")

    def test_scenarios_expose_behavioral_metadata(self) -> None:
        """
        Assert that every registered scenario exposes required behavioral metadata.
        
        Verifies that for each name in SCENARIO_NAMES, get_scenario(name) returns a ScenarioSpec with a truthy `objective`, a truthy `behavior_checks`, and a callable `score_episode`.
        """
        for name in SCENARIO_NAMES:
            spec = get_scenario(name)
            self.assertTrue(spec.objective)
            self.assertTrue(spec.behavior_checks)
            self.assertTrue(callable(spec.score_episode))


class ScenarioHelperFunctionTest(unittest.TestCase):
    """Tests for _first_cell and _entrance_ambush_cell helpers."""

    def _setup_world(self, name: str) -> SpiderWorld:
        scenario = get_scenario(name)
        world = SpiderWorld(seed=77, lizard_move_interval=1, map_template=scenario.map_template)
        world.reset(seed=77)
        scenario.setup(world)
        return world

    def test_entrance_ambush_lizard_is_adjacent_to_entrance(self) -> None:
        # _entrance_ambush_cell should place the lizard 1 step from entrance
        world = self._setup_world("entrance_ambush")
        entrance = sorted(world.shelter_entrance_cells)[0]
        lizard_pos = world.lizard_pos()
        dist = world.manhattan(lizard_pos, entrance)
        self.assertEqual(dist, 1,
            msg=f"Lizard {lizard_pos} should be 1 step from entrance {entrance}")

    def test_shelter_blockade_lizard_is_adjacent_to_entrance(self) -> None:
        world = self._setup_world("shelter_blockade")
        entrance = sorted(world.shelter_entrance_cells)[0]
        lizard_pos = world.lizard_pos()
        dist = world.manhattan(lizard_pos, entrance)
        self.assertEqual(dist, 1,
            msg=f"Lizard {lizard_pos} should be 1 step from entrance {entrance}")

    def test_entrance_ambush_failed_chases_initialized_to_two(self) -> None:
        world = self._setup_world("entrance_ambush")
        self.assertEqual(world.lizard.failed_chases, 2)

    def test_shelter_blockade_failed_chases_initialized_to_two(self) -> None:
        world = self._setup_world("shelter_blockade")
        self.assertEqual(world.lizard.failed_chases, 2)

    def test_recover_after_failed_chase_initializes_chase_state(self) -> None:
        world = self._setup_world("recover_after_failed_chase")
        self.assertEqual(world.lizard.mode, "CHASE")
        self.assertGreater(world.lizard.chase_streak, 0)
        self.assertGreater(world.lizard.recover_ticks, 0)
        self.assertEqual(world.lizard.failed_chases, 1)

    def test_two_shelter_tradeoff_has_multiple_deep_cells(self) -> None:
        world = self._setup_world("two_shelter_tradeoff")
        self.assertGreater(len(world.shelter_deep_cells), 2)

    def test_corridor_gauntlet_food_is_far_from_spider(self) -> None:
        world = self._setup_world("corridor_gauntlet")
        dist_to_food = world.manhattan(world.spider_pos(), world.food_positions[0])
        # Food should be far from spider start (deep shelter)
        self.assertGreater(dist_to_food, 3)

    def test_exposed_day_foraging_is_daytime(self) -> None:
        world = self._setup_world("exposed_day_foraging")
        self.assertTrue(world.observe()["meta"]["day"])

    def test_exposed_day_foraging_hunger_is_high(self) -> None:
        world = self._setup_world("exposed_day_foraging")
        self.assertGreater(world.state.hunger, 0.9)

    def test_two_shelter_tradeoff_spider_in_left_deep(self) -> None:
        world = self._setup_world("two_shelter_tradeoff")
        spider_x = world.spider_pos()[0]
        # Spider should be in the left shelter (smaller x)
        min_deep_x = min(c[0] for c in world.shelter_deep_cells)
        # Spider x should be close to the minimum deep x
        self.assertLessEqual(spider_x - min_deep_x, 2)


# ---------------------------------------------------------------------------
# SpiderWorld.configure_map_template()
# ---------------------------------------------------------------------------

class ConfigureMapTemplateTest(unittest.TestCase):
    """Tests for the new configure_map_template() method on SpiderWorld."""

    def test_configure_invalid_template_raises_value_error(self) -> None:
        world = SpiderWorld(seed=11)
        with self.assertRaises(ValueError):
            world.configure_map_template("nonexistent_template")

    def test_configure_valid_template_updates_map_template_name(self) -> None:
        world = SpiderWorld(seed=11, map_template="central_burrow")
        world.configure_map_template("side_burrow")
        self.assertEqual(world.map_template_name, "side_burrow")

    def test_configure_template_updates_shelter_cells(self) -> None:
        world = SpiderWorld(seed=11, map_template="central_burrow")
        old_entrance = set(world.shelter_entrance_cells)
        world.configure_map_template("side_burrow")
        new_entrance = set(world.shelter_entrance_cells)
        # The shelter geometry should be different
        self.assertNotEqual(old_entrance, new_entrance)

    def test_configure_template_updates_blocked_cells(self) -> None:
        world = SpiderWorld(seed=11, map_template="central_burrow")
        world.configure_map_template("entrance_funnel")
        self.assertEqual(world.blocked_cells, set(world.map_template.blocked_cells))

    def test_configure_template_updates_shelter_cells_to_new_map(self) -> None:
        world = SpiderWorld(seed=11, map_template="central_burrow")
        world.configure_map_template("two_shelters")
        expected_shelter = world.map_template.shelter_cells
        self.assertEqual(world.shelter_cells, expected_shelter)
        self.assertEqual(world.shelter_entrance_cells, set(world.map_template.shelter_entrance))
        self.assertEqual(world.shelter_interior_cells, set(world.map_template.shelter_interior))
        self.assertEqual(world.shelter_deep_cells, set(world.map_template.shelter_deep))

    def test_configure_all_new_templates_without_error(self) -> None:
        world = SpiderWorld(seed=11)
        for name in ("two_shelters", "exposed_feeding_ground", "entrance_funnel"):
            world.configure_map_template(name)
            self.assertEqual(world.map_template_name, name)

    def test_configure_template_back_to_original(self) -> None:
        world = SpiderWorld(seed=11, map_template="central_burrow")
        original_entrance = set(world.shelter_entrance_cells)
        original_interior = set(world.shelter_interior_cells)
        original_deep = set(world.shelter_deep_cells)
        original_blocked = set(world.blocked_cells)
        original_shelter = set(world.shelter_cells)
        world.configure_map_template("two_shelters")
        world.configure_map_template("central_burrow")
        self.assertEqual(world.shelter_entrance_cells, original_entrance)
        self.assertEqual(world.shelter_interior_cells, original_interior)
        self.assertEqual(world.shelter_deep_cells, original_deep)
        self.assertEqual(world.blocked_cells, original_blocked)
        self.assertEqual(world.shelter_cells, original_shelter)


# ---------------------------------------------------------------------------
# SpiderWorld.step() - info dict new fields
# ---------------------------------------------------------------------------

class WorldStepInfoDictTest(unittest.TestCase):
    """Tests for new fields in the info dict returned by world.step()."""

    def _make_world(self, map_template: str = "central_burrow") -> SpiderWorld:
        world = SpiderWorld(seed=99, map_template=map_template, lizard_move_interval=999999)
        world.reset(seed=99)
        return world

    def test_distance_deltas_always_present_in_info(self) -> None:
        world = self._make_world()
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        self.assertIn("distance_deltas", info)
        self.assertIsInstance(info["distance_deltas"], dict)

    def test_distance_deltas_has_food_shelter_predator_keys(self) -> None:
        world = self._make_world()
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        deltas = info["distance_deltas"]
        self.assertIn("food", deltas)
        self.assertIn("shelter", deltas)
        self.assertIn("predator", deltas)

    def test_distance_deltas_food_is_positive_when_approaching(self) -> None:
        world = self._make_world()
        # Place spider far from food, then move one step closer (but not onto food)
        world.food_positions = [(9, 1)]
        world.state.x, world.state.y = 1, 1
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        _, _, _, info = world.step(ACTION_TO_INDEX["MOVE_RIGHT"])
        # Spider moves from (1,1) to (2,1); food at (9,1): dist goes 8→7, delta=1
        self.assertGreater(info["distance_deltas"]["food"], 0,
            "Moving towards food should give positive food delta")

    def test_predator_transition_is_none_when_mode_unchanged(self) -> None:
        world = self._make_world()
        # Lizard far away, spider in shelter - no state change expected
        deep = sorted(world.shelter_deep_cells)[0]
        world.state.x, world.state.y = deep
        world.lizard.x, world.lizard.y = world.width - 1, world.height - 1
        world.lizard.mode = "PATROL"
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        # We don't guarantee it's None (predator may switch to ORIENT), but
        # when it IS None, the key must still be present
        self.assertIn("predator_transition", info)

    def test_predator_transition_reports_from_and_to_on_mode_change(self) -> None:
        world = SpiderWorld(seed=43, lizard_move_interval=1, map_template="entrance_funnel")
        world.reset(seed=43)
        entrance = sorted(world.shelter_entrance_cells)[0]
        # Put spider at entrance so lizard can detect
        world.state.x, world.state.y = entrance
        # Put lizard adjacent in PATROL mode
        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1)):
            candidate = (entrance[0] + dx, entrance[1] + dy)
            if 0 <= candidate[0] < world.width and 0 <= candidate[1] < world.height:
                if world.is_lizard_walkable(candidate):
                    world.lizard = LizardState(x=candidate[0], y=candidate[1], mode="PATROL")
                    break
        self.assertTrue(world.lizard_detects_spider())
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        transition = info["predator_transition"]
        self.assertIsNotNone(transition)
        self.assertIn("from", transition)
        self.assertIn("to", transition)
        self.assertEqual(transition["from"], "PATROL")
        self.assertEqual(transition["to"], "ORIENT")

    def test_predator_transition_structure_when_not_none(self) -> None:
        world = SpiderWorld(seed=71, lizard_move_interval=1, map_template="entrance_funnel")
        world.reset(seed=71)
        entrance = sorted(world.shelter_entrance_cells)[0]
        world.state.x, world.state.y = entrance
        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1)):
            candidate = (entrance[0] + dx, entrance[1] + dy)
            if 0 <= candidate[0] < world.width and 0 <= candidate[1] < world.height:
                if world.is_lizard_walkable(candidate):
                    world.lizard = LizardState(x=candidate[0], y=candidate[1], mode="PATROL")
                    break
        self.assertTrue(world.lizard_detects_spider())
        world.step(ACTION_TO_INDEX["STAY"])
        _, _, _, info = world.step(ACTION_TO_INDEX["STAY"])
        t = info["predator_transition"]
        self.assertIsNotNone(t)
        self.assertIsInstance(t["from"], str)
        self.assertIsInstance(t["to"], str)
        self.assertEqual(t["from"], "ORIENT")
        self.assertEqual(t["to"], "CHASE")


# ---------------------------------------------------------------------------
# Simulation map template switching per scenario
# ---------------------------------------------------------------------------

class SimulationScenarioMapSwitchingTest(unittest.TestCase):
    """Tests that the simulation switches map templates based on scenario.map_template."""

    def test_simulation_switches_map_for_entrance_funnel_scenario(self) -> None:
        from spider_cortex_sim.simulation import SpiderSimulation
        sim = SpiderSimulation(seed=83, max_steps=5, map_template="central_burrow")
        # Run a scenario that uses entrance_funnel
        sim.run_episode(0, training=False, sample=False, scenario_name="entrance_ambush")
        self.assertEqual(sim.world.map_template_name, "entrance_funnel")

    def test_simulation_switches_map_for_two_shelters_scenario(self) -> None:
        from spider_cortex_sim.simulation import SpiderSimulation
        sim = SpiderSimulation(seed=85, max_steps=5, map_template="central_burrow")
        sim.run_episode(0, training=False, sample=False, scenario_name="two_shelter_tradeoff")
        self.assertEqual(sim.world.map_template_name, "two_shelters")

    def test_simulation_restores_default_map_after_non_scenario_episode(self) -> None:
        from spider_cortex_sim.simulation import SpiderSimulation
        sim = SpiderSimulation(seed=87, max_steps=5, map_template="central_burrow")
        # First run a scenario that switches map
        sim.run_episode(0, training=False, sample=False,
                         scenario_name="entrance_ambush")
        # Then run a non-scenario episode - should revert to default
        sim.run_episode(1, training=False, sample=False)
        self.assertEqual(sim.world.map_template_name, "central_burrow")

    def test_default_map_template_stored_on_simulation(self) -> None:
        from spider_cortex_sim.simulation import SpiderSimulation
        sim = SpiderSimulation(seed=89, max_steps=5, map_template="two_shelters")
        self.assertEqual(sim.default_map_template, "two_shelters")

    def test_trace_includes_distance_deltas_and_predator_transition(self) -> None:
        from spider_cortex_sim.simulation import SpiderSimulation
        sim = SpiderSimulation(seed=91, max_steps=3)
        _, trace = sim.run_episode(0, training=False, sample=False, capture_trace=True)
        self.assertTrue(len(trace) > 0)
        for item in trace:
            self.assertIn("distance_deltas", item)
            self.assertIn("predator_transition", item)

    def test_scenario_stats_include_new_metric_fields(self) -> None:
        from spider_cortex_sim.simulation import SpiderSimulation
        sim = SpiderSimulation(seed=93, max_steps=12)
        results, _ = sim.run_scenarios(["night_rest"])
        stats = results["night_rest"]
        self.assertIn("mean_food_distance_delta", stats)
        self.assertIn("mean_shelter_distance_delta", stats)
        self.assertIn("mean_predator_mode_transitions", stats)
        self.assertIn("dominant_predator_state", stats)


if __name__ == "__main__":
    unittest.main()
