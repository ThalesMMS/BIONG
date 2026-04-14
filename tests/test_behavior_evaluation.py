"""
Comprehensive tests for behavior evaluation functionality introduced in the PR.

Covers:
- metrics.py: BehaviorCheckSpec, BehaviorCheckResult, BehavioralEpisodeScore dataclasses,
  build_behavior_check, build_behavior_score, aggregate_behavior_scores,
  summarize_behavior_suite, flatten_behavior_rows, _aggregate_values, _mean_like
- scenarios.py: _food_deprivation setup, trace helper functions,
  check spec constants, score functions
- simulation.py: save_behavior_csv, _compact_behavior_payload, trace new fields,
  EpisodeStats.seed
- cli.py: build_parser new behavior arguments
"""
from __future__ import annotations

from collections.abc import Mapping
import csv
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from numbers import Real
from pathlib import Path
from typing import Any

from spider_cortex_sim.metrics import (
    BehaviorCheckResult,
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeMetricAccumulator,
    EpisodeStats,
    REFLEX_MODULE_NAMES,
    _aggregate_values,
    _mean_like,
    aggregate_behavior_scores,
    aggregate_episode_stats,
    build_behavior_check,
    build_behavior_score,
    flatten_behavior_rows,
    summarize_behavior_suite,
)
from spider_cortex_sim.ablations import BrainAblationConfig, PROPOSAL_SOURCE_NAMES
from spider_cortex_sim.noise import LOW_NOISE_PROFILE
from spider_cortex_sim.scenarios import (
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    FOOD_DEPRIVATION_CHECKS,
    FOOD_VS_PREDATOR_CONFLICT_CHECKS,
    NIGHT_REST_INITIAL_SLEEP_DEBT,
    NIGHT_REST_CHECKS,
    PREDATOR_EDGE_CHECKS,
    SCENARIO_NAMES,
    SLEEP_VS_EXPLORATION_CONFLICT_CHECKS,
    SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
    _classify_corridor_gauntlet_failure,
    _classify_exposed_day_foraging_failure,
    _classify_food_deprivation_failure,
    _classify_open_field_foraging_failure,
    _clamp01,
    _extract_exposed_day_trace_metrics,
    _float_or_none,
    _mapping_predator_visible,
    _predator_visible_flag,
    _trace_predator_visible,
    _food_signal_strength,
    _hunger_valence_rate,
    _int_or_none,
    _memory_vector_freshness,
    _observation_food_distances,
    _payload_float,
    _payload_text,
    _state_alive,
    _state_food_distance,
    _state_position,
    _trace_action_selection_payloads,
    _trace_any_mode,
    _trace_any_sleep_phase,
    _trace_corridor_metrics,
    _trace_death_tick,
    _trace_escape_seen,
    _trace_food_distances,
    _trace_food_signal_strengths,
    _trace_post_observation_food_distances,
    _trace_predator_memory_seen,
    _trace_pre_observation_food_distances,
    _trace_shelter_cells,
    _trace_shelter_exit,
    _trace_states,
    _trace_tick,
    get_scenario,
)
from spider_cortex_sim.simulation import CURRICULUM_COLUMNS, SpiderSimulation
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES, SpiderWorld

NOISE_PROFILE_CONFIG_JSON = json.dumps(
    {
        "motor": {
            "action_flip_prob": 0.0,
            "fatigue_slip_factor": 0.0,
            "orientation_slip_factor": 0.0,
            "terrain_slip_factor": 0.0,
        },
        "name": "none",
        "olfactory": {
            "direction_jitter": 0.0,
            "strength_jitter": 0.0,
        },
        "predator": {
            "random_choice_prob": 0.0,
        },
        "spawn": {
            "uniform_mix": 0.0,
        },
        "visual": {
            "certainty_jitter": 0.0,
            "direction_jitter": 0.0,
            "dropout_prob": 0.0,
        },
    },
    sort_keys=True,
    separators=(",", ":"),
)
LOW_NOISE_PROFILE_CONFIG_JSON = json.dumps(
    LOW_NOISE_PROFILE.to_summary(),
    sort_keys=True,
    separators=(",", ":"),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode_stats(**overrides) -> EpisodeStats:
    """Create a minimal EpisodeStats for testing."""
    from spider_cortex_sim.predator import PREDATOR_STATES

    defaults = dict(
        episode=0,
        seed=42,
        training=False,
        scenario="test_scenario",
        total_reward=1.0,
        steps=10,
        food_eaten=1,
        sleep_events=0,
        shelter_entries=1,
        alert_events=0,
        predator_contacts=0,
        predator_sightings=0,
        predator_escapes=0,
        night_ticks=5,
        night_shelter_ticks=5,
        night_still_ticks=5,
        night_role_ticks={"outside": 0, "entrance": 0, "inside": 0, "deep": 5},
        night_shelter_occupancy_rate=1.0,
        night_stillness_rate=1.0,
        night_role_distribution={"outside": 0.0, "entrance": 0.0, "inside": 0.0, "deep": 1.0},
        predator_response_events=0,
        mean_predator_response_latency=0.0,
        mean_sleep_debt=0.3,
        food_distance_delta=2.0,
        shelter_distance_delta=0.0,
        final_hunger=0.4,
        final_fatigue=0.3,
        final_sleep_debt=0.15,
        final_health=1.0,
        alive=True,
        reward_component_totals={k: 0.0 for k in REWARD_COMPONENT_NAMES},
        predator_state_ticks={s: 0 for s in PREDATOR_STATES},
        predator_mode_transitions=0,
        dominant_predator_state="PATROL",
    )
    defaults.update(overrides)
    return EpisodeStats(**defaults)


def _make_action_selection_trace_item(payload: dict[str, object]) -> dict[str, object]:
    """Wrap an action.selection payload in the trace envelope used by scorers."""
    return {
        "messages": [
            {
                "sender": "action_center",
                "topic": "action.selection",
                "payload": payload,
            }
        ]
    }


def _make_food_deprivation_trace_item(
    *,
    tick: int,
    pos: tuple[int, int],
    health: float,
    food_dist: int,
    winning_valence: str,
    food_delta: int = 0,
    map_template: str = "central_burrow",
) -> dict[str, object]:
    """
    Create a trace item representing a single tick containing state, food-distance observation, and an action-selection message.
    
    Parameters:
        tick (int): Simulation tick index for the trace item.
        pos (tuple[int, int]): (x, y) position of the agent.
        health (float): Agent health value at this tick.
        food_dist (int): Reported distance-to-food included in the environment observation payload.
        winning_valence (str): Valence string recorded by the action selection message.
        food_delta (int): Change in food-distance since previous tick, placed in `distance_deltas.food`.
        map_template (str): Map template identifier stored in the state.
    
    Returns:
        dict[str, object]: A trace-item dictionary with keys `tick`, `state`, `distance_deltas`, and `messages` suitable for scenario scoring tests.
    """
    return {
        "tick": tick,
        "state": {
            "x": pos[0],
            "y": pos[1],
            "health": health,
            "map_template": map_template,
        },
        "distance_deltas": {"food": food_delta},
        "messages": [
            {
                "sender": "environment",
                "topic": "observation",
                "payload": {"meta": {"food_dist": food_dist}},
            },
            {
                "sender": "action_center",
                "topic": "action.selection",
                "payload": {"winning_valence": winning_valence},
            },
        ],
    }


def _make_open_field_trace_item(
    *,
    tick: int,
    pos: tuple[int, int],
    health: float,
    food_dist: int,
    winning_valence: str = "hunger",
    food_signal_strength: float = 0.0,
    food_delta: int = 0,
) -> dict[str, object]:
    """
    Builds a synthetic open-field trace item containing state, distance deltas, and environment/action-selection messages used by the open-field foraging scorer.
    
    Parameters:
        tick (int): Simulation tick for the trace item.
        pos (tuple[int, int]): (x, y) spider position.
        health (float): Spider health value for the state.
        food_dist (int): Reported observed distance to food in the environment observation metadata.
        winning_valence (str): Value placed in the action-selection payload's `winning_valence` field.
        food_signal_strength (float): Strength of the food sensory cue (0.0 means no signal); influences `food_trace`, vision/certainty, percept_traces, and memory_vectors.
        food_delta (int): Delta to apply to food distance (placed in `distance_deltas.food`) to simulate a post-step distance change.
    
    Returns:
        dict: A trace item dictionary with keys `tick`, `state` (including `map_template="exposed_feeding_ground"` and `food_trace`), `distance_deltas`, and `messages` (an environment observation payload with `meta.food_dist` and sensory fields, and an action.selection payload with `winning_valence` and evidence).
    """
    return {
        "tick": tick,
        "state": {
            "x": pos[0],
            "y": pos[1],
            "health": health,
            "map_template": "exposed_feeding_ground",
            "food_trace": {
                "strength": food_signal_strength,
                "dx": 0.1 if food_signal_strength > 0.0 else 0.0,
                "dy": -0.1 if food_signal_strength > 0.0 else 0.0,
            },
        },
        "distance_deltas": {"food": food_delta},
        "messages": [
            {
                "sender": "environment",
                "topic": "observation",
                "payload": {
                    "meta": {
                        "food_dist": food_dist,
                        "vision": {
                            "food": {
                                "visible": food_signal_strength,
                                "certainty": food_signal_strength,
                            },
                        },
                        "percept_traces": {
                            "food": {"strength": food_signal_strength},
                        },
                        "memory_vectors": {
                            "food": {
                                "dx": 0.1 if food_signal_strength > 0.0 else 0.0,
                                "dy": -0.1 if food_signal_strength > 0.0 else 0.0,
                                "age": 1.0 - food_signal_strength,
                                "ttl": 12,
                            },
                        },
                    },
                },
            },
            {
                "sender": "action_center",
                "topic": "action.selection",
                "payload": {
                    "winning_valence": winning_valence,
                    "evidence": {
                        "hunger": {
                            "food_visible": food_signal_strength,
                            "food_certainty": food_signal_strength,
                            "food_smell_strength": food_signal_strength,
                            "food_memory_freshness": food_signal_strength,
                        },
                    },
                },
            },
        ],
    }


def _make_exposed_day_trace_item(
    *,
    tick: int,
    pos: tuple[int, int],
    health: float,
    food_dist: int,
    predator_visible: bool = False,
    food_delta: int = 0,
) -> dict[str, object]:
    """
    Create a synthetic trace item for the "exposed_day" scenario used in tests.
    
    Parameters:
        tick (int): Trace tick index.
        pos (tuple[int, int]): Spider (x, y) position.
        health (float): Spider health recorded in the `state`.
        food_dist (int): Reported food distance placed in the observation `meta`.
        predator_visible (bool): If True, observation `meta` and action-selection evidence indicate a visible predator.
        food_delta (int): Delta to record under `distance_deltas["food"]`.
    
    Returns:
        dict: A trace item with keys:
            - `tick`: the provided tick,
            - `state`: dict containing `x`, `y`, `health`, and `map_template` set to "exposed_feeding_ground",
            - `distance_deltas`: dict with `"food"` set to `food_delta`,
            - `messages`: list containing an environment `observation` payload with `meta` (including `food_dist` and `predator_visible`) and an `action.selection` payload whose `evidence.threat.predator_visible` is `1.0` when `predator_visible` is True, otherwise `0.0`.
    """
    return {
        "tick": tick,
        "state": {
            "x": pos[0],
            "y": pos[1],
            "health": health,
            "map_template": "exposed_feeding_ground",
        },
        "distance_deltas": {"food": food_delta},
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
            },
            {
                "sender": "action_center",
                "topic": "action.selection",
                "payload": {
                    "winning_valence": "hunger",
                    "evidence": {
                        "threat": {
                            "predator_visible": 1.0 if predator_visible else 0.0,
                        },
                    },
                },
            },
        ],
    }


def _make_corridor_trace_item(
    *,
    tick: int,
    pos: tuple[int, int],
    health: float,
    food_dist: int,
    predator_visible: bool = False,
    food_delta: int = 0,
) -> dict[str, object]:
    """
    Create a synthetic trace item for the "corridor_gauntlet" scenario.
    
    Parameters:
        tick (int): Simulation tick index for the trace item.
        pos (tuple[int, int]): (x, y) position coordinates.
        health (float): Agent health value at this tick.
        food_dist (int): Reported distance to food in the observation meta.
        predator_visible (bool): Whether a predator is visible in this observation.
        food_delta (int): Change in food distance since the previous tick.
    
    Returns:
        dict[str, object]: A trace item dictionary containing keys:
            - "tick": tick index
            - "state": mapping with "x", "y", "health", and "map_template" set to "corridor_escape"
            - "distance_deltas": mapping with "food" delta
            - "messages": list with an environment observation (including meta.food_dist and meta.predator_visible)
              and an action.selection payload containing threat evidence for predator visibility
    """
    return {
        "tick": tick,
        "state": {
            "x": pos[0],
            "y": pos[1],
            "health": health,
            "map_template": "corridor_escape",
        },
        "distance_deltas": {"food": food_delta},
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
            },
            {
                "sender": "action_center",
                "topic": "action.selection",
                "payload": {
                    "winning_valence": "hunger",
                    "evidence": {
                        "threat": {
                            "predator_visible": 1.0 if predator_visible else 0.0,
                        },
                    },
                },
            },
        ],
    }


def _trace_positions_for_scenario(scenario_name: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Locate the shelter position and the nearest adjacent outside cell for a given scenario.
    
    Builds a SpiderWorld using the scenario's map template and setup, finds the spider's shelter cell, and selects a walkable neighboring cell that is not part of the shelter which is nearest (by Manhattan distance) to the scenario's first food position. Ties are broken by row then column.
    
    Parameters:
        scenario_name (str): Name of the scenario to use for world construction and setup.
    
    Returns:
        tuple[tuple[int, int], tuple[int, int]]: (shelter_pos, outside_pos) where each position is a (row, column) pair.
    
    Raises:
        AssertionError: If the scenario fixture has no adjacent outside cell or has no food positions.
    """
    scenario = get_scenario(scenario_name)
    world = SpiderWorld(seed=101, lizard_move_interval=1, map_template=scenario.map_template)
    world.reset(seed=101)
    scenario.setup(world)
    shelter_pos = world.spider_pos()
    outside_candidates = []
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        cell = (shelter_pos[0] + dx, shelter_pos[1] + dy)
        if not (0 <= cell[0] < world.width and 0 <= cell[1] < world.height):
            continue
        if world.is_walkable(cell) and cell not in world.shelter_cells:
            outside_candidates.append(cell)
    if not outside_candidates:
        raise AssertionError(f"{scenario_name} fixture has no adjacent outside cell")
    if not world.food_positions:
        raise AssertionError(
            f"{scenario_name} fixture has no world.food_positions "
            "in _trace_positions_for_scenario"
        )
    target = world.food_positions[0]
    outside_pos = min(
        outside_candidates,
        key=lambda cell: (world.manhattan(cell, target), cell[1], cell[0]),
    )
    return shelter_pos, outside_pos


def _open_field_trace_positions() -> tuple[tuple[int, int], tuple[int, int]]:
    """Return shelter and outside positions for open-field trace fixtures."""
    return _trace_positions_for_scenario("open_field_foraging")


def _exposed_day_trace_positions() -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Provide shelter and outside coordinate pairs for exposed-day-foraging trace fixtures.
    
    Returns:
        tuple[tuple[int, int], tuple[int, int]]: A pair (shelter_pos, outside_pos), each an (x, y) integer coordinate.
    """
    return _trace_positions_for_scenario("exposed_day_foraging")


def _corridor_trace_positions() -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Get shelter and outside positions for corridor-gauntlet trace fixtures.
    
    Returns:
        tuple[tuple[int, int], tuple[int, int]]: (shelter_pos, outside_pos) where each is an (x, y) coordinate pair.
    """
    return _trace_positions_for_scenario("corridor_gauntlet")


def _open_field_failure_base_metrics() -> dict[str, object]:
    """
    Provide a baseline metrics dictionary for open-field failure classifier tests.
    
    Returns:
        dict[str, object]: A metrics mapping representing a typical episode used in tests. Keys:
            - "checks_passed": `False` indicating one or more behavior checks failed.
            - "left_shelter": whether the agent exited shelter (True).
            - "hunger_valence_rate": fraction of action-selection payloads favoring hunger (0.0-1.0).
            - "initial_food_distance": distance to nearest food at start of episode.
            - "min_food_distance_reached": minimum distance to food observed during episode.
            - "food_distance_delta": change in distance to food (initial minus minimum).
            - "alive": whether the agent survived to episode end (True/False).
            - "food_eaten": count of food items consumed.
            - "max_food_signal_strength": maximum observed food-signal strength (0.0-1.0).
            - "initial_food_signal_strength": food-signal strength at episode start (0.0-1.0).
    """
    return {
        "checks_passed": False,
        "left_shelter": True,
        "hunger_valence_rate": 0.75,
        "initial_food_distance": 8.0,
        "min_food_distance_reached": 6.0,
        "food_distance_delta": 2.0,
        "alive": True,
        "food_eaten": 0,
        "max_food_signal_strength": 0.8,
        "initial_food_signal_strength": 0.5,
    }


def _make_behavior_check_result(
    name: str = "check_a",
    passed: bool = True,
    value: object | None = 1.0,
    description: str = "desc",
    expected: str = "true",
) -> BehaviorCheckResult:
    """
    Create a BehaviorCheckResult with the provided name, description, expected value, pass status, and observed value.
    
    Returns:
        BehaviorCheckResult: Instance whose fields mirror the function arguments.
    """
    return BehaviorCheckResult(name=name, description=description, expected=expected, passed=passed, value=value)


def _make_behavioral_episode_score(
    episode=0,
    seed=42,
    scenario="test_scenario",
    objective="test_objective",
    success=True,
    checks=None,
    behavior_metrics=None,
    failures=None,
) -> BehavioralEpisodeScore:
    if checks is None:
        checks = {}
    if behavior_metrics is None:
        behavior_metrics = {}
    if failures is None:
        failures = []
    return BehavioralEpisodeScore(
        episode=episode,
        seed=seed,
        scenario=scenario,
        objective=objective,
        success=success,
        checks=checks,
        behavior_metrics=behavior_metrics,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# BehaviorCheckSpec dataclass
# ---------------------------------------------------------------------------

class BehaviorCheckSpecTest(unittest.TestCase):
    """Tests for the BehaviorCheckSpec frozen dataclass."""

    def test_creation_stores_fields(self) -> None:
        spec = BehaviorCheckSpec("my_check", "some description", ">= 0.5")
        self.assertEqual(spec.name, "my_check")
        self.assertEqual(spec.description, "some description")
        self.assertEqual(spec.expected, ">= 0.5")

    def test_frozen_raises_on_assignment(self) -> None:
        spec = BehaviorCheckSpec("check", "desc", "val")
        with self.assertRaises((AttributeError, TypeError)):
            spec.name = "new_name"  # type: ignore[misc]

    def test_equality_with_same_values(self) -> None:
        a = BehaviorCheckSpec("x", "d", "e")
        b = BehaviorCheckSpec("x", "d", "e")
        self.assertEqual(a, b)

    def test_inequality_with_different_values(self) -> None:
        a = BehaviorCheckSpec("x", "d", "e1")
        b = BehaviorCheckSpec("x", "d", "e2")
        self.assertNotEqual(a, b)


# ---------------------------------------------------------------------------
# BehaviorCheckResult dataclass
# ---------------------------------------------------------------------------

class BehaviorCheckResultTest(unittest.TestCase):
    """Tests for the BehaviorCheckResult frozen dataclass."""

    def test_creation_stores_all_fields(self) -> None:
        result = BehaviorCheckResult("chk", "description", ">= 1", True, 1.5)
        self.assertEqual(result.name, "chk")
        self.assertEqual(result.description, "description")
        self.assertEqual(result.expected, ">= 1")
        self.assertTrue(result.passed)
        self.assertEqual(result.value, 1.5)

    def test_passed_false_is_stored(self) -> None:
        result = BehaviorCheckResult("chk", "desc", "true", False, False)
        self.assertFalse(result.passed)

    def test_value_can_be_any_type(self) -> None:
        for value in [0, 1.5, True, "string", [1, 2], {"a": 1}]:
            result = BehaviorCheckResult("chk", "desc", "exp", True, value)
            self.assertEqual(result.value, value)


# ---------------------------------------------------------------------------
# BehavioralEpisodeScore dataclass
# ---------------------------------------------------------------------------

class BehavioralEpisodeScoreTest(unittest.TestCase):
    """Tests for the BehavioralEpisodeScore dataclass."""

    def test_creation_stores_fields(self) -> None:
        check = _make_behavior_check_result()
        score = BehavioralEpisodeScore(
            episode=3,
            seed=7,
            scenario="night_rest",
            objective="test_obj",
            success=True,
            checks={"check_a": check},
            behavior_metrics={"rate": 0.9},
            failures=[],
        )
        self.assertEqual(score.episode, 3)
        self.assertEqual(score.seed, 7)
        self.assertEqual(score.scenario, "night_rest")
        self.assertTrue(score.success)
        self.assertIn("check_a", score.checks)
        self.assertEqual(score.behavior_metrics["rate"], 0.9)
        self.assertEqual(score.failures, [])

    def test_can_be_converted_to_dict_with_asdict(self) -> None:
        score = _make_behavioral_episode_score(behavior_metrics={"x": 1.0})
        d = asdict(score)
        self.assertIn("episode", d)
        self.assertIn("seed", d)
        self.assertIn("success", d)


# ---------------------------------------------------------------------------
# build_behavior_check
# ---------------------------------------------------------------------------

class BuildBehaviorCheckTest(unittest.TestCase):
    """Tests for build_behavior_check."""

    def test_creates_result_from_spec_when_passed(self) -> None:
        spec = BehaviorCheckSpec("chk", "desc", ">= 0.5")
        result = build_behavior_check(spec, passed=True, value=0.8)
        self.assertEqual(result.name, "chk")
        self.assertEqual(result.description, "desc")
        self.assertEqual(result.expected, ">= 0.5")
        self.assertTrue(result.passed)
        self.assertEqual(result.value, 0.8)

    def test_creates_result_from_spec_when_failed(self) -> None:
        spec = BehaviorCheckSpec("chk", "desc", ">= 0.5")
        result = build_behavior_check(spec, passed=False, value=0.2)
        self.assertFalse(result.passed)
        self.assertEqual(result.value, 0.2)

    def test_passed_is_coerced_to_bool(self) -> None:
        spec = BehaviorCheckSpec("chk", "desc", "true")
        result_truthy = build_behavior_check(spec, passed=1, value=1)
        result_falsy = build_behavior_check(spec, passed=0, value=0)
        self.assertIs(type(result_truthy.passed), bool)
        self.assertIs(type(result_falsy.passed), bool)
        self.assertTrue(result_truthy.passed)
        self.assertFalse(result_falsy.passed)

    def test_value_preserved_as_given(self) -> None:
        spec = BehaviorCheckSpec("chk", "desc", "exp")
        for val in [None, "str_val", [1, 2, 3], {"a": 1}]:
            result = build_behavior_check(spec, passed=True, value=val)
            self.assertEqual(result.value, val)

    def test_result_inherits_name_description_expected_from_spec(self) -> None:
        spec = BehaviorCheckSpec("my_name", "my_desc", "my_expected")
        result = build_behavior_check(spec, passed=True, value=42)
        self.assertEqual(result.name, spec.name)
        self.assertEqual(result.description, spec.description)
        self.assertEqual(result.expected, spec.expected)


# ---------------------------------------------------------------------------
# build_behavior_score
# ---------------------------------------------------------------------------

class BuildBehaviorScoreTest(unittest.TestCase):
    """Tests for build_behavior_score."""

    def test_success_when_all_checks_pass(self) -> None:
        stats = _make_episode_stats(episode=5, seed=7, scenario="night_rest")
        checks = [
            _make_behavior_check_result("a", passed=True),
            _make_behavior_check_result("b", passed=True),
        ]
        score = build_behavior_score(
            stats=stats,
            objective="test",
            checks=checks,
            behavior_metrics={},
        )
        self.assertTrue(score.success)
        self.assertEqual(score.failures, [])

    def test_failure_when_check_fails(self) -> None:
        stats = _make_episode_stats(scenario="night_rest")
        checks = [
            _make_behavior_check_result("a", passed=True),
            _make_behavior_check_result("b", passed=False),
        ]
        score = build_behavior_score(
            stats=stats,
            objective="test",
            checks=checks,
            behavior_metrics={},
        )
        self.assertFalse(score.success)
        self.assertIn("b", score.failures)
        self.assertNotIn("a", score.failures)

    def test_scenario_defaults_to_default_when_none(self) -> None:
        stats = _make_episode_stats(scenario=None)
        score = build_behavior_score(
            stats=stats,
            objective="test",
            checks=[],
            behavior_metrics={},
        )
        self.assertEqual(score.scenario, "default")

    def test_episode_and_seed_from_stats(self) -> None:
        stats = _make_episode_stats(episode=42, seed=99, scenario="test")
        score = build_behavior_score(
            stats=stats,
            objective="test",
            checks=[],
            behavior_metrics={},
        )
        self.assertEqual(score.episode, 42)
        self.assertEqual(score.seed, 99)

    def test_checks_indexed_by_name(self) -> None:
        stats = _make_episode_stats()
        check_a = _make_behavior_check_result("alpha", passed=True)
        check_b = _make_behavior_check_result("beta", passed=False)
        score = build_behavior_score(
            stats=stats,
            objective="obj",
            checks=[check_a, check_b],
            behavior_metrics={},
        )
        self.assertIn("alpha", score.checks)
        self.assertIn("beta", score.checks)
        self.assertTrue(score.checks["alpha"].passed)
        self.assertFalse(score.checks["beta"].passed)

    def test_behavior_metrics_copied(self) -> None:
        stats = _make_episode_stats()
        metrics = {"rate": 0.9, "count": 3}
        score = build_behavior_score(
            stats=stats,
            objective="obj",
            checks=[],
            behavior_metrics=metrics,
        )
        self.assertEqual(score.behavior_metrics["rate"], 0.9)
        self.assertEqual(score.behavior_metrics["count"], 3)

    def test_success_false_when_all_checks_fail(self) -> None:
        stats = _make_episode_stats()
        checks = [
            _make_behavior_check_result("x", passed=False),
            _make_behavior_check_result("y", passed=False),
        ]
        score = build_behavior_score(stats=stats, objective="o", checks=checks, behavior_metrics={})
        self.assertFalse(score.success)
        self.assertIn("x", score.failures)
        self.assertIn("y", score.failures)

    def test_empty_checks_yields_success(self) -> None:
        stats = _make_episode_stats()
        score = build_behavior_score(stats=stats, objective="o", checks=[], behavior_metrics={})
        self.assertTrue(score.success)
        self.assertEqual(score.failures, [])


# ---------------------------------------------------------------------------
# aggregate_behavior_scores
# ---------------------------------------------------------------------------

class AggregateBehaviorScoresTest(unittest.TestCase):
    """Tests for aggregate_behavior_scores."""

    def _make_spec(self, name="check_a") -> BehaviorCheckSpec:
        return BehaviorCheckSpec(name, f"desc_{name}", ">= 0")

    def _make_score(self, passed=True, check_name="check_a", value=1.0) -> BehavioralEpisodeScore:
        check = _make_behavior_check_result(check_name, passed=passed, value=value)
        failures = [] if passed else [check_name]
        return BehavioralEpisodeScore(
            episode=0, seed=1, scenario="s", objective="o",
            success=passed,
            checks={check_name: check},
            behavior_metrics={
                "metric_a": value,
                "outcome_band": "full_success" if passed else "stalled_and_died",
                "partial_progress": bool(passed),
                "died_without_contact": not passed,
            },
            failures=failures,
        )

    def test_success_rate_all_pass(self) -> None:
        scores = [self._make_score(True), self._make_score(True)]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertEqual(result["success_rate"], 1.0)

    def test_success_rate_none_pass(self) -> None:
        scores = [self._make_score(False), self._make_score(False)]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertEqual(result["success_rate"], 0.0)

    def test_success_rate_partial(self) -> None:
        scores = [self._make_score(True), self._make_score(False)]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertAlmostEqual(result["success_rate"], 0.5)

    def test_check_pass_rate_computed(self) -> None:
        scores = [
            self._make_score(True, value=1.0),
            self._make_score(True, value=2.0),
            self._make_score(False, value=0.0),
        ]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        check = result["checks"]["check_a"]
        self.assertAlmostEqual(check["pass_rate"], 2 / 3)

    def test_check_mean_value_computed(self) -> None:
        scores = [
            self._make_score(True, value=1.0),
            self._make_score(True, value=3.0),
        ]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertAlmostEqual(result["checks"]["check_a"]["mean_value"], 2.0)

    def test_failures_are_sorted_unique(self) -> None:
        score_fail = self._make_score(False)
        scores = [self._make_score(True), score_fail, score_fail]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertEqual(result["failures"], ["check_a"])

    def test_episodes_count_matches_input(self) -> None:
        scores = [self._make_score() for _ in range(7)]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertEqual(result["episodes"], 7)

    def test_legacy_metrics_included(self) -> None:
        scores = [self._make_score()]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
            legacy_metrics={"mean_reward": 3.14},
        )
        self.assertEqual(result["legacy_metrics"]["mean_reward"], 3.14)

    def test_legacy_metrics_defaults_to_empty(self) -> None:
        scores = [self._make_score()]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertEqual(result["legacy_metrics"], {})

    def test_episodes_detail_contains_per_episode_dicts(self) -> None:
        scores = [self._make_score(), self._make_score()]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertEqual(len(result["episodes_detail"]), 2)
        for ep in result["episodes_detail"]:
            self.assertIn("episode", ep)
            self.assertIn("success", ep)

    def test_empty_scores_returns_zero_success_rate(self) -> None:
        result = aggregate_behavior_scores(
            [],
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertEqual(result["success_rate"], 0.0)
        self.assertEqual(result["episodes"], 0)

    def test_scenario_description_objective_passed_through(self) -> None:
        result = aggregate_behavior_scores(
            [],
            scenario="my_scenario",
            description="my description",
            objective="my objective",
            check_specs=[],
        )
        self.assertEqual(result["scenario"], "my_scenario")
        self.assertEqual(result["description"], "my description")
        self.assertEqual(result["objective"], "my objective")

    def test_diagnostic_metadata_passed_through(self) -> None:
        result = aggregate_behavior_scores(
            [],
            scenario="my_scenario",
            description="my description",
            objective="my objective",
            diagnostic_focus="focus",
            success_interpretation="success",
            failure_interpretation="failure",
            budget_note="note",
            check_specs=[],
        )
        self.assertEqual(result["diagnostic_focus"], "focus")
        self.assertEqual(result["success_interpretation"], "success")
        self.assertEqual(result["failure_interpretation"], "failure")
        self.assertEqual(result["budget_note"], "note")

    def test_behavior_metrics_aggregated(self) -> None:
        scores = [self._make_score(value=2.0), self._make_score(value=4.0)]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertAlmostEqual(result["behavior_metrics"]["metric_a"], 3.0)

    def test_diagnostics_aggregated(self) -> None:
        scores = [
            BehavioralEpisodeScore(
                episode=0,
                seed=1,
                scenario="s",
                objective="o",
                success=False,
                checks={"check_a": _make_behavior_check_result("check_a", passed=False)},
                behavior_metrics={
                    "outcome_band": "partial_progress_died",
                    "partial_progress": True,
                    "died_without_contact": True,
                },
                failures=["check_a"],
            ),
            BehavioralEpisodeScore(
                episode=1,
                seed=2,
                scenario="s",
                objective="o",
                success=False,
                checks={"check_a": _make_behavior_check_result("check_a", passed=False)},
                behavior_metrics={
                    "outcome_band": "partial_progress_died",
                    "partial_progress": True,
                    "died_without_contact": False,
                },
                failures=["check_a"],
            ),
            BehavioralEpisodeScore(
                episode=2,
                seed=3,
                scenario="s",
                objective="o",
                success=True,
                checks={"check_a": _make_behavior_check_result("check_a", passed=True)},
                behavior_metrics={
                    "outcome_band": "full_success",
                    "partial_progress": True,
                    "died_without_contact": False,
                },
                failures=[],
            ),
        ]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertEqual(result["diagnostics"]["primary_outcome"], "partial_progress_died")
        self.assertAlmostEqual(
            result["diagnostics"]["outcome_distribution"]["partial_progress_died"],
            2 / 3,
        )
        self.assertAlmostEqual(result["diagnostics"]["partial_progress_rate"], 1.0)
        self.assertAlmostEqual(result["diagnostics"]["died_without_contact_rate"], 1 / 3)

    def test_failure_mode_diagnostics_aggregated_when_present(self) -> None:
        scores = [
            self._make_score(False),
            self._make_score(False),
            self._make_score(True),
        ]
        scores[0].behavior_metrics["failure_mode"] = "left_without_food_signal"
        scores[1].behavior_metrics["failure_mode"] = "left_without_food_signal"
        scores[2].behavior_metrics["failure_mode"] = "success"

        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )

        self.assertEqual(
            result["diagnostics"]["primary_failure_mode"],
            "left_without_food_signal",
        )
        self.assertAlmostEqual(
            result["diagnostics"]["failure_mode_distribution"]["left_without_food_signal"],
            2 / 3,
        )
        self.assertAlmostEqual(
            result["diagnostics"]["failure_mode_distribution"]["success"],
            1 / 3,
        )

    def test_diagnostic_rates_are_none_when_metric_is_absent(self) -> None:
        score = BehavioralEpisodeScore(
            episode=0,
            seed=1,
            scenario="s",
            objective="o",
            success=True,
            checks={"check_a": _make_behavior_check_result("check_a", passed=True)},
            behavior_metrics={"metric_a": 1.0},
            failures=[],
        )
        result = aggregate_behavior_scores(
            [score],
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )
        self.assertEqual(result["diagnostics"]["primary_outcome"], "not_available")
        self.assertIsNone(result["diagnostics"]["partial_progress_rate"])
        self.assertIsNone(result["diagnostics"]["died_without_contact_rate"])


# ---------------------------------------------------------------------------
# summarize_behavior_suite
# ---------------------------------------------------------------------------

class SummarizeBehaviorSuiteTest(unittest.TestCase):
    """Tests for summarize_behavior_suite."""

    def test_empty_suite_returns_zeros(self) -> None:
        result = summarize_behavior_suite({})
        self.assertEqual(result["scenario_count"], 0)
        self.assertEqual(result["episode_count"], 0)
        self.assertEqual(result["scenario_success_rate"], 0.0)
        self.assertEqual(result["episode_success_rate"], 0.0)
        self.assertEqual(result["competence_type"], "mixed")
        self.assertEqual(result["regressions"], [])

    def test_single_fully_passing_scenario(self) -> None:
        suite = {"s1": {"episodes": 3, "success_rate": 1.0, "failures": []}}
        result = summarize_behavior_suite(suite)
        self.assertEqual(result["scenario_count"], 1)
        self.assertEqual(result["episode_count"], 3)
        self.assertEqual(result["scenario_success_rate"], 1.0)
        self.assertAlmostEqual(result["episode_success_rate"], 1.0)
        self.assertEqual(result["regressions"], [])

    def test_single_failing_scenario(self) -> None:
        suite = {"s1": {"episodes": 2, "success_rate": 0.0, "failures": ["check_a"]}}
        result = summarize_behavior_suite(suite)
        self.assertEqual(result["scenario_success_rate"], 0.0)
        self.assertEqual(result["episode_success_rate"], 0.0)
        self.assertEqual(len(result["regressions"]), 1)
        self.assertEqual(result["regressions"][0]["scenario"], "s1")
        self.assertEqual(result["regressions"][0]["failures"], ["check_a"])

    def test_mixed_scenarios(self) -> None:
        suite = {
            "s1": {"episodes": 4, "success_rate": 1.0, "failures": []},
            "s2": {"episodes": 2, "success_rate": 0.0, "failures": ["f1"]},
        }
        result = summarize_behavior_suite(suite)
        self.assertEqual(result["scenario_count"], 2)
        self.assertEqual(result["episode_count"], 6)
        # scenario_success_rate: mean of [1.0 >= 1.0 → 1.0, 0.0 >= 1.0 → 0.0] = 0.5
        self.assertAlmostEqual(result["scenario_success_rate"], 0.5)
        # episode_success_rate: (4*1.0 + 2*0.0) / 6 = 4/6 ≈ 0.667
        self.assertAlmostEqual(result["episode_success_rate"], 4 / 6)
        self.assertEqual(len(result["regressions"]), 1)

    def test_partial_success_rate_not_counted_as_scenario_success(self) -> None:
        suite = {"s1": {"episodes": 2, "success_rate": 0.5, "failures": ["f1"]}}
        result = summarize_behavior_suite(suite)
        # 0.5 < 1.0, so scenario_success_rate should be 0.0
        self.assertEqual(result["scenario_success_rate"], 0.0)

    def test_regressions_only_for_scenarios_with_failures(self) -> None:
        suite = {
            "pass_scenario": {"episodes": 1, "success_rate": 1.0, "failures": []},
            "fail_scenario": {"episodes": 1, "success_rate": 0.5, "failures": ["broken_check"]},
        }
        result = summarize_behavior_suite(suite)
        regression_names = [r["scenario"] for r in result["regressions"]]
        self.assertIn("fail_scenario", regression_names)
        self.assertNotIn("pass_scenario", regression_names)

    def test_competence_label_is_recorded(self) -> None:
        suite = {"s1": {"episodes": 1, "success_rate": 1.0, "failures": []}}
        result = summarize_behavior_suite(
            suite,
            competence_label="self_sufficient",
        )
        self.assertEqual(result["competence_type"], "self_sufficient")

    def test_invalid_competence_label_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            summarize_behavior_suite({}, competence_label="assisted")


# ---------------------------------------------------------------------------
# flatten_behavior_rows
# ---------------------------------------------------------------------------

class FlattenBehaviorRowsTest(unittest.TestCase):
    """Tests for flatten_behavior_rows."""

    def _make_score_with_check(self, passed=True) -> BehavioralEpisodeScore:
        spec = BehaviorCheckSpec("chk_a", "desc", "true")
        check = build_behavior_check(spec, passed=passed, value=passed)
        failures = [] if passed else ["chk_a"]
        return BehavioralEpisodeScore(
            episode=0, seed=5, scenario="night_rest", objective="obj",
            success=passed, checks={"chk_a": check},
            behavior_metrics={"food_eaten": 1},
            failures=failures,
        )

    def test_one_row_per_score(self) -> None:
        scores = [self._make_score_with_check(), self._make_score_with_check()]
        rows = flatten_behavior_rows(
            scores,
            reward_profile="classic",
            scenario_map="central_burrow",
            scenario_description="desc",
            scenario_objective="obj",
            scenario_focus="focus",
            evaluation_map="side_burrow",
            simulation_seed=7,
        )
        self.assertEqual(len(rows), 2)

    def test_fixed_columns_present(self) -> None:
        scores = [self._make_score_with_check()]
        row = flatten_behavior_rows(
            scores,
            reward_profile="eco",
            scenario_map="side_burrow",
            scenario_description="desc",
            scenario_objective="obj",
            scenario_focus="focus",
            evaluation_map="central_burrow",
            simulation_seed=42,
        )[0]
        for col in ["reward_profile", "scenario_map", "evaluation_map", "competence_type", "is_primary_benchmark", "eval_reflex_scale", "simulation_seed", "episode_seed", "scenario", "scenario_description", "scenario_objective", "scenario_focus", "episode", "success", "failure_count", "failures"]:
            self.assertIn(col, row, msg=f"Column {col!r} missing from row")

    def test_reward_profile_and_maps_set(self) -> None:
        scores = [self._make_score_with_check()]
        row = flatten_behavior_rows(
            scores,
            reward_profile="ecological",
            scenario_map="two_shelters",
            scenario_description="desc",
            scenario_objective="obj",
            scenario_focus="focus",
            evaluation_map="central_burrow",
            simulation_seed=99,
        )[0]
        self.assertEqual(row["reward_profile"], "ecological")
        self.assertEqual(row["scenario_map"], "two_shelters")
        self.assertEqual(row["evaluation_map"], "central_burrow")
        self.assertEqual(row["simulation_seed"], 99)

    def test_episode_seed_from_score(self) -> None:
        scores = [self._make_score_with_check()]
        row = flatten_behavior_rows(scores, reward_profile="classic", scenario_map="central_burrow", scenario_description="desc", scenario_objective="obj", scenario_focus="focus", evaluation_map="central_burrow", simulation_seed=7)[0]
        self.assertEqual(row["episode_seed"], 5)

    def test_success_and_failure_count(self) -> None:
        passed_score = self._make_score_with_check(passed=True)
        failed_score = self._make_score_with_check(passed=False)
        rows = flatten_behavior_rows([passed_score, failed_score], reward_profile="classic", scenario_map="central_burrow", scenario_description="desc", scenario_objective="obj", scenario_focus="focus", evaluation_map="central_burrow", simulation_seed=0)
        self.assertTrue(rows[0]["success"])
        self.assertEqual(rows[0]["failure_count"], 0)
        self.assertFalse(rows[1]["success"])
        self.assertEqual(rows[1]["failure_count"], 1)
        self.assertIn("chk_a", rows[1]["failures"])

    def test_metric_columns_prefixed(self) -> None:
        scores = [self._make_score_with_check()]
        row = flatten_behavior_rows(scores, reward_profile="classic", scenario_map="central_burrow", scenario_description="desc", scenario_objective="obj", scenario_focus="focus", evaluation_map="central_burrow", simulation_seed=0)[0]
        self.assertIn("metric_food_eaten", row)
        self.assertEqual(row["metric_food_eaten"], 1)

    def test_check_columns_present_for_each_check(self) -> None:
        scores = [self._make_score_with_check()]
        row = flatten_behavior_rows(scores, reward_profile="classic", scenario_map="central_burrow", scenario_description="desc", scenario_objective="obj", scenario_focus="focus", evaluation_map="central_burrow", simulation_seed=0)[0]
        self.assertIn("check_chk_a_passed", row)
        self.assertIn("check_chk_a_value", row)
        self.assertIn("check_chk_a_expected", row)

    def test_empty_scores_returns_empty_list(self) -> None:
        rows = flatten_behavior_rows([], reward_profile="classic", scenario_map="central_burrow", scenario_description="desc", scenario_objective="obj", scenario_focus="focus", evaluation_map="central_burrow", simulation_seed=0)
        self.assertEqual(rows, [])

    def test_failures_joined_with_comma(self) -> None:
        check_a = _make_behavior_check_result("fa", passed=False)
        check_b = _make_behavior_check_result("fb", passed=False)
        score = BehavioralEpisodeScore(
            episode=0, seed=1, scenario="s", objective="o",
            success=False,
            checks={"fa": check_a, "fb": check_b},
            behavior_metrics={},
            failures=["fa", "fb"],
        )
        rows = flatten_behavior_rows([score], reward_profile="c", scenario_map="m", scenario_description="desc", scenario_objective="obj", scenario_focus="focus", evaluation_map="sweep", simulation_seed=0)
        self.assertIn("fa", rows[0]["failures"])
        self.assertIn("fb", rows[0]["failures"])

    def test_scenario_metadata_columns_populated(self) -> None:
        row = flatten_behavior_rows(
            [self._make_score_with_check()],
            reward_profile="classic",
            scenario_map="central_burrow",
            scenario_description="scenario desc",
            scenario_objective="scenario obj",
            scenario_focus="scenario focus",
            evaluation_map="central_burrow",
            simulation_seed=0,
        )[0]
        self.assertEqual(row["scenario_description"], "scenario desc")
        self.assertEqual(row["scenario_objective"], "scenario obj")
        self.assertEqual(row["scenario_focus"], "scenario focus")

    def test_competence_columns_derive_from_eval_reflex_scale(self) -> None:
        self_sufficient = flatten_behavior_rows(
            [self._make_score_with_check()],
            reward_profile="classic",
            scenario_map="central_burrow",
            scenario_description="desc",
            scenario_objective="obj",
            scenario_focus="focus",
            evaluation_map="central_burrow",
            simulation_seed=0,
            eval_reflex_scale=0.0,
        )[0]
        scaffolded = flatten_behavior_rows(
            [self._make_score_with_check()],
            reward_profile="classic",
            scenario_map="central_burrow",
            scenario_description="desc",
            scenario_objective="obj",
            scenario_focus="focus",
            evaluation_map="central_burrow",
            simulation_seed=0,
            eval_reflex_scale=0.5,
        )[0]

        self.assertEqual(self_sufficient["competence_type"], "self_sufficient")
        self.assertTrue(self_sufficient["is_primary_benchmark"])
        self.assertEqual(scaffolded["competence_type"], "scaffolded")
        self.assertFalse(scaffolded["is_primary_benchmark"])


# ---------------------------------------------------------------------------
# _mean_like
# ---------------------------------------------------------------------------

class MeanLikeTest(unittest.TestCase):
    """Tests for the _mean_like helper."""

    def test_empty_returns_zero(self) -> None:
        self.assertEqual(_mean_like([]), 0.0)

    def test_int_list_returns_float_mean(self) -> None:
        result = _mean_like([2, 4, 6])
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 4.0)

    def test_float_list_returns_mean(self) -> None:
        result = _mean_like([1.0, 3.0])
        self.assertAlmostEqual(result, 2.0)

    def test_bool_values_treated_as_numeric(self) -> None:
        result = _mean_like([True, False, True])
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 2 / 3)

    def test_mixed_int_float_bool_returns_mean(self) -> None:
        result = _mean_like([1, 2.0, True])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 4 / 3)

    def test_string_returns_none(self) -> None:
        result = _mean_like(["a", "b"])
        self.assertIsNone(result)

    def test_mixed_numeric_and_string_returns_none(self) -> None:
        result = _mean_like([1, "x"])
        self.assertIsNone(result)

    def test_single_numeric_value(self) -> None:
        result = _mean_like([5.5])
        self.assertAlmostEqual(result, 5.5)

    def test_single_bool_value(self) -> None:
        result = _mean_like([True])
        self.assertAlmostEqual(result, 1.0)


# ---------------------------------------------------------------------------
# _aggregate_values
# ---------------------------------------------------------------------------

class AggregateValuesTest(unittest.TestCase):
    """Tests for the _aggregate_values helper."""

    def test_empty_returns_zero_float(self) -> None:
        self.assertEqual(_aggregate_values([]), 0.0)

    def test_numeric_values_return_mean(self) -> None:
        result = _aggregate_values([10, 20, 30])
        self.assertAlmostEqual(result, 20.0)

    def test_bool_values_return_float_mean(self) -> None:
        result = _aggregate_values([True, False, True])
        self.assertAlmostEqual(result, 2 / 3)

    def test_string_values_return_most_common(self) -> None:
        result = _aggregate_values(["a", "b", "a", "a", "b"])
        self.assertEqual(result, "a")

    def test_mixed_types_returns_most_common_string(self) -> None:
        result = _aggregate_values(["x", "y", 1, "x"])
        # All stringified: ["x", "y", "1", "x"] → "x" is most common
        self.assertEqual(result, "x")

    def test_single_value_returns_itself_as_float(self) -> None:
        result = _aggregate_values([7.0])
        self.assertAlmostEqual(result, 7.0)

    def test_single_string_returns_that_string(self) -> None:
        result = _aggregate_values(["only"])
        self.assertEqual(result, "only")


# ---------------------------------------------------------------------------
# EpisodeStats.seed field
# ---------------------------------------------------------------------------

class EpisodeStatsSeedFieldTest(unittest.TestCase):
    """Test that EpisodeStats correctly stores the new seed field."""

    def test_seed_stored(self) -> None:
        stats = _make_episode_stats(seed=999)
        self.assertEqual(stats.seed, 999)

    def test_seed_zero(self) -> None:
        stats = _make_episode_stats(seed=0)
        self.assertEqual(stats.seed, 0)

    def test_seed_in_asdict(self) -> None:
        stats = _make_episode_stats(seed=123)
        d = asdict(stats)
        self.assertIn("seed", d)
        self.assertEqual(d["seed"], 123)


# ---------------------------------------------------------------------------
# Trace helper functions in scenarios.py
# ---------------------------------------------------------------------------

class TraceHelpersTest(unittest.TestCase):
    """Tests for _trace_states, _trace_any_mode, _trace_any_sleep_phase,
    _trace_predator_memory_seen, _trace_escape_seen."""

    def test_trace_states_extracts_dict_states(self) -> None:
        trace = [
            {"tick": 0, "state": {"lizard_mode": "PATROL"}},
            {"tick": 1, "state": {"lizard_mode": "CHASE"}},
            {"tick": 2, "no_state_key": True},
        ]
        states = _trace_states(trace)
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0]["lizard_mode"], "PATROL")
        self.assertEqual(states[1]["lizard_mode"], "CHASE")

    def test_trace_states_skips_non_dict_state(self) -> None:
        trace = [
            {"state": "not_a_dict"},
            {"state": {"lizard_mode": "PATROL"}},
        ]
        states = _trace_states(trace)
        self.assertEqual(len(states), 1)

    def test_trace_states_empty_trace(self) -> None:
        self.assertEqual(_trace_states([]), [])

    def test_trace_any_mode_found(self) -> None:
        trace = [
            {"state": {"lizard_mode": "PATROL"}},
            {"state": {"lizard_mode": "CHASE"}},
        ]
        self.assertTrue(_trace_any_mode(trace, "CHASE"))

    def test_trace_any_mode_not_found(self) -> None:
        trace = [{"state": {"lizard_mode": "PATROL"}}]
        self.assertFalse(_trace_any_mode(trace, "RECOVER"))

    def test_trace_any_mode_empty_trace(self) -> None:
        self.assertFalse(_trace_any_mode([], "PATROL"))

    def test_trace_any_sleep_phase_found(self) -> None:
        trace = [
            {"state": {"sleep_phase": "AWAKE"}},
            {"state": {"sleep_phase": "DEEP_SLEEP"}},
        ]
        self.assertTrue(_trace_any_sleep_phase(trace, "DEEP_SLEEP"))

    def test_trace_any_sleep_phase_not_found(self) -> None:
        trace = [{"state": {"sleep_phase": "AWAKE"}}]
        self.assertFalse(_trace_any_sleep_phase(trace, "DEEP_SLEEP"))

    def test_trace_any_sleep_phase_empty(self) -> None:
        self.assertFalse(_trace_any_sleep_phase([], "DEEP_SLEEP"))

    def test_trace_predator_memory_seen_found(self) -> None:
        trace = [
            {"state": {"predator_memory": {"target": None}}},
            {"state": {"predator_memory": {"target": (3, 4)}}},
        ]
        self.assertTrue(_trace_predator_memory_seen(trace))

    def test_trace_predator_memory_seen_not_found_when_all_none(self) -> None:
        trace = [{"state": {"predator_memory": {"target": None}}}]
        self.assertFalse(_trace_predator_memory_seen(trace))

    def test_trace_predator_memory_seen_not_found_empty(self) -> None:
        self.assertFalse(_trace_predator_memory_seen([]))

    def test_trace_predator_memory_seen_non_dict_predator_memory(self) -> None:
        trace = [{"state": {"predator_memory": "not_a_dict"}}]
        self.assertFalse(_trace_predator_memory_seen(trace))

    def test_trace_escape_seen_true(self) -> None:
        trace = [
            {"predator_escape": False},
            {"predator_escape": True},
        ]
        self.assertTrue(_trace_escape_seen(trace))

    def test_trace_escape_seen_false(self) -> None:
        trace = [{"predator_escape": False}, {"predator_escape": False}]
        self.assertFalse(_trace_escape_seen(trace))

    def test_trace_escape_seen_missing_key(self) -> None:
        trace = [{"tick": 0}, {"tick": 1}]
        self.assertFalse(_trace_escape_seen(trace))

    def test_trace_escape_seen_empty(self) -> None:
        self.assertFalse(_trace_escape_seen([]))

    def test_trace_action_selection_payloads_extracts_action_center_messages(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "threat"},
                    },
                    {
                        "sender": "motor_cortex",
                        "topic": "action.execution",
                        "payload": {"selected_action": "MOVE_LEFT"},
                    },
                ]
            }
        ]
        payloads = _trace_action_selection_payloads(trace)
        self.assertEqual(payloads, [{"winning_valence": "threat"}])

    def test_payload_float_and_text_helpers(self) -> None:
        payload = {"winning_valence": "sleep", "module_gates": {"visual_cortex": 0.48}}
        self.assertEqual(_payload_text(payload, "winning_valence"), "sleep")
        self.assertAlmostEqual(_payload_float(payload, "module_gates", "visual_cortex"), 0.48)
        self.assertIsNone(_payload_text(payload, "missing"))
        self.assertIsNone(_payload_float(payload, "module_gates", "missing"))


# ---------------------------------------------------------------------------
# Check spec constants
# ---------------------------------------------------------------------------

class CheckSpecConstantsTest(unittest.TestCase):
    """Tests for the BehaviorCheckSpec constants defined in scenarios.py."""

    def test_night_rest_checks_has_three_items(self) -> None:
        self.assertEqual(len(NIGHT_REST_CHECKS), 3)

    def test_night_rest_check_names(self) -> None:
        names = {spec.name for spec in NIGHT_REST_CHECKS}
        self.assertIn("deep_night_shelter", names)
        self.assertIn("deep_sleep_reached", names)
        self.assertIn("sleep_debt_reduced", names)

    def test_predator_edge_checks_has_three_items(self) -> None:
        self.assertEqual(len(PREDATOR_EDGE_CHECKS), 3)

    def test_predator_edge_check_names(self) -> None:
        names = {spec.name for spec in PREDATOR_EDGE_CHECKS}
        self.assertIn("predator_detected", names)
        self.assertIn("predator_memory_recorded", names)
        self.assertIn("predator_reacted", names)

    def test_food_deprivation_checks_has_four_items(self) -> None:
        self.assertEqual(len(FOOD_DEPRIVATION_CHECKS), 4)

    def test_food_deprivation_check_names(self) -> None:
        self.assertEqual(
            [spec.name for spec in FOOD_DEPRIVATION_CHECKS],
            [
                "hunger_reduced",
                "approaches_food",
                "commits_to_foraging",
                "survives_deprivation",
            ],
        )

    def test_food_vs_predator_conflict_checks_has_three_items(self) -> None:
        self.assertEqual(len(FOOD_VS_PREDATOR_CONFLICT_CHECKS), 3)

    def test_sleep_vs_exploration_conflict_checks_has_three_items(self) -> None:
        self.assertEqual(len(SLEEP_VS_EXPLORATION_CONFLICT_CHECKS), 3)

    def test_all_specs_have_non_empty_fields(self) -> None:
        for spec in (
            list(NIGHT_REST_CHECKS)
            + list(PREDATOR_EDGE_CHECKS)
            + list(FOOD_DEPRIVATION_CHECKS)
            + list(FOOD_VS_PREDATOR_CONFLICT_CHECKS)
            + list(SLEEP_VS_EXPLORATION_CONFLICT_CHECKS)
        ):
            self.assertTrue(spec.name)
            self.assertTrue(spec.description)
            self.assertTrue(spec.expected)


# ---------------------------------------------------------------------------
# Score functions per scenario (via ScenarioSpec.score_episode)
# ---------------------------------------------------------------------------

class ScoreFunctionTest(unittest.TestCase):
    """Tests that each scenario's score_episode function produces a valid BehavioralEpisodeScore."""

    def _make_alive_stats(self, scenario: str) -> EpisodeStats:
        return _make_episode_stats(
            episode=1,
            seed=42,
            scenario=scenario,
            alive=True,
            predator_contacts=0,
            predator_sightings=0,
            alert_events=0,
            predator_response_events=0,
            predator_mode_transitions=0,
            predator_escapes=0,
            food_distance_delta=2.0,
            food_eaten=1,
            night_shelter_occupancy_rate=1.0,
            night_stillness_rate=1.0,
            night_role_distribution={"outside": 0.0, "entrance": 0.0, "inside": 0.0, "deep": 1.0},
            final_sleep_debt=0.1,
            final_hunger=0.3,
        )

    def _minimal_trace(self) -> list:
        """
        Provide a minimal simulation trace containing a single trace item useful for tests.
        
        The single trace item is a dict with:
        - "state": a dict containing:
            - "lizard_mode": "PATROL"
            - "sleep_phase": "DEEP_SLEEP"
            - "predator_memory": {"target": None}
        - "predator_escape": False
        
        Returns:
            list: A list with the single trace item described above.
        """
        return [
            {"state": {"lizard_mode": "PATROL", "sleep_phase": "DEEP_SLEEP", "predator_memory": {"target": None}}, "predator_escape": False},
        ]

    def _run_conflict_scenario(
        self,
        scenario_name: str,
        *,
        use_learned_arbitration: bool,
        enable_deterministic_guards: bool = False,
    ) -> tuple[BehavioralEpisodeScore, list[dict[str, object]]]:
        """
        Run a single episode for the given conflict scenario using a SpiderSimulation configured for learned or fixed arbitration, and return the scenario's behavioral score along with the captured trace.
        
        Parameters:
            scenario_name (str): Name of the scenario to run.
            use_learned_arbitration (bool): If True, configure the brain to use learned arbitration; if False, use fixed arbitration.
            enable_deterministic_guards (bool): If True, opt into the legacy deterministic guard ablation for conflict-priority diagnostics.
        
        Returns:
            tuple: (BehavioralEpisodeScore, list[dict[str, object]]) — the scored result for the episode and the list of trace items captured during the run.
        """
        arbitration_mode = "learned" if use_learned_arbitration else "fixed"
        guard_mode = "guarded" if enable_deterministic_guards else "unguarded"
        config = BrainAblationConfig(
            name=f"{scenario_name}_{arbitration_mode}_{guard_mode}",
            module_dropout=0.0,
            use_learned_arbitration=use_learned_arbitration,
            enable_deterministic_guards=enable_deterministic_guards,
        )
        sim = SpiderSimulation(seed=7, max_steps=30, brain_config=config)
        stats, trace = sim.run_episode(
            0,
            training=False,
            sample=False,
            capture_trace=True,
            scenario_name=scenario_name,
            debug_trace=False,
        )
        return get_scenario(scenario_name).score_episode(stats, trace), trace

    def _assert_numeric_mapping(self, payload: dict[str, object], key: str) -> None:
        self.assertIn(key, payload)
        value = payload[key]
        self.assertIsInstance(value, dict)
        self.assertGreater(len(value), 0)
        for item in value.values():
            self.assertIsInstance(item, Real)
            self.assertNotIsInstance(item, bool)

    def _assert_learned_arbitration_payload_contract(self, payload: dict[str, object]) -> None:
        """
        Assert that a learned-arbitration action-selection payload contains the required fields with the expected types and numeric mappings.
        
        Parameters:
            payload (dict[str, object]): The action-selection payload to validate; must include:
                - `winning_valence` (str)
                - `strategy` (str)
                - `dominant_module` (str)
                - `arbitration_value` (numeric, not a bool)
                - `learned_adjustment` (bool)
                - numeric mapping keys: `module_gates`, `valence_scores`, `valence_logits`,
                  `base_gates`, `gate_adjustments`, `module_contribution_share`
                  (each must be a mapping whose values are real numbers and not booleans)
        """
        self.assertIn("winning_valence", payload)
        self.assertIsInstance(payload["winning_valence"], str)
        self.assertIn("strategy", payload)
        self.assertIsInstance(payload["strategy"], str)
        self.assertIn("dominant_module", payload)
        self.assertIsInstance(payload["dominant_module"], str)
        self.assertIn("arbitration_value", payload)
        self.assertIsInstance(payload["arbitration_value"], Real)
        self.assertNotIsInstance(payload["arbitration_value"], bool)
        self.assertIn("learned_adjustment", payload)
        self.assertIsInstance(payload["learned_adjustment"], bool)

        for key in (
            "module_gates",
            "valence_scores",
            "valence_logits",
            "base_gates",
            "gate_adjustments",
            "module_contribution_share",
        ):
            self._assert_numeric_mapping(payload, key)

    def _food_deprivation_trace(
        self,
        *,
        positions: list[tuple[int, int]],
        food_distances: list[int],
        winning_valences: list[str],
        healths: list[float] | None = None,
    ) -> list[dict[str, object]]:
        """
        Builds a sequence of food-deprivation trace items for testing from parallel lists of per-tick values.
        
        Parameters:
            positions (list[tuple[int, int]]): Spider positions for each tick.
            food_distances (list[int]): Food distance values corresponding to each tick.
            winning_valences (list[str]): Action-selection winning valence string for each tick.
            healths (list[float] | None): Optional health values per tick; when omitted, each tick uses 1.0.
        
        Returns:
            list[dict[str, object]]: A list of trace item dictionaries, one per tick, constructed from the provided inputs.
        
        Raises:
            AssertionError: If the input lists do not all have the same length.
        """
        if healths is None:
            healths = [1.0 for _ in positions]
        self.assertEqual(len(positions), len(food_distances))
        self.assertEqual(len(positions), len(winning_valences))
        self.assertEqual(len(positions), len(healths))
        return [
            _make_food_deprivation_trace_item(
                tick=tick,
                pos=pos,
                health=health,
                food_dist=food_dist,
                winning_valence=winning_valence,
            )
            for tick, (pos, food_dist, winning_valence, health) in enumerate(
                zip(positions, food_distances, winning_valences, healths, strict=True)
            )
        ]

    def _food_deprivation_shelter_and_outside_cells(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Select a representative deep-shelter cell and a traversable cell outside the shelter from the "central_burrow" map.
        
        Returns:
            tuple: A pair of 2-tuples of ints: `(deep_cell, outside_cell)`, where `deep_cell` is a coordinate inside the deep shelter and `outside_cell` is a coordinate of a traversable cell not in the shelter.
        """
        world = SpiderWorld(seed=11, map_template="central_burrow")
        deep = sorted(world.shelter_deep_cells)[0]
        outside = sorted(
            cell
            for cell in world.map_template.traversable_cells
            if cell not in world.shelter_cells
        )[0]
        return deep, outside

    def test_all_scenarios_score_episode_returns_behavioral_episode_score(self) -> None:
        empty_trace: list = []
        for name in SCENARIO_NAMES:
            spec = get_scenario(name)
            stats = self._make_alive_stats(name)
            score = spec.score_episode(stats, empty_trace)
            self.assertIsInstance(score, BehavioralEpisodeScore, msg=f"Scenario {name!r}")

    def test_all_scenarios_score_episode_has_checks(self) -> None:
        empty_trace: list = []
        for name in SCENARIO_NAMES:
            spec = get_scenario(name)
            stats = self._make_alive_stats(name)
            score = spec.score_episode(stats, empty_trace)
            self.assertIsInstance(score.checks, dict, msg=f"Scenario {name!r}")

    def test_all_scenarios_score_episode_has_behavior_metrics(self) -> None:
        for name in SCENARIO_NAMES:
            spec = get_scenario(name)
            stats = self._make_alive_stats(name)
            score = spec.score_episode(stats, [])
            self.assertIsInstance(score.behavior_metrics, dict, msg=f"Scenario {name!r}")

    def test_night_rest_score_uses_deep_night_shelter_check(self) -> None:
        spec = get_scenario("night_rest")
        stats = _make_episode_stats(
            scenario="night_rest",
            night_role_distribution={"outside": 0.0, "entrance": 0.0, "inside": 0.0, "deep": 1.0},
            final_sleep_debt=NIGHT_REST_INITIAL_SLEEP_DEBT - 0.45,
        )
        trace = [{"state": {"sleep_phase": "DEEP_SLEEP"}}]
        score = spec.score_episode(stats, trace)
        self.assertIn("deep_night_shelter", score.checks)

    def test_food_deprivation_score_checks_hunger_reduction(self) -> None:
        spec = get_scenario("food_deprivation")
        # With final_hunger=0.78, reduction matches the configured deprivation baseline and should pass.
        stats = _make_episode_stats(
            scenario="food_deprivation",
            final_hunger=0.78,
            food_distance_delta=1.0,
            food_eaten=0,
            alive=True,
        )
        score = spec.score_episode(stats, [])
        self.assertIn("hunger_reduced", score.checks)

    def test_food_deprivation_score_fails_when_no_progress(self) -> None:
        spec = get_scenario("food_deprivation")
        # High hunger remaining, no food eaten, no food progress → should fail
        stats = _make_episode_stats(
            scenario="food_deprivation",
            final_hunger=0.95,
            food_distance_delta=0.0,  # approaches_food fails
            food_eaten=0,
            alive=False,
        )
        score = spec.score_episode(stats, [])
        # At least one check should fail
        self.assertFalse(score.success)

    def test_food_deprivation_emits_partial_progress_diagnostics(self) -> None:
        spec = get_scenario("food_deprivation")
        stats = _make_episode_stats(
            scenario="food_deprivation",
            final_hunger=0.96,
            food_distance_delta=5.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        score = spec.score_episode(stats, [])
        self.assertEqual(score.behavior_metrics["progress_band"], "advanced")
        self.assertEqual(score.behavior_metrics["outcome_band"], "partial_progress_died")
        self.assertTrue(score.behavior_metrics["partial_progress"])
        self.assertTrue(score.behavior_metrics["died_after_progress"])
        self.assertTrue(score.behavior_metrics["died_without_contact"])

    def test_food_deprivation_emits_trace_diagnostic_metrics(self) -> None:
        spec = get_scenario("food_deprivation")
        deep, outside = self._food_deprivation_shelter_and_outside_cells()
        stats = _make_episode_stats(
            scenario="food_deprivation",
            final_hunger=0.96,
            food_distance_delta=3.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        trace = self._food_deprivation_trace(
            positions=[deep, outside, outside],
            food_distances=[10, 9, 7],
            winning_valences=["hunger", "hunger", "sleep"],
            healths=[1.0, 1.0, 0.0],
        )

        score = spec.score_episode(stats, trace)

        self.assertTrue(
            {
                "min_food_distance_reached",
                "left_shelter",
                "shelter_exit_tick",
                "death_tick",
                "hunger_valence_rate",
                "failure_mode",
            }.issubset(score.behavior_metrics)
        )
        self.assertEqual(score.behavior_metrics["min_food_distance_reached"], 7.0)
        self.assertTrue(score.behavior_metrics["left_shelter"])
        self.assertEqual(score.behavior_metrics["shelter_exit_tick"], 1)
        self.assertEqual(score.behavior_metrics["death_tick"], 2)
        self.assertAlmostEqual(score.behavior_metrics["hunger_valence_rate"], 2 / 3)
        self.assertEqual(score.behavior_metrics["failure_mode"], "timing_failure")
        self.assertTrue(score.checks["commits_to_foraging"].passed)

    def test_food_deprivation_failure_modes_from_stats_and_trace_fixtures(self) -> None:
        spec = get_scenario("food_deprivation")
        deep, outside = self._food_deprivation_shelter_and_outside_cells()
        cases = {
            "no_commitment": {
                "stats": {
                    "final_hunger": 0.96,
                    "food_distance_delta": 0.0,
                    "food_eaten": 0,
                    "alive": True,
                },
                "trace": {
                    "positions": [deep, deep],
                    "food_distances": [6, 6],
                    "winning_valences": ["hunger", "hunger"],
                },
            },
            "orientation_failure": {
                "stats": {
                    "final_hunger": 0.70,
                    "food_distance_delta": 0.0,
                    "food_eaten": 0,
                    "alive": True,
                },
                "trace": {
                    "positions": [deep, outside, outside],
                    "food_distances": [6, 6, 6],
                    "winning_valences": ["hunger", "hunger", "hunger"],
                },
            },
            "timing_failure": {
                "stats": {
                    "final_hunger": 0.96,
                    "food_distance_delta": 2.0,
                    "food_eaten": 0,
                    "alive": False,
                },
                "trace": {
                    "positions": [deep, outside, outside],
                    "food_distances": [6, 5, 4],
                    "winning_valences": ["hunger", "hunger", "sleep"],
                    "healths": [1.0, 1.0, 0.0],
                },
            },
            "scoring_mismatch": {
                "stats": {
                    "final_hunger": 0.96,
                    "food_distance_delta": 0.0,
                    "food_eaten": 0,
                    "alive": True,
                },
                "trace": {
                    "positions": [deep, outside, outside],
                    "food_distances": [6, 5, 4],
                    "winning_valences": ["hunger", "hunger", "sleep"],
                },
            },
            "success": {
                "stats": {
                    "final_hunger": 0.70,
                    "food_distance_delta": 2.0,
                    "food_eaten": 0,
                    "alive": True,
                },
                "trace": {
                    "positions": [deep, outside, outside],
                    "food_distances": [6, 5, 4],
                    "winning_valences": ["hunger", "hunger", "sleep"],
                },
            },
        }
        for expected, fixture in cases.items():
            with self.subTest(expected=expected):
                stats = _make_episode_stats(
                    scenario="food_deprivation",
                    predator_contacts=0,
                    **fixture["stats"],
                )
                trace = self._food_deprivation_trace(**fixture["trace"])

                score = spec.score_episode(stats, trace)

                self.assertEqual(score.behavior_metrics["failure_mode"], expected)

    def test_food_deprivation_timing_failure_when_approached_but_died_before_eating(self) -> None:
        spec = get_scenario("food_deprivation")
        deep, outside = self._food_deprivation_shelter_and_outside_cells()
        stats = _make_episode_stats(
            scenario="food_deprivation",
            final_hunger=0.96,
            food_distance_delta=2.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        trace = self._food_deprivation_trace(
            positions=[deep, outside, outside],
            food_distances=[8, 6, 5],
            winning_valences=["hunger", "hunger", "sleep"],
            healths=[1.0, 1.0, 0.0],
        )

        score = spec.score_episode(stats, trace)

        self.assertEqual(score.behavior_metrics["failure_mode"], "timing_failure")
        self.assertLess(score.behavior_metrics["min_food_distance_reached"], 8.0)
        self.assertEqual(score.behavior_metrics["food_eaten"], 0)
        self.assertFalse(score.behavior_metrics["alive"])

    def test_food_deprivation_commitment_check_passes_with_exit_and_hunger_priority(self) -> None:
        spec = get_scenario("food_deprivation")
        deep, outside = self._food_deprivation_shelter_and_outside_cells()
        stats = _make_episode_stats(
            scenario="food_deprivation",
            final_hunger=0.70,
            food_distance_delta=1.0,
            food_eaten=0,
            alive=True,
        )
        trace = self._food_deprivation_trace(
            positions=[deep, outside],
            food_distances=[6, 5],
            winning_valences=["hunger", "sleep"],
        )

        score = spec.score_episode(stats, trace)

        self.assertTrue(score.checks["commits_to_foraging"].passed)
        self.assertEqual(
            score.checks["commits_to_foraging"].value,
            {"left_shelter": True, "hunger_valence_rate": 0.5},
        )

    def test_food_deprivation_commitment_check_fails_without_exit_or_hunger_priority(self) -> None:
        spec = get_scenario("food_deprivation")
        deep, outside = self._food_deprivation_shelter_and_outside_cells()
        cases = {
            "stayed_sheltered": {
                "positions": [deep, deep],
                "food_distances": [6, 5],
                "winning_valences": ["hunger", "hunger"],
                "expected_value": {"left_shelter": False, "hunger_valence_rate": 1.0},
            },
            "hunger_suppressed": {
                "positions": [deep, outside, outside],
                "food_distances": [6, 5, 4],
                "winning_valences": ["sleep", "sleep", "hunger"],
                "expected_value": {"left_shelter": True, "hunger_valence_rate": 1 / 3},
            },
        }
        stats = _make_episode_stats(
            scenario="food_deprivation",
            final_hunger=0.70,
            food_distance_delta=1.0,
            food_eaten=0,
            alive=True,
        )
        for name, fixture in cases.items():
            with self.subTest(case=name):
                trace = self._food_deprivation_trace(
                    positions=fixture["positions"],
                    food_distances=fixture["food_distances"],
                    winning_valences=fixture["winning_valences"],
                )

                score = spec.score_episode(stats, trace)

                self.assertFalse(score.checks["commits_to_foraging"].passed)
                self.assertEqual(
                    score.checks["commits_to_foraging"].value,
                    fixture["expected_value"],
                )

    def test_food_deprivation_initial_distance_prefers_pre_step_observation(self) -> None:
        spec = get_scenario("food_deprivation")
        _, outside = self._food_deprivation_shelter_and_outside_cells()
        stats = _make_episode_stats(
            scenario="food_deprivation",
            final_hunger=0.96,
            food_distance_delta=0.0,
            food_eaten=0,
            alive=True,
            predator_contacts=0,
        )
        trace = [
            {
                "tick": 0,
                "state": {
                    "x": outside[0],
                    "y": outside[1],
                    "health": 1.0,
                    "map_template": "central_burrow",
                },
                "observation": {"meta": {"food_dist": 8.0}},
                "next_observation": {"meta": {"food_dist": 6.0}},
                "messages": [
                    {
                        "sender": "environment",
                        "topic": "observation",
                        "payload": {"meta": {"food_dist": 7.0}},
                    },
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "hunger"},
                    },
                ],
            }
        ]

        score = spec.score_episode(stats, trace)

        self.assertEqual(score.behavior_metrics["min_food_distance_reached"], 6.0)
        self.assertEqual(score.behavior_metrics["failure_mode"], "scoring_mismatch")

    def test_food_deprivation_initial_distance_reconstructs_from_snapshot(self) -> None:
        spec = get_scenario("food_deprivation")
        _, outside = self._food_deprivation_shelter_and_outside_cells()
        stats = _make_episode_stats(
            scenario="food_deprivation",
            final_hunger=0.96,
            food_distance_delta=0.0,
            food_eaten=0,
            alive=True,
            predator_contacts=0,
        )
        trace = [
            {
                "tick": 0,
                "state": {
                    "x": outside[0],
                    "y": outside[1],
                    "health": 1.0,
                    "map_template": "central_burrow",
                },
                "event_log": [
                    {
                        "stage": "pre_tick",
                        "name": "snapshot",
                        "payload": {"prev_food_dist": 8, "spider_pos": [outside[0], outside[1]]},
                    }
                ],
                "messages": [
                    {
                        "sender": "environment",
                        "topic": "observation",
                        "payload": {"meta": {"food_dist": 6.0}},
                    },
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "hunger"},
                    },
                ],
            }
        ]

        score = spec.score_episode(stats, trace)

        self.assertEqual(score.behavior_metrics["min_food_distance_reached"], 6.0)
        self.assertEqual(score.behavior_metrics["failure_mode"], "scoring_mismatch")

    def test_food_deprivation_failure_classifier_modes(self) -> None:
        base_metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "hunger_valence_rate": 0.75,
            "initial_food_distance": 10.0,
            "min_food_distance_reached": 6.0,
            "food_distance_delta": 4.0,
            "alive": True,
            "food_eaten": 0,
        }
        cases = {
            "success": {**base_metrics, "checks_passed": True},
            "no_commitment": {**base_metrics, "hunger_valence_rate": 0.25},
            "orientation_failure": {
                **base_metrics,
                "min_food_distance_reached": 10.0,
                "food_distance_delta": 0.0,
            },
            "timing_failure": {**base_metrics, "alive": False},
            "scoring_mismatch": base_metrics,
        }
        for expected, metrics in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    _classify_food_deprivation_failure(metrics),
                    expected,
                )

    def test_food_vs_predator_conflict_reads_action_selection_messages(self) -> None:
        spec = get_scenario("food_vs_predator_conflict")
        stats = _make_episode_stats(
            scenario="food_vs_predator_conflict",
            alive=True,
            predator_contacts=0,
        )
        trace = [
            _make_action_selection_trace_item(
                {
                    "winning_valence": "threat",
                    "module_gates": {"hunger_center": 0.18},
                    "evidence": {
                        "threat": {
                            "predator_visible": 1.0,
                            "predator_proximity": 0.9,
                            "predator_certainty": 0.8,
                        }
                    },
                }
            )
        ]
        score = spec.score_episode(stats, trace)
        self.assertTrue(score.success)
        self.assertIn("threat_priority_rate", score.behavior_metrics)

    def test_food_vs_predator_conflict_requires_rate_threshold(self) -> None:
        spec = get_scenario("food_vs_predator_conflict")
        stats = _make_episode_stats(
            scenario="food_vs_predator_conflict",
            alive=True,
            predator_contacts=0,
        )
        trace = [
            _make_action_selection_trace_item(
                {
                    "winning_valence": "threat",
                    "module_gates": {"hunger_center": 0.18},
                    "evidence": {
                        "threat": {
                            "predator_visible": 1.0,
                            "predator_proximity": 0.9,
                            "predator_certainty": 0.8,
                        }
                    },
                }
            ),
            _make_action_selection_trace_item(
                {
                    "winning_valence": "hunger",
                    "module_gates": {"hunger_center": 1.0},
                    "evidence": {
                        "threat": {
                            "predator_visible": 1.0,
                            "predator_proximity": 0.9,
                            "predator_certainty": 0.8,
                        }
                    },
                }
            ),
        ]
        score = spec.score_episode(stats, trace)
        self.assertFalse(score.checks["threat_priority"].passed)
        self.assertTrue(score.checks["foraging_suppressed_under_threat"].passed)
        self.assertAlmostEqual(score.behavior_metrics["threat_priority_rate"], 0.5)
        self.assertAlmostEqual(score.behavior_metrics["foraging_suppressed_rate"], 1.0)

    def test_sleep_vs_exploration_conflict_reads_action_selection_messages(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        stats = _make_episode_stats(
            scenario="sleep_vs_exploration_conflict",
            sleep_events=2,
            final_sleep_debt=SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - 0.2,
        )
        trace = [
            _make_action_selection_trace_item(
                {
                    "winning_valence": "sleep",
                    "module_gates": {"visual_cortex": 0.48, "sensory_cortex": 0.56},
                    "evidence": {
                        "sleep": {"sleep_debt": 0.92, "fatigue": 0.94},
                        "threat": {"predator_visible": 0.0},
                    },
                }
            )
        ]
        score = spec.score_episode(stats, trace)
        self.assertTrue(score.success)
        self.assertIn("sleep_priority_rate", score.behavior_metrics)

    def test_sleep_vs_exploration_conflict_requires_rate_threshold(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        stats = _make_episode_stats(
            scenario="sleep_vs_exploration_conflict",
            sleep_events=2,
            final_sleep_debt=SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - 0.2,
        )
        trace = [
            _make_action_selection_trace_item(
                {
                    "winning_valence": "sleep",
                    "module_gates": {"visual_cortex": 0.48, "sensory_cortex": 0.56},
                    "evidence": {
                        "sleep": {"sleep_debt": 0.92, "fatigue": 0.94},
                        "threat": {"predator_visible": 0.0},
                    },
                }
            ),
            _make_action_selection_trace_item(
                {
                    "winning_valence": "exploration",
                    "module_gates": {"visual_cortex": 0.96, "sensory_cortex": 0.92},
                    "evidence": {
                        "sleep": {"sleep_debt": 0.92, "fatigue": 0.94},
                        "threat": {"predator_visible": 0.0},
                    },
                }
            ),
        ]
        score = spec.score_episode(stats, trace)
        self.assertFalse(score.checks["sleep_priority"].passed)
        self.assertTrue(score.checks["exploration_suppressed_under_sleep_pressure"].passed)
        self.assertAlmostEqual(score.behavior_metrics["sleep_priority_rate"], 0.5)
        self.assertAlmostEqual(score.behavior_metrics["exploration_suppressed_rate"], 1.0)

    def test_food_vs_predator_conflict_passes_with_learned_arbitration_trace(self) -> None:
        """
        Integration test that runs the food_vs_predator_conflict scenario with learned arbitration
        and verifies the produced action-selection payloads conform to the learned-arbitration contract
        and that the scenario meets minimum diagnostic thresholds.
        
        Asserts:
        - At least one action-selection payload is present in the trace.
        - Each payload satisfies the learned-arbitration payload contract.
        - The scenario's `threat_priority_rate` is >= 0.8.
        - The scored episode is marked as successful.
        """
        score, trace = self._run_conflict_scenario(
            "food_vs_predator_conflict",
            use_learned_arbitration=True,
            enable_deterministic_guards=True,
        )
        payloads = list(_trace_action_selection_payloads(trace))

        self.assertGreater(len(payloads), 0)
        for payload in payloads:
            self._assert_learned_arbitration_payload_contract(payload)
        self.assertGreaterEqual(score.behavior_metrics["threat_priority_rate"], 0.8)
        self.assertTrue(score.success, msg=str(score.behavior_metrics))

    def test_food_vs_predator_conflict_unguarded_learned_trace_exercises_guards_off_path(self) -> None:
        score, trace = self._run_conflict_scenario(
            "food_vs_predator_conflict",
            use_learned_arbitration=True,
            enable_deterministic_guards=False,
        )
        payloads = list(_trace_action_selection_payloads(trace))

        self.assertGreater(len(payloads), 0)
        for payload in payloads:
            self._assert_learned_arbitration_payload_contract(payload)
            self.assertIn("guards_applied", payload)
            self.assertIs(payload["guards_applied"], False)
        self.assertIn("threat_priority_rate", score.behavior_metrics)

    def test_sleep_vs_exploration_conflict_passes_with_learned_arbitration_trace(self) -> None:
        """
        Verify the sleep-vs-exploration conflict scenario succeeds and produces action-selection trace diagnostics when using learned arbitration.
        
        Asserts that at least one action-selection payload is present and each payload contains a `winning_valence` key and a `module_gates` dictionary, and that the resulting score reports a `sleep_priority_rate` of at least 0.8 and overall success.
        """
        score, trace = self._run_conflict_scenario(
            "sleep_vs_exploration_conflict",
            use_learned_arbitration=True,
        )
        payloads = list(_trace_action_selection_payloads(trace))

        self.assertGreater(len(payloads), 0)
        for payload in payloads:
            self._assert_learned_arbitration_payload_contract(payload)
        self.assertGreaterEqual(score.behavior_metrics["sleep_priority_rate"], 0.8)
        self.assertTrue(score.success, msg=str(score.behavior_metrics))

    def test_learned_and_fixed_arbitration_conflict_diagnostics(self) -> None:
        scenarios = {
            "food_vs_predator_conflict": "threat_priority_rate",
            "sleep_vs_exploration_conflict": "sleep_priority_rate",
        }
        comparison: dict[str, dict[str, float]] = {"learned": {}, "fixed": {}}
        for scenario_name, metric_name in scenarios.items():
            learned_score, _ = self._run_conflict_scenario(
                scenario_name,
                use_learned_arbitration=True,
                enable_deterministic_guards=True,
            )
            fixed_score, _ = self._run_conflict_scenario(
                scenario_name,
                use_learned_arbitration=False,
                enable_deterministic_guards=True,
            )
            comparison["learned"][scenario_name] = float(learned_score.behavior_metrics[metric_name])
            comparison["fixed"][scenario_name] = float(fixed_score.behavior_metrics[metric_name])

        for mode, values in comparison.items():
            for value in values.values():
                self.assertGreaterEqual(
                    value,
                    0.8,
                    msg=f"{mode} arbitration diagnostics: {comparison}",
                )

    def test_open_field_foraging_score_passes_with_food_progress(self) -> None:
        spec = get_scenario("open_field_foraging")
        stats = _make_episode_stats(
            scenario="open_field_foraging",
            food_distance_delta=5.0,
            food_eaten=1,
            alive=True,
            predator_contacts=0,
        )
        score = spec.score_episode(stats, [])
        self.assertTrue(score.checks["made_food_progress"].passed)
        self.assertTrue(score.checks["survives_exposure"].passed)

    def test_open_field_foraging_emits_trace_diagnostic_metrics(self) -> None:
        spec = get_scenario("open_field_foraging")
        stats = _make_episode_stats(
            scenario="open_field_foraging",
            food_distance_delta=2.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        shelter_pos, outside_pos = _open_field_trace_positions()
        trace = [
            _make_open_field_trace_item(
                tick=0,
                pos=shelter_pos,
                health=1.0,
                food_dist=6,
                winning_valence="hunger",
                food_signal_strength=0.4,
            ),
            _make_open_field_trace_item(
                tick=1,
                pos=outside_pos,
                health=1.0,
                food_dist=5,
                winning_valence="hunger",
                food_signal_strength=0.5,
            ),
            _make_open_field_trace_item(
                tick=2,
                pos=outside_pos,
                health=0.0,
                food_dist=4,
                winning_valence="sleep",
                food_signal_strength=0.3,
            ),
        ]

        score = spec.score_episode(stats, trace)

        self.assertTrue(
            {
                "initial_food_distance",
                "min_food_distance_reached",
                "left_shelter",
                "shelter_exit_tick",
                "death_tick",
                "hunger_valence_rate",
                "predator_visible_ticks",
                "initial_food_signal_strength",
                "max_food_signal_strength",
                "food_signal_tick_rate",
                "failure_mode",
            }.issubset(score.behavior_metrics)
        )
        self.assertEqual(score.behavior_metrics["initial_food_distance"], 6.0)
        self.assertEqual(score.behavior_metrics["min_food_distance_reached"], 4.0)
        self.assertTrue(score.behavior_metrics["left_shelter"])
        self.assertEqual(score.behavior_metrics["shelter_exit_tick"], 1)
        self.assertEqual(score.behavior_metrics["death_tick"], 2)
        self.assertAlmostEqual(score.behavior_metrics["hunger_valence_rate"], 2 / 3)
        self.assertEqual(score.behavior_metrics["predator_visible_ticks"], 0)
        self.assertAlmostEqual(score.behavior_metrics["initial_food_signal_strength"], 0.4)
        self.assertAlmostEqual(score.behavior_metrics["max_food_signal_strength"], 0.5)
        self.assertAlmostEqual(score.behavior_metrics["food_signal_tick_rate"], 1.0)
        self.assertEqual(score.behavior_metrics["failure_mode"], "progressed_then_died")

    def test_food_signal_strength_reads_top_level_hunger_memory(self) -> None:
        signal = _food_signal_strength(
            {
                "hunger": {
                    "food_memory_dx": 0.25,
                    "food_memory_dy": -0.25,
                    "food_memory_age": 0.25,
                },
            }
        )

        expected_signal = 1.0 - 0.25
        self.assertAlmostEqual(signal, expected_signal, places=6)

    def test_open_field_foraging_failure_classifier_modes(self) -> None:
        base_metrics = _open_field_failure_base_metrics()
        cases = {
            "success": {**base_metrics, "checks_passed": True},
            "never_left_shelter": {**base_metrics, "left_shelter": False},
            "no_hunger_commitment": {**base_metrics, "hunger_valence_rate": 0.49},
            "left_without_food_signal": {
                **base_metrics,
                "max_food_signal_strength": 0.0,
                "initial_food_signal_strength": 0.0,
            },
            "orientation_failure": {
                **base_metrics,
                "min_food_distance_reached": 8.0,
                "food_distance_delta": 0.0,
                "alive": False,
            },
            "progressed_then_died": {**base_metrics, "alive": False},
            "stall": {
                **base_metrics,
                "min_food_distance_reached": 8.0,
                "food_distance_delta": 0.0,
                "predator_visible_ticks": 0,
            },
            "scoring_mismatch": base_metrics,
        }
        for expected, metrics in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    _classify_open_field_foraging_failure(metrics),
                    expected,
                )
        self.assertEqual(
            _classify_open_field_foraging_failure(
                {
                    **base_metrics,
                    "initial_food_signal_strength": 0.0,
                    "max_food_signal_strength": 0.8,
                }
            ),
            "left_without_food_signal",
        )

    def test_open_field_foraging_current_impossible_shape_is_left_without_food_signal(self) -> None:
        spec = get_scenario("open_field_foraging")
        stats = _make_episode_stats(
            scenario="open_field_foraging",
            food_distance_delta=-3.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        shelter_pos, outside_pos = _open_field_trace_positions()
        trace = [
            _make_open_field_trace_item(
                tick=0,
                pos=shelter_pos,
                health=1.0,
                food_dist=10,
                winning_valence="hunger",
                food_signal_strength=0.0,
            ),
            _make_open_field_trace_item(
                tick=1,
                pos=outside_pos,
                health=0.0,
                food_dist=11,
                winning_valence="hunger",
                food_signal_strength=0.0,
            ),
        ]

        score = spec.score_episode(stats, trace)

        self.assertEqual(score.behavior_metrics["failure_mode"], "left_without_food_signal")

    def test_open_field_foraging_alive_partial_progress_is_scoring_mismatch(self) -> None:
        spec = get_scenario("open_field_foraging")
        stats = _make_episode_stats(
            scenario="open_field_foraging",
            food_distance_delta=1.0,
            food_eaten=0,
            alive=True,
            predator_contacts=0,
        )
        shelter_pos, outside_pos = _open_field_trace_positions()
        trace = [
            _make_open_field_trace_item(
                tick=0,
                pos=shelter_pos,
                health=1.0,
                food_dist=6,
                winning_valence="hunger",
                food_signal_strength=0.5,
            ),
            _make_open_field_trace_item(
                tick=1,
                pos=outside_pos,
                health=1.0,
                food_dist=5,
                winning_valence="hunger",
                food_signal_strength=0.5,
            ),
        ]

        score = spec.score_episode(stats, trace)

        self.assertTrue(score.checks["made_food_progress"].passed)
        self.assertFalse(score.checks["foraging_viable"].passed)
        self.assertTrue(score.checks["survives_exposure"].passed)
        self.assertEqual(score.behavior_metrics["failure_mode"], "scoring_mismatch")

    def test_open_field_foraging_marks_regressed_and_died(self) -> None:
        spec = get_scenario("open_field_foraging")
        stats = _make_episode_stats(
            scenario="open_field_foraging",
            food_distance_delta=-3.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        score = spec.score_episode(stats, [])
        self.assertEqual(score.behavior_metrics["progress_band"], "regressed")
        self.assertEqual(score.behavior_metrics["outcome_band"], "regressed_and_died")

    def test_corridor_gauntlet_marks_stalled_and_died_when_no_progress(self) -> None:
        spec = get_scenario("corridor_gauntlet")
        stats = _make_episode_stats(
            scenario="corridor_gauntlet",
            food_distance_delta=0.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        score = spec.score_episode(stats, [])
        self.assertEqual(score.behavior_metrics["progress_band"], "stalled")
        self.assertEqual(score.behavior_metrics["outcome_band"], "stalled_and_died")

    def test_predator_edge_score_detects_sightings(self) -> None:
        spec = get_scenario("predator_edge")
        stats = _make_episode_stats(
            scenario="predator_edge",
            predator_sightings=1,
            alert_events=0,
            predator_response_events=1,
            predator_mode_transitions=0,
        )
        score = spec.score_episode(stats, [])
        self.assertTrue(score.checks["predator_detected"].passed)

    def test_recover_after_failed_chase_uses_trace_modes(self) -> None:
        spec = get_scenario("recover_after_failed_chase")
        stats = _make_episode_stats(scenario="recover_after_failed_chase", alive=True)
        trace = [
            {"state": {"lizard_mode": "RECOVER"}},
            {"state": {"lizard_mode": "WAIT"}},
        ]
        score = spec.score_episode(stats, trace)
        self.assertTrue(score.checks["predator_enters_recover"].passed)
        self.assertTrue(score.checks["predator_returns_to_wait"].passed)

    def test_recover_after_failed_chase_fails_without_recover(self) -> None:
        spec = get_scenario("recover_after_failed_chase")
        stats = _make_episode_stats(scenario="recover_after_failed_chase", alive=True)
        trace = [{"state": {"lizard_mode": "PATROL"}}]
        score = spec.score_episode(stats, trace)
        self.assertFalse(score.checks["predator_enters_recover"].passed)

    def test_score_episode_returns_correct_scenario_name(self) -> None:
        for name in SCENARIO_NAMES:
            spec = get_scenario(name)
            stats = _make_episode_stats(scenario=name)
            score = spec.score_episode(stats, [])
            self.assertEqual(score.scenario, name, msg=f"Expected scenario name '{name}'")


class CorridorGauntletClassifierTest(unittest.TestCase):
    def test_failure_classifier_modes(self) -> None:
        base_stats = _make_episode_stats(
            scenario="corridor_gauntlet",
            food_distance_delta=0.0,
            alive=True,
            predator_contacts=0,
        )
        trace_metrics = {
            "left_shelter": True,
            "shelter_exit_tick": 1,
            "predator_visible_ticks": 0,
            "peak_food_progress": 0.0,
            "death_tick": None,
        }
        cases = {
            "success": {
                "stats": base_stats,
                "trace_metrics": {**trace_metrics, "left_shelter": False},
                "full_success": True,
            },
            "frozen_in_shelter": {
                "stats": base_stats,
                "trace_metrics": {**trace_metrics, "left_shelter": False},
                "full_success": False,
            },
            "contact_failure_died": {
                "stats": _make_episode_stats(
                    scenario="corridor_gauntlet",
                    food_distance_delta=2.0,
                    alive=False,
                    predator_contacts=1,
                ),
                "trace_metrics": trace_metrics,
                "full_success": False,
            },
            "contact_failure_survived": {
                "stats": _make_episode_stats(
                    scenario="corridor_gauntlet",
                    food_distance_delta=2.0,
                    alive=True,
                    predator_contacts=1,
                ),
                "trace_metrics": trace_metrics,
                "full_success": False,
            },
            "survived_no_progress": {
                "stats": base_stats,
                "trace_metrics": trace_metrics,
                "full_success": False,
            },
            "progress_then_died": {
                "stats": _make_episode_stats(
                    scenario="corridor_gauntlet",
                    food_distance_delta=2.0,
                    alive=False,
                    predator_contacts=0,
                ),
                "trace_metrics": trace_metrics,
                "full_success": False,
            },
            "scoring_mismatch": {
                "stats": _make_episode_stats(
                    scenario="corridor_gauntlet",
                    food_distance_delta=0.0,
                    alive=False,
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
    def test_trace_metric_extractor_reads_trace_shape(self) -> None:
        shelter_pos, outside_pos = _corridor_trace_positions()
        trace = [
            _make_corridor_trace_item(
                tick=0,
                pos=shelter_pos,
                health=1.0,
                food_dist=9,
                predator_visible=False,
            ),
            _make_corridor_trace_item(
                tick=1,
                pos=outside_pos,
                health=1.0,
                food_dist=7,
                predator_visible=True,
            ),
            _make_corridor_trace_item(
                tick=2,
                pos=outside_pos,
                health=0.0,
                food_dist=8,
                predator_visible=True,
            ),
        ]

        metrics = _trace_corridor_metrics(trace)

        self.assertTrue(metrics["left_shelter"])
        self.assertEqual(metrics["shelter_exit_tick"], 1)
        self.assertEqual(metrics["predator_visible_ticks"], 2)
        self.assertEqual(metrics["peak_food_progress"], 2.0)
        self.assertEqual(metrics["death_tick"], 2)

    def test_emits_trace_diagnostic_metrics_and_failure_mode(self) -> None:
        spec = get_scenario("corridor_gauntlet")
        stats = _make_episode_stats(
            scenario="corridor_gauntlet",
            food_distance_delta=1.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        shelter_pos, outside_pos = _corridor_trace_positions()
        trace = [
            _make_corridor_trace_item(
                tick=0,
                pos=shelter_pos,
                health=1.0,
                food_dist=9,
                predator_visible=False,
            ),
            _make_corridor_trace_item(
                tick=1,
                pos=outside_pos,
                health=1.0,
                food_dist=7,
                predator_visible=True,
            ),
            _make_corridor_trace_item(
                tick=2,
                pos=outside_pos,
                health=0.0,
                food_dist=8,
                predator_visible=True,
            ),
        ]

        score = spec.score_episode(stats, trace)

        self.assertTrue(
            {
                "failure_mode",
                "left_shelter",
                "shelter_exit_tick",
                "predator_visible_ticks",
                "peak_food_progress",
                "death_tick",
            }.issubset(score.behavior_metrics)
        )
        self.assertTrue(score.behavior_metrics["left_shelter"])
        self.assertEqual(score.behavior_metrics["shelter_exit_tick"], 1)
        self.assertEqual(score.behavior_metrics["predator_visible_ticks"], 2)
        self.assertEqual(score.behavior_metrics["peak_food_progress"], 2.0)
        self.assertEqual(score.behavior_metrics["death_tick"], 2)
        self.assertEqual(score.behavior_metrics["failure_mode"], "progress_then_died")


class ExposedDayForagingClassifierTest(unittest.TestCase):
    def test_failure_classifier_modes(self) -> None:
        """
        Unit test that validates the exposed-day-foraging failure classifier maps representative metric combinations to the correct failure-mode labels.
        
        Verifies that `_classify_exposed_day_foraging_failure` returns the expected label for a set of predefined metric cases including "success", "cautious_inert", "foraging_and_died", "threatened_retreat", "partial_progress", "stall", and "scoring_mismatch".
        """
        base_metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "food_distance_delta": 0.0,
            "peak_food_progress": 0.0,
            "predator_visible_ticks": 0,
            "alive": True,
            "food_eaten": 0,
        }
        cases = {
            "success": {**base_metrics, "checks_passed": True},
            "cautious_inert": {**base_metrics, "left_shelter": False},
            "foraging_and_died": {
                **base_metrics,
                "food_distance_delta": 2.0,
                "alive": False,
            },
            "threatened_retreat": {
                **base_metrics,
                "predator_visible_ticks": 2,
            },
            "partial_progress": {
                **base_metrics,
                "peak_food_progress": 1.0,
            },
            "stall": base_metrics,
            "scoring_mismatch": {**base_metrics, "alive": False},
        }
        for expected, metrics in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(
                    _classify_exposed_day_foraging_failure(metrics),
                    expected,
                )


class ExposedDayForagingScorerTest(unittest.TestCase):
    def test_trace_metric_extractor_reads_trace_shape(self) -> None:
        shelter_pos, outside_pos = _exposed_day_trace_positions()
        trace = [
            _make_exposed_day_trace_item(
                tick=0,
                pos=shelter_pos,
                health=1.0,
                food_dist=8,
                predator_visible=False,
            ),
            _make_exposed_day_trace_item(
                tick=1,
                pos=outside_pos,
                health=1.0,
                food_dist=7,
                predator_visible=True,
            ),
            _make_exposed_day_trace_item(
                tick=2,
                pos=outside_pos,
                health=0.0,
                food_dist=5,
                predator_visible=True,
            ),
        ]

        metrics = _extract_exposed_day_trace_metrics(trace)

        self.assertTrue(metrics["left_shelter"])
        self.assertEqual(metrics["shelter_exit_tick"], 1)
        self.assertEqual(metrics["peak_food_progress"], 3.0)
        self.assertEqual(metrics["predator_visible_ticks"], 2)
        self.assertEqual(metrics["final_distance_to_food"], 5.0)

    def test_score_failure_modes_from_trace_fixtures(self) -> None:
        spec = get_scenario("exposed_day_foraging")
        shelter_pos, outside_pos = _exposed_day_trace_positions()
        cases = {
            "cautious_inert": {
                "stats": {
                    "food_distance_delta": 0.0,
                    "food_eaten": 0,
                    "alive": True,
                    "predator_contacts": 0,
                },
                "trace": [
                    _make_exposed_day_trace_item(
                        tick=0,
                        pos=shelter_pos,
                        health=1.0,
                        food_dist=8,
                    ),
                    _make_exposed_day_trace_item(
                        tick=1,
                        pos=shelter_pos,
                        health=1.0,
                        food_dist=8,
                    ),
                ],
            },
            "foraging_and_died": {
                "stats": {
                    "food_distance_delta": 2.0,
                    "food_eaten": 0,
                    "alive": False,
                    "predator_contacts": 0,
                },
                "trace": [
                    _make_exposed_day_trace_item(
                        tick=0,
                        pos=shelter_pos,
                        health=1.0,
                        food_dist=8,
                        predator_visible=False,
                    ),
                    _make_exposed_day_trace_item(
                        tick=1,
                        pos=outside_pos,
                        health=1.0,
                        food_dist=7,
                        predator_visible=True,
                    ),
                    _make_exposed_day_trace_item(
                        tick=2,
                        pos=outside_pos,
                        health=0.0,
                        food_dist=5,
                        predator_visible=True,
                    ),
                ],
            },
            "threatened_retreat": {
                "stats": {
                    "food_distance_delta": 0.0,
                    "food_eaten": 0,
                    "alive": True,
                    "predator_contacts": 0,
                },
                "trace": [
                    _make_exposed_day_trace_item(
                        tick=0,
                        pos=shelter_pos,
                        health=1.0,
                        food_dist=8,
                    ),
                    _make_exposed_day_trace_item(
                        tick=1,
                        pos=outside_pos,
                        health=1.0,
                        food_dist=8,
                        predator_visible=True,
                    ),
                    _make_exposed_day_trace_item(
                        tick=2,
                        pos=shelter_pos,
                        health=1.0,
                        food_dist=8,
                    ),
                ],
            },
            "partial_progress": {
                "stats": {
                    "food_distance_delta": 0.0,
                    "food_eaten": 0,
                    "alive": True,
                    "predator_contacts": 0,
                },
                "trace": [
                    _make_exposed_day_trace_item(
                        tick=0,
                        pos=shelter_pos,
                        health=1.0,
                        food_dist=8,
                    ),
                    _make_exposed_day_trace_item(
                        tick=1,
                        pos=outside_pos,
                        health=1.0,
                        food_dist=7,
                    ),
                    _make_exposed_day_trace_item(
                        tick=2,
                        pos=outside_pos,
                        health=1.0,
                        food_dist=7,
                    ),
                ],
            },
            "stall": {
                "stats": {
                    "food_distance_delta": 0.0,
                    "food_eaten": 0,
                    "alive": True,
                    "predator_contacts": 0,
                },
                "trace": [
                    _make_exposed_day_trace_item(
                        tick=0,
                        pos=shelter_pos,
                        health=1.0,
                        food_dist=8,
                    ),
                    _make_exposed_day_trace_item(
                        tick=1,
                        pos=outside_pos,
                        health=1.0,
                        food_dist=8,
                    ),
                    _make_exposed_day_trace_item(
                        tick=2,
                        pos=outside_pos,
                        health=1.0,
                        food_dist=8,
                    ),
                ],
            },
            "success": {
                "stats": {
                    "food_distance_delta": 2.0,
                    "food_eaten": 0,
                    "alive": True,
                    "predator_contacts": 0,
                },
                "trace": [
                    _make_exposed_day_trace_item(
                        tick=0,
                        pos=shelter_pos,
                        health=1.0,
                        food_dist=8,
                    ),
                    _make_exposed_day_trace_item(
                        tick=1,
                        pos=outside_pos,
                        health=1.0,
                        food_dist=7,
                    ),
                    _make_exposed_day_trace_item(
                        tick=2,
                        pos=outside_pos,
                        health=1.0,
                        food_dist=6,
                    ),
                ],
            },
        }
        for expected, fixture in cases.items():
            with self.subTest(expected=expected):
                stats = _make_episode_stats(
                    scenario="exposed_day_foraging",
                    **fixture["stats"],
                )

                score = spec.score_episode(stats, fixture["trace"])

                self.assertEqual(score.behavior_metrics["failure_mode"], expected)

    def test_emits_trace_diagnostic_metrics(self) -> None:
        """
        Verify that the exposed_day_foraging scenario produces expected diagnostic behavior metrics for a foraging-then-died episode.
        
        Asserts that the resulting BehavioralEpisodeScore includes diagnostic keys (failure_mode, left_shelter, shelter_exit_tick, peak_food_progress, predator_visible_ticks, progress_band, outcome_band) and that their values match the provided episode statistics and trace: agent left shelter at tick 1, peak food progress equals 3.0, two predator-visible ticks occurred, and the failure_mode is "foraging_and_died".
        """
        spec = get_scenario("exposed_day_foraging")
        stats = _make_episode_stats(
            scenario="exposed_day_foraging",
            food_distance_delta=2.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        shelter_pos, outside_pos = _exposed_day_trace_positions()
        trace = [
            _make_exposed_day_trace_item(
                tick=0,
                pos=shelter_pos,
                health=1.0,
                food_dist=8,
                predator_visible=False,
            ),
            _make_exposed_day_trace_item(
                tick=1,
                pos=outside_pos,
                health=1.0,
                food_dist=7,
                predator_visible=True,
            ),
            _make_exposed_day_trace_item(
                tick=2,
                pos=outside_pos,
                health=0.0,
                food_dist=5,
                predator_visible=True,
            ),
        ]

        score = spec.score_episode(stats, trace)

        self.assertTrue(
            {
                "failure_mode",
                "left_shelter",
                "shelter_exit_tick",
                "peak_food_progress",
                "predator_visible_ticks",
                "progress_band",
                "outcome_band",
            }.issubset(score.behavior_metrics)
        )
        self.assertTrue(score.behavior_metrics["left_shelter"])
        self.assertEqual(score.behavior_metrics["shelter_exit_tick"], 1)
        self.assertEqual(score.behavior_metrics["peak_food_progress"], 3.0)
        self.assertEqual(score.behavior_metrics["predator_visible_ticks"], 2)
        self.assertEqual(score.behavior_metrics["failure_mode"], "foraging_and_died")


# ---------------------------------------------------------------------------
# _predator_visible_flag unit tests
# ---------------------------------------------------------------------------

class PredatorVisibleFlagTest(unittest.TestCase):
    """Unit tests for _predator_visible_flag type-coercion logic."""

    def test_bool_true_returns_true(self) -> None:
        self.assertIs(_predator_visible_flag(True), True)

    def test_bool_false_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag(False), False)

    def test_numeric_at_threshold_returns_true(self) -> None:
        self.assertIs(_predator_visible_flag(1.0), True)
        self.assertIs(_predator_visible_flag(0.5), True)

    def test_numeric_below_threshold_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag(0.49), False)
        self.assertIs(_predator_visible_flag(0.1), False)

    def test_zero_numeric_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag(0.0), False)
        self.assertIs(_predator_visible_flag(0), False)

    def test_negative_numeric_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag(-1.0), False)

    def test_string_true_tokens_return_true(self) -> None:
        for token in ("true", "yes", "visible"):
            with self.subTest(token=token):
                self.assertIs(_predator_visible_flag(token), True)

    def test_string_true_tokens_case_insensitive(self) -> None:
        for token in ("TRUE", "YES", "VISIBLE", "True", "Yes", "Visible"):
            with self.subTest(token=token):
                self.assertIs(_predator_visible_flag(token), True)

    def test_string_false_tokens_return_false(self) -> None:
        for token in ("false", "no", "hidden", "none"):
            with self.subTest(token=token):
                self.assertIs(_predator_visible_flag(token), False)

    def test_string_false_tokens_case_insensitive(self) -> None:
        for token in ("FALSE", "NO", "HIDDEN", "NONE", "False", "No", "Hidden", "None"):
            with self.subTest(token=token):
                self.assertIs(_predator_visible_flag(token), False)

    def test_string_with_whitespace_stripped(self) -> None:
        self.assertIs(_predator_visible_flag("  true  "), True)
        self.assertIs(_predator_visible_flag("  false  "), False)

    def test_unknown_string_returns_none(self) -> None:
        self.assertIsNone(_predator_visible_flag("unknown"))
        self.assertIsNone(_predator_visible_flag("maybe"))
        self.assertIsNone(_predator_visible_flag(""))

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_predator_visible_flag(None))

    def test_list_returns_none(self) -> None:
        self.assertIsNone(_predator_visible_flag([]))

    def test_dict_returns_none(self) -> None:
        self.assertIsNone(_predator_visible_flag({}))

    def test_numeric_string_zero_returns_false(self) -> None:
        # "0" converts to float 0.0 via _float_or_none and returns False.
        self.assertIs(_predator_visible_flag("0"), False)

    def test_numeric_string_one_returns_true(self) -> None:
        # "1" converts to float 1.0 via _float_or_none and returns True.
        self.assertIs(_predator_visible_flag("1"), True)

    def test_numeric_string_below_threshold_returns_false(self) -> None:
        self.assertIs(_predator_visible_flag("0.49"), False)


# ---------------------------------------------------------------------------
# _mapping_predator_visible unit tests
# ---------------------------------------------------------------------------

class MappingPredatorVisibleTest(unittest.TestCase):
    """Unit tests for _mapping_predator_visible nested mapping inspection."""

    def test_empty_mapping_returns_false(self) -> None:
        self.assertFalse(_mapping_predator_visible({}))

    def test_predator_visible_true(self) -> None:
        self.assertTrue(_mapping_predator_visible({"predator_visible": True}))

    def test_predator_visible_false_returns_false(self) -> None:
        self.assertFalse(_mapping_predator_visible({"predator_visible": False}))

    def test_prev_predator_visible_true(self) -> None:
        self.assertTrue(_mapping_predator_visible({"prev_predator_visible": True}))

    def test_prev_predator_visible_false_returns_false(self) -> None:
        self.assertFalse(_mapping_predator_visible({"prev_predator_visible": False}))

    def test_any_visible_flag_returns_true(self) -> None:
        self.assertTrue(
            _mapping_predator_visible({"predator_visible": False, "prev_predator_visible": True})
        )

    def test_meta_nested_predator_visible(self) -> None:
        source = {"meta": {"predator_visible": True}}
        self.assertTrue(_mapping_predator_visible(source))

    def test_meta_non_mapping_is_ignored(self) -> None:
        self.assertFalse(_mapping_predator_visible({"meta": "string_meta"}))

    def test_vision_predator_visible_key(self) -> None:
        source = {"vision": {"predator": {"visible": True}}}
        self.assertTrue(_mapping_predator_visible(source))

    def test_vision_predator_certainty_key(self) -> None:
        for certainty in (0.5, 1.0):
            with self.subTest(certainty=certainty):
                source = {"vision": {"predator": {"certainty": certainty}}}
                self.assertTrue(_mapping_predator_visible(source))

    def test_vision_predator_certainty_zero_is_false(self) -> None:
        for certainty in (0.0, 0.49):
            with self.subTest(certainty=certainty):
                source = {"vision": {"predator": {"certainty": certainty}}}
                self.assertFalse(_mapping_predator_visible(source))

    def test_meta_predator_visible_bool_is_honored(self) -> None:
        self.assertTrue(_mapping_predator_visible({"meta": {"predator_visible": True}}))

    def test_vision_non_mapping_predator_ignored(self) -> None:
        source = {"vision": {"predator": "patrol"}}
        self.assertFalse(_mapping_predator_visible(source))

    def test_vision_non_mapping_ignored(self) -> None:
        source = {"vision": "something"}
        self.assertFalse(_mapping_predator_visible(source))

    def test_evidence_threat_predator_visible(self) -> None:
        source = {"evidence": {"threat": {"predator_visible": 1.0}}}
        self.assertTrue(_mapping_predator_visible(source))

    def test_evidence_threat_predator_not_visible(self) -> None:
        source = {"evidence": {"threat": {"predator_visible": 0.0}}}
        self.assertFalse(_mapping_predator_visible(source))

    def test_evidence_non_mapping_ignored(self) -> None:
        source = {"evidence": "some_string"}
        self.assertFalse(_mapping_predator_visible(source))

    def test_deeply_nested_via_evidence_threat(self) -> None:
        # evidence.threat.meta.predator_visible should also resolve
        source = {"evidence": {"threat": {"meta": {"predator_visible": True}}}}
        self.assertTrue(_mapping_predator_visible(source))

    def test_unrecognized_keys_only_returns_false(self) -> None:
        source = {"food_visible": True, "hunger": 0.9}
        self.assertFalse(_mapping_predator_visible(source))


# ---------------------------------------------------------------------------
# _trace_predator_visible unit tests
# ---------------------------------------------------------------------------

class TracePredatorVisibleTest(unittest.TestCase):
    """Unit tests for _trace_predator_visible trace-item inspection."""

    def test_empty_item_returns_false(self) -> None:
        self.assertFalse(_trace_predator_visible({}))

    def test_direct_predator_visible_true(self) -> None:
        self.assertTrue(_trace_predator_visible({"predator_visible": True}))

    def test_state_with_predator_visible(self) -> None:
        item = {"state": {"predator_visible": True}}
        self.assertTrue(_trace_predator_visible(item))

    def test_observation_with_predator_visible(self) -> None:
        item = {"observation": {"predator_visible": True}}
        self.assertTrue(_trace_predator_visible(item))

    def test_next_observation_with_predator_visible(self) -> None:
        item = {"next_observation": {"predator_visible": True}}
        self.assertTrue(_trace_predator_visible(item))

    def test_debug_with_predator_visible(self) -> None:
        item = {"debug": {"predator_visible": True}}
        self.assertTrue(_trace_predator_visible(item))

    def test_snapshot_with_predator_visible(self) -> None:
        item = {
            "event_log": [
                {
                    "stage": "pre_tick",
                    "name": "snapshot",
                    "payload": {"meta": {"predator_visible": True}},
                }
            ]
        }
        self.assertTrue(_trace_predator_visible(item))

    def test_snapshot_non_mapping_payload_ignored(self) -> None:
        item = {
            "event_log": [
                {
                    "stage": "pre_tick",
                    "name": "snapshot",
                    "payload": "not_a_mapping",
                }
            ]
        }
        self.assertFalse(_trace_predator_visible(item))

    def test_messages_payload_with_predator_visible(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"predator_visible": True},
                }
            ]
        }
        self.assertTrue(_trace_predator_visible(item))

    def test_messages_payload_meta_nested(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"predator_visible": True}},
                }
            ]
        }
        self.assertTrue(_trace_predator_visible(item))

    def test_non_mapping_message_ignored(self) -> None:
        item = {"messages": ["not_a_mapping"]}
        self.assertFalse(_trace_predator_visible(item))

    def test_message_without_payload_ignored(self) -> None:
        item = {"messages": [{"sender": "environment", "topic": "observation"}]}
        self.assertFalse(_trace_predator_visible(item))

    def test_evidence_threat_in_message_payload(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "action_center",
                    "topic": "action.selection",
                    "payload": {
                        "evidence": {"threat": {"predator_visible": 1.0}},
                    },
                }
            ]
        }
        self.assertTrue(_trace_predator_visible(item))

    def test_all_channels_absent_returns_false(self) -> None:
        item = {
            "state": {"x": 5, "y": 3, "health": 1.0},
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 6}},
                }
            ],
        }
        self.assertFalse(_trace_predator_visible(item))

    def test_state_non_mapping_ignored(self) -> None:
        item = {"state": "not_a_mapping"}
        self.assertFalse(_trace_predator_visible(item))


# ---------------------------------------------------------------------------
# Additional ExposedDayForagingClassifier edge cases
# ---------------------------------------------------------------------------

class ExposedDayForagingClassifierEdgeCaseTest(unittest.TestCase):
    """Edge-case and boundary tests for _classify_exposed_day_foraging_failure."""

    def test_food_eaten_triggers_partial_progress_when_alive(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "food_eaten": 1,
            "food_distance_delta": 0.0,
            "peak_food_progress": 0.0,
            "predator_visible_ticks": 0,
            "alive": True,
        }
        self.assertEqual(_classify_exposed_day_foraging_failure(metrics), "partial_progress")

    def test_food_eaten_and_dead_gives_foraging_and_died(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "food_eaten": 2,
            "food_distance_delta": 0.0,
            "peak_food_progress": 0.0,
            "predator_visible_ticks": 0,
            "alive": False,
        }
        self.assertEqual(_classify_exposed_day_foraging_failure(metrics), "foraging_and_died")

    def test_predator_visible_with_food_progress_gives_partial_progress(self) -> None:
        # progress wins over retreat when made_food_progress is True
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "food_eaten": 0,
            "food_distance_delta": 1.5,
            "peak_food_progress": 0.0,
            "predator_visible_ticks": 3,
            "alive": True,
        }
        self.assertEqual(_classify_exposed_day_foraging_failure(metrics), "partial_progress")

    def test_none_values_treated_as_absent(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "food_eaten": None,
            "food_distance_delta": None,
            "peak_food_progress": None,
            "predator_visible_ticks": None,
            "alive": True,
        }
        # All None coerces to zero/False: no progress, no predator, survived.
        self.assertEqual(_classify_exposed_day_foraging_failure(metrics), "stall")

    def test_missing_keys_treated_as_absent(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "alive": True,
        }
        self.assertEqual(_classify_exposed_day_foraging_failure(metrics), "stall")

    def test_peak_food_progress_triggers_partial_progress(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "food_eaten": 0,
            "food_distance_delta": 0.0,
            "peak_food_progress": 0.5,
            "predator_visible_ticks": 0,
            "alive": True,
        }
        self.assertEqual(_classify_exposed_day_foraging_failure(metrics), "partial_progress")

    def test_checks_passed_true_overrides_all(self) -> None:
        metrics = {
            "checks_passed": True,
            "left_shelter": False,
            "food_eaten": 0,
            "food_distance_delta": 0.0,
            "peak_food_progress": 0.0,
            "predator_visible_ticks": 5,
            "alive": False,
        }
        self.assertEqual(_classify_exposed_day_foraging_failure(metrics), "success")


# ---------------------------------------------------------------------------
# Additional ExposedDayForagingScorer edge cases
# ---------------------------------------------------------------------------

class ExposedDayForagingScorerEdgeCaseTest(unittest.TestCase):
    """Additional edge-case and boundary tests for the exposed_day_foraging scorer."""

    def test_empty_trace_produces_defaults(self) -> None:
        spec = get_scenario("exposed_day_foraging")
        stats = _make_episode_stats(
            scenario="exposed_day_foraging",
            food_distance_delta=0.0,
            food_eaten=0,
            alive=True,
            predator_contacts=0,
        )
        score = spec.score_episode(stats, [])
        self.assertFalse(score.behavior_metrics["left_shelter"])
        self.assertIsNone(score.behavior_metrics["shelter_exit_tick"])
        self.assertEqual(score.behavior_metrics["peak_food_progress"], 0.0)
        self.assertEqual(score.behavior_metrics["predator_visible_ticks"], 0)
        self.assertIsNone(score.behavior_metrics["final_distance_to_food"])

    def test_all_shelter_ticks_left_shelter_false(self) -> None:
        shelter_pos, _ = _exposed_day_trace_positions()
        trace = [
            _make_exposed_day_trace_item(
                tick=t,
                pos=shelter_pos,
                health=1.0,
                food_dist=8,
                predator_visible=False,
            )
            for t in range(3)
        ]
        spec = get_scenario("exposed_day_foraging")
        stats = _make_episode_stats(
            scenario="exposed_day_foraging",
            food_distance_delta=0.0,
            food_eaten=0,
            alive=True,
            predator_contacts=0,
        )
        score = spec.score_episode(stats, trace)
        self.assertFalse(score.behavior_metrics["left_shelter"])
        self.assertEqual(score.behavior_metrics["failure_mode"], "cautious_inert")

    def test_predator_visible_ticks_counted_from_trace(self) -> None:
        shelter_pos, outside_pos = _exposed_day_trace_positions()
        trace = [
            _make_exposed_day_trace_item(tick=0, pos=shelter_pos, health=1.0, food_dist=8, predator_visible=False),
            _make_exposed_day_trace_item(tick=1, pos=outside_pos, health=1.0, food_dist=7, predator_visible=True),
            _make_exposed_day_trace_item(tick=2, pos=outside_pos, health=1.0, food_dist=7, predator_visible=False),
            _make_exposed_day_trace_item(tick=3, pos=outside_pos, health=1.0, food_dist=7, predator_visible=True),
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["predator_visible_ticks"], 2)

    def test_peak_food_progress_uses_initial_distance(self) -> None:
        shelter_pos, outside_pos = _exposed_day_trace_positions()
        trace = [
            _make_exposed_day_trace_item(tick=0, pos=shelter_pos, health=1.0, food_dist=10),
            _make_exposed_day_trace_item(tick=1, pos=outside_pos, health=1.0, food_dist=6),
            _make_exposed_day_trace_item(tick=2, pos=outside_pos, health=1.0, food_dist=8),
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        # Peak reduction from 10 → 6 = 4.0
        self.assertEqual(metrics["peak_food_progress"], 4.0)

    def test_behavior_metrics_contains_food_eaten_and_predator_contacts(self) -> None:
        spec = get_scenario("exposed_day_foraging")
        stats = _make_episode_stats(
            scenario="exposed_day_foraging",
            food_distance_delta=1.0,
            food_eaten=1,
            alive=True,
            predator_contacts=2,
        )
        score = spec.score_episode(stats, [])
        self.assertEqual(score.behavior_metrics["food_eaten"], 1)
        self.assertEqual(score.behavior_metrics["predator_contacts"], 2)

    def test_score_with_food_eaten_gives_partial_progress_failure_mode(self) -> None:
        # food_eaten=1 and alive=True indicate progress; predator_contacts=1 fails the
        # "no predator contacts" check so checks_passed=False → partial_progress
        shelter_pos, outside_pos = _exposed_day_trace_positions()
        trace = [
            _make_exposed_day_trace_item(tick=0, pos=shelter_pos, health=1.0, food_dist=8),
            _make_exposed_day_trace_item(tick=1, pos=outside_pos, health=1.0, food_dist=4),
        ]
        spec = get_scenario("exposed_day_foraging")
        stats = _make_episode_stats(
            scenario="exposed_day_foraging",
            food_distance_delta=4.0,
            food_eaten=1,
            alive=True,
            predator_contacts=1,  # fails the "zero contacts" check → not success
        )
        score = spec.score_episode(stats, trace)
        self.assertEqual(score.behavior_metrics["failure_mode"], "partial_progress")


# ---------------------------------------------------------------------------
# food_deprivation scenario setup (_food_deprivation)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# simulation.py: save_behavior_csv
# ---------------------------------------------------------------------------

class SaveBehaviorCsvTest(unittest.TestCase):
    """Tests for SpiderSimulation.save_behavior_csv."""

    def setUp(self):
        from spider_cortex_sim.simulation import SpiderSimulation
        self.SpiderSimulation = SpiderSimulation

    def _make_rows(self) -> list[dict[str, Any]]:
        """
        Constructs example CSV row dictionaries used by the behavior-evaluation test suite.
        
        Each dictionary models a single flattened CSV row including simulation and curriculum metadata, competence/reflex/budget/checkpoint/noise fields, scenario identifiers and descriptive fields, outcome fields (`success`, `failure_count`, `failures`), metric columns prefixed with `metric_...`, and per-check columns named `check_<name>_passed`, `check_<name>_value`, and `check_<name>_expected`.
        
        Returns:
            list[dict[str, Any]]: A list of example row dictionaries conforming to the CSV/flattened behavior-row schema used in tests.
        """
        return [
            {
                "reward_profile": "classic",
                "scenario_map": "night_rest_map",
                "evaluation_map": "central_burrow",
                "training_regime": "curriculum",
                "curriculum_profile": "ecological_v1",
                "curriculum_phase": "phase_4_corridor_food_deprivation",
                "curriculum_skill": "corridor_navigation",
                "curriculum_phase_status": "max_budget_exhausted",
                "curriculum_promotion_reason": "threshold_fallback",
                "learning_evidence_condition": "trained_final",
                "learning_evidence_policy_mode": "normal",
                "learning_evidence_training_regime": "baseline",
                "learning_evidence_train_episodes": 6,
                "learning_evidence_frozen_after_episode": "",
                "learning_evidence_checkpoint_source": "final",
                "learning_evidence_budget_profile": "smoke",
                "learning_evidence_budget_benchmark_strength": "quick",
                "reflex_scale": 1.0,
                "reflex_anneal_final_scale": 0.5,
                "competence_type": "self_sufficient",
                "is_primary_benchmark": True,
                "eval_reflex_scale": 0.0,
                "budget_profile": "dev",
                "benchmark_strength": "quick",
                "operational_profile": "default_v1",
                "operational_profile_version": 1,
                "train_noise_profile": "low",
                "train_noise_profile_config": LOW_NOISE_PROFILE_CONFIG_JSON,
                "eval_noise_profile": "none",
                "eval_noise_profile_config": NOISE_PROFILE_CONFIG_JSON,
                "noise_profile": "none",
                "noise_profile_config": NOISE_PROFILE_CONFIG_JSON,
                "checkpoint_source": "best",
                "simulation_seed": 7,
                "episode_seed": 100000,
                "scenario": "night_rest",
                "scenario_description": "night desc",
                "scenario_objective": "night obj",
                "scenario_focus": "night focus",
                "episode": 0,
                "success": True,
                "failure_count": 0,
                "failures": "",
                "metric_food_eaten": 1,
                "check_deep_night_shelter_passed": True,
                "check_deep_night_shelter_value": 0.99,
                "check_deep_night_shelter_expected": ">= 0.95",
            }
        ]

    def test_creates_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.SpiderSimulation.save_behavior_csv(self._make_rows(), path)
            self.assertTrue(path.exists())

    def test_csv_has_header_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.SpiderSimulation.save_behavior_csv(self._make_rows(), path)
            content = path.read_text(encoding="utf-8")
            lines = content.strip().splitlines()
            self.assertGreater(len(lines), 1)

    def test_preferred_columns_appear_in_header(self) -> None:
        """
        Verify the CSV header produced by save_behavior_csv includes all preferred behavior CSV columns.
        
        Checks presence of columns for reward/profile/map metadata, curriculum and learning-evidence fields, reflex/budget/checkpoint settings, competence and reflex evaluation fields, noise-profile configuration, scenario metadata, and outcome columns such as episode, success, failure_count, and failures.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.SpiderSimulation.save_behavior_csv(self._make_rows(), path)
            header_cols = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            for col in [
                "reward_profile",
                "scenario_map",
                "evaluation_map",
                *CURRICULUM_COLUMNS,
                "learning_evidence_condition",
                "learning_evidence_policy_mode",
                "learning_evidence_training_regime",
                "learning_evidence_train_episodes",
                "learning_evidence_frozen_after_episode",
                "learning_evidence_checkpoint_source",
                "learning_evidence_budget_profile",
                "learning_evidence_budget_benchmark_strength",
                "reflex_scale",
                "reflex_anneal_final_scale",
                "competence_type",
                "is_primary_benchmark",
                "eval_reflex_scale",
                "budget_profile",
                "benchmark_strength",
                "operational_profile",
                "operational_profile_version",
                "train_noise_profile",
                "train_noise_profile_config",
                "eval_noise_profile",
                "eval_noise_profile_config",
                "noise_profile",
                "noise_profile_config",
                "checkpoint_source",
                "simulation_seed",
                "episode_seed",
                "scenario",
                "scenario_description",
                "scenario_objective",
                "scenario_focus",
                "episode",
                "success",
                "failure_count",
                "failures",
            ]:
                self.assertIn(
                    col,
                    header_cols,
                    msg=f"Column {col!r} missing from CSV header",
                )

    def test_curriculum_columns_appear_in_preferred_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.SpiderSimulation.save_behavior_csv(self._make_rows(), path)
            header_cols = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            curriculum_header = [
                column for column in header_cols if column in CURRICULUM_COLUMNS
            ]
            self.assertEqual(curriculum_header, list(CURRICULUM_COLUMNS))

    def test_data_rows_count(self) -> None:
        rows = self._make_rows() * 3
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.SpiderSimulation.save_behavior_csv(rows, path)
            content = path.read_text(encoding="utf-8")
            lines = [line for line in content.strip().splitlines() if line.strip()]
            self.assertEqual(len(lines), 4)  # 1 header + 3 data rows

    def test_extra_columns_sorted_after_preferred(self) -> None:
        rows = [{
            "reward_profile": "classic",
            "scenario_map": "night_rest_map",
            "evaluation_map": "central_burrow",
            "simulation_seed": 7,
            "episode_seed": 100000,
            "scenario": "s",
            "episode": 0,
            "success": True,
            "failure_count": 0,
            "failures": "",
            "zzz_metric": 1.0,
            "aaa_metric": 2.0,
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.SpiderSimulation.save_behavior_csv(rows, path)
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            # 'failures' is last preferred column, extra columns come after
            failures_idx = header.index("failures")
            aaa_idx = header.index("aaa_metric")
            zzz_idx = header.index("zzz_metric")
            self.assertGreater(aaa_idx, failures_idx)
            self.assertLess(aaa_idx, zzz_idx)

    def test_empty_rows_creates_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.SpiderSimulation.save_behavior_csv([], path)
            content = path.read_text(encoding="utf-8")
            lines = [line for line in content.strip().splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)  # header only

    def test_csv_is_readable_by_csv_reader(self) -> None:
        """
        Verify the CSV produced by SpiderSimulation.save_behavior_csv is readable by csv.DictReader and that key fields serialize to the expected string values.
        
        Writes a single-row CSV to a temporary file, reads it back with csv.DictReader, asserts exactly one data row is present, and checks that metadata (including reward/profile/map fields, curriculum and learning-evidence columns, reflex/budget/checkpoint settings, seed fields, scenario metadata, outcome fields, and noise profile/config) are serialized exactly as expected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            rows = self._make_rows()
            self.SpiderSimulation.save_behavior_csv(rows, path)
            with open(path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                data_rows = list(reader)
            self.assertEqual(len(data_rows), 1)
            expected = {
                "reward_profile": "classic",
                "scenario_map": "night_rest_map",
                "evaluation_map": "central_burrow",
                "training_regime": "curriculum",
                "curriculum_profile": "ecological_v1",
                "curriculum_phase": "phase_4_corridor_food_deprivation",
                "curriculum_skill": "corridor_navigation",
                "curriculum_phase_status": "max_budget_exhausted",
                "curriculum_promotion_reason": "threshold_fallback",
                "learning_evidence_condition": "trained_final",
                "learning_evidence_policy_mode": "normal",
                "learning_evidence_training_regime": "baseline",
                "learning_evidence_train_episodes": "6",
                "learning_evidence_frozen_after_episode": "",
                "learning_evidence_checkpoint_source": "final",
                "learning_evidence_budget_profile": "smoke",
                "learning_evidence_budget_benchmark_strength": "quick",
                "reflex_scale": "1.0",
                "reflex_anneal_final_scale": "0.5",
                "competence_type": "self_sufficient",
                "is_primary_benchmark": "True",
                "eval_reflex_scale": "0.0",
                "budget_profile": "dev",
                "benchmark_strength": "quick",
                "operational_profile": "default_v1",
                "operational_profile_version": "1",
                "train_noise_profile": "low",
                "train_noise_profile_config": LOW_NOISE_PROFILE_CONFIG_JSON,
                "eval_noise_profile": "none",
                "eval_noise_profile_config": NOISE_PROFILE_CONFIG_JSON,
                "noise_profile": "none",
                "noise_profile_config": NOISE_PROFILE_CONFIG_JSON,
                "checkpoint_source": "best",
                "simulation_seed": "7",
                "episode_seed": "100000",
                "scenario": "night_rest",
                "scenario_description": "night desc",
                "scenario_objective": "night obj",
                "scenario_focus": "night focus",
                "episode": "0",
                "success": "True",
                "failure_count": "0",
                "failures": "",
            }
            for field, value in expected.items():
                self.assertEqual(data_rows[0].get(field), value)


# ---------------------------------------------------------------------------
# simulation.py: _compact_behavior_payload
# ---------------------------------------------------------------------------

class CompactBehaviorPayloadTest(unittest.TestCase):
    """Tests for SpiderSimulation._compact_behavior_payload."""

    def setUp(self):
        from spider_cortex_sim.simulation import SpiderSimulation
        self.SpiderSimulation = SpiderSimulation

    def _make_payload(self):
        return {
            "summary": {"scenario_count": 1, "episode_count": 2, "scenario_success_rate": 0.5, "episode_success_rate": 0.5, "regressions": []},
            "suite": {
                "night_rest": {
                    "scenario": "night_rest",
                    "description": "desc",
                    "objective": "obj",
                    "episodes": 2,
                    "success_rate": 1.0,
                    "checks": {},
                    "behavior_metrics": {},
                    "failures": [],
                    "episodes_detail": [{"episode": 0}, {"episode": 1}],
                    "legacy_metrics": {"mean_reward": 1.0, "episodes_detail": [{"episode": 0}]},
                }
            },
            "legacy_scenarios": {
                "night_rest": {"mean_reward": 1.0, "episodes_detail": [{"episode": 0}]}
            },
        }

    def test_episodes_detail_removed_from_suite(self) -> None:
        payload = self._make_payload()
        compact = self.SpiderSimulation._compact_behavior_payload(payload)
        self.assertNotIn("episodes_detail", compact["suite"]["night_rest"])

    def test_legacy_metrics_episodes_detail_removed(self) -> None:
        payload = self._make_payload()
        compact = self.SpiderSimulation._compact_behavior_payload(payload)
        self.assertNotIn("episodes_detail", compact["suite"]["night_rest"]["legacy_metrics"])

    def test_legacy_scenarios_episodes_detail_removed(self) -> None:
        payload = self._make_payload()
        compact = self.SpiderSimulation._compact_behavior_payload(payload)
        self.assertNotIn("episodes_detail", compact["legacy_scenarios"]["night_rest"])

    def test_summary_preserved(self) -> None:
        payload = self._make_payload()
        compact = self.SpiderSimulation._compact_behavior_payload(payload)
        self.assertEqual(compact["summary"]["scenario_count"], 1)
        self.assertEqual(compact["summary"]["episode_count"], 2)

    def test_suite_fields_preserved(self) -> None:
        payload = self._make_payload()
        compact = self.SpiderSimulation._compact_behavior_payload(payload)
        suite_item = compact["suite"]["night_rest"]
        self.assertEqual(suite_item["success_rate"], 1.0)
        self.assertEqual(suite_item["episodes"], 2)

    def test_empty_payload_does_not_raise(self) -> None:
        compact = self.SpiderSimulation._compact_behavior_payload({})
        self.assertIn("summary", compact)
        self.assertIn("suite", compact)
        self.assertIn("legacy_scenarios", compact)


# ---------------------------------------------------------------------------
# simulation.py: trace new fields
# ---------------------------------------------------------------------------

class TraceNewFieldsTest(unittest.TestCase):
    """Tests that run_episode trace items include new fields added in this PR."""

    def setUp(self):
        from spider_cortex_sim.simulation import SpiderSimulation
        self.sim = SpiderSimulation(seed=7, max_steps=3)

    def test_trace_items_include_seed(self) -> None:
        _, trace = self.sim.run_episode(0, training=False, sample=False, capture_trace=True)
        self.assertTrue(trace)
        for item in trace:
            self.assertIn("seed", item)

    def test_trace_items_include_action(self) -> None:
        _, trace = self.sim.run_episode(0, training=False, sample=False, capture_trace=True)
        self.assertTrue(trace)
        for item in trace:
            self.assertIn("action", item)

    def test_trace_items_include_ate(self) -> None:
        _, trace = self.sim.run_episode(0, training=False, sample=False, capture_trace=True)
        self.assertTrue(trace)
        for item in trace:
            self.assertIn("ate", item)
            self.assertIsInstance(item["ate"], bool)

    def test_trace_items_include_slept(self) -> None:
        _, trace = self.sim.run_episode(0, training=False, sample=False, capture_trace=True)
        self.assertTrue(trace)
        for item in trace:
            self.assertIn("slept", item)
            self.assertIsInstance(item["slept"], bool)

    def test_trace_items_include_predator_contact(self) -> None:
        _, trace = self.sim.run_episode(0, training=False, sample=False, capture_trace=True)
        self.assertTrue(trace)
        for item in trace:
            self.assertIn("predator_contact", item)
            self.assertIsInstance(item["predator_contact"], bool)

    def test_trace_items_include_predator_escape(self) -> None:
        _, trace = self.sim.run_episode(0, training=False, sample=False, capture_trace=True)
        self.assertTrue(trace)
        for item in trace:
            self.assertIn("predator_escape", item)
            self.assertIsInstance(item["predator_escape"], bool)

    def test_trace_seed_matches_episode_index(self) -> None:
        # Seed in trace should be deterministic for same episode index
        _, trace1 = self.sim.run_episode(42, training=False, sample=False, capture_trace=True)
        seed1 = trace1[0]["seed"] if trace1 else None
        sim2 = type(self.sim)(seed=7, max_steps=3)
        _, trace2 = sim2.run_episode(42, training=False, sample=False, capture_trace=True)
        seed2 = trace2[0]["seed"] if trace2 else None
        self.assertEqual(seed1, seed2)

    def test_stats_include_seed_field(self) -> None:
        stats, _ = self.sim.run_episode(0, training=False, sample=False, capture_trace=False)
        self.assertTrue(hasattr(stats, "seed"))
        self.assertIsInstance(stats.seed, int)


# ---------------------------------------------------------------------------
# cli.py: build_parser new behavior arguments
# ---------------------------------------------------------------------------

class CLIBehaviorArgumentsTest(unittest.TestCase):
    """Tests for the new behavior-related CLI arguments added to build_parser."""

    def setUp(self):
        from spider_cortex_sim.cli import build_parser
        self.parser = build_parser()

    def test_default_behavior_evaluation_matches_suite_shape(self) -> None:
        """
        Verify that the default behavior evaluation payload has the expected top-level shape and default competence.
        
        Asserts that the payload contains "suite", "legacy_scenarios", and "summary" keys, that "legacy_scenarios" is an empty dict, and that the summary's "competence_type" is "mixed".
        """
        from spider_cortex_sim.cli import _default_behavior_evaluation

        payload = _default_behavior_evaluation()
        self.assertIn("suite", payload)
        self.assertIn("legacy_scenarios", payload)
        self.assertIn("summary", payload)
        self.assertEqual(payload["legacy_scenarios"], {})
        self.assertEqual(payload["summary"]["competence_type"], "mixed")

    def test_behavior_scenario_arg_exists(self) -> None:
        args = self.parser.parse_args(["--behavior-scenario", "night_rest"])
        self.assertEqual(args.behavior_scenario, ["night_rest"])

    def test_behavior_scenario_can_be_multiple(self) -> None:
        args = self.parser.parse_args([
            "--behavior-scenario", "night_rest",
            "--behavior-scenario", "food_deprivation",
        ])
        self.assertEqual(set(args.behavior_scenario), {"night_rest", "food_deprivation"})

    def test_behavior_scenario_invalid_choice_raises(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--behavior-scenario", "nonexistent_scenario"])

    def test_behavior_suite_flag(self) -> None:
        args = self.parser.parse_args(["--behavior-suite"])
        self.assertTrue(args.behavior_suite)

    def test_behavior_suite_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.behavior_suite)

    def test_behavior_seeds_arg(self) -> None:
        args = self.parser.parse_args(["--behavior-seeds", "7", "17", "29"])
        self.assertEqual(args.behavior_seeds, [7, 17, 29])

    def test_behavior_seeds_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.behavior_seeds)

    def test_behavior_compare_profiles_flag(self) -> None:
        args = self.parser.parse_args(["--behavior-compare-profiles"])
        self.assertTrue(args.behavior_compare_profiles)

    def test_behavior_compare_profiles_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.behavior_compare_profiles)

    def test_behavior_compare_maps_flag(self) -> None:
        args = self.parser.parse_args(["--behavior-compare-maps"])
        self.assertTrue(args.behavior_compare_maps)

    def test_behavior_compare_maps_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.behavior_compare_maps)

    def test_noise_robustness_flag(self) -> None:
        args = self.parser.parse_args(["--noise-robustness"])
        self.assertTrue(args.noise_robustness)

    def test_behavior_noise_robustness_alias_flag(self) -> None:
        args = self.parser.parse_args(["--behavior-noise-robustness"])
        self.assertTrue(args.noise_robustness)

    def test_noise_robustness_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.noise_robustness)

    def test_behavior_csv_arg(self) -> None:
        args = self.parser.parse_args(["--behavior-csv", "output.csv"])
        self.assertEqual(args.behavior_csv, Path("output.csv"))

    def test_behavior_csv_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.behavior_csv)

    def test_curriculum_profile_arg(self) -> None:
        args = self.parser.parse_args(["--curriculum-profile", "ecological_v1"])
        self.assertEqual(args.curriculum_profile, "ecological_v1")

    def test_curriculum_profile_ecological_v2_arg(self) -> None:
        args = self.parser.parse_args(["--curriculum-profile", "ecological_v2"])
        self.assertEqual(args.curriculum_profile, "ecological_v2")

    def test_curriculum_profile_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.curriculum_profile, "none")

    def test_curriculum_profile_invalid_value(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--curriculum-profile", "invalid_value"])

    def test_operational_profile_arg(self) -> None:
        args = self.parser.parse_args(["--operational-profile", "default_v1"])
        self.assertEqual(args.operational_profile, "default_v1")

    def test_operational_profile_default(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.operational_profile, "default_v1")

    def test_noise_profile_arg(self) -> None:
        args = self.parser.parse_args(["--noise-profile", "medium"])
        self.assertEqual(args.noise_profile, "medium")

    def test_noise_profile_default(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.noise_profile, "none")

    def test_reflex_scale_arg(self) -> None:
        args = self.parser.parse_args(["--reflex-scale", "0.5"])
        self.assertAlmostEqual(args.reflex_scale, 0.5)

    def test_module_reflex_scale_arg(self) -> None:
        args = self.parser.parse_args(
            ["--module-reflex-scale", "alert_center=0.25", "--module-reflex-scale", "sleep_center=0.75"]
        )
        self.assertEqual(args.module_reflex_scale, ["alert_center=0.25", "sleep_center=0.75"])

    def test_reflex_anneal_final_scale_arg(self) -> None:
        args = self.parser.parse_args(["--reflex-anneal-final-scale", "0.2"])
        self.assertAlmostEqual(args.reflex_anneal_final_scale, 0.2)

    def test_budget_profile_arg(self) -> None:
        args = self.parser.parse_args(["--budget-profile", "smoke"])
        self.assertEqual(args.budget_profile, "smoke")

    def test_budget_profile_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.budget_profile)

    def test_checkpoint_selection_arg(self) -> None:
        args = self.parser.parse_args(["--checkpoint-selection", "best"])
        self.assertEqual(args.checkpoint_selection, "best")

    def test_checkpoint_selection_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.checkpoint_selection, "none")

    def test_checkpoint_metric_arg(self) -> None:
        args = self.parser.parse_args(["--checkpoint-metric", "mean_reward"])
        self.assertEqual(args.checkpoint_metric, "mean_reward")

    def test_checkpoint_metric_default(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.checkpoint_metric, "scenario_success_rate")

    def test_checkpoint_penalty_args(self) -> None:
        args = self.parser.parse_args(
            [
                "--checkpoint-override-penalty",
                "0.25",
                "--checkpoint-dominance-penalty",
                "0.5",
                "--checkpoint-penalty-mode",
                "direct",
            ]
        )
        self.assertAlmostEqual(args.checkpoint_override_penalty, 0.25)
        self.assertAlmostEqual(args.checkpoint_dominance_penalty, 0.5)
        self.assertEqual(args.checkpoint_penalty_mode, "direct")

    def test_checkpoint_penalty_defaults(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.checkpoint_override_penalty, 0.0)
        self.assertEqual(args.checkpoint_dominance_penalty, 0.0)
        self.assertEqual(args.checkpoint_penalty_mode, "tiebreaker")

    def test_checkpoint_interval_arg(self) -> None:
        args = self.parser.parse_args(["--checkpoint-interval", "5"])
        self.assertEqual(args.checkpoint_interval, 5)

    def test_checkpoint_dir_arg(self) -> None:
        args = self.parser.parse_args(["--checkpoint-dir", "checkpoints"])
        self.assertEqual(args.checkpoint_dir, Path("checkpoints"))

    def test_ablation_suite_flag(self) -> None:
        args = self.parser.parse_args(["--ablation-suite"])
        self.assertTrue(args.ablation_suite)

    def test_ablation_suite_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.ablation_suite)

    def test_ablation_variant_arg(self) -> None:
        args = self.parser.parse_args(["--ablation-variant", "monolithic_policy"])
        self.assertEqual(args.ablation_variant, ["monolithic_policy"])

    def test_ablation_variant_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.ablation_variant)

    def test_ablation_seeds_arg(self) -> None:
        args = self.parser.parse_args(["--ablation-seeds", "7", "17", "29"])
        self.assertEqual(args.ablation_seeds, [7, 17, 29])

    def test_ablation_seeds_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.ablation_seeds)

    def test_ablation_seeds_single_value(self) -> None:
        args = self.parser.parse_args(["--ablation-seeds", "42"])
        self.assertEqual(args.ablation_seeds, [42])

    def test_learning_evidence_flag(self) -> None:
        args = self.parser.parse_args(["--learning-evidence"])
        self.assertTrue(args.learning_evidence)

    def test_learning_evidence_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.learning_evidence)

    def test_experiment_of_record_flag(self) -> None:
        args = self.parser.parse_args(["--experiment-of-record"])
        self.assertTrue(args.experiment_of_record)

    def test_experiment_of_record_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.experiment_of_record)

    def test_claim_test_suite_flag(self) -> None:
        args = self.parser.parse_args(["--claim-test-suite"])
        self.assertTrue(args.claim_test_suite)

    def test_claim_test_suite_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.claim_test_suite)

    def test_claim_test_arg(self) -> None:
        args = self.parser.parse_args(
            ["--claim-test", "learning_without_privileged_signals"]
        )
        self.assertEqual(args.claim_test, ["learning_without_privileged_signals"])

    def test_claim_test_multiple_flags(self) -> None:
        args = self.parser.parse_args(
            [
                "--claim-test",
                "learning_without_privileged_signals",
                "--claim-test",
                "escape_without_reflex_support",
            ]
        )
        self.assertEqual(
            args.claim_test,
            [
                "learning_without_privileged_signals",
                "escape_without_reflex_support",
            ],
        )

    def test_claim_test_invalid_choice_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--claim-test", "not_a_claim_test"])

    def test_short_claim_test_suite_summary_preserves_skipped_state(self) -> None:
        from spider_cortex_sim.cli import _short_claim_test_suite_summary

        summary = _short_claim_test_suite_summary(
            {
                "claims": {
                    "learning_without_privileged_signals": {
                        "status": "skipped",
                        "passed": False,
                    }
                },
                "summary": {
                    "claims_passed": 0,
                    "claims_failed": 0,
                    "claims_skipped": 1,
                    "all_primary_claims_passed": False,
                },
            }
        )

        self.assertEqual(
            summary["claims"]["learning_without_privileged_signals"],
            {"passed": None, "skipped": True},
        )
        self.assertEqual(summary["claims_skipped"], 1)

    def test_learning_evidence_long_budget_profile_default(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.learning_evidence_long_budget_profile, "report")

    def test_learning_evidence_long_budget_profile_arg(self) -> None:
        args = self.parser.parse_args(
            ["--learning-evidence-long-budget-profile", "smoke"]
        )
        self.assertEqual(args.learning_evidence_long_budget_profile, "smoke")

    def test_ablation_variant_multiple_flags(self) -> None:
        args = self.parser.parse_args([
            "--ablation-variant", "monolithic_policy",
            "--ablation-variant", "no_module_dropout",
        ])
        self.assertEqual(args.ablation_variant, ["monolithic_policy", "no_module_dropout"])

    def test_ablation_variant_invalid_choice_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--ablation-variant", "not_a_real_variant"])

    def test_ablation_suite_and_variant_are_independent(self) -> None:
        args = self.parser.parse_args(["--ablation-suite", "--ablation-variant", "modular_full"])
        self.assertTrue(args.ablation_suite)
        self.assertEqual(args.ablation_variant, ["modular_full"])

    def test_behavior_scenario_none_by_default(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.behavior_scenario)


# ---------------------------------------------------------------------------
# Integration: evaluate_behavior_suite uses EpisodeStats.seed
# ---------------------------------------------------------------------------

class EvaluateBehaviorSuiteIntegrationTest(unittest.TestCase):
    """Integration tests for evaluate_behavior_suite and behavior payload structure."""

    def setUp(self):
        from spider_cortex_sim.simulation import SpiderSimulation
        self.SimClass = SpiderSimulation

    def test_payload_has_required_keys(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12)
        payload, _trace, _rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertIn("suite", payload)
        self.assertIn("summary", payload)
        self.assertIn("legacy_scenarios", payload)

    def test_summary_has_required_keys(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12)
        payload, _, _ = sim.evaluate_behavior_suite(["night_rest"])
        summary = payload["summary"]
        for key in ["scenario_count", "episode_count", "scenario_success_rate", "episode_success_rate", "regressions"]:
            self.assertIn(key, summary)

    def test_rows_have_scenario_column(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12)
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertTrue(rows)
        self.assertIn("scenario", rows[0])
        self.assertIn("ablation_variant", rows[0])
        self.assertIn("ablation_architecture", rows[0])

    def test_rows_have_check_columns(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12)
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        self.assertTrue(rows)
        self.assertIn("check_deep_night_shelter_passed", rows[0])

    def test_rows_separate_scenario_map_from_evaluation_map(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12, map_template="central_burrow")
        _, _, rows = sim.evaluate_behavior_suite(["open_field_foraging"])
        self.assertTrue(rows)
        self.assertEqual(rows[0]["scenario_map"], "exposed_feeding_ground")
        self.assertEqual(rows[0]["evaluation_map"], "central_burrow")

    def test_suite_scenario_success_rate_is_float(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12)
        payload, _, _ = sim.evaluate_behavior_suite(["food_deprivation"])
        success_rate = payload["suite"]["food_deprivation"]["success_rate"]
        self.assertIsInstance(success_rate, float)

    def test_multiple_episodes_per_scenario(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12)
        payload, _, rows = sim.evaluate_behavior_suite(["night_rest"], episodes_per_scenario=2)
        self.assertEqual(payload["suite"]["night_rest"]["episodes"], 2)
        self.assertEqual(len(rows), 2)

    def test_compare_behavior_suite_includes_shaping_audit(self) -> None:
        payload, _rows = self.SimClass.compare_behavior_suite(
            episodes=0,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
            reward_profiles=("classic", "austere"),
        )
        self.assertIn("reward_audit", payload)
        self.assertEqual(payload["reward_audit"]["minimal_profile"], "austere")
        self.assertIn("comparison", payload["reward_audit"])
        self.assertIn(
            "classic",
            payload["reward_audit"]["comparison"]["deltas_vs_minimal"],
        )

    def test_compare_behavior_suite_omits_minimal_baseline_when_austere_not_compared(self) -> None:
        payload, _rows = self.SimClass.compare_behavior_suite(
            episodes=0,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
            reward_profiles=("classic", "ecological"),
        )
        self.assertIsNone(payload["reward_audit"]["minimal_profile"])
        self.assertEqual(
            payload["reward_audit"]["comparison"]["deltas_vs_minimal"],
            {},
        )


# ---------------------------------------------------------------------------
# Tests for _default_checkpointing_summary from cli.py
# ---------------------------------------------------------------------------

class DefaultCheckpointingSummaryTest(unittest.TestCase):
    """Tests for the _default_checkpointing_summary helper in cli.py."""

    def _get_fn(self):
        from spider_cortex_sim.cli import _default_checkpointing_summary
        return _default_checkpointing_summary

    def test_enabled_true_when_selection_is_best(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="scenario_success_rate",
            checkpoint_interval=4,
            selection_scenario_episodes=1,
        )
        self.assertTrue(result["enabled"])

    def test_enabled_false_when_selection_is_none(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="none",
            metric="scenario_success_rate",
            checkpoint_interval=4,
            selection_scenario_episodes=1,
        )
        self.assertFalse(result["enabled"])

    def test_selection_field_preserved(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="mean_reward",
            checkpoint_interval=6,
            selection_scenario_episodes=2,
        )
        self.assertEqual(result["selection"], "best")

    def test_metric_field_preserved(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="episode_success_rate",
            checkpoint_interval=4,
            selection_scenario_episodes=1,
        )
        self.assertEqual(result["metric"], "episode_success_rate")

    def test_checkpoint_interval_field_preserved(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="scenario_success_rate",
            checkpoint_interval=10,
            selection_scenario_episodes=1,
        )
        self.assertEqual(result["checkpoint_interval"], 10)

    def test_evaluation_source_is_behavior_suite(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="scenario_success_rate",
            checkpoint_interval=4,
            selection_scenario_episodes=1,
        )
        self.assertEqual(result["evaluation_source"], "behavior_suite")

    def test_selection_scenario_episodes_preserved(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="scenario_success_rate",
            checkpoint_interval=4,
            selection_scenario_episodes=3,
        )
        self.assertEqual(result["selection_scenario_episodes"], 3)

    def test_generated_checkpoints_is_empty_list(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="scenario_success_rate",
            checkpoint_interval=4,
            selection_scenario_episodes=1,
        )
        self.assertEqual(result["generated_checkpoints"], [])

    def test_selected_checkpoint_has_scope_per_run(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="scenario_success_rate",
            checkpoint_interval=4,
            selection_scenario_episodes=1,
        )
        self.assertIn("selected_checkpoint", result)
        self.assertEqual(result["selected_checkpoint"]["scope"], "per_run")

    def test_penalty_config_preserved(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="mean_reward",
            checkpoint_interval=4,
            selection_scenario_episodes=1,
            override_penalty_weight=0.25,
            dominance_penalty_weight=0.5,
            penalty_mode="direct",
        )
        self.assertEqual(result["penalty_mode"], "direct")
        self.assertEqual(
            result["penalty_config"],
            {
                "metric": "mean_reward",
                "override_penalty_weight": 0.25,
                "dominance_penalty_weight": 0.5,
                "penalty_mode": "direct",
            },
        )

    def test_full_structure_keys(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="scenario_success_rate",
            checkpoint_interval=4,
            selection_scenario_episodes=1,
        )
        expected_keys = {
            "enabled",
            "selection",
            "metric",
            "penalty_mode",
            "penalty_config",
            "checkpoint_interval",
            "evaluation_source",
            "selection_scenario_episodes",
            "generated_checkpoints",
            "selected_checkpoint",
        }
        self.assertEqual(set(result.keys()), expected_keys)


# ---------------------------------------------------------------------------
# Tests for budget-related fields added to CLI budget_profile arg choices
# ---------------------------------------------------------------------------

class CLIBudgetProfileChoicesTest(unittest.TestCase):
    """Verify the --budget-profile argument accepts only canonical names."""

    def setUp(self) -> None:
        from spider_cortex_sim.cli import build_parser
        self.parser = build_parser()

    def test_budget_profile_accepts_smoke(self) -> None:
        args = self.parser.parse_args(["--budget-profile", "smoke"])
        self.assertEqual(args.budget_profile, "smoke")

    def test_budget_profile_accepts_dev(self) -> None:
        args = self.parser.parse_args(["--budget-profile", "dev"])
        self.assertEqual(args.budget_profile, "dev")

    def test_budget_profile_accepts_report(self) -> None:
        args = self.parser.parse_args(["--budget-profile", "report"])
        self.assertEqual(args.budget_profile, "report")

    def test_budget_profile_rejects_invalid_choice(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--budget-profile", "invalid_profile"])

    def test_checkpoint_selection_rejects_invalid_choice(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--checkpoint-selection", "auto"])

    def test_checkpoint_metric_rejects_invalid_choice(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--checkpoint-metric", "not_a_metric"])

    def test_checkpoint_penalty_mode_rejects_invalid_choice(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--checkpoint-penalty-mode", "soft"])

    def test_episodes_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.episodes)

    def test_eval_episodes_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.eval_episodes)

    def test_max_steps_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.max_steps)

    def test_scenario_episodes_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.scenario_episodes)

    def test_checkpoint_interval_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.checkpoint_interval)

    def test_checkpoint_dir_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.checkpoint_dir)


class EpisodeMetricAccumulatorReflexTest(unittest.TestCase):
    """Tests for EpisodeMetricAccumulator.record_decision reflex tracking (new in this PR)."""

    def _make_accumulator(self) -> EpisodeMetricAccumulator:
        return EpisodeMetricAccumulator(
            reward_component_names=[],
            predator_states=["PATROL"],
        )

    def _make_decision(
        self,
        *,
        final_reflex_override: bool = False,
        module_results: list | None = None,
        arbitration_decision: object | None = None,
    ) -> object:
        class FakeDecision:
            pass

        decision = FakeDecision()
        decision.final_reflex_override = final_reflex_override
        decision.module_results = module_results or []
        decision.arbitration_decision = arbitration_decision
        return decision

    def _make_arbitration(
        self,
        *,
        module_contribution_share: dict[str, float] | None = None,
        dominant_module: str = "alert_center",
        dominant_module_share: float = 0.0,
        effective_module_count: float = 0.0,
        module_agreement_rate: float = 0.0,
        module_disagreement_rate: float = 0.0,
    ) -> object:
        class FakeArbitration:
            pass

        arbitration = FakeArbitration()
        arbitration.module_contribution_share = module_contribution_share or {}
        arbitration.dominant_module = dominant_module
        arbitration.dominant_module_share = dominant_module_share
        arbitration.effective_module_count = effective_module_count
        arbitration.module_agreement_rate = module_agreement_rate
        arbitration.module_disagreement_rate = module_disagreement_rate
        return arbitration

    def _make_module_result(
        self,
        name: str,
        *,
        reflex_applied: bool = False,
        module_reflex_override: bool = False,
        module_reflex_dominance: float = 0.0,
    ) -> object:
        class FakeModuleResult:
            pass

        result = FakeModuleResult()
        result.name = name
        result.reflex_applied = reflex_applied
        result.module_reflex_override = module_reflex_override
        result.module_reflex_dominance = module_reflex_dominance
        return result

    def _record_motor_transition(
        self,
        acc: EpisodeMetricAccumulator,
        *,
        occurred: bool,
        terrain: str,
        orientation_alignment: float,
        terrain_difficulty: float,
    ) -> None:
        class FakeState:
            sleep_debt = 0.0
            last_move_dx = 0
            last_move_dy = 0

        meta = {
            "food_dist": 3,
            "shelter_dist": 2,
            "predator_visible": False,
            "diagnostic": {"diagnostic_predator_dist": 9},
            "night": False,
            "shelter_role": "outside",
            "on_shelter": False,
        }
        acc.record_transition(
            step=0,
            observation_meta=meta,
            next_meta=meta,
            info={
                "reward_components": {},
                "predator_contact": False,
                "motor_noise_applied": occurred,
                "motor_execution_components": {
                    "orientation_alignment": orientation_alignment,
                    "terrain_difficulty": terrain_difficulty,
                },
                "motor_slip": {
                    "occurred": occurred,
                    "terrain": terrain,
                    "components": {
                        "orientation_alignment": orientation_alignment,
                        "terrain_difficulty": terrain_difficulty,
                    },
                },
            },
            state=FakeState(),
            predator_state_before="PATROL",
            predator_state="PATROL",
        )

    def test_decision_steps_increments_each_call(self) -> None:
        acc = self._make_accumulator()
        self.assertEqual(acc.decision_steps, 0)
        acc.record_decision(self._make_decision())
        self.assertEqual(acc.decision_steps, 1)
        acc.record_decision(self._make_decision())
        self.assertEqual(acc.decision_steps, 2)

    def test_no_reflex_applied_leaves_reflex_steps_zero(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=False),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.reflex_steps, 0)

    def test_reflex_applied_increments_reflex_steps(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.reflex_steps, 1)

    def test_reflex_steps_counts_steps_not_modules(self) -> None:
        # Two modules both apply reflex in one step: reflex_steps should be 1, not 2
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=True),
                self._make_module_result("hunger_center", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.reflex_steps, 1)

    def test_final_reflex_override_increments_counter(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision(final_reflex_override=True))
        self.assertEqual(acc.final_reflex_override_steps, 1)

    def test_final_reflex_override_false_does_not_increment(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision(final_reflex_override=False))
        self.assertEqual(acc.final_reflex_override_steps, 0)

    def test_module_reflex_usage_tracked_by_module_name(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.module_reflex_usage_steps["alert_center"], 1)

    def test_module_reflex_override_tracked_by_module_name(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", module_reflex_override=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.module_reflex_override_steps["alert_center"], 1)

    def test_module_reflex_dominance_accumulated(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", module_reflex_dominance=0.4),
            ]
        )
        acc.record_decision(decision)
        self.assertAlmostEqual(acc.module_reflex_dominance_sums["alert_center"], 0.4)

    def test_unknown_module_name_in_results_is_ignored(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("nonexistent_module", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        self.assertEqual(acc.reflex_steps, 0)

    def test_snapshot_includes_reflex_usage_rate(self) -> None:
        acc = self._make_accumulator()
        decision = self._make_decision(
            module_results=[
                self._make_module_result("alert_center", reflex_applied=True),
            ]
        )
        acc.record_decision(decision)
        snapshot = acc.snapshot()
        self.assertIn("reflex_usage_rate", snapshot)
        self.assertAlmostEqual(snapshot["reflex_usage_rate"], 1.0)

    def test_snapshot_reflex_usage_rate_is_zero_when_no_reflex(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision())
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["reflex_usage_rate"], 0.0)

    def test_snapshot_includes_final_reflex_override_rate(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision(final_reflex_override=True))
        acc.record_decision(self._make_decision(final_reflex_override=False))
        snapshot = acc.snapshot()
        self.assertIn("final_reflex_override_rate", snapshot)
        self.assertAlmostEqual(snapshot["final_reflex_override_rate"], 0.5)

    def test_snapshot_includes_module_reflex_usage_rates(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision())
        snapshot = acc.snapshot()
        self.assertIn("module_reflex_usage_rates", snapshot)
        for name in REFLEX_MODULE_NAMES:
            self.assertIn(name, snapshot["module_reflex_usage_rates"])

    def test_snapshot_includes_mean_reflex_dominance(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(self._make_decision())
        snapshot = acc.snapshot()
        self.assertIn("mean_reflex_dominance", snapshot)
        self.assertIsInstance(snapshot["mean_reflex_dominance"], float)

    def test_module_contribution_share_accumulated_from_arbitration(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(
            self._make_decision(
                arbitration_decision=self._make_arbitration(
                    module_contribution_share={"alert_center": 0.75},
                )
            )
        )
        self.assertAlmostEqual(
            acc.module_contribution_share_sums["alert_center"],
            0.75,
        )

    def test_snapshot_includes_independence_metrics(self) -> None:
        acc = self._make_accumulator()
        acc.record_decision(
            self._make_decision(
                arbitration_decision=self._make_arbitration(
                    module_contribution_share={"alert_center": 0.7, "hunger_center": 0.3},
                    dominant_module="alert_center",
                    dominant_module_share=0.7,
                    effective_module_count=1.72,
                    module_agreement_rate=0.8,
                    module_disagreement_rate=0.2,
                )
            )
        )
        snapshot = acc.snapshot()
        self.assertIn("module_contribution_share", snapshot)
        self.assertIn("dominant_module_distribution", snapshot)
        self.assertEqual(snapshot["dominant_module"], "alert_center")
        self.assertAlmostEqual(snapshot["dominant_module_share"], 0.7)
        self.assertAlmostEqual(snapshot["effective_module_count"], 1.72)
        self.assertAlmostEqual(snapshot["module_agreement_rate"], 0.8)
        self.assertAlmostEqual(snapshot["module_disagreement_rate"], 0.2)
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, snapshot["module_contribution_share"])
            self.assertIn(name, snapshot["dominant_module_distribution"])

    def test_record_learning_accumulates_credit_weights_and_gradient_norms(self) -> None:
        acc = self._make_accumulator()
        acc.record_learning(
            {
                "module_credit_weights": {"alert_center": 0.25},
                "module_gradient_norms": {"alert_center": 1.5},
            }
        )
        acc.record_learning(
            {
                "module_credit_weights": {"alert_center": 0.75},
                "module_gradient_norms": {"alert_center": 2.5},
            }
        )
        snapshot = acc.snapshot()
        self.assertAlmostEqual(
            snapshot["mean_module_credit_weights"]["alert_center"],
            0.5,
        )
        self.assertAlmostEqual(
            snapshot["module_gradient_norm_means"]["alert_center"],
            2.0,
        )

    def test_snapshot_includes_learning_credit_maps(self) -> None:
        acc = self._make_accumulator()
        snapshot = acc.snapshot()
        self.assertIn("mean_module_credit_weights", snapshot)
        self.assertIn("module_gradient_norm_means", snapshot)
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, snapshot["mean_module_credit_weights"])
            self.assertIn(name, snapshot["module_gradient_norm_means"])

    def test_snapshot_denominator_uses_max_one_when_no_decisions(self) -> None:
        # With zero decisions, snapshot should still not raise division-by-zero
        acc = self._make_accumulator()
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["reflex_usage_rate"], 0.0)
        self.assertAlmostEqual(snapshot["final_reflex_override_rate"], 0.0)

    def test_snapshot_includes_motor_execution_metrics(self) -> None:
        acc = self._make_accumulator()
        self._record_motor_transition(
            acc,
            occurred=False,
            terrain="open",
            orientation_alignment=1.0,
            terrain_difficulty=0.0,
        )
        self._record_motor_transition(
            acc,
            occurred=True,
            terrain="narrow",
            orientation_alignment=0.25,
            terrain_difficulty=0.7,
        )
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["motor_slip_rate"], 0.5)
        self.assertAlmostEqual(snapshot["mean_orientation_alignment"], 0.625)
        self.assertAlmostEqual(snapshot["mean_terrain_difficulty"], 0.35)
        self.assertAlmostEqual(snapshot["terrain_slip_rates"]["open"], 0.0)
        self.assertAlmostEqual(snapshot["terrain_slip_rates"]["narrow"], 1.0)

    def test_post_init_initializes_all_module_reflex_dicts(self) -> None:
        acc = self._make_accumulator()
        for name in REFLEX_MODULE_NAMES:
            self.assertIn(name, acc.module_reflex_usage_steps)
            self.assertIn(name, acc.module_reflex_override_steps)
            self.assertIn(name, acc.module_reflex_dominance_sums)
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, acc.module_credit_weight_sums)
            self.assertIn(name, acc.module_gradient_norm_sums)


class AggregateEpisodeStatsReflexTest(unittest.TestCase):
    """Tests for aggregate_episode_stats reflex metrics (new in this PR)."""

    def _make_episode_stats(
        self,
        *,
        reflex_usage_rate: float = 0.0,
        final_reflex_override_rate: float = 0.0,
        mean_reflex_dominance: float = 0.0,
        module_reflex_usage_rates: dict | None = None,
        module_reflex_override_rates: dict | None = None,
        module_reflex_dominance: dict | None = None,
        module_contribution_share: dict | None = None,
        dominant_module: str = "",
        dominant_module_share: float = 0.0,
        effective_module_count: float = 0.0,
        module_agreement_rate: float = 0.0,
        module_disagreement_rate: float = 0.0,
        mean_module_credit_weights: dict | None = None,
        module_gradient_norm_means: dict | None = None,
        motor_slip_rate: float = 0.0,
        mean_orientation_alignment: float = 0.0,
        mean_terrain_difficulty: float = 0.0,
        terrain_slip_rates: dict | None = None,
    ) -> EpisodeStats:
        """
        Create deterministic EpisodeStats for reflex and module metric tests.

        Key rates include reflex_usage_rate, final_reflex_override_rate,
        mean_reflex_dominance, and dominant_module_share (0.0-1.0).
        """
        module_rates = {name: 0.0 for name in REFLEX_MODULE_NAMES}
        module_overrides = {name: 0.0 for name in REFLEX_MODULE_NAMES}
        module_dominance = {name: 0.0 for name in REFLEX_MODULE_NAMES}
        module_contribution = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        module_credit = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        module_gradients = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        if module_reflex_usage_rates:
            module_rates.update(module_reflex_usage_rates)
        if module_reflex_override_rates:
            module_overrides.update(module_reflex_override_rates)
        if module_reflex_dominance:
            module_dominance.update(module_reflex_dominance)
        if module_contribution_share:
            module_contribution.update(module_contribution_share)
        if mean_module_credit_weights:
            module_credit.update(mean_module_credit_weights)
        if module_gradient_norm_means:
            module_gradients.update(module_gradient_norm_means)
        return EpisodeStats(
            episode=0,
            seed=7,
            training=False,
            scenario=None,
            total_reward=0.0,
            steps=10,
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
            final_hunger=0.0,
            final_fatigue=0.0,
            final_sleep_debt=0.0,
            final_health=1.0,
            alive=True,
            reward_component_totals={},
            predator_state_ticks={"PATROL": 10},
            predator_mode_transitions=0,
            dominant_predator_state="PATROL",
            reflex_usage_rate=reflex_usage_rate,
            final_reflex_override_rate=final_reflex_override_rate,
            mean_reflex_dominance=mean_reflex_dominance,
            module_reflex_usage_rates=module_rates,
            module_reflex_override_rates=module_overrides,
            module_reflex_dominance=module_dominance,
            module_contribution_share=module_contribution,
            dominant_module=dominant_module,
            dominant_module_share=dominant_module_share,
            effective_module_count=effective_module_count,
            module_agreement_rate=module_agreement_rate,
            module_disagreement_rate=module_disagreement_rate,
            mean_module_credit_weights=module_credit,
            module_gradient_norm_means=module_gradients,
            motor_slip_rate=motor_slip_rate,
            mean_orientation_alignment=mean_orientation_alignment,
            mean_terrain_difficulty=mean_terrain_difficulty,
            terrain_slip_rates=dict(terrain_slip_rates or {}),
        )

    def test_aggregate_empty_history_returns_zeros(self) -> None:
        result = aggregate_episode_stats([])
        self.assertAlmostEqual(result["mean_reflex_usage_rate"], 0.0)
        self.assertAlmostEqual(result["mean_final_reflex_override_rate"], 0.0)
        self.assertAlmostEqual(result["mean_reflex_dominance"], 0.0)
        self.assertAlmostEqual(result["mean_motor_slip_rate"], 0.0)
        self.assertAlmostEqual(result["mean_orientation_alignment"], 0.0)
        self.assertAlmostEqual(result["mean_terrain_difficulty"], 0.0)
        self.assertEqual(result["mean_terrain_slip_rates"], {})
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, result["mean_module_credit_weights"])
            self.assertIn(name, result["module_gradient_norm_means"])

    def test_aggregate_includes_mean_reflex_usage_rate(self) -> None:
        history = [
            self._make_episode_stats(reflex_usage_rate=0.4),
            self._make_episode_stats(reflex_usage_rate=0.6),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_reflex_usage_rate"], 0.5)

    def test_aggregate_includes_motor_execution_metrics(self) -> None:
        history = [
            self._make_episode_stats(
                motor_slip_rate=0.2,
                mean_orientation_alignment=0.8,
                mean_terrain_difficulty=0.1,
                terrain_slip_rates={"open": 0.0, "narrow": 0.4},
            ),
            self._make_episode_stats(
                motor_slip_rate=0.6,
                mean_orientation_alignment=0.4,
                mean_terrain_difficulty=0.5,
                terrain_slip_rates={"narrow": 0.8},
            ),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_motor_slip_rate"], 0.4)
        self.assertAlmostEqual(result["mean_orientation_alignment"], 0.6)
        self.assertAlmostEqual(result["mean_terrain_difficulty"], 0.3)
        self.assertAlmostEqual(result["mean_terrain_slip_rates"]["open"], 0.0)
        self.assertAlmostEqual(result["mean_terrain_slip_rates"]["narrow"], 0.6)

    def test_aggregate_includes_mean_final_reflex_override_rate(self) -> None:
        history = [
            self._make_episode_stats(final_reflex_override_rate=0.2),
            self._make_episode_stats(final_reflex_override_rate=0.4),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_final_reflex_override_rate"], 0.3)

    def test_aggregate_includes_mean_reflex_dominance(self) -> None:
        history = [
            self._make_episode_stats(mean_reflex_dominance=0.1),
            self._make_episode_stats(mean_reflex_dominance=0.3),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_reflex_dominance"], 0.2)

    def test_aggregate_includes_mean_module_reflex_usage_rate(self) -> None:
        history = [
            self._make_episode_stats(module_reflex_usage_rates={"alert_center": 0.4}),
            self._make_episode_stats(module_reflex_usage_rates={"alert_center": 0.8}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_reflex_usage_rate", result)
        self.assertAlmostEqual(result["mean_module_reflex_usage_rate"]["alert_center"], 0.6)

    def test_aggregate_includes_mean_module_reflex_override_rate(self) -> None:
        history = [
            self._make_episode_stats(module_reflex_override_rates={"alert_center": 0.2}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_reflex_override_rate", result)
        self.assertAlmostEqual(result["mean_module_reflex_override_rate"]["alert_center"], 0.2)

    def test_aggregate_includes_mean_module_reflex_dominance(self) -> None:
        history = [
            self._make_episode_stats(module_reflex_dominance={"alert_center": 0.35}),
            self._make_episode_stats(module_reflex_dominance={"alert_center": 0.65}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_reflex_dominance", result)
        self.assertAlmostEqual(result["mean_module_reflex_dominance"]["alert_center"], 0.5)

    def test_aggregate_module_rate_keys_include_all_module_names(self) -> None:
        history = [self._make_episode_stats()]
        result = aggregate_episode_stats(history)
        for name in REFLEX_MODULE_NAMES:
            self.assertIn(name, result["mean_module_reflex_usage_rate"])
            self.assertIn(name, result["mean_module_reflex_override_rate"])
            self.assertIn(name, result["mean_module_reflex_dominance"])

    def test_aggregate_includes_mean_module_contribution_share(self) -> None:
        history = [
            self._make_episode_stats(module_contribution_share={"alert_center": 0.2}),
            self._make_episode_stats(module_contribution_share={"alert_center": 0.6}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_contribution_share", result)
        self.assertAlmostEqual(
            result["mean_module_contribution_share"]["alert_center"],
            0.4,
        )

    def test_aggregate_includes_mean_module_credit_weights(self) -> None:
        history = [
            self._make_episode_stats(mean_module_credit_weights={"alert_center": 0.25}),
            self._make_episode_stats(mean_module_credit_weights={"alert_center": 0.75}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("mean_module_credit_weights", result)
        self.assertAlmostEqual(
            result["mean_module_credit_weights"]["alert_center"],
            0.5,
        )

    def test_aggregate_includes_module_gradient_norm_means(self) -> None:
        history = [
            self._make_episode_stats(module_gradient_norm_means={"alert_center": 1.5}),
            self._make_episode_stats(module_gradient_norm_means={"alert_center": 2.5}),
        ]
        result = aggregate_episode_stats(history)
        self.assertIn("module_gradient_norm_means", result)
        self.assertAlmostEqual(
            result["module_gradient_norm_means"]["alert_center"],
            2.0,
        )

    def test_aggregate_includes_dominant_module_distribution(self) -> None:
        history = [
            self._make_episode_stats(dominant_module="alert_center"),
            self._make_episode_stats(dominant_module="alert_center"),
            self._make_episode_stats(dominant_module="hunger_center"),
        ]
        result = aggregate_episode_stats(history)
        self.assertEqual(result["dominant_module"], "alert_center")
        self.assertAlmostEqual(
            result["dominant_module_distribution"]["alert_center"],
            2 / 3,
        )

    def test_aggregate_includes_mean_independence_scalars(self) -> None:
        history = [
            self._make_episode_stats(
                dominant_module_share=0.7,
                effective_module_count=1.5,
                module_agreement_rate=0.8,
                module_disagreement_rate=0.2,
            ),
            self._make_episode_stats(
                dominant_module_share=0.5,
                effective_module_count=2.0,
                module_agreement_rate=0.6,
                module_disagreement_rate=0.4,
            ),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_dominant_module_share"], 0.6)
        self.assertAlmostEqual(result["mean_effective_module_count"], 1.75)
        self.assertAlmostEqual(result["mean_module_agreement_rate"], 0.7)
        self.assertAlmostEqual(result["mean_module_disagreement_rate"], 0.3)


class ParseModuleReflexScalesTest(unittest.TestCase):
    """Tests for spider_cortex_sim.cli._parse_module_reflex_scales."""

    def setUp(self) -> None:
        from spider_cortex_sim.cli import _parse_module_reflex_scales
        self._parse = _parse_module_reflex_scales

    def test_none_input_returns_empty_dict(self) -> None:
        result = self._parse(None)
        self.assertEqual(result, {})

    def test_empty_list_returns_empty_dict(self) -> None:
        result = self._parse([])
        self.assertEqual(result, {})

    def test_single_valid_entry(self) -> None:
        result = self._parse(["alert_center=0.5"])
        self.assertAlmostEqual(result["alert_center"], 0.5)

    def test_multiple_valid_entries(self) -> None:
        result = self._parse(["alert_center=0.25", "sleep_center=0.75"])
        self.assertAlmostEqual(result["alert_center"], 0.25)
        self.assertAlmostEqual(result["sleep_center"], 0.75)

    def test_later_entry_overwrites_earlier_for_same_module(self) -> None:
        result = self._parse(["alert_center=0.3", "alert_center=0.9"])
        self.assertAlmostEqual(result["alert_center"], 0.9)

    def test_missing_equals_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self._parse(["alert_center0.5"])

    def test_empty_module_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self._parse(["=0.5"])

    def test_invalid_scale_float_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self._parse(["alert_center=not_a_float"])

    def test_zero_scale_accepted(self) -> None:
        result = self._parse(["alert_center=0.0"])
        self.assertAlmostEqual(result["alert_center"], 0.0)

    def test_integer_scale_converted_to_float(self) -> None:
        result = self._parse(["alert_center=1"])
        self.assertIsInstance(result["alert_center"], float)
        self.assertAlmostEqual(result["alert_center"], 1.0)

    def test_whitespace_stripped_from_module_name(self) -> None:
        result = self._parse([" alert_center = 0.5"])
        self.assertIn("alert_center", result)
        self.assertAlmostEqual(result["alert_center"], 0.5)

    def test_scale_text_with_equals_in_value_parses_first_equals(self) -> None:
        # "module=1.0=extra" - split on first "=" gives module_name="module", scale_text="1.0=extra"
        # which should fail as non-float
        with self.assertRaises(ValueError):
            self._parse(["alert_center=1.0=extra"])


class CLIReflexScaleDefaultsTest(unittest.TestCase):
    """Tests for default values of new reflex-related CLI arguments."""

    def setUp(self) -> None:
        from spider_cortex_sim.cli import build_parser
        self.parser = build_parser()

    def test_reflex_scale_default_is_one(self) -> None:
        args = self.parser.parse_args([])
        self.assertAlmostEqual(args.reflex_scale, 1.0)

    def test_module_reflex_scale_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.module_reflex_scale)

    def test_reflex_anneal_final_scale_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.reflex_anneal_final_scale)

    def test_module_reflex_scale_multiple_appends(self) -> None:
        args = self.parser.parse_args([
            "--module-reflex-scale", "alert_center=0.5",
            "--module-reflex-scale", "sleep_center=0.25",
        ])
        self.assertIn("alert_center=0.5", args.module_reflex_scale)
        self.assertIn("sleep_center=0.25", args.module_reflex_scale)


# ---------------------------------------------------------------------------
# _payload_float edge cases
# ---------------------------------------------------------------------------

class PayloadFloatEdgeCasesTest(unittest.TestCase):
    """Comprehensive edge case tests for _payload_float."""

    def test_missing_top_level_key_returns_none(self) -> None:
        payload = {"other_key": 1.0}
        self.assertIsNone(_payload_float(payload, "missing_key"))

    def test_missing_nested_key_returns_none(self) -> None:
        payload = {"outer": {"inner": 0.5}}
        self.assertIsNone(_payload_float(payload, "outer", "missing"))

    def test_none_value_returns_none(self) -> None:
        payload = {"key": None}
        self.assertIsNone(_payload_float(payload, "key"))

    def test_non_numeric_string_returns_none(self) -> None:
        payload = {"key": "not_a_number"}
        self.assertIsNone(_payload_float(payload, "key"))

    def test_valid_int_returns_float(self) -> None:
        payload = {"key": 3}
        result = _payload_float(payload, "key")
        self.assertAlmostEqual(result, 3.0)
        self.assertIsInstance(result, float)

    def test_valid_float_returned(self) -> None:
        payload = {"key": 0.75}
        self.assertAlmostEqual(_payload_float(payload, "key"), 0.75)

    def test_empty_path_returns_none(self) -> None:
        # No path arguments -> current is the dict itself, float({}) raises TypeError
        payload = {"key": 1.0}
        self.assertIsNone(_payload_float(payload))

    def test_intermediate_not_dict_returns_none(self) -> None:
        payload = {"outer": "string_not_dict"}
        self.assertIsNone(_payload_float(payload, "outer", "inner"))

    def test_deep_nested_path(self) -> None:
        payload = {"a": {"b": {"c": 0.42}}}
        self.assertAlmostEqual(_payload_float(payload, "a", "b", "c"), 0.42)

    def test_bool_true_converts_to_one(self) -> None:
        payload = {"flag": True}
        self.assertAlmostEqual(_payload_float(payload, "flag"), 1.0)

    def test_bool_false_converts_to_zero(self) -> None:
        payload = {"flag": False}
        self.assertAlmostEqual(_payload_float(payload, "flag"), 0.0)

    def test_zero_float_value(self) -> None:
        payload = {"key": 0.0}
        self.assertEqual(_payload_float(payload, "key"), 0.0)

    def test_list_as_value_returns_none(self) -> None:
        payload = {"key": [1, 2, 3]}
        self.assertIsNone(_payload_float(payload, "key"))


# ---------------------------------------------------------------------------
# _payload_text edge cases
# ---------------------------------------------------------------------------

class PayloadTextEdgeCasesTest(unittest.TestCase):
    """Comprehensive edge case tests for _payload_text."""

    def test_missing_top_level_key_returns_none(self) -> None:
        payload = {"other_key": "value"}
        self.assertIsNone(_payload_text(payload, "missing_key"))

    def test_none_value_returns_none(self) -> None:
        payload = {"key": None}
        self.assertIsNone(_payload_text(payload, "key"))

    def test_string_value_returned_as_is(self) -> None:
        payload = {"key": "threat"}
        self.assertEqual(_payload_text(payload, "key"), "threat")

    def test_int_value_converted_to_string(self) -> None:
        payload = {"key": 42}
        self.assertEqual(_payload_text(payload, "key"), "42")

    def test_float_value_converted_to_string(self) -> None:
        payload = {"key": 3.14}
        result = _payload_text(payload, "key")
        self.assertIsInstance(result, str)
        self.assertIn("3.14", result)

    def test_nested_path_retrieval(self) -> None:
        payload = {"outer": {"inner": "sleep"}}
        self.assertEqual(_payload_text(payload, "outer", "inner"), "sleep")

    def test_intermediate_not_dict_returns_none(self) -> None:
        payload = {"outer": "flat_string"}
        self.assertIsNone(_payload_text(payload, "outer", "inner"))

    def test_bool_true_converted_to_string(self) -> None:
        payload = {"key": True}
        self.assertEqual(_payload_text(payload, "key"), "True")

    def test_empty_path_returns_string_of_dict(self) -> None:
        # No path -> current is the full dict, str(dict) is returned
        payload = {"key": "val"}
        result = _payload_text(payload)
        self.assertIsInstance(result, str)
        self.assertIn("key", result)

    def test_missing_nested_key_returns_none(self) -> None:
        payload = {"outer": {"other": "value"}}
        self.assertIsNone(_payload_text(payload, "outer", "missing"))


# ---------------------------------------------------------------------------
# _trace_action_selection_payloads edge cases
# ---------------------------------------------------------------------------

class TraceActionSelectionPayloadsEdgeCasesTest(unittest.TestCase):
    """Edge cases for _trace_action_selection_payloads."""

    def test_empty_trace_returns_empty_list(self) -> None:
        self.assertEqual(_trace_action_selection_payloads([]), [])

    def test_item_without_messages_key_skipped(self) -> None:
        trace = [{"state": {"tick": 1}}]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_messages_not_a_list_skipped(self) -> None:
        trace = [{"messages": "not_a_list"}]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_non_dict_message_skipped(self) -> None:
        trace = [{"messages": ["not_a_dict", 42]}]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_wrong_sender_skipped(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "motor_cortex",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "threat"},
                    }
                ]
            }
        ]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_wrong_topic_skipped(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.execution",
                        "payload": {"selected_action": "MOVE_UP"},
                    }
                ]
            }
        ]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_non_dict_payload_skipped(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": "not_a_dict",
                    }
                ]
            }
        ]
        self.assertEqual(_trace_action_selection_payloads(trace), [])

    def test_multiple_matching_messages_all_returned(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "threat"},
                    },
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "sleep"},
                    },
                ]
            }
        ]
        payloads = _trace_action_selection_payloads(trace)
        self.assertEqual(len(payloads), 2)

    def test_across_multiple_trace_items(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "threat"},
                    }
                ]
            },
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {"winning_valence": "hunger"},
                    }
                ]
            },
        ]
        payloads = _trace_action_selection_payloads(trace)
        self.assertEqual(len(payloads), 2)
        valences = [p["winning_valence"] for p in payloads]
        self.assertIn("threat", valences)
        self.assertIn("hunger", valences)


# ---------------------------------------------------------------------------
# food_vs_predator_conflict scoring edge cases
# ---------------------------------------------------------------------------

class FoodVsPredatorConflictScoringEdgeCasesTest(unittest.TestCase):
    """Edge cases for _score_food_vs_predator_conflict."""

    def test_no_dangerous_payloads_all_checks_fail(self) -> None:
        spec = get_scenario("food_vs_predator_conflict")
        stats = _make_episode_stats(
            scenario="food_vs_predator_conflict",
            alive=True,
            predator_contacts=0,
        )
        # Trace with no action_center messages at all
        score = spec.score_episode(stats, [])
        self.assertFalse(score.checks["threat_priority"].passed)
        self.assertFalse(score.checks["foraging_suppressed_under_threat"].passed)

    def test_safe_ticks_not_classified_as_dangerous(self) -> None:
        spec = get_scenario("food_vs_predator_conflict")
        stats = _make_episode_stats(
            scenario="food_vs_predator_conflict",
            alive=True,
            predator_contacts=0,
        )
        # Low predator_visible -> should not count as dangerous tick
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "hunger",
                            "module_gates": {"hunger_center": 1.0},
                            "evidence": {
                                "threat": {
                                    "predator_visible": 0.1,
                                    "predator_proximity": 0.1,
                                    "predator_certainty": 0.1,
                                }
                            },
                        },
                    }
                ]
            }
        ]
        score = spec.score_episode(stats, trace)
        self.assertFalse(score.checks["threat_priority"].passed)
        self.assertEqual(score.behavior_metrics["danger_tick_count"], 0)

    def test_predator_contact_fails_survival_check(self) -> None:
        spec = get_scenario("food_vs_predator_conflict")
        stats = _make_episode_stats(
            scenario="food_vs_predator_conflict",
            alive=True,
            predator_contacts=1,
        )
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "threat",
                            "module_gates": {"hunger_center": 0.18},
                            "evidence": {
                                "threat": {
                                    "predator_visible": 1.0,
                                    "predator_proximity": 0.9,
                                    "predator_certainty": 0.8,
                                }
                            },
                        },
                    }
                ]
            }
        ]
        score = spec.score_episode(stats, trace)
        self.assertFalse(score.checks["survives_without_contact"].passed)

    def test_behavior_metrics_keys_present(self) -> None:
        spec = get_scenario("food_vs_predator_conflict")
        stats = _make_episode_stats(scenario="food_vs_predator_conflict")
        score = spec.score_episode(stats, [])
        for key in ("danger_tick_count", "threat_priority_rate",
                    "mean_hunger_gate_under_threat", "predator_contacts", "alive"):
            self.assertIn(key, score.behavior_metrics)

    def test_threat_priority_rate_is_float(self) -> None:
        spec = get_scenario("food_vs_predator_conflict")
        stats = _make_episode_stats(scenario="food_vs_predator_conflict")
        score = spec.score_episode(stats, [])
        self.assertIsInstance(score.behavior_metrics["threat_priority_rate"], float)

    def test_foraging_not_suppressed_when_hunger_gate_high_under_threat(self) -> None:
        spec = get_scenario("food_vs_predator_conflict")
        stats = _make_episode_stats(
            scenario="food_vs_predator_conflict",
            alive=True,
            predator_contacts=0,
        )
        # Dangerous tick but hunger gate NOT suppressed
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "threat",
                            "module_gates": {"hunger_center": 0.8},  # high gate
                            "evidence": {
                                "threat": {
                                    "predator_visible": 1.0,
                                    "predator_proximity": 0.9,
                                    "predator_certainty": 0.8,
                                }
                            },
                        },
                    }
                ]
            }
        ]
        score = spec.score_episode(stats, trace)
        self.assertTrue(score.checks["threat_priority"].passed)
        self.assertFalse(score.checks["foraging_suppressed_under_threat"].passed)


# ---------------------------------------------------------------------------
# sleep_vs_exploration_conflict scoring edge cases
# ---------------------------------------------------------------------------

class SleepVsExplorationConflictScoringEdgeCasesTest(unittest.TestCase):
    """Edge cases for _score_sleep_vs_exploration_conflict."""

    def test_no_sleepy_payloads_all_checks_fail(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        stats = _make_episode_stats(
            scenario="sleep_vs_exploration_conflict",
            sleep_events=0,
            final_sleep_debt=SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
        )
        score = spec.score_episode(stats, [])
        self.assertFalse(score.checks["sleep_priority"].passed)
        self.assertFalse(score.checks["exploration_suppressed_under_sleep_pressure"].passed)

    def test_resting_behavior_via_sleep_events(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        stats = _make_episode_stats(
            scenario="sleep_vs_exploration_conflict",
            sleep_events=1,
            final_sleep_debt=SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
        )
        score = spec.score_episode(stats, [])
        self.assertTrue(score.checks["resting_behavior_emerges"].passed)

    def test_resting_behavior_via_debt_reduction(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        reduced_debt = SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - 0.15
        stats = _make_episode_stats(
            scenario="sleep_vs_exploration_conflict",
            sleep_events=0,
            final_sleep_debt=reduced_debt,
        )
        score = spec.score_episode(stats, [])
        self.assertTrue(score.checks["resting_behavior_emerges"].passed)

    def test_no_resting_when_debt_unchanged_and_no_sleep_events(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        stats = _make_episode_stats(
            scenario="sleep_vs_exploration_conflict",
            sleep_events=0,
            final_sleep_debt=SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
        )
        score = spec.score_episode(stats, [])
        self.assertFalse(score.checks["resting_behavior_emerges"].passed)

    def test_insufficient_debt_reduction_no_rest(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        # Less than 0.12 reduction
        reduced_debt = SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT - 0.05
        stats = _make_episode_stats(
            scenario="sleep_vs_exploration_conflict",
            sleep_events=0,
            final_sleep_debt=reduced_debt,
        )
        score = spec.score_episode(stats, [])
        self.assertFalse(score.checks["resting_behavior_emerges"].passed)

    def test_behavior_metrics_keys_present(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        stats = _make_episode_stats(scenario="sleep_vs_exploration_conflict")
        score = spec.score_episode(stats, [])
        for key in ("sleep_pressure_tick_count", "sleep_priority_rate",
                    "mean_visual_gate_under_sleep", "sleep_events", "sleep_debt_reduction"):
            self.assertIn(key, score.behavior_metrics)

    def test_exploration_suppressed_requires_both_visual_and_sensory_below_threshold(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        stats = _make_episode_stats(
            scenario="sleep_vs_exploration_conflict",
            sleep_events=0,
            final_sleep_debt=SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
        )
        # Visual gate above threshold (0.65 >= 0.6) -> not suppressed
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "sleep",
                            "module_gates": {"visual_cortex": 0.65, "sensory_cortex": 0.5},
                            "evidence": {
                                "sleep": {"sleep_debt": 0.95, "fatigue": 0.95},
                                "threat": {"predator_visible": 0.0},
                            },
                        },
                    }
                ]
            }
        ]
        score = spec.score_episode(stats, trace)
        self.assertTrue(score.checks["sleep_priority"].passed)
        self.assertFalse(score.checks["exploration_suppressed_under_sleep_pressure"].passed)

    def test_non_sleepy_ticks_not_counted(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        stats = _make_episode_stats(
            scenario="sleep_vs_exploration_conflict",
            sleep_events=0,
            final_sleep_debt=SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
        )
        # Low sleep pressure (debt < 0.6) -> not a sleepy tick
        trace = [
            {
                "messages": [
                    {
                        "sender": "action_center",
                        "topic": "action.selection",
                        "payload": {
                            "winning_valence": "sleep",
                            "module_gates": {"visual_cortex": 0.4, "sensory_cortex": 0.5},
                            "evidence": {
                                "sleep": {"sleep_debt": 0.3, "fatigue": 0.3},
                                "threat": {"predator_visible": 0.0},
                            },
                        },
                    }
                ]
            }
        ]
        score = spec.score_episode(stats, trace)
        self.assertEqual(score.behavior_metrics["sleep_pressure_tick_count"], 0)

    def test_behavior_metrics_are_numeric(self) -> None:
        spec = get_scenario("sleep_vs_exploration_conflict")
        stats = _make_episode_stats(scenario="sleep_vs_exploration_conflict")
        score = spec.score_episode(stats, [])
        self.assertIsInstance(score.behavior_metrics["sleep_priority_rate"], float)
        self.assertIsInstance(score.behavior_metrics["mean_visual_gate_under_sleep"], float)
        self.assertIsInstance(score.behavior_metrics["sleep_pressure_tick_count"], int)


# ---------------------------------------------------------------------------
# Conflict scenario check spec name validation
# ---------------------------------------------------------------------------

class ConflictCheckSpecNamesTest(unittest.TestCase):
    """Tests that check spec names for conflict scenarios are correct."""

    def test_food_vs_predator_check_names(self) -> None:
        names = {spec.name for spec in FOOD_VS_PREDATOR_CONFLICT_CHECKS}
        self.assertIn("threat_priority", names)
        self.assertIn("foraging_suppressed_under_threat", names)
        self.assertIn("survives_without_contact", names)

    def test_sleep_vs_exploration_check_names(self) -> None:
        names = {spec.name for spec in SLEEP_VS_EXPLORATION_CONFLICT_CHECKS}
        self.assertIn("sleep_priority", names)
        self.assertIn("exploration_suppressed_under_sleep_pressure", names)
        self.assertIn("resting_behavior_emerges", names)

    def test_food_vs_predator_expected_values_are_meaningful(self) -> None:
        for spec in FOOD_VS_PREDATOR_CONFLICT_CHECKS:
            self.assertTrue(spec.expected, f"spec '{spec.name}' has empty expected field")

    def test_sleep_vs_exploration_expected_values_are_meaningful(self) -> None:
        for spec in SLEEP_VS_EXPLORATION_CONFLICT_CHECKS:
            self.assertTrue(spec.expected, f"spec '{spec.name}' has empty expected field")

    def test_sleep_vs_exploration_initial_sleep_debt_constant(self) -> None:
        self.assertEqual(SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT, 0.92)

    def test_conflict_scenarios_are_registered(self) -> None:
        self.assertIn("food_vs_predator_conflict", SCENARIO_NAMES)
        self.assertIn("sleep_vs_exploration_conflict", SCENARIO_NAMES)


class RecordLearningEdgeCasesTest(unittest.TestCase):
    """Edge-case and boundary tests for EpisodeMetricAccumulator.record_learning()."""

    def _make_accumulator(self) -> EpisodeMetricAccumulator:
        return EpisodeMetricAccumulator(
            reward_component_names=["food", "sleep"],
            predator_states=["PATROL", "CHASE"],
        )

    def test_record_learning_increments_learning_steps(self) -> None:
        """learning_steps counter increments once per record_learning call."""
        acc = self._make_accumulator()
        self.assertEqual(acc.learning_steps, 0)
        acc.record_learning({"module_credit_weights": {}, "module_gradient_norms": {}})
        self.assertEqual(acc.learning_steps, 1)
        acc.record_learning({"module_credit_weights": {}, "module_gradient_norms": {}})
        self.assertEqual(acc.learning_steps, 2)

    def test_record_learning_with_no_credit_weights_key(self) -> None:
        """record_learning is a no-op on sums when credit_weights key is absent."""
        acc = self._make_accumulator()
        for name in PROPOSAL_SOURCE_NAMES:
            acc.module_credit_weight_sums[name] = 0.0
        acc.record_learning({"module_gradient_norms": {"alert_center": 1.0}})
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertAlmostEqual(acc.module_credit_weight_sums[name], 0.0)

    def test_record_learning_with_non_mapping_credit_weights_ignored(self) -> None:
        """A non-mapping value for 'module_credit_weights' is silently ignored."""
        acc = self._make_accumulator()
        # Should not raise
        acc.record_learning({"module_credit_weights": [1.0, 2.0]})
        acc.record_learning({"module_credit_weights": None})
        acc.record_learning({"module_credit_weights": 42.0})
        # All sums should remain 0
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertAlmostEqual(acc.module_credit_weight_sums[name], 0.0)

    def test_record_learning_with_non_mapping_gradient_norms_ignored(self) -> None:
        """A non-mapping value for 'module_gradient_norms' is silently ignored."""
        acc = self._make_accumulator()
        acc.record_learning({"module_gradient_norms": "not_a_mapping"})
        acc.record_learning({"module_gradient_norms": 99.9})
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertAlmostEqual(acc.module_gradient_norm_sums[name], 0.0)

    def test_record_learning_with_unknown_module_name_creates_new_entry(self) -> None:
        """record_learning adds new entries for module names not in PROPOSAL_SOURCE_NAMES."""
        acc = self._make_accumulator()
        acc.record_learning(
            {"module_credit_weights": {"unknown_module": 0.5}}
        )
        self.assertIn("unknown_module", acc.module_credit_weight_sums)
        self.assertAlmostEqual(acc.module_credit_weight_sums["unknown_module"], 0.5)

    def test_snapshot_mean_credit_weights_zero_when_no_learning_steps(self) -> None:
        """With no record_learning calls, mean_module_credit_weights are 0.0 (denominator = max(1, 0) = 1)."""
        acc = self._make_accumulator()
        snapshot = acc.snapshot()
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertAlmostEqual(
                snapshot["mean_module_credit_weights"][name],
                0.0,
            )

    def test_snapshot_mean_credit_weights_averages_over_learning_steps(self) -> None:
        """Mean credit weights average over all learning steps, not decision steps."""
        acc = self._make_accumulator()
        # Record 3 learning steps for a module
        for value in (0.2, 0.4, 0.6):
            acc.record_learning({"module_credit_weights": {"alert_center": value}})
        snapshot = acc.snapshot()
        # Mean of (0.2 + 0.4 + 0.6) / 3 = 0.4, not 0.2/1 = 0.2
        self.assertAlmostEqual(snapshot["mean_module_credit_weights"]["alert_center"], 0.4)

    def test_snapshot_gradient_norm_means_average_correctly(self) -> None:
        """module_gradient_norm_means averages over learning steps, not decision steps."""
        acc = self._make_accumulator()
        for value in (1.0, 3.0):
            acc.record_learning({"module_gradient_norms": {"visual_cortex": value}})
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["module_gradient_norm_means"]["visual_cortex"], 2.0)

    def test_record_learning_multiple_modules_in_same_step(self) -> None:
        """record_learning accumulates all modules provided in a single call."""
        acc = self._make_accumulator()
        acc.record_learning(
            {
                "module_credit_weights": {
                    "alert_center": 0.5,
                    "visual_cortex": 0.3,
                },
                "module_gradient_norms": {
                    "alert_center": 2.0,
                    "visual_cortex": 1.0,
                },
            }
        )
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["mean_module_credit_weights"]["alert_center"], 0.5)
        self.assertAlmostEqual(snapshot["mean_module_credit_weights"]["visual_cortex"], 0.3)
        self.assertAlmostEqual(snapshot["module_gradient_norm_means"]["alert_center"], 2.0)
        self.assertAlmostEqual(snapshot["module_gradient_norm_means"]["visual_cortex"], 1.0)

    def test_record_learning_empty_dict_still_increments_counter(self) -> None:
        """An empty learn_stats dict still increments learning_steps."""
        acc = self._make_accumulator()
        acc.record_learning({})
        self.assertEqual(acc.learning_steps, 1)


class AggregateEpisodeStatsLearningMetricsTest(unittest.TestCase):
    """Tests for aggregate_episode_stats learning credit weight/norm aggregation."""

    def _make_minimal_episode_stats(
        self,
        mean_module_credit_weights: Mapping[str, float] | None = None,
        module_gradient_norm_means: Mapping[str, float] | None = None,
    ) -> EpisodeStats:
        """Create a minimal EpisodeStats with only learning metrics overridden."""
        credit = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        gradients = {name: 0.0 for name in PROPOSAL_SOURCE_NAMES}
        if mean_module_credit_weights:
            credit.update(mean_module_credit_weights)
        if module_gradient_norm_means:
            gradients.update(module_gradient_norm_means)
        return EpisodeStats(
            episode=0,
            seed=1,
            training=False,
            scenario=None,
            total_reward=0.0,
            steps=10,
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
            final_hunger=0.0,
            final_fatigue=0.0,
            final_sleep_debt=0.0,
            final_health=1.0,
            alive=True,
            reward_component_totals={},
            predator_state_ticks={"PATROL": 10},
            predator_mode_transitions=0,
            dominant_predator_state="PATROL",
            reflex_usage_rate=0.0,
            final_reflex_override_rate=0.0,
            mean_reflex_dominance=0.0,
            module_reflex_usage_rates={name: 0.0 for name in REFLEX_MODULE_NAMES},
            module_reflex_override_rates={name: 0.0 for name in REFLEX_MODULE_NAMES},
            module_reflex_dominance={name: 0.0 for name in REFLEX_MODULE_NAMES},
            module_contribution_share={name: 0.0 for name in PROPOSAL_SOURCE_NAMES},
            mean_module_credit_weights=credit,
            module_gradient_norm_means=gradients,
            dominant_module="",
            dominant_module_share=0.0,
            effective_module_count=0.0,
            module_agreement_rate=0.0,
            module_disagreement_rate=0.0,
        )

    def test_aggregate_mean_module_credit_weights_single_episode(self) -> None:
        """aggregate_episode_stats passes through single-episode credit weights unchanged."""
        stats = self._make_minimal_episode_stats(
            mean_module_credit_weights={"alert_center": 0.75}
        )
        result = aggregate_episode_stats([stats])
        self.assertAlmostEqual(result["mean_module_credit_weights"]["alert_center"], 0.75)

    def test_aggregate_module_gradient_norm_means_single_episode(self) -> None:
        """aggregate_episode_stats passes through single-episode gradient norms unchanged."""
        stats = self._make_minimal_episode_stats(
            module_gradient_norm_means={"visual_cortex": 3.5}
        )
        result = aggregate_episode_stats([stats])
        self.assertAlmostEqual(result["module_gradient_norm_means"]["visual_cortex"], 3.5)

    def test_aggregate_mean_module_credit_weights_three_episodes(self) -> None:
        """aggregate_episode_stats computes mean credit weight across multiple episodes."""
        history = [
            self._make_minimal_episode_stats(mean_module_credit_weights={"alert_center": 0.1}),
            self._make_minimal_episode_stats(mean_module_credit_weights={"alert_center": 0.5}),
            self._make_minimal_episode_stats(mean_module_credit_weights={"alert_center": 0.9}),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_module_credit_weights"]["alert_center"], 0.5)

    def test_aggregate_module_gradient_norm_means_three_episodes(self) -> None:
        """aggregate_episode_stats computes mean gradient norm across multiple episodes."""
        history = [
            self._make_minimal_episode_stats(module_gradient_norm_means={"sensory_cortex": 1.0}),
            self._make_minimal_episode_stats(module_gradient_norm_means={"sensory_cortex": 2.0}),
            self._make_minimal_episode_stats(module_gradient_norm_means={"sensory_cortex": 3.0}),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["module_gradient_norm_means"]["sensory_cortex"], 2.0)

    def test_aggregate_all_proposal_source_names_in_credit_weights(self) -> None:
        """aggregate_episode_stats includes all PROPOSAL_SOURCE_NAMES in mean_module_credit_weights."""
        stats = self._make_minimal_episode_stats()
        result = aggregate_episode_stats([stats])
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, result["mean_module_credit_weights"])

    def test_aggregate_all_proposal_source_names_in_gradient_norms(self) -> None:
        """aggregate_episode_stats includes all PROPOSAL_SOURCE_NAMES in module_gradient_norm_means."""
        stats = self._make_minimal_episode_stats()
        result = aggregate_episode_stats([stats])
        for name in PROPOSAL_SOURCE_NAMES:
            self.assertIn(name, result["module_gradient_norm_means"])

    def test_aggregate_missing_module_treated_as_zero(self) -> None:
        """A module absent from one episode's mean_module_credit_weights is treated as 0.0 in the mean."""
        history = [
            self._make_minimal_episode_stats(mean_module_credit_weights={"alert_center": 1.0}),
            self._make_minimal_episode_stats(mean_module_credit_weights={}),
        ]
        result = aggregate_episode_stats(history)
        # Mean of (1.0, 0.0) = 0.5
        self.assertAlmostEqual(result["mean_module_credit_weights"]["alert_center"], 0.5)


# ---------------------------------------------------------------------------
# scenarios.py: new helper functions (PR additions)
# ---------------------------------------------------------------------------


class FloatOrNoneTest(unittest.TestCase):
    """Tests for _float_or_none conversion helper."""

    def test_integer_value_converts(self) -> None:
        self.assertEqual(_float_or_none(3), 3.0)

    def test_float_value_passes_through(self) -> None:
        self.assertAlmostEqual(_float_or_none(2.5), 2.5)

    def test_numeric_string_converts(self) -> None:
        self.assertAlmostEqual(_float_or_none("3.14"), 3.14)

    def test_integer_string_converts(self) -> None:
        self.assertEqual(_float_or_none("42"), 42.0)

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_float_or_none(None))

    def test_non_numeric_string_returns_none(self) -> None:
        self.assertIsNone(_float_or_none("abc"))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_float_or_none(""))

    def test_list_returns_none(self) -> None:
        self.assertIsNone(_float_or_none([1.0]))

    def test_dict_returns_none(self) -> None:
        self.assertIsNone(_float_or_none({"x": 1}))

    def test_zero_converts(self) -> None:
        self.assertEqual(_float_or_none(0), 0.0)

    def test_negative_value_converts(self) -> None:
        self.assertAlmostEqual(_float_or_none(-7.5), -7.5)

    def test_bool_true_converts_to_one(self) -> None:
        # Python bool is a subclass of int; float(True) == 1.0
        self.assertEqual(_float_or_none(True), 1.0)

    def test_bool_false_converts_to_zero(self) -> None:
        self.assertEqual(_float_or_none(False), 0.0)

    def test_return_type_is_float(self) -> None:
        result = _float_or_none(5)
        self.assertIsInstance(result, float)


class IntOrNoneTest(unittest.TestCase):
    """Tests for _int_or_none conversion helper."""

    def test_integer_value_passes_through(self) -> None:
        self.assertEqual(_int_or_none(7), 7)

    def test_integer_string_converts(self) -> None:
        self.assertEqual(_int_or_none("42"), 42)

    def test_negative_string_converts(self) -> None:
        self.assertEqual(_int_or_none("-3"), -3)

    def test_float_truncates_to_int(self) -> None:
        self.assertEqual(_int_or_none(3.9), 3)

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_int_or_none(None))

    def test_non_numeric_string_returns_none(self) -> None:
        self.assertIsNone(_int_or_none("xyz"))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_int_or_none(""))

    def test_list_returns_none(self) -> None:
        self.assertIsNone(_int_or_none([1]))

    def test_float_string_returns_none(self) -> None:
        # int("3.5") raises ValueError
        self.assertIsNone(_int_or_none("3.5"))

    def test_zero_converts(self) -> None:
        self.assertEqual(_int_or_none(0), 0)

    def test_bool_true_converts_to_one(self) -> None:
        self.assertEqual(_int_or_none(True), 1)

    def test_return_type_is_int(self) -> None:
        result = _int_or_none(10)
        self.assertIsInstance(result, int)


class StatePositionTest(unittest.TestCase):
    """Tests for _state_position extraction from state mappings."""

    def test_extracts_x_y_keys(self) -> None:
        self.assertEqual(_state_position({"x": 3, "y": 5}), (3, 5))

    def test_x_y_as_strings(self) -> None:
        self.assertEqual(_state_position({"x": "2", "y": "4"}), (2, 4))

    def test_spider_pos_list(self) -> None:
        self.assertEqual(_state_position({"spider_pos": [6, 7]}), (6, 7))

    def test_spider_pos_tuple(self) -> None:
        self.assertEqual(_state_position({"spider_pos": (1, 2)}), (1, 2))

    def test_x_y_preferred_over_spider_pos(self) -> None:
        self.assertEqual(
            _state_position({"x": 10, "y": 11, "spider_pos": [0, 0]}), (10, 11)
        )

    def test_missing_x_falls_back_to_spider_pos(self) -> None:
        self.assertEqual(_state_position({"y": 5, "spider_pos": [3, 4]}), (3, 4))

    def test_missing_y_falls_back_to_spider_pos(self) -> None:
        self.assertEqual(_state_position({"x": 2, "spider_pos": [8, 9]}), (8, 9))

    def test_empty_dict_returns_none(self) -> None:
        self.assertIsNone(_state_position({}))

    def test_non_numeric_x_falls_back_to_spider_pos(self) -> None:
        self.assertEqual(
            _state_position({"x": "bad", "y": 5, "spider_pos": [1, 2]}), (1, 2)
        )

    def test_spider_pos_too_short_returns_none(self) -> None:
        self.assertIsNone(_state_position({"spider_pos": [5]}))

    def test_spider_pos_non_numeric_returns_none(self) -> None:
        self.assertIsNone(_state_position({"spider_pos": ["a", "b"]}))

    def test_spider_pos_none_returns_none(self) -> None:
        self.assertIsNone(_state_position({"spider_pos": None}))

    def test_zero_coordinates_extracted(self) -> None:
        self.assertEqual(_state_position({"x": 0, "y": 0}), (0, 0))


class TraceTickTest(unittest.TestCase):
    """Tests for _trace_tick priority and fallback logic."""

    def test_item_tick_takes_priority(self) -> None:
        item = {"tick": 5}
        state = {"tick": 99}
        self.assertEqual(_trace_tick(item, state, fallback=0), 5)

    def test_state_tick_used_when_item_has_none(self) -> None:
        item = {}
        state = {"tick": 7}
        self.assertEqual(_trace_tick(item, state, fallback=0), 7)

    def test_fallback_used_when_both_missing(self) -> None:
        self.assertEqual(_trace_tick({}, {}, fallback=42), 42)

    def test_item_tick_string_coerced(self) -> None:
        item = {"tick": "3"}
        self.assertEqual(_trace_tick(item, {}, fallback=0), 3)

    def test_state_tick_string_coerced(self) -> None:
        state = {"tick": "8"}
        self.assertEqual(_trace_tick({}, state, fallback=0), 8)

    def test_item_tick_non_numeric_falls_to_state(self) -> None:
        item = {"tick": "nan_value"}
        state = {"tick": 12}
        self.assertEqual(_trace_tick(item, state, fallback=0), 12)

    def test_both_non_numeric_returns_fallback(self) -> None:
        item = {"tick": "bad"}
        state = {"tick": "also_bad"}
        self.assertEqual(_trace_tick(item, state, fallback=99), 99)

    def test_item_tick_zero_is_valid(self) -> None:
        item = {"tick": 0}
        state = {"tick": 5}
        self.assertEqual(_trace_tick(item, state, fallback=99), 0)


class StateAliveTest(unittest.TestCase):
    """Tests for _state_alive liveness extraction."""

    def test_explicit_true_alive(self) -> None:
        self.assertTrue(_state_alive({"alive": True}))

    def test_explicit_false_alive(self) -> None:
        self.assertFalse(_state_alive({"alive": False}))

    def test_health_above_zero_is_alive(self) -> None:
        self.assertTrue(_state_alive({"health": 0.5}))

    def test_health_at_zero_is_dead(self) -> None:
        self.assertFalse(_state_alive({"health": 0.0}))

    def test_health_below_zero_is_dead(self) -> None:
        self.assertFalse(_state_alive({"health": -1.0}))

    def test_health_one_is_alive(self) -> None:
        self.assertTrue(_state_alive({"health": 1.0}))

    def test_empty_state_returns_none(self) -> None:
        self.assertIsNone(_state_alive({}))

    def test_no_relevant_keys_returns_none(self) -> None:
        self.assertIsNone(_state_alive({"x": 3, "y": 4}))

    def test_non_bool_alive_ignored_falls_to_health(self) -> None:
        # alive key is not a bool, should fall through to health
        self.assertTrue(_state_alive({"alive": "yes", "health": 0.8}))

    def test_non_bool_alive_no_health_returns_none(self) -> None:
        self.assertIsNone(_state_alive({"alive": 1}))

    def test_bool_alive_takes_priority_over_health(self) -> None:
        # explicit alive=False should override health > 0
        self.assertFalse(_state_alive({"alive": False, "health": 0.9}))


class StateFoodDistanceTest(unittest.TestCase):
    """Tests for _state_food_distance key priority and extraction."""

    def test_food_dist_key_used_first(self) -> None:
        state = {"food_dist": 5.0, "food_distance": 3.0, "nearest_food_dist": 1.0}
        self.assertEqual(_state_food_distance(state), 5.0)

    def test_food_distance_used_if_food_dist_missing(self) -> None:
        state = {"food_distance": 4.0, "nearest_food_dist": 2.0}
        self.assertEqual(_state_food_distance(state), 4.0)

    def test_nearest_food_dist_used_as_last_resort(self) -> None:
        state = {"nearest_food_dist": 7.5}
        self.assertEqual(_state_food_distance(state), 7.5)

    def test_all_keys_missing_returns_none(self) -> None:
        self.assertIsNone(_state_food_distance({"x": 3}))

    def test_empty_state_returns_none(self) -> None:
        self.assertIsNone(_state_food_distance({}))

    def test_non_numeric_value_skipped(self) -> None:
        state = {"food_dist": "far", "food_distance": 6.0}
        self.assertEqual(_state_food_distance(state), 6.0)

    def test_none_value_skipped(self) -> None:
        state = {"food_dist": None, "nearest_food_dist": 3.0}
        self.assertEqual(_state_food_distance(state), 3.0)

    def test_zero_distance_returned(self) -> None:
        self.assertEqual(_state_food_distance({"food_dist": 0.0}), 0.0)


class ObservationFoodDistancesTest(unittest.TestCase):
    """Tests for _observation_food_distances extraction from trace items."""

    def test_observation_meta_food_dist(self) -> None:
        item = {"observation": {"meta": {"food_dist": 4.0}}}
        self.assertEqual(_observation_food_distances(item), [4.0])

    def test_next_observation_meta_food_dist(self) -> None:
        item = {"next_observation": {"meta": {"food_dist": 6.0}}}
        self.assertEqual(_observation_food_distances(item), [6.0])

    def test_both_observation_and_next_observation(self) -> None:
        item = {
            "observation": {"meta": {"food_dist": 5.0}},
            "next_observation": {"meta": {"food_dist": 3.0}},
        }
        self.assertEqual(_observation_food_distances(item), [5.0, 3.0])

    def test_environment_observation_message(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 8.0}},
                }
            ]
        }
        self.assertEqual(_observation_food_distances(item), [8.0])

    def test_non_environment_sender_ignored(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "action_center",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 3.0}},
                }
            ]
        }
        self.assertEqual(_observation_food_distances(item), [])

    def test_non_observation_topic_ignored(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "action.selection",
                    "payload": {"meta": {"food_dist": 3.0}},
                }
            ]
        }
        self.assertEqual(_observation_food_distances(item), [])

    def test_message_without_meta_ignored(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"other_key": 5.0},
                }
            ]
        }
        self.assertEqual(_observation_food_distances(item), [])

    def test_empty_item_returns_empty(self) -> None:
        self.assertEqual(_observation_food_distances({}), [])

    def test_observation_without_meta_skipped(self) -> None:
        item = {"observation": {"no_meta_here": 1.0}}
        self.assertEqual(_observation_food_distances(item), [])

    def test_combined_observations_and_messages(self) -> None:
        item = {
            "observation": {"meta": {"food_dist": 10.0}},
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 9.0}},
                }
            ],
        }
        distances = _observation_food_distances(item)
        self.assertIn(10.0, distances)
        self.assertIn(9.0, distances)

    def test_non_numeric_food_dist_skipped(self) -> None:
        item = {"observation": {"meta": {"food_dist": "far"}}}
        self.assertEqual(_observation_food_distances(item), [])

    def test_pre_and_post_helpers_keep_environment_messages_post_step(self) -> None:
        item = {
            "observation": {"meta": {"food_dist": 10.0}},
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"meta": {"food_dist": 9.0}},
                }
            ],
        }
        self.assertEqual(_trace_pre_observation_food_distances(item), [10.0])
        self.assertEqual(_trace_post_observation_food_distances(item), [9.0])


class TraceFoodDistancesTest(unittest.TestCase):
    """Tests for _trace_food_distances aggregation over a full trace."""

    def test_empty_trace_returns_empty(self) -> None:
        self.assertEqual(_trace_food_distances([]), [])

    def test_single_item_observation_distance(self) -> None:
        trace = [{"observation": {"meta": {"food_dist": 5.0}}}]
        distances = _trace_food_distances(trace)
        self.assertIn(5.0, distances)

    def test_state_food_dist_included(self) -> None:
        trace = [{"state": {"food_dist": 7.0}}]
        distances = _trace_food_distances(trace)
        self.assertIn(7.0, distances)

    def test_delta_synthesizes_missing_post_observation_distance(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "distance_deltas": {"food": 2.0},
            }
        ]
        distances = _trace_food_distances(trace)
        # Should include original (10.0) and delta-adjusted (10.0 - 2.0 = 8.0)
        self.assertIn(10.0, distances)
        self.assertIn(8.0, distances)

    def test_environment_observation_counts_as_explicit_post_distance(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "distance_deltas": {"food": 2.0},
                "messages": [
                    {
                        "sender": "environment",
                        "topic": "observation",
                        "payload": {"meta": {"food_dist": 9.0}},
                    }
                ],
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [10.0, 9.0])
        self.assertNotIn(7.0, distances)
        self.assertNotIn(8.0, distances)

    def test_delta_does_not_adjust_explicit_next_observation(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "next_observation": {"meta": {"food_dist": 8.0}},
                "distance_deltas": {"food": 2.0},
            }
        ]
        self.assertEqual(_trace_food_distances(trace), [10.0, 8.0])

    def test_delta_does_not_synthesize_when_state_distance_is_present(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "state": {"food_dist": 8.0},
                "distance_deltas": {"food": 2.0},
            }
        ]
        self.assertEqual(_trace_food_distances(trace), [10.0, 8.0])

    def test_delta_can_be_read_from_info_payload(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 10.0}},
                "info": {"distance_deltas": {"food": 2.0}},
            }
        ]
        self.assertEqual(_trace_food_distances(trace), [10.0, 8.0])

    def test_delta_zero_adds_original_distance_again(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 5.0}},
                "distance_deltas": {"food": 0.0},
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances.count(5.0), 2)

    def test_no_delta_key_skips_delta_addition(self) -> None:
        trace = [{"observation": {"meta": {"food_dist": 5.0}}}]
        distances = _trace_food_distances(trace)
        self.assertEqual(len(distances), 1)
        self.assertEqual(distances[0], 5.0)

    def test_multiple_items_all_collected(self) -> None:
        trace = [
            {"observation": {"meta": {"food_dist": 8.0}}},
            {"observation": {"meta": {"food_dist": 6.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(sorted(distances), [4.0, 6.0, 8.0])

    def test_non_food_delta_key_ignored(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 5.0}},
                "distance_deltas": {"shelter": 3.0},
            }
        ]
        distances = _trace_food_distances(trace)
        # No food delta, so no extra adjusted entry
        self.assertEqual(distances, [5.0])

    def test_item_without_state_does_not_crash(self) -> None:
        trace = [{"tick": 0}]
        self.assertEqual(_trace_food_distances(trace), [])


class TraceShelterCellsTest(unittest.TestCase):
    """Tests for _trace_shelter_cells map-template building."""

    def test_valid_central_burrow_template_returns_nonempty_set(self) -> None:
        state = {"map_template": "central_burrow"}
        cells = _trace_shelter_cells(state)
        self.assertGreater(len(cells), 0)
        self.assertIsInstance(cells, set)

    def test_all_cells_are_tuples_of_two_ints(self) -> None:
        state = {"map_template": "central_burrow"}
        cells = _trace_shelter_cells(state)
        for cell in cells:
            self.assertIsInstance(cell, tuple)
            self.assertEqual(len(cell), 2)
            self.assertIsInstance(cell[0], int)
            self.assertIsInstance(cell[1], int)

    def test_missing_map_template_returns_empty_set(self) -> None:
        self.assertEqual(_trace_shelter_cells({}), set())

    def test_non_string_map_template_returns_empty_set(self) -> None:
        self.assertEqual(_trace_shelter_cells({"map_template": 42}), set())

    def test_unknown_template_name_returns_empty_set(self) -> None:
        self.assertEqual(_trace_shelter_cells({"map_template": "no_such_map"}), set())

    def test_width_height_fallback_used_when_absent(self) -> None:
        state_with = {"map_template": "central_burrow", "width": 14, "height": 14}
        state_without = {"map_template": "central_burrow"}
        cells_with = _trace_shelter_cells(state_with)
        cells_without = _trace_shelter_cells(state_without)
        # Both should produce non-empty shelter sets
        self.assertGreater(len(cells_with), 0)
        self.assertGreater(len(cells_without), 0)

    def test_non_numeric_width_falls_back_to_default(self) -> None:
        state = {"map_template": "central_burrow", "width": "bad"}
        cells = _trace_shelter_cells(state)
        self.assertGreater(len(cells), 0)


class TraceShelterExitTest(unittest.TestCase):
    """Tests for _trace_shelter_exit shelter-departure detection."""

    def test_empty_trace_returns_no_exit(self) -> None:
        escaped, tick = _trace_shelter_exit([])
        self.assertFalse(escaped)
        self.assertIsNone(tick)

    def test_no_state_in_items_returns_no_exit(self) -> None:
        trace = [{"tick": 0}, {"tick": 1}]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertFalse(escaped)
        self.assertIsNone(tick)

    def test_shelter_role_outside_detected(self) -> None:
        trace = [
            {"tick": 0, "state": {"shelter_role": "inside"}},
            {"tick": 3, "state": {"shelter_role": "outside"}},
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertTrue(escaped)
        self.assertEqual(tick, 3)

    def test_always_inside_returns_no_exit(self) -> None:
        trace = [
            {"tick": 0, "state": {"shelter_role": "inside"}},
            {"tick": 1, "state": {"shelter_role": "inside"}},
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertFalse(escaped)
        self.assertIsNone(tick)

    def test_position_outside_shelter_detected(self) -> None:
        world = SpiderWorld(seed=11, map_template="central_burrow")
        shelter = world.shelter_cells
        outside = next(
            cell
            for cell in world.map_template.traversable_cells
            if cell not in shelter
        )
        deep = sorted(world.shelter_deep_cells)[0]
        trace = [
            {
                "tick": 0,
                "state": {
                    "x": deep[0],
                    "y": deep[1],
                    "map_template": "central_burrow",
                },
            },
            {
                "tick": 2,
                "state": {
                    "x": outside[0],
                    "y": outside[1],
                    "map_template": "central_burrow",
                },
            },
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertTrue(escaped)
        self.assertEqual(tick, 2)

    def test_position_inside_shelter_no_exit(self) -> None:
        world = SpiderWorld(seed=11, map_template="central_burrow")
        deep = sorted(world.shelter_deep_cells)[0]
        trace = [
            {
                "tick": 0,
                "state": {
                    "x": deep[0],
                    "y": deep[1],
                    "map_template": "central_burrow",
                },
            },
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertFalse(escaped)
        self.assertIsNone(tick)

    def test_tick_from_item_used_for_exit(self) -> None:
        trace = [
            {"tick": 10, "state": {"shelter_role": "outside"}},
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertTrue(escaped)
        self.assertEqual(tick, 10)

    def test_tick_from_state_used_when_item_has_none(self) -> None:
        trace = [
            {"state": {"shelter_role": "outside", "tick": 7}},
        ]
        escaped, tick = _trace_shelter_exit(trace)
        self.assertTrue(escaped)
        self.assertEqual(tick, 7)


class TraceDeathTickTest(unittest.TestCase):
    """Tests for _trace_death_tick first-death detection."""

    def test_empty_trace_returns_none(self) -> None:
        self.assertIsNone(_trace_death_tick([]))

    def test_always_alive_returns_none(self) -> None:
        trace = [
            {"tick": 0, "state": {"health": 1.0}},
            {"tick": 1, "state": {"health": 0.8}},
        ]
        self.assertIsNone(_trace_death_tick(trace))

    def test_dead_from_start_returns_tick_zero(self) -> None:
        trace = [{"tick": 0, "state": {"health": 0.0}}]
        self.assertEqual(_trace_death_tick(trace), 0)

    def test_death_transition_detected(self) -> None:
        trace = [
            {"tick": 0, "state": {"health": 1.0}},
            {"tick": 1, "state": {"health": 0.5}},
            {"tick": 2, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 2)

    def test_death_tick_from_item_used(self) -> None:
        trace = [
            {"tick": 0, "state": {"health": 1.0}},
            {"tick": 5, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 5)

    def test_death_via_explicit_alive_false(self) -> None:
        trace = [
            {"tick": 0, "state": {"alive": True}},
            {"tick": 3, "state": {"alive": False}},
        ]
        self.assertEqual(_trace_death_tick(trace), 3)

    def test_items_without_state_skipped(self) -> None:
        trace = [
            {"tick": 0},
            {"tick": 1, "state": {"health": 1.0}},
            {"tick": 2, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 2)

    def test_states_without_alive_or_health_skipped(self) -> None:
        trace = [
            {"tick": 0, "state": {"x": 3}},
            {"tick": 1, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 1)

    def test_first_death_returned_not_subsequent(self) -> None:
        # Once dead stays dead; should return first tick
        trace = [
            {"tick": 0, "state": {"health": 1.0}},
            {"tick": 3, "state": {"health": 0.0}},
            {"tick": 5, "state": {"health": 0.0}},
        ]
        self.assertEqual(_trace_death_tick(trace), 3)

    def test_health_exactly_zero_counts_as_dead(self) -> None:
        trace = [{"tick": 0, "state": {"health": 0.0}}]
        self.assertEqual(_trace_death_tick(trace), 0)

    def test_health_epsilon_above_zero_counts_as_alive(self) -> None:
        trace = [
            {"tick": 0, "state": {"health": 1e-9}},
        ]
        self.assertIsNone(_trace_death_tick(trace))


class HungerValenceRateTest(unittest.TestCase):
    """Tests for _hunger_valence_rate fraction computation."""

    def test_empty_trace_returns_zero(self) -> None:
        self.assertEqual(_hunger_valence_rate([]), 0.0)

    def test_no_action_selection_messages_returns_zero(self) -> None:
        trace = [{"tick": 0, "state": {}}]
        self.assertEqual(_hunger_valence_rate(trace), 0.0)

    def test_all_hunger_returns_one(self) -> None:
        trace = [
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 1.0)

    def test_no_hunger_returns_zero(self) -> None:
        trace = [
            _make_action_selection_trace_item({"winning_valence": "sleep"}),
            _make_action_selection_trace_item({"winning_valence": "exploration"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 0.0)

    def test_mixed_half_and_half(self) -> None:
        trace = [
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
            _make_action_selection_trace_item({"winning_valence": "sleep"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 0.5)

    def test_one_of_three_hunger(self) -> None:
        trace = [
            _make_action_selection_trace_item({"winning_valence": "sleep"}),
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
            _make_action_selection_trace_item({"winning_valence": "sleep"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 1 / 3)

    def test_return_type_is_float(self) -> None:
        trace = [_make_action_selection_trace_item({"winning_valence": "hunger"})]
        result = _hunger_valence_rate(trace)
        self.assertIsInstance(result, float)

    def test_missing_winning_valence_counts_as_non_hunger(self) -> None:
        trace = [
            _make_action_selection_trace_item({}),
            _make_action_selection_trace_item({"winning_valence": "hunger"}),
        ]
        self.assertAlmostEqual(_hunger_valence_rate(trace), 0.5)


class ClassifyFoodDeprivationFailureEdgeCasesTest(unittest.TestCase):
    """Edge-case tests for _classify_food_deprivation_failure beyond the standard fixture cases."""

    def test_empty_metrics_is_no_commitment(self) -> None:
        self.assertEqual(_classify_food_deprivation_failure({}), "no_commitment")

    def test_checks_passed_true_is_success(self) -> None:
        self.assertEqual(
            _classify_food_deprivation_failure({"checks_passed": True}), "success"
        )

    def test_checks_passed_truthy_string_is_success(self) -> None:
        # bool("non-empty") is True
        self.assertEqual(
            _classify_food_deprivation_failure({"checks_passed": "yes"}), "success"
        )

    def test_hunger_valence_rate_0_5_not_classified_as_no_commitment(self) -> None:
        # threshold is hunger_valence_rate < 0.5 → 0.5 is NOT < 0.5 → passes
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "hunger_valence_rate": 0.5,
            "initial_food_distance": 6.0,
            "min_food_distance_reached": 4.0,
            "alive": True,
            "food_eaten": 0,
        }
        result = _classify_food_deprivation_failure(metrics)
        # hunger_valence_rate=0.5 is NOT < 0.5, so commitment condition passes
        self.assertNotEqual(result, "no_commitment")

    def test_hunger_valence_rate_just_below_0_5_is_no_commitment(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "hunger_valence_rate": 0.49,
        }
        self.assertEqual(_classify_food_deprivation_failure(metrics), "no_commitment")

    def test_food_distance_delta_fallback_when_no_min_initial(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "hunger_valence_rate": 0.8,
            "food_distance_delta": 0.0,  # no progress
            "alive": True,
            "food_eaten": 0,
        }
        self.assertEqual(_classify_food_deprivation_failure(metrics), "orientation_failure")

    def test_food_distance_delta_positive_is_approached(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "hunger_valence_rate": 0.8,
            "food_distance_delta": 1.0,
            "alive": True,
            "food_eaten": 0,
        }
        self.assertEqual(_classify_food_deprivation_failure(metrics), "scoring_mismatch")

    def test_dead_with_food_eaten_is_timing_failure(self) -> None:
        # food_eaten > 0 but not alive; last return in function returns "timing_failure"
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "hunger_valence_rate": 0.8,
            "food_distance_delta": 2.0,
            "alive": False,
            "food_eaten": 1,
        }
        result = _classify_food_deprivation_failure(metrics)
        # alive=False, food_eaten=1 > 0 → neither "not alive and food_eaten <= 0" nor "alive"
        # falls through to the last return "timing_failure"
        self.assertEqual(result, "timing_failure")

    def test_not_left_shelter_is_no_commitment(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": False,
            "hunger_valence_rate": 1.0,
        }
        self.assertEqual(_classify_food_deprivation_failure(metrics), "no_commitment")

    def test_min_less_than_initial_is_approached(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "hunger_valence_rate": 0.9,
            "min_food_distance_reached": 3.0,
            "initial_food_distance": 8.0,
            "alive": True,
            "food_eaten": 0,
        }
        self.assertEqual(_classify_food_deprivation_failure(metrics), "scoring_mismatch")

    def test_min_equal_to_initial_is_not_approached(self) -> None:
        metrics = {
            "checks_passed": False,
            "left_shelter": True,
            "hunger_valence_rate": 0.9,
            "min_food_distance_reached": 6.0,
            "initial_food_distance": 6.0,
            "alive": True,
            "food_eaten": 0,
        }
        self.assertEqual(_classify_food_deprivation_failure(metrics), "orientation_failure")


# ---------------------------------------------------------------------------
# _clamp01
# ---------------------------------------------------------------------------

class Clamp01Test(unittest.TestCase):
    """Tests for the _clamp01 utility introduced in this PR."""

    def test_value_below_zero_returns_zero(self) -> None:
        self.assertEqual(_clamp01(-1.0), 0.0)

    def test_value_above_one_returns_one(self) -> None:
        self.assertEqual(_clamp01(2.5), 1.0)

    def test_exact_zero_returns_zero(self) -> None:
        self.assertEqual(_clamp01(0.0), 0.0)

    def test_exact_one_returns_one(self) -> None:
        self.assertEqual(_clamp01(1.0), 1.0)

    def test_midrange_value_unchanged(self) -> None:
        self.assertAlmostEqual(_clamp01(0.5), 0.5)

    def test_small_positive_value_unchanged(self) -> None:
        self.assertAlmostEqual(_clamp01(0.01), 0.01)

    def test_value_just_below_zero_returns_zero(self) -> None:
        self.assertEqual(_clamp01(-0.0001), 0.0)

    def test_value_just_above_one_returns_one(self) -> None:
        self.assertEqual(_clamp01(1.0001), 1.0)

    def test_returns_float(self) -> None:
        result = _clamp01(0.7)
        self.assertIsInstance(result, float)

    def test_negative_large_value_returns_zero(self) -> None:
        self.assertEqual(_clamp01(-1000.0), 0.0)


# ---------------------------------------------------------------------------
# _memory_vector_freshness
# ---------------------------------------------------------------------------

class MemoryVectorFreshnessTest(unittest.TestCase):
    """Tests for _memory_vector_freshness introduced in this PR."""

    def test_non_mapping_returns_zero(self) -> None:
        self.assertEqual(_memory_vector_freshness(None), 0.0)
        self.assertEqual(_memory_vector_freshness(42), 0.0)
        self.assertEqual(_memory_vector_freshness("string"), 0.0)
        self.assertEqual(_memory_vector_freshness([1, 2]), 0.0)

    def test_zero_displacement_returns_zero(self) -> None:
        self.assertEqual(_memory_vector_freshness({"dx": 0.0, "dy": 0.0, "age": 0.1}), 0.0)

    def test_zero_dx_nonzero_dy_uses_displacement(self) -> None:
        result = _memory_vector_freshness({"dx": 0.0, "dy": 1.0, "age": 0.2})
        self.assertAlmostEqual(result, 0.8)

    def test_missing_age_returns_zero(self) -> None:
        self.assertEqual(_memory_vector_freshness({"dx": 1.0, "dy": 0.0}), 0.0)

    def test_age_zero_returns_one(self) -> None:
        result = _memory_vector_freshness({"dx": 0.5, "dy": 0.5, "age": 0.0})
        self.assertAlmostEqual(result, 1.0)

    def test_age_one_returns_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 0.5, "dy": 0.5, "age": 1.0})
        self.assertAlmostEqual(result, 0.0)

    def test_age_in_range_returns_one_minus_age(self) -> None:
        result = _memory_vector_freshness({"dx": 0.5, "dy": 0.5, "age": 0.3})
        self.assertAlmostEqual(result, 0.7)

    def test_age_above_one_with_no_ttl_returns_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 5.0})
        self.assertEqual(result, 0.0)

    def test_age_above_one_with_ttl_zero_returns_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 5.0, "ttl": 0.0})
        self.assertEqual(result, 0.0)

    def test_age_above_one_with_valid_ttl(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 4.0, "ttl": 8.0})
        self.assertAlmostEqual(result, 0.5)

    def test_age_greater_than_ttl_clamped_to_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 12.0, "ttl": 8.0})
        self.assertEqual(result, 0.0)

    def test_negative_ttl_returns_zero(self) -> None:
        result = _memory_vector_freshness({"dx": 1.0, "dy": 0.0, "age": 3.0, "ttl": -1.0})
        self.assertEqual(result, 0.0)


# ---------------------------------------------------------------------------
# _food_signal_strength
# ---------------------------------------------------------------------------

class FoodSignalStrengthTest(unittest.TestCase):
    """Tests for _food_signal_strength introduced in this PR."""

    def test_empty_mapping_returns_zero(self) -> None:
        self.assertEqual(_food_signal_strength({}), 0.0)

    def test_food_visible_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_visible": 0.6}), 0.6)

    def test_food_certainty_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_certainty": 0.4}), 0.4)

    def test_food_smell_strength_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_smell_strength": 0.7}), 0.7)

    def test_food_trace_strength_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_trace_strength": 0.55}), 0.55)

    def test_food_memory_freshness_key(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_memory_freshness": 0.3}), 0.3)

    def test_value_above_one_is_clamped(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_visible": 2.5}), 1.0)

    def test_value_below_zero_is_clamped(self) -> None:
        self.assertAlmostEqual(_food_signal_strength({"food_visible": -0.5}), 0.0)

    def test_returns_max_of_multiple_direct_keys(self) -> None:
        source = {"food_visible": 0.3, "food_smell_strength": 0.8, "food_certainty": 0.5}
        self.assertAlmostEqual(_food_signal_strength(source), 0.8)

    def test_evidence_hunger_nesting(self) -> None:
        source = {"evidence": {"hunger": {"food_visible": 0.65}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.65)

    def test_evidence_visual_nesting(self) -> None:
        source = {"evidence": {"visual": {"food_certainty": 0.45}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.45)

    def test_evidence_sensory_nesting(self) -> None:
        source = {"evidence": {"sensory": {"food_smell_strength": 0.72}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.72)

    def test_meta_nesting(self) -> None:
        source = {"meta": {"food_visible": 0.55}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.55)

    def test_vision_food_visible(self) -> None:
        source = {"vision": {"food": {"visible": 0.88}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.88)

    def test_vision_food_certainty(self) -> None:
        source = {"vision": {"food": {"certainty": 0.77}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.77)

    def test_percept_traces_food_strength(self) -> None:
        source = {"percept_traces": {"food": {"strength": 0.6}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.6)

    def test_memory_vectors_food_freshness(self) -> None:
        source = {"memory_vectors": {"food": {"dx": 1.0, "dy": 0.0, "age": 0.4}}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.6)

    def test_food_trace_state_key(self) -> None:
        source = {"food_trace": {"strength": 0.5, "dx": 1.0, "dy": 0.0}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.5)

    def test_food_memory_with_target_age_ttl(self) -> None:
        source = {"food_memory": {"target": (5, 3), "age": 3.0, "ttl": 12.0}}
        self.assertAlmostEqual(_food_signal_strength(source), 0.75)

    def test_food_memory_without_target_ignored(self) -> None:
        source = {"food_memory": {"target": None, "age": 1.0, "ttl": 12.0}}
        self.assertEqual(_food_signal_strength(source), 0.0)

    def test_max_across_all_nested_sources(self) -> None:
        source = {
            "food_visible": 0.2,
            "meta": {"food_smell_strength": 0.9},
            "evidence": {"hunger": {"food_certainty": 0.5}},
        }
        self.assertAlmostEqual(_food_signal_strength(source), 0.9)

    def test_vision_food_not_mapping_ignored(self) -> None:
        source = {"vision": {"food": "not_a_mapping"}}
        self.assertEqual(_food_signal_strength(source), 0.0)

    def test_percept_traces_food_not_mapping_ignored(self) -> None:
        source = {"percept_traces": {"food": "absent"}}
        self.assertEqual(_food_signal_strength(source), 0.0)


# ---------------------------------------------------------------------------
# _trace_food_signal_strengths
# ---------------------------------------------------------------------------

class TraceFoodSignalStrengthsTest(unittest.TestCase):
    """Tests for _trace_food_signal_strengths introduced in this PR."""

    def test_empty_trace_returns_empty_list(self) -> None:
        self.assertEqual(_trace_food_signal_strengths([]), [])

    def test_single_item_with_no_signal_returns_zero(self) -> None:
        result = _trace_food_signal_strengths([{"tick": 0}])
        self.assertEqual(result, [0.0])

    def test_signal_from_state(self) -> None:
        item = {"state": {"food_visible": 0.7}}
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.7)

    def test_signal_from_observation_key(self) -> None:
        item = {"observation": {"food_certainty": 0.6}}
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.6)

    def test_signal_from_next_observation_key(self) -> None:
        item = {"next_observation": {"food_smell_strength": 0.5}}
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.5)

    def test_signal_from_message_payload(self) -> None:
        item = {
            "messages": [
                {
                    "sender": "environment",
                    "topic": "observation",
                    "payload": {"food_visible": 0.8},
                }
            ]
        }
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.8)

    def test_per_tick_maximum_across_sources(self) -> None:
        item = {
            "state": {"food_visible": 0.3},
            "observation": {"food_smell_strength": 0.9},
        }
        result = _trace_food_signal_strengths([item])
        self.assertAlmostEqual(result[0], 0.9)

    def test_multiple_items_aligned_to_trace(self) -> None:
        items = [
            {"state": {"food_visible": 0.5}},
            {"state": {"food_visible": 0.0}},
            {"state": {"food_visible": 0.8}},
        ]
        result = _trace_food_signal_strengths(items)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.8)

    def test_message_with_non_mapping_payload_ignored(self) -> None:
        item = {
            "messages": [
                {"sender": "env", "topic": "obs", "payload": "not_a_mapping"},
            ]
        }
        result = _trace_food_signal_strengths([item])
        self.assertEqual(result[0], 0.0)

    def test_messages_not_a_list_ignored(self) -> None:
        item = {"messages": "invalid"}
        result = _trace_food_signal_strengths([item])
        self.assertEqual(result[0], 0.0)

    def test_full_open_field_trace_item_signal(self) -> None:
        item = _make_open_field_trace_item(
            tick=0,
            pos=(4, 5),
            health=1.0,
            food_dist=6,
            food_signal_strength=0.6,
        )
        result = _trace_food_signal_strengths([item])
        self.assertGreater(result[0], 0.0)

    def test_full_open_field_trace_item_no_signal(self) -> None:
        item = _make_open_field_trace_item(
            tick=0,
            pos=(4, 5),
            health=1.0,
            food_dist=6,
            food_signal_strength=0.0,
        )
        result = _trace_food_signal_strengths([item])
        self.assertEqual(result[0], 0.0)


# ---------------------------------------------------------------------------
# aggregate_behavior_scores: new optional metadata params + failure_mode absent
# ---------------------------------------------------------------------------

class AggregateBehaviorScoresMetadataTest(unittest.TestCase):
    """Additional tests for aggregate_behavior_scores covering new optional params."""

    def _make_score(self, success: bool = True) -> BehavioralEpisodeScore:
        return BehavioralEpisodeScore(
            episode=0,
            seed=1,
            scenario="s",
            objective="o",
            success=success,
            checks={"check_a": _make_behavior_check_result("check_a", passed=success)},
            behavior_metrics={},
            failures=[] if success else ["check_a"],
        )

    def test_none_metadata_params_become_empty_strings(self) -> None:
        result = aggregate_behavior_scores(
            [],
            scenario="s",
            description="d",
            objective="o",
            diagnostic_focus=None,
            success_interpretation=None,
            failure_interpretation=None,
            budget_note=None,
            check_specs=[],
        )
        self.assertEqual(result["diagnostic_focus"], "")
        self.assertEqual(result["success_interpretation"], "")
        self.assertEqual(result["failure_interpretation"], "")
        self.assertEqual(result["budget_note"], "")

    def test_no_failure_mode_in_scores_omits_distribution_keys(self) -> None:
        scores = [self._make_score(False), self._make_score(True)]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[BehaviorCheckSpec("check_a", "desc", "true")],
        )
        self.assertNotIn("primary_failure_mode", result["diagnostics"])
        self.assertNotIn("failure_mode_distribution", result["diagnostics"])

    def test_single_failure_mode_is_primary(self) -> None:
        score = self._make_score(False)
        score.behavior_metrics["failure_mode"] = "orientation_failure"
        result = aggregate_behavior_scores(
            [score],
            scenario="s",
            description="d",
            objective="o",
            check_specs=[BehaviorCheckSpec("check_a", "desc", "true")],
        )
        self.assertEqual(result["diagnostics"]["primary_failure_mode"], "orientation_failure")
        self.assertAlmostEqual(
            result["diagnostics"]["failure_mode_distribution"]["orientation_failure"],
            1.0,
        )

    def test_metadata_keys_present_even_with_no_scores(self) -> None:
        result = aggregate_behavior_scores(
            [],
            scenario="scenario_x",
            description="desc_x",
            objective="obj_x",
            diagnostic_focus="focus_x",
            success_interpretation="success_x",
            failure_interpretation="failure_x",
            budget_note="note_x",
            check_specs=[],
        )
        self.assertEqual(result["diagnostic_focus"], "focus_x")
        self.assertEqual(result["success_interpretation"], "success_x")
        self.assertEqual(result["failure_interpretation"], "failure_x")
        self.assertEqual(result["budget_note"], "note_x")
        self.assertEqual(result["episodes"], 0)


# ---------------------------------------------------------------------------
# _classify_open_field_foraging_failure: boundary and edge cases
# ---------------------------------------------------------------------------

class ClassifyOpenFieldForagingFailureBoundaryTest(unittest.TestCase):
    """Boundary and edge case tests for _classify_open_field_foraging_failure."""

    def test_hunger_valence_rate_exactly_0_5_not_no_commitment(self) -> None:
        metrics = {**_open_field_failure_base_metrics(), "hunger_valence_rate": 0.5}
        # The check is `< 0.5`, so 0.5 passes: should NOT be no_hunger_commitment
        self.assertNotEqual(_classify_open_field_foraging_failure(metrics), "no_hunger_commitment")

    def test_hunger_valence_rate_just_below_0_5_is_no_commitment(self) -> None:
        metrics = {**_open_field_failure_base_metrics(), "hunger_valence_rate": 0.499}
        self.assertEqual(_classify_open_field_foraging_failure(metrics), "no_hunger_commitment")

    def test_food_eaten_positive_overrides_distance_fallback(self) -> None:
        # food_eaten > 0 makes approached_food True even if distances show no progress.
        metrics = {
            **_open_field_failure_base_metrics(),
            "food_eaten": 1,
            "min_food_distance_reached": 8.0,  # no improvement in distance
            "initial_food_distance": 8.0,
            "food_distance_delta": 0.0,
            "alive": False,
        }
        self.assertEqual(_classify_open_field_foraging_failure(metrics), "progressed_then_died")

    def test_food_eaten_positive_and_alive_is_scoring_mismatch(self) -> None:
        metrics = {
            **_open_field_failure_base_metrics(),
            "food_eaten": 2,
            "min_food_distance_reached": 8.0,
            "initial_food_distance": 8.0,
            "food_distance_delta": 0.0,
            "alive": True,
        }
        self.assertEqual(_classify_open_field_foraging_failure(metrics), "scoring_mismatch")

    def test_food_distance_delta_fallback_when_distances_missing(self) -> None:
        # When min and initial distances are both None, fall back to food_distance_delta
        metrics = {
            **_open_field_failure_base_metrics(),
            "min_food_distance_reached": None,
            "initial_food_distance": None,
            "food_distance_delta": 3.0,
            "alive": False,
        }
        self.assertEqual(_classify_open_field_foraging_failure(metrics), "progressed_then_died")

    def test_food_distance_delta_zero_when_distances_missing_is_orientation_failure(self) -> None:
        metrics = {
            **_open_field_failure_base_metrics(),
            "min_food_distance_reached": None,
            "initial_food_distance": None,
            "food_distance_delta": 0.0,
            "food_eaten": 0,
            "alive": False,
        }
        self.assertEqual(_classify_open_field_foraging_failure(metrics), "orientation_failure")

    def test_missing_predator_visible_ticks_does_not_stall(self) -> None:
        metrics = {
            **_open_field_failure_base_metrics(),
            "min_food_distance_reached": 8.0,
            "food_distance_delta": 0.0,
            "food_eaten": 0,
            "alive": True,
        }
        self.assertEqual(_classify_open_field_foraging_failure(metrics), "orientation_failure")

    def test_explicit_zero_predator_visible_ticks_stalls(self) -> None:
        metrics = {
            **_open_field_failure_base_metrics(),
            "min_food_distance_reached": 8.0,
            "food_distance_delta": 0.0,
            "food_eaten": 0,
            "alive": True,
            "predator_visible_ticks": 0,
        }
        self.assertEqual(_classify_open_field_foraging_failure(metrics), "stall")

    def test_initial_food_signal_fallback_when_max_is_none(self) -> None:
        # max_food_signal_strength not present, but initial_food_signal_strength is
        metrics = {
            **_open_field_failure_base_metrics(),
            "max_food_signal_strength": None,
            "initial_food_signal_strength": 0.5,
        }
        # Should NOT return left_without_food_signal because initial_food_signal_strength > 0
        self.assertNotEqual(_classify_open_field_foraging_failure(metrics), "left_without_food_signal")

    def test_both_signal_strengths_none_treated_as_zero(self) -> None:
        metrics = {
            **_open_field_failure_base_metrics(),
            "max_food_signal_strength": None,
            "initial_food_signal_strength": None,
        }
        self.assertEqual(_classify_open_field_foraging_failure(metrics), "left_without_food_signal")


# ---------------------------------------------------------------------------
# _score_open_field_foraging: additional edge cases
# ---------------------------------------------------------------------------

class ScoreOpenFieldForagingEdgeCasesTest(unittest.TestCase):
    """Additional edge-case tests for the open_field_foraging scorer."""

    def test_empty_trace_produces_zero_signal_metrics(self) -> None:
        spec = get_scenario("open_field_foraging")
        stats = _make_episode_stats(
            scenario="open_field_foraging",
            food_distance_delta=0.0,
            food_eaten=0,
            alive=False,
            predator_contacts=0,
        )
        score = spec.score_episode(stats, [])
        self.assertEqual(score.behavior_metrics["initial_food_signal_strength"], 0.0)
        self.assertEqual(score.behavior_metrics["max_food_signal_strength"], 0.0)
        self.assertEqual(score.behavior_metrics["food_signal_tick_rate"], 0.0)
        self.assertIsNone(score.behavior_metrics["initial_food_distance"])
        self.assertIsNone(score.behavior_metrics["min_food_distance_reached"])

    def test_partial_signal_tick_rate(self) -> None:
        spec = get_scenario("open_field_foraging")
        stats = _make_episode_stats(
            scenario="open_field_foraging",
            food_distance_delta=1.0,
            food_eaten=0,
            alive=True,
            predator_contacts=0,
        )
        trace = [
            _make_open_field_trace_item(
                tick=0, pos=(4, 5), health=1.0, food_dist=6,
                food_signal_strength=0.5,
            ),
            _make_open_field_trace_item(
                tick=1, pos=(4, 4), health=1.0, food_dist=5,
                food_signal_strength=0.0,
            ),
            _make_open_field_trace_item(
                tick=2, pos=(4, 3), health=1.0, food_dist=4,
                food_signal_strength=0.0,
            ),
            _make_open_field_trace_item(
                tick=3, pos=(4, 2), health=1.0, food_dist=3,
                food_signal_strength=0.4,
            ),
        ]
        score = spec.score_episode(stats, trace)
        self.assertAlmostEqual(score.behavior_metrics["food_signal_tick_rate"], 2 / 4)
        self.assertAlmostEqual(score.behavior_metrics["initial_food_signal_strength"], 0.5)
        self.assertAlmostEqual(score.behavior_metrics["max_food_signal_strength"], 0.5)

    def test_food_eaten_via_stats_yields_success_failure_mode(self) -> None:
        spec = get_scenario("open_field_foraging")
        stats = _make_episode_stats(
            scenario="open_field_foraging",
            food_distance_delta=4.0,
            food_eaten=1,
            alive=True,
            predator_contacts=0,
        )
        trace = [
            _make_open_field_trace_item(
                tick=0, pos=(4, 5), health=1.0, food_dist=5,
                winning_valence="hunger",
                food_signal_strength=0.5,
            ),
        ]
        score = spec.score_episode(stats, trace)
        self.assertTrue(score.checks["foraging_viable"].passed)
        self.assertEqual(score.behavior_metrics["failure_mode"], "success")

    def test_no_shelter_exit_detected_when_spider_stays_in_shelter(self) -> None:
        spec = get_scenario("open_field_foraging")
        stats = _make_episode_stats(
            scenario="open_field_foraging",
            food_distance_delta=0.0,
            food_eaten=0,
            alive=True,
            predator_contacts=0,
        )
        shelter_pos, _ = _open_field_trace_positions()
        trace = [
            _make_open_field_trace_item(
                tick=0, pos=shelter_pos, health=1.0, food_dist=8,
                winning_valence="sleep",
                food_signal_strength=0.0,
            ),
        ]
        score = spec.score_episode(stats, trace)
        self.assertFalse(score.behavior_metrics["left_shelter"])
        self.assertIsNone(score.behavior_metrics["shelter_exit_tick"])


if __name__ == "__main__":
    unittest.main()
