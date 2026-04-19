from __future__ import annotations

from collections.abc import Mapping
import unittest
from dataclasses import asdict
from numbers import Real
from typing import Any

from spider_cortex_sim.ablations import BrainAblationConfig, PROPOSAL_SOURCE_NAMES
from spider_cortex_sim.metrics import (
    BehaviorCheckResult,
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeStats,
    build_behavior_check,
    build_behavior_score,
)
from spider_cortex_sim.noise import LOW_NOISE_PROFILE
from spider_cortex_sim.predator import PREDATOR_STATES
from spider_cortex_sim.reward import SCENARIO_AUSTERE_REQUIREMENTS, SHAPING_GAP_POLICY
from spider_cortex_sim.scenarios import SCENARIO_NAMES, get_scenario
from spider_cortex_sim.scenarios.scoring import (
    CONFLICT_PASS_RATE,
    FOOD_DEPRIVATION_CHECKS,
    FOOD_VS_PREDATOR_CONFLICT_CHECKS,
    NIGHT_REST_CHECKS,
    OLFACTORY_AMBUSH_CHECKS,
    PREDATOR_EDGE_CHECKS,
    SLEEP_VS_EXPLORATION_CONFLICT_CHECKS,
    VISUAL_HUNTER_OPEN_FIELD_CHECKS,
    _classify_corridor_gauntlet_failure,
    _classify_exposed_day_foraging_failure,
    _classify_food_deprivation_failure,
    _classify_open_field_foraging_failure,
    _score_corridor_gauntlet,
)
from spider_cortex_sim.scenarios.specs import (
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    NIGHT_REST_INITIAL_SLEEP_DEBT,
    SLEEP_VS_EXPLORATION_INITIAL_SLEEP_DEBT,
)
from spider_cortex_sim.scenarios.trace import (
    _extract_exposed_day_trace_metrics,
    _food_signal_strength,
    _trace_action_selection_payloads,
    _trace_corridor_metrics,
)
from spider_cortex_sim.simulation import (
    CAPABILITY_PROBE_SCENARIOS,
    CURRICULUM_COLUMNS,
    SpiderSimulation,
    is_capability_probe,
)
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES, SpiderWorld

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
