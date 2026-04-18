from __future__ import annotations

import unittest

from spider_cortex_sim.metrics import EpisodeStats
from spider_cortex_sim.scenarios import get_scenario
from spider_cortex_sim.scenarios.trace import (
    _clamp01,
    _float_or_none,
    _food_signal_strength,
    _hunger_valence_rate,
    _int_or_none,
    _mapping_predator_visible,
    _memory_vector_freshness,
    _observation_food_distances,
    _payload_float,
    _payload_text,
    _predator_visible_flag,
    _state_alive,
    _state_food_distance,
    _state_position,
    _trace_action_selection_payloads,
    _trace_any_mode,
    _trace_any_sleep_phase,
    _extract_exposed_day_trace_metrics,
    _trace_death_tick,
    _trace_dominant_predator_types,
    _trace_escape_seen,
    _trace_food_distances,
    _trace_food_signal_strengths,
    _trace_post_observation_food_distances,
    _trace_predator_memory_seen,
    _trace_predator_visible,
    _trace_pre_observation_food_distances,
    _trace_shelter_cells,
    _trace_shelter_exit,
    _trace_states,
    _trace_tick,
)
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES, SpiderWorld

def _make_episode_stats(**overrides: object) -> EpisodeStats:
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

    def test_delta_zero_preserves_unique_distance_once(self) -> None:
        trace = [
            {
                "observation": {"meta": {"food_dist": 5.0}},
                "distance_deltas": {"food": 0.0},
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [5.0])

    def test_exposed_day_final_distance_uses_raw_trace_endpoint(self) -> None:
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
            {"observation": {"meta": {"food_dist": 5.0}}},
        ]
        self.assertEqual(_trace_food_distances(trace), [5.0, 4.0])
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 5.0)

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


class TraceFoodDistancesDeduplicationTest(unittest.TestCase):
    """Tests for the new deduplication behavior in _trace_food_distances (dict.fromkeys)."""

    def test_exact_duplicates_removed(self) -> None:
        """Same distance appearing in observation and state should appear only once."""
        trace = [
            {
                "observation": {"meta": {"food_dist": 5.0}},
                "state": {"food_dist": 5.0},
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances.count(5.0), 1)

    def test_distinct_values_all_kept(self) -> None:
        """Different distances are preserved in order."""
        trace = [
            {"observation": {"meta": {"food_dist": 8.0}}},
            {"observation": {"meta": {"food_dist": 6.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [8.0, 6.0, 4.0])

    def test_repeated_value_across_items_deduplicated(self) -> None:
        """Same distance seen in multiple items appears only at first occurrence."""
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
            {"observation": {"meta": {"food_dist": 5.0}}},  # duplicate of first
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [5.0, 4.0])
        self.assertEqual(distances.count(5.0), 1)

    def test_insertion_order_preserved(self) -> None:
        """First-seen order is preserved after deduplication."""
        trace = [
            {"observation": {"meta": {"food_dist": 10.0}}},
            {"observation": {"meta": {"food_dist": 7.0}}},
            {"observation": {"meta": {"food_dist": 10.0}}},
            {"observation": {"meta": {"food_dist": 3.0}}},
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances, [10.0, 7.0, 3.0])

    def test_synthesized_distance_same_as_explicit_deduplicated(self) -> None:
        """Synthesized post-distance equal to observation is deduplicated."""
        trace = [
            {
                "observation": {"meta": {"food_dist": 5.0}},
                "distance_deltas": {"food": 0.0},  # synthesized post = 5.0 - 0.0 = 5.0
            }
        ]
        distances = _trace_food_distances(trace)
        # 5.0 (observation) + synthesized 5.0 -> deduplicated to [5.0]
        self.assertEqual(distances, [5.0])

    def test_empty_trace_returns_empty_list(self) -> None:
        self.assertEqual(_trace_food_distances([]), [])

    def test_return_type_is_list(self) -> None:
        trace = [{"observation": {"meta": {"food_dist": 5.0}}}]
        result = _trace_food_distances(trace)
        self.assertIsInstance(result, list)

    def test_observation_and_next_observation_same_value_deduplicated(self) -> None:
        """Same distance in observation and next_observation deduplicated."""
        trace = [
            {
                "observation": {"meta": {"food_dist": 6.0}},
                "next_observation": {"meta": {"food_dist": 6.0}},
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertEqual(distances.count(6.0), 1)

    def test_regression_observation_state_two_unique_distances(self) -> None:
        """Observation and state with different values both appear."""
        trace = [
            {
                "observation": {"meta": {"food_dist": 8.0}},
                "state": {"food_dist": 7.0},
            }
        ]
        distances = _trace_food_distances(trace)
        self.assertIn(8.0, distances)
        self.assertIn(7.0, distances)


class ExtractExposedDayTraceMetricsFinalDistanceTest(unittest.TestCase):
    """Tests for final_distance_to_food in _extract_exposed_day_trace_metrics.

    In the new implementation, final_distance_to_food uses reversed(trace) to find
    the last trace item that has food distance data, not _trace_food_distances[-1].
    """

    def test_final_distance_is_raw_last_item_value(self) -> None:
        """final_distance_to_food is from the last trace item, not the last unique value."""
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
            {"observation": {"meta": {"food_dist": 5.0}}},  # last item, same as first
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        # The last item has food_dist=5.0, so that's the final distance
        self.assertEqual(metrics["final_distance_to_food"], 5.0)

    def test_final_distance_different_from_deduplicated_last(self) -> None:
        """Demonstrates that final_distance_to_food differs from _trace_food_distances[-1]."""
        trace = [
            {"observation": {"meta": {"food_dist": 10.0}}},
            {"observation": {"meta": {"food_dist": 7.0}}},
            {"observation": {"meta": {"food_dist": 10.0}}},  # repeat, dedup would give 7.0 as last
        ]
        food_distances = _trace_food_distances(trace)
        # Dedup gives [10.0, 7.0] - so [-1] would be 7.0
        self.assertEqual(food_distances[-1], 7.0)
        # But final_distance_to_food uses the actual last item with data: 10.0
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 10.0)

    def test_final_distance_is_none_when_no_food_data(self) -> None:
        """Returns None when no food distance info exists in any trace item."""
        trace = [{"tick": 0}, {"tick": 1}]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertIsNone(metrics["final_distance_to_food"])

    def test_final_distance_from_single_item(self) -> None:
        """Single trace item with food_dist returns that distance."""
        trace = [{"observation": {"meta": {"food_dist": 9.0}}}]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 9.0)

    def test_final_distance_uses_last_item_with_food_data(self) -> None:
        """Uses last item that has food data even if some items lack it."""
        trace = [
            {"observation": {"meta": {"food_dist": 8.0}}},
            {"tick": 1},  # no food data
            {"tick": 2},  # no food data
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 8.0)

    def test_final_distance_from_next_observation(self) -> None:
        """final_distance_to_food can come from next_observation meta."""
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"next_observation": {"meta": {"food_dist": 3.0}}},
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 3.0)

    def test_final_distance_from_state(self) -> None:
        """final_distance_to_food can come from state field."""
        trace = [
            {"observation": {"meta": {"food_dist": 5.0}}},
            {"state": {"food_dist": 2.0}},
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 2.0)

    def test_other_metrics_keys_present(self) -> None:
        """All expected metric keys are present in the result."""
        trace = [{"observation": {"meta": {"food_dist": 5.0}}}]
        metrics = _extract_exposed_day_trace_metrics(trace)
        for key in ("shelter_exit_tick", "left_shelter", "peak_food_progress",
                    "predator_visible_ticks", "final_distance_to_food"):
            self.assertIn(key, metrics)

    def test_monotonically_decreasing_distances_final_is_minimum(self) -> None:
        """For strictly decreasing distances, final is the minimum."""
        trace = [
            {"observation": {"meta": {"food_dist": 8.0}}},
            {"observation": {"meta": {"food_dist": 6.0}}},
            {"observation": {"meta": {"food_dist": 4.0}}},
        ]
        metrics = _extract_exposed_day_trace_metrics(trace)
        self.assertEqual(metrics["final_distance_to_food"], 4.0)


class TraceDominantPredatorTypesTest(unittest.TestCase):
    """Tests for _trace_dominant_predator_types from the new trace module."""

    def _make_meta_item(self, label: str) -> dict:
        return {"observation": {"meta": {"dominant_predator_type_label": label}}}

    def test_empty_trace_returns_empty_set(self) -> None:
        result = _trace_dominant_predator_types([])
        self.assertEqual(result, set())

    def test_visual_label_detected(self) -> None:
        trace = [self._make_meta_item("visual")]
        result = _trace_dominant_predator_types(trace)
        self.assertIn("visual", result)

    def test_olfactory_label_detected(self) -> None:
        trace = [self._make_meta_item("olfactory")]
        result = _trace_dominant_predator_types(trace)
        self.assertIn("olfactory", result)

    def test_both_labels_detected_across_items(self) -> None:
        trace = [
            self._make_meta_item("visual"),
            self._make_meta_item("olfactory"),
        ]
        result = _trace_dominant_predator_types(trace)
        self.assertEqual(result, {"visual", "olfactory"})

    def test_unknown_label_ignored(self) -> None:
        trace = [self._make_meta_item("thermal")]
        result = _trace_dominant_predator_types(trace)
        self.assertEqual(result, set())

    def test_case_normalized_to_lowercase(self) -> None:
        trace = [self._make_meta_item("VISUAL")]
        result = _trace_dominant_predator_types(trace)
        self.assertIn("visual", result)

    def test_none_label_ignored(self) -> None:
        trace = [{"observation": {"meta": {"dominant_predator_type_label": None}}}]
        result = _trace_dominant_predator_types(trace)
        self.assertEqual(result, set())

    def test_message_meta_also_scanned(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "environment",
                        "topic": "observation",
                        "payload": {"meta": {"dominant_predator_type_label": "olfactory"}},
                    }
                ]
            }
        ]
        result = _trace_dominant_predator_types(trace)
        self.assertIn("olfactory", result)

    def test_returns_set_type(self) -> None:
        result = _trace_dominant_predator_types([])
        self.assertIsInstance(result, set)
