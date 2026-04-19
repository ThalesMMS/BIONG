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
                zip(positions, food_distances, winning_valences, healths)
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

class SpecializedScenarioScoringTest(unittest.TestCase):
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

    def test_olfactory_ambush_detection_uses_module_response_fallback(self) -> None:
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

        self.assertTrue(score.checks["olfactory_threat_detected"].passed)
        self.assertEqual(score.behavior_metrics["olfactory_predator_threat_peak_initial"], 0.0)

    def test_visual_olfactory_pincer_detection_uses_module_response_fallback(self) -> None:
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

        self.assertTrue(score.checks["dual_threat_detected"].passed)

    def test_olfactory_ambush_detection_fails_without_peak_or_response(self) -> None:
        spec = get_scenario("olfactory_ambush")
        stats = _make_corridor_episode_stats(
            scenario="olfactory_ambush",
            module_response_by_predator_type={
                "olfactory": {"sensory_cortex": 0.0, "visual_cortex": 0.0},
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

    def test_visual_hunter_open_field_detection_fails_without_peak_or_response(self) -> None:
        spec = get_scenario("visual_hunter_open_field")
        stats = _make_corridor_episode_stats(
            scenario="visual_hunter_open_field",
            module_response_by_predator_type={
                "visual": {"visual_cortex": 0.0, "sensory_cortex": 0.0},
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
