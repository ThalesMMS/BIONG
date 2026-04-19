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

from tests.fixtures.scenario_trace_builders import (
    _make_episode_stats,
    _make_action_selection_trace_item,
    _make_food_deprivation_trace_item,
    _make_open_field_trace_item,
    _make_exposed_day_trace_item,
    _make_corridor_trace_item,
    _trace_positions_for_scenario,
    _open_field_trace_positions,
    _exposed_day_trace_positions,
    _corridor_trace_positions,
    _open_field_failure_base_metrics,
    _make_behavior_check_result,
    _make_behavioral_episode_score,
    _make_corridor_episode_stats,
)

class ScoreFunctionTestBase(unittest.TestCase):
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
