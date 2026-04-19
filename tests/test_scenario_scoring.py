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

from tests.fixtures.scenario_scoring import ScoreFunctionTestBase

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

class ScoreFunctionCoreTest(ScoreFunctionTestBase):
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

    def test_score_episode_returns_correct_scenario_name(self) -> None:
        for name in SCENARIO_NAMES:
            spec = get_scenario(name)
            stats = _make_episode_stats(scenario=name)
            score = spec.score_episode(stats, [])
            self.assertEqual(score.scenario, name, msg=f"Expected scenario name '{name}'")

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
