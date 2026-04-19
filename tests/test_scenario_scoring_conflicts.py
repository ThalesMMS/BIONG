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

class ScoreFunctionConflictTest(ScoreFunctionTestBase):
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
