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

class ScoreFunctionFoodOpenTest(ScoreFunctionTestBase):
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
