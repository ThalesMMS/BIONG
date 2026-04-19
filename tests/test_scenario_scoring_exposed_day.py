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
