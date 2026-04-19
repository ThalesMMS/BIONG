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

class ScoreFunctionCorridorTest(ScoreFunctionTestBase):
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
