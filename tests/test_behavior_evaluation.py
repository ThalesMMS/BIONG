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

import csv
import io
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

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
from spider_cortex_sim.scenarios import (
    FOOD_DEPRIVATION_INITIAL_HUNGER,
    FOOD_DEPRIVATION_CHECKS,
    NIGHT_REST_INITIAL_SLEEP_DEBT,
    NIGHT_REST_CHECKS,
    PREDATOR_EDGE_CHECKS,
    SCENARIO_NAMES,
    _trace_any_mode,
    _trace_any_sleep_phase,
    _trace_escape_seen,
    _trace_predator_memory_seen,
    _trace_states,
    get_scenario,
)
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES, SpiderWorld

NOISE_PROFILE_CONFIG_JSON = (
    "{\"motor\":{\"action_flip_prob\":0.0},\"name\":\"none\","
    "\"olfactory\":{\"direction_jitter\":0.0,\"strength_jitter\":0.0},"
    "\"predator\":{\"random_choice_prob\":0.0},\"spawn\":{\"uniform_mix\":0.0},"
    "\"visual\":{\"certainty_jitter\":0.0,\"direction_jitter\":0.0,\"dropout_prob\":0.0}}"
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


def _make_behavior_check_result(name="check_a", passed=True, value=1.0, description="desc", expected="true") -> BehaviorCheckResult:
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
        for col in ["reward_profile", "scenario_map", "evaluation_map", "simulation_seed", "episode_seed", "scenario", "scenario_description", "scenario_objective", "scenario_focus", "episode", "success", "failure_count", "failures"]:
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

    def test_food_deprivation_checks_has_three_items(self) -> None:
        self.assertEqual(len(FOOD_DEPRIVATION_CHECKS), 3)

    def test_food_deprivation_check_names(self) -> None:
        names = {spec.name for spec in FOOD_DEPRIVATION_CHECKS}
        self.assertIn("hunger_reduced", names)
        self.assertIn("approaches_food", names)
        self.assertIn("survives_deprivation", names)

    def test_all_specs_have_non_empty_fields(self) -> None:
        for spec in list(NIGHT_REST_CHECKS) + list(PREDATOR_EDGE_CHECKS) + list(FOOD_DEPRIVATION_CHECKS):
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
        return [
            {"state": {"lizard_mode": "PATROL", "sleep_phase": "DEEP_SLEEP", "predator_memory": {"target": None}}, "predator_escape": False},
        ]

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
        self.assertGreater(dist, 3)

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

    def _make_rows(self):
        """
        Builds example CSV-ready behavior evaluation rows used by the test suite.
        
        Each returned element is a dict representing one episode row with fields found in the behavior-evaluation CSV export: general metadata (reward/evaluation maps, operational and noise profiles, seeds, checkpoint source), learning-evidence configuration (condition, policy mode, train episodes, frozen checkpoint, checkpoint source, budget profile and benchmark strength), reflex and budget settings (reflex_scale, reflex_anneal_final_scale, eval_reflex_scale, budget_profile, benchmark_strength), scenario metadata (name, description, objective, focus), outcome fields (episode, success, failure_count, failures), numeric/behavior metrics (prefixed with `metric_`) and per-check columns (prefixed with `check_` containing `_passed`, `_value`, and `_expected` variants).
        
        Returns:
            list[dict]: A list of row dictionaries (one per episode) containing CSV-ready fields used by behavior tests.
        """
        return [
            {
                "reward_profile": "classic",
                "scenario_map": "night_rest_map",
                "evaluation_map": "central_burrow",
                "learning_evidence_condition": "trained_final",
                "learning_evidence_policy_mode": "normal",
                "learning_evidence_train_episodes": 6,
                "learning_evidence_frozen_after_episode": "",
                "learning_evidence_checkpoint_source": "final",
                "learning_evidence_budget_profile": "smoke",
                "learning_evidence_budget_benchmark_strength": "quick",
                "reflex_scale": 1.0,
                "reflex_anneal_final_scale": 0.5,
                "eval_reflex_scale": 0.0,
                "budget_profile": "dev",
                "benchmark_strength": "quick",
                "operational_profile": "default_v1",
                "operational_profile_version": 1,
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
        Verifies the CSV header produced by save_behavior_csv contains all expected preferred columns, including learning-evidence fields, reflex/budget/checkpoint settings, scenario metadata, and outcome columns.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.csv"
            self.SpiderSimulation.save_behavior_csv(self._make_rows(), path)
            header_cols = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            for col in [
                "reward_profile",
                "scenario_map",
                "evaluation_map",
                "learning_evidence_condition",
                "learning_evidence_policy_mode",
                "learning_evidence_train_episodes",
                "learning_evidence_frozen_after_episode",
                "learning_evidence_checkpoint_source",
                "learning_evidence_budget_profile",
                "learning_evidence_budget_benchmark_strength",
                "reflex_scale",
                "reflex_anneal_final_scale",
                "eval_reflex_scale",
                "budget_profile",
                "benchmark_strength",
                "operational_profile",
                "operational_profile_version",
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
        Verify the CSV produced by SpiderSimulation.save_behavior_csv can be read by csv.DictReader and contains the expected stringified fields.
        
        Creates a temporary CSV file from a single behavior row, reads it back with csv.DictReader, asserts exactly one data row is present, and checks that a set of expected columns (including reward/profile/map metadata, seed fields, scenario metadata, outcome fields, and the `noise_profile` column) are serialized to the exact string values.
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
                "learning_evidence_condition": "trained_final",
                "learning_evidence_policy_mode": "normal",
                "learning_evidence_train_episodes": "6",
                "learning_evidence_frozen_after_episode": "",
                "learning_evidence_checkpoint_source": "final",
                "learning_evidence_budget_profile": "smoke",
                "learning_evidence_budget_benchmark_strength": "quick",
                "reflex_scale": "1.0",
                "reflex_anneal_final_scale": "0.5",
                "eval_reflex_scale": "0.0",
                "budget_profile": "dev",
                "benchmark_strength": "quick",
                "operational_profile": "default_v1",
                "operational_profile_version": "1",
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
        from spider_cortex_sim.cli import _default_behavior_evaluation

        payload = _default_behavior_evaluation()
        self.assertIn("suite", payload)
        self.assertIn("legacy_scenarios", payload)
        self.assertIn("summary", payload)
        self.assertEqual(payload["legacy_scenarios"], {})

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

    def test_behavior_csv_arg(self) -> None:
        args = self.parser.parse_args(["--behavior-csv", "output.csv"])
        self.assertEqual(args.behavior_csv, Path("output.csv"))

    def test_behavior_csv_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.behavior_csv)

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
    ) -> object:
        class FakeDecision:
            pass

        decision = FakeDecision()
        decision.final_reflex_override = final_reflex_override
        decision.module_results = module_results or []
        return decision

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

    def test_snapshot_denominator_uses_max_one_when_no_decisions(self) -> None:
        # With zero decisions, snapshot should still not raise division-by-zero
        acc = self._make_accumulator()
        snapshot = acc.snapshot()
        self.assertAlmostEqual(snapshot["reflex_usage_rate"], 0.0)
        self.assertAlmostEqual(snapshot["final_reflex_override_rate"], 0.0)

    def test_post_init_initializes_all_module_reflex_dicts(self) -> None:
        acc = self._make_accumulator()
        for name in REFLEX_MODULE_NAMES:
            self.assertIn(name, acc.module_reflex_usage_steps)
            self.assertIn(name, acc.module_reflex_override_steps)
            self.assertIn(name, acc.module_reflex_dominance_sums)


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
    ) -> EpisodeStats:
        module_rates = {name: 0.0 for name in REFLEX_MODULE_NAMES}
        module_overrides = {name: 0.0 for name in REFLEX_MODULE_NAMES}
        module_dominance = {name: 0.0 for name in REFLEX_MODULE_NAMES}
        if module_reflex_usage_rates:
            module_rates.update(module_reflex_usage_rates)
        if module_reflex_override_rates:
            module_overrides.update(module_reflex_override_rates)
        if module_reflex_dominance:
            module_dominance.update(module_reflex_dominance)
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
        )

    def test_aggregate_empty_history_returns_zeros(self) -> None:
        result = aggregate_episode_stats([])
        self.assertAlmostEqual(result["mean_reflex_usage_rate"], 0.0)
        self.assertAlmostEqual(result["mean_final_reflex_override_rate"], 0.0)
        self.assertAlmostEqual(result["mean_reflex_dominance"], 0.0)

    def test_aggregate_includes_mean_reflex_usage_rate(self) -> None:
        history = [
            self._make_episode_stats(reflex_usage_rate=0.4),
            self._make_episode_stats(reflex_usage_rate=0.6),
        ]
        result = aggregate_episode_stats(history)
        self.assertAlmostEqual(result["mean_reflex_usage_rate"], 0.5)

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


if __name__ == "__main__":
    unittest.main()
