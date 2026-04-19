"""Focused metrics and behavior-evaluation tests."""

from __future__ import annotations

from collections.abc import Mapping
import unittest

from spider_cortex_sim.ablations import PROPOSAL_SOURCE_NAMES
from spider_cortex_sim.metrics import (
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeStats,
    REFLEX_MODULE_NAMES,
    _aggregate_values,
    _mean_like,
    _mean_map,
    aggregate_behavior_scores,
    aggregate_episode_stats,
    build_behavior_check,
    build_behavior_score,
    flatten_behavior_rows,
    summarize_behavior_suite,
)
from spider_cortex_sim.predator import PREDATOR_STATES
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES
from tests.fixtures.shared_builders import (
    make_behavior_check_result,
    make_episode_stats,
)

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

class BuildBehaviorScoreTest(unittest.TestCase):
    """Tests for build_behavior_score."""

    def test_success_when_all_checks_pass(self) -> None:
        stats = make_episode_stats(episode=5, seed=7, scenario="night_rest")
        checks = [
            make_behavior_check_result("a", passed=True),
            make_behavior_check_result("b", passed=True),
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
        stats = make_episode_stats(scenario="night_rest")
        checks = [
            make_behavior_check_result("a", passed=True),
            make_behavior_check_result("b", passed=False),
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
        stats = make_episode_stats(scenario=None)
        score = build_behavior_score(
            stats=stats,
            objective="test",
            checks=[],
            behavior_metrics={},
        )
        self.assertEqual(score.scenario, "default")

    def test_episode_and_seed_from_stats(self) -> None:
        stats = make_episode_stats(episode=42, seed=99, scenario="test")
        score = build_behavior_score(
            stats=stats,
            objective="test",
            checks=[],
            behavior_metrics={},
        )
        self.assertEqual(score.episode, 42)
        self.assertEqual(score.seed, 99)

    def test_checks_indexed_by_name(self) -> None:
        stats = make_episode_stats()
        check_a = make_behavior_check_result("alpha", passed=True)
        check_b = make_behavior_check_result("beta", passed=False)
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
        stats = make_episode_stats()
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
        stats = make_episode_stats()
        checks = [
            make_behavior_check_result("x", passed=False),
            make_behavior_check_result("y", passed=False),
        ]
        score = build_behavior_score(stats=stats, objective="o", checks=checks, behavior_metrics={})
        self.assertFalse(score.success)
        self.assertIn("x", score.failures)
        self.assertIn("y", score.failures)

    def test_empty_checks_yields_success(self) -> None:
        stats = make_episode_stats()
        score = build_behavior_score(stats=stats, objective="o", checks=[], behavior_metrics={})
        self.assertTrue(score.success)
        self.assertEqual(score.failures, [])

class AggregateBehaviorScoresTest(unittest.TestCase):
    """Tests for aggregate_behavior_scores."""

    def _make_spec(self, name="check_a") -> BehaviorCheckSpec:
        """
        Create a BehaviorCheckSpec for tests with a sensible default description and expectation.
        
        Parameters:
            name (str): Identifier for the check (default: "check_a").
        
        Returns:
            BehaviorCheckSpec: A spec with the given name, description "desc_<name>", and expected condition ">= 0".
        """
        return BehaviorCheckSpec(name, f"desc_{name}", ">= 0")

    def _make_score(self, passed=True, check_name="check_a", value=1.0) -> BehavioralEpisodeScore:
        """
        Create a BehavioralEpisodeScore test fixture containing a single check and a small set of behavior metrics.
        
        Parameters:
            passed (bool): Whether the single check should be marked as passed.
            check_name (str): Name to assign to the single check.
            value (float): Numeric value to attach to the check and `metric_a`.
        
        Returns:
            BehavioralEpisodeScore: A score with episode=0, seed=1, scenario='s', objective='o', `success` set to `passed`, `checks` containing one `BehaviorCheckResult` keyed by `check_name`, `behavior_metrics` including `"metric_a"` (set to `value`), `"outcome_band"` (`"full_success"` if `passed` else `"stalled_and_died"`), `"partial_progress"` (bool of `passed`), and `"died_without_contact"` (inverse of `passed`), and `failures` containing `[check_name]` when `passed` is False or `[]` when True.
        """
        check = make_behavior_check_result(check_name, passed=passed, value=value)
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
                checks={"check_a": make_behavior_check_result("check_a", passed=False)},
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
                checks={"check_a": make_behavior_check_result("check_a", passed=False)},
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
                checks={"check_a": make_behavior_check_result("check_a", passed=True)},
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

    def test_failure_mode_diagnostics_aggregated_when_present(self) -> None:
        scores = [
            self._make_score(False),
            self._make_score(False),
            self._make_score(True),
        ]
        scores[0].behavior_metrics["failure_mode"] = "left_without_food_signal"
        scores[1].behavior_metrics["failure_mode"] = "left_without_food_signal"
        scores[2].behavior_metrics["failure_mode"] = "success"

        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[self._make_spec()],
        )

        self.assertEqual(
            result["diagnostics"]["primary_failure_mode"],
            "left_without_food_signal",
        )
        self.assertAlmostEqual(
            result["diagnostics"]["failure_mode_distribution"]["left_without_food_signal"],
            2 / 3,
        )
        self.assertAlmostEqual(
            result["diagnostics"]["failure_mode_distribution"]["success"],
            1 / 3,
        )

    def test_diagnostic_rates_are_none_when_metric_is_absent(self) -> None:
        score = BehavioralEpisodeScore(
            episode=0,
            seed=1,
            scenario="s",
            objective="o",
            success=True,
            checks={"check_a": make_behavior_check_result("check_a", passed=True)},
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

class SummarizeBehaviorSuiteTest(unittest.TestCase):
    """Tests for summarize_behavior_suite."""

    def test_empty_suite_returns_zeros(self) -> None:
        result = summarize_behavior_suite({})
        self.assertEqual(result["scenario_count"], 0)
        self.assertEqual(result["episode_count"], 0)
        self.assertEqual(result["scenario_success_rate"], 0.0)
        self.assertEqual(result["episode_success_rate"], 0.0)
        self.assertEqual(result["competence_type"], "mixed")
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

    def test_competence_label_is_recorded(self) -> None:
        suite = {"s1": {"episodes": 1, "success_rate": 1.0, "failures": []}}
        result = summarize_behavior_suite(
            suite,
            competence_label="self_sufficient",
        )
        self.assertEqual(result["competence_type"], "self_sufficient")

    def test_invalid_competence_label_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            summarize_behavior_suite({}, competence_label="assisted")

class FlattenBehaviorRowsTest(unittest.TestCase):
    """Tests for flatten_behavior_rows."""

    def _make_score_with_check(self, passed=True) -> BehavioralEpisodeScore:
        """
        Constructs a BehavioralEpisodeScore containing a single check named "chk_a".
        
        Parameters:
            passed (bool): Whether the check should be marked as passed; when False the returned score will list "chk_a" in `failures`. Defaults to True.
        
        Returns:
            BehavioralEpisodeScore: Score with episode=0, seed=5, scenario="night_rest", objective="obj", success set to `passed`, checks containing the built check under "chk_a" (with `value` equal to `passed`), behavior_metrics={"food_eaten": 1}, and failures either [] or ["chk_a"].
        """
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
        for col in ["reward_profile", "scenario_map", "evaluation_map", "competence_type", "is_primary_benchmark", "eval_reflex_scale", "simulation_seed", "episode_seed", "scenario", "scenario_description", "scenario_objective", "scenario_focus", "episode", "success", "failure_count", "failures"]:
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
        check_a = make_behavior_check_result("fa", passed=False)
        check_b = make_behavior_check_result("fb", passed=False)
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

    def test_competence_columns_derive_from_eval_reflex_scale(self) -> None:
        self_sufficient = flatten_behavior_rows(
            [self._make_score_with_check()],
            reward_profile="classic",
            scenario_map="central_burrow",
            scenario_description="desc",
            scenario_objective="obj",
            scenario_focus="focus",
            evaluation_map="central_burrow",
            simulation_seed=0,
            eval_reflex_scale=0.0,
        )[0]
        scaffolded = flatten_behavior_rows(
            [self._make_score_with_check()],
            reward_profile="classic",
            scenario_map="central_burrow",
            scenario_description="desc",
            scenario_objective="obj",
            scenario_focus="focus",
            evaluation_map="central_burrow",
            simulation_seed=0,
            eval_reflex_scale=0.5,
        )[0]

        self.assertEqual(self_sufficient["competence_type"], "self_sufficient")
        self.assertTrue(self_sufficient["is_primary_benchmark"])
        self.assertEqual(scaffolded["competence_type"], "scaffolded")
        self.assertFalse(scaffolded["is_primary_benchmark"])

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
