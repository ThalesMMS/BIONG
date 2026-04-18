"""Tests for behavior evaluation pipeline introduced/changed in this PR.

Covers the complete workflow from EpisodeStats → BehavioralEpisodeScore →
aggregate_behavior_scores → summarize_behavior_suite → flatten_behavior_rows,
focusing on:
- Correct scenario/objective threading through the pipeline
- Competence-label derivation from eval_reflex_scale
- is_primary_benchmark flag semantics
- Diagnostic aggregation (outcome_band, failure_mode) end-to-end
- Suite-level regression detection and episode-weighted success rate
- flatten_behavior_rows column coverage and metric prefix convention
- Edge cases: empty suite, single episode, all-pass, all-fail
"""

from __future__ import annotations

import unittest
from dataclasses import asdict

from spider_cortex_sim.metrics import (
    BehaviorCheckResult,
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeStats,
    aggregate_behavior_scores,
    build_behavior_check,
    build_behavior_score,
    competence_label_from_eval_reflex_scale,
    flatten_behavior_rows,
    summarize_behavior_suite,
)
from spider_cortex_sim.world import REWARD_COMPONENT_NAMES


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_episode_stats(**overrides) -> EpisodeStats:
    from spider_cortex_sim.predator import PREDATOR_STATES

    defaults: dict = dict(
        episode=0,
        seed=42,
        training=False,
        scenario="night_rest",
        total_reward=1.0,
        steps=100,
        food_eaten=2,
        sleep_events=1,
        shelter_entries=3,
        alert_events=0,
        predator_contacts=0,
        predator_sightings=0,
        predator_escapes=0,
        night_ticks=20,
        night_shelter_ticks=18,
        night_still_ticks=16,
        night_role_ticks={"outside": 2, "entrance": 0, "inside": 2, "deep": 16},
        night_shelter_occupancy_rate=0.9,
        night_stillness_rate=0.8,
        night_role_distribution={
            "outside": 0.1, "entrance": 0.0, "inside": 0.1, "deep": 0.8
        },
        predator_response_events=0,
        mean_predator_response_latency=0.0,
        mean_sleep_debt=0.2,
        food_distance_delta=3.0,
        shelter_distance_delta=1.0,
        final_hunger=0.3,
        final_fatigue=0.2,
        final_sleep_debt=0.1,
        final_health=1.0,
        alive=True,
        reward_component_totals={k: 0.0 for k in REWARD_COMPONENT_NAMES},
        predator_state_ticks={s: 0 for s in PREDATOR_STATES},
        predator_mode_transitions=0,
        dominant_predator_state="PATROL",
    )
    defaults.update(overrides)
    return EpisodeStats(**defaults)


def _make_check_spec(name: str = "check_a", expected: str = ">= 0.8") -> BehaviorCheckSpec:
    return BehaviorCheckSpec(name, f"Description for {name}", expected)


def _make_check_result(
    name: str = "check_a",
    passed: bool = True,
    value: object = 0.9,
) -> BehaviorCheckResult:
    return BehaviorCheckResult(
        name=name,
        description=f"Description for {name}",
        expected=">= 0.8",
        passed=passed,
        value=value,
    )


def _make_score(
    episode: int = 0,
    seed: int = 1,
    scenario: str = "night_rest",
    objective: str = "survive_night",
    passed: bool = True,
    check_name: str = "check_a",
    value: float = 0.9,
    behavior_metrics: dict | None = None,
) -> BehavioralEpisodeScore:
    check = _make_check_result(check_name, passed=passed, value=value)
    return BehavioralEpisodeScore(
        episode=episode,
        seed=seed,
        scenario=scenario,
        objective=objective,
        success=passed,
        checks={check_name: check},
        behavior_metrics=behavior_metrics if behavior_metrics is not None else {},
        failures=[] if passed else [check_name],
    )


# ---------------------------------------------------------------------------
# build_behavior_score integration tests
# ---------------------------------------------------------------------------

class BuildBehaviorScoreIntegrationTest(unittest.TestCase):
    """Integration-level tests for build_behavior_score."""

    def test_full_round_trip_from_episode_stats(self) -> None:
        stats = _make_episode_stats(episode=3, seed=77, scenario="predator_edge")
        spec = _make_check_spec("night_shelter_rate", ">= 0.8")
        check = build_behavior_check(
            spec,
            passed=stats.night_shelter_occupancy_rate >= 0.8,
            value=stats.night_shelter_occupancy_rate,
        )
        score = build_behavior_score(
            stats=stats,
            objective="shelter_survival",
            checks=[check],
            behavior_metrics={"night_shelter_rate": stats.night_shelter_occupancy_rate},
        )
        self.assertEqual(score.episode, 3)
        self.assertEqual(score.seed, 77)
        self.assertEqual(score.scenario, "predator_edge")
        self.assertEqual(score.objective, "shelter_survival")
        self.assertTrue(score.success)
        self.assertIn("night_shelter_rate", score.checks)
        self.assertAlmostEqual(
            score.behavior_metrics["night_shelter_rate"],
            stats.night_shelter_occupancy_rate,
        )

    def test_multi_check_success_requires_all_pass(self) -> None:
        stats = _make_episode_stats()
        checks = [
            build_behavior_check(_make_check_spec("a"), passed=True, value=1.0),
            build_behavior_check(_make_check_spec("b"), passed=False, value=0.3),
            build_behavior_check(_make_check_spec("c"), passed=True, value=0.9),
        ]
        score = build_behavior_score(
            stats=stats, objective="o", checks=checks, behavior_metrics={}
        )
        self.assertFalse(score.success)
        self.assertIn("b", score.failures)
        self.assertNotIn("a", score.failures)
        self.assertNotIn("c", score.failures)

    def test_none_scenario_becomes_default(self) -> None:
        stats = _make_episode_stats(scenario=None)
        score = build_behavior_score(
            stats=stats, objective="o", checks=[], behavior_metrics={}
        )
        self.assertEqual(score.scenario, "default")

    def test_checks_dict_keyed_by_name(self) -> None:
        stats = _make_episode_stats()
        spec_a = _make_check_spec("alpha")
        spec_b = _make_check_spec("beta")
        checks = [
            build_behavior_check(spec_a, passed=True, value=1.0),
            build_behavior_check(spec_b, passed=True, value=0.95),
        ]
        score = build_behavior_score(
            stats=stats, objective="o", checks=checks, behavior_metrics={}
        )
        self.assertIn("alpha", score.checks)
        self.assertIn("beta", score.checks)

    def test_empty_checks_yields_successful_score(self) -> None:
        stats = _make_episode_stats()
        score = build_behavior_score(
            stats=stats, objective="o", checks=[], behavior_metrics={}
        )
        self.assertTrue(score.success)
        self.assertEqual(score.failures, [])


# ---------------------------------------------------------------------------
# aggregate_behavior_scores integration tests
# ---------------------------------------------------------------------------

class AggregateBehaviorScoresPipelineTest(unittest.TestCase):
    """Tests for aggregate_behavior_scores with realistic multi-episode data."""

    def _scenario_scores(
        self,
        n_pass: int,
        n_fail: int,
        scenario: str = "night_rest",
    ) -> list[BehavioralEpisodeScore]:
        scores = []
        for i in range(n_pass):
            scores.append(
                _make_score(
                    episode=i,
                    seed=i + 1,
                    scenario=scenario,
                    passed=True,
                    behavior_metrics={
                        "night_shelter_rate": 0.9,
                        "outcome_band": "full_success",
                        "partial_progress": True,
                        "died_without_contact": False,
                    },
                )
            )
        for i in range(n_fail):
            scores.append(
                _make_score(
                    episode=n_pass + i,
                    seed=n_pass + i + 1,
                    scenario=scenario,
                    passed=False,
                    behavior_metrics={
                        "night_shelter_rate": 0.4,
                        "outcome_band": "stalled_and_died",
                        "partial_progress": False,
                        "died_without_contact": True,
                    },
                )
            )
        return scores

    def test_all_pass_gives_success_rate_one(self) -> None:
        scores = self._scenario_scores(5, 0)
        result = aggregate_behavior_scores(
            scores,
            scenario="night_rest",
            description="Night rest",
            objective="survive_night",
            check_specs=[_make_check_spec()],
        )
        self.assertAlmostEqual(result["success_rate"], 1.0)

    def test_all_fail_gives_success_rate_zero(self) -> None:
        scores = self._scenario_scores(0, 5)
        result = aggregate_behavior_scores(
            scores,
            scenario="night_rest",
            description="Night rest",
            objective="survive_night",
            check_specs=[_make_check_spec()],
        )
        self.assertAlmostEqual(result["success_rate"], 0.0)

    def test_mixed_gives_partial_success_rate(self) -> None:
        scores = self._scenario_scores(3, 1)
        result = aggregate_behavior_scores(
            scores,
            scenario="night_rest",
            description="desc",
            objective="obj",
            check_specs=[_make_check_spec()],
        )
        self.assertAlmostEqual(result["success_rate"], 0.75)

    def test_episodes_count_matches_input(self) -> None:
        scores = self._scenario_scores(4, 2)
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[_make_check_spec()],
        )
        self.assertEqual(result["episodes"], 6)

    def test_check_pass_rate_computed_correctly(self) -> None:
        scores = self._scenario_scores(3, 1)
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[_make_check_spec()],
        )
        self.assertAlmostEqual(result["checks"]["check_a"]["pass_rate"], 3 / 4)

    def test_behavior_metrics_aggregated_as_mean(self) -> None:
        scores = self._scenario_scores(2, 2)
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[_make_check_spec()],
        )
        # 2 pass (0.9) + 2 fail (0.4) → mean = 0.65
        self.assertAlmostEqual(
            result["behavior_metrics"]["night_shelter_rate"], 0.65
        )

    def test_failures_are_unique_and_sorted(self) -> None:
        scores = [
            _make_score(passed=False, check_name="check_a"),
            _make_score(passed=False, check_name="check_a"),
        ]
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[_make_check_spec()],
        )
        self.assertEqual(result["failures"], ["check_a"])

    def test_diagnostic_outcome_band_aggregated(self) -> None:
        scores = self._scenario_scores(1, 3)
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[_make_check_spec()],
        )
        self.assertEqual(
            result["diagnostics"]["primary_outcome"], "stalled_and_died"
        )

    def test_diagnostic_partial_progress_rate(self) -> None:
        scores = self._scenario_scores(2, 2)
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[_make_check_spec()],
        )
        # 2 pass have partial_progress=True, 2 fail have partial_progress=False
        self.assertAlmostEqual(result["diagnostics"]["partial_progress_rate"], 0.5)

    def test_diagnostic_died_without_contact_rate(self) -> None:
        scores = self._scenario_scores(2, 2)
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[_make_check_spec()],
        )
        # Only fail episodes have died_without_contact=True → 2/4 = 0.5
        self.assertAlmostEqual(
            result["diagnostics"]["died_without_contact_rate"], 0.5
        )

    def test_episodes_detail_length_matches_input(self) -> None:
        scores = self._scenario_scores(3, 2)
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[_make_check_spec()],
        )
        self.assertEqual(len(result["episodes_detail"]), 5)

    def test_episodes_detail_contains_required_fields(self) -> None:
        scores = self._scenario_scores(1, 1)
        result = aggregate_behavior_scores(
            scores,
            scenario="s",
            description="d",
            objective="o",
            check_specs=[_make_check_spec()],
        )
        for ep_detail in result["episodes_detail"]:
            for field in ("episode", "seed", "success", "failures"):
                self.assertIn(field, ep_detail)

    def test_legacy_metrics_passed_through(self) -> None:
        result = aggregate_behavior_scores(
            [],
            scenario="s",
            description="d",
            objective="o",
            check_specs=[],
            legacy_metrics={"mean_reward": 2.5},
        )
        self.assertEqual(result["legacy_metrics"]["mean_reward"], 2.5)

    def test_optional_diagnostic_metadata_threaded_correctly(self) -> None:
        result = aggregate_behavior_scores(
            [],
            scenario="my_scenario",
            description="my desc",
            objective="my obj",
            diagnostic_focus="predator_response",
            success_interpretation="agent survived night",
            failure_interpretation="agent died",
            budget_note="reflexes disabled",
            check_specs=[],
        )
        self.assertEqual(result["diagnostic_focus"], "predator_response")
        self.assertEqual(result["success_interpretation"], "agent survived night")
        self.assertEqual(result["failure_interpretation"], "agent died")
        self.assertEqual(result["budget_note"], "reflexes disabled")


# ---------------------------------------------------------------------------
# summarize_behavior_suite integration tests
# ---------------------------------------------------------------------------

class SummarizeBehaviorSuiteIntegrationTest(unittest.TestCase):
    """Integration tests for summarize_behavior_suite."""

    def _suite(self, *scenario_specs: tuple) -> dict:
        """Build a fake suite dict from (name, episodes, success_rate, failures) tuples."""
        return {
            name: {
                "episodes": episodes,
                "success_rate": success_rate,
                "failures": failures,
            }
            for name, episodes, success_rate, failures in scenario_specs
        }

    def test_all_pass_reports_full_scenario_and_episode_success(self) -> None:
        suite = self._suite(
            ("night_rest", 5, 1.0, []),
            ("predator_edge", 4, 1.0, []),
        )
        result = summarize_behavior_suite(suite)
        self.assertAlmostEqual(result["scenario_success_rate"], 1.0)
        # Episode success: (5*1.0 + 4*1.0) / 9 = 1.0
        self.assertAlmostEqual(result["episode_success_rate"], 1.0)
        self.assertEqual(result["regressions"], [])

    def test_single_failing_scenario_reported_as_regression(self) -> None:
        suite = self._suite(
            ("night_rest", 5, 1.0, []),
            ("predator_edge", 3, 0.0, ["check_a"]),
        )
        result = summarize_behavior_suite(suite)
        regression_names = [r["scenario"] for r in result["regressions"]]
        self.assertIn("predator_edge", regression_names)
        self.assertNotIn("night_rest", regression_names)

    def test_episode_success_rate_is_episode_weighted(self) -> None:
        suite = self._suite(
            ("s1", 10, 1.0, []),
            ("s2", 10, 0.0, ["f1"]),
        )
        result = summarize_behavior_suite(suite)
        # 10*1.0 + 10*0.0 = 10/20 = 0.5
        self.assertAlmostEqual(result["episode_success_rate"], 0.5)

    def test_scenario_success_rate_is_count_weighted(self) -> None:
        suite = self._suite(
            ("s1", 2, 1.0, []),
            ("s2", 100, 1.0, []),
        )
        result = summarize_behavior_suite(suite)
        # Both pass → 2/2 = 1.0
        self.assertAlmostEqual(result["scenario_success_rate"], 1.0)

    def test_partial_success_not_counted_as_full_scenario_pass(self) -> None:
        suite = self._suite(("s1", 5, 0.6, ["f1"]))
        result = summarize_behavior_suite(suite)
        self.assertAlmostEqual(result["scenario_success_rate"], 0.0)

    def test_competence_label_threaded_into_summary(self) -> None:
        suite = self._suite(("s1", 3, 1.0, []))
        result = summarize_behavior_suite(suite, competence_label="scaffolded")
        self.assertEqual(result["competence_type"], "scaffolded")

    def test_default_competence_label_is_mixed(self) -> None:
        result = summarize_behavior_suite({})
        self.assertEqual(result["competence_type"], "mixed")

    def test_invalid_competence_label_raises(self) -> None:
        with self.assertRaises(ValueError):
            summarize_behavior_suite({}, competence_label="invalid_label")

    def test_empty_suite_zeros(self) -> None:
        result = summarize_behavior_suite({})
        self.assertEqual(result["scenario_count"], 0)
        self.assertEqual(result["episode_count"], 0)
        self.assertAlmostEqual(result["scenario_success_rate"], 0.0)
        self.assertAlmostEqual(result["episode_success_rate"], 0.0)

    def test_regression_entry_has_scenario_and_failures(self) -> None:
        suite = self._suite(("s1", 2, 0.0, ["broken_check"]))
        result = summarize_behavior_suite(suite)
        reg = result["regressions"][0]
        self.assertEqual(reg["scenario"], "s1")
        self.assertEqual(reg["failures"], ["broken_check"])


# ---------------------------------------------------------------------------
# flatten_behavior_rows integration tests
# ---------------------------------------------------------------------------

class FlattenBehaviorRowsIntegrationTest(unittest.TestCase):
    """Integration tests for flatten_behavior_rows."""

    def _build_scores(self, n: int = 2) -> list[BehavioralEpisodeScore]:
        return [
            _make_score(
                episode=i,
                seed=i + 10,
                passed=(i % 2 == 0),
                behavior_metrics={
                    "food_eaten": float(i + 1),
                    "night_shelter_rate": 0.9 if i % 2 == 0 else 0.3,
                },
            )
            for i in range(n)
        ]

    def _common_kwargs(self, **overrides) -> dict:
        defaults = dict(
            reward_profile="classic",
            scenario_map="central_burrow",
            scenario_description="Survive the night in the central burrow",
            scenario_objective="shelter_at_night",
            scenario_focus="night_shelter",
            evaluation_map="central_burrow",
            simulation_seed=1,
        )
        defaults.update(overrides)
        return defaults

    def test_one_row_per_score(self) -> None:
        scores = self._build_scores(4)
        rows = flatten_behavior_rows(scores, **self._common_kwargs())
        self.assertEqual(len(rows), 4)

    def test_empty_scores_returns_empty_list(self) -> None:
        rows = flatten_behavior_rows([], **self._common_kwargs())
        self.assertEqual(rows, [])

    def test_required_columns_present(self) -> None:
        scores = self._build_scores(1)
        row = flatten_behavior_rows(scores, **self._common_kwargs())[0]
        required = [
            "reward_profile", "scenario_map", "evaluation_map",
            "competence_type", "is_primary_benchmark",
            "eval_reflex_scale", "simulation_seed", "episode_seed",
            "scenario", "scenario_description", "scenario_objective",
            "scenario_focus", "episode", "success",
            "failure_count", "failures",
        ]
        for col in required:
            self.assertIn(col, row, f"Column {col!r} missing")

    def test_reward_profile_and_maps_stored(self) -> None:
        scores = self._build_scores(1)
        row = flatten_behavior_rows(
            scores,
            **self._common_kwargs(
                reward_profile="ecological",
                scenario_map="two_shelters",
                evaluation_map="side_burrow",
                simulation_seed=77,
            ),
        )[0]
        self.assertEqual(row["reward_profile"], "ecological")
        self.assertEqual(row["scenario_map"], "two_shelters")
        self.assertEqual(row["evaluation_map"], "side_burrow")
        self.assertEqual(row["simulation_seed"], 77)

    def test_episode_seed_comes_from_score(self) -> None:
        score = _make_score(seed=99)
        rows = flatten_behavior_rows([score], **self._common_kwargs())
        self.assertEqual(rows[0]["episode_seed"], 99)

    def test_success_and_failure_count_accurate(self) -> None:
        pass_score = _make_score(passed=True)
        fail_score = _make_score(passed=False)
        rows = flatten_behavior_rows([pass_score, fail_score], **self._common_kwargs())
        self.assertTrue(rows[0]["success"])
        self.assertEqual(rows[0]["failure_count"], 0)
        self.assertFalse(rows[1]["success"])
        self.assertEqual(rows[1]["failure_count"], 1)

    def test_metric_columns_prefixed_with_metric(self) -> None:
        score = _make_score(
            behavior_metrics={"food_eaten": 3, "night_shelter_rate": 0.8}
        )
        row = flatten_behavior_rows([score], **self._common_kwargs())[0]
        self.assertIn("metric_food_eaten", row)
        self.assertIn("metric_night_shelter_rate", row)

    def test_check_columns_present_for_each_check(self) -> None:
        score = _make_score(check_name="my_check")
        row = flatten_behavior_rows([score], **self._common_kwargs())[0]
        self.assertIn("check_my_check_passed", row)
        self.assertIn("check_my_check_value", row)
        self.assertIn("check_my_check_expected", row)

    def test_scenario_description_objective_focus_columns(self) -> None:
        scores = self._build_scores(1)
        row = flatten_behavior_rows(
            scores,
            **self._common_kwargs(
                scenario_description="Survive for 100 ticks",
                scenario_objective="night_survival",
                scenario_focus="predator_avoidance",
            ),
        )[0]
        self.assertEqual(row["scenario_description"], "Survive for 100 ticks")
        self.assertEqual(row["scenario_objective"], "night_survival")
        self.assertEqual(row["scenario_focus"], "predator_avoidance")

    def test_self_sufficient_competence_when_reflex_scale_zero(self) -> None:
        scores = self._build_scores(1)
        row = flatten_behavior_rows(
            scores,
            **self._common_kwargs(),
            eval_reflex_scale=0.0,
        )[0]
        self.assertEqual(row["competence_type"], "self_sufficient")
        self.assertTrue(row["is_primary_benchmark"])

    def test_scaffolded_competence_when_reflex_scale_positive(self) -> None:
        scores = self._build_scores(1)
        row = flatten_behavior_rows(
            scores,
            **self._common_kwargs(),
            eval_reflex_scale=0.75,
        )[0]
        self.assertEqual(row["competence_type"], "scaffolded")
        self.assertFalse(row["is_primary_benchmark"])

    def test_mixed_competence_when_reflex_scale_none(self) -> None:
        scores = self._build_scores(1)
        row = flatten_behavior_rows(
            scores,
            **self._common_kwargs(),
        )[0]
        # Default eval_reflex_scale is None → mixed
        self.assertEqual(row["competence_type"], "mixed")

    def test_failures_listed_in_row(self) -> None:
        score = _make_score(passed=False, check_name="bad_check")
        row = flatten_behavior_rows([score], **self._common_kwargs())[0]
        self.assertIn("bad_check", row["failures"])

    def test_multiple_checks_all_columns_present(self) -> None:
        checks = {
            "chk_x": _make_check_result("chk_x", passed=True, value=1.0),
            "chk_y": _make_check_result("chk_y", passed=False, value=0.2),
        }
        score = BehavioralEpisodeScore(
            episode=0,
            seed=1,
            scenario="night_rest",
            objective="o",
            success=False,
            checks=checks,
            behavior_metrics={},
            failures=["chk_y"],
        )
        row = flatten_behavior_rows([score], **self._common_kwargs())[0]
        for col in ("check_chk_x_passed", "check_chk_y_passed",
                    "check_chk_x_value", "check_chk_y_value"):
            self.assertIn(col, row, f"Missing column {col!r}")

    def test_metric_values_preserved(self) -> None:
        score = _make_score(
            behavior_metrics={"food_eaten": 5, "night_shelter_rate": 0.95}
        )
        row = flatten_behavior_rows([score], **self._common_kwargs())[0]
        self.assertEqual(row["metric_food_eaten"], 5)
        self.assertAlmostEqual(row["metric_night_shelter_rate"], 0.95)


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

class EndToEndBehaviorEvaluationPipelineTest(unittest.TestCase):
    """Smoke test for the full pipeline: stats → scores → aggregation → suite."""

    def test_full_pipeline_with_two_scenarios(self) -> None:
        night_rest_scores = []
        predator_edge_scores = []

        for i in range(5):
            stats = _make_episode_stats(
                episode=i, seed=i + 1, scenario="night_rest",
                night_shelter_occupancy_rate=0.85 if i < 4 else 0.5,
                alive=True,
            )
            passed = stats.night_shelter_occupancy_rate >= 0.8
            spec = _make_check_spec("shelter_rate", ">= 0.8")
            check = build_behavior_check(
                spec,
                passed=passed,
                value=stats.night_shelter_occupancy_rate,
            )
            score = build_behavior_score(
                stats=stats,
                objective="shelter_survival",
                checks=[check],
                behavior_metrics={
                    "shelter_rate": stats.night_shelter_occupancy_rate,
                    "outcome_band": "full_success" if passed else "stalled_and_died",
                    "partial_progress": passed,
                    "died_without_contact": not passed,
                },
            )
            night_rest_scores.append(score)

        for i in range(3):
            stats = _make_episode_stats(
                episode=i, seed=i + 10, scenario="predator_edge",
                predator_escapes=1 if i == 0 else 0,
                alive=i == 0,
            )
            passed = stats.alive and stats.predator_escapes > 0
            spec = _make_check_spec("escaped_predator", "true")
            check = build_behavior_check(spec, passed=passed, value=passed)
            score = build_behavior_score(
                stats=stats,
                objective="predator_survival",
                checks=[check],
                behavior_metrics={
                    "escaped_predator": passed,
                    "outcome_band": "full_success" if passed else "stalled_and_died",
                    "partial_progress": stats.alive,
                    "died_without_contact": not stats.alive,
                },
            )
            predator_edge_scores.append(score)

        night_rest_agg = aggregate_behavior_scores(
            night_rest_scores,
            scenario="night_rest",
            description="Survive the night",
            objective="shelter_survival",
            check_specs=[_make_check_spec("shelter_rate", ">= 0.8")],
        )
        predator_edge_agg = aggregate_behavior_scores(
            predator_edge_scores,
            scenario="predator_edge",
            description="Escape predator",
            objective="predator_survival",
            check_specs=[_make_check_spec("escaped_predator", "true")],
        )

        self.assertAlmostEqual(night_rest_agg["success_rate"], 4 / 5)
        self.assertAlmostEqual(predator_edge_agg["success_rate"], 1 / 3)

        suite = {
            "night_rest": night_rest_agg,
            "predator_edge": predator_edge_agg,
        }
        summary = summarize_behavior_suite(suite, competence_label="self_sufficient")

        self.assertEqual(summary["scenario_count"], 2)
        self.assertEqual(summary["episode_count"], 8)
        self.assertEqual(summary["competence_type"], "self_sufficient")

        # predator_edge has success_rate < 1.0 → regression
        regression_names = [r["scenario"] for r in summary["regressions"]]
        self.assertIn("predator_edge", regression_names)

        # flatten rows for both scenarios
        night_rows = flatten_behavior_rows(
            night_rest_scores,
            reward_profile="classic",
            scenario_map="central_burrow",
            scenario_description="Survive the night",
            scenario_objective="shelter_survival",
            scenario_focus="night_shelter",
            evaluation_map="central_burrow",
            simulation_seed=42,
            eval_reflex_scale=0.0,
        )
        self.assertEqual(len(night_rows), 5)
        for row in night_rows:
            self.assertEqual(row["competence_type"], "self_sufficient")
            self.assertTrue(row["is_primary_benchmark"])
            self.assertIn("check_shelter_rate_passed", row)


if __name__ == "__main__":
    unittest.main()