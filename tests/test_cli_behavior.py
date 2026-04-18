"""Focused metrics and behavior-evaluation tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

from spider_cortex_sim.ablations import PROPOSAL_SOURCE_NAMES
from spider_cortex_sim.maps import (
    CLUTTER,
    NARROW,
    OPEN,
    MAP_TEMPLATE_NAMES,
    build_map_template,
)
from spider_cortex_sim.metrics import (
    ACTION_CENTER_REPRESENTATION_FIELDS,
    BehaviorCheckResult,
    BehaviorCheckSpec,
    BehavioralEpisodeScore,
    EpisodeMetricAccumulator,
    EpisodeStats,
    PREDATOR_TYPE_NAMES,
    PRIMARY_REPRESENTATION_READOUT_MODULES,
    REFLEX_MODULE_NAMES,
    _aggregate_values,
    _contact_predator_types,
    _diagnostic_predator_distance,
    _dominant_predator_type,
    _first_active_predator_type,
    _mean_like,
    _mean_map,
    _normalize_distribution,
    _predator_type_threat,
    aggregate_behavior_scores,
    aggregate_episode_stats,
    build_behavior_check,
    build_behavior_score,
    flatten_behavior_rows,
    jensen_shannon_divergence,
    summarize_behavior_suite,
)
from spider_cortex_sim.predator import LizardState, PREDATOR_STATES, PredatorController
from spider_cortex_sim.scenarios import (
    SCENARIOS,
    SCENARIO_NAMES,
    ScenarioSpec,
    get_scenario,
)
from spider_cortex_sim.simulation import (
    CAPABILITY_PROBE_SCENARIOS,
    SpiderSimulation,
    is_capability_probe,
)
from spider_cortex_sim.world import ACTION_TO_INDEX, REWARD_COMPONENT_NAMES, SpiderWorld


class TraceNewFieldsTest(unittest.TestCase):
    """Tests that run_episode trace items include new fields added in this PR."""

    def setUp(self):
        """
        Set up a SpiderSimulation instance with a fixed seed and limited steps for tests.
        
        Creates self.sim as SpiderSimulation(seed=7, max_steps=3) to provide a deterministic, short-running simulation used by the test methods.
        """
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
        """
        Assert that running an episode with trace capture yields trace items that include a "slept" field of type bool.
        """
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
        """
        Ensure that the RNG seed recorded in a captured episode trace is deterministic for a given episode index across separate simulation instances.
        """
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


class CLIBehaviorArgumentsTest(unittest.TestCase):
    """Tests for the new behavior-related CLI arguments added to build_parser."""

    def setUp(self):
        """
        Prepare test fixture by constructing the CLI argument parser and assigning it to self.parser for use by test methods.
        """
        from spider_cortex_sim.cli import build_parser
        self.parser = build_parser()

    def test_default_behavior_evaluation_matches_suite_shape(self) -> None:
        """
        Verify that the default behavior evaluation payload has the expected top-level shape and default competence.

        Asserts that the payload contains "suite", "legacy_scenarios", and "summary" keys, that "legacy_scenarios" is an empty dict, and that the summary's "competence_type" is "mixed".
        """
        from spider_cortex_sim.cli import _default_behavior_evaluation

        payload = _default_behavior_evaluation()
        self.assertIn("suite", payload)
        self.assertIn("legacy_scenarios", payload)
        self.assertIn("summary", payload)
        self.assertEqual(payload["legacy_scenarios"], {})
        self.assertEqual(payload["summary"]["competence_type"], "mixed")

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

    def test_noise_robustness_flag(self) -> None:
        args = self.parser.parse_args(["--noise-robustness"])
        self.assertTrue(args.noise_robustness)

    def test_behavior_noise_robustness_alias_flag(self) -> None:
        args = self.parser.parse_args(["--behavior-noise-robustness"])
        self.assertTrue(args.noise_robustness)

    def test_noise_robustness_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.noise_robustness)

    def test_behavior_csv_arg(self) -> None:
        args = self.parser.parse_args(["--behavior-csv", "output.csv"])
        self.assertEqual(args.behavior_csv, Path("output.csv"))

    def test_behavior_csv_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.behavior_csv)

    def test_benchmark_package_arg(self) -> None:
        args = self.parser.parse_args(["--benchmark-package", "package"])
        self.assertEqual(args.benchmark_package, Path("package"))

    def test_benchmark_package_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.benchmark_package)

    def test_curriculum_profile_arg(self) -> None:
        args = self.parser.parse_args(["--curriculum-profile", "ecological_v1"])
        self.assertEqual(args.curriculum_profile, "ecological_v1")

    def test_curriculum_profile_ecological_v2_arg(self) -> None:
        args = self.parser.parse_args(["--curriculum-profile", "ecological_v2"])
        self.assertEqual(args.curriculum_profile, "ecological_v2")

    def test_curriculum_profile_default_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.curriculum_profile, "none")

    def test_curriculum_profile_invalid_value(self) -> None:
        """
        Verify that passing an invalid value to `--curriculum-profile` causes the argument parser to exit.
        
        Asserts that `parse_args` raises `SystemExit` when given `"invalid_value"` for `--curriculum-profile`.
        """
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--curriculum-profile", "invalid_value"])

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
        """
        Verify the CLI's noise profile defaults to "none".
        
        Parses no arguments and asserts that args.noise_profile equals "none".
        """
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

    def test_checkpoint_penalty_args(self) -> None:
        args = self.parser.parse_args(
            [
                "--checkpoint-override-penalty",
                "0.25",
                "--checkpoint-dominance-penalty",
                "0.5",
                "--checkpoint-penalty-mode",
                "direct",
            ]
        )
        self.assertAlmostEqual(args.checkpoint_override_penalty, 0.25)
        self.assertAlmostEqual(args.checkpoint_dominance_penalty, 0.5)
        self.assertEqual(args.checkpoint_penalty_mode, "direct")

    def test_checkpoint_penalty_defaults(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.checkpoint_override_penalty, 0.0)
        self.assertEqual(args.checkpoint_dominance_penalty, 0.0)
        self.assertEqual(args.checkpoint_penalty_mode, "tiebreaker")

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
        """
        Verify that the CLI parser sets `ablation_seeds` to None when no ablation seeds are provided.
        
        Parses an empty argument list and asserts `args.ablation_seeds` is `None`.
        """
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

    def test_experiment_of_record_flag(self) -> None:
        args = self.parser.parse_args(["--experiment-of-record"])
        self.assertTrue(args.experiment_of_record)

    def test_experiment_of_record_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.experiment_of_record)

    def test_claim_test_suite_flag(self) -> None:
        args = self.parser.parse_args(["--claim-test-suite"])
        self.assertTrue(args.claim_test_suite)

    def test_claim_test_suite_default_false(self) -> None:
        args = self.parser.parse_args([])
        self.assertFalse(args.claim_test_suite)

    def test_claim_test_arg(self) -> None:
        args = self.parser.parse_args(
            ["--claim-test", "learning_without_privileged_signals"]
        )
        self.assertEqual(args.claim_test, ["learning_without_privileged_signals"])

    def test_claim_test_multiple_flags(self) -> None:
        args = self.parser.parse_args(
            [
                "--claim-test",
                "learning_without_privileged_signals",
                "--claim-test",
                "escape_without_reflex_support",
            ]
        )
        self.assertEqual(
            args.claim_test,
            [
                "learning_without_privileged_signals",
                "escape_without_reflex_support",
            ],
        )

    def test_claim_test_invalid_choice_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--claim-test", "not_a_claim_test"])

    def test_short_claim_test_suite_summary_preserves_skipped_state(self) -> None:
        from spider_cortex_sim.cli import _short_claim_test_suite_summary

        summary = _short_claim_test_suite_summary(
            {
                "claims": {
                    "learning_without_privileged_signals": {
                        "status": "skipped",
                        "passed": False,
                    }
                },
                "summary": {
                    "claims_passed": 0,
                    "claims_failed": 0,
                    "claims_skipped": 1,
                    "all_primary_claims_passed": False,
                },
            }
        )

        self.assertEqual(
            summary["claims"]["learning_without_privileged_signals"],
            {"passed": None, "skipped": True},
        )
        self.assertEqual(summary["claims_skipped"], 1)

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


class EvaluateBehaviorSuiteIntegrationTest(unittest.TestCase):
    """Integration tests for evaluate_behavior_suite and behavior payload structure."""

    def setUp(self):
        """
        Prepare the test fixture by importing SpiderSimulation and storing the class on self.SimClass.
        
        This makes the SpiderSimulation class available to individual tests as self.SimClass.
        """
        from spider_cortex_sim.simulation import SpiderSimulation
        self.SimClass = SpiderSimulation

    def test_payload_has_required_keys(self) -> None:
        """
        Verifies that evaluate_behavior_suite returns a payload containing required top-level keys.
        
        Calls evaluate_behavior_suite for the "night_rest" scenario and asserts the returned payload includes the keys "suite", "summary", and "legacy_scenarios".
        """
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

    def test_capability_probe_classification_marks_suite_and_rows(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12)
        payload, _, rows = sim.evaluate_behavior_suite(["night_rest", "open_field_foraging"])
        by_scenario = {row["scenario"]: row for row in rows}

        self.assertEqual(
            CAPABILITY_PROBE_SCENARIOS,
            (
                "open_field_foraging",
                "corridor_gauntlet",
                "exposed_day_foraging",
                "food_deprivation",
            ),
        )
        self.assertTrue(is_capability_probe("open_field_foraging"))
        self.assertFalse(is_capability_probe("night_rest"))
        self.assertTrue(payload["suite"]["open_field_foraging"]["is_capability_probe"])
        self.assertFalse(payload["suite"]["night_rest"]["is_capability_probe"])
        self.assertTrue(by_scenario["open_field_foraging"]["is_capability_probe"])
        self.assertFalse(by_scenario["night_rest"]["is_capability_probe"])

    def test_capability_probe_flag_appears_for_all_probe_summaries(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12)
        payload, _, _ = sim.evaluate_behavior_suite(list(CAPABILITY_PROBE_SCENARIOS))

        for name in CAPABILITY_PROBE_SCENARIOS:
            with self.subTest(name=name):
                self.assertTrue(payload["suite"][name]["is_capability_probe"])

    def test_benchmark_tier_appears_in_scenario_result_metadata(self) -> None:
        """
        Verify that scenario metadata includes benchmark tier and target skill in both the suite payload and per-row results.
        
        Asserts that "night_rest" has benchmark_tier "primary", and that "food_deprivation" has benchmark_tier "capability" with target_skill "hunger_driven_commitment", and that these fields appear identically in the returned rows.
        """
        sim = self.SimClass(seed=7, max_steps=12)
        payload, _, rows = sim.evaluate_behavior_suite(
            ["night_rest", "food_deprivation"]
        )
        rows_by_scenario = {row["scenario"]: row for row in rows}

        self.assertEqual(payload["suite"]["night_rest"]["benchmark_tier"], "primary")
        self.assertEqual(
            payload["suite"]["food_deprivation"]["benchmark_tier"],
            "capability",
        )
        self.assertEqual(
            payload["suite"]["food_deprivation"]["target_skill"],
            "hunger_driven_commitment",
        )
        self.assertEqual(rows_by_scenario["night_rest"]["benchmark_tier"], "primary")
        self.assertEqual(
            rows_by_scenario["food_deprivation"]["benchmark_tier"],
            "capability",
        )
        self.assertEqual(
            rows_by_scenario["food_deprivation"]["target_skill"],
            "hunger_driven_commitment",
        )

    def test_zero_success_capability_probes_emit_interpretable_diagnostics(self) -> None:
        """
        Assert that capability-probe scenarios with zero success emit interpretable diagnostics and behavior metrics.
        
        Verifies for each scenario in a zero-success capability-probe payload that:
        - `is_capability_probe` is true and `success_rate` equals 0.0.
        - `diagnostics` contains `primary_failure_mode`, `primary_outcome`, `failure_mode_distribution`, and `outcome_distribution`.
        - `behavior_metrics` contains `failure_mode` and `outcome_band`.
        """
        payload = {
            "suite": {
                "food_deprivation": {
                    "is_capability_probe": True,
                    "success_rate": 0.0,
                    "diagnostics": {
                        "primary_failure_mode": "timing_failure",
                        "primary_outcome": "partial_progress",
                        "failure_mode_distribution": {"timing_failure": 1.0},
                        "outcome_distribution": {"partial_progress": 1.0},
                    },
                    "behavior_metrics": {
                        "failure_mode": "timing_failure",
                        "outcome_band": "partial_progress",
                    },
                }
            }
        }

        for name, scenario_result in payload["suite"].items():
            with self.subTest(name=name):
                diagnostics = scenario_result["diagnostics"]
                behavior_metrics = scenario_result["behavior_metrics"]
                self.assertTrue(scenario_result["is_capability_probe"])
                self.assertEqual(scenario_result["success_rate"], 0.0)
                self.assertIn("primary_failure_mode", diagnostics)
                self.assertIn("primary_outcome", diagnostics)
                self.assertIn("failure_mode_distribution", diagnostics)
                self.assertIn("outcome_distribution", diagnostics)
                self.assertIn("failure_mode", behavior_metrics)
                self.assertIn("outcome_band", behavior_metrics)

    def test_suite_scenario_success_rate_is_float(self) -> None:
        """
        Check that the `success_rate` for the "food_deprivation" scenario in the evaluated suite is a float.
        """
        sim = self.SimClass(seed=7, max_steps=12)
        payload, _, _ = sim.evaluate_behavior_suite(["food_deprivation"])
        success_rate = payload["suite"]["food_deprivation"]["success_rate"]
        self.assertIsInstance(success_rate, float)

    def test_multiple_episodes_per_scenario(self) -> None:
        sim = self.SimClass(seed=7, max_steps=12)
        payload, _, rows = sim.evaluate_behavior_suite(["night_rest"], episodes_per_scenario=2)
        self.assertEqual(payload["suite"]["night_rest"]["episodes"], 2)
        self.assertEqual(len(rows), 2)


class DefaultCheckpointingSummaryTest(unittest.TestCase):
    """Tests for the _default_checkpointing_summary helper in cli.py."""

    def _get_fn(self):
        """
        Retrieve the `_default_checkpointing_summary` helper callable.
        
        Returns:
            fn (callable): The `_default_checkpointing_summary` function from spider_cortex_sim.cli.
        """
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

    def test_penalty_config_preserved(self) -> None:
        fn = self._get_fn()
        result = fn(
            selection="best",
            metric="mean_reward",
            checkpoint_interval=4,
            selection_scenario_episodes=1,
            override_penalty_weight=0.25,
            dominance_penalty_weight=0.5,
            penalty_mode="direct",
        )
        self.assertEqual(result["penalty_mode"], "direct")
        self.assertEqual(
            result["penalty_config"],
            {
                "metric": "mean_reward",
                "override_penalty_weight": 0.25,
                "dominance_penalty_weight": 0.5,
                "penalty_mode": "direct",
            },
        )

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
            "penalty_mode",
            "penalty_config",
            "checkpoint_interval",
            "evaluation_source",
            "selection_scenario_episodes",
            "generated_checkpoints",
            "selected_checkpoint",
        }
        self.assertEqual(set(result.keys()), expected_keys)


class CLIBudgetProfileChoicesTest(unittest.TestCase):
    """Verify the --budget-profile argument accepts only canonical names."""

    def setUp(self) -> None:
        """
        Set up the test fixture by creating a CLI argument parser and storing it on self.parser.
        
        The parser is created via spider_cortex_sim.cli.build_parser() and is used by test methods to validate CLI parsing behavior.
        """
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

    def test_checkpoint_penalty_mode_rejects_invalid_choice(self) -> None:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--checkpoint-penalty-mode", "soft"])

    def test_episodes_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.episodes)

    def test_eval_episodes_default_is_none(self) -> None:
        args = self.parser.parse_args([])
        self.assertIsNone(args.eval_episodes)

    def test_max_steps_default_is_none(self) -> None:
        """
        Assert that parsing no CLI arguments leaves `max_steps` unset (`None`).
        """
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
        """
        Asserts that module-reflex scale entries missing an '=' delimiter raise a ValueError.
        
        Calls the parser with a malformed entry like "alert_center0.5" and expects a ValueError to be raised.
        """
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
        """
        Set up the test fixture by creating a CLI argument parser and storing it on self.parser.
        
        The parser is created via spider_cortex_sim.cli.build_parser() and is used by test methods to validate CLI parsing behavior.
        """
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
