import unittest
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import canonical_ablation_configs
from spider_cortex_sim.comparison import (
    build_ablation_deltas,
    build_learning_evidence_deltas,
    build_learning_evidence_summary,
    build_predator_type_specialization_summary,
    build_reward_audit,
    build_reward_audit_comparison,
    austere_survival_gate_passed,
    compare_ablation_suite,
    compare_behavior_suite,
    compare_configurations,
    compare_learning_evidence,
    compare_noise_robustness,
    compare_reward_profiles,
    compare_training_regimes,
    condition_compact_summary,
    profile_comparison_metrics,
)
from spider_cortex_sim.curriculum import CURRICULUM_FOCUS_SCENARIOS
from spider_cortex_sim.learning_evidence import LearningEvidenceConditionSpec
from spider_cortex_sim.noise import RobustnessMatrixSpec
from spider_cortex_sim.reward import SCENARIO_AUSTERE_REQUIREMENTS, SHAPING_GAP_POLICY
from spider_cortex_sim.simulation import SpiderSimulation

class CompareAblationSuiteAlwaysIncludesReferenceTest(unittest.TestCase):
    """Tests that compare_ablation_suite always includes modular_full as reference."""

    def test_empty_seeds_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty seeds"):
            compare_ablation_suite(
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                seeds=(),
            )

    def test_reference_always_present_when_only_variant_requested(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["no_module_dropout"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertIn("modular_full", payload["variants"])
        self.assertEqual(payload["reference_variant"], "modular_full")

    def test_reference_not_duplicated_when_explicitly_requested(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["modular_full", "monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        variant_keys = list(payload["variants"].keys())
        self.assertEqual(variant_keys.count("modular_full"), 1)

    def test_true_monolithic_policy_present_in_canonical_variants(self) -> None:
        self.assertIn("true_monolithic_policy", canonical_ablation_configs())

    def test_deltas_always_has_reference_entry(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["no_module_reflexes"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertIn("modular_full", payload["deltas_vs_reference"])

    def test_scenario_names_in_payload(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest", "predator_edge"),
            seeds=(7,),
        )
        self.assertEqual(payload["scenario_names"], ["night_rest", "predator_edge"])

    def test_seeds_in_payload(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7, 17),
        )
        self.assertEqual(payload["seeds"], [7, 17])

    def test_rows_contain_ablation_columns(self) -> None:
        _, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["local_credit_only", "monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("ablation_variant", row)
            self.assertIn("ablation_architecture", row)
            self.assertIn("metric_module_contribution_alert_center", row)
            self.assertIn("metric_dominant_module", row)
            self.assertIn("metric_effective_module_count", row)

    def test_payload_includes_local_credit_only_variant(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["local_credit_only"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertIn("local_credit_only", payload["variants"])
        self.assertEqual(
            payload["variants"]["local_credit_only"]["config"]["name"],
            "local_credit_only",
        )
        self.assertIn(
            "mean_module_contribution_share",
            payload["variants"]["local_credit_only"]["legacy_scenarios"]["night_rest"],
        )

    def test_true_monolithic_payload_includes_architecture_description(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["true_monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        variant = payload["variants"]["true_monolithic_policy"]
        self.assertEqual(
            variant["config"]["architecture_description"],
            "true monolithic direct control",
        )
        self.assertEqual(
            variant["summary"]["architecture_description"],
            "true monolithic direct control",
        )

    def test_true_monolithic_deltas_are_computed_vs_modular_full_reference(self) -> None:
        payload, _ = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["true_monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertEqual(payload["reference_variant"], "modular_full")
        self.assertIn("true_monolithic_policy", payload["deltas_vs_reference"])
        delta = payload["deltas_vs_reference"]["true_monolithic_policy"]["summary"]
        variant_summary = payload["variants"]["true_monolithic_policy"]["summary"]
        reference_summary = payload["variants"]["modular_full"]["summary"]
        expected = (
            float(variant_summary["scenario_success_rate"])
            - float(reference_summary["scenario_success_rate"])
        )
        self.assertAlmostEqual(
            float(delta["scenario_success_rate_delta"]),
            expected,
            places=6,
        )

    def test_true_monolithic_completes_full_comparison_workflow_without_errors(self) -> None:
        payload, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["true_monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertIn("true_monolithic_policy", payload["variants"])
        self.assertTrue(rows)

    def test_rows_contain_operational_profile_columns(self) -> None:
        _, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("operational_profile", row)
            self.assertIn("operational_profile_version", row)
            self.assertIn("noise_profile", row)
            self.assertIn("noise_profile_config", row)

    def test_rows_contain_architecture_traceability_columns(self) -> None:
        _, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("architecture_version", row)
            self.assertIn("architecture_fingerprint", row)

    def test_payload_includes_without_reflex_support_and_rows_track_eval_scale(self) -> None:
        """
        Verify that compare_ablation_suite marks the primary evaluation as "without_reflex_support" and that reflex-scale variant rows include both evaluation runs with eval_reflex_scale == 0.0 and eval_reflex_scale > 0.0.

        This test calls compare_ablation_suite for the single variant "reflex_scale_0_50" and asserts:
        - The payload's variant entries for "modular_full" and "reflex_scale_0_50" include "without_reflex_support".
        - The overall primary evaluation is "without_reflex_support" and the reference evaluation reflex scale is 0.0.
        - The modular_full variant's primary evaluation and its summary.eval_reflex_scale are set to "without_reflex_support" and 0.0 respectively.
        - The reflex_scale_0_50 config reflex_scale is approximately 0.5.
        - The returned rows contain entries for the "reflex_scale_0_50" variant and include at least one row with eval_reflex_scale == 0.0 and at least one row with eval_reflex_scale > 0.0.
        """
        payload, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["reflex_scale_0_50"],
            names=("night_rest",),
            seeds=(7,),
        )

        self.assertIn("without_reflex_support", payload["variants"]["modular_full"])
        self.assertIn("without_reflex_support", payload["variants"]["reflex_scale_0_50"])
        self.assertEqual(payload["primary_evaluation"], "without_reflex_support")
        self.assertEqual(payload["reference_eval_reflex_scale"], 0.0)
        self.assertEqual(
            payload["variants"]["modular_full"]["primary_evaluation"],
            "without_reflex_support",
        )
        self.assertEqual(
            payload["variants"]["modular_full"]["summary"]["eval_reflex_scale"],
            0.0,
        )
        self.assertEqual(
            payload["variants"]["modular_full"]["summary"]["competence_type"],
            "self_sufficient",
        )
        self.assertEqual(
            payload["variants"]["modular_full"]["without_reflex_support"]["summary"][
                "competence_type"
            ],
            "self_sufficient",
        )
        self.assertEqual(
            payload["variants"]["modular_full"]["with_reflex_support"]["summary"][
                "competence_type"
            ],
            "scaffolded",
        )
        self.assertAlmostEqual(
            float(payload["variants"]["reflex_scale_0_50"]["config"]["reflex_scale"]),
            0.5,
        )
        reflex_scale_rows = [
            row for row in rows if row["ablation_variant"] == "reflex_scale_0_50"
        ]
        self.assertTrue(reflex_scale_rows)
        self.assertTrue(
            any(float(row["eval_reflex_scale"]) == 0.0 for row in reflex_scale_rows)
        )
        self.assertTrue(
            any(float(row["eval_reflex_scale"]) > 0.0 for row in reflex_scale_rows)
        )
        self.assertTrue(
            any(row["competence_type"] == "self_sufficient" for row in reflex_scale_rows)
        )
        self.assertTrue(
            any(row["competence_type"] == "scaffolded" for row in reflex_scale_rows)
        )
        self.assertTrue(
            any(row["is_primary_benchmark"] is True for row in reflex_scale_rows)
        )

    def test_without_reflex_support_reuses_behavior_base_index(self) -> None:
        """
        Verifies that behavior evaluations run without reflex support reuse the same base_index across related suite executions.

        Patches SpiderSimulation._execute_behavior_suite to record (brain config name, current_reflex_scale, base_index) for each executed suite, runs compare_ablation_suite for a reflex-scale variant, and asserts the sequence of recorded calls and their base_index values match the expected order: first the reference variant with reflex support and without, then the reflex-scale variant with reflex support and without — all using the same base_index (300000).
        """
        original_execute = SpiderSimulation._execute_behavior_suite
        recorded_calls: list[tuple[str, float, int]] = []

        def wrapped_execute(
            self,
            *,
            names,
            episodes_per_scenario,
            capture_trace,
            debug_trace,
            base_index=100_000,
        ):
            recorded_calls.append(
                (
                    self.brain.config.name,
                    float(self.brain.current_reflex_scale),
                    int(base_index),
                )
            )
            return original_execute(
                self,
                names=names,
                episodes_per_scenario=episodes_per_scenario,
                capture_trace=capture_trace,
                debug_trace=debug_trace,
                base_index=base_index,
            )

        SpiderSimulation._execute_behavior_suite = wrapped_execute  # type: ignore[method-assign]
        try:
            compare_ablation_suite(
                episodes=0,
                evaluation_episodes=0,
                variant_names=["reflex_scale_0_50"],
                names=("night_rest",),
                seeds=(7,),
            )
        finally:
            SpiderSimulation._execute_behavior_suite = original_execute  # type: ignore[method-assign]

        self.assertEqual(
            recorded_calls,
            [
                ("modular_full", 1.0, 300_000),
                ("modular_full", 0.0, 300_000),
                ("reflex_scale_0_50", 0.5, 300_000),
                ("reflex_scale_0_50", 0.0, 300_000),
            ],
        )

    def test_rows_operational_profile_value_is_default(self) -> None:
        _, rows = compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        for row in rows:
            self.assertEqual(row["operational_profile"], "default_v1")
            self.assertEqual(row["operational_profile_version"], 1)
            self.assertEqual(row["noise_profile"], "none")
            self.assertIn("noise_profile_config", row)

class ProfileComparisonMetricsTest(unittest.TestCase):
    """Tests for comparison.profile_comparison_metrics."""

    def test_returns_zero_defaults_for_none_input(self) -> None:
        result = profile_comparison_metrics(None)
        self.assertEqual(result["scenario_success_rate"], 0.0)
        self.assertEqual(result["episode_success_rate"], 0.0)
        self.assertEqual(result["mean_reward"], 0.0)

    def test_returns_zero_defaults_for_non_dict_input(self) -> None:
        result = profile_comparison_metrics("bad")
        self.assertEqual(result["scenario_success_rate"], 0.0)
        self.assertEqual(result["episode_success_rate"], 0.0)

    def test_extracts_from_summary_subdict(self) -> None:
        payload = {
            "summary": {
                "scenario_success_rate": 0.75,
                "episode_success_rate": 0.60,
            }
        }
        result = profile_comparison_metrics(payload)
        self.assertAlmostEqual(result["scenario_success_rate"], 0.75)
        self.assertAlmostEqual(result["episode_success_rate"], 0.60)

    def test_extracts_from_flat_payload_when_no_summary(self) -> None:
        payload = {
            "scenario_success_rate": 0.55,
            "episode_success_rate": 0.40,
            "mean_reward": 1.23,
        }
        result = profile_comparison_metrics(payload)
        self.assertAlmostEqual(result["scenario_success_rate"], 0.55)
        self.assertAlmostEqual(result["episode_success_rate"], 0.40)
        self.assertAlmostEqual(result["mean_reward"], 1.23)

    def test_returns_three_metric_keys(self) -> None:
        result = profile_comparison_metrics({})
        self.assertEqual(
            set(result.keys()),
            {"scenario_success_rate", "episode_success_rate", "mean_reward"},
        )

    def test_missing_rate_fields_default_to_zero(self) -> None:
        payload = {"summary": {}}
        result = profile_comparison_metrics(payload)
        self.assertEqual(result["scenario_success_rate"], 0.0)
        self.assertEqual(result["episode_success_rate"], 0.0)

    def test_values_are_float(self) -> None:
        payload = {
            "summary": {
                "scenario_success_rate": 1,
                "episode_success_rate": 0,
            }
        }
        result = profile_comparison_metrics(payload)
        self.assertIsInstance(result["scenario_success_rate"], float)
        self.assertIsInstance(result["episode_success_rate"], float)
