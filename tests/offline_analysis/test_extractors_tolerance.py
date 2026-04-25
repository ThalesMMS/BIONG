from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.offline_analysis.constants import (
    DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    SHAPING_DEPENDENCE_WARNING_THRESHOLD,
)
from spider_cortex_sim.offline_analysis.extractors import (
    _aggregate_specialization_from_scenarios,
    _noise_robustness_cell_summary,
    _noise_robustness_metrics,
    _normalize_module_response_by_predator_type,
    _normalize_noise_marginals,
    _ordered_noise_conditions,
    extract_ablations,
    extract_architecture_capacity,
    extract_model_capacity,
    extract_noise_robustness,
    extract_predator_type_specialization,
    extract_reflex_frequency,
    extract_ladder_comparison,
    extract_representation_specialization,
    extract_shaping_audit,
    extract_training_eval_series,
    extract_unified_ladder_report,
)
from spider_cortex_sim.offline_analysis.ingestion import load_summary, normalize_behavior_rows
from spider_cortex_sim.offline_analysis.report import build_report_data, write_report
from spider_cortex_sim.offline_analysis.tables import build_diagnostics
from spider_cortex_sim.simulation import SpiderSimulation

from .conftest import (
    CHECKIN_SUMMARY,
    EPISODE_SHAPING_GAP,
    LARGE_SHAPING_GAP,
    MEAN_REWARD_SHAPING_GAP,
    SHAPING_GAP_EPSILON,
    SMALL_SHAPING_GAP,
    build_uncertainty_summary,
)

from .extractor_fixtures import OfflineAnalysisToleranceFixtures


class OfflineAnalysisToleranceTest(OfflineAnalysisToleranceFixtures, unittest.TestCase):
    def test_build_report_tolerates_summary_without_behavior_evaluation(self) -> None:
        summary = load_summary(CHECKIN_SUMMARY)
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertTrue(report["training_eval"]["available"])
        self.assertFalse(report["scenario_success"]["available"])
        self.assertIn("No scenario-level success data was available.", report["limitations"])

    def test_build_report_tolerates_missing_comparisons_and_ablations(self) -> None:
        sim = SpiderSimulation(seed=3, max_steps=5)
        summary, _ = sim.train(
            episodes=2,
            evaluation_episodes=1,
            capture_evaluation_trace=False,
        )
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertFalse(report["comparisons"]["available"])
        self.assertFalse(report["ablations"]["available"])
        self.assertTrue(report["training_eval"]["available"])

    def test_training_eval_source_uses_evaluation_detail_when_no_training(self) -> None:
        result = extract_training_eval_series(
            {
                "evaluation": {
                    "episodes_detail": [
                        {"episode": 1, "reward": 2.5},
                    ],
                }
            }
        )

        self.assertEqual(result["source"], "summary.evaluation.episodes_detail")

    def test_training_eval_source_uses_evaluation_aggregate_when_no_training(self) -> None:
        result = extract_training_eval_series(
            {
                "evaluation": {
                    "mean_reward": 2.5,
                }
            }
        )

        self.assertEqual(result["source"], "summary.evaluation.aggregate")

    def test_extract_shaping_audit_surfaces_gap_and_survival_flags(self) -> None:
        result = extract_shaping_audit(self._summary_with_shaping_program())

        self.assertTrue(result["available"])
        self.assertEqual(result["dense_profile"], "classic")
        self.assertEqual(result["minimal_profile"], "austere")
        self.assertTrue(result["interpretive_flags"]["shaping_dependent"])
        self.assertAlmostEqual(
            result["gap_metrics"]["scenario_success_rate_delta"],
            LARGE_SHAPING_GAP,
        )
        self.assertAlmostEqual(result["removed_weight_gap"], 1.2)
        self.assertIn("food_progress", result["disposition_summary"]["removed"]["components"])
        self.assertEqual(
            result["behavior_survival"]["surviving_scenario_count"],
            1,
        )

    def test_build_report_data_includes_shaping_program(self) -> None:
        report = build_report_data(
            summary=self._summary_with_shaping_program(),
            trace=[],
            behavior_rows=[],
        )

        self.assertIn("shaping_program", report)
        self.assertTrue(report["shaping_program"]["available"])
        self.assertTrue(
            report["shaping_program"]["interpretive_flags"]["shaping_dependent"]
        )

    def test_write_report_renders_shaping_program_section(self) -> None:
        """
        Verifies that writing a report produces a complete "Shaping Minimization Program" section when a shaping program is present in the summary.
        
        Asserts that the rendered report Markdown contains:
        - The main section header "## Shaping Minimization Program" and the subsection "### Dense vs Minimal Gap".
        - A warning line indicating high shaping dependence when applicable.
        - A table row for the `food_progress` component with disposition and removal status.
        - A summary line reporting surviving scenarios under minimal shaping ("1/2 scenarios survive minimal shaping").
        - A per-scenario row for "night_rest" showing its survival, success rate, and episode count.
        """
        report = build_report_data(
            summary=self._summary_with_shaping_program(),
            trace=[],
            behavior_rows=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Shaping Minimization Program", report_md)
        self.assertIn("### Dense vs Minimal Gap", report_md)
        self.assertIn("WARNING: High shaping dependence detected", report_md)
        self.assertIn("| food_progress | progress | high | removed |", report_md)
        self.assertIn("1/2 scenarios survive minimal shaping", report_md)
        self.assertIn("| night_rest | 1.00 | yes | 2 |", report_md)

    def test_build_report_data_includes_credit_analysis(self) -> None:
        report = build_report_data(
            summary=self._summary_with_credit_analysis_program(),
            trace=[],
            behavior_rows=[],
        )

        self.assertIn("credit_analysis", report)
        self.assertTrue(report["credit_analysis"]["available"])
        strategy_rows = report["credit_analysis"]["strategy_comparison_table"]["rows"]
        self.assertEqual(len(strategy_rows), 6)
        matrix_rows = report["credit_analysis"]["architecture_strategy_matrix"]["rows"]
        self.assertEqual(len(matrix_rows), 6)
        patterns = {
            item["pattern"]
            for item in report["credit_analysis"]["findings"]
            if isinstance(item, dict)
        }
        self.assertIn("excessive global credit", patterns)
        self.assertIn("insufficient local credit", patterns)
        self.assertIn("counterfactual improvement", patterns)
        self.assertIn("local_only differential failure", patterns)
        self.assertIn("counterfactual benefit scales with module count", patterns)

    def test_write_report_renders_credit_assignment_analysis_section(self) -> None:
        report = build_report_data(
            summary=self._summary_with_credit_analysis_program(),
            trace=[],
            behavior_rows=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Credit Assignment Analysis", report_md)
        self.assertIn("| architecture_rung | strategy | variant |", report_md)
        self.assertIn("insufficient local credit", report_md)
        self.assertIn("counterfactual benefit scales with module count", report_md)

    def test_extract_unified_ladder_report_marks_capacity_confounded(self) -> None:
        result = extract_unified_ladder_report(build_uncertainty_summary())

        self.assertTrue(result["available"])
        self.assertEqual(result["conclusion"], "capacity/interface confounded")
        self.assertEqual(len(result["ladder_table"]["rows"]), 5)
        self.assertTrue(
            any(
                item["experiment_name"] == "capacity_matched_ladder"
                for item in result["missing_experiments"]
            )
        )

    def test_extract_unified_ladder_report_supports_capacity_matched_positive_case(
        self,
    ) -> None:
        summary = build_uncertainty_summary()
        variants = summary["behavior_evaluation"]["ablations"]["variants"]
        for variant_name, total in (
            ("true_monolithic_policy", 1000),
            ("monolithic_policy", 1020),
            ("three_center_modular", 980),
            ("four_center_modular", 1010),
            ("modular_full", 990),
        ):
            variants[variant_name]["parameter_counts"]["total"] = total
            variants[variant_name]["parameter_counts"]["total_trainable"] = total
        summary["behavior_evaluation"]["claim_tests"]["claims"][
            "escape_without_reflex_support"
        ] = {
            "status": "passed",
            "passed": True,
            "reference_value": 0.45,
            "comparison_values": {"trained_without_reflex_support": 0.75},
            "delta": {"trained_without_reflex_support": 0.30},
            "cohens_d": {"trained_without_reflex_support": 1.2},
            "primary_metric": "predator_response_scenario_success_rate",
            "scenarios_evaluated": [
                "predator_edge",
                "entrance_ambush",
                "shelter_blockade",
            ],
        }

        result = extract_unified_ladder_report(summary)

        self.assertEqual(result["conclusion"], "modularity supported")
        self.assertIn("A4 exceeded A0", result["conclusion_rationale"])

    def test_extract_ablations_falls_back_to_behavior_csv_rows(self) -> None:
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "night_rest",
                    "success": True,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 1.0,
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "ablation_variant": "monolithic_policy",
                    "ablation_architecture": "monolithic",
                    "eval_reflex_scale": 1.0,
                },
            ]
        )
        ablations = extract_ablations({}, rows)

        self.assertTrue(ablations["available"])
        self.assertIn("modular_full", ablations["variants"])
        self.assertIn("monolithic_policy", ablations["variants"])
        self.assertEqual(ablations["variants"]["modular_full"]["ladder_rung"], "A4")

    def test_extract_ablations_fallback_uses_minimal_reflex_rows(self) -> None:
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "night_rest",
                    "success": False,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 1.0,
                },
                {
                    "scenario": "night_rest",
                    "success": True,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "night_rest",
                    "success": True,
                    "ablation_variant": "monolithic_policy",
                    "ablation_architecture": "monolithic",
                    "eval_reflex_scale": 1.0,
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "ablation_variant": "monolithic_policy",
                    "ablation_architecture": "monolithic",
                    "eval_reflex_scale": 0.0,
                },
            ]
        )

        ablations = extract_ablations({}, rows)

        self.assertEqual(ablations["reference_variant"], "modular_full")
        self.assertEqual(
            ablations["variants"]["modular_full"]["summary"]["eval_reflex_scale"],
            0.0,
        )
        self.assertAlmostEqual(
            ablations["variants"]["modular_full"]["summary"]["scenario_success_rate"],
            1.0,
        )
        self.assertAlmostEqual(
            ablations["variants"]["monolithic_policy"]["summary"]["scenario_success_rate"],
            0.0,
        )
        self.assertAlmostEqual(
            ablations["deltas_vs_reference"]["monolithic_policy"]["summary"][
                "scenario_success_rate_delta"
            ],
            -1.0,
        )
        self.assertEqual(
            ablations["variants"]["monolithic_policy"]["ladder_rung"],
            "A1",
        )

    def test_extract_ablations_builds_architectural_ladder(self) -> None:
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "true_monolithic_policy": {
                            "config": {"architecture": "true_monolithic"},
                            "summary": {"scenario_success_rate": 0.2},
                            "suite": {"night_rest": {"success_rate": 0.2}},
                        },
                        "monolithic_policy": {
                            "config": {"architecture": "monolithic"},
                            "summary": {"scenario_success_rate": 0.5},
                            "suite": {"night_rest": {"success_rate": 0.5}},
                        },
                        "three_center_modular": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 0.6},
                            "suite": {"night_rest": {"success_rate": 0.6}},
                        },
                        "four_center_modular": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 0.7},
                            "suite": {"night_rest": {"success_rate": 0.7}},
                        },
                        "modular_full": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 0.8},
                            "suite": {"night_rest": {"success_rate": 0.8}},
                        },
                        "no_module_dropout": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 0.75},
                            "suite": {"night_rest": {"success_rate": 0.75}},
                        },
                    },
                }
            }
        }

        ablations = extract_ablations(summary, [])

        self.assertIn("ladder_rungs", ablations)
        self.assertIn("ladder_comparison", ablations)
        self.assertEqual(
            ablations["variants"]["true_monolithic_policy"]["ladder_rung"],
            "A0",
        )
        self.assertEqual(
            ablations["variants"]["monolithic_policy"]["ladder_rung"],
            "A1",
        )
        self.assertEqual(
            ablations["variants"]["three_center_modular"]["ladder_rung"],
            "A2",
        )
        self.assertEqual(
            ablations["variants"]["modular_full"]["ladder_rung"],
            "A4",
        )
        self.assertEqual(
            ablations["variants"]["four_center_modular"]["ladder_rung"],
            "A3",
        )
        self.assertIsNone(
            ablations["variants"]["no_module_dropout"]["ladder_rung"],
        )
        self.assertEqual(
            ablations["ladder_rungs"]["recognized_rungs"],
            ["A0", "A1", "A2", "A3", "A4"],
        )
        self.assertEqual(
            ablations["ladder_rungs"]["descriptions"]["A1"],
            "Monolithic representation with the existing action_center plus motor_cortex pipeline, isolating whether separating decision from execution helps even without modular sensory processing.",
        )
        self.assertEqual(
            ablations["ladder_rungs"]["applicable_comparisons"],
            [
                {"baseline_rung": "A0", "comparison_rung": "A1"},
                {"baseline_rung": "A1", "comparison_rung": "A2"},
                {"baseline_rung": "A2", "comparison_rung": "A3"},
                {"baseline_rung": "A3", "comparison_rung": "A4"},
            ],
        )
        ladder = ablations["ladder"]
        self.assertTrue(ladder["available"])
        self.assertEqual(
            ladder["rungs"]["A0"]["technical_variant"],
            "true_monolithic_policy",
        )
        self.assertEqual(
            ladder["rungs"]["A1"]["technical_variant"],
            "monolithic_policy",
        )
        self.assertEqual(
            ladder["rungs"]["A2"]["technical_variant"],
            "three_center_modular",
        )
        self.assertEqual(
            ladder["rungs"]["A3"]["technical_variant"],
            "four_center_modular",
        )
        self.assertEqual(
            ladder["rungs"]["A4"]["technical_variant"],
            "modular_full",
        )
        a0_vs_a1 = next(
            item
            for item in ladder["adjacent_comparisons"]
            if item["baseline_rung"] == "A0" and item["comparison_rung"] == "A1"
        )
        self.assertAlmostEqual(
            a0_vs_a1["summary"]["scenario_success_rate_delta"],
            0.3,
        )
        a2_vs_a3 = next(
            item
            for item in ladder["adjacent_comparisons"]
            if item["baseline_rung"] == "A2" and item["comparison_rung"] == "A3"
        )
        self.assertAlmostEqual(
            a2_vs_a3["summary"]["scenario_success_rate_delta"],
            0.1,
        )
