from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.export import save_behavior_csv
from spider_cortex_sim.offline_analysis.cli import run_offline_analysis
from spider_cortex_sim.offline_analysis.extractors import extract_shaping_audit
from spider_cortex_sim.offline_analysis.report import build_report_data, write_report
from spider_cortex_sim.simulation import SpiderSimulation

from .conftest import (
    CHECKIN_SUMMARY,
    CHECKIN_TRACE,
    EPISODE_SHAPING_GAP,
    LARGE_SHAPING_GAP,
    MEAN_REWARD_SHAPING_GAP,
    SMALL_SHAPING_GAP,
    build_uncertainty_summary,
)

class ShapingReportTest(unittest.TestCase):
    def _summary(
        self,
        scenario_success_rate_delta: float = LARGE_SHAPING_GAP,
    ) -> dict[str, object]:
        """
        Create a synthetic `summary` dictionary representing a reward audit used in tests.
        
        The `scenario_success_rate_delta` value is applied to the `"classic"` profile in
        `reward_audit.comparison.deltas_vs_minimal.classic.scenario_success_rate_delta`.
        
        Parameters:
            scenario_success_rate_delta (float): Delta for the classic profile's scenario-level success rate.
        
        Returns:
            dict: A summary-like dictionary containing `reward_audit` with:
                - `minimal_profile` (str)
                - `reward_components` (mapping of component metadata)
                - `reward_profiles` (profiles with disposition summaries)
                - `comparison` (including `deltas_vs_minimal` and `behavior_survival`)
        """
        return {
            "reward_audit": {
                "minimal_profile": "austere",
                "reward_components": {
                    "food_progress": {
                        "category": "progress",
                        "shaping_risk": "high",
                        "shaping_disposition": "removed",
                        "disposition_rationale": "Zeroed in the austere profile.",
                    }
                },
                "reward_profiles": {
                    "classic": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 1.2}
                        }
                    },
                    "austere": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 0.0}
                        }
                    },
                },
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": scenario_success_rate_delta,
                            "episode_success_rate_delta": EPISODE_SHAPING_GAP,
                            "mean_reward_delta": MEAN_REWARD_SHAPING_GAP,
                        },
                        "austere": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        },
                    },
                    "behavior_survival": {
                        "minimal_profile": "austere",
                        "survival_threshold": 0.5,
                        "scenario_count": 1,
                        "surviving_scenario_count": 1,
                        "survival_rate": 1.0,
                        "scenarios": {
                            "night_rest": {
                                "austere_success_rate": 1.0,
                                "survives": True,
                                "episodes": 1,
                            }
                        },
                    },
                },
            }
        }

    def test_extract_shaping_audit_extracts_delta_metrics(self) -> None:
        result = extract_shaping_audit(self._summary())
        self.assertAlmostEqual(
            result["gap_metrics"]["scenario_success_rate_delta"],
            LARGE_SHAPING_GAP,
        )
        self.assertAlmostEqual(
            result["gap_metrics"]["episode_success_rate_delta"],
            EPISODE_SHAPING_GAP,
        )
        self.assertAlmostEqual(
            result["gap_metrics"]["mean_reward_delta"],
            MEAN_REWARD_SHAPING_GAP,
        )

    def test_build_report_data_includes_shaping_program_when_comparison_available(self) -> None:
        report = build_report_data(
            summary=self._summary(),
            trace=[],
            behavior_rows=[],
        )
        self.assertIn("shaping_program", report)
        self.assertTrue(report["shaping_program"]["available"])

    def test_write_report_contains_shaping_minimization_section(self) -> None:
        report = build_report_data(
            summary=self._summary(),
            trace=[],
            behavior_rows=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Shaping Minimization Program", report_md)
        self.assertIn("### Dense vs Minimal Gap", report_md)
        self.assertIn("### Profile-Level Summary", report_md)
        self.assertIn("### Component Dispositions", report_md)
        self.assertIn("### Behavior Survival", report_md)

    def test_write_report_renders_profile_level_shaping_summary(self) -> None:
        """
        Verify the generated Markdown report includes the profile-level shaping summary.
        
        Asserts the report contains the removed disposition weight gap for classic vs austere (1.20),
        the profile table header and rows for `classic` and `austere`, and the component disposition
        table header and a row for the `food_progress` component.
        """
        report = build_report_data(
            summary=self._summary(),
            trace=[],
            behavior_rows=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("Removed disposition weight gap (classic - austere): 1.20.", report_md)
        self.assertIn("| profile | disposition | total_weight_proxy |", report_md)
        self.assertIn("| classic | removed | 1.20 |", report_md)
        self.assertIn("| austere | removed | 0.00 |", report_md)
        self.assertIn("| disposition | component_count | components |", report_md)
        self.assertIn("| removed | 1 | food_progress |", report_md)

    def test_write_report_renders_warning_for_large_shaping_gap(self) -> None:
        report = build_report_data(
            summary=self._summary(scenario_success_rate_delta=LARGE_SHAPING_GAP),
            trace=[],
            behavior_rows=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("WARNING: High shaping dependence detected", report_md)

    def test_write_report_omits_warning_for_small_shaping_gap(self) -> None:
        report = build_report_data(
            summary=self._summary(scenario_success_rate_delta=SMALL_SHAPING_GAP),
            trace=[],
            behavior_rows=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("WARNING: High shaping dependence detected", report_md)

    def test_write_report_omits_placeholder_gap_without_reward_audit(self) -> None:
        report = build_report_data(summary={}, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("| classic - austere | 0.00 | 0.00 | 0.00 |", report_md)
        self.assertNotIn("Removed disposition weight gap", report_md)

class OfflineAnalysisUncertaintyReportTest(unittest.TestCase):
    def test_build_report_data_includes_uncertainty_tables(self) -> None:
        report = build_report_data(summary=build_uncertainty_summary(), trace=[], behavior_rows=[])

        self.assertIn("aggregate_benchmark_tables", report)
        self.assertIn("claim_test_tables", report)
        self.assertIn("effect_size_tables", report)

    def test_write_report_renders_uncertainty_sections(self) -> None:
        report = build_report_data(summary=build_uncertainty_summary(), trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Benchmark-of-Record Summary", report_md)
        self.assertIn("## Claim Test Results with Uncertainty", report_md)
        self.assertIn("## Effect Sizes Against Baselines", report_md)
        self.assertIn("[^ci-method]:", report_md)
        self.assertIn("[^effect-size]:", report_md)

class OfflineAnalysisOutputTest(unittest.TestCase):
    def test_run_offline_analysis_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "report"
            report = run_offline_analysis(
                summary_path=CHECKIN_SUMMARY,
                trace_path=CHECKIN_TRACE,
                output_dir=output_dir,
            )

            generated = report["generated_files"]
            self.assertTrue((output_dir / "report.md").exists())
            self.assertTrue((output_dir / "report.json").exists())
            self.assertTrue((output_dir / "training_eval.svg").exists())
            self.assertTrue((output_dir / "scenario_success.svg").exists())
            self.assertTrue((output_dir / "robustness_matrix.svg").exists())
            self.assertTrue((output_dir / "representation_specialization.svg").exists())
            self.assertTrue((output_dir / "scenario_checks.csv").exists())
            self.assertTrue((output_dir / "reward_components.csv").exists())
            self.assertTrue(generated["robustness_matrix_svg"])
            self.assertTrue(generated["reflex_frequency_svg"])
            self.assertTrue(generated["representation_specialization_svg"])

    def test_run_offline_analysis_report_mentions_real_variant_and_scenario(self) -> None:
        """
        Verifies that run_offline_analysis includes scenario and ablation variant identifiers from a behavior CSV in the generated report outputs.
        
        Creates a behavior CSV containing a real scenario row and a custom ablation variant, runs offline analysis, and asserts that the resulting report Markdown and JSON contain the scenario name and ablation variant.
        """
        sim = SpiderSimulation(seed=5, max_steps=8)
        _, _, rows = sim.evaluate_behavior_suite(["night_rest"])
        rows.append(
            {
                "reward_profile": "classic",
                "scenario_map": "central_burrow",
                "evaluation_map": "central_burrow",
                "ablation_variant": "monolithic_policy",
                "ablation_architecture": "monolithic",
                "reflex_scale": 0.0,
                "reflex_anneal_final_scale": 0.0,
                "eval_reflex_scale": 0.0,
                "budget_profile": "smoke",
                "benchmark_strength": "quick",
                "operational_profile": "default_v1",
                "operational_profile_version": 1,
                "checkpoint_source": "final",
                "simulation_seed": 7,
                "episode_seed": 7001,
                "scenario": "night_rest",
                "scenario_description": "night rest",
                "scenario_objective": "sleep",
                "scenario_focus": "sleep",
                "episode": 0,
                "success": False,
                "failure_count": 1,
                "failures": "check_missing",
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "behavior.csv"
            save_behavior_csv(rows, csv_path)
            output_dir = Path(tmpdir) / "report"
            run_offline_analysis(
                behavior_csv_path=csv_path,
                output_dir=output_dir,
            )
            report_md = (output_dir / "report.md").read_text(encoding="utf-8")
            report_json = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))

        self.assertIn("night_rest", report_md)
        self.assertIn("monolithic_policy", report_md)
        self.assertIn("night_rest", json.dumps(report_json, ensure_ascii=False))
        self.assertIn("monolithic_policy", json.dumps(report_json, ensure_ascii=False))
