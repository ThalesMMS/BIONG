from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from spider_cortex_sim.comparison_capacity import compare_capacity_sweep
from spider_cortex_sim.export import save_behavior_csv
from spider_cortex_sim.offline_analysis.combined import build_combined_ladder_report
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
        self.assertIn("ladder_comparison", report)
        self.assertIn("parameter_counts", report)
        self.assertIn("capacity_analysis", report)
        self.assertEqual(
            report["capacity_analysis"]["status"],
            "monolithic_policy 2.7x larger",
        )

    def test_write_report_renders_uncertainty_sections(self) -> None:
        report = build_report_data(summary=build_uncertainty_summary(), trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Architecture Capacity", report_md)
        self.assertIn("| network | parameters | proportion |", report_md)
        self.assertIn(
            "| variant | architecture | total_trainable | key_components | capacity_status | ratio_vs_reference |",
            report_md,
        )
        self.assertIn("Capacity status: monolithic_policy 2.7x larger", report_md)
        self.assertIn("not matched", report_md)
        self.assertIn("## Benchmark-of-Record Summary", report_md)
        self.assertIn("## Claim Test Results with Uncertainty", report_md)
        self.assertIn("## Effect Sizes Against Baselines", report_md)
        self.assertIn("## Architectural Ladder Comparison", report_md)
        self.assertIn("A0_true_monolithic", report_md)
        self.assertIn("A1_monolithic_with_action_motor", report_md)
        self.assertIn("A2_three_center", report_md)
        self.assertIn("A3_four_center", report_md)
        self.assertIn("A4_current_full_modular", report_md)
        self.assertIn(
            "The ladder validation asks whether gains or regressions come from adding the decision-execution pipeline itself or from later sensory/proposer modularization.",
            report_md,
        )
        self.assertIn(
            "| A0 | A0_true_monolithic | Direct monolithic control: one network maps shared observations straight to action logits with no action_center or motor_cortex stage. |",
            report_md,
        )
        self.assertIn(
            "| A1 | A1_monolithic_with_action_motor | Monolithic representation with the existing action_center plus motor_cortex pipeline, isolating whether separating decision from execution helps even without modular sensory processing. |",
            report_md,
        )
        self.assertIn(
            "| A0 | A1 | 0.15 | 1.34 | positive large effect |",
            report_md,
        )
        self.assertIn(
            "| A1 | A2 | 0.20 | 1.79 | positive large effect |",
            report_md,
        )
        self.assertIn(
            "| A2 | A3 | 0.10 | 1.41 | positive large effect |",
            report_md,
        )
        self.assertIn(
            "| A3 | A4 | 0.10 | 0.89 | positive large effect |",
            report_md,
        )
        self.assertIn("## Unified Architectural Ladder Report", report_md)
        self.assertIn("Conclusion: `capacity/interface confounded`.", report_md)
        self.assertIn(
            "| A4 | A4_current_full_modular | modular_full | 840 | 0.80 | 0.70 | 0.90 | 4.92 | large | 2 |",
            report_md,
        )
        self.assertIn("Adjacent rung comparisons:", report_md)
        self.assertIn("Capacity matching summary:", report_md)
        self.assertIn("Credit assignment comparison summary:", report_md)
        self.assertIn("Reward shaping sensitivity:", report_md)
        self.assertIn("No-reflex competence:", report_md)
        self.assertIn("Capability probe boundaries:", report_md)
        self.assertIn("> Conclusion", report_md)
        self.assertIn("Missing experiments before asserting modular emergence:", report_md)
        self.assertIn("Unified ladder limitations:", report_md)
        self.assertIn("[^ci-method]:", report_md)
        self.assertIn("[^effect-size]:", report_md)

    def test_write_report_reports_partial_ladder_coverage(self) -> None:
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "monolithic_policy": {
                            "config": {"architecture": "monolithic"},
                            "summary": {
                                "scenario_success_rate": 0.5,
                                "episode_success_rate": 0.4,
                                "mean_reward": 4.0,
                            },
                            "seed_level": [
                                {
                                    "metric_name": "scenario_success_rate",
                                    "seed": 1,
                                    "value": 0.4,
                                    "condition": "monolithic_policy",
                                },
                                {
                                    "metric_name": "scenario_success_rate",
                                    "seed": 2,
                                    "value": 0.6,
                                    "condition": "monolithic_policy",
                                },
                            ],
                        },
                        "modular_full": {
                            "config": {"architecture": "modular"},
                            "summary": {
                                "scenario_success_rate": 0.8,
                                "episode_success_rate": 0.7,
                                "mean_reward": 7.0,
                            },
                            "seed_level": [
                                {
                                    "metric_name": "scenario_success_rate",
                                    "seed": 1,
                                    "value": 0.7,
                                    "condition": "modular_full",
                                },
                                {
                                    "metric_name": "scenario_success_rate",
                                    "seed": 2,
                                    "value": 0.9,
                                    "condition": "modular_full",
                                },
                            ],
                        },
                    },
                }
            }
        }
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("Missing variants:", report_md)
        self.assertIn("`true_monolithic_policy`", report_md)
        self.assertIn("`three_center_modular`", report_md)
        self.assertIn("`four_center_modular`", report_md)
        self.assertIn("Missing adjacent ladder pairs:", report_md)
        self.assertIn("`A0->A1`", report_md)
        self.assertIn("`A1->A2`", report_md)
        self.assertIn("`A2->A3`", report_md)
        self.assertIn("`A3->A4`", report_md)

    def test_build_report_data_skips_broadcast_limitation_for_empty_rungs(self) -> None:
        report = build_report_data(
            summary={
                "behavior_evaluation": {
                    "ablations": {
                        "variants": {
                            "three_center_modular_local_credit": {
                                "config": {
                                    "architecture": "modular",
                                    "credit_strategy": "local_only",
                                },
                                "summary": {
                                    "scenario_success_rate": 0.4,
                                    "mean_module_credit_weights": {
                                        "perception_center": 0.0,
                                    },
                                    "module_gradient_norm_means": {
                                        "perception_center": 0.0,
                                    },
                                },
                            },
                        },
                    },
                },
            },
            trace=[],
            behavior_rows=[],
        )

        self.assertIn(
            "Credit diagnostics for A2 lacked a broadcast reference.",
            report["credit_analysis"]["limitations"],
        )
        self.assertNotIn(
            "Credit diagnostics for A3 lacked a broadcast reference.",
            report["credit_analysis"]["limitations"],
        )
        self.assertNotIn(
            "Credit diagnostics for A4 lacked a broadcast reference.",
            report["credit_analysis"]["limitations"],
        )

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


class OfflineAnalysisArtifactStateTest(unittest.TestCase):
    def test_capacity_sweeps_marked_not_in_artifact_when_key_absent(self) -> None:
        report = build_report_data(summary={}, trace=[], behavior_rows=[])
        self.assertEqual(report["capacity_sweeps"]["artifact_state"], "not_in_this_artifact")

    def test_capacity_sweeps_marked_expected_but_missing_when_key_present_without_rows(self) -> None:
        report = build_report_data(
            summary={"behavior_evaluation": {"capacity_sweeps": {}}},
            trace=[],
            behavior_rows=[],
        )
        self.assertEqual(report["capacity_sweeps"]["artifact_state"], "expected_but_missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("expected in this artifact but missing", report_md)

    def test_capacity_sweep_report_uses_fallback_note_when_only_interpretations_exist(self) -> None:
        capacity_payload, _ = compare_capacity_sweep(
            episodes=0,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
        )
        report = build_report_data(
            summary={
                "behavior_evaluation": {"capacity_sweeps": capacity_payload}
            },
            trace=[],
            behavior_rows=[],
        )
        report["aggregate_benchmark_tables"]["capacity_sweep_curves"] = {
            "available": False,
            "artifact_state": "expected_but_missing",
            "rows": [],
            "limitations": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        capacity_section = report_md.split("## Capacity Sweep", 1)[1].split(
            "## Primary Benchmark",
            1,
        )[0]

        self.assertIn("fallback interpretations are reported below", capacity_section)
        self.assertNotIn("expected in this artifact but missing", capacity_section)
        self.assertIn("| three_center_modular |", capacity_section)

    def test_claim_tests_marked_not_in_artifact_when_absent(self) -> None:
        report = build_report_data(summary={}, trace=[], behavior_rows=[])
        self.assertEqual(report["claim_test_tables"]["artifact_state"], "not_in_this_artifact")


class OfflineAnalysisCreditFallbackTest(unittest.TestCase):
    def test_credit_analysis_falls_back_to_credit_assignment_strategy_rows(self) -> None:
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "three_center_modular": {
                            "config": {"architecture": "modular", "credit_strategy": "broadcast"},
                            "summary": {
                                "scenario_success_rate": 0.55,
                                "episode_success_rate": 0.55,
                                "dominant_module": "perception_center",
                                "mean_dominant_module_share": 0.4,
                                "mean_effective_module_count": 3.0,
                            },
                            "suite": {"night_rest": {"success_rate": 0.55}},
                        },
                        "three_center_modular_local_credit": {
                            "config": {"architecture": "modular", "credit_strategy": "local_only"},
                            "summary": {
                                "scenario_success_rate": 0.35,
                                "episode_success_rate": 0.35,
                                "dominant_module": "perception_center",
                                "mean_dominant_module_share": 0.5,
                                "mean_effective_module_count": 1.0,
                            },
                            "suite": {"night_rest": {"success_rate": 0.35}},
                        },
                        "modular_full": {
                            "config": {"architecture": "modular", "credit_strategy": "broadcast"},
                            "summary": {
                                "scenario_success_rate": 0.65,
                                "episode_success_rate": 0.65,
                                "dominant_module": "visual_cortex",
                                "mean_dominant_module_share": 0.45,
                                "mean_effective_module_count": 4.0,
                            },
                            "suite": {"night_rest": {"success_rate": 0.65}},
                        },
                    },
                }
            }
        }

        report = build_report_data(summary=summary, trace=[], behavior_rows=[])
        rows = report["credit_analysis"]["architecture_strategy_matrix"]["rows"]
        strategies = {(row["architecture_rung"], row["credit_strategy"]) for row in rows}

        self.assertEqual(report["credit_analysis"]["artifact_state"], "available")
        self.assertIn(("A2", "broadcast"), strategies)
        self.assertIn(("A2", "local_only"), strategies)
        self.assertIn(("A4", "broadcast"), strategies)


class CombinedLadderReportTest(unittest.TestCase):
    def test_build_combined_ladder_report_merges_local_tasks_and_distillation(self) -> None:
        capacity_payload, _ = compare_capacity_sweep(
            episodes=0,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
        )
        capacity_report = build_report_data(
            summary={"config": {"brain": {"name": "modular_full", "architecture": "modular"}}, "behavior_evaluation": {"capacity_sweeps": capacity_payload}},
            trace=[],
            behavior_rows=[],
        )
        ablation_report = build_report_data(
            summary=build_uncertainty_summary(),
            trace=[],
            behavior_rows=[],
        )
        profile_report = {
            "reward_profile_ladder_tables": {
                "available": True,
                "artifact_state": "available",
                "rows": {"ladder_by_profile": [{"protocol_name": "A2_three_center"}]},
                "limitations": [],
            },
            "ladder_profile_comparison": {
                "available": True,
                "artifact_state": "available",
                "classification_summary": {"rows": [{"protocol_name": "A2_three_center"}]},
                "limitations": [],
            },
            "shaping_program": dict(ablation_report["shaping_program"]),
            "limitations": [],
        }
        module_local_payload = {
            "variant_modules": {
                "visual_cortex": [
                    {
                        "seed": 7,
                        "report": {
                            "module_name": "visual_cortex",
                            "minimal_sufficient_level": 3,
                            "levels": [{"level": 4, "all_tasks_passed": True}],
                        },
                    }
                ]
            },
            "canonical_only_modules": {},
            "paper_gate_pass": True,
            "blocked_reasons": [],
            "partial_variant_coverage": ["visual_cortex"],
        }
        distillation_summary = {
            "distillation": {
                "post_distillation_comparison": {
                    "teacher": {"episodes": 2, "survival_rate": 0.5, "mean_reward": -1.0, "mean_food_distance_delta": 1.0},
                    "modular_distilled": {"episodes": 2, "survival_rate": 0.25, "mean_reward": -2.0, "mean_food_distance_delta": 0.5},
                    "assessment": {
                        "answer": "inconclusive",
                        "rationale": "Teacher reference remained weak.",
                    },
                }
            }
        }

        report = build_combined_ladder_report(
            ablation_report=ablation_report,
            profile_report=profile_report,
            capacity_report=capacity_report,
            module_local_payload=module_local_payload,
            distillation_summary=distillation_summary,
            ablation_summary_path="ablation_summary.json",
            ablation_behavior_csv_path="ablation_rows.csv",
            profile_summary_path="profile_summary.json",
            profile_behavior_csv_path="profile_rows.csv",
            capacity_summary_path="capacity_summary.json",
            capacity_behavior_csv_path="capacity_rows.csv",
            module_local_report_path="module_variant_sufficiency.json",
            distillation_summary_path="student_summary.json",
        )

        self.assertTrue(report["module_local_sufficiency"]["available"])
        self.assertTrue(report["distillation_analysis"]["available"])
        self.assertTrue(report["capacity_sweeps"]["available"])
        self.assertTrue(report["reward_profile_ladder_tables"]["available"])

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Module-Local Sufficiency", report_md)
        self.assertIn("## Distillation Diagnostic", report_md)
