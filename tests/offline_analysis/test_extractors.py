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

class OfflineAnalysisToleranceTest(unittest.TestCase):
    def _summary_with_shaping_program(self) -> dict[str, object]:
        """
        Return a synthetic summary dictionary that encodes a shaping-minimization program suitable for tests.
        
        The dictionary contains a `reward_audit` entry with:
        - `minimal_profile`: the identifier of the minimal shaping profile (`"austere"`).
        - `reward_components`: two example components (`"feeding"`, `"food_progress"`) with shaping risk, disposition, and optional rationale.
        - `reward_profiles`: `classic` and `austere` profiles, each providing `disposition_summary` values (`total_weight_proxy`) for dispositions such as `removed` and `outcome_signal`.
        - `comparison`: contains `minimal_profile`, `deltas_vs_minimal` (per-profile deltas for scenario/episode success rates and mean reward), and `behavior_survival` (availability flag, survival threshold, counts, and per-scenario survival details).
        
        Returns:
            dict: A synthetic summary shaped for shaping-audit and report-generation tests.
        """
        return {
            "reward_audit": {
                "minimal_profile": "austere",
                "reward_components": {
                    "feeding": {
                        "category": "event",
                        "shaping_risk": "low",
                        "shaping_disposition": "outcome_signal",
                    },
                    "food_progress": {
                        "category": "progress",
                        "shaping_risk": "high",
                        "shaping_disposition": "removed",
                        "disposition_rationale": "Zeroed in the austere profile.",
                    },
                },
                "reward_profiles": {
                    "classic": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 1.2},
                            "outcome_signal": {"total_weight_proxy": 3.48},
                        }
                    },
                    "austere": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 0.0},
                            "outcome_signal": {"total_weight_proxy": 3.48},
                        }
                    },
                },
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": LARGE_SHAPING_GAP,
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
                        "available": True,
                        "minimal_profile": "austere",
                        "survival_threshold": 0.5,
                        "scenario_count": 2,
                        "surviving_scenario_count": 1,
                        "survival_rate": 0.5,
                        "scenarios": {
                            "night_rest": {
                                "austere_success_rate": 1.0,
                                "survives": True,
                                "episodes": 2,
                            },
                            "open_field_foraging": {
                                "austere_success_rate": 0.0,
                                "survives": False,
                                "episodes": 2,
                            },
                        },
                    },
                },
            }
        }

    def _summary_with_credit_analysis_program(self) -> dict[str, object]:
        return {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "three_center_modular": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "broadcast",
                            },
                            "summary": {
                                "scenario_success_rate": 0.6,
                                "episode_success_rate": 0.6,
                                "mean_module_credit_weights": {
                                    "perception_center": 0.34,
                                    "action_center": 0.33,
                                    "context_center": 0.33,
                                },
                                "module_gradient_norm_means": {
                                    "perception_center": 0.4,
                                    "action_center": 0.4,
                                    "context_center": 0.4,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "perception_center": 0.2,
                                    "action_center": 0.2,
                                    "context_center": 0.2,
                                },
                                "dominant_module": "perception_center",
                                "mean_dominant_module_share": 0.34,
                                "mean_effective_module_count": 3.0,
                            },
                            "suite": {"night_rest": {"success_rate": 0.6}},
                        },
                        "three_center_modular_local_credit": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "local_only",
                            },
                            "summary": {
                                "scenario_success_rate": 0.25,
                                "episode_success_rate": 0.25,
                                "mean_module_credit_weights": {
                                    "perception_center": 0.0,
                                    "action_center": 0.0,
                                    "context_center": 0.0,
                                },
                                "module_gradient_norm_means": {
                                    "perception_center": 0.0,
                                    "action_center": 0.0,
                                    "context_center": 0.0,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "perception_center": 0.0,
                                    "action_center": 0.0,
                                    "context_center": 0.0,
                                },
                                "dominant_module": "",
                                "mean_dominant_module_share": 0.0,
                                "mean_effective_module_count": 0.0,
                            },
                            "suite": {"night_rest": {"success_rate": 0.25}},
                        },
                        "three_center_modular_counterfactual": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "counterfactual",
                            },
                            "summary": {
                                "scenario_success_rate": 0.75,
                                "episode_success_rate": 0.75,
                                "mean_module_credit_weights": {
                                    "perception_center": 0.65,
                                    "action_center": 0.2,
                                    "context_center": 0.15,
                                },
                                "module_gradient_norm_means": {
                                    "perception_center": 1.8,
                                    "action_center": 0.8,
                                    "context_center": 0.6,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "perception_center": 0.7,
                                    "action_center": 0.2,
                                    "context_center": 0.1,
                                },
                                "dominant_module": "perception_center",
                                "mean_dominant_module_share": 0.65,
                                "mean_effective_module_count": 1.9,
                            },
                            "suite": {"night_rest": {"success_rate": 0.75}},
                        },
                        "modular_full": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "broadcast",
                            },
                            "summary": {
                                "scenario_success_rate": 0.5,
                                "episode_success_rate": 0.5,
                                "mean_module_credit_weights": {
                                    "visual_cortex": 0.26,
                                    "motor_cortex": 0.25,
                                    "decision_cortex": 0.24,
                                    "memory_cortex": 0.25,
                                },
                                "module_gradient_norm_means": {
                                    "visual_cortex": 0.35,
                                    "motor_cortex": 0.35,
                                    "decision_cortex": 0.35,
                                    "memory_cortex": 0.35,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "visual_cortex": 0.15,
                                    "motor_cortex": 0.15,
                                    "decision_cortex": 0.15,
                                    "memory_cortex": 0.15,
                                },
                                "dominant_module": "visual_cortex",
                                "mean_dominant_module_share": 0.26,
                                "mean_effective_module_count": 4.0,
                            },
                            "suite": {"night_rest": {"success_rate": 0.5}},
                        },
                        "local_credit_only": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "local_only",
                            },
                            "summary": {
                                "scenario_success_rate": 0.05,
                                "episode_success_rate": 0.05,
                                "mean_module_credit_weights": {
                                    "visual_cortex": 0.0,
                                    "motor_cortex": 0.0,
                                    "decision_cortex": 0.0,
                                    "memory_cortex": 0.0,
                                },
                                "module_gradient_norm_means": {
                                    "visual_cortex": 0.0,
                                    "motor_cortex": 0.0,
                                    "decision_cortex": 0.0,
                                    "memory_cortex": 0.0,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "visual_cortex": 0.0,
                                    "motor_cortex": 0.0,
                                    "decision_cortex": 0.0,
                                    "memory_cortex": 0.0,
                                },
                                "dominant_module": "",
                                "mean_dominant_module_share": 0.0,
                                "mean_effective_module_count": 0.0,
                            },
                            "suite": {"night_rest": {"success_rate": 0.05}},
                        },
                        "counterfactual_credit": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "counterfactual",
                            },
                            "summary": {
                                "scenario_success_rate": 0.82,
                                "episode_success_rate": 0.82,
                                "mean_module_credit_weights": {
                                    "visual_cortex": 0.72,
                                    "motor_cortex": 0.12,
                                    "decision_cortex": 0.1,
                                    "memory_cortex": 0.06,
                                },
                                "module_gradient_norm_means": {
                                    "visual_cortex": 2.1,
                                    "motor_cortex": 0.8,
                                    "decision_cortex": 0.7,
                                    "memory_cortex": 0.5,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "visual_cortex": 0.75,
                                    "motor_cortex": 0.1,
                                    "decision_cortex": 0.08,
                                    "memory_cortex": 0.07,
                                },
                                "dominant_module": "visual_cortex",
                                "mean_dominant_module_share": 0.72,
                                "mean_effective_module_count": 1.7,
                            },
                            "suite": {"night_rest": {"success_rate": 0.82}},
                        },
                    },
                }
            }
        }

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

    def test_extract_ladder_comparison_returns_requested_shape(self) -> None:
        variants = {
            "true_monolithic_policy": {
                "summary": {
                    "scenario_success_rate": 0.2,
                    "episode_success_rate": 0.2,
                    "mean_reward": 2.0,
                }
            },
            "monolithic_policy": {
                "summary": {
                    "scenario_success_rate": 0.5,
                    "episode_success_rate": 0.4,
                    "mean_reward": 4.0,
                }
            },
            "three_center_modular": {
                "summary": {
                    "scenario_success_rate": 0.6,
                    "episode_success_rate": 0.55,
                    "mean_reward": 6.0,
                }
            },
            "four_center_modular": {
                "summary": {
                    "scenario_success_rate": 0.7,
                    "episode_success_rate": 0.625,
                    "mean_reward": 6.5,
                }
            },
            "modular_full": {
                "summary": {
                    "scenario_success_rate": 0.8,
                    "episode_success_rate": 0.7,
                    "mean_reward": 7.0,
                }
            },
        }

        result = extract_ladder_comparison(variants, source="test")

        self.assertTrue(result["available"])
        self.assertEqual(len(result["comparisons"]), 4)
        self.assertIn("A0", result["explanatory_notes"])
        self.assertIn("A1", result["explanatory_notes"])
        self.assertIn("A2", result["explanatory_notes"])
        self.assertIn("A3", result["explanatory_notes"])
        self.assertIn("A4", result["explanatory_notes"])
        a0_vs_a1 = next(
            item
            for item in result["comparisons"]
            if item["baseline_rung"] == "A0" and item["comparison_rung"] == "A1"
        )
        self.assertEqual(
            a0_vs_a1["metrics"]["baseline_variant"],
            "true_monolithic_policy",
        )
        self.assertEqual(
            a0_vs_a1["metrics"]["comparison_variant"],
            "monolithic_policy",
        )
        self.assertAlmostEqual(
            a0_vs_a1["deltas"]["scenario_success_rate_delta"],
            0.3,
        )
        self.assertAlmostEqual(
            a0_vs_a1["deltas"]["episode_success_rate_delta"],
            0.2,
        )
        self.assertAlmostEqual(
            a0_vs_a1["deltas"]["mean_reward_delta"],
            2.0,
        )
        a1_vs_a2 = next(
            item
            for item in result["comparisons"]
            if item["baseline_rung"] == "A1" and item["comparison_rung"] == "A2"
        )
        self.assertAlmostEqual(
            a1_vs_a2["deltas"]["scenario_success_rate_delta"],
            0.1,
        )
        a2_vs_a3 = next(
            item
            for item in result["comparisons"]
            if item["baseline_rung"] == "A2" and item["comparison_rung"] == "A3"
        )
        self.assertAlmostEqual(
            a2_vs_a3["deltas"]["scenario_success_rate_delta"],
            0.1,
        )
        a3_vs_a4 = next(
            item
            for item in result["comparisons"]
            if item["baseline_rung"] == "A3" and item["comparison_rung"] == "A4"
        )
        self.assertAlmostEqual(
            a3_vs_a4["deltas"]["scenario_success_rate_delta"],
            0.1,
        )

    def test_extract_ladder_comparison_preserves_missing_mean_reward(self) -> None:
        variants = {
            "true_monolithic_policy": {
                "summary": {
                    "scenario_success_rate": 0.2,
                    "episode_success_rate": 0.2,
                }
            },
            "monolithic_policy": {
                "summary": {
                    "scenario_success_rate": 0.5,
                    "episode_success_rate": 0.4,
                }
            },
        }

        result = extract_ladder_comparison(variants, source="test")

        a0_vs_a1 = next(
            item
            for item in result["comparisons"]
            if item["baseline_rung"] == "A0" and item["comparison_rung"] == "A1"
        )
        self.assertIsNone(a0_vs_a1["metrics"]["baseline"]["mean_reward"])
        self.assertIsNone(a0_vs_a1["metrics"]["comparison"]["mean_reward"])
        self.assertIsNone(a0_vs_a1["deltas"]["mean_reward_delta"])

    def test_extract_ladder_comparison_skips_missing_variant_gracefully(self) -> None:
        variants = {
            "monolithic_policy": {
                "summary": {
                    "scenario_success_rate": 0.5,
                    "episode_success_rate": 0.4,
                    "mean_reward": 4.0,
                }
            },
            "three_center_modular": {
                "summary": {
                    "scenario_success_rate": 0.6,
                    "episode_success_rate": 0.5,
                    "mean_reward": 5.5,
                }
            },
            "modular_full": {
                "summary": {
                    "scenario_success_rate": 0.8,
                    "episode_success_rate": 0.7,
                    "mean_reward": 7.0,
                }
            },
        }

        result = extract_ladder_comparison(variants, source="test")

        self.assertTrue(result["available"])
        self.assertEqual(len(result["comparisons"]), 1)
        only_comparison = next(
            item
            for item in result["comparisons"]
            if item["baseline_rung"] == "A1" and item["comparison_rung"] == "A2"
        )
        self.assertEqual(only_comparison["baseline_rung"], "A1")
        self.assertEqual(only_comparison["comparison_rung"], "A2")
        self.assertTrue(
            any("A0 vs A1" in item for item in result["limitations"]),
        )
        self.assertTrue(
            any("A2 vs A3" in item for item in result["limitations"]),
        )

    def test_extract_model_capacity_flat_payload_ignores_metadata_keys(self) -> None:
        result = extract_model_capacity(
            {
                "config": {
                    "brain": {
                        "name": "modular_full",
                        "architecture": "modular",
                    }
                },
                "parameter_counts": {
                    "architecture": "modular",
                    "visual_cortex": 120,
                    "motor_cortex": 80,
                    "total": 200,
                    "total_trainable": 200,
                },
            }
        )

        self.assertTrue(result["available"])
        self.assertEqual(result["total_trainable"], 200)
        self.assertEqual(
            [item["network"] for item in result["networks"]],
            ["visual_cortex", "motor_cortex"],
        )

    def test_extract_architecture_capacity_is_unavailable_when_variant_counts_missing(
        self,
    ) -> None:
        summary = build_uncertainty_summary()
        summary["behavior_evaluation"]["ablations"]["variants"]["broken_variant"] = {
            "config": {"architecture": "modular"}
        }

        result = extract_architecture_capacity(summary)

        self.assertFalse(result["available"])
        self.assertEqual(result["rows"], [])
        self.assertIn(
            "Variant broken_variant was missing parameter counts.",
            result["limitations"],
        )

    def test_extract_architecture_capacity_is_unavailable_for_non_mapping_variant(
        self,
    ) -> None:
        summary = build_uncertainty_summary()
        summary["behavior_evaluation"]["ablations"]["variants"]["broken_variant"] = None

        result = extract_architecture_capacity(summary)

        self.assertFalse(result["available"])
        self.assertEqual(result["rows"], [])
        self.assertIn(
            "Variant broken_variant had a non-mapping payload.",
            result["limitations"],
        )

    def test_extract_ablations_summary_promotes_without_reflex_support(self) -> None:
        """
        Verifies that `extract_ablations` promotes `without_reflex_support` metrics into primary variant summaries and that diagnostics reflect the promoted values.
        
        Asserts that when ablation variants include a `without_reflex_support` branch:
        - The variant's `summary` fields (e.g., `scenario_success_rate`) are replaced by the `without_reflex_support.summary` values.
        - The promoted variant receives `eval_reflex_scale == 0.0`.
        - Deltas versus the reference variant are computed from the promoted summaries.
        - `build_diagnostics` selects the best ablation variant based on the promoted scenario success rate and formats its label/value accordingly.
        """
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "modular_full": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 1.0},
                            "suite": {"night_rest": {"success_rate": 1.0}},
                            "without_reflex_support": {
                                "summary": {
                                    "scenario_success_rate": 0.2,
                                    "episode_success_rate": 0.2,
                                    "mean_reward": 2.0,
                                },
                                "suite": {"night_rest": {"success_rate": 0.2}},
                            },
                        },
                        "no_module_reflexes": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 0.1},
                            "suite": {"night_rest": {"success_rate": 0.1}},
                            "without_reflex_support": {
                                "summary": {
                                    "scenario_success_rate": 0.8,
                                    "episode_success_rate": 0.8,
                                    "mean_reward": 8.0,
                                },
                                "suite": {"night_rest": {"success_rate": 0.8}},
                            },
                        },
                    },
                }
            }
        }

        ablations = extract_ablations(summary, [])
        diagnostics = build_diagnostics(
            {},
            {"scenarios": []},
            ablations,
            {"modules": []},
        )

        self.assertAlmostEqual(
            ablations["variants"]["modular_full"]["summary"]["scenario_success_rate"],
            0.2,
        )
        self.assertEqual(
            ablations["variants"]["modular_full"]["summary"]["eval_reflex_scale"],
            0.0,
        )
        self.assertAlmostEqual(
            ablations["variants"]["no_module_reflexes"]["summary"]["scenario_success_rate"],
            0.8,
        )
        self.assertAlmostEqual(
            ablations["deltas_vs_reference"]["no_module_reflexes"]["summary"][
                "scenario_success_rate_delta"
            ],
            0.6,
        )
        best = next(
            item for item in diagnostics if item["label"] == "Best ablation variant"
        )
        self.assertEqual(best["value"], "no_module_reflexes (0.80)")

    def test_build_report_data_surfaces_no_reflex_primary_benchmark(self) -> None:
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "modular_full": {
                            "config": {"architecture": "modular"},
                            "summary": {"scenario_success_rate": 0.9},
                            "suite": {"night_rest": {"success_rate": 0.9}},
                            "without_reflex_support": {
                                "summary": {
                                    "scenario_success_rate": 0.4,
                                    "episode_success_rate": 0.4,
                                    "mean_reward": 4.0,
                                },
                                "suite": {"night_rest": {"success_rate": 0.4}},
                            },
                        }
                    },
                }
            },
            "evaluation_with_reflex_support": {
                "summary": {
                    "mean_final_reflex_override_rate": 0.2,
                    "mean_reflex_dominance": 0.5,
                }
            },
        }

        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertTrue(report["primary_benchmark"]["available"])
        self.assertEqual(report["primary_benchmark"]["metric"], "scenario_success_rate")
        self.assertAlmostEqual(
            report["primary_benchmark"]["scenario_success_rate"],
            0.4,
        )
        self.assertEqual(report["primary_benchmark"]["eval_reflex_scale"], 0.0)
        self.assertEqual(report["primary_benchmark"]["reference_variant"], "modular_full")
        diagnostics = {item["label"]: item for item in report["diagnostics"]}
        self.assertEqual(
            diagnostics["Reflex Dependence: override rate"]["status"],
            "warning",
        )
        self.assertTrue(
            diagnostics["Reflex Dependence: override rate"]["failure_indicator"]
        )
        self.assertEqual(
            diagnostics["Reflex Dependence: dominance"]["status"],
            "warning",
        )

    def test_build_report_data_does_not_treat_legacy_evaluation_as_no_reflex(self) -> None:
        summary = {"evaluation": {"scenario_success_rate": 0.9}}

        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertFalse(report["primary_benchmark"]["available"])
        self.assertIsNone(report["primary_benchmark"]["eval_reflex_scale"])
        self.assertEqual(report["primary_benchmark"]["source"], "none")

    def test_build_report_data_suppresses_reflex_dependence_without_debug_reflexes(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "visual_cortex",
                        "topic": "action.proposal",
                        "payload": {"reflex": True},
                    }
                ]
            }
        ]

        report = build_report_data(summary={}, trace=trace, behavior_rows=[])

        self.assertFalse(report["reflex_dependence"]["available"])
        diagnostics = {item["label"]: item for item in report["diagnostics"]}
        self.assertNotIn("Reflex Dependence: override rate", diagnostics)
        self.assertNotIn("Reflex Dependence: dominance", diagnostics)

    def test_write_report_ablation_table_includes_no_reflex_scale(self) -> None:
        report = build_report_data(
            summary={},
            trace=[],
            behavior_rows=[
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
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Primary Benchmark", report_md)
        self.assertIn(
            "| metric | scenario_success_rate | eval_reflex_scale | reference_variant | source |",
            report_md,
        )
        self.assertIn(
            "| variant | architecture | eval_reflex_scale | scenario_success_rate |",
            report_md,
        )
        self.assertIn("| modular_full | modular | 0.00 | 1.00 |", report_md)

    def test_extract_ablations_includes_predator_type_comparisons(self) -> None:
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "visual_hunter_open_field",
                    "success": False,
                    "ablation_variant": "drop_visual_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "olfactory_ambush",
                    "success": True,
                    "ablation_variant": "drop_visual_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "visual_hunter_open_field",
                    "success": True,
                    "ablation_variant": "drop_sensory_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "olfactory_ambush",
                    "success": False,
                    "ablation_variant": "drop_sensory_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
            ]
        )

        ablations = extract_ablations({}, rows)

        self.assertTrue(ablations["predator_type_comparisons"]["available"])
        comparisons = ablations["predator_type_comparisons"]["comparisons"]
        self.assertIn("drop_visual_cortex", comparisons)
        self.assertIn("drop_sensory_cortex", comparisons)

        unrelated_rows = normalize_behavior_rows(
            [
                {
                    "scenario": "open_field_foraging",
                    "success": True,
                    "ablation_variant": "drop_visual_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
                {
                    "scenario": "corridor_gauntlet",
                    "success": False,
                    "ablation_variant": "drop_sensory_cortex",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 0.0,
                },
            ]
        )
        unrelated_ablations = extract_ablations({}, unrelated_rows)
        self.assertFalse(unrelated_ablations["predator_type_comparisons"]["available"])

class ExtractNoiseRobustnessTest(unittest.TestCase):
    def test_extract_noise_robustness_reads_summary_payload(self) -> None:
        """
        Verifies that extract_noise_robustness reads a complete robustness_matrix from a summary payload and returns the expected matrix metadata and scores.
        
        Asserts that the result is marked available, that the source points to the summary robustness_matrix path, that train conditions in matrix_spec match the summary, that a specific matrix cell's scenario_success_rate matches the summary value, and that the reported robustness_score equals the provided value.
        """
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none", "low"],
                        "eval_conditions": ["none", "low"],
                        "cell_count": 4,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}},
                            "low": {"summary": {"scenario_success_rate": 0.5, "episode_success_rate": 0.5}},
                        },
                        "low": {
                            "none": {"summary": {"scenario_success_rate": 0.75, "episode_success_rate": 0.75}},
                            "low": {"summary": {"scenario_success_rate": 0.25, "episode_success_rate": 0.25}},
                        },
                    },
                    "train_marginals": {"none": 0.75, "low": 0.5},
                    "eval_marginals": {"none": 0.875, "low": 0.375},
                    "robustness_score": 0.625,
                    "diagonal_score": 0.625,
                    "off_diagonal_score": 0.625,
                }
            }
        }

        result = extract_noise_robustness(summary, [])

        self.assertTrue(result["available"])
        self.assertEqual(
            result["source"],
            "summary.behavior_evaluation.robustness_matrix",
        )
        self.assertEqual(result["matrix_spec"]["train_conditions"], ["none", "low"])
        self.assertEqual(result["matrix"]["none"]["low"]["scenario_success_rate"], 0.5)
        self.assertAlmostEqual(result["robustness_score"], 0.625)

    def test_extract_noise_robustness_falls_back_to_behavior_rows(self) -> None:
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "night_rest",
                    "success": True,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "none",
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "high",
                },
                {
                    "scenario": "night_rest",
                    "success": True,
                    "train_noise_profile": "low",
                    "eval_noise_profile": "none",
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "low",
                    "eval_noise_profile": "high",
                },
            ]
        )

        result = extract_noise_robustness({}, rows)

        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "behavior_csv")
        self.assertEqual(result["matrix_spec"]["cell_count"], 4)
        self.assertEqual(result["matrix"]["none"]["none"]["scenario_success_rate"], 1.0)
        self.assertEqual(result["matrix"]["low"]["high"]["scenario_success_rate"], 0.0)
        self.assertAlmostEqual(result["train_marginals"]["none"], 0.5)
        self.assertAlmostEqual(result["eval_marginals"]["high"], 0.0)

    def test_extract_noise_robustness_missing_data_returns_unavailable_structure(self) -> None:
        result = extract_noise_robustness({}, [])

        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "none")
        self.assertEqual(result["matrix_spec"]["cell_count"], 0)
        self.assertFalse(result["metadata"]["complete"])
        self.assertTrue(
            any(
                "No noise robustness matrix payload" in limitation
                for limitation in result["limitations"]
            )
        )

    def test_extract_noise_robustness_partial_summary_marks_incomplete(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none", "low"],
                        "eval_conditions": ["none", "high"],
                        "cell_count": 4,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}},
                        },
                    },
                }
            }
        }

        result = extract_noise_robustness(summary, [])

        self.assertTrue(result["available"])
        self.assertEqual(
            result["source"],
            "summary.behavior_evaluation.robustness_matrix",
        )
        self.assertFalse(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["available_cell_count"], 1)
        self.assertTrue(
            any(
                "some train/eval cells were missing" in limitation
                for limitation in result["limitations"]
            )
        )
        self.assertEqual(result["matrix"]["none"]["none"]["scenario_success_rate"], 1.0)
        self.assertIsNone(result["robustness_score"])

    def test_extract_noise_robustness_partial_summary_with_behavior_rows_reconstructs(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none", "low"],
                        "eval_conditions": ["none", "high"],
                        "cell_count": 4,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}},
                        },
                    },
                }
            }
        }
        rows = normalize_behavior_rows(
            [
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "high",
                },
                {
                    "scenario": "night_rest",
                    "success": True,
                    "train_noise_profile": "low",
                    "eval_noise_profile": "none",
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "low",
                    "eval_noise_profile": "high",
                },
            ]
        )

        result = extract_noise_robustness(summary, rows)

        self.assertTrue(result["available"])
        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["available_cell_count"], 4)
        self.assertEqual(result["matrix_spec"]["cell_count"], 4)
        self.assertEqual(result["matrix"]["none"]["none"]["scenario_success_rate"], 1.0)
        self.assertEqual(result["matrix"]["low"]["high"]["scenario_success_rate"], 0.0)
        self.assertFalse(
            any(
                "some train/eval cells were missing" in limitation
                for limitation in result["limitations"]
            )
        )

    def test_build_report_data_includes_noise_robustness(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none"],
                        "eval_conditions": ["none"],
                        "cell_count": 1,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0, "episode_success_rate": 1.0}}
                        }
                    },
                    "train_marginals": {"none": 1.0},
                    "eval_marginals": {"none": 1.0},
                    "robustness_score": 1.0,
                    "diagonal_score": 1.0,
                    "off_diagonal_score": 0.0,
                }
            }
        }

        report = build_report_data(summary=summary, trace=[], behavior_rows=[])

        self.assertIn("noise_robustness", report)
        self.assertTrue(report["noise_robustness"]["available"])

    def test_write_report_contains_noise_robustness_section_and_svg(self) -> None:
        report = build_report_data(
            summary={},
            trace=[],
            behavior_rows=[
                {
                    "scenario": "night_rest",
                    "success": True,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "none",
                },
                {
                    "scenario": "night_rest",
                    "success": False,
                    "train_noise_profile": "none",
                    "eval_noise_profile": "high",
                },
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Noise Robustness Matrix", report_md)
        self.assertIn("| train \\ eval |", report_md)
        self.assertIn("Overall robustness score:", report_md)
        self.assertTrue(generated["robustness_matrix_svg"])

    def test_write_report_marks_missing_robustness_cells_and_scores(self) -> None:
        report = build_report_data(
            summary={
                "behavior_evaluation": {
                    "robustness_matrix": {
                        "matrix_spec": {
                            "train_conditions": ["none", "low"],
                            "eval_conditions": ["none", "high"],
                            "cell_count": 4,
                        },
                        "matrix": {
                            "none": {
                                "none": {
                                    "summary": {
                                        "scenario_success_rate": 1.0,
                                        "episode_success_rate": 1.0,
                                    }
                                }
                            }
                        },
                    }
                }
            },
            trace=[],
            behavior_rows=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
            robustness_svg = Path(generated["robustness_matrix_svg"]).read_text(
                encoding="utf-8"
            )

        self.assertIn("| none | 1.00 | — | 1.00 |", report_md)
        self.assertIn("| low | — | — | — |", report_md)
        self.assertIn("| mean | 1.00 | — | — |", report_md)
        self.assertIn("Overall robustness score: —.", report_md)
        self.assertIn(">—</text>", robustness_svg)

class ExtractNoiseRobustnessEdgeCasesTest(unittest.TestCase):
    """Additional edge case and regression tests for extract_noise_robustness (new in this PR)."""

    def test_summary_without_matrix_spec_infers_conditions_from_matrix_keys(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix": {
                        "low": {
                            "high": {"summary": {"scenario_success_rate": 0.6}},
                        },
                    },
                }
            }
        }
        result = extract_noise_robustness(summary, [])
        self.assertIn("low", result["matrix_spec"]["train_conditions"])
        self.assertIn("high", result["matrix_spec"]["eval_conditions"])

    def test_behavior_rows_with_partial_noise_metadata_count_as_missing(self) -> None:
        rows = normalize_behavior_rows([
            {
                "scenario": "night_rest",
                "success": True,
                "noise_profile": "none",
                "train_noise_profile": "",
                "eval_noise_profile": "",
            }
        ])
        result = extract_noise_robustness({}, rows)
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "none")

    def test_diagonal_and_off_diagonal_scores_for_full_canonical_matrix(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none", "low"],
                        "eval_conditions": ["none", "low"],
                        "cell_count": 4,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 0.9}},
                            "low": {"summary": {"scenario_success_rate": 0.5}},
                        },
                        "low": {
                            "none": {"summary": {"scenario_success_rate": 0.6}},
                            "low": {"summary": {"scenario_success_rate": 0.8}},
                        },
                    },
                    "robustness_score": 0.7,
                    "diagonal_score": 0.85,
                    "off_diagonal_score": 0.55,
                }
            }
        }
        result = extract_noise_robustness(summary, [])
        # summary-provided scores are used when present
        self.assertAlmostEqual(result["robustness_score"], 0.7)
        self.assertAlmostEqual(result["diagonal_score"], 0.85)
        self.assertAlmostEqual(result["off_diagonal_score"], 0.55)

    def test_behavior_csv_source_computes_episode_count_per_cell(self) -> None:
        rows = normalize_behavior_rows([
            {"scenario": "night_rest", "success": True, "train_noise_profile": "none", "eval_noise_profile": "none"},
            {"scenario": "night_rest", "success": True, "train_noise_profile": "none", "eval_noise_profile": "none"},
            {"scenario": "night_rest", "success": False, "train_noise_profile": "none", "eval_noise_profile": "none"},
        ])
        result = extract_noise_robustness({}, rows)
        self.assertEqual(result["matrix"]["none"]["none"]["episode_count"], 3)
        self.assertAlmostEqual(result["matrix"]["none"]["none"]["scenario_success_rate"], 2 / 3)

    def test_metadata_complete_true_when_all_cells_present(self) -> None:
        summary = {
            "behavior_evaluation": {
                "robustness_matrix": {
                    "matrix_spec": {
                        "train_conditions": ["none"],
                        "eval_conditions": ["none"],
                        "cell_count": 1,
                    },
                    "matrix": {
                        "none": {
                            "none": {"summary": {"scenario_success_rate": 1.0}},
                        }
                    },
                }
            }
        }
        result = extract_noise_robustness(summary, [])
        self.assertTrue(result["metadata"]["complete"])
        self.assertEqual(result["metadata"]["available_cell_count"], 1)
        self.assertEqual(result["metadata"]["expected_cell_count"], 1)
