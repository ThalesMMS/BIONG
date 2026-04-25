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


class OfflineAnalysisLadderAndAblationTest(OfflineAnalysisToleranceFixtures, unittest.TestCase):
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
