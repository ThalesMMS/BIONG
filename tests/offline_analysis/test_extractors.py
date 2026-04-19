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
    extract_noise_robustness,
    extract_predator_type_specialization,
    extract_reflex_frequency,
    extract_representation_specialization,
    extract_shaping_audit,
    extract_training_eval_series,
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
