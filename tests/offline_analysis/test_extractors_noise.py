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
