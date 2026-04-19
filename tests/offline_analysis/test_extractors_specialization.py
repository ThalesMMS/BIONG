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

class RepresentationSpecializationReportTest(unittest.TestCase):
    def _summary(self) -> dict[str, object]:
        return {
            "evaluation": {
                "mean_proposer_divergence_by_module": {
                    "visual_cortex": 0.72,
                    "sensory_cortex": 0.58,
                },
                "mean_action_center_gate_differential": {
                    "visual_cortex": 0.31,
                    "sensory_cortex": -0.27,
                },
                "mean_action_center_contribution_differential": {
                    "visual_cortex": 0.22,
                    "sensory_cortex": -0.19,
                },
                "mean_representation_specialization_score": 0.65,
            }
        }

    def test_extract_representation_specialization_reads_evaluation_aggregate(self) -> None:
        result = extract_representation_specialization(self._summary(), [])

        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.evaluation")
        self.assertEqual(result["interpretation"], "high")
        self.assertAlmostEqual(
            result["proposer_divergence"]["visual_cortex"],
            0.72,
        )
        self.assertAlmostEqual(
            result["action_center_gate_differential"]["sensory_cortex"],
            -0.27,
        )
        self.assertAlmostEqual(
            result["representation_specialization_score"],
            0.65,
        )

    def test_extract_representation_specialization_falls_back_to_suite(self) -> None:
        summary = {
            "behavior_evaluation": {
                "suite": {
                    "visual_olfactory_pincer": {
                        "mean_proposer_divergence_by_module": {
                            "visual_cortex": 0.6,
                            "sensory_cortex": 0.4,
                        },
                        "mean_action_center_gate_differential": {
                            "visual_cortex": 0.2,
                        },
                        "mean_action_center_contribution_differential": {
                            "visual_cortex": 0.1,
                        },
                        "mean_representation_specialization_score": 0.5,
                    },
                    "olfactory_ambush": {
                        "mean_proposer_divergence_by_module": {
                            "visual_cortex": 0.4,
                            "sensory_cortex": 0.2,
                        },
                        "mean_action_center_gate_differential": {
                            "visual_cortex": 0.0,
                        },
                        "mean_action_center_contribution_differential": {
                            "visual_cortex": -0.1,
                        },
                        "mean_representation_specialization_score": 0.3,
                    },
                }
            }
        }

        result = extract_representation_specialization(summary, [])

        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.behavior_evaluation.suite")
        self.assertAlmostEqual(
            result["proposer_divergence"]["visual_cortex"],
            0.5,
        )
        self.assertAlmostEqual(
            result["action_center_gate_differential"]["visual_cortex"],
            0.1,
        )
        self.assertAlmostEqual(
            result["action_center_contribution_differential"]["visual_cortex"],
            0.0,
        )
        self.assertAlmostEqual(
            result["representation_specialization_score"],
            0.4,
        )
        self.assertEqual(result["interpretation"], "moderate")

    def test_extract_representation_specialization_reads_suite_legacy_metrics(self) -> None:
        summary = {
            "behavior_evaluation": {
                "suite": {
                    "visual_olfactory_pincer": {
                        "legacy_metrics": {
                            "mean_proposer_divergence_by_module": {
                                "visual_cortex": 0.42,
                            },
                            "mean_action_center_gate_differential": {
                                "visual_cortex": 0.18,
                            },
                            "mean_action_center_contribution_differential": {
                                "visual_cortex": 0.12,
                            },
                            "mean_representation_specialization_score": 0.42,
                        }
                    }
                }
            }
        }

        result = extract_representation_specialization(summary, [])

        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.behavior_evaluation.suite")
        self.assertAlmostEqual(
            result["proposer_divergence"]["visual_cortex"],
            0.42,
        )
        self.assertAlmostEqual(
            result["action_center_gate_differential"]["visual_cortex"],
            0.18,
        )
        self.assertAlmostEqual(
            result["action_center_contribution_differential"]["visual_cortex"],
            0.12,
        )
        self.assertAlmostEqual(
            result["representation_specialization_score"],
            0.42,
        )

    def test_extract_representation_specialization_returns_unavailable_when_missing(self) -> None:
        result = extract_representation_specialization({}, [])

        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "none")
        self.assertEqual(result["interpretation"], "insufficient_data")

    def test_extract_representation_specialization_classifies_low_score(self) -> None:
        summary = {
            "evaluation": {
                "mean_proposer_divergence_by_module": {
                    "visual_cortex": 0.08,
                },
                "mean_representation_specialization_score": 0.08,
            }
        }

        result = extract_representation_specialization(summary, [])

        # Low behavioral specialization with only emerging internal separation
        # should stay in the "low" interpretation bucket until the aggregate
        # score clears the moderate threshold.
        self.assertTrue(result["available"])
        self.assertEqual(result["interpretation"], "low")

    def test_extract_representation_specialization_skips_invalid_alias_values(self) -> None:
        summary = {
            "evaluation": {
                "mean_proposer_divergence_by_module": {
                    "visual_cortex": object(),
                },
                "proposer_divergence_by_module": {
                    "visual_cortex": 0.44,
                },
                "mean_representation_specialization_score": "not-a-number",
                "representation_specialization_score": 0.37,
            }
        }

        result = extract_representation_specialization(summary, [])

        self.assertTrue(result["available"])
        self.assertAlmostEqual(
            result["proposer_divergence"]["visual_cortex"],
            0.44,
        )
        self.assertAlmostEqual(
            result["representation_specialization_score"],
            0.37,
        )

    def test_build_report_data_includes_representation_specialization(self) -> None:
        report = build_report_data(summary=self._summary(), trace=[], behavior_rows=[])

        self.assertIn("representation_specialization", report)
        self.assertTrue(report["representation_specialization"]["available"])

    def test_write_report_contains_representation_specialization_section_and_svg(self) -> None:
        report = build_report_data(summary=self._summary(), trace=[], behavior_rows=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            generated = write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
            self.assertTrue(
                Path(generated["representation_specialization_svg"]).exists()
            )

        self.assertIn("## Representation Specialization", report_md)
        self.assertIn("representation_specialization.svg", report_md)
        self.assertIn("| visual_cortex | 0.72 | high |", report_md)
        self.assertIn(
            "| visual_cortex | 0.31 | 0.22 |",
            report_md,
        )

    def test_write_report_json_includes_representation_specialization(self) -> None:
        report = build_report_data(summary=self._summary(), trace=[], behavior_rows=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_json = json.loads(
                (Path(tmpdir) / "report.json").read_text(encoding="utf-8")
            )

        self.assertIn("representation_specialization", report_json)
        self.assertTrue(report_json["representation_specialization"]["available"])

class PredatorTypeSpecializationReportTest(unittest.TestCase):
    def _summary(self) -> dict[str, object]:
        """
        Return a synthetic evaluation payload containing mean module responses by predator type.
        
        The returned dictionary mimics the structure produced by evaluation aggregates and includes
        per-predator-type mean responses for modules such as `visual_cortex`, `sensory_cortex`, and `alert_center`.
        
        Returns:
            dict[str, object]: A mapping with key `"evaluation"` containing
            `mean_module_response_by_predator_type`, where each predator type maps module names
            to their mean response values (floats between 0.0 and 1.0).
        """
        return {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {
                        "visual_cortex": 0.75,
                        "sensory_cortex": 0.20,
                        "alert_center": 0.05,
                    },
                    "olfactory": {
                        "visual_cortex": 0.10,
                        "sensory_cortex": 0.80,
                        "alert_center": 0.10,
                    },
                }
            }
        }

    def test_extract_predator_type_specialization_reads_evaluation_aggregate(self) -> None:
        result = extract_predator_type_specialization(self._summary(), [], [])

        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.evaluation")
        self.assertEqual(
            result["predator_types"]["visual"]["dominant_module"],
            "visual_cortex",
        )
        self.assertEqual(
            result["predator_types"]["olfactory"]["dominant_module"],
            "sensory_cortex",
        )
        self.assertGreater(result["specialization_score"], 0.5)

    def test_build_report_data_includes_predator_type_specialization(self) -> None:
        report = build_report_data(summary=self._summary(), trace=[], behavior_rows=[])

        self.assertIn("predator_type_specialization", report)
        self.assertTrue(report["predator_type_specialization"]["available"])

    def test_write_report_contains_predator_type_specialization_section(self) -> None:
        report = build_report_data(summary=self._summary(), trace=[], behavior_rows=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")

        self.assertIn("## Predator Type Specialization", report_md)
        self.assertIn("Specialization score:", report_md)
        self.assertIn("| visual | 0.75 | 0.20 | visual_cortex |", report_md)

class ExtractPredatorTypeSpecializationEdgeCasesTest(unittest.TestCase):
    """Edge-case and boundary tests for extract_predator_type_specialization() - new in this PR."""

    def test_empty_summary_returns_unavailable(self) -> None:
        result = extract_predator_type_specialization({}, [], [])
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "none")

    def test_unavailable_result_has_required_keys(self) -> None:
        result = extract_predator_type_specialization({}, [], [])
        for key in ("available", "source", "predator_types", "differential_activation",
                    "type_module_correlation", "specialization_score", "interpretation", "limitations"):
            self.assertIn(key, result)

    def test_unavailable_result_has_unavailable_interpretation(self) -> None:
        result = extract_predator_type_specialization({}, [], [])
        self.assertEqual(result["interpretation"], "unavailable")

    def test_low_specialization_score_interpretation(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.5, "sensory_cortex": 0.5},
                    "olfactory": {"visual_cortex": 0.5, "sensory_cortex": 0.5},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertIn(result["interpretation"], ("low", "moderate", "high"))
        # Same distributions -> specialization near 0 -> low
        self.assertEqual(result["interpretation"], "low")

    def test_high_specialization_score_interpretation(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.95, "sensory_cortex": 0.05},
                    "olfactory": {"visual_cortex": 0.05, "sensory_cortex": 0.95},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["interpretation"], "high")
        self.assertGreater(result["specialization_score"], 0.5)

    def test_moderate_specialization_score_interpretation(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.65, "sensory_cortex": 0.35},
                    "olfactory": {"visual_cortex": 0.30, "sensory_cortex": 0.70},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["interpretation"], "moderate")

    def test_specialization_score_bounded_between_zero_and_one(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 1.0, "sensory_cortex": 0.0},
                    "olfactory": {"visual_cortex": 0.0, "sensory_cortex": 1.0},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertGreaterEqual(result["specialization_score"], 0.0)
        self.assertLessEqual(result["specialization_score"], 1.0)

    def test_type_module_correlation_bounded_between_zero_and_one(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                    "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertGreaterEqual(result["type_module_correlation"], 0.0)
        self.assertLessEqual(result["type_module_correlation"], 1.0)

    def test_falls_back_to_behavior_evaluation_legacy_scenarios(self) -> None:
        summary = {
            "behavior_evaluation": {
                "legacy_scenarios": {
                    "visual_olfactory_pincer": {
                        "module_response_by_predator_type": {
                            "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                            "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                        }
                    }
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertIn("legacy_scenarios", result["source"])

    def test_prefers_paired_suite_data_over_partial_earlier_candidate(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.9, "sensory_cortex": 0.1},
                }
            },
            "behavior_evaluation": {
                "suite": {
                    "visual_hunter_open_field": {
                        "module_response_by_predator_type": {
                            "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                            "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                        }
                    }
                }
            },
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.behavior_evaluation.suite")
        self.assertGreater(result["specialization_score"], 0.0)
        self.assertNotEqual(result["interpretation"], "insufficient_data")

    def test_prefers_suite_over_legacy_when_both_have_paired_data(self) -> None:
        summary = {
            "behavior_evaluation": {
                "legacy_scenarios": {
                    "visual_olfactory_pincer": {
                        "module_response_by_predator_type": {
                            "visual": {"visual_cortex": 0.6, "sensory_cortex": 0.4},
                            "olfactory": {"visual_cortex": 0.5, "sensory_cortex": 0.5},
                        }
                    }
                },
                "suite": {
                    "visual_hunter_open_field": {
                        "module_response_by_predator_type": {
                            "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                            "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                        }
                    }
                },
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "summary.behavior_evaluation.suite")

    def test_result_includes_differential_activation_keys(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
                    "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertIn("visual_cortex_visual_minus_olfactory", result["differential_activation"])
        self.assertIn("sensory_cortex_olfactory_minus_visual", result["differential_activation"])

    def test_result_predator_types_include_visual_and_olfactory(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
                    "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertIn("visual", result["predator_types"])
        self.assertIn("olfactory", result["predator_types"])

    def test_single_predator_type_returns_insufficient_data(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        self.assertTrue(result["available"])
        self.assertEqual(result["specialization_score"], 0.0)
        self.assertEqual(result["type_module_correlation"], 0.0)
        self.assertEqual(result["interpretation"], "insufficient_data")
        self.assertEqual(
            result["differential_activation"]["visual_cortex_visual_minus_olfactory"],
            0.0,
        )
        self.assertEqual(
            result["differential_activation"]["sensory_cortex_olfactory_minus_visual"],
            0.0,
        )
        self.assertTrue(
            any("both visual and olfactory predators" in limitation for limitation in result["limitations"])
        )

    def test_differential_activation_visual_cortex_positive_for_specialized(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                    "olfactory": {"visual_cortex": 0.1, "sensory_cortex": 0.9},
                }
            }
        }
        result = extract_predator_type_specialization(summary, [], [])
        differential = result["differential_activation"]
        # visual cortex responds more to visual predators -> positive
        self.assertGreater(differential["visual_cortex_visual_minus_olfactory"], 0.0)
        # sensory cortex responds more to olfactory predators -> positive
        self.assertGreater(differential["sensory_cortex_olfactory_minus_visual"], 0.0)

    def test_write_report_includes_differential_activation_table(self) -> None:
        summary = {
            "evaluation": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.75, "sensory_cortex": 0.25},
                    "olfactory": {"visual_cortex": 0.15, "sensory_cortex": 0.85},
                }
            }
        }
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("sensory_cortex (olfactory - visual)", report_md)
        self.assertIn("visual_cortex (visual - olfactory)", report_md)

class AggregateSpecializationFromScenariosTest(unittest.TestCase):
    """Tests for offline_analysis._aggregate_specialization_from_scenarios() - new in this PR."""

    def test_returns_mean_across_scenarios(self) -> None:
        scenarios = {
            "scenario_a": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.6, "sensory_cortex": 0.4},
                }
            },
            "scenario_b": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.8, "sensory_cortex": 0.2},
                }
            },
        }
        result = _aggregate_specialization_from_scenarios(scenarios)
        self.assertIn("visual", result)
        self.assertAlmostEqual(result["visual"]["visual_cortex"], 0.7)
        self.assertAlmostEqual(result["visual"]["sensory_cortex"], 0.3)

    def test_empty_scenarios_returns_empty(self) -> None:
        result = _aggregate_specialization_from_scenarios({})
        self.assertEqual(result, {})

    def test_non_mapping_scenario_payload_is_skipped(self) -> None:
        scenarios = {
            "scenario_a": "not_a_mapping",
            "scenario_b": {
                "mean_module_response_by_predator_type": {
                    "visual": {"visual_cortex": 0.5},
                }
            },
        }
        result = _aggregate_specialization_from_scenarios(scenarios)
        self.assertAlmostEqual(result["visual"]["visual_cortex"], 0.5)

    def test_falls_back_to_module_response_by_predator_type_key(self) -> None:
        scenarios = {
            "scenario_a": {
                "module_response_by_predator_type": {
                    "olfactory": {"sensory_cortex": 0.9},
                }
            }
        }
        result = _aggregate_specialization_from_scenarios(scenarios)
        self.assertIn("olfactory", result)
        self.assertAlmostEqual(result["olfactory"]["sensory_cortex"], 0.9)

    def test_aggregates_across_predator_types(self) -> None:
        scenarios = {
            "s1": {
                "mean_module_response_by_predator_type": {
                    "visual": {"vc": 0.7},
                    "olfactory": {"sc": 0.8},
                }
            }
        }
        result = _aggregate_specialization_from_scenarios(scenarios)
        self.assertIn("visual", result)
        self.assertIn("olfactory", result)

class NormalizeModuleResponseByPredatorTypeTest(unittest.TestCase):
    """Tests for offline_analysis._normalize_module_response_by_predator_type() - new in this PR."""

    def test_valid_nested_mapping_converts_values_to_float(self) -> None:
        value = {
            "visual": {"visual_cortex": 0.7, "sensory_cortex": 0.3},
            "olfactory": {"visual_cortex": 0.2, "sensory_cortex": 0.8},
        }
        result = _normalize_module_response_by_predator_type(value)
        self.assertAlmostEqual(result["visual"]["visual_cortex"], 0.7)
        self.assertAlmostEqual(result["olfactory"]["sensory_cortex"], 0.8)

    def test_non_mapping_input_returns_empty(self) -> None:
        result = _normalize_module_response_by_predator_type("not_a_mapping")
        self.assertEqual(result, {})

    def test_none_input_returns_empty(self) -> None:
        result = _normalize_module_response_by_predator_type(None)
        self.assertEqual(result, {})

    def test_list_input_returns_empty(self) -> None:
        result = _normalize_module_response_by_predator_type([1, 2, 3])
        self.assertEqual(result, {})

    def test_non_mapping_predator_entry_is_skipped(self) -> None:
        value = {
            "visual": "not_a_mapping",
            "olfactory": {"sensory_cortex": 0.5},
        }
        result = _normalize_module_response_by_predator_type(value)
        self.assertNotIn("visual", result)
        self.assertIn("olfactory", result)

    def test_keys_are_strings(self) -> None:
        value = {"visual": {"a": 1.0}}
        result = _normalize_module_response_by_predator_type(value)
        for key in result:
            self.assertIsInstance(key, str)
        for key in result.get("visual", {}):
            self.assertIsInstance(key, str)

    def test_values_are_floats(self) -> None:
        value = {"visual": {"module_a": 1}}
        result = _normalize_module_response_by_predator_type(value)
        for val in result.get("visual", {}).values():
            self.assertIsInstance(val, float)

    def test_empty_mapping_returns_empty(self) -> None:
        result = _normalize_module_response_by_predator_type({})
        self.assertEqual(result, {})

    def test_invalid_value_coerced_to_zero(self) -> None:
        value = {"visual": {"module_a": None}}
        result = _normalize_module_response_by_predator_type(value)
        self.assertAlmostEqual(result["visual"]["module_a"], 0.0)

class ExtractAblationsPredatorTypeComparisonsTest(unittest.TestCase):
    """Tests that extract_ablations includes predator_type_comparisons - new in this PR."""

    def test_extract_ablations_from_summary_includes_predator_type_comparisons(self) -> None:
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "reference_variant": "modular_full",
                    "variants": {
                        "modular_full": {
                            "suite": {
                                "visual_olfactory_pincer": {"success_rate": 0.8},
                                "olfactory_ambush": {"success_rate": 0.7},
                                "visual_hunter_open_field": {"success_rate": 0.9},
                            }
                        },
                        "drop_visual_cortex": {
                            "suite": {
                                "visual_olfactory_pincer": {"success_rate": 0.3},
                                "olfactory_ambush": {"success_rate": 0.7},
                                "visual_hunter_open_field": {"success_rate": 0.2},
                            }
                        },
                    },
                }
            }
        }
        result = extract_ablations(summary, [])
        self.assertIn("predator_type_comparisons", result)
        comparisons = result["predator_type_comparisons"]
        self.assertIn("available", comparisons)

    def test_extract_ablations_from_csv_includes_predator_type_comparisons(self) -> None:
        rows = normalize_behavior_rows([
            {
                "scenario": "visual_olfactory_pincer",
                "success": True,
                "ablation_variant": "drop_visual_cortex",
                "ablation_architecture": "modular",
                "eval_reflex_scale": 1.0,
            },
            {
                "scenario": "olfactory_ambush",
                "success": False,
                "ablation_variant": "drop_visual_cortex",
                "ablation_architecture": "modular",
                "eval_reflex_scale": 1.0,
            },
        ])
        result = extract_ablations({}, rows)
        self.assertIn("predator_type_comparisons", result)
