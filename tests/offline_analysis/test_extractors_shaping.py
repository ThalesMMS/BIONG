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

class OfflineAnalysisReflexFrequencyTest(unittest.TestCase):
    def test_extract_reflex_frequency_uses_messages_without_debug(self) -> None:
        trace = [
            {
                "messages": [
                    {
                        "sender": "alert_center",
                        "topic": "action.proposal",
                        "payload": {"reflex": {"action": "MOVE_LEFT", "reason": "threat"}},
                    }
                ]
            },
            {
                "messages": [
                    {
                        "sender": "sleep_center",
                        "topic": "action.proposal",
                        "payload": {"reflex": {"action": "STAY", "reason": "rest"}},
                    }
                ]
            },
        ]
        result = extract_reflex_frequency(trace)

        self.assertTrue(result["available"])
        modules = {item["module"]: item for item in result["modules"]}
        self.assertEqual(modules["alert_center"]["reflex_events"], 1)
        self.assertEqual(modules["sleep_center"]["reflex_events"], 1)
        self.assertFalse(result["uses_debug_reflexes"])

    def test_extract_reflex_frequency_enriches_with_debug_payload(self) -> None:
        trace = [
            {
                "messages": [],
                "debug": {
                    "reflexes": {
                        "alert_center": {
                            "reflex": {"action": "MOVE_LEFT", "reason": "threat"},
                            "module_reflex_override": True,
                            "module_reflex_dominance": 0.75,
                        }
                    }
                },
            }
        ]
        result = extract_reflex_frequency(trace)

        modules = {item["module"]: item for item in result["modules"]}
        self.assertTrue(result["uses_debug_reflexes"])
        self.assertEqual(modules["alert_center"]["debug_reflex_events"], 1)
        self.assertGreater(modules["alert_center"]["override_rate"], 0.0)
        self.assertGreater(modules["alert_center"]["mean_dominance"], 0.0)

    def test_extract_reflex_frequency_ignores_non_reflex_debug_override_payloads(self) -> None:
        result = extract_reflex_frequency(
            [
                {
                    "debug": {
                        "reflexes": {
                            "visual_cortex": {
                                "reflex": False,
                                "module_reflex_override": True,
                                "module_reflex_dominance": 1.0,
                            }
                        }
                    }
                },
                {
                    "debug": {
                        "reflexes": {
                            "visual_cortex": {
                                "reflex": True,
                                "module_reflex_override": True,
                                "module_reflex_dominance": 0.4,
                            }
                        }
                    }
                },
            ]
        )

        row = next(
            item for item in result["modules"] if item["module"] == "visual_cortex"
        )
        self.assertEqual(row["debug_reflex_events"], 1)
        self.assertAlmostEqual(row["override_rate"], 1.0)
        self.assertAlmostEqual(row["mean_dominance"], 0.4)

    def test_extract_ablations_csv_scenario_success_is_unweighted_by_scenario(self) -> None:
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
                    "success": True,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 1.0,
                },
                {
                    "scenario": "open_field_foraging",
                    "success": False,
                    "ablation_variant": "modular_full",
                    "ablation_architecture": "modular",
                    "eval_reflex_scale": 1.0,
                },
            ]
        )

        ablations = extract_ablations({}, rows)

        summary = ablations["variants"]["modular_full"]["summary"]
        self.assertAlmostEqual(summary["scenario_success_rate"], 0.5)
        self.assertAlmostEqual(summary["episode_success_rate"], 2.0 / 3.0)

class ExtractShapingAuditEdgeCasesTest(unittest.TestCase):
    """Edge-case and boundary tests for extract_shaping_audit."""

    def test_extract_shaping_audit_empty_summary_returns_unavailable(self) -> None:
        result = extract_shaping_audit({})
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "none")
        self.assertEqual(result["dense_profile"], "classic")
        self.assertEqual(result["minimal_profile"], "austere")

    def test_extract_shaping_audit_no_reward_audit_sets_limitations(self) -> None:
        result = extract_shaping_audit({})
        self.assertIn("No reward_audit payload was available.", result["limitations"])

    def test_extract_shaping_audit_no_reward_audit_gap_metrics_are_zero(self) -> None:
        result = extract_shaping_audit({})
        self.assertAlmostEqual(result["gap_metrics"]["scenario_success_rate_delta"], 0.0)
        self.assertAlmostEqual(result["gap_metrics"]["episode_success_rate_delta"], 0.0)
        self.assertAlmostEqual(result["gap_metrics"]["mean_reward_delta"], 0.0)

    def test_extract_shaping_audit_no_reward_audit_flags_are_false(self) -> None:
        result = extract_shaping_audit({})
        self.assertFalse(result["interpretive_flags"]["gap_available"])
        self.assertFalse(result["interpretive_flags"]["shaping_dependent"])

    def test_extract_shaping_audit_available_with_survival_only_payload(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "behavior_survival": {
                        "available": True,
                        "minimal_profile": "austere",
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
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertTrue(result["available"])
        self.assertFalse(result["interpretive_flags"]["gap_available"])
        self.assertTrue(result["behavior_survival"]["available"])

    def test_extract_shaping_audit_thresholds_exposed_in_result(self) -> None:
        result = extract_shaping_audit({})
        self.assertAlmostEqual(
            result["thresholds"]["shaping_dependence"],
            SHAPING_DEPENDENCE_WARNING_THRESHOLD,
        )
        self.assertAlmostEqual(
            result["thresholds"]["behavior_survival"],
            DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        )

    def test_extract_shaping_audit_gap_exactly_at_threshold_is_not_dependent(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": SHAPING_DEPENDENCE_WARNING_THRESHOLD,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertFalse(result["interpretive_flags"]["shaping_dependent"])

    def test_extract_shaping_audit_gap_above_threshold_is_dependent(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": (
                                SHAPING_DEPENDENCE_WARNING_THRESHOLD
                                + SHAPING_GAP_EPSILON
                            ),
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertTrue(result["interpretive_flags"]["shaping_dependent"])

    def test_extract_shaping_audit_gap_zero_interpretation_says_matches_or_exceeds(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn("matches or exceeds", result["interpretation"])

    def test_extract_shaping_audit_gap_below_threshold_but_positive_interpretation(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": SMALL_SHAPING_GAP,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn("below", result["interpretation"])
        self.assertFalse(result["interpretive_flags"]["shaping_dependent"])

    def test_extract_shaping_audit_high_gap_interpretation_mentions_high_shaping(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": LARGE_SHAPING_GAP,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn("High shaping dependence", result["interpretation"])

    def test_extract_shaping_audit_falls_back_to_non_minimal_profile_as_dense(self) -> None:
        # If 'classic' is not in deltas_vs_minimal, picks another non-minimal profile
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "ecological": {
                            "scenario_success_rate_delta": 0.15,
                            "episode_success_rate_delta": 0.10,
                            "mean_reward_delta": 0.5,
                        },
                        "austere": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        },
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertEqual(result["dense_profile"], "ecological")
        self.assertTrue(result["interpretive_flags"]["gap_available"])

    def test_extract_shaping_audit_limitation_when_no_component_classification(self) -> None:
        # reward_audit present but no reward_components → limitation listed
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn("No reward component disposition table was available.", result["limitations"])

    def test_extract_shaping_audit_limitation_when_no_behavior_survival(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertIn(
            "No austere per-scenario behavior survival data was available.",
            result["limitations"],
        )

    def test_extract_shaping_audit_removed_weight_gap_computed_from_profiles(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                },
                "reward_profiles": {
                    "classic": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 2.5},
                        }
                    },
                    "austere": {
                        "disposition_summary": {
                            "removed": {"total_weight_proxy": 0.0},
                        }
                    },
                },
            }
        }
        result = extract_shaping_audit(summary)
        self.assertAlmostEqual(result["removed_weight_gap"], 2.5)

    def test_extract_shaping_audit_source_is_summary_reward_audit(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    }
                }
            }
        }
        result = extract_shaping_audit(summary)
        self.assertEqual(result["source"], "summary.reward_audit")

    def test_extract_shaping_audit_behavior_survival_normalized(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                    "behavior_survival": {
                        "available": True,
                        "minimal_profile": "austere",
                        "survival_threshold": 0.5,
                        "scenario_count": 2,
                        "surviving_scenario_count": 1,
                        "survival_rate": 0.5,
                        "scenarios": {
                            "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 2},
                            "open_field": {"austere_success_rate": 0.0, "survives": False, "episodes": 2},
                        },
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        bs = result["behavior_survival"]
        self.assertTrue(bs["available"])
        self.assertEqual(bs["surviving_scenario_count"], 1)
        self.assertAlmostEqual(bs["survival_rate"], 0.5)

    def test_extract_shaping_audit_no_limitations_when_all_data_present(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                    "behavior_survival": {
                        "available": True,
                        "scenarios": {
                            "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1}
                        },
                    },
                },
                "reward_components": {
                    "food_progress": {
                        "category": "progress",
                        "shaping_risk": "high",
                        "shaping_disposition": "removed",
                        "disposition_rationale": "Zeroed in austere.",
                    }
                },
            }
        }
        result = extract_shaping_audit(summary)
        self.assertEqual(result["limitations"], [])

    def test_extract_shaping_audit_write_report_no_warning_for_gap_at_threshold(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": SHAPING_DEPENDENCE_WARNING_THRESHOLD,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("WARNING: High shaping dependence detected", report_md)

    def test_write_report_mentions_missing_survival_when_shaping_program_exists(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.0,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("_No shaping survival data available._", report_md)

    def test_extract_shaping_audit_write_report_warning_just_above_threshold(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": SHAPING_DEPENDENCE_WARNING_THRESHOLD + 0.001,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                }
            }
        }
        report = build_report_data(summary=summary, trace=[], behavior_rows=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            write_report(tmpdir, report)
            report_md = (Path(tmpdir) / "report.md").read_text(encoding="utf-8")
        self.assertIn("WARNING: High shaping dependence detected", report_md)

    def test_extract_shaping_audit_build_report_shaping_program_unavailable_without_reward_audit(self) -> None:
        report = build_report_data(summary={}, trace=[], behavior_rows=[])
        self.assertIn("shaping_program", report)
        self.assertFalse(report["shaping_program"]["available"])

    def test_extract_shaping_audit_behavior_survival_scenarios_as_list_normalized(self) -> None:
        summary = {
            "reward_audit": {
                "comparison": {
                    "minimal_profile": "austere",
                    "deltas_vs_minimal": {
                        "classic": {
                            "scenario_success_rate_delta": 0.1,
                            "episode_success_rate_delta": 0.0,
                            "mean_reward_delta": 0.0,
                        }
                    },
                    "behavior_survival": {
                        "available": True,
                        "scenarios": [
                            {
                                "scenario": "night_rest",
                                "austere_success_rate": 1.0,
                                "survives": True,
                                "episodes": 5,
                            }
                        ],
                    },
                }
            }
        }
        result = extract_shaping_audit(summary)
        bs = result["behavior_survival"]
        self.assertTrue(bs["available"])
        scenarios = bs["scenarios"]
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]["scenario"], "night_rest")
