"""Tests for tables.py helpers not covered by test_tables.py.

Covers: build_scenario_checks_rows, build_reward_component_rows,
build_diagnostics, and _claim_uncertainty_for_condition.
"""
from __future__ import annotations

import unittest

from spider_cortex_sim.offline_analysis.extractors import (
    _compare_credit_across_architectures,
    _interpret_credit_failure,
    extract_credit_metrics,
)
from spider_cortex_sim.offline_analysis.tables import (
    _claim_uncertainty_for_condition,
    build_credit_assignment_tables,
    build_credit_table,
    build_diagnostics,
    build_reward_component_rows,
    build_scenario_checks_rows,
)


class BuildScenarioChecksRowsTest(unittest.TestCase):
    def _make_scenario_success(
        self,
        scenarios: list[dict],
        source: str = "summary.behavior_evaluation.suite",
    ) -> dict:
        return {"scenarios": scenarios, "source": source}

    def test_empty_scenarios_returns_empty_list(self) -> None:
        result = build_scenario_checks_rows(self._make_scenario_success([]))
        self.assertEqual(result, [])

    def test_scenario_with_no_checks_produces_no_rows(self) -> None:
        scenario_success = self._make_scenario_success([
            {"scenario": "night_rest", "checks": {}}
        ])
        result = build_scenario_checks_rows(scenario_success)
        self.assertEqual(result, [])

    def test_single_check_produces_one_row(self) -> None:
        scenario_success = self._make_scenario_success([
            {
                "scenario": "night_rest",
                "checks": {
                    "deep_night_shelter": {
                        "pass_rate": 0.9,
                        "mean_value": 0.95,
                        "expected": ">= 0.95",
                        "description": "Must sleep",
                    }
                },
            }
        ])
        rows = build_scenario_checks_rows(scenario_success)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["scenario"], "night_rest")
        self.assertEqual(row["check_name"], "deep_night_shelter")
        self.assertAlmostEqual(row["pass_rate"], 0.9)
        self.assertAlmostEqual(row["mean_value"], 0.95)
        self.assertEqual(row["expected"], ">= 0.95")

    def test_multiple_scenarios_with_checks(self) -> None:
        scenario_success = self._make_scenario_success([
            {
                "scenario": "s1",
                "checks": {
                    "check_a": {"pass_rate": 1.0, "mean_value": 1.0, "expected": "", "description": ""},
                    "check_b": {"pass_rate": 0.5, "mean_value": 0.5, "expected": "", "description": ""},
                },
            },
            {
                "scenario": "s2",
                "checks": {
                    "check_c": {"pass_rate": 0.8, "mean_value": 0.8, "expected": "", "description": ""},
                },
            },
        ])
        rows = build_scenario_checks_rows(scenario_success)
        self.assertEqual(len(rows), 3)
        scenarios_in_rows = {r["scenario"] for r in rows}
        self.assertIn("s1", scenarios_in_rows)
        self.assertIn("s2", scenarios_in_rows)

    def test_row_includes_source_from_scenario_success(self) -> None:
        scenario_success = self._make_scenario_success(
            [{"scenario": "s", "checks": {"c": {"pass_rate": 1.0, "mean_value": 1.0, "expected": "", "description": ""}}}],
            source="behavior_csv",
        )
        rows = build_scenario_checks_rows(scenario_success)
        self.assertEqual(rows[0]["source"], "behavior_csv")

    def test_pass_rate_rounded_to_six_decimals(self) -> None:
        scenario_success = self._make_scenario_success([
            {
                "scenario": "s",
                "checks": {
                    "c": {
                        "pass_rate": 1.0 / 3.0,
                        "mean_value": 0.0,
                        "expected": "",
                        "description": "",
                    }
                },
            }
        ])
        rows = build_scenario_checks_rows(scenario_success)
        self.assertEqual(rows[0]["pass_rate"], round(1.0 / 3.0, 6))

    def test_non_mapping_check_payload_skipped(self) -> None:
        scenario_success = self._make_scenario_success([
            {
                "scenario": "s",
                "checks": {
                    "valid_check": {"pass_rate": 1.0, "mean_value": 1.0, "expected": "", "description": ""},
                    "bad_check": "not_a_mapping",
                },
            }
        ])
        rows = build_scenario_checks_rows(scenario_success)
        check_names = [r["check_name"] for r in rows]
        self.assertIn("valid_check", check_names)
        self.assertNotIn("bad_check", check_names)

    def test_non_mapping_scenario_skipped(self) -> None:
        valid_check = {
            "pass_rate": 0.75,
            "mean_value": 0.5,
            "expected": ">=0.5",
            "description": "valid scenario check",
        }
        scenario_success = {
            "scenarios": [
                "not_a_mapping",
                {"scenario": "s", "checks": {"valid_check": valid_check}},
            ],
            "source": "s",
        }
        rows = build_scenario_checks_rows(scenario_success)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scenario"], "s")
        self.assertEqual(rows[0]["check_name"], "valid_check")
        self.assertEqual(rows[0]["pass_rate"], 0.75)


class BuildCreditAssignmentTablesTest(unittest.TestCase):
    def test_build_credit_assignment_tables_groups_a2_and_a4_strategies(self) -> None:
        ablations = {
            "variants": {
                "three_center_modular": {
                    "config": {"architecture": "modular", "credit_strategy": "broadcast"},
                    "summary": {
                        "scenario_success_rate": 0.6,
                        "episode_success_rate": 0.6,
                        "mean_module_credit_weights": {"perception_center": 1.0},
                        "module_gradient_norm_means": {"perception_center": 2.0},
                        "mean_counterfactual_credit_weights": {"perception_center": 0.0},
                        "mean_module_contribution_share": {"perception_center": 0.5},
                        "dominant_module_distribution": {"perception_center": 0.5},
                        "dominant_module": "perception_center",
                        "mean_dominant_module_share": 0.5,
                        "mean_effective_module_count": 2.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.6}},
                },
                "three_center_modular_local_credit": {
                    "config": {"architecture": "modular", "credit_strategy": "local_only"},
                    "summary": {
                        "scenario_success_rate": 0.4,
                        "episode_success_rate": 0.4,
                        "mean_module_credit_weights": {"perception_center": 0.0},
                        "module_gradient_norm_means": {"perception_center": 1.5},
                        "mean_counterfactual_credit_weights": {"perception_center": 0.0},
                        "mean_module_contribution_share": {"perception_center": 0.4},
                        "dominant_module_distribution": {"perception_center": 0.4},
                        "dominant_module": "perception_center",
                        "mean_dominant_module_share": 0.4,
                        "mean_effective_module_count": 1.8,
                    },
                    "suite": {"night_rest": {"success_rate": 0.4}},
                },
                "three_center_modular_counterfactual": {
                    "config": {"architecture": "modular", "credit_strategy": "counterfactual"},
                    "summary": {
                        "scenario_success_rate": 0.7,
                        "episode_success_rate": 0.7,
                        "mean_module_credit_weights": {"perception_center": 0.7},
                        "module_gradient_norm_means": {"perception_center": 2.2},
                        "mean_counterfactual_credit_weights": {"perception_center": 0.7},
                        "mean_module_contribution_share": {"perception_center": 0.55},
                        "dominant_module_distribution": {"perception_center": 0.55},
                        "dominant_module": "perception_center",
                        "mean_dominant_module_share": 0.55,
                        "mean_effective_module_count": 2.1,
                    },
                    "suite": {"night_rest": {"success_rate": 0.7}},
                },
                "modular_full": {
                    "config": {"architecture": "modular", "credit_strategy": "broadcast"},
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.5,
                        "mean_module_credit_weights": {"visual_cortex": 1.0},
                        "module_gradient_norm_means": {"visual_cortex": 3.0},
                        "mean_counterfactual_credit_weights": {"visual_cortex": 0.0},
                        "mean_module_contribution_share": {"visual_cortex": 0.3},
                        "dominant_module_distribution": {"visual_cortex": 0.3},
                        "dominant_module": "visual_cortex",
                        "mean_dominant_module_share": 0.3,
                        "mean_effective_module_count": 3.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.5}},
                },
                "local_credit_only": {
                    "config": {"architecture": "modular", "credit_strategy": "local_only"},
                    "summary": {
                        "scenario_success_rate": 0.55,
                        "episode_success_rate": 0.55,
                        "mean_module_credit_weights": {"visual_cortex": 0.0},
                        "module_gradient_norm_means": {"visual_cortex": 2.5},
                        "mean_counterfactual_credit_weights": {"visual_cortex": 0.0},
                        "mean_module_contribution_share": {"visual_cortex": 0.28},
                        "dominant_module_distribution": {"visual_cortex": 0.28},
                        "dominant_module": "visual_cortex",
                        "mean_dominant_module_share": 0.28,
                        "mean_effective_module_count": 2.8,
                    },
                    "suite": {"night_rest": {"success_rate": 0.55}},
                },
                "counterfactual_credit": {
                    "config": {"architecture": "modular", "credit_strategy": "counterfactual"},
                    "summary": {
                        "scenario_success_rate": 0.65,
                        "episode_success_rate": 0.65,
                        "mean_module_credit_weights": {"visual_cortex": 0.65},
                        "module_gradient_norm_means": {"visual_cortex": 3.2},
                        "mean_counterfactual_credit_weights": {"visual_cortex": 0.65},
                        "mean_module_contribution_share": {"visual_cortex": 0.33},
                        "dominant_module_distribution": {"visual_cortex": 0.33},
                        "dominant_module": "visual_cortex",
                        "mean_dominant_module_share": 0.33,
                        "mean_effective_module_count": 3.1,
                    },
                    "suite": {"night_rest": {"success_rate": 0.65}},
                },
            }
        }

        result = build_credit_assignment_tables(ablations)

        self.assertTrue(result["available"])
        summary_rows = result["strategy_summary"]["rows"]
        self.assertEqual(len(summary_rows), 6)
        a2_local = next(
            row for row in summary_rows
            if row["rung"] == "A2" and row["credit_strategy"] == "local_only"
        )
        self.assertAlmostEqual(a2_local["scenario_success_delta_vs_broadcast"], -0.2)
        module_rows = result["module_credit"]["rows"]
        self.assertTrue(
            any(
                row["rung"] == "A4"
                and row["credit_strategy"] == "counterfactual"
                and row["module"] == "visual_cortex"
                and row["mean_counterfactual_credit_weight"] == 0.65
                for row in module_rows
            )
        )
        self.assertFalse(
            any(
                row["rung"] == "A2"
                and row["credit_strategy"] == "broadcast"
                and row["module"] == "alert_center"
                for row in module_rows
            )
        )
        findings = [item["finding"] for item in result["interpretations"]]
        self.assertIn("Failure by local credit insufficiency", findings)
        self.assertIn("Counterfactual scaling across the ladder", findings)

    def test_build_credit_assignment_tables_uses_without_reflex_support_payloads(
        self,
    ) -> None:
        ablations = {
            "variants": {
                "three_center_modular": {
                    "config": {"architecture": "modular", "credit_strategy": "broadcast"},
                    "summary": {
                        "scenario_success_rate": 0.8,
                        "episode_success_rate": 0.8,
                        "mean_module_credit_weights": {"perception_center": 1.0},
                        "module_gradient_norm_means": {"perception_center": 2.0},
                        "mean_counterfactual_credit_weights": {"perception_center": 0.0},
                        "mean_effective_module_count": 2.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.8}},
                    "without_reflex_support": {
                        "summary": {
                            "scenario_success_rate": 0.55,
                            "episode_success_rate": 0.55,
                            "mean_module_credit_weights": {"perception_center": 0.75},
                            "module_gradient_norm_means": {"perception_center": 1.4},
                            "mean_counterfactual_credit_weights": {"perception_center": 0.0},
                            "mean_effective_module_count": 1.7,
                        },
                        "suite": {"night_rest": {"success_rate": 0.55}},
                    },
                },
                "modular_full": {
                    "config": {"architecture": "modular", "credit_strategy": "broadcast"},
                    "summary": {
                        "scenario_success_rate": 0.5,
                        "episode_success_rate": 0.5,
                        "mean_module_credit_weights": {"visual_cortex": 1.0},
                        "module_gradient_norm_means": {"visual_cortex": 3.0},
                        "mean_counterfactual_credit_weights": {"visual_cortex": 0.0},
                        "mean_effective_module_count": 3.0,
                    },
                    "suite": {"night_rest": {"success_rate": 0.5}},
                },
            }
        }

        result = build_credit_assignment_tables(ablations)

        a2_broadcast = next(
            row
            for row in result["strategy_summary"]["rows"]
            if row["rung"] == "A2" and row["credit_strategy"] == "broadcast"
        )
        a2_scenario = next(
            row
            for row in result["scenario_success"]["rows"]
            if row["rung"] == "A2" and row["credit_strategy"] == "broadcast"
        )
        self.assertEqual(a2_broadcast["scenario_success_rate"], 0.55)
        self.assertEqual(a2_scenario["success_rate"], 0.55)


class CreditMetricsExtractionAndTableTest(unittest.TestCase):
    def _summary(self) -> dict[str, object]:
        return {
            "config": {
                "brain": {
                    "name": "modular_full",
                    "architecture": "modular",
                    "credit_strategy": "broadcast",
                }
            },
            "behavior_evaluation": {
                "ablations": {
                    "variants": {
                        "three_center_modular": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "broadcast",
                            },
                            "summary": {
                                "scenario_success_rate": 0.6,
                                "mean_module_credit_weights": {
                                    "perception_center": 1.0,
                                },
                                "module_gradient_norm_means": {
                                    "perception_center": 2.0,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "perception_center": 0.0,
                                },
                                "mean_effective_module_count": 2.0,
                            },
                        },
                        "three_center_modular_counterfactual": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "counterfactual",
                            },
                            "summary": {
                                "scenario_success_rate": 0.7,
                                "mean_module_credit_weights": {
                                    "perception_center": 0.7,
                                },
                                "module_gradient_norm_means": {
                                    "perception_center": 2.2,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "perception_center": 0.7,
                                },
                                "mean_effective_module_count": 2.1,
                            },
                        },
                        "modular_full": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "broadcast",
                            },
                            "summary": {
                                "scenario_success_rate": 0.5,
                                "mean_module_credit_weights": {
                                    "visual_cortex": 1.0,
                                },
                                "module_gradient_norm_means": {
                                    "visual_cortex": 3.0,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "visual_cortex": 0.0,
                                },
                                "mean_effective_module_count": 3.0,
                            },
                            "without_reflex_support": {
                                "summary": {
                                    "scenario_success_rate": 0.45,
                                    "mean_module_credit_weights": {
                                        "visual_cortex": 0.9,
                                    },
                                    "module_gradient_norm_means": {
                                        "visual_cortex": 2.7,
                                    },
                                    "mean_counterfactual_credit_weights": {
                                        "visual_cortex": 0.0,
                                    },
                                    "mean_effective_module_count": 2.8,
                                }
                            },
                        },
                        "counterfactual_credit": {
                            "config": {
                                "architecture": "modular",
                                "credit_strategy": "counterfactual",
                            },
                            "summary": {
                                "scenario_success_rate": 0.65,
                                "mean_module_credit_weights": {
                                    "visual_cortex": 0.65,
                                },
                                "module_gradient_norm_means": {
                                    "visual_cortex": 3.2,
                                },
                                "mean_counterfactual_credit_weights": {
                                    "visual_cortex": 0.65,
                                },
                                "mean_effective_module_count": 3.1,
                            },
                        },
                    }
                }
            },
        }

    def test_extract_credit_metrics_returns_variant_mapping(self) -> None:
        result = extract_credit_metrics(self._summary(), [])

        self.assertIn("three_center_modular", result)
        self.assertEqual(result["three_center_modular"]["strategy"], "broadcast")
        self.assertEqual(
            result["three_center_modular_counterfactual"]["counterfactual_weights"][
                "perception_center"
            ],
            0.7,
        )
        self.assertEqual(
            result["modular_full"]["gradient_norms"]["visual_cortex"],
            2.7,
        )
        self.assertEqual(
            result["modular_full"]["scenario_success_rate"],
            0.45,
        )

    def test_build_credit_table_emits_rows_and_summary_statistics(self) -> None:
        result = build_credit_table(self._summary())

        self.assertTrue(result["available"])
        table_rows = result["table"]["rows"]
        self.assertTrue(
            any(
                row["variant"] == "counterfactual_credit"
                and row["architecture_rung"] == "A4"
                and row["credit_strategy"] == "counterfactual"
                and row["module_name"] == "visual_cortex"
                and row["counterfactual_weight"] == 0.65
                and row["scenario_success_rate"] == 0.65
                for row in table_rows
            )
        )
        self.assertTrue(
            any(
                row["variant"] == "modular_full"
                and row["module_name"] == "visual_cortex"
                and row["credit_weight"] == 0.9
                and row["scenario_success_rate"] == 0.45
                for row in table_rows
            )
        )
        mean_credit_rows = result["summary_statistics"][
            "mean_credit_per_module_by_strategy"
        ]["rows"]
        self.assertTrue(
            any(
                row["credit_strategy"] == "broadcast"
                and row["module_name"] == "visual_cortex"
                and row["mean_credit_weight"] == 0.9
                for row in mean_credit_rows
            )
        )
        concentration_rows = result["summary_statistics"]["credit_concentration"][
            "rows"
        ]
        self.assertTrue(
            any(
                row["credit_strategy"] == "counterfactual"
                and row["mean_effective_module_count"] == 2.6
                for row in concentration_rows
            )
        )
        self.assertFalse(
            any(
                row["credit_strategy"] == "broadcast"
                and row["module_name"] == "alert_center"
                for row in mean_credit_rows
            )
        )

    def test_build_credit_table_uses_top_level_credit_strategy_fallback(self) -> None:
        result = build_credit_table(
            {
                "config": {
                    "name": "current_run",
                    "credit_strategy": "counterfactual",
                },
                "evaluation": {
                    "scenario_success_rate": 0.4,
                    "mean_module_credit_weights": {"visual_cortex": 0.4},
                    "module_gradient_norm_means": {"visual_cortex": 1.2},
                    "mean_counterfactual_credit_weights": {"visual_cortex": 0.6},
                    "mean_effective_module_count": 1.0,
                },
            }
        )

        self.assertTrue(result["available"])
        self.assertTrue(
            all(
                row["credit_strategy"] == "counterfactual"
                for row in result["table"]["rows"]
            )
        )

    def test_build_credit_table_excludes_gradient_only_modules_from_credit_means(
        self,
    ) -> None:
        result = build_credit_table(
            {
                "config": {
                    "brain": {
                        "name": "modular_full",
                        "architecture": "modular",
                        "credit_strategy": "broadcast",
                    }
                },
                "behavior_evaluation": {
                    "ablations": {
                        "variants": {
                            "modular_full": {
                                "config": {
                                    "architecture": "modular",
                                    "credit_strategy": "broadcast",
                                },
                                "summary": {
                                    "scenario_success_rate": 0.5,
                                    "mean_module_credit_weights": {
                                        "visual_cortex": 1.0,
                                    },
                                    "module_gradient_norm_means": {
                                        "visual_cortex": 3.0,
                                        "motor_cortex": 1.5,
                                    },
                                    "mean_counterfactual_credit_weights": {
                                        "visual_cortex": 0.0,
                                        "motor_cortex": 0.4,
                                    },
                                    "mean_effective_module_count": 2.0,
                                },
                            }
                        }
                    }
                },
            }
        )

        table_rows = result["table"]["rows"]
        self.assertTrue(
            any(
                row["module_name"] == "motor_cortex"
                and row["gradient_norm"] == 1.5
                and row["counterfactual_weight"] == 0.4
                for row in table_rows
            )
        )
        mean_credit_rows = result["summary_statistics"][
            "mean_credit_per_module_by_strategy"
        ]["rows"]
        self.assertFalse(
            any(row["module_name"] == "motor_cortex" for row in mean_credit_rows)
        )


class CreditInterpretationHelpersTest(unittest.TestCase):
    def test_interpret_credit_failure_detects_local_insufficiency(self) -> None:
        result = _interpret_credit_failure(
            {
                "strategy": "local_only",
                "weights": {"perception_center": 0.0, "action_center": 0.0},
                "gradient_norms": {"perception_center": 0.0, "action_center": 0.0},
                "counterfactual_weights": {},
                "scenario_success_rate": 0.2,
            },
            {
                "strategy": "broadcast",
                "weights": {"perception_center": 0.5, "action_center": 0.5},
                "gradient_norms": {"perception_center": 0.4, "action_center": 0.4},
                "counterfactual_weights": {},
                "scenario_success_rate": 0.6,
            },
        )

        self.assertTrue(
            any(
                finding["pattern"] == "insufficient local credit"
                for finding in result["findings"]
            )
        )

    def test_compare_credit_across_architectures_detects_scaling_patterns(self) -> None:
        result = _compare_credit_across_architectures(
            {
                "broadcast": {"scenario_success_rate": 0.6},
                "local_only": {"scenario_success_rate": 0.3},
                "counterfactual": {"scenario_success_rate": 0.72},
            },
            {
                "broadcast": {"scenario_success_rate": 0.5},
                "local_only": {"scenario_success_rate": 0.05},
                "counterfactual": {"scenario_success_rate": 0.82},
            },
        )

        patterns = {finding["pattern"] for finding in result["findings"]}
        self.assertIn("local_only differential failure", patterns)
        self.assertIn("counterfactual benefit scales with module count", patterns)

    def test_compare_credit_across_architectures_rejects_ambiguous_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "single-rung strategy mapping"):
            _compare_credit_across_architectures(
                {
                    "broadcast": {"scenario_success_rate": 0.6},
                    "local_only": {"scenario_success_rate": 0.3},
                }
            )


class BuildRewardComponentRowsTest(unittest.TestCase):
    def _make_scenario_success(self, scenarios: list[dict]) -> dict:
        return {"scenarios": scenarios, "source": "test"}

    def test_empty_inputs_returns_empty_list(self) -> None:
        result = build_reward_component_rows({}, self._make_scenario_success([]), [])
        self.assertEqual(result, [])

    def test_summary_training_components_extracted(self) -> None:
        summary = {
            "training": {
                "mean_reward_components": {
                    "food_progress": 0.5,
                    "feeding": 0.3,
                }
            }
        }
        rows = build_reward_component_rows(summary, self._make_scenario_success([]), [])
        sources = [r["source"] for r in rows]
        self.assertTrue(all(s == "summary" for s in sources))
        components = [r["component"] for r in rows]
        self.assertIn("food_progress", components)
        self.assertIn("feeding", components)

    def test_summary_evaluation_components_extracted(self) -> None:
        summary = {
            "evaluation": {
                "mean_reward_components": {"reward_x": 1.0}
            }
        }
        rows = build_reward_component_rows(summary, self._make_scenario_success([]), [])
        self.assertTrue(any(r["component"] == "reward_x" for r in rows))

    def test_scenario_legacy_metrics_components_extracted(self) -> None:
        scenario_success = self._make_scenario_success([
            {
                "scenario": "night_rest",
                "legacy_metrics": {
                    "mean_reward_components": {
                        "shelter_bonus": 0.2,
                    }
                },
            }
        ])
        rows = build_reward_component_rows({}, scenario_success, [])
        self.assertTrue(any(r["component"] == "shelter_bonus" for r in rows))

    def test_trace_components_aggregated(self) -> None:
        trace = [
            {"reward_components": {"food_progress": 0.1, "feeding": 0.2}},
            {"reward_components": {"food_progress": 0.3}},
        ]
        rows = build_reward_component_rows({}, self._make_scenario_success([]), trace)
        trace_rows = [r for r in rows if r["source"] == "trace"]
        self.assertTrue(any(r["component"] == "food_progress" for r in trace_rows))
        food_row = next(r for r in trace_rows if r["component"] == "food_progress")
        self.assertAlmostEqual(food_row["value"], 0.4, places=5)

    def test_trace_rows_without_reward_components_skipped(self) -> None:
        trace = [{"step": 1}, {"step": 2}]
        rows = build_reward_component_rows({}, self._make_scenario_success([]), trace)
        self.assertEqual(rows, [])

    def test_value_rounded_to_six_decimals(self) -> None:
        summary = {
            "training": {
                "mean_reward_components": {"c": 1.0 / 3.0}
            }
        }
        rows = build_reward_component_rows(summary, self._make_scenario_success([]), [])
        row = next(r for r in rows if r["component"] == "c")
        self.assertEqual(row["value"], round(1.0 / 3.0, 6))


class BuildDiagnosticsTest(unittest.TestCase):
    def _make_empty_scenario_success(self) -> dict:
        return {"scenarios": [], "source": "none"}

    def _make_empty_ablations(self) -> dict:
        return {"variants": {}, "source": "none"}

    def _make_empty_reflex_frequency(self) -> dict:
        return {"modules": []}

    def test_returns_list(self) -> None:
        result = build_diagnostics(
            {},
            self._make_empty_scenario_success(),
            self._make_empty_ablations(),
            self._make_empty_reflex_frequency(),
        )
        self.assertIsInstance(result, list)

    def test_includes_evaluation_metrics_when_present(self) -> None:
        summary = {
            "evaluation": {
                "mean_reward": 2.5,
                "mean_food_distance_delta": 0.1,
                "mean_shelter_distance_delta": 0.2,
                "mean_predator_mode_transitions": 3.0,
                "dominant_predator_state": "fleeing",
            }
        }
        result = build_diagnostics(
            summary,
            self._make_empty_scenario_success(),
            self._make_empty_ablations(),
            self._make_empty_reflex_frequency(),
        )
        labels = [r["label"] for r in result]
        self.assertIn("Evaluation mean reward", labels)
        self.assertIn("Evaluation mean food distance delta", labels)
        self.assertIn("Evaluation mean shelter distance delta", labels)
        self.assertIn("Evaluation predator mode transitions", labels)
        self.assertIn("Evaluation dominant predator state", labels)

    def test_evaluation_mean_reward_value(self) -> None:
        summary = {"evaluation": {"mean_reward": 3.14}}
        result = build_diagnostics(
            summary,
            self._make_empty_scenario_success(),
            self._make_empty_ablations(),
            self._make_empty_reflex_frequency(),
        )
        reward_row = next(r for r in result if r["label"] == "Evaluation mean reward")
        self.assertAlmostEqual(reward_row["value"], 3.14, places=5)

    def test_weakest_scenario_appended(self) -> None:
        scenario_success = {
            "scenarios": [
                {"scenario": "hard", "success_rate": 0.2},
                {"scenario": "easy", "success_rate": 0.9},
            ]
        }
        result = build_diagnostics(
            {},
            scenario_success,
            self._make_empty_ablations(),
            self._make_empty_reflex_frequency(),
        )
        labels = [r["label"] for r in result]
        self.assertIn("Weakest scenario", labels)
        weakest = next(r for r in result if r["label"] == "Weakest scenario")
        self.assertIn("hard", str(weakest["value"]))

    def test_best_ablation_variant_appended(self) -> None:
        ablations = {
            "variants": {
                "modular_full": {
                    "summary": {"scenario_success_rate": 0.9}
                },
                "monolithic": {
                    "summary": {"scenario_success_rate": 0.4}
                },
            }
        }
        result = build_diagnostics(
            {},
            self._make_empty_scenario_success(),
            ablations,
            self._make_empty_reflex_frequency(),
        )
        labels = [r["label"] for r in result]
        self.assertIn("Best ablation variant", labels)
        best = next(r for r in result if r["label"] == "Best ablation variant")
        self.assertIn("modular_full", str(best["value"]))

    def test_most_frequent_reflex_source_appended(self) -> None:
        reflex_frequency = {
            "modules": [
                {"module": "visual_cortex", "reflex_events": 100},
                {"module": "sensory_cortex", "reflex_events": 50},
            ]
        }
        result = build_diagnostics(
            {},
            self._make_empty_scenario_success(),
            self._make_empty_ablations(),
            reflex_frequency,
        )
        labels = [r["label"] for r in result]
        self.assertIn("Most frequent reflex source", labels)
        most_freq = next(r for r in result if r["label"] == "Most frequent reflex source")
        self.assertIn("visual_cortex", str(most_freq["value"]))

    def test_reflex_dependence_indicators_added_when_available(self) -> None:
        summary = {
            "evaluation_with_reflex_support": {
                "summary": {
                    "mean_final_reflex_override_rate": 0.15,
                    "mean_reflex_dominance": 0.30,
                }
            }
        }
        result = build_diagnostics(
            summary,
            self._make_empty_scenario_success(),
            self._make_empty_ablations(),
            self._make_empty_reflex_frequency(),
        )
        labels = [r["label"] for r in result]
        self.assertIn("Reflex Dependence: override rate", labels)
        self.assertIn("Reflex Dependence: dominance", labels)

    def test_failure_indicator_has_status_and_threshold_fields(self) -> None:
        summary = {
            "evaluation_with_reflex_support": {
                "summary": {
                    "mean_final_reflex_override_rate": 0.5,
                    "mean_reflex_dominance": 0.5,
                }
            }
        }
        result = build_diagnostics(
            summary,
            self._make_empty_scenario_success(),
            self._make_empty_ablations(),
            self._make_empty_reflex_frequency(),
        )
        indicator_rows = [r for r in result if r.get("failure_indicator")]
        self.assertTrue(indicator_rows)
        for row in indicator_rows:
            self.assertIn("status", row)
            self.assertIn("warning_threshold", row)
            self.assertTrue(row["failure_indicator"])

    def test_empty_summary_produces_zero_valued_evaluation_rows(self) -> None:
        # Even an empty summary produces evaluation rows (with zero values) because
        # summary.get("evaluation", {}) returns {} which IS a Mapping.
        result = build_diagnostics(
            {},
            self._make_empty_scenario_success(),
            self._make_empty_ablations(),
            self._make_empty_reflex_frequency(),
        )
        labels = [r["label"] for r in result]
        self.assertIn("Evaluation mean reward", labels)
        reward_row = next(r for r in result if r["label"] == "Evaluation mean reward")
        self.assertAlmostEqual(reward_row["value"], 0.0)


class ClaimUncertaintyForConditionTest(unittest.TestCase):
    def test_condition_key_present_in_mapping_returns_nested(self) -> None:
        nested = {"ci_lower": 0.4, "ci_upper": 0.8}
        uncertainty = {"condition_a": nested}
        result = _claim_uncertainty_for_condition(uncertainty, "condition_a")
        self.assertIs(result, nested)

    def test_condition_key_absent_falls_back_to_top_level_if_has_ci_lower(self) -> None:
        uncertainty = {"ci_lower": 0.1, "ci_upper": 0.5}
        result = _claim_uncertainty_for_condition(uncertainty, "nonexistent")
        self.assertIs(result, uncertainty)

    def test_condition_key_absent_falls_back_to_top_level_if_has_mean(self) -> None:
        uncertainty = {"mean": 0.7}
        result = _claim_uncertainty_for_condition(uncertainty, "nonexistent")
        self.assertIs(result, uncertainty)

    def test_condition_key_absent_no_ci_lower_or_mean_returns_empty(self) -> None:
        uncertainty = {"other_key": 0.5}
        result = _claim_uncertainty_for_condition(uncertainty, "nonexistent")
        self.assertEqual(dict(result), {})

    def test_non_mapping_uncertainty_returns_empty(self) -> None:
        result = _claim_uncertainty_for_condition(None, "cond")
        self.assertEqual(dict(result), {})

    def test_nested_value_not_mapping_falls_back_correctly(self) -> None:
        uncertainty = {"condition_a": "not_a_mapping", "ci_lower": 0.3}
        # condition_a is not a Mapping so falls through
        # But: it checks isinstance(nested, Mapping) first, if True returns nested
        # Since "not_a_mapping" is a str and has nested = uncertainty.get(condition_a) = "not_a_mapping"
        # That is not a Mapping, so it falls back to check for ci_lower/mean on top-level
        result = _claim_uncertainty_for_condition(uncertainty, "condition_a")
        # Expected: the nested value "not_a_mapping" is not returned (not a Mapping)
        # The function returns payload if "ci_lower" or "mean" is in payload
        # payload = _mapping_or_empty(uncertainty) = uncertainty itself
        # nested = payload.get("condition_a") = "not_a_mapping"
        # isinstance(nested, Mapping) => False, so falls to:
        # return payload if "ci_lower" in payload or "mean" in payload else {}
        self.assertIs(result, uncertainty)
