from __future__ import annotations

import unittest

from spider_cortex_sim.offline_analysis.tables import (
    build_credit_assignment_tables,
    build_credit_table,
    build_diagnostics,
    build_reward_component_rows,
)
from spider_cortex_sim.offline_analysis.table_builders import diagnostics as diagnostics_builder


class CreditAssignmentReviewRegressionTest(unittest.TestCase):
    def test_missing_counterfactual_delta_is_not_treated_as_zero(self) -> None:
        ablations = {
            "variants": {
                "three_center_modular": {
                    "config": {"architecture": "modular", "credit_strategy": "broadcast"},
                    "summary": {"scenario_success_rate": 0.6},
                    "suite": {"night_rest": {"success_rate": 0.6}},
                },
                "three_center_modular_counterfactual": {
                    "config": {"architecture": "modular", "credit_strategy": "counterfactual"},
                    "summary": {},
                    "suite": {"night_rest": {"success_rate": 0.6}},
                },
                "modular_full": {
                    "config": {"architecture": "modular", "credit_strategy": "broadcast"},
                    "summary": {"scenario_success_rate": 0.7},
                    "suite": {"night_rest": {"success_rate": 0.7}},
                },
                "counterfactual_credit": {
                    "config": {"architecture": "modular", "credit_strategy": "counterfactual"},
                    "summary": {},
                    "suite": {"night_rest": {"success_rate": 0.7}},
                },
            }
        }

        result = build_credit_assignment_tables(ablations)

        counterfactual_rows = [
            row
            for row in result["strategy_summary"]["rows"]
            if row["credit_strategy"] == "counterfactual"
        ]
        self.assertTrue(counterfactual_rows)
        self.assertTrue(
            all(
                row["scenario_success_delta_vs_broadcast"] is None
                for row in counterfactual_rows
            )
        )
        findings = [item["finding"] for item in result["interpretations"]]
        self.assertNotIn("Counterfactual scaling across the ladder", findings)

    def test_missing_success_rates_remain_none_in_credit_assignment_rows(self) -> None:
        ablations = {
            "variants": {
                "three_center_modular": {
                    "config": {"architecture": "modular", "credit_strategy": "broadcast"},
                    "summary": {"mean_module_credit_weights": {"visual_cortex": 0.5}},
                    "suite": {},
                }
            }
        }

        result = build_credit_assignment_tables(ablations)

        summary_row = result["strategy_summary"]["rows"][0]
        self.assertIsNone(summary_row["scenario_success_rate"])
        self.assertIsNone(summary_row["episode_success_rate"])
        module_row = result["module_credit"]["rows"][0]
        self.assertIsNone(module_row["scenario_success_rate"])

    def test_credit_table_normalizes_raw_strategy_strings(self) -> None:
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "variants": {
                        "local_credit_only": {
                            "config": {"credit_strategy": "legacy_local"},
                            "summary": {
                                "mean_module_credit_weights": {"visual_cortex": 0.5}
                            },
                        }
                    }
                }
            }
        }

        result = build_credit_table(summary)

        self.assertEqual(result["table"]["rows"][0]["credit_strategy"], "local_only")

    def test_credit_table_reads_normalized_variant_config(self) -> None:
        summary = {
            "behavior_evaluation": {
                "ablations": {
                    "variants": {
                        "custom_variant": {
                            "config": {"credit_strategy": "broadcast"},
                            "summary": {},
                            "without_reflex_support": {
                                "config": {"credit_strategy": "counterfactual"},
                                "summary": {
                                    "mean_module_credit_weights": {
                                        "visual_cortex": 0.5
                                    },
                                    "mean_counterfactual_credit_weights": {
                                        "visual_cortex": 0.4
                                    },
                                },
                            },
                        }
                    }
                }
            }
        }

        result = build_credit_table(summary)

        self.assertEqual(
            result["table"]["rows"][0]["credit_strategy"],
            "counterfactual",
        )

    def test_scenario_check_falsey_text_values_are_preserved(self) -> None:
        from spider_cortex_sim.offline_analysis.tables import build_scenario_checks_rows

        rows = build_scenario_checks_rows(
            {
                "source": 0,
                "scenarios": [
                    {
                        "scenario": 0,
                        "checks": {
                            "check": {
                                "pass_rate": 1.0,
                                "mean_value": 0.0,
                                "expected": False,
                                "description": 0,
                            }
                        },
                    }
                ],
            }
        )

        self.assertEqual(rows[0]["scenario"], "0")
        self.assertEqual(rows[0]["expected"], "False")
        self.assertEqual(rows[0]["description"], "0")
        self.assertEqual(rows[0]["source"], "0")


class RewardComponentReviewRegressionTest(unittest.TestCase):
    def test_non_finite_summary_components_are_skipped(self) -> None:
        summary = {
            "training": {
                "mean_reward_components": {
                    "bad": float("nan"),
                    "good": 0.5,
                }
            }
        }
        rows = build_reward_component_rows(summary, {"scenarios": []}, [])
        components = {row["component"] for row in rows}
        self.assertNotIn("bad", components)
        self.assertIn("good", components)

    def test_non_finite_trace_components_are_skipped(self) -> None:
        trace = [
            {"reward_components": {"bad": float("inf"), "good": 0.25}},
            {"reward_components": {"good": 0.25}},
        ]
        rows = build_reward_component_rows({}, {"scenarios": []}, trace)
        components = {row["component"] for row in rows}
        self.assertNotIn("bad", components)
        good_row = next(row for row in rows if row["component"] == "good")
        self.assertEqual(good_row["value"], 0.5)


class DiagnosticsReviewRegressionTest(unittest.TestCase):
    def test_reflex_dependence_top_level_shape_is_consumed(self) -> None:
        reflex_data = {
            "available": True,
            "source": "summary.evaluation",
            "override_rate": 0.2,
            "override_warning_threshold": 0.1,
            "override_status": "warning",
            "dominance_rate": 0.3,
            "dominance_warning_threshold": 0.25,
            "dominance_status": "warning",
        }
        original = diagnostics_builder.build_reflex_dependence_indicators
        diagnostics_builder.build_reflex_dependence_indicators = (
            lambda _summary, _reflex_frequency: reflex_data
        )
        try:
            rows = build_diagnostics({}, {"scenarios": []}, {"variants": {}}, {})
        finally:
            diagnostics_builder.build_reflex_dependence_indicators = original

        diagnostics = {row["label"]: row for row in rows}
        self.assertEqual(
            diagnostics["Reflex Dependence: override rate"]["status"],
            "warning",
        )
        self.assertEqual(
            diagnostics["Reflex Dependence: dominance"]["warning_threshold"],
            0.25,
        )

    def test_capacity_diagnostic_uses_capacity_summary_keys(self) -> None:
        rows = build_diagnostics(
            {},
            {"scenarios": []},
            {
                "variants": {
                    "single": {
                        "parameter_counts": {"total": 100},
                        "summary": {"scenario_success_rate": 0.5},
                    }
                }
            },
            {},
        )

        diagnostics = {row["label"]: row for row in rows}
        self.assertEqual(
            diagnostics["Architecture capacity match"]["value"],
            "single architecture",
        )


if __name__ == "__main__":
    unittest.main()
