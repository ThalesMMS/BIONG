from __future__ import annotations

import unittest

from spider_cortex_sim.offline_analysis.tables import (
    build_aggregate_benchmark_tables,
    build_claim_test_tables,
    build_diagnostics,
    build_effect_size_tables,
)

from .conftest import (
    assert_uncertainty_fields,
    build_uncertainty_summary,
    uncertainty_condition,
    uncertainty_payload,
)

class OfflineAnalysisUncertaintyTablesTest(unittest.TestCase):
    def test_build_aggregate_benchmark_tables_includes_cis(self) -> None:
        tables = build_aggregate_benchmark_tables(build_uncertainty_summary())

        primary_rows = tables["primary_benchmark"]["rows"]
        scenario_rows = tables["per_scenario_success_rates"]["rows"]
        learning_rows = tables["learning_evidence_deltas"]["rows"]

        primary_row = next(
            row for row in primary_rows if row["metric"] == "scenario_success_rate"
        )
        scenario_row = next(
            row for row in scenario_rows if row["scenario"] == "night_rest"
        )
        learning_row = next(
            row
            for row in learning_rows
            if row["metric"] == "scenario_success_rate_delta"
            and row["comparison"] == "random_init"
        )
        self.assertEqual(primary_row["ci_lower"], 0.7)
        assert_uncertainty_fields(self, primary_row)
        assert_uncertainty_fields(self, scenario_row)
        assert_uncertainty_fields(self, learning_row)
        for row in learning_rows:
            assert_uncertainty_fields(self, row)
    def test_build_aggregate_benchmark_tables_uses_primary_benchmark_guard(self) -> None:
        tables = build_aggregate_benchmark_tables(
            {
                "evaluation": {
                    "scenario_success_rate": 0.9,
                    "eval_reflex_scale": 1.0,
                },
                "behavior_evaluation": {
                    "summary": {"scenario_success_rate": 0.8},
                    "suite": {"night_rest": {"success_rate": 0.8}},
                },
            }
        )

        self.assertEqual(tables["primary_benchmark"]["rows"], [])
        self.assertEqual(tables["per_scenario_success_rates"]["rows"], [])
    def test_build_aggregate_benchmark_tables_uses_zero_reflex_payload_ci(self) -> None:
        tables = build_aggregate_benchmark_tables(
            {
                "behavior_evaluation": {
                    "summary": {
                        "scenario_success_rate": 0.6,
                        "eval_reflex_scale": 0.0,
                    },
                    "suite": {
                        "night_rest": {
                            "success_rate": 0.6,
                            "uncertainty": {
                                "success_rate": uncertainty_payload(
                                    0.6,
                                    0.5,
                                    0.7,
                                    [0.5, 0.7],
                                )
                            },
                        }
                    },
                    "uncertainty": {
                        "scenario_success_rate": uncertainty_payload(
                            0.6,
                            0.5,
                            0.7,
                            [0.5, 0.7],
                        )
                    },
                }
            }
        )

        primary_row = tables["primary_benchmark"]["rows"][0]
        scenario_row = tables["per_scenario_success_rates"]["rows"][0]
        self.assertEqual(primary_row["source"], "summary.behavior_evaluation.summary")
        self.assertEqual(primary_row["ci_lower"], 0.5)
        self.assertEqual(primary_row["ci_upper"], 0.7)
        self.assertEqual(scenario_row["source"], "summary.behavior_evaluation.suite.night_rest")
        self.assertEqual(scenario_row["ci_lower"], 0.5)
        self.assertEqual(scenario_row["ci_upper"], 0.7)
    def test_build_claim_test_tables_includes_uncertainty_roles(self) -> None:
        tables = build_claim_test_tables(build_uncertainty_summary())
        rows = tables["claim_results"]["rows"]
        roles = {row["role"] for row in rows}

        self.assertIn("reference", roles)
        self.assertIn("comparison", roles)
        self.assertIn("delta", roles)
        self.assertIn("effect_size", roles)
        effect_rows = [row for row in rows if row["role"] == "effect_size"]
        for role in ("reference", "comparison", "delta", "effect_size"):
            row = next(item for item in rows if item["role"] == role)
            assert_uncertainty_fields(self, row)
        self.assertEqual(effect_rows[0]["cohens_d"], 4.0)
        self.assertEqual(effect_rows[0]["effect_magnitude"], "large")
    def test_build_claim_test_tables_uses_metric_specific_reference_uncertainty(self) -> None:
        summary = build_uncertainty_summary()
        claim = summary["behavior_evaluation"]["claim_tests"]["claims"][
            "learning_without_privileged_signals"
        ]
        claim["reference_value"] = {
            "scenario_success_rate": 0.3,
            "episode_success_rate": 0.4,
        }
        claim["reference_uncertainty"] = {
            "scenario_success_rate": uncertainty_payload(0.3, 0.2, 0.4, [0.2, 0.4]),
            "episode_success_rate": uncertainty_payload(0.4, 0.3, 0.5, [0.3, 0.5]),
        }

        tables = build_claim_test_tables(summary)
        rows = tables["claim_results"]["rows"]
        episode_row = next(
            row
            for row in rows
            if row["role"] == "reference"
            and row["metric"] == "episode_success_rate"
        )

        self.assertEqual(episode_row["ci_lower"], 0.3)
        self.assertEqual(episode_row["ci_upper"], 0.5)
    def test_build_effect_size_tables_includes_main_baselines(self) -> None:
        tables = build_effect_size_tables(build_uncertainty_summary())
        rows = tables["effect_sizes"]["rows"]
        pairs = {(row["domain"], row["baseline"]) for row in rows}

        self.assertIn(("learning_evidence", "random_init"), pairs)
        self.assertIn(("learning_evidence", "reflex_only"), pairs)
        self.assertIn(("ablation", "modular_full"), pairs)
        self.assertIn(("ablation", "monolithic_policy"), pairs)
        for baseline in (
            "random_init",
            "reflex_only",
            "modular_full",
            "monolithic_policy",
        ):
            row = next(item for item in rows if item["baseline"] == baseline)
            for key in (
                "raw_delta",
                "cohens_d",
                "magnitude_label",
                "ci_lower",
                "ci_upper",
                "std_error",
                "n_seeds",
                "confidence_level",
            ):
                self.assertIn(key, row)
        ablation_rows = [row for row in rows if row["domain"] == "ablation"]
        self.assertTrue(ablation_rows)
        for row in ablation_rows:
            self.assertEqual(row["value"], row["cohens_d"])
            self.assertIsInstance(row["ci_lower"], float)
            self.assertIsInstance(row["ci_upper"], float)
            self.assertEqual(row["effect_size_ci_lower"], row["ci_lower"])
            self.assertEqual(row["effect_size_ci_upper"], row["ci_upper"])
            self.assertIsInstance(row["delta_ci_lower"], float)
            self.assertIsInstance(row["delta_ci_upper"], float)
            self.assertEqual(row["n_seeds"], 2)
            self.assertEqual(row["effect_size_n_seeds"], 2)
            self.assertEqual(row["delta_n_seeds"], 2)


class OfflineAnalysisDiagnosticsTablesTest(unittest.TestCase):
    def test_build_diagnostics_ignores_non_mapping_scenarios(self) -> None:
        diagnostics = build_diagnostics(
            {},
            {"scenarios": ["invalid"]},
            {},
            {},
        )

        labels = {row["label"] for row in diagnostics}
        self.assertNotIn("Weakest scenario", labels)

    def test_build_diagnostics_ignores_non_mapping_reflex_modules(self) -> None:
        diagnostics = build_diagnostics(
            {},
            {"scenarios": []},
            {},
            {"modules": ["invalid"]},
        )

        labels = {row["label"] for row in diagnostics}
        self.assertNotIn("Most frequent reflex source", labels)

    def test_uncertainty_condition_rejects_empty_values(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "uncertainty_condition requires at least one value",
        ):
            uncertainty_condition("empty", [])
