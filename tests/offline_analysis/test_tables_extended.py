"""Tests for tables.py helpers not covered by test_tables.py.

Covers: build_scenario_checks_rows, build_reward_component_rows,
build_diagnostics, and _claim_uncertainty_for_condition.
"""
from __future__ import annotations

import unittest

from spider_cortex_sim.offline_analysis.tables import (
    _claim_uncertainty_for_condition,
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
        self.assertEqual(rows[0]["mean_value"], 0.5)

    def test_non_mapping_checks_field_skipped(self) -> None:
        scenario_success = self._make_scenario_success([
            {"scenario": "s", "checks": "not_a_mapping"}
        ])
        rows = build_scenario_checks_rows(scenario_success)
        self.assertEqual(rows, [])


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
