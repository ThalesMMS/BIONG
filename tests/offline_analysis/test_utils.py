from __future__ import annotations

import unittest

from spider_cortex_sim.offline_analysis.constants import DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD
from spider_cortex_sim.offline_analysis.extractors import (
    _component_disposition_rows,
    _component_disposition_summary,
    _normalize_behavior_survival,
    _profile_disposition_weights,
)
from spider_cortex_sim.offline_analysis.utils import _mapping_or_empty

class ShapingHelperFunctionsTest(unittest.TestCase):
    """Tests for private helper functions added in this PR."""

    # --- _mapping_or_empty ---

    def test_mapping_or_empty_returns_dict_unchanged(self) -> None:
        d = {"a": 1, "b": 2}
        result = _mapping_or_empty(d)
        self.assertIs(result, d)

    def test_mapping_or_empty_returns_empty_for_none(self) -> None:
        result = _mapping_or_empty(None)
        self.assertEqual(dict(result), {})

    def test_mapping_or_empty_returns_empty_for_list(self) -> None:
        result = _mapping_or_empty([1, 2, 3])
        self.assertEqual(dict(result), {})

    def test_mapping_or_empty_returns_empty_for_string(self) -> None:
        result = _mapping_or_empty("not a mapping")
        self.assertEqual(dict(result), {})

    def test_mapping_or_empty_returns_empty_for_int(self) -> None:
        result = _mapping_or_empty(42)
        self.assertEqual(dict(result), {})

    def test_mapping_or_empty_accepts_empty_dict(self) -> None:
        d: dict = {}
        result = _mapping_or_empty(d)
        self.assertIs(result, d)

    # --- _profile_disposition_weights ---

    def test_profile_disposition_weights_empty_input(self) -> None:
        result = _profile_disposition_weights({})
        self.assertEqual(result, {})

    def test_profile_disposition_weights_skips_non_mapping_profiles(self) -> None:
        profiles = {
            "good": {
                "disposition_summary": {
                    "removed": {"total_weight_proxy": 1.5},
                }
            },
            "bad": "not a mapping",
        }
        result = _profile_disposition_weights(profiles)
        self.assertIn("good", result)
        self.assertNotIn("bad", result)

    def test_profile_disposition_weights_skips_profile_with_non_mapping_disposition_summary(self) -> None:
        # When disposition_summary is not a Mapping, the entire profile is skipped
        profiles = {
            "profile_a": {
                "disposition_summary": "not_a_dict",
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertNotIn("profile_a", result)

    def test_profile_disposition_weights_extracts_weight_proxies(self) -> None:
        profiles = {
            "classic": {
                "disposition_summary": {
                    "removed": {"total_weight_proxy": 1.234567890},
                    "defended": {"total_weight_proxy": 2.5},
                }
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertIn("classic", result)
        self.assertAlmostEqual(result["classic"]["removed"], 1.234568, places=5)
        self.assertAlmostEqual(result["classic"]["defended"], 2.5)

    def test_profile_disposition_weights_rounds_to_six_decimals(self) -> None:
        profiles = {
            "p": {
                "disposition_summary": {
                    "removed": {"total_weight_proxy": 1.1234567},
                }
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertEqual(result["p"]["removed"], round(1.1234567, 6))

    def test_profile_disposition_weights_non_mapping_disposition_entry_yields_zero(self) -> None:
        profiles = {
            "p": {
                "disposition_summary": {
                    "removed": "not_a_dict",
                }
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertAlmostEqual(result["p"]["removed"], 0.0)

    def test_profile_disposition_weights_missing_total_weight_proxy_yields_zero(self) -> None:
        profiles = {
            "p": {
                "disposition_summary": {
                    "removed": {"some_other_key": 99.0},
                }
            }
        }
        result = _profile_disposition_weights(profiles)
        self.assertAlmostEqual(result["p"]["removed"], 0.0)

    def test_profile_disposition_weights_sorted_profile_keys(self) -> None:
        profiles = {
            "zzz": {"disposition_summary": {"removed": {"total_weight_proxy": 1.0}}},
            "aaa": {"disposition_summary": {"removed": {"total_weight_proxy": 2.0}}},
        }
        result = _profile_disposition_weights(profiles)
        self.assertEqual(list(result.keys()), ["aaa", "zzz"])

    # --- _component_disposition_rows ---

    def test_component_disposition_rows_empty_input(self) -> None:
        result = _component_disposition_rows({})
        self.assertEqual(result, [])

    def test_component_disposition_rows_skips_non_mapping_metadata(self) -> None:
        components = {
            "food_progress": {
                "category": "progress",
                "shaping_risk": "high",
                "shaping_disposition": "removed",
                "disposition_rationale": "Test",
            },
            "bad_entry": "not a mapping",
        }
        result = _component_disposition_rows(components)
        names = [row["component"] for row in result]
        self.assertIn("food_progress", names)
        self.assertNotIn("bad_entry", names)

    def test_component_disposition_rows_row_has_required_keys(self) -> None:
        components = {
            "food_progress": {
                "category": "progress",
                "shaping_risk": "high",
                "shaping_disposition": "removed",
                "disposition_rationale": "Some rationale",
            }
        }
        result = _component_disposition_rows(components)
        self.assertEqual(len(result), 1)
        row = result[0]
        for key in ("component", "category", "risk", "disposition", "rationale"):
            self.assertIn(key, row)

    def test_component_disposition_rows_coerces_fields_to_string(self) -> None:
        components = {
            "test_comp": {
                "category": None,
                "shaping_risk": None,
                "shaping_disposition": None,
                "disposition_rationale": None,
            }
        }
        result = _component_disposition_rows(components)
        self.assertEqual(len(result), 1)
        row = result[0]
        # None values coerced to empty string via `str(None or "")`
        self.assertIsInstance(row["category"], str)
        self.assertIsInstance(row["risk"], str)
        self.assertIsInstance(row["disposition"], str)
        self.assertIsInstance(row["rationale"], str)

    def test_component_disposition_rows_is_sorted_by_component_name(self) -> None:
        components = {
            "zzz_component": {"category": "event", "shaping_risk": "low", "shaping_disposition": "outcome_signal", "disposition_rationale": ""},
            "aaa_component": {"category": "progress", "shaping_risk": "high", "shaping_disposition": "removed", "disposition_rationale": "test"},
        }
        result = _component_disposition_rows(components)
        names = [row["component"] for row in result]
        self.assertEqual(names, sorted(names))

    # --- _component_disposition_summary ---

    def test_component_disposition_summary_empty_input(self) -> None:
        result = _component_disposition_summary([])
        self.assertEqual(result, {})

    def test_component_disposition_summary_groups_by_disposition(self) -> None:
        rows = [
            {"component": "food_progress", "disposition": "removed"},
            {"component": "shelter_progress", "disposition": "removed"},
            {"component": "feeding", "disposition": "outcome_signal"},
        ]
        result = _component_disposition_summary(rows)
        self.assertIn("removed", result)
        self.assertIn("outcome_signal", result)
        self.assertIn("food_progress", result["removed"]["components"])
        self.assertIn("shelter_progress", result["removed"]["components"])
        self.assertIn("feeding", result["outcome_signal"]["components"])

    def test_component_disposition_summary_skips_rows_with_empty_disposition(self) -> None:
        rows = [
            {"component": "food_progress", "disposition": ""},
            {"component": "feeding", "disposition": "outcome_signal"},
        ]
        result = _component_disposition_summary(rows)
        self.assertNotIn("", result)
        self.assertIn("outcome_signal", result)

    def test_component_disposition_summary_skips_rows_with_empty_component(self) -> None:
        rows = [
            {"component": "", "disposition": "removed"},
            {"component": "feeding", "disposition": "outcome_signal"},
        ]
        result = _component_disposition_summary(rows)
        self.assertEqual(result["outcome_signal"]["component_count"], 1)

    def test_component_disposition_summary_components_are_sorted(self) -> None:
        rows = [
            {"component": "zzz", "disposition": "removed"},
            {"component": "aaa", "disposition": "removed"},
        ]
        result = _component_disposition_summary(rows)
        self.assertEqual(result["removed"]["components"], ["aaa", "zzz"])

    def test_component_disposition_summary_component_count_matches_list(self) -> None:
        rows = [
            {"component": "food_progress", "disposition": "removed"},
            {"component": "shelter_progress", "disposition": "removed"},
            {"component": "feeding", "disposition": "outcome_signal"},
        ]
        result = _component_disposition_summary(rows)
        self.assertEqual(result["removed"]["component_count"], 2)
        self.assertEqual(result["outcome_signal"]["component_count"], 1)

    # --- _normalize_behavior_survival ---

    def test_normalize_behavior_survival_empty_input_returns_unavailable(self) -> None:
        result = _normalize_behavior_survival({})
        self.assertFalse(result["available"])
        self.assertEqual(result["scenario_count"], 0)
        self.assertEqual(result["surviving_scenario_count"], 0)
        self.assertAlmostEqual(result["survival_rate"], 0.0)
        self.assertEqual(result["scenarios"], [])

    def test_normalize_behavior_survival_uses_default_threshold_when_missing(self) -> None:
        result = _normalize_behavior_survival({})
        self.assertAlmostEqual(
            result["survival_threshold"],
            DEFAULT_MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        )

    def test_normalize_behavior_survival_respects_provided_threshold(self) -> None:
        result = _normalize_behavior_survival({"survival_threshold": 0.75})
        self.assertAlmostEqual(result["survival_threshold"], 0.75)

    def test_normalize_behavior_survival_scenarios_as_mapping(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {
                    "austere_success_rate": 1.0,
                    "survives": True,
                    "episodes": 3,
                }
            }
        }
        result = _normalize_behavior_survival(payload)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["scenarios"]), 1)
        self.assertEqual(result["scenarios"][0]["scenario"], "night_rest")
        self.assertAlmostEqual(result["scenarios"][0]["austere_success_rate"], 1.0)
        self.assertTrue(result["scenarios"][0]["survives"])
        self.assertEqual(result["scenarios"][0]["episodes"], 3)

    def test_normalize_behavior_survival_accepts_success_rate_alias(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {
                    "success_rate": 0.75,
                    "survives": True,
                    "episodes": 3,
                }
            }
        }
        result = _normalize_behavior_survival(payload)
        self.assertAlmostEqual(result["scenarios"][0]["austere_success_rate"], 0.75)

    def test_normalize_behavior_survival_derives_survives_when_missing(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {"success_rate": 0.75, "episodes": 3},
                "open_field": {"austere_success_rate": 0.0, "episodes": 3},
            }
        }
        result = _normalize_behavior_survival(payload)
        scenarios = {row["scenario"]: row for row in result["scenarios"]}
        self.assertTrue(scenarios["night_rest"]["survives"])
        self.assertFalse(scenarios["open_field"]["survives"])

    def test_normalize_behavior_survival_derives_survives_for_list_payloads(self) -> None:
        payload = {
            "scenarios": [
                {
                    "scenario": "night_rest",
                    "austere_success_rate": 0.5,
                    "episodes": 3,
                }
            ]
        }
        result = _normalize_behavior_survival(payload)
        self.assertTrue(result["scenarios"][0]["survives"])

    def test_normalize_behavior_survival_derives_survives_using_threshold(self) -> None:
        payload = {
            "survival_threshold": 0.5,
            "scenarios": {
                "open_field": {
                    "austere_success_rate": 0.1,
                    "episodes": 3,
                }
            },
        }
        result = _normalize_behavior_survival(payload)
        self.assertFalse(result["scenarios"][0]["survives"])

    def test_normalize_behavior_survival_scenarios_as_list(self) -> None:
        payload = {
            "scenarios": [
                {
                    "scenario": "night_rest",
                    "austere_success_rate": 0.6,
                    "survives": True,
                    "episodes": 2,
                }
            ]
        }
        result = _normalize_behavior_survival(payload)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["scenarios"]), 1)
        self.assertEqual(result["scenarios"][0]["scenario"], "night_rest")

    def test_normalize_behavior_survival_scenarios_as_neither_yields_empty(self) -> None:
        payload = {"scenarios": "not_a_collection"}
        result = _normalize_behavior_survival(payload)
        self.assertEqual(result["scenarios"], [])
        self.assertFalse(result["available"])

    def test_normalize_behavior_survival_explicit_available_false_overrides(self) -> None:
        payload = {
            "available": False,
            "scenarios": {
                "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1}
            },
        }
        result = _normalize_behavior_survival(payload)
        self.assertFalse(result["available"])

    def test_normalize_behavior_survival_uses_provided_scenario_count(self) -> None:
        payload = {
            "scenario_count": 5,
            "surviving_scenario_count": 3,
            "survival_rate": 0.6,
            "scenarios": {
                "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1}
            },
        }
        result = _normalize_behavior_survival(payload)
        self.assertEqual(result["scenario_count"], 5)
        self.assertEqual(result["surviving_scenario_count"], 3)
        self.assertAlmostEqual(result["survival_rate"], 0.6)

    def test_normalize_behavior_survival_computes_rate_when_not_provided(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1},
                "open_field": {"austere_success_rate": 0.0, "survives": False, "episodes": 1},
            }
        }
        result = _normalize_behavior_survival(payload)
        self.assertAlmostEqual(result["survival_rate"], 0.5)
        self.assertEqual(result["surviving_scenario_count"], 1)
        self.assertEqual(result["scenario_count"], 2)

    def test_normalize_behavior_survival_scenarios_sorted_alphabetically(self) -> None:
        payload = {
            "scenarios": {
                "zzz_scenario": {"austere_success_rate": 1.0, "survives": True, "episodes": 1},
                "aaa_scenario": {"austere_success_rate": 0.0, "survives": False, "episodes": 1},
            }
        }
        result = _normalize_behavior_survival(payload)
        names = [row["scenario"] for row in result["scenarios"]]
        self.assertEqual(names, sorted(names))

    def test_normalize_behavior_survival_skips_non_mapping_scenario_entries(self) -> None:
        payload = {
            "scenarios": {
                "night_rest": {"austere_success_rate": 1.0, "survives": True, "episodes": 1},
                "bad_entry": "not a mapping",
            }
        }
        result = _normalize_behavior_survival(payload)
        names = [row["scenario"] for row in result["scenarios"]]
        self.assertIn("night_rest", names)
        self.assertNotIn("bad_entry", names)

    def test_normalize_behavior_survival_minimal_profile_is_captured(self) -> None:
        payload = {
            "minimal_profile": "austere",
            "scenarios": {
                "s": {"austere_success_rate": 1.0, "survives": True, "episodes": 1}
            },
        }
        result = _normalize_behavior_survival(payload)
        self.assertEqual(result["minimal_profile"], "austere")

    def test_normalize_behavior_survival_empty_scenario_name_skipped(self) -> None:
        payload = {
            "scenarios": [
                {"scenario": "", "austere_success_rate": 1.0, "survives": True, "episodes": 1},
                {"scenario": "valid", "austere_success_rate": 0.5, "survives": False, "episodes": 2},
            ]
        }
        result = _normalize_behavior_survival(payload)
        names = [row["scenario"] for row in result["scenarios"]]
        self.assertNotIn("", names)
        self.assertIn("valid", names)
