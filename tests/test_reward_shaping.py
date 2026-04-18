"""Focused reward tests grouped by reward subpackage responsibility."""

from __future__ import annotations

import math
import unittest

from spider_cortex_sim.claim_tests import canonical_claim_tests
from spider_cortex_sim.noise import NoiseConfig
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.reward.audit import (
    REWARD_COMPONENT_AUDIT,
    _roadmap_status_for_profile,
    reward_component_audit,
    reward_profile_audit,
    shaping_disposition_summary,
)
from spider_cortex_sim.reward.computation import (
    apply_action_and_terrain_effects,
    apply_pressure_penalties,
    apply_progress_and_event_rewards,
    compute_predator_threat,
    copy_reward_components,
    empty_reward_components,
    reward_total,
)
from spider_cortex_sim.reward.profiles import (
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    REWARD_COMPONENT_NAMES,
    REWARD_PROFILES,
)
from spider_cortex_sim.reward.shaping import (
    DISPOSITION_EVIDENCE_CRITERIA,
    SCENARIO_AUSTERE_REQUIREMENTS,
    SHAPING_DISPOSITIONS,
    SHAPING_GAP_POLICY,
    SHAPING_REDUCTION_ROADMAP,
    shaping_reduction_roadmap,
    validate_gap_policy,
)
from spider_cortex_sim.scenarios import CAPABILITY_PROBE_SCENARIOS, SCENARIOS
from spider_cortex_sim.world import SpiderWorld
from spider_cortex_sim.world_types import TickContext, TickSnapshot


class ShapingDispositionTest(unittest.TestCase):
    def test_high_risk_components_have_shaping_disposition(self) -> None:
        """
        Verify that every reward component marked with shaping_risk == "high" includes a valid shaping_disposition.
        
        Asserts that each audited component whose metadata["shaping_risk"] is "high" contains the "shaping_disposition" key and that its value is one of the values in SHAPING_DISPOSITIONS.
        """
        audit = reward_component_audit()
        for component_name, metadata in audit.items():
            if metadata["shaping_risk"] != "high":
                continue
            self.assertIn(
                "shaping_disposition",
                metadata,
                f"High-risk component {component_name!r} needs a disposition",
            )
            self.assertIn(metadata["shaping_disposition"], SHAPING_DISPOSITIONS)

    def test_progress_components_have_shaping_disposition(self) -> None:
        audit = reward_component_audit()
        for component_name, metadata in audit.items():
            if metadata["category"] != "progress":
                continue
            self.assertIn(
                "shaping_disposition",
                metadata,
                f"Progress component {component_name!r} needs a disposition",
            )
            self.assertIn(metadata["shaping_disposition"], SHAPING_DISPOSITIONS)

    def test_shaping_disposition_summary_has_expected_categories(self) -> None:
        summary = shaping_disposition_summary()
        self.assertEqual(set(summary), set(SHAPING_DISPOSITIONS) | {"counts"})
        self.assertIsInstance(summary["counts"], dict)
        self.assertEqual(set(summary["counts"]), set(SHAPING_DISPOSITIONS))
        for disposition in SHAPING_DISPOSITIONS:
            self.assertIsInstance(summary[disposition], list)

    def test_under_investigation_components_have_rationale(self) -> None:
        # REWARD_COMPONENT_AUDIT currently has no under-investigation entries;
        # if one is added, it must carry an explicit rationale.
        """
        Ensure every component whose `shaping_disposition` is "under_investigation" provides a non-empty `disposition_rationale`.
        
        The test iterates over the reward component audit and asserts that for each component with `shaping_disposition == "under_investigation"`,
        `disposition_rationale` exists, is a `str`, and contains non-whitespace characters.
        """
        audit = reward_component_audit()
        for component_name, metadata in audit.items():
            if metadata["shaping_disposition"] != "under_investigation":
                continue
            rationale = metadata.get("disposition_rationale")
            self.assertIsInstance(
                rationale,
                str,
                f"Under-investigation component {component_name!r} needs a rationale",
            )
            self.assertGreater(len(rationale.strip()), 0)

    def test_missing_shaping_disposition_raises_value_error(self) -> None:
        metadata = dict(REWARD_COMPONENT_AUDIT["feeding"])
        metadata.pop("shaping_disposition")
        REWARD_COMPONENT_AUDIT["missing_disposition_test"] = metadata
        try:
            with self.assertRaisesRegex(ValueError, "missing shaping_disposition"):
                shaping_disposition_summary()
            with self.assertRaisesRegex(ValueError, "missing shaping_disposition"):
                reward_profile_audit("classic")
        finally:
            del REWARD_COMPONENT_AUDIT["missing_disposition_test"]


class ShapingReductionRoadmapTest(unittest.TestCase):
    def test_reduction_roadmap_contains_required_targets_and_fields(self) -> None:
        expected_targets = {
            "resting",
            "night_exposure",
            "hunger_pressure",
            "fatigue_pressure",
            "sleep_debt_pressure",
            "homeostasis_penalty",
            "terrain_cost",
            "action_cost",
        }
        self.assertEqual(set(SHAPING_REDUCTION_ROADMAP), expected_targets)
        required_fields = {
            "current_disposition",
            "target_disposition",
            "reduction_priority",
            "evidence_required",
            "blocking_scenarios",
            "notes",
        }
        audit = reward_component_audit()
        for component_name, entry in SHAPING_REDUCTION_ROADMAP.items():
            with self.subTest(component=component_name):
                self.assertEqual(set(entry), required_fields)
                self.assertEqual(
                    entry["current_disposition"],
                    audit[component_name]["shaping_disposition"],
                )
                self.assertIn(entry["target_disposition"], SHAPING_DISPOSITIONS)
                self.assertIn(entry["reduction_priority"], {"high", "medium", "low"})
                self.assertGreater(len(entry["evidence_required"]), 0)
                self.assertGreater(len(entry["blocking_scenarios"]), 0)

    def test_roadmap_current_dispositions_match_reward_component_audit(self) -> None:
        audit = reward_component_audit()
        for component_name, entry in SHAPING_REDUCTION_ROADMAP.items():
            with self.subTest(component=component_name):
                self.assertIn(component_name, audit)
                self.assertEqual(
                    entry["current_disposition"],
                    audit[component_name]["shaping_disposition"],
                )

    def test_gap_policy_uses_minimal_shaping_survival_threshold(self) -> None:
        self.assertEqual(
            SHAPING_GAP_POLICY["min_austere_survival_rate"],
            MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        )
        self.assertIn(
            "classic_minus_austere",
            SHAPING_GAP_POLICY["max_scenario_success_rate_delta"],
        )
        self.assertIn(
            "ecological_minus_austere",
            SHAPING_GAP_POLICY["max_mean_reward_delta"],
        )

    def test_gap_policy_thresholds_are_positive_and_consistent(self) -> None:
        self.assertGreater(SHAPING_GAP_POLICY["min_austere_survival_rate"], 0.0)
        self.assertLessEqual(SHAPING_GAP_POLICY["min_austere_survival_rate"], 1.0)
        self.assertGreater(SHAPING_GAP_POLICY["warning_threshold_multiplier"], 0.0)
        self.assertLess(SHAPING_GAP_POLICY["warning_threshold_multiplier"], 1.0)
        for policy_key in (
            "max_scenario_success_rate_delta",
            "max_mean_reward_delta",
        ):
            policy_values = SHAPING_GAP_POLICY[policy_key]
            self.assertIsInstance(policy_values, dict)
            for comparison_name, threshold in policy_values.items():
                with self.subTest(policy=policy_key, comparison=comparison_name):
                    self.assertGreater(threshold, 0.0)
                    self.assertGreater(
                        threshold,
                        threshold * SHAPING_GAP_POLICY["warning_threshold_multiplier"],
                    )
        self.assertEqual(
            set(SHAPING_GAP_POLICY["max_scenario_success_rate_delta"]),
            set(SHAPING_GAP_POLICY["max_mean_reward_delta"]),
        )

    def test_disposition_evidence_criteria_cover_existing_dispositions(self) -> None:
        """
        Verify that DISPOSITION_EVIDENCE_CRITERIA exactly covers all SHAPING_DISPOSITIONS and that each disposition's criteria contain the required fields with non-empty definitions, required evidence, and transition conditions.
        
        This test asserts:
        - The set of keys in DISPOSITION_EVIDENCE_CRITERIA equals SHAPING_DISPOSITIONS.
        - Each criteria mapping has exactly the fields: "definition", "required_evidence", "example_components", and "transition_conditions".
        - "definition" is a non-empty string.
        - "required_evidence" and "transition_conditions" are non-empty collections.
        """
        self.assertEqual(
            set(DISPOSITION_EVIDENCE_CRITERIA),
            set(SHAPING_DISPOSITIONS),
        )
        required_fields = {
            "definition",
            "required_evidence",
            "example_components",
            "transition_conditions",
        }
        for disposition, criteria in DISPOSITION_EVIDENCE_CRITERIA.items():
            with self.subTest(disposition=disposition):
                self.assertEqual(set(criteria), required_fields)
                self.assertIsInstance(criteria["definition"], str)
                self.assertGreater(len(criteria["definition"].strip()), 0)
                self.assertGreater(len(criteria["required_evidence"]), 0)
                self.assertGreater(len(criteria["transition_conditions"]), 0)

    def test_scenario_austere_requirements_have_expected_levels(self) -> None:
        """
        Verify SCENARIO_AUSTERE_REQUIREMENTS assigns the expected scenarios to gate, warning, and diagnostic levels and that each scenario entry contains required non-empty metadata.
        
        Asserts that:
        - The set of scenarios with "gate" level equals the expected gate_scenarios set.
        - The set of scenarios with "warning" level equals the expected warning_scenarios set.
        - The set of scenarios with "diagnostic" level equals the expected diagnostic_scenarios set.
        - For every scenario entry, `requirement_level` is one of {"gate","warning","diagnostic","excluded"}, `rationale` is non-empty after stripping, and `claim_test_linkage` is non-empty.
        """
        gate_scenarios = {
            "night_rest",
            "predator_edge",
            "entrance_ambush",
            "shelter_blockade",
            "two_shelter_tradeoff",
            "visual_olfactory_pincer",
            "olfactory_ambush",
            "visual_hunter_open_field",
        }
        warning_scenarios = {
            "recover_after_failed_chase",
            "food_vs_predator_conflict",
            "sleep_vs_exploration_conflict",
        }
        diagnostic_scenarios = {
            "open_field_foraging",
            "corridor_gauntlet",
            "exposed_day_foraging",
            "food_deprivation",
        }
        self.assertEqual(
            {
                name
                for name, entry in SCENARIO_AUSTERE_REQUIREMENTS.items()
                if entry["requirement_level"] == "gate"
            },
            gate_scenarios,
        )
        self.assertEqual(
            {
                name
                for name, entry in SCENARIO_AUSTERE_REQUIREMENTS.items()
                if entry["requirement_level"] == "warning"
            },
            warning_scenarios,
        )
        self.assertEqual(
            {
                name
                for name, entry in SCENARIO_AUSTERE_REQUIREMENTS.items()
                if entry["requirement_level"] == "diagnostic"
            },
            diagnostic_scenarios,
        )
        for scenario_name, entry in SCENARIO_AUSTERE_REQUIREMENTS.items():
            with self.subTest(scenario=scenario_name):
                self.assertIn(
                    entry["requirement_level"],
                    {"gate", "warning", "diagnostic", "excluded"},
                )
                self.assertGreater(len(entry["rationale"].strip()), 0)
                self.assertGreater(len(entry["claim_test_linkage"]), 0)

    def test_scenario_austere_requirements_cover_all_scenarios(self) -> None:
        self.assertEqual(
            set(SCENARIO_AUSTERE_REQUIREMENTS),
            set(SCENARIOS),
        )

    def test_gate_scenarios_cover_primary_claim_scenarios(self) -> None:
        """
        Asserts that every scenario required by primary claim tests is classified as a "gate" scenario.
        
        For each spec returned by canonical_claim_tests() with austere_survival_required == True,
        verifies that all scenarios referenced by the spec appear in SCENARIO_AUSTERE_REQUIREMENTS with
        requirement_level == "gate". Also verifies the "learning_without_privileged_signals" spec's
        scenarios are a subset of gate scenarios.
        """
        gate_scenarios = {
            name
            for name, entry in SCENARIO_AUSTERE_REQUIREMENTS.items()
            if entry["requirement_level"] == "gate"
        }
        for spec in canonical_claim_tests():
            if not spec.austere_survival_required:
                continue
            with self.subTest(claim=spec.name):
                self.assertTrue(set(spec.scenarios).issubset(gate_scenarios))
        learning_spec = next(
            spec
            for spec in canonical_claim_tests()
            if spec.name == "learning_without_privileged_signals"
        )
        self.assertTrue(set(learning_spec.scenarios).issubset(gate_scenarios))

    def test_capability_probe_scenarios_are_diagnostic(self) -> None:
        self.assertTrue(CAPABILITY_PROBE_SCENARIOS)
        for scenario_name in CAPABILITY_PROBE_SCENARIOS:
            with self.subTest(scenario=scenario_name):
                self.assertEqual(
                    SCENARIO_AUSTERE_REQUIREMENTS[scenario_name]["requirement_level"],
                    "diagnostic",
                )

    def test_no_scenario_has_conflicting_requirement_levels(self) -> None:
        levels_by_scenario: dict[str, set[str]] = {}
        for scenario_name, entry in SCENARIO_AUSTERE_REQUIREMENTS.items():
            levels_by_scenario.setdefault(scenario_name, set()).add(
                entry["requirement_level"]
            )
        for scenario_name, levels in levels_by_scenario.items():
            with self.subTest(scenario=scenario_name):
                self.assertEqual(len(levels), 1)
                self.assertIn(next(iter(levels)), {"gate", "warning", "diagnostic", "excluded"})

    def test_shaping_reduction_roadmap_returns_defensive_copy(self) -> None:
        roadmap = shaping_reduction_roadmap()
        self.assertEqual(roadmap["minimal_profile"], "austere")
        self.assertIn("reduction_targets", roadmap)
        roadmap["reduction_targets"]["resting"]["notes"] = "mutated"
        self.assertNotEqual(
            SHAPING_REDUCTION_ROADMAP["resting"]["notes"],
            "mutated",
        )

    def test_shaping_reduction_roadmap_returns_valid_structure(self) -> None:
        roadmap = shaping_reduction_roadmap()
        self.assertEqual(
            set(roadmap),
            {
                "minimal_profile",
                "survival_threshold",
                "reduction_targets",
                "gap_policy",
                "evidence_criteria",
                "scenario_requirements",
                "notes",
            },
        )
        self.assertEqual(roadmap["reduction_targets"], SHAPING_REDUCTION_ROADMAP)
        self.assertEqual(roadmap["gap_policy"], SHAPING_GAP_POLICY)
        self.assertEqual(roadmap["evidence_criteria"], DISPOSITION_EVIDENCE_CRITERIA)
        self.assertEqual(roadmap["scenario_requirements"], SCENARIO_AUSTERE_REQUIREMENTS)
        self.assertGreater(len(roadmap["notes"]), 0)

    def test_reward_profile_audit_exposes_reduction_roadmap_status(self) -> None:
        audit = reward_profile_audit("classic")
        status = audit["reduction_roadmap_status"]
        self.assertEqual(set(status), set(SHAPING_REDUCTION_ROADMAP))
        self.assertTrue(status["resting"]["has_configurable_weight"])
        self.assertEqual(status["resting"]["weight_status"], "active_configurable")
        self.assertGreater(status["resting"]["hardcoded_weight"], 0.0)
        self.assertEqual(
            status["homeostasis_penalty"]["target_disposition"],
            "under_investigation",
        )
        for component_name, entry in status.items():
            with self.subTest(component=component_name):
                self.assertIn("configured_weight_proxy", entry)
                self.assertIn("active_configured_weight_keys", entry)
                self.assertIn("weight_status", entry)

    def test_roadmap_status_reports_zeroed_configurable_weights(self) -> None:
        profile = dict(REWARD_PROFILES["classic"])
        for key in REWARD_COMPONENT_AUDIT["action_cost"]["configured_weight_keys"]:
            profile[key] = 0.0

        status = _roadmap_status_for_profile(profile)["action_cost"]

        self.assertTrue(status["configured_weight_keys"])
        self.assertEqual(status["active_configured_weight_keys"], [])
        self.assertTrue(status["has_configurable_weight"])
        self.assertEqual(status["configured_weight_proxy"], 0.0)
        self.assertEqual(status["weight_status"], "zeroed")

    def test_validate_gap_policy_reports_violations_and_warnings(self) -> None:
        comparison = {
            "minimal_profile": "austere",
            "deltas_vs_minimal": {
                "classic": {
                    "scenario_success_rate_delta": 0.21,
                    "mean_reward_delta": 0.60,
                },
                "ecological": {
                    "scenario_success_rate_delta": 0.13,
                    "mean_reward_delta": 0.35,
                },
            },
            "behavior_survival": {
                "available": True,
                "survival_rate": 0.48,
                "scenarios": {
                    "night_rest": {
                        "austere_success_rate": 0.49,
                        "survives": False,
                    },
                    "food_vs_predator_conflict": {
                        "austere_success_rate": 0.40,
                        "survives": False,
                    },
                    "open_field_foraging": {
                        "austere_success_rate": 0.10,
                        "survives": False,
                    },
                },
            },
        }
        result = validate_gap_policy(comparison)
        self.assertFalse(result["passes"])
        violation_metrics = {
            item["metric"]
            for item in result["violations"]
        }
        self.assertIn("scenario_success_rate_delta", violation_metrics)
        self.assertIn("mean_reward_delta", violation_metrics)
        self.assertIn("austere_survival_rate", violation_metrics)
        self.assertIn(
            {
                "scenario": "night_rest",
                "requirement_level": "gate",
                "metric": "austere_success_rate",
                "observed_rate": 0.49,
                "minimum": MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
            },
            result["violations"],
        )
        warning_metrics = {
            item["metric"]
            for item in result["warnings"]
        }
        self.assertIn("scenario_success_rate_delta", warning_metrics)
        self.assertIn("mean_reward_delta", warning_metrics)
        self.assertIn("austere_success_rate", warning_metrics)
        self.assertIn("open_field_foraging", result["checked_scenarios"])

    def test_validate_gap_policy_skips_unavailable_survival_payload(self) -> None:
        result = validate_gap_policy(
            {
                "minimal_profile": None,
                "deltas_vs_minimal": {},
                "behavior_survival": {
                    "available": False,
                    "survival_rate": 0.0,
                    "scenarios": {},
                },
            }
        )
        self.assertTrue(result["passes"])
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["warnings"], [])

    def test_validate_gap_policy_ignores_missing_or_nonfinite_scenario_rates(self) -> None:
        result = validate_gap_policy(
            {
                "minimal_profile": "austere",
                "deltas_vs_minimal": {},
                "behavior_survival": {
                    "available": True,
                    "survival_rate": 1.0,
                    "scenarios": {
                        "night_rest": {"survives": False},
                        "predator_edge": {
                            "austere_success_rate": float("nan"),
                            "survives": False,
                        },
                        "entrance_ambush": {
                            "austere_success_rate": "not-a-number",
                            "survives": False,
                        },
                        "food_vs_predator_conflict": {
                            "austere_success_rate": float("inf"),
                            "survives": False,
                        },
                    },
                },
            }
        )
        self.assertTrue(result["passes"])
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["checked_scenarios"], [])


class ShapingDispositionInvariantsTest(unittest.TestCase):
    def test_shaping_dispositions_sequence_has_five_values(self) -> None:
        self.assertEqual(len(SHAPING_DISPOSITIONS), 5)
        self.assertIn("defended", SHAPING_DISPOSITIONS)
        self.assertIn("weakened", SHAPING_DISPOSITIONS)
        self.assertIn("removed", SHAPING_DISPOSITIONS)
        self.assertIn("outcome_signal", SHAPING_DISPOSITIONS)
        self.assertIn("under_investigation", SHAPING_DISPOSITIONS)


if __name__ == "__main__":
    unittest.main()
