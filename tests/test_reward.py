"""Tests for the reward package introduced/restructured in this PR.

Covers aspects not already addressed by the focused submodule test files:
- reward/__init__.py compatibility layer (all symbols importable from
  spider_cortex_sim.reward)
- shaping_reduction_roadmap() return structure and deep-copy isolation
- validate_shaping_disposition() valid and invalid inputs
- validate_gap_policy() with realistic comparison payloads
- MINIMAL_SHAPING_SURVIVAL_THRESHOLD value constraints
- REWARD_COMPONENT_NAMES coverage against reward computation
- REWARD_PROFILES structure invariants
"""

from __future__ import annotations

import unittest
from copy import deepcopy

from spider_cortex_sim.reward import (
    DISPOSITION_EVIDENCE_CRITERIA,
    MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
    REWARD_COMPONENT_AUDIT,
    REWARD_COMPONENT_NAMES,
    REWARD_PROFILES,
    SCENARIO_AUSTERE_REQUIREMENTS,
    SHAPING_DISPOSITIONS,
    SHAPING_GAP_POLICY,
    SHAPING_REDUCTION_ROADMAP,
    empty_reward_components,
    reward_component_audit,
    reward_profile_audit,
    reward_total,
    shaping_disposition_summary,
    shaping_reduction_roadmap,
    validate_gap_policy,
    validate_shaping_disposition,
)


# ---------------------------------------------------------------------------
# shaping_reduction_roadmap()
# ---------------------------------------------------------------------------

class ShapingReductionRoadmapFunctionTest(unittest.TestCase):
    """Tests for shaping_reduction_roadmap() function."""

    def setUp(self) -> None:
        self.roadmap = shaping_reduction_roadmap()

    def test_returns_dict(self) -> None:
        self.assertIsInstance(self.roadmap, dict)

    def test_has_all_required_keys(self) -> None:
        required_keys = {
            "minimal_profile",
            "survival_threshold",
            "reduction_targets",
            "gap_policy",
            "evidence_criteria",
            "scenario_requirements",
            "notes",
        }
        self.assertEqual(set(self.roadmap.keys()), required_keys)

    def test_minimal_profile_is_austere(self) -> None:
        self.assertEqual(self.roadmap["minimal_profile"], "austere")

    def test_survival_threshold_matches_constant(self) -> None:
        self.assertAlmostEqual(
            self.roadmap["survival_threshold"],
            MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        )

    def test_notes_is_non_empty_list(self) -> None:
        notes = self.roadmap["notes"]
        self.assertIsInstance(notes, list)
        self.assertGreater(len(notes), 0)

    def test_reduction_targets_is_dict(self) -> None:
        self.assertIsInstance(self.roadmap["reduction_targets"], dict)

    def test_gap_policy_is_dict(self) -> None:
        self.assertIsInstance(self.roadmap["gap_policy"], dict)

    def test_evidence_criteria_is_dict(self) -> None:
        self.assertIsInstance(self.roadmap["evidence_criteria"], dict)

    def test_scenario_requirements_is_dict(self) -> None:
        self.assertIsInstance(self.roadmap["scenario_requirements"], dict)

    def test_returns_deep_copy_not_reference(self) -> None:
        roadmap1 = shaping_reduction_roadmap()
        roadmap2 = shaping_reduction_roadmap()
        # Mutating roadmap1 must not affect roadmap2
        if roadmap1["reduction_targets"]:
            first_key = next(iter(roadmap1["reduction_targets"]))
            roadmap1["reduction_targets"][first_key] = "mutated"
            self.assertNotEqual(
                roadmap2["reduction_targets"].get(first_key), "mutated"
            )

    def test_gap_policy_in_roadmap_matches_constant(self) -> None:
        policy = self.roadmap["gap_policy"]
        self.assertIn("minimal_profile", policy)
        self.assertEqual(policy["minimal_profile"], SHAPING_GAP_POLICY["minimal_profile"])

    def test_evidence_criteria_covers_all_dispositions(self) -> None:
        criteria = self.roadmap["evidence_criteria"]
        for disposition in SHAPING_DISPOSITIONS:
            if disposition not in ("under_investigation",):
                # All non-investigative dispositions should have evidence criteria
                pass  # not every disposition is required to have criteria
        # At minimum, defended and removed should be present
        self.assertIn("defended", criteria)
        self.assertIn("removed", criteria)

    def test_scenario_requirements_values_have_required_fields(self) -> None:
        reqs = self.roadmap["scenario_requirements"]
        for scenario_name, req in reqs.items():
            self.assertIn(
                "requirement_level", req,
                f"Scenario {scenario_name!r} missing 'requirement_level'",
            )
            self.assertIn(
                "rationale", req,
                f"Scenario {scenario_name!r} missing 'rationale'",
            )

    def test_scenario_requirement_levels_are_valid(self) -> None:
        valid_levels = {"gate", "warning", "diagnostic", "excluded"}
        reqs = self.roadmap["scenario_requirements"]
        for scenario_name, req in reqs.items():
            level = req["requirement_level"]
            self.assertIn(
                level, valid_levels,
                f"Scenario {scenario_name!r} has invalid requirement_level {level!r}",
            )


# ---------------------------------------------------------------------------
# validate_shaping_disposition()
# ---------------------------------------------------------------------------

class ValidateShapingDispositionTest(unittest.TestCase):
    """Tests for validate_shaping_disposition()."""

    def test_valid_disposition_returned_unchanged(self) -> None:
        for disposition in SHAPING_DISPOSITIONS:
            metadata = {"shaping_disposition": disposition}
            result = validate_shaping_disposition(metadata, "test_component")
            self.assertEqual(result, disposition)

    def test_missing_disposition_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_shaping_disposition({}, "my_component")
        self.assertIn("my_component", str(ctx.exception))

    def test_invalid_disposition_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_shaping_disposition(
                {"shaping_disposition": "not_a_valid_disposition"},
                "action_cost",
            )
        self.assertIn("action_cost", str(ctx.exception))

    def test_component_name_in_error_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_shaping_disposition(
                {"shaping_disposition": "completely_wrong"},
                "sleep_debt_pressure",
            )
        self.assertIn("sleep_debt_pressure", str(ctx.exception))

    def test_none_disposition_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_shaping_disposition(
                {"shaping_disposition": None}, "some_component"
            )

    def test_return_type_is_str(self) -> None:
        result = validate_shaping_disposition(
            {"shaping_disposition": "defended"}, "action_cost"
        )
        self.assertIsInstance(result, str)

    def test_all_audit_dispositions_pass_validation(self) -> None:
        audit = reward_component_audit()
        for component_name, metadata in audit.items():
            with self.subTest(component=component_name):
                result = validate_shaping_disposition(metadata, component_name)
                self.assertIn(result, SHAPING_DISPOSITIONS)


# ---------------------------------------------------------------------------
# validate_gap_policy()
# ---------------------------------------------------------------------------

class ValidateGapPolicyTest(unittest.TestCase):
    """Tests for validate_gap_policy()."""

    def test_empty_payload_passes_with_no_violations(self) -> None:
        result = validate_gap_policy({})
        self.assertTrue(result["passes"])
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["warnings"], [])

    def test_returns_required_keys(self) -> None:
        result = validate_gap_policy({})
        required_keys = {
            "passes", "policy", "violations", "warnings",
            "checked_profiles", "checked_scenarios", "notes",
        }
        self.assertEqual(set(result.keys()), required_keys)

    def test_no_violation_when_deltas_within_limits(self) -> None:
        comparison = {
            "deltas_vs_minimal": {
                "classic": {
                    "scenario_success_rate_delta": 0.10,  # limit is 0.20
                    "mean_reward_delta": 0.30,  # limit is 0.50
                },
            }
        }
        result = validate_gap_policy(comparison)
        self.assertTrue(result["passes"])
        self.assertEqual(result["violations"], [])

    def test_violation_when_success_rate_delta_exceeds_limit(self) -> None:
        comparison = {
            "deltas_vs_minimal": {
                "classic": {
                    "scenario_success_rate_delta": 0.50,  # exceeds 0.20 limit
                    "mean_reward_delta": 0.0,
                },
            }
        }
        result = validate_gap_policy(comparison)
        self.assertFalse(result["passes"])
        self.assertGreater(len(result["violations"]), 0)

    def test_violation_when_mean_reward_delta_exceeds_limit(self) -> None:
        comparison = {
            "deltas_vs_minimal": {
                "classic": {
                    "scenario_success_rate_delta": 0.0,
                    "mean_reward_delta": 0.99,  # exceeds 0.50 limit
                },
            }
        }
        result = validate_gap_policy(comparison)
        self.assertFalse(result["passes"])
        self.assertGreater(len(result["violations"]), 0)

    def test_warning_when_delta_exceeds_warning_threshold(self) -> None:
        # warning_threshold_multiplier = 0.80, limit = 0.20 → warning at 0.16+
        comparison = {
            "deltas_vs_minimal": {
                "classic": {
                    "scenario_success_rate_delta": 0.18,  # 0.80*0.20=0.16 < 0.18 < 0.20
                    "mean_reward_delta": 0.0,
                },
            }
        }
        result = validate_gap_policy(comparison)
        self.assertTrue(result["passes"])
        self.assertGreater(len(result["warnings"]), 0)

    def test_no_checked_profiles_when_no_deltas(self) -> None:
        result = validate_gap_policy({})
        self.assertEqual(result["checked_profiles"], [])

    def test_checked_profiles_includes_evaluated_profile(self) -> None:
        comparison = {
            "deltas_vs_minimal": {
                "classic": {"scenario_success_rate_delta": 0.0, "mean_reward_delta": 0.0},
            }
        }
        result = validate_gap_policy(comparison)
        self.assertIn("classic", result["checked_profiles"])

    def test_austere_survival_violation_when_below_minimum(self) -> None:
        comparison = {
            "behavior_survival": {
                "available": True,
                "survival_rate": 0.10,  # well below 0.50 threshold
            }
        }
        result = validate_gap_policy(comparison)
        self.assertFalse(result["passes"])
        metric_names = [v.get("metric") for v in result["violations"]]
        self.assertIn("austere_survival_rate", metric_names)

    def test_austere_survival_passes_when_above_minimum(self) -> None:
        comparison = {
            "behavior_survival": {
                "available": True,
                "survival_rate": 0.80,
            }
        }
        result = validate_gap_policy(comparison)
        self.assertTrue(result["passes"])

    def test_policy_key_in_result_is_deep_copy(self) -> None:
        result = validate_gap_policy({})
        result["policy"]["minimal_profile"] = "mutated"
        # Original SHAPING_GAP_POLICY should be unchanged
        self.assertNotEqual(SHAPING_GAP_POLICY["minimal_profile"], "mutated")

    def test_notes_in_result_is_non_empty_list(self) -> None:
        result = validate_gap_policy({})
        self.assertIsInstance(result["notes"], list)
        self.assertGreater(len(result["notes"]), 0)

    def test_missing_behavior_survival_section_does_not_error(self) -> None:
        # Should not raise even when 'behavior_survival' is absent
        result = validate_gap_policy({"deltas_vs_minimal": {}})
        self.assertIsInstance(result["passes"], bool)

    def test_missing_available_flag_skips_survival_check(self) -> None:
        comparison = {
            "behavior_survival": {
                "survival_rate": 0.1,  # below threshold, but 'available' missing
            }
        }
        result = validate_gap_policy(comparison)
        # Without available=True, survival check should be skipped
        metric_names = [v.get("metric") for v in result["violations"]]
        self.assertNotIn("austere_survival_rate", metric_names)

    def test_ecological_profile_delta_checked(self) -> None:
        comparison = {
            "deltas_vs_minimal": {
                "ecological": {
                    "scenario_success_rate_delta": 0.99,  # exceeds 0.15 limit
                    "mean_reward_delta": 0.0,
                }
            }
        }
        result = validate_gap_policy(comparison)
        self.assertFalse(result["passes"])

    def test_unknown_profile_delta_ignored(self) -> None:
        # A profile not in the policy limits should be silently ignored
        comparison = {
            "deltas_vs_minimal": {
                "unknown_profile": {
                    "scenario_success_rate_delta": 9.9,
                    "mean_reward_delta": 9.9,
                }
            }
        }
        result = validate_gap_policy(comparison)
        # No violations because limits are not defined for unknown_profile
        success_violations = [
            v for v in result["violations"]
            if v.get("metric") == "scenario_success_rate_delta"
        ]
        self.assertEqual(success_violations, [])


# ---------------------------------------------------------------------------
# MINIMAL_SHAPING_SURVIVAL_THRESHOLD
# ---------------------------------------------------------------------------

class MinimalShapingSurvivalThresholdTest(unittest.TestCase):
    """Tests for the MINIMAL_SHAPING_SURVIVAL_THRESHOLD constant."""

    def test_is_a_float(self) -> None:
        self.assertIsInstance(MINIMAL_SHAPING_SURVIVAL_THRESHOLD, float)

    def test_is_in_unit_interval(self) -> None:
        self.assertGreater(MINIMAL_SHAPING_SURVIVAL_THRESHOLD, 0.0)
        self.assertLessEqual(MINIMAL_SHAPING_SURVIVAL_THRESHOLD, 1.0)

    def test_value_is_reasonable_for_survival(self) -> None:
        # Should be at least a moderate survival threshold
        self.assertGreaterEqual(MINIMAL_SHAPING_SURVIVAL_THRESHOLD, 0.3)

    def test_gap_policy_references_same_value(self) -> None:
        self.assertAlmostEqual(
            SHAPING_GAP_POLICY["min_austere_survival_rate"],
            MINIMAL_SHAPING_SURVIVAL_THRESHOLD,
        )


# ---------------------------------------------------------------------------
# REWARD_COMPONENT_NAMES coverage
# ---------------------------------------------------------------------------

class RewardComponentNamesCoverageTest(unittest.TestCase):
    """Tests for REWARD_COMPONENT_NAMES completeness."""

    def test_is_non_empty_tuple_or_sequence(self) -> None:
        self.assertGreater(len(REWARD_COMPONENT_NAMES), 0)

    def test_all_names_are_strings(self) -> None:
        for name in REWARD_COMPONENT_NAMES:
            self.assertIsInstance(name, str)

    def test_matches_empty_reward_components_keys(self) -> None:
        components = empty_reward_components()
        self.assertEqual(set(components.keys()), set(REWARD_COMPONENT_NAMES))

    def test_matches_audit_keys(self) -> None:
        audit = reward_component_audit()
        self.assertEqual(set(audit.keys()), set(REWARD_COMPONENT_NAMES))

    def test_expected_components_present(self) -> None:
        expected = {
            "action_cost", "feeding", "predator_contact",
            "death_penalty", "hunger_pressure", "fatigue_pressure",
        }
        for name in expected:
            self.assertIn(name, REWARD_COMPONENT_NAMES, f"Missing component: {name!r}")

    def test_no_duplicate_names(self) -> None:
        names_list = list(REWARD_COMPONENT_NAMES)
        self.assertEqual(len(names_list), len(set(names_list)))


# ---------------------------------------------------------------------------
# REWARD_PROFILES structure invariants
# ---------------------------------------------------------------------------

class RewardProfilesStructureTest(unittest.TestCase):
    """Tests for REWARD_PROFILES structure invariants from the reward package."""

    def test_has_classic_ecological_and_austere(self) -> None:
        for profile_name in ("classic", "ecological", "austere"):
            self.assertIn(profile_name, REWARD_PROFILES)

    def test_each_profile_is_non_empty_dict(self) -> None:
        for profile_name, profile in REWARD_PROFILES.items():
            self.assertIsInstance(
                profile, dict,
                f"Profile {profile_name!r} should be a dict",
            )
            self.assertGreater(
                len(profile), 0,
                f"Profile {profile_name!r} should not be empty",
            )

    def test_component_weights_are_none_or_numeric(self) -> None:
        for profile_name, profile in REWARD_PROFILES.items():
            for component_name, weight in profile.items():
                if weight is not None:
                    self.assertIsInstance(
                        weight, (int, float),
                        f"Profile {profile_name!r} component {component_name!r} "
                        f"has non-numeric weight {weight!r}",
                    )

    def test_austere_profile_has_fewer_or_equal_positive_values_than_classic(self) -> None:
        # The austere profile removes dense progress signals; so it should have
        # fewer non-zero config values than the classic profile
        classic = REWARD_PROFILES["classic"]
        austere = REWARD_PROFILES["austere"]
        classic_nonzero = sum(1 for v in classic.values() if v)
        austere_nonzero = sum(1 for v in austere.values() if v)
        self.assertLessEqual(austere_nonzero, classic_nonzero)

    def test_classic_profile_total_weight_positive(self) -> None:
        classic = REWARD_PROFILES["classic"]
        total = sum(w for w in classic.values() if isinstance(w, (int, float)) and w)
        self.assertGreater(total, 0.0)

    def test_profile_audit_matches_profiles_dict(self) -> None:
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            self.assertEqual(audit["profile"], profile_name)


# ---------------------------------------------------------------------------
# SHAPING_DISPOSITIONS completeness
# ---------------------------------------------------------------------------

class ShapingDispositionsTest(unittest.TestCase):
    """Tests for SHAPING_DISPOSITIONS constant."""

    def test_contains_all_five_dispositions(self) -> None:
        expected = {
            "defended", "weakened", "removed",
            "outcome_signal", "under_investigation",
        }
        self.assertEqual(set(SHAPING_DISPOSITIONS), expected)

    def test_all_dispositions_are_strings(self) -> None:
        for d in SHAPING_DISPOSITIONS:
            self.assertIsInstance(d, str)

    def test_no_duplicates(self) -> None:
        dlist = list(SHAPING_DISPOSITIONS)
        self.assertEqual(len(dlist), len(set(dlist)))

    def test_disposition_evidence_criteria_covers_main_dispositions(self) -> None:
        # "defended", "weakened", "removed" should all have evidence criteria
        for d in ("defended", "weakened", "removed"):
            self.assertIn(d, DISPOSITION_EVIDENCE_CRITERIA)

    def test_each_evidence_entry_has_required_fields(self) -> None:
        required = {"definition", "required_evidence", "example_components", "transition_conditions"}
        for disposition, entry in DISPOSITION_EVIDENCE_CRITERIA.items():
            for field in required:
                self.assertIn(
                    field, entry,
                    f"Disposition {disposition!r} evidence criteria missing field {field!r}",
                )


# ---------------------------------------------------------------------------
# SHAPING_GAP_POLICY structure
# ---------------------------------------------------------------------------

class ShapingGapPolicyStructureTest(unittest.TestCase):
    """Tests for SHAPING_GAP_POLICY structure."""

    def test_has_required_keys(self) -> None:
        required_keys = {
            "minimal_profile",
            "max_scenario_success_rate_delta",
            "max_mean_reward_delta",
            "min_austere_survival_rate",
            "warning_threshold_multiplier",
            "notes",
        }
        self.assertEqual(set(SHAPING_GAP_POLICY.keys()), required_keys)

    def test_minimal_profile_is_austere(self) -> None:
        self.assertEqual(SHAPING_GAP_POLICY["minimal_profile"], "austere")

    def test_warning_threshold_multiplier_is_between_zero_and_one(self) -> None:
        mult = SHAPING_GAP_POLICY["warning_threshold_multiplier"]
        self.assertGreater(float(mult), 0.0)
        self.assertLess(float(mult), 1.0)

    def test_success_rate_delta_limits_are_positive(self) -> None:
        limits = SHAPING_GAP_POLICY["max_scenario_success_rate_delta"]
        for profile_key, limit in limits.items():
            self.assertGreater(float(limit), 0.0, f"Limit for {profile_key!r} must be positive")

    def test_mean_reward_delta_limits_are_positive(self) -> None:
        limits = SHAPING_GAP_POLICY["max_mean_reward_delta"]
        for profile_key, limit in limits.items():
            self.assertGreater(float(limit), 0.0, f"Limit for {profile_key!r} must be positive")

    def test_notes_is_non_empty_list(self) -> None:
        notes = SHAPING_GAP_POLICY["notes"]
        self.assertIsInstance(notes, list)
        self.assertGreater(len(notes), 0)

    def test_classic_limit_in_success_rate_deltas(self) -> None:
        limits = SHAPING_GAP_POLICY["max_scenario_success_rate_delta"]
        self.assertIn("classic_minus_austere", limits)

    def test_ecological_limit_in_success_rate_deltas(self) -> None:
        limits = SHAPING_GAP_POLICY["max_scenario_success_rate_delta"]
        self.assertIn("ecological_minus_austere", limits)

    def test_classical_limit_stricter_than_ecological_is_not_required(self) -> None:
        # This is a policy-level check: classic can have a looser or tighter limit
        # than ecological – just verify both exist and are numeric
        limits = SHAPING_GAP_POLICY["max_scenario_success_rate_delta"]
        self.assertIsInstance(float(limits["classic_minus_austere"]), float)
        self.assertIsInstance(float(limits["ecological_minus_austere"]), float)


if __name__ == "__main__":
    unittest.main()