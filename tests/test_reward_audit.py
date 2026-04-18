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


class RewardAuditModuleTest(unittest.TestCase):
    def test_reward_component_audit_covers_declared_components(self) -> None:
        audit = reward_component_audit()
        self.assertEqual(set(audit.keys()), set(REWARD_COMPONENT_NAMES))
        self.assertEqual(set(REWARD_COMPONENT_AUDIT.keys()), set(REWARD_COMPONENT_NAMES))

    def test_reward_component_audit_classifies_progress_components(self) -> None:
        audit = reward_component_audit()
        self.assertEqual(audit["food_progress"]["category"], "progress")
        self.assertEqual(audit["shelter_progress"]["category"], "progress")
        self.assertEqual(audit["predator_escape"]["category"], "progress")

    def test_reward_profile_audit_reduces_progress_weight_in_austere_profile(self) -> None:
        classic_audit = reward_profile_audit("classic")
        austere_audit = reward_profile_audit("austere")
        self.assertEqual(austere_audit["profile"], "austere")
        self.assertEqual(
            austere_audit["component_weight_proxy"]["food_progress"],
            0.0,
        )
        self.assertEqual(
            austere_audit["component_weight_proxy"]["shelter_progress"],
            0.0,
        )
        self.assertLess(
            austere_audit["category_weight_proxy"]["progress"],
            classic_audit["category_weight_proxy"]["progress"],
        )

    def test_reward_profile_audit_exposes_configurable_and_hardcoded_mass(self) -> None:
        audit = reward_profile_audit("austere")
        self.assertIn("configurable_category_weight_proxy", audit)
        self.assertIn("hardcoded_mass", audit)
        self.assertGreater(audit["hardcoded_mass"], 0.0)
        self.assertLessEqual(
            audit["configurable_category_weight_proxy"]["progress"],
            audit["category_weight_proxy"]["progress"],
        )

    def test_reward_component_audit_returns_deep_copy_not_reference(self) -> None:
        audit1 = reward_component_audit()
        audit2 = reward_component_audit()
        audit1["food_progress"]["category"] = "mutated"
        self.assertEqual(audit2["food_progress"]["category"], "progress")

    def test_reward_component_audit_event_category_components_present(self) -> None:
        audit = reward_component_audit()
        expected_event_components = {
            "action_cost", "terrain_cost", "night_exposure",
            "predator_contact", "feeding", "resting", "shelter_entry",
            "night_shelter_bonus", "death_penalty",
        }
        actual_event_components = {
            name for name, data in audit.items()
            if data["category"] == "event"
        }
        self.assertEqual(actual_event_components, expected_event_components)

    def test_reward_component_audit_internal_pressure_components_present(self) -> None:
        audit = reward_component_audit()
        expected_pressure_components = {
            "hunger_pressure", "fatigue_pressure",
            "sleep_debt_pressure", "homeostasis_penalty",
        }
        actual_pressure_components = {
            name for name, data in audit.items()
            if data["category"] == "internal_pressure"
        }
        self.assertEqual(actual_pressure_components, expected_pressure_components)

    def test_reward_component_audit_all_entries_have_required_fields(self) -> None:
        """
        Verify every entry returned by `reward_component_audit()` includes the required metadata fields.
        
        Asserts that each component's metadata contains the keys:
        `category`, `source`, `gates`, `config_keys`, `configured_weight_keys`,
        `shaping_risk`, `shaping_disposition`, and `notes`. A missing key causes the test to fail.
        """
        audit = reward_component_audit()
        required_fields = {
            "category", "source", "gates", "config_keys",
            "configured_weight_keys", "shaping_risk",
            "shaping_disposition", "notes",
        }
        for component_name, metadata in audit.items():
            for field in required_fields:
                self.assertIn(
                    field,
                    metadata,
                    f"Reward component {component_name!r} missing field {field!r}",
                )

    def test_reward_component_audit_shaping_risk_values_are_valid(self) -> None:
        audit = reward_component_audit()
        valid_risk_levels = {"low", "medium", "high"}
        for name, data in audit.items():
            self.assertIn(
                data["shaping_risk"],
                valid_risk_levels,
                f"Component {name!r} has unexpected shaping_risk {data['shaping_risk']!r}",
            )

    def test_reward_component_audit_shaping_disposition_values_are_valid(self) -> None:
        audit = reward_component_audit()
        valid_dispositions = set(SHAPING_DISPOSITIONS)
        for name, data in audit.items():
            self.assertIn(
                data["shaping_disposition"],
                valid_dispositions,
                f"Component {name!r} has unexpected shaping_disposition {data['shaping_disposition']!r}",
            )

    def test_reward_component_audit_requires_rationale_for_non_outcome_signals(self) -> None:
        audit = reward_component_audit()
        for name, data in audit.items():
            with self.subTest(component=name):
                if data["shaping_disposition"] == "outcome_signal":
                    continue
                self.assertIn(
                    "disposition_rationale",
                    data,
                    f"Component {name!r} needs a shaping disposition rationale",
                )
                self.assertIsInstance(data["disposition_rationale"], str)
                self.assertGreater(len(data["disposition_rationale"].strip()), 0)

    def test_high_risk_progress_components_are_marked_removed(self) -> None:
        """
        Verify that the high-risk progress reward components are assigned the "removed" shaping disposition and that their disposition rationale mentions "austere".
        
        Checks the audit entries for "food_progress", "shelter_progress", and "predator_escape".
        """
        audit = reward_component_audit()
        for component_name in ("food_progress", "shelter_progress", "predator_escape"):
            self.assertEqual(audit[component_name]["shaping_disposition"], "removed")
            self.assertIn("austere", audit[component_name]["disposition_rationale"])

    def test_low_risk_outcome_components_are_marked_outcome_signals(self) -> None:
        """
        Asserts that specific low-risk reward components are labeled with the "outcome_signal" shaping disposition.
        
        Verifies that the audit entries for "predator_contact", "feeding", and "death_penalty" have `shaping_disposition == "outcome_signal"`.
        """
        audit = reward_component_audit()
        for component_name in ("predator_contact", "feeding", "death_penalty"):
            self.assertEqual(audit[component_name]["shaping_disposition"], "outcome_signal")
            self.assertIsInstance(
                audit[component_name]["disposition_rationale"],
                str,
            )
            self.assertGreater(
                len(audit[component_name]["disposition_rationale"].strip()),
                0,
            )

    def test_shaping_disposition_summary_groups_components_and_counts(self) -> None:
        summary = shaping_disposition_summary()
        counts = summary["counts"]
        for disposition in SHAPING_DISPOSITIONS:
            components = summary[disposition]
            self.assertEqual(counts[disposition], len(components))
        self.assertIn("food_progress", summary["removed"])
        self.assertIn("feeding", summary["outcome_signal"])
        self.assertIn("resting", summary["weakened"])

    def test_reward_profile_audit_raises_key_error_for_unknown_profile(self) -> None:
        with self.assertRaises(KeyError):
            reward_profile_audit("nonexistent_profile")

    def test_reward_profile_audit_classic_has_all_expected_keys(self) -> None:
        audit = reward_profile_audit("classic")
        expected_keys = {
            "profile", "component_weight_proxy", "category_weight_proxy",
            "configurable_category_weight_proxy", "disposition_summary",
            "reduction_roadmap_status",
            "dominant_category", "hardcoded_mass",
            "non_configurable_components", "notes",
        }
        self.assertEqual(set(audit.keys()), expected_keys)

    def test_reward_profile_audit_disposition_summary_contains_all_dispositions(self) -> None:
        audit = reward_profile_audit("classic")
        disposition_summary = audit["disposition_summary"]
        self.assertEqual(set(disposition_summary.keys()), set(SHAPING_DISPOSITIONS))
        for disposition, data in disposition_summary.items():
            self.assertIn("components", data, disposition)
            self.assertIn("component_count", data, disposition)
            self.assertIn("total_weight_proxy", data, disposition)
            self.assertEqual(data["component_count"], len(data["components"]))

    def test_reward_profile_audit_reports_removed_disposition_weight_gap(self) -> None:
        """
        Verify that the 'removed' shaping disposition carries positive total weight in the classic profile and exactly zero in the austere profile.
        
        Calls `reward_profile_audit("classic")` and `reward_profile_audit("austere")` and asserts that:
        - classic profile's `disposition_summary["removed"]["total_weight_proxy"]` is greater than 0.0
        - austere profile's `disposition_summary["removed"]["total_weight_proxy"]` is equal to 0.0
        """
        cases = [
            ("classic", self.assertGreater, 0.0),
            ("austere", self.assertEqual, 0.0),
        ]
        for profile_name, assertion, expected in cases:
            with self.subTest(profile=profile_name):
                audit = reward_profile_audit(profile_name)
                assertion(
                    audit["disposition_summary"]["removed"]["total_weight_proxy"],
                    expected,
                )

    def test_reward_profile_audit_dominant_category_is_valid(self) -> None:
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            self.assertIn(
                audit["dominant_category"],
                {"event", "progress", "internal_pressure"},
                f"Profile {profile_name!r} has unexpected dominant_category",
            )

    def test_reward_profile_audit_non_configurable_components_includes_expected(self) -> None:
        audit = reward_profile_audit("classic")
        non_configurable = audit["non_configurable_components"]
        for expected in ("predator_contact", "feeding"):
            self.assertIn(
                expected,
                non_configurable,
                f"Expected {expected!r} in non_configurable_components",
            )
        self.assertNotIn("death_penalty", non_configurable)

    def test_reward_profile_audit_non_configurable_components_is_sorted(self) -> None:
        audit = reward_profile_audit("classic")
        components = audit["non_configurable_components"]
        self.assertEqual(components, sorted(components))

    def test_reward_profile_audit_notes_is_non_empty_list(self) -> None:
        audit = reward_profile_audit("classic")
        self.assertIsInstance(audit["notes"], list)
        self.assertGreater(len(audit["notes"]), 0)

    def test_reward_profile_audit_category_weight_proxy_has_three_categories(self) -> None:
        audit = reward_profile_audit("classic")
        self.assertEqual(
            set(audit["category_weight_proxy"].keys()),
            {"event", "progress", "internal_pressure"},
        )

    def test_reward_profile_audit_classic_total_weight_is_positive(self) -> None:
        audit = reward_profile_audit("classic")
        total = sum(audit["category_weight_proxy"].values())
        self.assertGreater(total, 0.0)

    def test_reward_profile_audit_component_weights_are_non_negative(self) -> None:
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            for comp_name, weight in audit["component_weight_proxy"].items():
                self.assertGreaterEqual(
                    weight,
                    0.0,
                    f"Profile {profile_name!r} component {comp_name!r} has negative weight proxy",
                )

    def test_reward_profile_audit_ecological_has_progress_category(self) -> None:
        audit = reward_profile_audit("ecological")
        self.assertIn("progress", audit["category_weight_proxy"])
        self.assertGreater(audit["category_weight_proxy"]["progress"], 0.0)

    def test_reward_component_audit_resting_has_hardcoded_weight(self) -> None:
        audit = reward_component_audit()
        self.assertIn("hardcoded_weight", audit["resting"])
        self.assertGreater(float(audit["resting"]["hardcoded_weight"]), 0.0)

    def test_reward_component_audit_death_penalty_is_configurable(self) -> None:
        audit = reward_component_audit()
        self.assertEqual(audit["death_penalty"]["config_keys"], ["death_penalty"])
        self.assertEqual(audit["death_penalty"]["configured_weight_keys"], ["death_penalty"])
        self.assertNotIn("hardcoded_weight", audit["death_penalty"])


class ShapingDispositionSummaryAdditionalTest(unittest.TestCase):
    def test_shaping_disposition_summary_components_are_sorted_alphabetically(self) -> None:
        summary = shaping_disposition_summary()
        for disposition in SHAPING_DISPOSITIONS:
            components = summary[disposition]
            self.assertEqual(
                components,
                sorted(components),
                f"Components for disposition {disposition!r} are not sorted",
            )

    def test_shaping_disposition_summary_total_count_equals_all_components(self) -> None:
        summary = shaping_disposition_summary()
        total_from_counts = sum(summary["counts"][d] for d in SHAPING_DISPOSITIONS)
        total_components = len(REWARD_COMPONENT_NAMES)
        self.assertEqual(total_from_counts, total_components)

    def test_shaping_disposition_summary_no_duplicate_components_across_dispositions(self) -> None:
        summary = shaping_disposition_summary()
        seen = []
        for disposition in SHAPING_DISPOSITIONS:
            seen.extend(summary[disposition])
        self.assertEqual(len(seen), len(set(seen)), "Some components appear in multiple dispositions")

    def test_shaping_disposition_summary_every_component_name_appears_exactly_once(self) -> None:
        summary = shaping_disposition_summary()
        all_components = []
        for disposition in SHAPING_DISPOSITIONS:
            all_components.extend(summary[disposition])
        self.assertEqual(sorted(all_components), sorted(REWARD_COMPONENT_NAMES))

    def test_shaping_disposition_summary_defended_components_are_non_empty(self) -> None:
        summary = shaping_disposition_summary()
        self.assertGreater(len(summary["defended"]), 0)

    def test_shaping_disposition_summary_removed_components_include_progress_terms(self) -> None:
        """
        Verifies the 'removed' shaping disposition contains expected progress-related components.
        
        Asserts that the components 'food_progress', 'shelter_progress', 'predator_escape',
        'day_exploration', 'shelter_entry', and 'night_shelter_bonus' are present in
        summary['removed'] returned by shaping_disposition_summary().
        """
        summary = shaping_disposition_summary()
        for component in ("food_progress", "shelter_progress", "predator_escape",
                          "day_exploration", "shelter_entry", "night_shelter_bonus"):
            self.assertIn(component, summary["removed"])

    def test_shaping_disposition_summary_outcome_signal_components_include_feeding_and_death(self) -> None:
        summary = shaping_disposition_summary()
        self.assertIn("feeding", summary["outcome_signal"])
        self.assertIn("predator_contact", summary["outcome_signal"])
        self.assertIn("death_penalty", summary["outcome_signal"])

    def test_shaping_disposition_summary_weakened_contains_resting(self) -> None:
        summary = shaping_disposition_summary()
        self.assertIn("resting", summary["weakened"])

    def test_reward_profile_audit_disposition_components_are_sorted(self) -> None:
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            for disposition, data in audit["disposition_summary"].items():
                components = data["components"]
                self.assertEqual(
                    components,
                    sorted(components),
                    f"Profile {profile_name!r} disposition {disposition!r} components not sorted",
                )

    def test_reward_profile_audit_disposition_weight_proxies_are_non_negative(self) -> None:
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            for disposition, data in audit["disposition_summary"].items():
                self.assertGreaterEqual(
                    data["total_weight_proxy"],
                    0.0,
                    f"Profile {profile_name!r} disposition {disposition!r} has negative weight proxy",
                )

    def test_reward_profile_audit_disposition_component_counts_sum_to_total(self) -> None:
        audit = reward_profile_audit("classic")
        total_from_disposition = sum(
            data["component_count"]
            for data in audit["disposition_summary"].values()
        )
        self.assertEqual(total_from_disposition, len(REWARD_COMPONENT_NAMES))

    def test_reward_profile_audit_disposition_components_union_equals_all_components(self) -> None:
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            all_in_disposition = []
            for data in audit["disposition_summary"].values():
                all_in_disposition.extend(data["components"])
            self.assertEqual(
                sorted(all_in_disposition),
                sorted(REWARD_COMPONENT_NAMES),
                f"Profile {profile_name!r} disposition components do not cover all reward components",
            )

    def test_reward_profile_audit_austere_removed_weight_is_zero(self) -> None:
        audit = reward_profile_audit("austere")
        self.assertEqual(audit["disposition_summary"]["removed"]["total_weight_proxy"], 0.0)

    def test_reward_profile_audit_classic_removed_weight_exceeds_austere(self) -> None:
        removed_weights = {
            profile_name: reward_profile_audit(profile_name)["disposition_summary"][
                "removed"
            ]["total_weight_proxy"]
            for profile_name in ("classic", "austere")
        }
        self.assertGreater(
            removed_weights["classic"],
            removed_weights["austere"],
        )

    def test_reward_profile_audit_defended_weight_positive_in_all_profiles(self) -> None:
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            self.assertGreater(
                audit["disposition_summary"]["defended"]["total_weight_proxy"],
                0.0,
                f"Profile {profile_name!r} has no defended weight",
            )

    def test_reward_profile_audit_outcome_signal_summary_is_well_formed(self) -> None:
        """
        Verifies the outcome-signal disposition summary is present and well-formed for every reward profile.
        
        Checks that each profile's `disposition_summary["outcome_signal"]` is a dict containing the keys
        `components`, `component_count`, and `total_weight_proxy`; includes the components
        `predator_contact`, `feeding`, and `death_penalty`; has `component_count` equal to the length of
        `components`; and has `total_weight_proxy` as a float.
        """
        expected_components = {"predator_contact", "feeding", "death_penalty"}
        for profile_name in REWARD_PROFILES:
            audit = reward_profile_audit(profile_name)
            outcome_signal = audit["disposition_summary"]["outcome_signal"]
            self.assertIsInstance(outcome_signal, dict)
            self.assertIn("components", outcome_signal)
            self.assertIn("component_count", outcome_signal)
            self.assertIn("total_weight_proxy", outcome_signal)
            self.assertTrue(
                expected_components.issubset(set(outcome_signal["components"])),
                f"Profile {profile_name!r} missing expected outcome-signal components",
            )
            self.assertEqual(
                outcome_signal["component_count"],
                len(outcome_signal["components"]),
            )
            self.assertIsInstance(outcome_signal["total_weight_proxy"], float)

    def test_reward_profile_audit_disposition_all_dispositions_present_in_ecological(self) -> None:
        """
        Assert that the ecological reward profile's disposition_summary includes every shaping disposition.
        
        Checks that the set of keys in the profile audit's `disposition_summary` exactly matches `SHAPING_DISPOSITIONS`.
        """
        audit = reward_profile_audit("ecological")
        self.assertEqual(set(audit["disposition_summary"].keys()), set(SHAPING_DISPOSITIONS))


if __name__ == "__main__":
    unittest.main()
