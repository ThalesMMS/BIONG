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


class RewardProfileModuleTest(unittest.TestCase):
    def test_reward_profiles_contain_required_keys(self) -> None:
        """
        Verify each reward profile in REWARD_PROFILES includes the full set of required reward configuration keys.
        
        This test checks that every profile contains the following keys:
        "action_cost_stay", "action_cost_move", "hunger_pressure", "fatigue_pressure",
        "sleep_debt_pressure", "food_progress", "shelter_progress", "predator_escape",
        "predator_escape_bonus", "sleep_debt_night_awake", "sleep_debt_day_awake",
        "sleep_debt_overdue_threshold", "sleep_debt_health_penalty", "recent_contact_decay",
        "recent_pain_decay", and "death_penalty".
        
        If any key is missing from a profile, the test fails with a message identifying
        the profile name and the missing key.
        """
        required_keys = {
            "action_cost_stay", "action_cost_move",
            "hunger_pressure", "fatigue_pressure", "sleep_debt_pressure",
            "food_progress", "shelter_progress",
            "predator_escape", "predator_escape_bonus",
            "sleep_debt_night_awake", "sleep_debt_day_awake",
            "sleep_debt_overdue_threshold", "sleep_debt_health_penalty",
            "recent_contact_decay", "recent_pain_decay", "death_penalty",
            "turn_fatigue_cost", "reverse_fatigue_cost",
        }
        for profile_name, profile in REWARD_PROFILES.items():
            for key in required_keys:
                self.assertIn(key, profile, f"Profile '{profile_name}' missing key: {key}")

    def test_reward_profiles_has_classic_and_ecological(self) -> None:
        self.assertIn("classic", REWARD_PROFILES)
        self.assertIn("ecological", REWARD_PROFILES)
        self.assertIn("austere", REWARD_PROFILES)

    def test_reward_profiles_ecological_stricter_penalties(self) -> None:
        classic = REWARD_PROFILES["classic"]
        ecological = REWARD_PROFILES["ecological"]
        self.assertGreater(ecological["hunger_pressure"], classic["hunger_pressure"])
        self.assertGreater(ecological["fatigue_pressure"], classic["fatigue_pressure"])

    def test_austere_profile_zeroes_progress_guidance_terms(self) -> None:
        """
        Verify the "austere" reward profile zeros specific progress and guidance terms.
        
        Asserts that the following keys in REWARD_PROFILES["austere"] are exactly 0.0:
        `food_progress`, `shelter_progress`, `threat_shelter_progress`,
        `predator_escape`, `predator_escape_bonus`,
        `day_exploration_hungry`, and `day_exploration_calm`.
        """
        austere = REWARD_PROFILES["austere"]
        self.assertEqual(austere["food_progress"], 0.0)
        self.assertEqual(austere["shelter_progress"], 0.0)
        self.assertEqual(austere["threat_shelter_progress"], 0.0)
        self.assertEqual(austere["predator_escape"], 0.0)
        self.assertEqual(austere["predator_escape_bonus"], 0.0)
        self.assertEqual(austere["day_exploration_hungry"], 0.0)
        self.assertEqual(austere["day_exploration_calm"], 0.0)

    def test_austere_profile_has_non_zero_action_costs_and_pressures(self) -> None:
        austere = REWARD_PROFILES["austere"]
        self.assertGreater(austere["action_cost_stay"], 0.0)
        self.assertGreater(austere["action_cost_move"], 0.0)
        self.assertGreater(austere["hunger_pressure"], 0.0)
        self.assertGreater(austere["fatigue_pressure"], 0.0)
        self.assertGreater(austere["sleep_debt_pressure"], 0.0)

    def test_austere_profile_zeroes_shelter_entry_and_night_bonus(self) -> None:
        austere = REWARD_PROFILES["austere"]
        self.assertEqual(austere["shelter_entry"], 0.0)
        self.assertEqual(austere["night_shelter_bonus"], 0.0)


class AustereProfileStructureTest(unittest.TestCase):
    """Regression tests for the austere reward profile added in this PR."""

    def test_austere_profile_has_same_keys_as_classic(self) -> None:
        self.assertEqual(
            set(REWARD_PROFILES["austere"].keys()),
            set(REWARD_PROFILES["classic"].keys()),
        )

    def test_austere_profile_has_same_keys_as_ecological(self) -> None:
        """
        Check that the "austere" and "ecological" reward profiles have identical key sets.
        """
        self.assertEqual(
            set(REWARD_PROFILES["austere"].keys()),
            set(REWARD_PROFILES["ecological"].keys()),
        )

    def test_all_three_profiles_have_identical_key_sets(self) -> None:
        classic_keys = set(REWARD_PROFILES["classic"].keys())
        ecological_keys = set(REWARD_PROFILES["ecological"].keys())
        austere_keys = set(REWARD_PROFILES["austere"].keys())
        self.assertEqual(classic_keys, ecological_keys)
        self.assertEqual(classic_keys, austere_keys)

    def test_austere_predator_escape_stay_penalty_is_zero(self) -> None:
        self.assertEqual(REWARD_PROFILES["austere"]["predator_escape_stay_penalty"], 0.0)

    def test_austere_sleep_debt_terms_are_positive(self) -> None:
        """
        Verify austere profile's sleep-debt related terms are strictly greater than 0.0.
        
        Asserts that `sleep_debt_night_awake`, `sleep_debt_resting_night`, and `sleep_debt_deep_night` in `REWARD_PROFILES["austere"]` are each greater than 0.0.
        """
        austere = REWARD_PROFILES["austere"]
        self.assertGreater(austere["sleep_debt_night_awake"], 0.0)
        self.assertGreater(austere["sleep_debt_resting_night"], 0.0)
        self.assertGreater(austere["sleep_debt_deep_night"], 0.0)

    def test_austere_pressure_terms_are_stronger_than_ecological(self) -> None:
        # austere should have higher or equal homeostatic pressure vs. ecological
        """
        Assert that the 'austere' reward profile's homeostatic pressure terms are greater than or equal to the 'ecological' profile's.
        
        Checks the `hunger_pressure`, `fatigue_pressure`, and `sleep_debt_pressure` entries in `REWARD_PROFILES` and fails the test if any austere value is less than the corresponding ecological value.
        """
        austere = REWARD_PROFILES["austere"]
        ecological = REWARD_PROFILES["ecological"]
        self.assertGreaterEqual(austere["hunger_pressure"], ecological["hunger_pressure"])
        self.assertGreaterEqual(austere["fatigue_pressure"], ecological["fatigue_pressure"])
        self.assertGreaterEqual(austere["sleep_debt_pressure"], ecological["sleep_debt_pressure"])

    def test_all_reward_profile_names_are_registered(self) -> None:
        """
        Assert that the module registers exactly the expected reward profile names.
        
        Checks that the set of keys in REWARD_PROFILES is precisely {"classic", "ecological", "austere"}.
        """
        expected = {"classic", "ecological", "austere"}
        self.assertEqual(set(REWARD_PROFILES.keys()), expected)

    def test_austere_profile_all_values_are_finite(self) -> None:
        """
        Check that every numeric value in the "austere" reward profile is finite.
        
        The test iterates over all key/value pairs in REWARD_PROFILES["austere"] and fails if any value is not finite, reporting the offending key and value.
        """
        import math
        for key, value in REWARD_PROFILES["austere"].items():
            self.assertTrue(
                math.isfinite(value),
                f"austere[{key!r}] = {value!r} is not finite",
            )


class RewardProfileFatigueCostValuesTest(unittest.TestCase):
    """Regression tests for the current fatigue cost values across all profiles."""

    def _check_profile_fatigue_costs(self, profile_name: str) -> None:
        """
        Assert that the given reward profile's fatigue-cost constants match the canonical expected values.
        
        Parameters:
            profile_name (str): Key of the profile in `REWARD_PROFILES` to validate. The method asserts exact values (within 1e-6) for `base_fatigue_cost`, `move_fatigue_cost`, `idle_fatigue_cost`, `turn_fatigue_cost`, and `reverse_fatigue_cost`.
        """
        profile = REWARD_PROFILES[profile_name]
        self.assertAlmostEqual(profile["base_fatigue_cost"], 0.0035, places=6)
        self.assertAlmostEqual(profile["move_fatigue_cost"], 0.003, places=6)
        self.assertAlmostEqual(profile["idle_fatigue_cost"], 0.001, places=6)
        self.assertAlmostEqual(profile["turn_fatigue_cost"], 0.002, places=6)
        self.assertAlmostEqual(profile["reverse_fatigue_cost"], 0.004, places=6)

    def test_classic_fatigue_costs(self) -> None:
        self._check_profile_fatigue_costs("classic")

    def test_ecological_fatigue_costs(self) -> None:
        """
        Assert that the "ecological" reward profile defines the canonical fatigue cost constants:
        `base_fatigue_cost`, `move_fatigue_cost`, `idle_fatigue_cost`, `turn_fatigue_cost`,
        and `reverse_fatigue_cost` match the expected numeric values.
        """
        self._check_profile_fatigue_costs("ecological")

    def test_austere_fatigue_costs(self) -> None:
        self._check_profile_fatigue_costs("austere")

    def test_move_fatigue_cost_exceeds_idle_fatigue_cost_in_all_profiles(self) -> None:
        """
        Asserts that every reward profile sets move_fatigue_cost greater than idle_fatigue_cost.
        
        Iterates through REWARD_PROFILES and fails a subTest for any profile whose `move_fatigue_cost` is not strictly greater than its `idle_fatigue_cost`.
        """
        for name in REWARD_PROFILES:
            with self.subTest(profile=name):
                profile = REWARD_PROFILES[name]
                self.assertGreater(
                    profile["move_fatigue_cost"],
                    profile["idle_fatigue_cost"],
                    f"{name}: move_fatigue_cost should exceed idle_fatigue_cost",
                )


class RewardProfileNightExposureValuesTest(unittest.TestCase):
    """Regression tests for the current night-exposure values across all profiles."""

    def test_classic_night_exposure_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["night_exposure_fatigue"], 0.006, places=6)

    def test_classic_night_exposure_debt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["night_exposure_debt"], 0.009, places=6)

    def test_ecological_night_exposure_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["night_exposure_fatigue"], 0.009, places=6)

    def test_ecological_night_exposure_debt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["night_exposure_debt"], 0.014, places=6)

    def test_austere_night_exposure_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["night_exposure_fatigue"], 0.010, places=6)

    def test_austere_night_exposure_debt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["night_exposure_debt"], 0.015, places=6)

    def test_austere_exposure_fatigue_exceeds_classic(self) -> None:
        self.assertGreater(
            REWARD_PROFILES["austere"]["night_exposure_fatigue"],
            REWARD_PROFILES["classic"]["night_exposure_fatigue"],
        )

    def test_austere_exposure_debt_exceeds_classic(self) -> None:
        self.assertGreater(
            REWARD_PROFILES["austere"]["night_exposure_debt"],
            REWARD_PROFILES["classic"]["night_exposure_debt"],
        )


class RewardProfileSleepDebtPenaltyValuesTest(unittest.TestCase):
    """Regression tests for the current sleep-debt penalty values across all profiles."""

    def test_classic_sleep_debt_interrupt(self) -> None:
        """
        Asserts the classic profile's `sleep_debt_interrupt` penalty equals 0.035 within six decimal places.
        
        This test fails if `REWARD_PROFILES["classic"]["sleep_debt_interrupt"]` differs from 0.035 by more than 1e-6.
        """
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["sleep_debt_interrupt"], 0.035, places=6)

    def test_ecological_sleep_debt_interrupt(self) -> None:
        """
        Check that the 'ecological' reward profile specifies `sleep_debt_interrupt` equal to 0.050 within 6 decimal places.
        """
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["sleep_debt_interrupt"], 0.050, places=6)

    def test_austere_sleep_debt_interrupt(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["sleep_debt_interrupt"], 0.060, places=6)

    def test_classic_sleep_debt_night_awake(self) -> None:
        """
        Asserts that the classic reward profile's `sleep_debt_night_awake` equals 0.007.
        
        Compares REWARD_PROFILES["classic"]["sleep_debt_night_awake"] to 0.007 using an assertion with precision of 6 decimal places.
        """
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["sleep_debt_night_awake"], 0.007, places=6)

    def test_ecological_sleep_debt_night_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["sleep_debt_night_awake"], 0.011, places=6)

    def test_austere_sleep_debt_night_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["sleep_debt_night_awake"], 0.012, places=6)

    def test_classic_sleep_debt_day_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["sleep_debt_day_awake"], 0.0015, places=6)

    def test_ecological_sleep_debt_day_awake(self) -> None:
        """
        Asserts that the ecological reward profile's sleep debt penalty for daytime wakefulness equals 0.002.
        
        Performs an equality check within 1e-6 using assertAlmostEqual on REWARD_PROFILES["ecological"]["sleep_debt_day_awake"].
        """
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["sleep_debt_day_awake"], 0.002, places=6)

    def test_austere_sleep_debt_day_awake(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["sleep_debt_day_awake"], 0.0025, places=6)

    def test_sleep_debt_interrupt_is_greater_than_night_awake_in_all_profiles(self) -> None:
        for name in REWARD_PROFILES:
            with self.subTest(profile=name):
                profile = REWARD_PROFILES[name]
                self.assertGreater(
                    profile["sleep_debt_interrupt"],
                    profile["sleep_debt_night_awake"],
                    f"{name}: sleep_debt_interrupt should be greater than sleep_debt_night_awake",
                )

    def test_sleep_debt_night_awake_exceeds_day_awake_in_all_profiles(self) -> None:
        for name in REWARD_PROFILES:
            with self.subTest(profile=name):
                profile = REWARD_PROFILES[name]
                self.assertGreater(
                    profile["sleep_debt_night_awake"],
                    profile["sleep_debt_day_awake"],
                    f"{name}: sleep_debt_night_awake should exceed sleep_debt_day_awake",
                )


class RewardProfileClutterNarrowFatigueValuesTest(unittest.TestCase):
    """Regression tests for the current clutter/narrow fatigue values."""

    def test_classic_clutter_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["clutter_fatigue"], 0.002, places=6)

    def test_ecological_clutter_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["clutter_fatigue"], 0.003, places=6)

    def test_austere_clutter_fatigue(self) -> None:
        """
        Asserts that the "austere" reward profile's clutter_fatigue equals 0.003.
        
        Checks the austere profile's 'clutter_fatigue' numeric regression value with a tolerance of six decimal places.
        """
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["clutter_fatigue"], 0.003, places=6)

    def test_classic_narrow_fatigue(self) -> None:
        """
        Verifies that the 'classic' reward profile's `narrow_fatigue` parameter equals 0.0015 within six decimal places.
        
        Asserts that REWARD_PROFILES["classic"]["narrow_fatigue"] is approximately 0.0015 (places=6).
        """
        self.assertAlmostEqual(REWARD_PROFILES["classic"]["narrow_fatigue"], 0.0015, places=6)

    def test_ecological_narrow_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["ecological"]["narrow_fatigue"], 0.0025, places=6)

    def test_austere_narrow_fatigue(self) -> None:
        self.assertAlmostEqual(REWARD_PROFILES["austere"]["narrow_fatigue"], 0.0025, places=6)

    def test_clutter_fatigue_exceeds_narrow_fatigue_in_all_profiles(self) -> None:
        for name in REWARD_PROFILES:
            with self.subTest(profile=name):
                profile = REWARD_PROFILES[name]
                self.assertGreaterEqual(
                    profile["clutter_fatigue"],
                    profile["narrow_fatigue"],
                    f"{name}: clutter_fatigue should be >= narrow_fatigue",
                )


if __name__ == "__main__":
    unittest.main()
