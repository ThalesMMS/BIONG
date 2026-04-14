import unittest

from spider_cortex_sim.operational_profiles import (
    DEFAULT_OPERATIONAL_PROFILE,
    OPERATIONAL_PROFILES,
    OPTIONAL_PERCEPTION_DEFAULTS,
    OperationalProfile,
    canonical_operational_profile_names,
    resolve_operational_profile,
)


class OperationalProfileRegistryTest(unittest.TestCase):
    def test_default_profile_registered(self) -> None:
        self.assertIn("default_v1", OPERATIONAL_PROFILES)
        self.assertEqual(DEFAULT_OPERATIONAL_PROFILE.name, "default_v1")

    def test_canonical_names_include_default(self) -> None:
        self.assertEqual(canonical_operational_profile_names(), ("default_v1",))

    def test_resolve_by_name_returns_registered_profile(self) -> None:
        self.assertIs(resolve_operational_profile("default_v1"), DEFAULT_OPERATIONAL_PROFILE)

    def test_resolve_none_returns_default_profile(self) -> None:
        self.assertIs(resolve_operational_profile(None), DEFAULT_OPERATIONAL_PROFILE)

    def test_from_summary_round_trip_preserves_default_profile(self) -> None:
        rebuilt = OperationalProfile.from_summary(DEFAULT_OPERATIONAL_PROFILE.to_summary())
        self.assertEqual(rebuilt.to_summary(), DEFAULT_OPERATIONAL_PROFILE.to_summary())

    def test_default_profile_preserves_current_reflex_values(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        self.assertEqual(summary["brain"]["aux_weights"]["alert_center"], 0.32)
        self.assertEqual(summary["brain"]["reflex_logit_strengths"]["sleep_center"], 1.0)
        self.assertEqual(summary["brain"]["reflex_thresholds"]["visual_cortex"]["predator_certainty"], 0.45)

    def test_default_profile_preserves_current_perception_values(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        self.assertEqual(summary["perception"]["night_vision_range_penalty"], 1.0)
        self.assertEqual(summary["perception"]["visibility_clutter_penalty"], 0.18)
        self.assertEqual(summary["perception"]["lizard_detection_threshold"], 0.45)
        self.assertEqual(summary["perception"]["fov_half_angle"], 60.0)
        self.assertEqual(summary["perception"]["peripheral_half_angle"], 90.0)
        self.assertEqual(summary["perception"]["peripheral_certainty_penalty"], 0.35)
        self.assertEqual(summary["perception"]["perceptual_delay_ticks"], 1.0)
        self.assertEqual(summary["perception"]["perceptual_delay_noise"], 0.5)

    def test_default_profile_preserves_current_reward_values(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        self.assertEqual(summary["reward"]["predator_threat_smell_threshold"], 0.14)
        self.assertEqual(summary["reward"]["food_progress_hunger_threshold"], 0.30)
        self.assertEqual(summary["reward"]["predator_escape_distance_gain_threshold"], 2.0)


class ResolveOperationalProfileTest(unittest.TestCase):
    def test_resolve_instance_returns_same_object(self) -> None:
        profile = DEFAULT_OPERATIONAL_PROFILE
        self.assertIs(resolve_operational_profile(profile), profile)

    def test_resolve_unknown_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_operational_profile("nonexistent_profile_xyz")
        self.assertIn("nonexistent_profile_xyz", str(ctx.exception))

    def test_resolve_unknown_name_mentions_available_profiles(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_operational_profile("bad_name")
        self.assertIn("default_v1", str(ctx.exception))

    def test_canonical_names_returns_tuple(self) -> None:
        names = canonical_operational_profile_names()
        self.assertIsInstance(names, tuple)

    def test_canonical_names_nonempty(self) -> None:
        self.assertGreater(len(canonical_operational_profile_names()), 0)


class OperationalProfileImmutabilityTest(unittest.TestCase):
    def test_profile_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            DEFAULT_OPERATIONAL_PROFILE.name = "tampered"  # type: ignore[misc]

    def test_to_summary_mutation_does_not_affect_profile(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["brain"]["aux_weights"]["alert_center"] = 999.0
        self.assertEqual(DEFAULT_OPERATIONAL_PROFILE.brain_aux_weights["alert_center"], 0.32)

    def test_to_summary_perception_mutation_does_not_affect_profile(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["perception"]["night_vision_range_penalty"] = 999.0
        self.assertEqual(DEFAULT_OPERATIONAL_PROFILE.perception["night_vision_range_penalty"], 1.0)

    def test_to_summary_reward_mutation_does_not_affect_profile(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["reward"]["food_progress_hunger_threshold"] = 999.0
        self.assertEqual(DEFAULT_OPERATIONAL_PROFILE.reward["food_progress_hunger_threshold"], 0.30)


class OperationalProfileFromSummaryTest(unittest.TestCase):
    def test_from_summary_missing_required_top_level_key_raises_value_error(self) -> None:
        bad_summary = {
            "name": "bad",
            "version": 1,
            "brain": DEFAULT_OPERATIONAL_PROFILE.to_summary()["brain"],
            "reward": {},
        }
        with self.assertRaises(ValueError) as ctx:
            OperationalProfile.from_summary(bad_summary)
        self.assertIn("perception", str(ctx.exception))

    def test_from_summary_invalid_brain_block_raises_value_error(self) -> None:
        bad_summary = {
            "name": "bad",
            "version": 1,
            "brain": "not_a_mapping",
            "perception": {},
            "reward": {},
        }
        with self.assertRaises(ValueError) as ctx:
            OperationalProfile.from_summary(bad_summary)
        self.assertIn("brain", str(ctx.exception))

    def test_from_summary_missing_required_brain_key_raises_value_error(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        del summary["brain"]["reflex_thresholds"]
        with self.assertRaises(ValueError) as ctx:
            OperationalProfile.from_summary(summary)
        self.assertIn("reflex_thresholds", str(ctx.exception))

    def test_from_summary_coerces_version_to_int(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["version"] = "42"
        profile = OperationalProfile.from_summary(summary)
        self.assertEqual(profile.version, 42)
        self.assertIsInstance(profile.version, int)

    def test_from_summary_non_string_name_raises_value_error(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = 123
        with self.assertRaises(ValueError) as ctx:
            OperationalProfile.from_summary(summary)
        self.assertIn("name", str(ctx.exception))

    def test_from_summary_custom_values_preserved(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "custom_v2"
        summary["version"] = 2
        summary["brain"]["aux_weights"]["alert_center"] = 0.99
        summary["perception"]["night_vision_range_penalty"] = 2.0
        summary["reward"]["predator_threat_smell_threshold"] = 0.01
        profile = OperationalProfile.from_summary(summary)
        self.assertEqual(profile.name, "custom_v2")
        self.assertEqual(profile.version, 2)
        self.assertAlmostEqual(profile.brain_aux_weights["alert_center"], 0.99)
        self.assertAlmostEqual(profile.perception["night_vision_range_penalty"], 2.0)
        self.assertAlmostEqual(profile.reward["predator_threat_smell_threshold"], 0.01)

    def test_from_summary_adds_optional_perception_defaults_for_legacy_summary(self) -> None:
        """
        Ensure legacy perception summaries missing optional keys receive defaults when loaded.
        
        Verifies that `OperationalProfile.from_summary()` fills missing perception keys with expected defaults:
        - `fov_half_angle` -> 60.0
        - `peripheral_half_angle` -> 90.0
        - `peripheral_certainty_penalty` -> 0.35
        - `perceptual_delay_ticks` -> 1.0
        - `perceptual_delay_noise` -> 0.5
        """
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        del summary["perception"]["fov_half_angle"]
        del summary["perception"]["peripheral_half_angle"]
        del summary["perception"]["peripheral_certainty_penalty"]
        del summary["perception"]["perceptual_delay_ticks"]
        del summary["perception"]["perceptual_delay_noise"]

        profile = OperationalProfile.from_summary(summary)

        self.assertAlmostEqual(profile.perception["fov_half_angle"], 60.0)
        self.assertAlmostEqual(profile.perception["peripheral_half_angle"], 90.0)
        self.assertAlmostEqual(profile.perception["peripheral_certainty_penalty"], 0.35)
        self.assertAlmostEqual(profile.perception["perceptual_delay_ticks"], 1.0)
        self.assertAlmostEqual(profile.perception["perceptual_delay_noise"], 0.5)

    def test_from_summary_missing_percept_trace_ttl_raises_value_error(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        del summary["perception"]["percept_trace_ttl"]
        with self.assertRaises(ValueError) as ctx:
            OperationalProfile.from_summary(summary)
        self.assertIn("percept_trace_ttl", str(ctx.exception))

    def test_from_summary_missing_percept_trace_decay_raises_value_error(self) -> None:
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        del summary["perception"]["percept_trace_decay"]
        with self.assertRaises(ValueError) as ctx:
            OperationalProfile.from_summary(summary)
        self.assertIn("percept_trace_decay", str(ctx.exception))


class OperationalProfileDefaultKeysTest(unittest.TestCase):
    def test_all_brain_modules_in_aux_weights(self) -> None:
        expected_modules = {"visual_cortex", "sensory_cortex", "hunger_center", "sleep_center", "alert_center"}
        self.assertEqual(set(DEFAULT_OPERATIONAL_PROFILE.brain_aux_weights.keys()), expected_modules)

    def test_all_brain_modules_in_reflex_logit_strengths(self) -> None:
        expected_modules = {"visual_cortex", "sensory_cortex", "hunger_center", "sleep_center", "alert_center"}
        self.assertEqual(set(DEFAULT_OPERATIONAL_PROFILE.brain_reflex_logit_strengths.keys()), expected_modules)

    def test_all_brain_modules_in_reflex_thresholds(self) -> None:
        expected_modules = {"visual_cortex", "sensory_cortex", "hunger_center", "sleep_center", "alert_center"}
        self.assertEqual(set(DEFAULT_OPERATIONAL_PROFILE.brain_reflex_thresholds.keys()), expected_modules)

    def test_required_perception_keys_present(self) -> None:
        """
        Verify the default operational profile's perception mapping contains exactly the required keys.
        
        Asserts that DEFAULT_OPERATIONAL_PROFILE.perception.keys() matches the expected set of perception configuration keys, including the newly required "percept_trace_ttl" and "percept_trace_decay".
        """
        required = {
            "night_vision_range_penalty",
            "night_vision_min_range",
            "visibility_clutter_penalty",
            "visibility_narrow_penalty",
            "visibility_source_narrow_penalty",
            "visibility_night_penalty",
            "visibility_binary_threshold",
            "occluded_certainty_min",
            "occluded_certainty_base",
            "occluded_certainty_decay_per_step",
            "lizard_detection_range_motion_bonus",
            "lizard_detection_range_clutter_penalty",
            "lizard_detection_range_narrow_penalty",
            "lizard_detection_range_inside_penalty",
            "lizard_detection_motion_bonus",
            "lizard_detection_inside_penalty",
            "lizard_detection_threshold",
            "predator_motion_bonus",
            "percept_trace_ttl",
            "percept_trace_decay",
            "fov_half_angle",
            "peripheral_half_angle",
            "peripheral_certainty_penalty",
            "perceptual_delay_ticks",
            "perceptual_delay_noise",
        }
        self.assertEqual(set(DEFAULT_OPERATIONAL_PROFILE.perception.keys()), required)

    def test_required_reward_keys_present(self) -> None:
        required = {
            "predator_threat_contact_threshold",
            "predator_threat_recent_pain_threshold",
            "predator_threat_distance_threshold",
            "predator_threat_smell_threshold",
            "narrow_predator_risk_max_distance",
            "food_progress_hunger_threshold",
            "food_progress_hunger_bias",
            "shelter_progress_fatigue_threshold",
            "shelter_progress_sleep_debt_threshold",
            "shelter_progress_bias",
            "shelter_progress_sleep_debt_weight",
            "predator_escape_contact_threshold",
            "day_exploration_hunger_threshold",
            "predator_visibility_threshold",
            "predator_escape_distance_gain_threshold",
            "reset_sleep_predator_distance_threshold",
            "reset_sleep_contact_threshold",
            "reset_sleep_recent_pain_threshold",
        }
        self.assertEqual(set(DEFAULT_OPERATIONAL_PROFILE.reward.keys()), required)

    def test_all_aux_weight_values_are_floats(self) -> None:
        for key, val in DEFAULT_OPERATIONAL_PROFILE.brain_aux_weights.items():
            self.assertIsInstance(val, float, f"aux_weight[{key!r}] is not a float")

    def test_all_perception_values_are_floats(self) -> None:
        for key, val in DEFAULT_OPERATIONAL_PROFILE.perception.items():
            self.assertIsInstance(val, float, f"perception[{key!r}] is not a float")

    def test_all_reward_values_are_floats(self) -> None:
        for key, val in DEFAULT_OPERATIONAL_PROFILE.reward.items():
            self.assertIsInstance(val, float, f"reward[{key!r}] is not a float")

    def test_default_version_is_one(self) -> None:
        self.assertEqual(DEFAULT_OPERATIONAL_PROFILE.version, 1)

    def test_default_name_is_string(self) -> None:
        self.assertIsInstance(DEFAULT_OPERATIONAL_PROFILE.name, str)


class OptionalPerceptionDefaultsTest(unittest.TestCase):
    """Tests for the OPTIONAL_PERCEPTION_DEFAULTS constant."""

    def test_contains_fov_half_angle(self) -> None:
        self.assertIn("fov_half_angle", OPTIONAL_PERCEPTION_DEFAULTS)

    def test_contains_peripheral_half_angle(self) -> None:
        self.assertIn("peripheral_half_angle", OPTIONAL_PERCEPTION_DEFAULTS)

    def test_contains_peripheral_certainty_penalty(self) -> None:
        self.assertIn("peripheral_certainty_penalty", OPTIONAL_PERCEPTION_DEFAULTS)

    def test_contains_perceptual_delay_ticks(self) -> None:
        self.assertIn("perceptual_delay_ticks", OPTIONAL_PERCEPTION_DEFAULTS)

    def test_contains_perceptual_delay_noise(self) -> None:
        self.assertIn("perceptual_delay_noise", OPTIONAL_PERCEPTION_DEFAULTS)

    def test_has_exactly_five_keys(self) -> None:
        self.assertEqual(len(OPTIONAL_PERCEPTION_DEFAULTS), 5)

    def test_fov_half_angle_default_is_60(self) -> None:
        self.assertAlmostEqual(OPTIONAL_PERCEPTION_DEFAULTS["fov_half_angle"], 60.0)

    def test_peripheral_half_angle_default_is_90(self) -> None:
        self.assertAlmostEqual(OPTIONAL_PERCEPTION_DEFAULTS["peripheral_half_angle"], 90.0)

    def test_peripheral_certainty_penalty_default_is_0_35(self) -> None:
        self.assertAlmostEqual(OPTIONAL_PERCEPTION_DEFAULTS["peripheral_certainty_penalty"], 0.35)

    def test_perceptual_delay_ticks_default_is_1(self) -> None:
        self.assertAlmostEqual(OPTIONAL_PERCEPTION_DEFAULTS["perceptual_delay_ticks"], 1.0)

    def test_perceptual_delay_noise_default_is_0_5(self) -> None:
        self.assertAlmostEqual(OPTIONAL_PERCEPTION_DEFAULTS["perceptual_delay_noise"], 0.5)

    def test_all_values_are_floats(self) -> None:
        for key, val in OPTIONAL_PERCEPTION_DEFAULTS.items():
            self.assertIsInstance(val, float, f"OPTIONAL_PERCEPTION_DEFAULTS[{key!r}] is not a float")

    def test_peripheral_half_angle_exceeds_fov_half_angle(self) -> None:
        self.assertGreater(
            OPTIONAL_PERCEPTION_DEFAULTS["peripheral_half_angle"],
            OPTIONAL_PERCEPTION_DEFAULTS["fov_half_angle"],
        )

    def test_from_summary_uses_optional_defaults_for_partial_perception(self) -> None:
        """from_summary with only required keys should add all optional defaults."""
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        for key in list(OPTIONAL_PERCEPTION_DEFAULTS):
            summary["perception"].pop(key, None)

        profile = OperationalProfile.from_summary(summary)

        for key, expected_val in OPTIONAL_PERCEPTION_DEFAULTS.items():
            self.assertAlmostEqual(
                profile.perception[key],
                expected_val,
                msg=f"Expected OPTIONAL_PERCEPTION_DEFAULTS[{key!r}]={expected_val} for legacy summary",
            )

    def test_from_summary_does_not_override_existing_optional_keys(self) -> None:
        """Explicit values in the summary must not be replaced by defaults."""
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["perception"]["fov_half_angle"] = 45.0
        summary["perception"]["peripheral_half_angle"] = 120.0

        profile = OperationalProfile.from_summary(summary)

        self.assertAlmostEqual(profile.perception["fov_half_angle"], 45.0)
        self.assertAlmostEqual(profile.perception["peripheral_half_angle"], 120.0)


class DefaultOperationalProfilePerceptionKeysTest(unittest.TestCase):
    def test_fov_half_angle_present_with_default_value(self) -> None:
        self.assertAlmostEqual(DEFAULT_OPERATIONAL_PROFILE.perception["fov_half_angle"], 60.0)

    def test_peripheral_half_angle_present_with_default_value(self) -> None:
        self.assertAlmostEqual(DEFAULT_OPERATIONAL_PROFILE.perception["peripheral_half_angle"], 90.0)

    def test_peripheral_certainty_penalty_present_with_default_value(self) -> None:
        self.assertAlmostEqual(DEFAULT_OPERATIONAL_PROFILE.perception["peripheral_certainty_penalty"], 0.35)

    def test_perceptual_delay_ticks_present_with_default_value(self) -> None:
        self.assertAlmostEqual(DEFAULT_OPERATIONAL_PROFILE.perception["perceptual_delay_ticks"], 1.0)

    def test_perceptual_delay_noise_present_with_default_value(self) -> None:
        self.assertAlmostEqual(DEFAULT_OPERATIONAL_PROFILE.perception["perceptual_delay_noise"], 0.5)


if __name__ == "__main__":
    unittest.main()