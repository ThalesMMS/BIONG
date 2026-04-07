from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from spider_cortex_sim.noise import (
    HIGH_NOISE_PROFILE,
    LOW_NOISE_PROFILE,
    MEDIUM_NOISE_PROFILE,
    NONE_NOISE_PROFILE,
    NoiseConfig,
    canonical_noise_profile_names,
    resolve_noise_profile,
)


def _minimal_noise_config(**overrides: object) -> NoiseConfig:
    """Return a valid NoiseConfig with zero noise, optionally overriding fields."""
    defaults: dict[str, object] = dict(
        name="test",
        visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
        olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
        motor={"action_flip_prob": 0.0},
        spawn={"uniform_mix": 0.0},
        predator={"random_choice_prob": 0.0},
    )
    defaults.update(overrides)
    return NoiseConfig(**defaults)  # type: ignore[arg-type]


class NoiseConfigConstructionTest(unittest.TestCase):
    def test_name_is_coerced_to_str(self) -> None:
        cfg = _minimal_noise_config(name=42)  # type: ignore[arg-type]
        self.assertIsInstance(cfg.name, str)
        self.assertEqual(cfg.name, "42")

    def test_visual_values_are_coerced_to_float(self) -> None:
        cfg = _minimal_noise_config(visual={"certainty_jitter": 1, "direction_jitter": 0, "dropout_prob": 0})
        for value in cfg.visual.values():
            self.assertIsInstance(value, float)

    def test_olfactory_values_are_coerced_to_float(self) -> None:
        cfg = _minimal_noise_config(olfactory={"strength_jitter": 1, "direction_jitter": 0})
        for value in cfg.olfactory.values():
            self.assertIsInstance(value, float)

    def test_motor_values_are_coerced_to_float(self) -> None:
        cfg = _minimal_noise_config(motor={"action_flip_prob": 1})
        for value in cfg.motor.values():
            self.assertIsInstance(value, float)

    def test_spawn_values_are_coerced_to_float(self) -> None:
        cfg = _minimal_noise_config(spawn={"uniform_mix": 1})
        for value in cfg.spawn.values():
            self.assertIsInstance(value, float)

    def test_predator_values_are_coerced_to_float(self) -> None:
        cfg = _minimal_noise_config(predator={"random_choice_prob": 1})
        for value in cfg.predator.values():
            self.assertIsInstance(value, float)

    def test_construction_copies_dict_not_alias(self) -> None:
        original_visual = {"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0}
        cfg = _minimal_noise_config(visual=original_visual)
        original_visual["certainty_jitter"] = 999.0
        self.assertEqual(cfg.visual["certainty_jitter"], 0.0)

    def test_key_names_coerced_to_str(self) -> None:
        class KeyLike:
            def __init__(self, value: str) -> None:
                self.value = value

            def __str__(self) -> str:
                return self.value

        cfg = NoiseConfig(
            name="strkey_test",
            visual={
                KeyLike("certainty_jitter"): 0.0,  # type: ignore[dict-item]
                KeyLike("direction_jitter"): 0.0,  # type: ignore[dict-item]
                KeyLike("dropout_prob"): 0.0,  # type: ignore[dict-item]
            },
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={"action_flip_prob": 0.0},
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )
        for key in cfg.visual:
            self.assertIsInstance(key, str)
        self.assertIn("certainty_jitter", cfg.visual)


class NoiseConfigImmutabilityTest(unittest.TestCase):
    def test_frozen_dataclass_raises_on_attribute_set(self) -> None:
        cfg = _minimal_noise_config()
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            cfg.name = "tampered"  # type: ignore[misc]

    def test_to_summary_mutation_does_not_affect_visual(self) -> None:
        cfg = _minimal_noise_config(visual={"certainty_jitter": 0.05, "direction_jitter": 0.0, "dropout_prob": 0.0})
        summary = cfg.to_summary()
        summary["visual"]["certainty_jitter"] = 999.0
        self.assertAlmostEqual(cfg.visual["certainty_jitter"], 0.05)

    def test_to_summary_mutation_does_not_affect_olfactory(self) -> None:
        cfg = _minimal_noise_config(olfactory={"strength_jitter": 0.07, "direction_jitter": 0.0})
        summary = cfg.to_summary()
        summary["olfactory"]["strength_jitter"] = 999.0
        self.assertAlmostEqual(cfg.olfactory["strength_jitter"], 0.07)

    def test_to_summary_mutation_does_not_affect_motor(self) -> None:
        cfg = _minimal_noise_config(motor={"action_flip_prob": 0.03})
        summary = cfg.to_summary()
        summary["motor"]["action_flip_prob"] = 999.0
        self.assertAlmostEqual(cfg.motor["action_flip_prob"], 0.03)

    def test_channel_mappings_are_immutable(self) -> None:
        cfg = _minimal_noise_config()
        with self.assertRaises(TypeError):
            cfg.visual["certainty_jitter"] = 0.5  # type: ignore[index]


class NoiseConfigToSummaryTest(unittest.TestCase):
    def test_to_summary_returns_dict_with_required_keys(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        for key in ("name", "visual", "olfactory", "motor", "spawn", "predator"):
            self.assertIn(key, summary)

    def test_to_summary_name_matches(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        self.assertEqual(summary["name"], "none")

    def test_to_summary_low_profile_name_matches(self) -> None:
        summary = LOW_NOISE_PROFILE.to_summary()
        self.assertEqual(summary["name"], "low")

    def test_to_summary_visual_section_is_dict(self) -> None:
        summary = MEDIUM_NOISE_PROFILE.to_summary()
        self.assertIsInstance(summary["visual"], dict)

    def test_to_summary_high_profile_visual_certainty_jitter(self) -> None:
        summary = HIGH_NOISE_PROFILE.to_summary()
        self.assertAlmostEqual(summary["visual"]["certainty_jitter"], 0.14)

    def test_to_summary_none_profile_all_zeros(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        self.assertAlmostEqual(summary["visual"]["certainty_jitter"], 0.0)
        self.assertAlmostEqual(summary["visual"]["dropout_prob"], 0.0)
        self.assertAlmostEqual(summary["olfactory"]["strength_jitter"], 0.0)
        self.assertAlmostEqual(summary["motor"]["action_flip_prob"], 0.0)
        self.assertAlmostEqual(summary["spawn"]["uniform_mix"], 0.0)
        self.assertAlmostEqual(summary["predator"]["random_choice_prob"], 0.0)


class NoiseConfigFromSummaryTest(unittest.TestCase):
    def test_round_trip_preserves_none_profile(self) -> None:
        rebuilt = NoiseConfig.from_summary(NONE_NOISE_PROFILE.to_summary())
        self.assertEqual(rebuilt.to_summary(), NONE_NOISE_PROFILE.to_summary())

    def test_round_trip_preserves_high_profile(self) -> None:
        rebuilt = NoiseConfig.from_summary(HIGH_NOISE_PROFILE.to_summary())
        self.assertEqual(rebuilt.to_summary(), HIGH_NOISE_PROFILE.to_summary())

    def test_round_trip_preserves_custom_values(self) -> None:
        cfg = _minimal_noise_config(
            name="custom_round_trip",
            visual={"certainty_jitter": 0.11, "direction_jitter": 0.05, "dropout_prob": 0.02},
            olfactory={"strength_jitter": 0.09, "direction_jitter": 0.04},
            motor={"action_flip_prob": 0.07},
            spawn={"uniform_mix": 0.25},
            predator={"random_choice_prob": 0.33},
        )
        rebuilt = NoiseConfig.from_summary(cfg.to_summary())
        self.assertEqual(rebuilt.name, "custom_round_trip")
        self.assertAlmostEqual(rebuilt.visual["certainty_jitter"], 0.11)
        self.assertAlmostEqual(rebuilt.motor["action_flip_prob"], 0.07)
        self.assertAlmostEqual(rebuilt.spawn["uniform_mix"], 0.25)
        self.assertAlmostEqual(rebuilt.predator["random_choice_prob"], 0.33)

    def test_from_summary_missing_name_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        del summary["name"]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("name", str(ctx.exception))

    def test_from_summary_missing_visual_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        del summary["visual"]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("visual", str(ctx.exception))

    def test_from_summary_missing_olfactory_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        del summary["olfactory"]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("olfactory", str(ctx.exception))

    def test_from_summary_missing_motor_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        del summary["motor"]
        with self.assertRaises(ValueError):
            NoiseConfig.from_summary(summary)

    def test_from_summary_missing_spawn_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        del summary["spawn"]
        with self.assertRaises(ValueError):
            NoiseConfig.from_summary(summary)

    def test_from_summary_missing_predator_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        del summary["predator"]
        with self.assertRaises(ValueError):
            NoiseConfig.from_summary(summary)

    def test_from_summary_non_mapping_visual_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["visual"] = "not_a_mapping"  # type: ignore[assignment]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("visual", str(ctx.exception))

    def test_from_summary_non_mapping_olfactory_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["olfactory"] = [1, 2, 3]  # type: ignore[assignment]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("olfactory", str(ctx.exception))

    def test_from_summary_non_mapping_motor_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["motor"] = 42  # type: ignore[assignment]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("motor", str(ctx.exception))

    def test_from_summary_non_mapping_spawn_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["spawn"] = None  # type: ignore[assignment]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("spawn", str(ctx.exception))

    def test_from_summary_non_mapping_predator_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["predator"] = 0.5  # type: ignore[assignment]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("predator", str(ctx.exception))

    def test_from_summary_missing_visual_channel_key_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        del summary["visual"]["dropout_prob"]  # type: ignore[index]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("visual", str(ctx.exception))
        self.assertIn("dropout_prob", str(ctx.exception))

    def test_from_summary_unknown_visual_channel_key_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["visual"]["dropout_probability"] = 0.1  # type: ignore[index]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("visual", str(ctx.exception))
        self.assertIn("dropout_probability", str(ctx.exception))

    def test_from_summary_non_str_name_raises_value_error(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["name"] = 123  # type: ignore[assignment]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("name", str(ctx.exception))

    def test_from_summary_empty_summary_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            NoiseConfig.from_summary({})


class ResolveNoiseProfileTest(unittest.TestCase):
    def test_resolve_none_returns_none_profile(self) -> None:
        resolved = resolve_noise_profile(None)
        self.assertEqual(resolved.to_summary(), NONE_NOISE_PROFILE.to_summary())
        self.assertIsNot(resolved, NONE_NOISE_PROFILE)

    def test_resolve_string_none_returns_none_profile(self) -> None:
        resolved = resolve_noise_profile("none")
        self.assertEqual(resolved.to_summary(), NONE_NOISE_PROFILE.to_summary())
        self.assertIsNot(resolved, NONE_NOISE_PROFILE)

    def test_resolve_string_low_returns_low_profile(self) -> None:
        resolved = resolve_noise_profile("low")
        self.assertEqual(resolved.to_summary(), LOW_NOISE_PROFILE.to_summary())
        self.assertIsNot(resolved, LOW_NOISE_PROFILE)

    def test_resolve_string_medium_returns_medium_profile(self) -> None:
        resolved = resolve_noise_profile("medium")
        self.assertEqual(resolved.to_summary(), MEDIUM_NOISE_PROFILE.to_summary())
        self.assertIsNot(resolved, MEDIUM_NOISE_PROFILE)

    def test_resolve_string_high_returns_high_profile(self) -> None:
        resolved = resolve_noise_profile("high")
        self.assertEqual(resolved.to_summary(), HIGH_NOISE_PROFILE.to_summary())
        self.assertIsNot(resolved, HIGH_NOISE_PROFILE)

    def test_resolve_instance_returns_defensive_copy(self) -> None:
        cfg = _minimal_noise_config(name="passthrough_test")
        resolved = resolve_noise_profile(cfg)
        self.assertEqual(resolved.to_summary(), cfg.to_summary())
        self.assertIsNot(resolved, cfg)

    def test_resolve_unknown_string_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_noise_profile("unknown_profile_xyz")
        self.assertIn("unknown_profile_xyz", str(ctx.exception))

    def test_resolve_unknown_string_raises_value_error_for_empty(self) -> None:
        with self.assertRaises(ValueError):
            resolve_noise_profile("")

    def test_resolve_unknown_string_raises_value_error_for_case_mismatch(self) -> None:
        # Profile names are case-sensitive; "None" (capital N) should fail.
        with self.assertRaises(ValueError):
            resolve_noise_profile("None")

    def test_resolve_unknown_string_raises_value_error_for_high_capital(self) -> None:
        with self.assertRaises(ValueError):
            resolve_noise_profile("High")


class CanonicalNoiseProfileNamesTest(unittest.TestCase):
    def test_returns_tuple(self) -> None:
        names = canonical_noise_profile_names()
        self.assertIsInstance(names, tuple)

    def test_contains_none(self) -> None:
        self.assertIn("none", canonical_noise_profile_names())

    def test_contains_low(self) -> None:
        self.assertIn("low", canonical_noise_profile_names())

    def test_contains_medium(self) -> None:
        self.assertIn("medium", canonical_noise_profile_names())

    def test_contains_high(self) -> None:
        self.assertIn("high", canonical_noise_profile_names())

    def test_has_exactly_four_profiles(self) -> None:
        self.assertEqual(len(canonical_noise_profile_names()), 4)

    def test_none_is_first(self) -> None:
        # "none" should be the first/default name in canonical order.
        self.assertEqual(canonical_noise_profile_names()[0], "none")


class NoneNoiseProfileValuesTest(unittest.TestCase):
    def test_name_is_none_string(self) -> None:
        self.assertEqual(NONE_NOISE_PROFILE.name, "none")

    def test_visual_certainty_jitter_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.visual["certainty_jitter"], 0.0)

    def test_visual_direction_jitter_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.visual["direction_jitter"], 0.0)

    def test_visual_dropout_prob_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.visual["dropout_prob"], 0.0)

    def test_olfactory_strength_jitter_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.olfactory["strength_jitter"], 0.0)

    def test_olfactory_direction_jitter_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.olfactory["direction_jitter"], 0.0)

    def test_motor_action_flip_prob_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.motor["action_flip_prob"], 0.0)

    def test_spawn_uniform_mix_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.spawn["uniform_mix"], 0.0)

    def test_predator_random_choice_prob_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.predator["random_choice_prob"], 0.0)


class LowNoiseProfileValuesTest(unittest.TestCase):
    def test_name_is_low_string(self) -> None:
        self.assertEqual(LOW_NOISE_PROFILE.name, "low")

    def test_visual_certainty_jitter_positive(self) -> None:
        self.assertGreater(LOW_NOISE_PROFILE.visual["certainty_jitter"], 0.0)

    def test_motor_action_flip_prob_positive(self) -> None:
        self.assertGreater(LOW_NOISE_PROFILE.motor["action_flip_prob"], 0.0)

    def test_predator_random_choice_prob_positive(self) -> None:
        self.assertGreater(LOW_NOISE_PROFILE.predator["random_choice_prob"], 0.0)

    def test_all_values_below_one(self) -> None:
        for section in (LOW_NOISE_PROFILE.visual, LOW_NOISE_PROFILE.olfactory,
                        LOW_NOISE_PROFILE.motor, LOW_NOISE_PROFILE.spawn,
                        LOW_NOISE_PROFILE.predator):
            for key, val in section.items():
                self.assertLessEqual(val, 1.0, f"LOW profile {key}={val} exceeds 1.0")


class NoiseProfileMonotonicityTest(unittest.TestCase):
    """Noise levels should increase from none < low < medium < high for each channel."""

    def test_visual_certainty_jitter_monotone(self) -> None:
        self.assertLess(
            NONE_NOISE_PROFILE.visual["certainty_jitter"],
            LOW_NOISE_PROFILE.visual["certainty_jitter"],
        )
        self.assertLess(
            LOW_NOISE_PROFILE.visual["certainty_jitter"],
            MEDIUM_NOISE_PROFILE.visual["certainty_jitter"],
        )
        self.assertLess(
            MEDIUM_NOISE_PROFILE.visual["certainty_jitter"],
            HIGH_NOISE_PROFILE.visual["certainty_jitter"],
        )

    def test_motor_flip_prob_monotone(self) -> None:
        self.assertLess(
            NONE_NOISE_PROFILE.motor["action_flip_prob"],
            LOW_NOISE_PROFILE.motor["action_flip_prob"],
        )
        self.assertLess(
            LOW_NOISE_PROFILE.motor["action_flip_prob"],
            MEDIUM_NOISE_PROFILE.motor["action_flip_prob"],
        )
        self.assertLess(
            MEDIUM_NOISE_PROFILE.motor["action_flip_prob"],
            HIGH_NOISE_PROFILE.motor["action_flip_prob"],
        )

    def test_olfactory_strength_jitter_monotone(self) -> None:
        self.assertLess(
            NONE_NOISE_PROFILE.olfactory["strength_jitter"],
            LOW_NOISE_PROFILE.olfactory["strength_jitter"],
        )
        self.assertLess(
            LOW_NOISE_PROFILE.olfactory["strength_jitter"],
            MEDIUM_NOISE_PROFILE.olfactory["strength_jitter"],
        )
        self.assertLess(
            MEDIUM_NOISE_PROFILE.olfactory["strength_jitter"],
            HIGH_NOISE_PROFILE.olfactory["strength_jitter"],
        )

    def test_olfactory_direction_jitter_monotone(self) -> None:
        self.assertLess(
            NONE_NOISE_PROFILE.olfactory["direction_jitter"],
            LOW_NOISE_PROFILE.olfactory["direction_jitter"],
        )
        self.assertLess(
            LOW_NOISE_PROFILE.olfactory["direction_jitter"],
            MEDIUM_NOISE_PROFILE.olfactory["direction_jitter"],
        )
        self.assertLess(
            MEDIUM_NOISE_PROFILE.olfactory["direction_jitter"],
            HIGH_NOISE_PROFILE.olfactory["direction_jitter"],
        )

    def test_predator_random_choice_prob_monotone(self) -> None:
        self.assertLess(
            NONE_NOISE_PROFILE.predator["random_choice_prob"],
            LOW_NOISE_PROFILE.predator["random_choice_prob"],
        )
        self.assertLess(
            LOW_NOISE_PROFILE.predator["random_choice_prob"],
            MEDIUM_NOISE_PROFILE.predator["random_choice_prob"],
        )
        self.assertLess(
            MEDIUM_NOISE_PROFILE.predator["random_choice_prob"],
            HIGH_NOISE_PROFILE.predator["random_choice_prob"],
        )

    def test_spawn_uniform_mix_monotone(self) -> None:
        self.assertLess(
            NONE_NOISE_PROFILE.spawn["uniform_mix"],
            LOW_NOISE_PROFILE.spawn["uniform_mix"],
        )
        self.assertLess(
            LOW_NOISE_PROFILE.spawn["uniform_mix"],
            MEDIUM_NOISE_PROFILE.spawn["uniform_mix"],
        )
        self.assertLess(
            MEDIUM_NOISE_PROFILE.spawn["uniform_mix"],
            HIGH_NOISE_PROFILE.spawn["uniform_mix"],
        )


class NoiseConfigFromSummaryEdgeCasesTest(unittest.TestCase):
    def test_from_summary_with_extra_keys_succeeds(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["unexpected_key"] = "ignored_value"  # type: ignore[assignment]
        rebuilt = NoiseConfig.from_summary(summary)
        self.assertEqual(rebuilt.name, "none")

    def test_from_summary_rejects_unknown_channel_keys(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["visual"]["extra_channel"] = 0.01  # type: ignore[index]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("visual", str(ctx.exception))
        self.assertIn("extra_channel", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
