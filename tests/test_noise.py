from __future__ import annotations

import json
import unittest
from collections.abc import Sequence
from dataclasses import FrozenInstanceError

from spider_cortex_sim.maps import CLUTTER, NARROW, OPEN
from spider_cortex_sim.noise import (
    CANONICAL_ROBUSTNESS_CONDITIONS,
    HIGH_NOISE_PROFILE,
    LOW_NOISE_PROFILE,
    MEDIUM_NOISE_PROFILE,
    NONE_NOISE_PROFILE,
    NoiseConfig,
    RobustnessMatrixSpec,
    SLIP_ADJACENT_ACTIONS,
    apply_motor_noise,
    canonical_robustness_matrix,
    canonical_noise_profile_names,
    compute_execution_difficulty,
    motor_slip_reason,
    resolve_noise_profile,
    sample_slip_action,
)
from spider_cortex_sim.world import SpiderWorld


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


class _FakeChoiceMotorRng:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.count: int | None = None
        self.weights: list[float] = []

    def choice(self, count: int, p: Sequence[float]) -> int:
        self.count = count
        self.weights = [float(value) for value in p]
        return self.index


def _terrain_with_cleanup(
    test_case: unittest.TestCase,
    world: SpiderWorld,
) -> tuple[dict[tuple[int, int], str], tuple[int, int]]:
    """Save terrain at the spider position and restore it after the test."""
    pos = world.spider_pos()
    terrain = world.map_template.terrain
    original_present = pos in terrain
    original_terrain = terrain.get(pos)

    def restore_terrain() -> None:
        if original_present:
            terrain[pos] = original_terrain
        else:
            terrain.pop(pos, None)

    test_case.addCleanup(restore_terrain)
    return terrain, pos


def _compute_slip_and_difficulty(
    world: SpiderWorld,
    terrain: dict[tuple[int, int], str],
    pos: tuple[int, int],
    terrain_type: str,
    heading: tuple[float, float],
    fatigue: float,
    momentum: float = 0.0,
) -> dict[str, object]:
    """Configure a MOVE_RIGHT motor-noise case and return diagnostics."""
    terrain[pos] = terrain_type
    world.state.heading_dx = float(heading[0])
    world.state.heading_dy = float(heading[1])
    world.state.fatigue = float(fatigue)
    world.state.momentum = float(momentum)
    return apply_motor_noise(world, "MOVE_RIGHT")


def _assert_execution_difficulty(
    test_case: unittest.TestCase,
    *,
    heading: tuple[float, float],
    intended_direction: tuple[float, float],
    terrain: str,
    fatigue: float,
    expected_difficulty: float,
    expected_components: dict[str, float],
    momentum: float = 0.0,
) -> None:
    difficulty, components = compute_execution_difficulty(
        heading=heading,
        intended_direction=intended_direction,
        terrain=terrain,
        fatigue=fatigue,
        momentum=momentum,
    )

    for key, expected_value in expected_components.items():
        test_case.assertAlmostEqual(components[key], expected_value)
    test_case.assertAlmostEqual(difficulty, expected_difficulty)


class MotorNoiseFunctionsTest(unittest.TestCase):
    def test_compute_execution_difficulty_uses_terrain_heading_and_fatigue(self) -> None:
        _assert_execution_difficulty(
            self,
            heading=(1.0, 0.0),
            intended_direction=(-1.0, 0.0),
            terrain=NARROW,
            fatigue=1.0,
            expected_difficulty=1.0,
            expected_components={
                "orientation_alignment": 0.0,
                "terrain_difficulty": 0.7,
                "fatigue_factor": 1.0,
                "raw_difficulty": 1.4,
            },
        )

    def test_compute_execution_difficulty_stay_has_no_mismatch(self) -> None:
        _assert_execution_difficulty(
            self,
            heading=(0.0, -1.0),
            intended_direction=(0.0, 0.0),
            terrain=OPEN,
            fatigue=1.0,
            expected_difficulty=0.0,
            expected_components={
                "orientation_alignment": 1.0,
                "orientation_mismatch": 0.0,
            },
        )

    def test_sample_slip_action_biases_toward_stay(self) -> None:
        fake_rng = _FakeChoiceMotorRng()

        self.assertEqual(sample_slip_action("MOVE_UP", fake_rng), "STAY")
        self.assertEqual(fake_rng.count, 3)
        self.assertGreater(fake_rng.weights[0], fake_rng.weights[1])
        self.assertGreater(fake_rng.weights[0], fake_rng.weights[2])
        self.assertAlmostEqual(sum(fake_rng.weights), 1.0)

    def test_motor_slip_reason_uses_base_for_zero_difficulty_base_slip(self) -> None:
        self.assertEqual(
            motor_slip_reason(
                {
                    "raw_difficulty": 0.0,
                    "terrain_difficulty": 0.0,
                    "orientation_mismatch": 0.0,
                    "fatigue_factor": 0.0,
                },
                base_slip_rate=1.0,
            ),
            "base",
        )

    def test_apply_motor_noise_handles_orientation_actions_without_slip(self) -> None:
        world = SpiderWorld(
            seed=17,
            noise_profile=_minimal_noise_config(
                motor={
                    "action_flip_prob": 1.0,
                    "orientation_slip_factor": 1.0,
                    "terrain_slip_factor": 1.0,
                    "fatigue_slip_factor": 1.0,
                }
            ),
            lizard_move_interval=999999,
        )

        result = apply_motor_noise(world, "ORIENT_LEFT")

        self.assertFalse(result["occurred"])
        self.assertEqual(result["reason"], "none")
        self.assertEqual(result["executed_action"], "ORIENT_LEFT")
        self.assertAlmostEqual(result["slip_probability"], 0.0)
        self.assertIn("momentum", result["components"])

    def test_apply_motor_noise_reports_base_slip_for_movement(self) -> None:
        world = SpiderWorld(
            seed=19,
            noise_profile=_minimal_noise_config(
                motor={
                    "action_flip_prob": 1.0,
                    "orientation_slip_factor": 0.0,
                    "terrain_slip_factor": 0.0,
                    "fatigue_slip_factor": 0.0,
                }
            ),
            lizard_move_interval=999999,
        )

        result = apply_motor_noise(world, "MOVE_RIGHT")

        self.assertTrue(result["occurred"])
        self.assertEqual(result["reason"], "base")
        self.assertEqual(result["original_action"], "MOVE_RIGHT")
        self.assertNotEqual(result["executed_action"], "MOVE_RIGHT")
        self.assertAlmostEqual(result["slip_probability"], 1.0)

    def test_apply_motor_noise_reports_zero_slip_probability_for_stay(self) -> None:
        world = SpiderWorld(
            seed=23,
            noise_profile=_minimal_noise_config(
                motor={
                    "action_flip_prob": 1.0,
                    "orientation_slip_factor": 1.0,
                    "terrain_slip_factor": 1.0,
                    "fatigue_slip_factor": 1.0,
                }
            ),
            lizard_move_interval=999999,
        )

        result = apply_motor_noise(world, "STAY")

        self.assertFalse(result["occurred"])
        self.assertEqual(result["reason"], "none")
        self.assertEqual(result["executed_action"], "STAY")
        self.assertAlmostEqual(result["slip_probability"], 0.0)


class ExecutionDifficultyTest(unittest.TestCase):
    def test_open_aligned_movement_has_zero_difficulty(self) -> None:
        _assert_execution_difficulty(
            self,
            heading=(1.0, 0.0),
            intended_direction=(1.0, 0.0),
            terrain=OPEN,
            fatigue=0.0,
            expected_difficulty=0.0,
            expected_components={
                "orientation_alignment": 1.0,
                "terrain_difficulty": 0.0,
            },
        )

    def test_stay_has_no_orientation_mismatch(self) -> None:
        _assert_execution_difficulty(
            self,
            heading=(1.0, 0.0),
            intended_direction=(0.0, 0.0),
            terrain=CLUTTER,
            fatigue=1.0,
            expected_difficulty=0.0,
            expected_components={
                "orientation_alignment": 1.0,
            },
        )

    def test_high_momentum_reduces_difficulty_for_aligned_move(self) -> None:
        low_momentum, low_components = compute_execution_difficulty(
            heading=(1.0, 1.0),
            intended_direction=(1.0, 0.0),
            terrain=NARROW,
            fatigue=0.0,
            momentum=0.0,
        )
        high_momentum, high_components = compute_execution_difficulty(
            heading=(1.0, 1.0),
            intended_direction=(1.0, 0.0),
            terrain=NARROW,
            fatigue=0.0,
            momentum=0.8,
        )

        self.assertGreater(low_components["orientation_alignment"], 0.7)
        self.assertAlmostEqual(high_components["momentum"], 0.8)
        self.assertLess(high_momentum, low_momentum)

    def test_momentum_does_not_reduce_difficulty_for_misaligned_move(self) -> None:
        no_momentum, _ = compute_execution_difficulty(
            heading=(-1.0, 0.0),
            intended_direction=(1.0, 0.0),
            terrain=NARROW,
            fatigue=0.0,
            momentum=0.0,
        )
        high_momentum, components = compute_execution_difficulty(
            heading=(-1.0, 0.0),
            intended_direction=(1.0, 0.0),
            terrain=NARROW,
            fatigue=0.0,
            momentum=0.8,
        )

        self.assertLessEqual(components["orientation_alignment"], 0.7)
        self.assertAlmostEqual(high_momentum, no_momentum)

    def test_momentum_component_in_difficulty_diagnostics(self) -> None:
        _, components = compute_execution_difficulty(
            heading=(1.0, 0.0),
            intended_direction=(1.0, 0.0),
            terrain=OPEN,
            fatigue=0.0,
            momentum=0.65,
        )

        self.assertIn("momentum", components)
        self.assertAlmostEqual(components["momentum"], 0.65)


class MotorExecutionSlipMechanismTest(unittest.TestCase):
    def _slip_profile(self, *, base: float = 0.0) -> NoiseConfig:
        return NoiseConfig(
            name="action_center_slip_test",
            visual={"certainty_jitter": 0.0, "direction_jitter": 0.0, "dropout_prob": 0.0},
            olfactory={"strength_jitter": 0.0, "direction_jitter": 0.0},
            motor={
                "action_flip_prob": base,
                "orientation_slip_factor": 0.2,
                "terrain_slip_factor": 0.4,
                "fatigue_slip_factor": 0.2,
            },
            spawn={"uniform_mix": 0.0},
            predator={"random_choice_prob": 0.0},
        )

    def test_slip_probability_increases_with_execution_difficulty(self) -> None:
        world = SpiderWorld(
            seed=211,
            noise_profile=self._slip_profile(),
            lizard_move_interval=999999,
        )
        terrain, pos = _terrain_with_cleanup(self, world)

        easy = _compute_slip_and_difficulty(
            world,
            terrain,
            pos,
            OPEN,
            heading=(1.0, 0.0),
            fatigue=0.0,
        )
        hard = _compute_slip_and_difficulty(
            world,
            terrain,
            pos,
            NARROW,
            heading=(-1.0, 0.0),
            fatigue=1.0,
        )

        self.assertLess(easy["slip_probability"], hard["slip_probability"])
        self.assertLess(easy["execution_difficulty"], hard["execution_difficulty"])

    def test_apply_motor_noise_passes_world_momentum_to_components(self) -> None:
        world = SpiderWorld(
            seed=213,
            noise_profile=self._slip_profile(),
            lizard_move_interval=999999,
        )
        terrain, pos = _terrain_with_cleanup(self, world)

        result = _compute_slip_and_difficulty(
            world,
            terrain,
            pos,
            NARROW,
            heading=(1.0, 1.0),
            fatigue=0.0,
            momentum=0.75,
        )

        self.assertAlmostEqual(result["components"]["momentum"], 0.75)

    def test_slip_sampler_deviates_to_adjacent_not_reverse(self) -> None:
        adjacent_actions = SLIP_ADJACENT_ACTIONS["MOVE_RIGHT"]
        reverse_action = "MOVE_LEFT"

        for index in range(1, len(adjacent_actions) + 1):
            with self.subTest(index=index):
                action = sample_slip_action("MOVE_RIGHT", _FakeChoiceMotorRng(index))
                self.assertIn(action, adjacent_actions)
                self.assertNotEqual(action, reverse_action)


class OrientActionHeadingTest(unittest.TestCase):
    """Subsystem checks for orientation actions handled by motor noise."""

    def test_orient_action_motor_slip_result_has_zero_probability(self) -> None:
        world = SpiderWorld(seed=81, lizard_move_interval=999999)

        for action_name in ("ORIENT_UP", "ORIENT_DOWN", "ORIENT_LEFT", "ORIENT_RIGHT"):
            with self.subTest(action=action_name):
                result = apply_motor_noise(world, action_name)
                self.assertAlmostEqual(result["slip_probability"], 0.0)
                self.assertFalse(result["occurred"])
                self.assertEqual(result["original_action"], action_name)
                self.assertEqual(result["executed_action"], action_name)

    def test_orient_action_motor_noise_result_reason_is_none(self) -> None:
        world = SpiderWorld(seed=81, lizard_move_interval=999999)

        result = apply_motor_noise(world, "ORIENT_LEFT")

        self.assertEqual(result["reason"], "none")


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
        cfg = _minimal_noise_config(
            motor={
                "action_flip_prob": 1,
                "orientation_slip_factor": 0,
                "terrain_slip_factor": 0,
                "fatigue_slip_factor": 0,
            }
        )
        for value in cfg.motor.values():
            self.assertIsInstance(value, float)

    def test_motor_slip_factors_default_to_zero_for_legacy_configs(self) -> None:
        cfg = _minimal_noise_config(motor={"action_flip_prob": 0.25})
        self.assertAlmostEqual(cfg.motor["action_flip_prob"], 0.25)
        self.assertAlmostEqual(cfg.motor["orientation_slip_factor"], 0.0)
        self.assertAlmostEqual(cfg.motor["terrain_slip_factor"], 0.0)
        self.assertAlmostEqual(cfg.motor["fatigue_slip_factor"], 0.0)

    def test_spawn_values_are_coerced_to_float(self) -> None:
        cfg = _minimal_noise_config(spawn={"uniform_mix": 1})
        for value in cfg.spawn.values():
            self.assertIsInstance(value, float)

    def test_predator_values_are_coerced_to_float(self) -> None:
        cfg = _minimal_noise_config(predator={"random_choice_prob": 1})
        for value in cfg.predator.values():
            self.assertIsInstance(value, float)

    def test_delay_values_are_coerced_to_float(self) -> None:
        cfg = _minimal_noise_config(
            delay={"certainty_decay_per_tick": 1, "direction_jitter_per_tick": 0}
        )
        for value in cfg.delay.values():
            self.assertIsInstance(value, float)

    def test_delay_defaults_to_zero_for_legacy_configs(self) -> None:
        cfg = _minimal_noise_config()
        self.assertAlmostEqual(cfg.delay["certainty_decay_per_tick"], 0.0)
        self.assertAlmostEqual(cfg.delay["direction_jitter_per_tick"], 0.0)

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
        for key in (
            "name",
            "visual",
            "olfactory",
            "motor",
            "spawn",
            "predator",
            "delay",
            "robustness_order",
        ):
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
        self.assertAlmostEqual(summary["motor"]["orientation_slip_factor"], 0.0)
        self.assertAlmostEqual(summary["motor"]["terrain_slip_factor"], 0.0)
        self.assertAlmostEqual(summary["motor"]["fatigue_slip_factor"], 0.0)
        self.assertAlmostEqual(summary["spawn"]["uniform_mix"], 0.0)
        self.assertAlmostEqual(summary["predator"]["random_choice_prob"], 0.0)
        self.assertAlmostEqual(summary["delay"]["certainty_decay_per_tick"], 0.0)
        self.assertAlmostEqual(summary["delay"]["direction_jitter_per_tick"], 0.0)

    def test_to_summary_includes_canonical_robustness_order(self) -> None:
        """
        Asserts that canonical noise profiles expose a monotonic `robustness_order` mapping.
        
        Verifies that the canonical profiles map to the expected integer orders: "none" → 0, "low" → 1, "medium" → 2, "high" → 3.
        """
        self.assertEqual(NONE_NOISE_PROFILE.to_summary()["robustness_order"], 0)
        self.assertEqual(LOW_NOISE_PROFILE.to_summary()["robustness_order"], 1)
        self.assertEqual(MEDIUM_NOISE_PROFILE.to_summary()["robustness_order"], 2)
        self.assertEqual(HIGH_NOISE_PROFILE.to_summary()["robustness_order"], 3)

    def test_to_summary_uses_minus_one_for_noncanonical_profile(self) -> None:
        summary = _minimal_noise_config(name="custom_noise").to_summary()
        self.assertEqual(summary["robustness_order"], -1)


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
            delay={"certainty_decay_per_tick": 0.04, "direction_jitter_per_tick": 0.02},
        )
        rebuilt = NoiseConfig.from_summary(cfg.to_summary())
        self.assertEqual(rebuilt.name, "custom_round_trip")
        self.assertAlmostEqual(rebuilt.visual["certainty_jitter"], 0.11)
        self.assertAlmostEqual(rebuilt.motor["action_flip_prob"], 0.07)
        self.assertAlmostEqual(rebuilt.motor["orientation_slip_factor"], 0.0)
        self.assertAlmostEqual(rebuilt.spawn["uniform_mix"], 0.25)
        self.assertAlmostEqual(rebuilt.predator["random_choice_prob"], 0.33)
        self.assertAlmostEqual(rebuilt.delay["certainty_decay_per_tick"], 0.04)
        self.assertAlmostEqual(rebuilt.delay["direction_jitter_per_tick"], 0.02)

    def test_from_summary_missing_delay_defaults_to_zero(self) -> None:
        summary = NONE_NOISE_PROFILE.to_summary()
        del summary["delay"]
        rebuilt = NoiseConfig.from_summary(summary)
        self.assertAlmostEqual(rebuilt.delay["certainty_decay_per_tick"], 0.0)
        self.assertAlmostEqual(rebuilt.delay["direction_jitter_per_tick"], 0.0)

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

    def test_from_summary_non_mapping_delay_raises_value_error(self) -> None:
        """Reject non-mapping delay sections."""
        summary = NONE_NOISE_PROFILE.to_summary()
        summary["delay"] = 0.5  # type: ignore[assignment]
        with self.assertRaises(ValueError) as ctx:
            NoiseConfig.from_summary(summary)
        self.assertIn("delay", str(ctx.exception))

    def test_from_summary_missing_visual_channel_key_raises_value_error(self) -> None:
        """Reject visual summaries missing dropout_prob."""
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


class RobustnessMatrixSpecTest(unittest.TestCase):
    def test_matrix_spec_coerces_condition_names_to_strings(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=("none", 1),  # type: ignore[arg-type]
            eval_conditions=("medium", 2),  # type: ignore[arg-type]
        )
        self.assertEqual(spec.train_conditions, ("none", "1"))
        self.assertEqual(spec.eval_conditions, ("medium", "2"))

    def test_matrix_spec_is_frozen(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=("none",),
            eval_conditions=("low",),
        )
        with self.assertRaises(FrozenInstanceError):
            spec.train_conditions = ("low",)  # type: ignore[misc]

    def test_matrix_spec_rejects_empty_train_condition_names(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            RobustnessMatrixSpec(
                train_conditions=("none", ""),
                eval_conditions=("low",),
            )
        self.assertIn("train_conditions", str(ctx.exception))

    def test_matrix_spec_strips_whitespace_before_validation(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=(" none ", " low\t"),
            eval_conditions=(" medium ",),
        )
        self.assertEqual(spec.train_conditions, ("none", "low"))
        self.assertEqual(spec.eval_conditions, ("medium",))

    def test_matrix_spec_rejects_duplicate_names_after_stripping(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            RobustnessMatrixSpec(
                train_conditions=("none", " none "),
                eval_conditions=("low",),
            )
        self.assertIn("train_conditions", str(ctx.exception))
        self.assertIn("'none'", str(ctx.exception))

    def test_matrix_spec_rejects_duplicate_eval_condition_names(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            RobustnessMatrixSpec(
                train_conditions=("none",),
                eval_conditions=("low", "low"),
            )
        self.assertIn("eval_conditions", str(ctx.exception))

    def test_canonical_conditions_match_expected_order(self) -> None:
        self.assertEqual(
            CANONICAL_ROBUSTNESS_CONDITIONS,
            ("none", "low", "medium", "high"),
        )

    def test_canonical_conditions_have_monotone_robustness_order(self) -> None:
        """
        Assert that the canonical robustness condition names map to robustness orders 0, 1, 2, 3 and that those orders are in ascending order.
        """
        orders = [
            resolve_noise_profile(name).to_summary()["robustness_order"]
            for name in CANONICAL_ROBUSTNESS_CONDITIONS
        ]
        self.assertEqual(orders, [0, 1, 2, 3])
        self.assertEqual(orders, sorted(orders))

    def test_matrix_spec_cells_returns_cartesian_product(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=("none", "low"),
            eval_conditions=("medium", "high"),
        )
        self.assertEqual(
            list(spec.cells()),
            [
                ("none", "medium"),
                ("none", "high"),
                ("low", "medium"),
                ("low", "high"),
            ],
        )

    def test_canonical_robustness_matrix_is_four_by_four(self) -> None:
        spec = canonical_robustness_matrix()
        self.assertEqual(spec.train_conditions, CANONICAL_ROBUSTNESS_CONDITIONS)
        self.assertEqual(spec.eval_conditions, CANONICAL_ROBUSTNESS_CONDITIONS)
        self.assertEqual(len(list(spec.cells())), 16)

    def test_matrix_spec_to_summary_includes_dimensions_and_cells(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=("none", "low"),
            eval_conditions=("medium",),
        )
        summary = spec.to_summary()
        self.assertEqual(summary["train_conditions"], ["none", "low"])
        self.assertEqual(summary["eval_conditions"], ["medium"])
        self.assertEqual(summary["cell_count"], 2)
        self.assertEqual(
            summary["cells"],
            [
                {
                    "train_noise_profile": "none",
                    "eval_noise_profile": "medium",
                },
                {
                    "train_noise_profile": "low",
                    "eval_noise_profile": "medium",
                },
            ],
        )


class RobustnessMatrixSpecEdgeCasesTest(unittest.TestCase):
    """Additional edge case and regression tests for RobustnessMatrixSpec (new in this PR)."""

    def test_empty_train_conditions_yields_no_cells(self) -> None:
        spec = RobustnessMatrixSpec(train_conditions=(), eval_conditions=("none",))
        self.assertEqual(list(spec.cells()), [])

    def test_empty_eval_conditions_yields_no_cells(self) -> None:
        spec = RobustnessMatrixSpec(train_conditions=("none",), eval_conditions=())
        self.assertEqual(list(spec.cells()), [])

    def test_both_empty_yields_no_cells(self) -> None:
        spec = RobustnessMatrixSpec(train_conditions=(), eval_conditions=())
        self.assertEqual(list(spec.cells()), [])

    def test_single_condition_yields_one_cell(self) -> None:
        spec = RobustnessMatrixSpec(train_conditions=("none",), eval_conditions=("none",))
        cells = list(spec.cells())
        self.assertEqual(cells, [("none", "none")])

    def test_to_summary_cell_count_zero_when_conditions_are_empty(self) -> None:
        spec = RobustnessMatrixSpec(train_conditions=(), eval_conditions=())
        summary = spec.to_summary()
        self.assertEqual(summary["cell_count"], 0)
        self.assertEqual(summary["cells"], [])

    def test_to_summary_is_json_serializable(self) -> None:
        spec = canonical_robustness_matrix()
        summary = spec.to_summary()
        serialized = json.dumps(summary)
        self.assertIsInstance(serialized, str)

    def test_cells_iteration_order_is_train_outer_eval_inner(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=("none", "low", "medium"),
            eval_conditions=("high", "none"),
        )
        cells = list(spec.cells())
        expected = [
            ("none", "high"),
            ("none", "none"),
            ("low", "high"),
            ("low", "none"),
            ("medium", "high"),
            ("medium", "none"),
        ]
        self.assertEqual(cells, expected)

    def test_non_string_conditions_coerced_via_str(self) -> None:
        spec = RobustnessMatrixSpec(
            train_conditions=(0, 1.5),
            eval_conditions=(True, False),
        )
        self.assertEqual(spec.train_conditions, ("0", "1.5"))
        self.assertEqual(spec.eval_conditions, ("True", "False"))

    def test_canonical_robustness_matrix_cell_count_is_sixteen(self) -> None:
        spec = canonical_robustness_matrix()
        summary = spec.to_summary()
        self.assertEqual(summary["cell_count"], 16)
        self.assertEqual(len(summary["cells"]), 16)

    def test_robustness_order_in_to_summary_is_integer(self) -> None:
        for profile in (NONE_NOISE_PROFILE, LOW_NOISE_PROFILE, MEDIUM_NOISE_PROFILE, HIGH_NOISE_PROFILE):
            order = profile.to_summary()["robustness_order"]
            self.assertIsInstance(order, int)

    def test_canonical_noise_profile_names_equals_canonical_robustness_conditions(self) -> None:
        self.assertEqual(canonical_noise_profile_names(), CANONICAL_ROBUSTNESS_CONDITIONS)

    def test_resolve_noise_profile_with_each_canonical_name(self) -> None:
        """All canonical condition names should resolve without raising."""
        for name in CANONICAL_ROBUSTNESS_CONDITIONS:
            profile = resolve_noise_profile(name)
            self.assertEqual(profile.name, name)

    def test_resolve_noise_profile_unknown_string_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            resolve_noise_profile("unknown_profile_xyz")


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

    def test_motor_slip_factors_are_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.motor["orientation_slip_factor"], 0.0)
        self.assertAlmostEqual(NONE_NOISE_PROFILE.motor["terrain_slip_factor"], 0.0)
        self.assertAlmostEqual(NONE_NOISE_PROFILE.motor["fatigue_slip_factor"], 0.0)

    def test_spawn_uniform_mix_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.spawn["uniform_mix"], 0.0)

    def test_predator_random_choice_prob_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.predator["random_choice_prob"], 0.0)

    def test_delay_noise_is_zero(self) -> None:
        self.assertAlmostEqual(NONE_NOISE_PROFILE.delay["certainty_decay_per_tick"], 0.0)
        self.assertAlmostEqual(NONE_NOISE_PROFILE.delay["direction_jitter_per_tick"], 0.0)


class LowNoiseProfileValuesTest(unittest.TestCase):
    def test_name_is_low_string(self) -> None:
        self.assertEqual(LOW_NOISE_PROFILE.name, "low")

    def test_visual_certainty_jitter_positive(self) -> None:
        self.assertGreater(LOW_NOISE_PROFILE.visual["certainty_jitter"], 0.0)

    def test_motor_action_flip_prob_positive(self) -> None:
        self.assertGreater(LOW_NOISE_PROFILE.motor["action_flip_prob"], 0.0)

    def test_motor_slip_factors_positive(self) -> None:
        self.assertGreater(LOW_NOISE_PROFILE.motor["orientation_slip_factor"], 0.0)
        self.assertGreater(LOW_NOISE_PROFILE.motor["terrain_slip_factor"], 0.0)
        self.assertGreater(LOW_NOISE_PROFILE.motor["fatigue_slip_factor"], 0.0)

    def test_predator_random_choice_prob_positive(self) -> None:
        self.assertGreater(LOW_NOISE_PROFILE.predator["random_choice_prob"], 0.0)

    def test_all_values_below_one(self) -> None:
        for section in (LOW_NOISE_PROFILE.visual, LOW_NOISE_PROFILE.olfactory,
                        LOW_NOISE_PROFILE.motor, LOW_NOISE_PROFILE.spawn,
                        LOW_NOISE_PROFILE.predator, LOW_NOISE_PROFILE.delay):
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

    def test_motor_slip_factors_monotone(self) -> None:
        for key in ("orientation_slip_factor", "terrain_slip_factor", "fatigue_slip_factor"):
            with self.subTest(key=key):
                self.assertLess(NONE_NOISE_PROFILE.motor[key], LOW_NOISE_PROFILE.motor[key])
                self.assertLess(LOW_NOISE_PROFILE.motor[key], MEDIUM_NOISE_PROFILE.motor[key])
                self.assertLess(MEDIUM_NOISE_PROFILE.motor[key], HIGH_NOISE_PROFILE.motor[key])

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

    def test_delay_noise_monotone(self) -> None:
        """
        Asserts that each delay-related noise parameter increases monotonically across the profiles NONE < LOW < MEDIUM < HIGH.
        
        Checks the keys "certainty_decay_per_tick" and "direction_jitter_per_tick" to ensure NONE < LOW < MEDIUM < HIGH for each.
        """
        for key in ("certainty_decay_per_tick", "direction_jitter_per_tick"):
            with self.subTest(key=key):
                self.assertLess(NONE_NOISE_PROFILE.delay[key], LOW_NOISE_PROFILE.delay[key])
                self.assertLess(LOW_NOISE_PROFILE.delay[key], MEDIUM_NOISE_PROFILE.delay[key])
                self.assertLess(MEDIUM_NOISE_PROFILE.delay[key], HIGH_NOISE_PROFILE.delay[key])


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


class NoiseConfigDelayImmutabilityTest(unittest.TestCase):
    def test_delay_mapping_is_immutable(self) -> None:
        cfg = _minimal_noise_config(
            delay={"certainty_decay_per_tick": 0.05, "direction_jitter_per_tick": 0.02}
        )
        with self.assertRaises(TypeError):
            cfg.delay["certainty_decay_per_tick"] = 0.99  # type: ignore[index]

    def test_to_summary_delay_mutation_does_not_affect_config(self) -> None:
        cfg = _minimal_noise_config(
            delay={"certainty_decay_per_tick": 0.07, "direction_jitter_per_tick": 0.03}
        )
        summary = cfg.to_summary()
        summary["delay"]["certainty_decay_per_tick"] = 999.0  # type: ignore[index]
        self.assertAlmostEqual(cfg.delay["certainty_decay_per_tick"], 0.07)

    def test_delay_section_is_plain_dict_in_summary(self) -> None:
        cfg = _minimal_noise_config(
            delay={"certainty_decay_per_tick": 0.0, "direction_jitter_per_tick": 0.0}
        )
        summary = cfg.to_summary()
        self.assertIsInstance(summary["delay"], dict)


class NoiseProfileDelayValuesTest(unittest.TestCase):
    def test_low_profile_delay_certainty_decay_is_positive(self) -> None:
        self.assertGreater(LOW_NOISE_PROFILE.delay["certainty_decay_per_tick"], 0.0)

    def test_low_profile_delay_direction_jitter_is_positive(self) -> None:
        self.assertGreater(LOW_NOISE_PROFILE.delay["direction_jitter_per_tick"], 0.0)

    def test_medium_profile_delay_certainty_decay_positive(self) -> None:
        self.assertGreater(MEDIUM_NOISE_PROFILE.delay["certainty_decay_per_tick"], 0.0)

    def test_high_profile_delay_certainty_decay_positive(self) -> None:
        self.assertGreater(HIGH_NOISE_PROFILE.delay["certainty_decay_per_tick"], 0.0)

    def test_high_profile_delay_direction_jitter_positive(self) -> None:
        self.assertGreater(HIGH_NOISE_PROFILE.delay["direction_jitter_per_tick"], 0.0)

    def test_all_profile_delay_values_are_floats(self) -> None:
        for profile in (NONE_NOISE_PROFILE, LOW_NOISE_PROFILE, MEDIUM_NOISE_PROFILE, HIGH_NOISE_PROFILE):
            for key, val in profile.delay.items():
                self.assertIsInstance(
                    val, float, f"Profile {profile.name!r} delay[{key!r}] is not float"
                )

    def test_all_profile_delay_values_below_one(self) -> None:
        for profile in (LOW_NOISE_PROFILE, MEDIUM_NOISE_PROFILE, HIGH_NOISE_PROFILE):
            for key, val in profile.delay.items():
                self.assertLessEqual(
                    val, 1.0, f"Profile {profile.name!r} delay[{key!r}]={val} exceeds 1.0"
                )

    def test_from_summary_preserves_delay_values_on_round_trip(self) -> None:
        cfg = _minimal_noise_config(
            delay={"certainty_decay_per_tick": 0.08, "direction_jitter_per_tick": 0.06}
        )
        rebuilt = NoiseConfig.from_summary(cfg.to_summary())
        self.assertAlmostEqual(rebuilt.delay["certainty_decay_per_tick"], 0.08)
        self.assertAlmostEqual(rebuilt.delay["direction_jitter_per_tick"], 0.06)

    def test_none_profile_delay_has_required_keys(self) -> None:
        self.assertIn("certainty_decay_per_tick", NONE_NOISE_PROFILE.delay)
        self.assertIn("direction_jitter_per_tick", NONE_NOISE_PROFILE.delay)

    def test_low_profile_delay_has_required_keys(self) -> None:
        self.assertIn("certainty_decay_per_tick", LOW_NOISE_PROFILE.delay)
        self.assertIn("direction_jitter_per_tick", LOW_NOISE_PROFILE.delay)


if __name__ == "__main__":
    unittest.main()
