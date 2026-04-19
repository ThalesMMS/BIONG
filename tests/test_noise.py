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

from tests.fixtures.noise import (
    _minimal_noise_config,
    _terrain_with_cleanup,
    _compute_slip_and_difficulty,
    _assert_execution_difficulty,
    _FakeChoiceMotorRng,
)

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
