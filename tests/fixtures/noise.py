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


class _FakeChoiceMotorRng:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.count: int | None = None
        self.weights: list[float] = []

    def choice(self, count: int, p: Sequence[float]) -> int:
        self.count = count
        self.weights = [float(value) for value in p]
        return self.index
