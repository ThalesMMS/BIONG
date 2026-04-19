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
