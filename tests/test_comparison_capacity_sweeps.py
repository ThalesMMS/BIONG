from __future__ import annotations

import unittest

from spider_cortex_sim.comparison_capacity import (
    CAPACITY_SWEEP_VARIANT_NAMES,
    compare_capacity_axis_sweep,
    compare_capacity_sweep,
    compare_capacity_sweeps,
)


class CompareCapacitySweepsTest(unittest.TestCase):
    def test_empty_seeds_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty seeds"):
            compare_capacity_sweep(
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                seeds=(),
            )

    def test_capacity_sweep_returns_all_profiles_and_architectures(self) -> None:
        payload, rows = compare_capacity_sweep(
            episodes=0,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
        )

        self.assertEqual(
            [item["profile"] for item in payload["capacity_profiles"]],
            ["small", "current", "medium", "large"],
        )
        self.assertEqual(
            payload["architecture_variants"],
            list(CAPACITY_SWEEP_VARIANT_NAMES),
        )
        for profile_name in ("small", "current", "medium", "large"):
            self.assertIn(profile_name, payload["profiles"])
            variant_names = set(payload["profiles"][profile_name]["variants"].keys())
            self.assertEqual(variant_names, set(CAPACITY_SWEEP_VARIANT_NAMES))
        self.assertTrue(rows)
        for row in rows:
            self.assertIn(row["capacity_profile"], {"small", "current", "medium", "large"})
        self.assertIn("performance_matrices", payload)
        self.assertEqual(
            set(payload["performance_matrices"]["scenario_success_rate"].keys()),
            set(CAPACITY_SWEEP_VARIANT_NAMES),
        )

    def test_plural_name_remains_compatible_alias(self) -> None:
        payload, rows = compare_capacity_sweeps(
            episodes=0,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
        )

        self.assertEqual(payload["architecture_variants"], list(CAPACITY_SWEEP_VARIANT_NAMES))
        self.assertTrue(rows)

    def test_duplicate_profile_names_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique capacity profiles"):
            compare_capacity_sweep(
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                seeds=(7,),
                capacity_profiles=("small", "small", "current"),
            )

    def test_explicit_empty_scenarios_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "No scenario names resolved"):
            compare_capacity_sweep(
                episodes=0,
                evaluation_episodes=0,
                names=(),
                architecture_variants=(),
                capacity_profiles=(),
                seeds=(7,),
            )

    def test_explicit_empty_variants_and_profiles_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "No architecture variants or capacity profiles resolved",
        ):
            compare_capacity_sweep(
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                architecture_variants=(),
                capacity_profiles=(),
                seeds=(7,),
            )

    def test_single_canonical_variant_and_profile_flow(self) -> None:
        payload, rows = compare_capacity_sweep(
            episodes=0,
            evaluation_episodes=0,
            names=("night_rest",),
            architecture_variants=("modular_full",),
            capacity_profiles=("current",),
            seeds=(7,),
        )

        self.assertEqual(payload["architecture_variants"], ["modular_full"])
        self.assertIn("modular_full", payload["profiles"]["current"]["variants"])
        self.assertTrue(rows)

    def test_duplicate_architecture_variants_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique architecture variants"):
            compare_capacity_sweep(
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                seeds=(7,),
                architecture_variants=("modular_full", "modular_full"),
            )

    def test_capacity_axis_sweep_annotates_axis_and_profiles(self) -> None:
        payload, rows = compare_capacity_axis_sweep(
            episodes=0,
            evaluation_episodes=0,
            names=("night_rest",),
            seeds=(7,),
            capacity_axes=("action_center",),
            capacity_profiles=("small",),
            architecture_variants=("modular_full",),
        )

        self.assertEqual(payload["capacity_axes"], ["action_center"])
        self.assertIn("action_center", payload["axes"])
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["capacity_axis"], "action_center")
            self.assertEqual(row["capacity_profile"], "action_center_small_from_current")
        profile = payload["axes"]["action_center"]["capacity_profiles"][0]
        self.assertEqual(profile["profile"], "action_center_small_from_current")
        self.assertEqual(profile["action_center_hidden_dim"], 20)
        self.assertEqual(profile["arbitration_hidden_dim"], 32)
        self.assertEqual(profile["motor_hidden_dim"], 32)
        self.assertEqual(profile["scale_factor"], 1.0)

    def test_capacity_axis_sweep_rejects_duplicate_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique capacity axes"):
            compare_capacity_axis_sweep(
                episodes=0,
                evaluation_episodes=0,
                names=("night_rest",),
                seeds=(7,),
                capacity_axes=("motor", "motor"),
            )


if __name__ == "__main__":
    unittest.main()
