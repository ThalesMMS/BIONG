from __future__ import annotations

import unittest

from spider_cortex_sim.ablations import BrainAblationConfig
from spider_cortex_sim.capacity_profiles import (
    CAPACITY_PROFILES,
    CapacityProfile,
    canonical_capacity_profile_names,
    resolve_capacity_profile,
)
from spider_cortex_sim.simulation import SpiderSimulation


class CapacityProfileRegistryTest(unittest.TestCase):
    def test_canonical_profile_names_are_stable(self) -> None:
        self.assertEqual(
            canonical_capacity_profile_names(),
            ("small", "current", "medium", "large"),
        )

    def test_resolve_none_returns_current_profile(self) -> None:
        self.assertEqual(resolve_capacity_profile(None).name, "current")

    def test_resolve_returns_defensive_immutable_profile(self) -> None:
        first = resolve_capacity_profile("current")
        second = resolve_capacity_profile("current")

        self.assertIsNot(first, second)
        with self.assertRaises(TypeError):
            first.module_hidden_dims["visual_cortex"] = 999

    def test_unknown_profile_raises_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown capacity profile"):
            resolve_capacity_profile("gigantic")


class CapacityConfigIntegrationTest(unittest.TestCase):
    def test_ablation_config_tracks_capacity_profile_and_hidden_dims(self) -> None:
        config = BrainAblationConfig(capacity_profile="large")
        summary = config.to_summary()

        self.assertEqual(summary["capacity_profile_name"], "large")
        self.assertEqual(summary["capacity_profile"], "large")
        self.assertEqual(summary["integration_hidden_dim"], 64)
        self.assertEqual(
            summary["module_hidden_dims"]["perception_center"],
            CAPACITY_PROFILES["large"].module_hidden_dims["perception_center"],
        )
        self.assertEqual(
            summary["monolithic_hidden_dim"],
            sum(summary["module_hidden_dims"].values()),
        )

    def test_simulation_summary_includes_capacity_and_compute_cost(self) -> None:
        sim = SpiderSimulation(
            seed=7,
            max_steps=5,
            brain_config=BrainAblationConfig(capacity_profile="small"),
        )
        summary = sim._build_summary([], [])
        architecture = sim.brain._architecture_signature()

        self.assertEqual(summary["config"]["brain"]["capacity_profile"], "small")
        self.assertEqual(summary["config"]["capacity_profile"]["profile"], "small")
        self.assertEqual(
            summary["config"]["capacity_profile"]["integration_hidden_dim"],
            20,
        )
        self.assertEqual(architecture["capacity_profile_name"], "small")
        self.assertIn("approximate_compute_cost", summary)
        self.assertEqual(
            summary["approximate_compute_cost"]["unit"],
            "approx_forward_macs",
        )
        self.assertGreater(summary["approximate_compute_cost"]["total"], 0)

    def test_simulation_summary_lists_frozen_modules_when_present(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        sim.brain.freeze_proposers(("alert_center", "sleep_center"))

        summary = sim._build_summary([], [])

        self.assertEqual(
            summary["config"]["frozen_modules"],
            ["alert_center", "sleep_center"],
        )

    def test_explicit_capacity_profile_override_updates_runtime_metadata(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=5,
            brain_config=BrainAblationConfig(capacity_profile="current"),
            capacity_profile="large",
        )
        summary = sim._build_summary([], [])
        architecture = sim.brain._architecture_signature()

        self.assertEqual(summary["config"]["capacity_profile"]["profile"], "large")
        self.assertEqual(summary["config"]["brain"]["capacity_profile"], "large")
        self.assertEqual(architecture["capacity_profile_name"], "large")
        self.assertEqual(
            summary["config"]["capacity_profile"]["integration_hidden_dim"],
            CAPACITY_PROFILES["large"].integration_hidden_dim,
        )
        self.assertEqual(
            summary["config"]["capacity_profile"]["module_hidden_dims"],
            dict(CAPACITY_PROFILES["large"].module_hidden_dims),
        )
        self.assertEqual(
            architecture["capacity"]["integration_hidden_dim"],
            CAPACITY_PROFILES["large"].integration_hidden_dim,
        )
        self.assertEqual(
            architecture["capacity"]["module_hidden_dims"],
            dict(CAPACITY_PROFILES["large"].module_hidden_dims),
        )
        self.assertEqual(
            summary["config"]["brain"]["monolithic_hidden_dim"],
            sum(CAPACITY_PROFILES["large"].module_hidden_dims.values()),
        )
        self.assertEqual(
            architecture["capacity"]["monolithic_hidden_dim"],
            sum(CAPACITY_PROFILES["large"].module_hidden_dims.values()),
        )

    def test_custom_capacity_profile_object_is_preserved_in_runtime_metadata(self) -> None:
        custom_profile = CapacityProfile(
            name="custom_profile",
            version="v-test",
            module_hidden_dims={
                "visual_cortex": 11,
                "sensory_cortex": 12,
                "hunger_center": 13,
                "sleep_center": 14,
                "alert_center": 15,
                "perception_center": 16,
                "homeostasis_center": 17,
                "threat_center": 18,
            },
            integration_hidden_dim=19,
            scale_factor=0.7,
        )
        sim = SpiderSimulation(
            seed=13,
            max_steps=5,
            brain_config=BrainAblationConfig(capacity_profile=custom_profile),
        )
        summary = sim._build_summary([], [])
        architecture = sim.brain._architecture_signature()

        self.assertEqual(summary["config"]["brain"]["capacity_profile"], "custom_profile")
        self.assertEqual(summary["config"]["brain"]["capacity_profile_name"], "custom_profile")
        self.assertEqual(summary["config"]["capacity_profile"]["profile"], "custom_profile")
        self.assertEqual(
            summary["config"]["capacity_profile"]["module_hidden_dims"],
            dict(custom_profile.module_hidden_dims),
        )
        self.assertEqual(
            summary["config"]["brain"]["monolithic_hidden_dim"],
            sum(custom_profile.module_hidden_dims.values()),
        )
        self.assertEqual(architecture["capacity_profile_name"], "custom_profile")
        self.assertEqual(
            architecture["capacity"]["module_hidden_dims"],
            dict(custom_profile.module_hidden_dims),
        )
        self.assertEqual(
            architecture["capacity"]["monolithic_hidden_dim"],
            sum(custom_profile.module_hidden_dims.values()),
        )

    def test_explicit_override_wins_over_custom_profile_object(self) -> None:
        custom_profile = CapacityProfile(
            name="custom_profile",
            version="v-test",
            module_hidden_dims={
                "visual_cortex": 11,
                "sensory_cortex": 12,
                "hunger_center": 13,
                "sleep_center": 14,
                "alert_center": 15,
                "perception_center": 16,
                "homeostasis_center": 17,
                "threat_center": 18,
            },
            integration_hidden_dim=19,
            scale_factor=0.7,
        )
        sim = SpiderSimulation(
            seed=17,
            max_steps=5,
            brain_config=BrainAblationConfig(capacity_profile=custom_profile),
            capacity_profile="small",
        )
        summary = sim._build_summary([], [])
        architecture = sim.brain._architecture_signature()

        self.assertEqual(summary["config"]["brain"]["capacity_profile"], "small")
        self.assertEqual(summary["config"]["capacity_profile"]["profile"], "small")
        self.assertEqual(
            summary["config"]["brain"]["monolithic_hidden_dim"],
            sum(CAPACITY_PROFILES["small"].module_hidden_dims.values()),
        )
        self.assertEqual(architecture["capacity_profile_name"], "small")
        self.assertEqual(
            architecture["capacity"]["monolithic_hidden_dim"],
            sum(CAPACITY_PROFILES["small"].module_hidden_dims.values()),
        )

    def test_from_summary_defaults_capacity_profile_name_to_current(self) -> None:
        rebuilt = BrainAblationConfig.from_summary(
            {
                "name": "legacy",
                "architecture": "modular",
            }
        )
        self.assertEqual(rebuilt.capacity_profile_name, "current")

    def test_from_summary_preserves_explicit_zero_values(self) -> None:
        rebuilt = BrainAblationConfig.from_summary(
            {
                "name": "zeroed",
                "architecture": "modular",
                "module_dropout": 0.0,
                "warm_start_scale": 0.0,
                "reflex_scale": 0.0,
            }
        )

        self.assertEqual(rebuilt.module_dropout, 0.0)
        self.assertEqual(rebuilt.warm_start_scale, 0.0)
        self.assertEqual(rebuilt.reflex_scale, 0.0)


if __name__ == "__main__":
    unittest.main()
