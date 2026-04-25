from __future__ import annotations

import unittest

from spider_cortex_sim.ablations import BrainAblationConfig
from spider_cortex_sim.capacity_profiles import (
    CAPACITY_PROFILES,
    CapacityProfile,
    canonical_capacity_profile_names,
    capacity_profile_for_axis,
    resolve_capacity_profile,
)
from spider_cortex_sim.agent import SpiderBrain
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

    def test_partial_axis_hidden_dims_require_legacy_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "axis-specific hidden dims"):
            CapacityProfile(
                name="partial",
                version="v-test",
                module_hidden_dims=CAPACITY_PROFILES["current"].module_hidden_dims,
                action_center_hidden_dim=32,
                arbitration_hidden_dim=32,
                scale_factor=1.0,
            )

    def test_legacy_hidden_dim_cannot_mix_with_axis_hidden_dims(self) -> None:
        with self.assertRaisesRegex(ValueError, "Do not mix integration_hidden_dim"):
            CapacityProfile(
                name="mixed",
                version="v-test",
                module_hidden_dims=CAPACITY_PROFILES["current"].module_hidden_dims,
                action_center_hidden_dim=32,
                integration_hidden_dim=32,
                scale_factor=1.0,
            )

    def test_mixed_axis_summary_omits_legacy_integration_hidden_dim(self) -> None:
        mixed = capacity_profile_for_axis(
            axis="motor",
            target_profile="large",
            base_profile="current",
        )
        uniform = resolve_capacity_profile("large")

        self.assertNotIn("integration_hidden_dim", mixed.to_summary())
        self.assertEqual(
            uniform.to_summary()["integration_hidden_dim"],
            uniform.integration_hidden_dim,
        )

    def test_mixed_axis_profile_identity_includes_base_profile(self) -> None:
        from_small = capacity_profile_for_axis(
            axis="motor",
            target_profile="large",
            base_profile="small",
        )
        from_current = capacity_profile_for_axis(
            axis="motor",
            target_profile="large",
            base_profile="current",
        )

        self.assertEqual(from_small.name, "motor_large_from_small")
        self.assertEqual(from_current.name, "motor_large_from_current")
        self.assertNotEqual(from_small.to_summary(), from_current.to_summary())
        self.assertIn("base:small:v1", from_small.version)
        self.assertIn("base:current:v1", from_current.version)


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

    def test_capacity_profile_override_preserves_explicit_config_dims(self) -> None:
        config = BrainAblationConfig(
            capacity_profile="current",
            module_hidden_dims={"visual_cortex": 77},
            action_center_hidden_dim=41,
            arbitration_hidden_dim=42,
            motor_hidden_dim=43,
            integration_hidden_dim=None,
        )

        sim = SpiderSimulation(
            seed=12,
            max_steps=5,
            brain_config=config,
            capacity_profile="large",
        )
        summary = sim._build_summary([], [])
        architecture = sim.brain._architecture_signature()
        row = sim._annotate_behavior_rows([{"scenario": "night_rest"}])[0]

        self.assertEqual(summary["config"]["brain"]["capacity_profile"], "large")
        self.assertEqual(summary["config"]["capacity_profile"]["profile"], "large")
        self.assertEqual(
            summary["config"]["capacity_profile"]["module_hidden_dims"]["visual_cortex"],
            77,
        )
        self.assertEqual(
            summary["config"]["capacity_profile"]["module_hidden_dims"]["sensory_cortex"],
            CAPACITY_PROFILES["large"].module_hidden_dims["sensory_cortex"],
        )
        self.assertEqual(summary["config"]["brain"]["action_center_hidden_dim"], 41)
        self.assertEqual(summary["config"]["brain"]["arbitration_hidden_dim"], 42)
        self.assertEqual(summary["config"]["brain"]["motor_hidden_dim"], 43)
        self.assertEqual(
            summary["config"]["brain"]["integration_hidden_dim"],
            41,
        )
        self.assertEqual(
            architecture["capacity"]["module_hidden_dims"],
            summary["config"]["capacity_profile"]["module_hidden_dims"],
        )
        self.assertEqual(row["capacity_profile"], "large")
        self.assertEqual(row["action_center_hidden_dim"], 41)
        self.assertEqual(row["arbitration_hidden_dim"], 42)
        self.assertEqual(row["motor_hidden_dim"], 43)
        self.assertEqual(
            row["integration_hidden_dim"],
            41,
        )

    def test_direct_brain_accepts_mixed_capacity_profile_object(self) -> None:
        mixed = capacity_profile_for_axis(
            axis="motor",
            target_profile="large",
            base_profile="small",
        )

        brain = SpiderBrain(seed=21, capacity_profile=mixed)

        self.assertEqual(brain.config.capacity_profile_name, mixed.name)
        self.assertEqual(brain.config.motor_hidden_dim, mixed.motor_hidden_dim)
        self.assertEqual(
            brain.config.module_hidden_dims,
            dict(mixed.module_hidden_dims),
        )

    def test_name_only_brain_config_uses_capacity_profile_name_as_source(self) -> None:
        config = BrainAblationConfig(capacity_profile_name="small")
        # Simulate a legacy/name-only config: the constructor now always resolves
        # capacity_profile, but compatibility paths still need the name fallback.
        object.__setattr__(config, "module_hidden_dims", {
            **config.module_hidden_dims,
            "visual_cortex": 77,
        })
        object.__setattr__(config, "capacity_profile", None)

        brain = SpiderBrain(
            seed=23,
            config=config,
            capacity_profile="large",
        )

        self.assertEqual(brain.config.capacity_profile_name, "large")
        self.assertEqual(brain.config.module_hidden_dims["visual_cortex"], 77)
        self.assertEqual(
            brain.config.module_hidden_dims["sensory_cortex"],
            CAPACITY_PROFILES["large"].module_hidden_dims["sensory_cortex"],
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

    def test_capacity_axis_profile_keeps_base_scale_for_mixed_axes(self) -> None:
        profile = capacity_profile_for_axis(
            axis="action_center",
            target_profile="small",
            base_profile="current",
        )
        all_profile = capacity_profile_for_axis(
            axis="all",
            target_profile="small",
            base_profile="current",
        )

        self.assertEqual(profile.scale_factor, CAPACITY_PROFILES["current"].scale_factor)
        self.assertEqual(all_profile.scale_factor, CAPACITY_PROFILES["small"].scale_factor)

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
