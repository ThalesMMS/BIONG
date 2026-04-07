from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    MODULE_NAMES,
    canonical_ablation_configs,
    canonical_ablation_variant_names,
    default_brain_config,
    resolve_ablation_configs,
)
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.simulation import SpiderSimulation


def _blank_mapping(interface) -> dict[str, float]:
    return {
        signal.name: (1.0 if signal.name.endswith("_age") else 0.0)
        for signal in interface.inputs
    }


def _build_observation(**overrides: dict[str, float]) -> dict[str, np.ndarray]:
    """
    Builds a complete observation dict for all module, action-context, and motor-context interfaces, applying any per-interface signal overrides.
    
    Parameters:
        overrides (dict[str, dict[str, float]]): Optional per-interface overrides keyed by each interface's
            `observation_key`. Each value is a mapping from signal name to numeric value that will replace
            the default for that signal.
    
    Returns:
        dict[str, np.ndarray]: A mapping from each interface's `observation_key` to its vectorized observation
        (NumPy array).
    """
    observation: dict[str, np.ndarray] = {}
    for interface in MODULE_INTERFACES:
        mapping = _blank_mapping(interface)
        mapping.update({name: float(value) for name, value in overrides.get(interface.observation_key, {}).items()})
        observation[interface.observation_key] = interface.vector_from_mapping(mapping)
    action_mapping = _blank_mapping(ACTION_CONTEXT_INTERFACE)
    action_mapping.update({name: float(value) for name, value in overrides.get(ACTION_CONTEXT_INTERFACE.observation_key, {}).items()})
    observation[ACTION_CONTEXT_INTERFACE.observation_key] = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_mapping)
    motor_mapping = _blank_mapping(MOTOR_CONTEXT_INTERFACE)
    motor_mapping.update({name: float(value) for name, value in overrides.get(MOTOR_CONTEXT_INTERFACE.observation_key, {}).items()})
    observation[MOTOR_CONTEXT_INTERFACE.observation_key] = MOTOR_CONTEXT_INTERFACE.vector_from_mapping(motor_mapping)
    return observation


def _profile_with_updates(**reward_updates: float) -> OperationalProfile:
    summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
    summary["name"] = "ablation_test_profile"
    summary["version"] = 21
    summary["reward"].update({name: float(value) for name, value in reward_updates.items()})
    return OperationalProfile.from_summary(summary)


class BrainAblationBehaviorTest(unittest.TestCase):
    def _module_result(self, module_name: str, **signals: float) -> ModuleResult:
        interface = MODULE_INTERFACE_BY_NAME[module_name]
        mapping = _blank_mapping(interface)
        mapping.update({name: float(value) for name, value in signals.items()})
        observation = interface.vector_from_mapping(mapping)
        action_dim = len(LOCOMOTION_ACTIONS)
        return ModuleResult(
            interface=interface,
            name=module_name,
            observation_key=interface.observation_key,
            observation=observation,
            logits=np.zeros(action_dim, dtype=float),
            probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
            active=True,
        )

    def test_disabled_module_is_inactive_and_zeroed(self) -> None:
        config = BrainAblationConfig(
            name="drop_alert_center",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=True,
            enable_auxiliary_targets=True,
            disabled_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=7, max_steps=8, brain_config=config)
        observation = sim.world.reset(seed=7)

        decision = sim.brain.act(observation, sample=False)
        alert_result = next(result for result in decision.module_results if result.name == "alert_center")

        self.assertFalse(alert_result.active)
        self.assertIsNone(alert_result.reflex)
        np.testing.assert_allclose(alert_result.logits, np.zeros(len(LOCOMOTION_ACTIONS), dtype=float))
        np.testing.assert_allclose(
            alert_result.probs,
            np.full(len(LOCOMOTION_ACTIONS), 1.0 / len(LOCOMOTION_ACTIONS), dtype=float),
        )

    def test_reflex_disabled_leaves_reflex_none(self) -> None:
        """
        Ensures that turning off reflexes leaves the alert_center module's reflex unset.
        
        Constructs an observation containing predator-related signals, runs two SpiderBrain instances
        with identical seeds and dropout but with reflexes enabled vs. disabled, and asserts that
        the alert_center ModuleResult contains a reflex when enabled and `None` when reflexes are disabled.
        """
        observation = _build_observation(
            alert={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
                "predator_dx": 1.0,
                "predator_dy": 0.0,
                "predator_dist": 0.2,
            },
            action_context={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
                "predator_dist": 0.2,
            },
        )
        control = SpiderBrain(
            seed=11,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="with_reflexes",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=True,
                enable_auxiliary_targets=True,
                disabled_modules=(),
            ),
        )
        no_reflexes = SpiderBrain(
            seed=11,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="no_reflexes",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=True,
                disabled_modules=(),
            ),
        )

        control_result = next(result for result in control.act(observation, sample=False).module_results if result.name == "alert_center")
        ablated_result = next(result for result in no_reflexes.act(observation, sample=False).module_results if result.name == "alert_center")

        self.assertIsNotNone(control_result.reflex)
        self.assertIsNone(ablated_result.reflex)

    def test_auxiliary_targets_can_be_disabled_independently(self) -> None:
        brain = SpiderBrain(
            seed=3,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="no_aux",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=True,
                enable_auxiliary_targets=False,
                disabled_modules=(),
            ),
        )
        result = self._module_result(
            "alert_center",
            predator_visible=1.0,
            predator_certainty=0.8,
            predator_dx=1.0,
            predator_dy=0.0,
        )
        result.reflex = brain._module_reflex_decision(result)

        self.assertEqual(brain._auxiliary_module_gradients([result]), {})


class BrainAblationConfigValidationTest(unittest.TestCase):
    """Tests for BrainAblationConfig construction and validation."""

    def test_default_architecture_is_modular(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertEqual(config.architecture, "modular")

    def test_default_module_dropout_is_float(self) -> None:
        config = BrainAblationConfig(name="test", module_dropout=0.1)
        self.assertIsInstance(config.module_dropout, float)
        self.assertAlmostEqual(config.module_dropout, 0.1)

    def test_default_enable_reflexes_true(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertTrue(config.enable_reflexes)

    def test_default_enable_auxiliary_targets_true(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertTrue(config.enable_auxiliary_targets)

    def test_default_disabled_modules_empty(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertEqual(config.disabled_modules, ())

    def test_invalid_architecture_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(name="test", architecture="invalid_arch")

    def test_invalid_module_name_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="modular",
                disabled_modules=("nonexistent_module",),
            )

    def test_monolithic_with_disabled_modules_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="monolithic",
                disabled_modules=("alert_center",),
            )

    def test_disabled_modules_are_sorted_and_deduplicated(self) -> None:
        config = BrainAblationConfig(
            name="test",
            architecture="modular",
            disabled_modules=("sleep_center", "alert_center", "alert_center"),
        )
        self.assertEqual(config.disabled_modules, ("alert_center", "sleep_center"))

    def test_is_modular_property_true_for_modular(self) -> None:
        config = BrainAblationConfig(name="test", architecture="modular")
        self.assertTrue(config.is_modular)
        self.assertFalse(config.is_monolithic)

    def test_is_monolithic_property_true_for_monolithic(self) -> None:
        config = BrainAblationConfig(name="test", architecture="monolithic")
        self.assertTrue(config.is_monolithic)
        self.assertFalse(config.is_modular)

    def test_to_summary_returns_correct_keys(self) -> None:
        config = BrainAblationConfig(
            name="drop_visual_cortex",
            architecture="modular",
            module_dropout=0.03,
            enable_reflexes=True,
            enable_auxiliary_targets=False,
            disabled_modules=("visual_cortex",),
        )
        summary = config.to_summary()
        self.assertEqual(summary["name"], "drop_visual_cortex")
        self.assertEqual(summary["architecture"], "modular")
        self.assertAlmostEqual(summary["module_dropout"], 0.03)
        self.assertTrue(summary["enable_reflexes"])
        self.assertFalse(summary["enable_auxiliary_targets"])
        self.assertEqual(summary["disabled_modules"], ["visual_cortex"])

    def test_to_summary_disabled_modules_is_list(self) -> None:
        config = BrainAblationConfig(name="test", disabled_modules=("hunger_center",))
        summary = config.to_summary()
        self.assertIsInstance(summary["disabled_modules"], list)

    def test_frozen_dataclass_is_immutable(self) -> None:
        config = BrainAblationConfig(name="test")
        with self.assertRaises((AttributeError, TypeError)):
            config.name = "other"  # type: ignore[misc]

    def test_module_dropout_coerced_to_float(self) -> None:
        config = BrainAblationConfig(name="test", module_dropout=0)
        self.assertIsInstance(config.module_dropout, float)
        self.assertEqual(config.module_dropout, 0.0)

    def test_reflex_scale_default_is_one(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertAlmostEqual(config.reflex_scale, 1.0)

    def test_reflex_scale_coerced_to_float(self) -> None:
        config = BrainAblationConfig(name="test", reflex_scale=0)
        self.assertIsInstance(config.reflex_scale, float)
        self.assertEqual(config.reflex_scale, 0.0)

    def test_reflex_scale_negative_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(name="test", reflex_scale=-0.1)

    def test_reflex_scale_non_finite_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(name="test", reflex_scale=float("nan"))

    def test_module_dropout_non_finite_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(name="test", module_dropout=float("inf"))

    def test_reflex_scale_zero_accepted(self) -> None:
        config = BrainAblationConfig(name="test", reflex_scale=0.0)
        self.assertAlmostEqual(config.reflex_scale, 0.0)

    def test_reflex_scale_large_value_accepted(self) -> None:
        config = BrainAblationConfig(name="test", reflex_scale=2.5)
        self.assertAlmostEqual(config.reflex_scale, 2.5)

    def test_module_reflex_scales_default_is_empty(self) -> None:
        config = BrainAblationConfig(name="test")
        self.assertEqual(config.module_reflex_scales, {})

    def test_module_reflex_scales_valid_module_accepted(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={"alert_center": 0.5},
        )
        self.assertAlmostEqual(config.module_reflex_scales["alert_center"], 0.5)

    def test_module_reflex_scales_invalid_module_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                module_reflex_scales={"nonexistent_module": 0.5},
            )

    def test_module_reflex_scales_reject_non_reflex_module(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                module_reflex_scales={"motor_cortex_context": 0.5},
            )

    def test_module_reflex_scales_negative_value_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                module_reflex_scales={"alert_center": -0.1},
            )

    def test_module_reflex_scales_non_finite_value_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                module_reflex_scales={"alert_center": float("nan")},
            )

    def test_module_reflex_scales_zero_accepted(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={"alert_center": 0.0},
        )
        self.assertAlmostEqual(config.module_reflex_scales["alert_center"], 0.0)

    def test_monolithic_with_module_reflex_scales_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            BrainAblationConfig(
                name="test",
                architecture="monolithic",
                module_reflex_scales={"alert_center": 0.5},
            )

    def test_module_reflex_scales_are_sorted_by_key(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={
                "visual_cortex": 0.8,
                "alert_center": 0.5,
                "hunger_center": 0.3,
            },
        )
        keys = list(config.module_reflex_scales.keys())
        self.assertEqual(keys, sorted(keys))

    def test_to_summary_includes_reflex_scale(self) -> None:
        config = BrainAblationConfig(name="test", reflex_scale=0.75)
        summary = config.to_summary()
        self.assertIn("reflex_scale", summary)
        self.assertAlmostEqual(summary["reflex_scale"], 0.75)

    def test_to_summary_includes_module_reflex_scales(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={"alert_center": 0.5},
        )
        summary = config.to_summary()
        self.assertIn("module_reflex_scales", summary)
        self.assertIsInstance(summary["module_reflex_scales"], dict)
        self.assertAlmostEqual(summary["module_reflex_scales"]["alert_center"], 0.5)

    def test_to_summary_module_reflex_scales_is_plain_dict(self) -> None:
        config = BrainAblationConfig(name="test")
        summary = config.to_summary()
        self.assertIsInstance(summary["module_reflex_scales"], dict)

    def test_module_reflex_scales_values_coerced_to_float(self) -> None:
        config = BrainAblationConfig(
            name="test",
            module_reflex_scales={"alert_center": 1},
        )
        self.assertIsInstance(config.module_reflex_scales["alert_center"], float)


class DefaultBrainConfigTest(unittest.TestCase):
    """Tests for default_brain_config factory."""

    def test_default_brain_config_name(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.name, "modular_full")

    def test_default_brain_config_architecture(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.architecture, "modular")

    def test_default_brain_config_custom_dropout(self) -> None:
        config = default_brain_config(module_dropout=0.10)
        self.assertAlmostEqual(config.module_dropout, 0.10)

    def test_default_brain_config_reflexes_and_aux_enabled(self) -> None:
        config = default_brain_config()
        self.assertTrue(config.enable_reflexes)
        self.assertTrue(config.enable_auxiliary_targets)

    def test_default_brain_config_no_disabled_modules(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.disabled_modules, ())

    def test_default_brain_config_reflex_scale_is_one(self) -> None:
        config = default_brain_config()
        self.assertAlmostEqual(config.reflex_scale, 1.0)

    def test_default_brain_config_module_reflex_scales_is_empty(self) -> None:
        config = default_brain_config()
        self.assertEqual(config.module_reflex_scales, {})


class CanonicalAblationConfigsTest(unittest.TestCase):
    """Tests for canonical_ablation_configs and canonical_ablation_variant_names."""

    def test_contains_modular_full(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("modular_full", configs)

    def test_contains_no_module_dropout(self) -> None:
        configs = canonical_ablation_configs()
        no_drop = configs["no_module_dropout"]
        self.assertEqual(no_drop.module_dropout, 0.0)
        self.assertTrue(no_drop.enable_reflexes)

    def test_contains_no_module_reflexes(self) -> None:
        configs = canonical_ablation_configs()
        no_reflex = configs["no_module_reflexes"]
        self.assertFalse(no_reflex.enable_reflexes)
        self.assertFalse(no_reflex.enable_auxiliary_targets)

    def test_contains_monolithic_policy(self) -> None:
        configs = canonical_ablation_configs()
        mono = configs["monolithic_policy"]
        self.assertEqual(mono.architecture, "monolithic")
        self.assertFalse(mono.enable_reflexes)

    def test_contains_drop_variants_for_all_modules(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in MODULE_NAMES:
            key = f"drop_{module_name}"
            self.assertIn(key, configs, f"Missing drop variant: {key}")
            self.assertIn(module_name, configs[key].disabled_modules)

    def test_custom_dropout_propagates_to_modular_full(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.08)
        self.assertAlmostEqual(configs["modular_full"].module_dropout, 0.08)

    def test_variant_names_match_config_keys(self) -> None:
        configs = canonical_ablation_configs()
        names = canonical_ablation_variant_names()
        self.assertEqual(names, tuple(configs.keys()))

    def test_variant_names_starts_with_modular_full(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertEqual(names[0], "modular_full")

    def test_variant_names_include_monolithic_policy(self) -> None:
        names = canonical_ablation_variant_names()
        self.assertIn("monolithic_policy", names)

    def test_variant_names_count(self) -> None:
        names = canonical_ablation_variant_names()
        # modular_full + no_module_dropout + no_module_reflexes
        # + reflex_scale_0_25/_0_50/_0_75 + 5 drop_ variants + monolithic_policy = 12
        expected_count = 6 + len(MODULE_NAMES) + 1
        self.assertEqual(len(names), expected_count)

    def test_contains_reflex_scale_0_25(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("reflex_scale_0_25", configs)
        variant = configs["reflex_scale_0_25"]
        self.assertAlmostEqual(variant.reflex_scale, 0.25)
        self.assertEqual(variant.architecture, "modular")
        self.assertTrue(variant.enable_reflexes)
        self.assertTrue(variant.enable_auxiliary_targets)

    def test_contains_reflex_scale_0_50(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("reflex_scale_0_50", configs)
        variant = configs["reflex_scale_0_50"]
        self.assertAlmostEqual(variant.reflex_scale, 0.50)
        self.assertEqual(variant.architecture, "modular")
        self.assertTrue(variant.enable_reflexes)

    def test_contains_reflex_scale_0_75(self) -> None:
        configs = canonical_ablation_configs()
        self.assertIn("reflex_scale_0_75", configs)
        variant = configs["reflex_scale_0_75"]
        self.assertAlmostEqual(variant.reflex_scale, 0.75)
        self.assertEqual(variant.architecture, "modular")
        self.assertTrue(variant.enable_reflexes)

    def test_reflex_scale_variants_have_no_disabled_modules(self) -> None:
        configs = canonical_ablation_configs()
        for name in ("reflex_scale_0_25", "reflex_scale_0_50", "reflex_scale_0_75"):
            self.assertEqual(configs[name].disabled_modules, ())

    def test_reflex_scale_variants_have_empty_module_reflex_scales(self) -> None:
        configs = canonical_ablation_configs()
        for name in ("reflex_scale_0_25", "reflex_scale_0_50", "reflex_scale_0_75"):
            self.assertEqual(configs[name].module_reflex_scales, {})

    def test_no_module_reflexes_has_reflex_scale_zero(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["no_module_reflexes"]
        self.assertAlmostEqual(variant.reflex_scale, 0.0)

    def test_monolithic_policy_has_reflex_scale_zero(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["monolithic_policy"]
        self.assertAlmostEqual(variant.reflex_scale, 0.0)

    def test_modular_full_has_reflex_scale_one(self) -> None:
        configs = canonical_ablation_configs()
        variant = configs["modular_full"]
        self.assertAlmostEqual(variant.reflex_scale, 1.0)

    def test_drop_variants_have_reflex_scale_one(self) -> None:
        configs = canonical_ablation_configs()
        for module_name in MODULE_NAMES:
            variant = configs[f"drop_{module_name}"]
            self.assertAlmostEqual(variant.reflex_scale, 1.0)

    def test_reflex_scale_variants_custom_dropout_propagates(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.09)
        for name in ("reflex_scale_0_25", "reflex_scale_0_50", "reflex_scale_0_75"):
            self.assertAlmostEqual(configs[name].module_dropout, 0.09)


class ResolveAblationConfigsTest(unittest.TestCase):
    """Tests for resolve_ablation_configs."""

    def test_none_returns_all_canonical_variants(self) -> None:
        configs = resolve_ablation_configs(None)
        names = [c.name for c in configs]
        self.assertEqual(names, list(canonical_ablation_variant_names()))

    def test_specific_names_returns_correct_order(self) -> None:
        configs = resolve_ablation_configs(["monolithic_policy", "modular_full"])
        self.assertEqual(configs[0].name, "monolithic_policy")
        self.assertEqual(configs[1].name, "modular_full")

    def test_single_name_returns_single_config(self) -> None:
        configs = resolve_ablation_configs(["no_module_dropout"])
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].name, "no_module_dropout")

    def test_unknown_name_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            resolve_ablation_configs(["nonexistent_variant"])

    def test_module_dropout_propagates(self) -> None:
        configs = resolve_ablation_configs(["modular_full"], module_dropout=0.07)
        self.assertAlmostEqual(configs[0].module_dropout, 0.07)

    def test_empty_list_preserves_empty_selection(self) -> None:
        configs = resolve_ablation_configs([])
        self.assertEqual(configs, [])


class CorticalModuleBankDisabledModulesTest(unittest.TestCase):
    """Tests for CorticalModuleBank.disabled_modules feature (new in this PR)."""

    def _make_bank(self, disabled: tuple[str, ...] = ()) -> CorticalModuleBank:
        rng = np.random.default_rng(42)
        return CorticalModuleBank(action_dim=5, rng=rng, module_dropout=0.0, disabled_modules=disabled)

    def _blank_observation(self) -> dict:
        obs = {}
        for spec in self._make_bank().specs:
            obs[spec.observation_key] = np.zeros(spec.input_dim, dtype=float)
        return obs

    def test_disabled_modules_stored_as_set(self) -> None:
        bank = self._make_bank(("alert_center",))
        self.assertIsInstance(bank.disabled_modules, set)

    def test_unknown_disabled_module_raises_value_error(self) -> None:
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(ValueError, "nonexistent_module"):
            CorticalModuleBank(
                action_dim=5,
                rng=rng,
                module_dropout=0.0,
                disabled_modules=("nonexistent_module",),
            )

    def test_disabled_module_produces_zero_logits(self) -> None:
        bank = self._make_bank(("hunger_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        hunger_result = next(r for r in results if r.name == "hunger_center")
        np.testing.assert_allclose(hunger_result.logits, np.zeros(5, dtype=float))

    def test_disabled_module_skips_forward_pass(self) -> None:
        bank = self._make_bank(("hunger_center",))
        obs = self._blank_observation()

        def _disabled_forward(*args: object, **kwargs: object) -> np.ndarray:
            raise AssertionError("disabled module forward should not run")

        bank.modules["hunger_center"].forward = _disabled_forward  # type: ignore[assignment]
        results = bank.forward(obs, store_cache=False, training=False)

        hunger_result = next(r for r in results if r.name == "hunger_center")
        self.assertFalse(hunger_result.active)
        np.testing.assert_allclose(hunger_result.logits, np.zeros(5, dtype=float))

    def test_disabled_module_marked_inactive(self) -> None:
        bank = self._make_bank(("sleep_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        sleep_result = next(r for r in results if r.name == "sleep_center")
        self.assertFalse(sleep_result.active)

    def test_enabled_modules_remain_active(self) -> None:
        bank = self._make_bank(("alert_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        active_names = [r.name for r in results if r.active]
        self.assertNotIn("alert_center", active_names)
        self.assertIn("visual_cortex", active_names)
        self.assertIn("hunger_center", active_names)

    def test_no_disabled_modules_all_active(self) -> None:
        bank = self._make_bank(())
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        self.assertTrue(all(r.active for r in results))

    def test_dropout_keeps_one_non_disabled_module_active(self) -> None:
        rng = np.random.default_rng(42)
        bank = CorticalModuleBank(
            action_dim=5,
            rng=rng,
            module_dropout=1.0,
            disabled_modules=(
                "sensory_cortex",
                "hunger_center",
                "sleep_center",
                "alert_center",
            ),
        )
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=True)

        active_names = [result.name for result in results if result.active]
        self.assertEqual(active_names, ["visual_cortex"])


class ModuleResultNamedObservationTest(unittest.TestCase):
    """Tests for ModuleResult.named_observation (new None-guard added in this PR)."""

    def test_named_observation_returns_empty_dict_for_none_interface(self) -> None:
        result = ModuleResult(
            interface=None,
            name="monolithic_policy",
            observation_key="monolithic_policy",
            observation=np.zeros(5, dtype=float),
            logits=np.zeros(5, dtype=float),
            probs=np.full(5, 0.2, dtype=float),
            active=True,
        )
        self.assertEqual(result.named_observation(), {})

    def test_named_observation_works_with_valid_interface(self) -> None:
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        obs = np.zeros(interface.input_dim, dtype=float)
        result = ModuleResult(
            interface=interface,
            name="alert_center",
            observation_key=interface.observation_key,
            observation=obs,
            logits=np.zeros(5, dtype=float),
            probs=np.full(5, 0.2, dtype=float),
            active=True,
        )
        named = result.named_observation()
        self.assertIsInstance(named, dict)
        self.assertIn("predator_visible", named)


class ModuleResultReflexFieldsTest(unittest.TestCase):
    """Tests for the new reflex-related fields on ModuleResult (new in this PR)."""

    def _make_result(self, **kwargs) -> ModuleResult:
        interface = MODULE_INTERFACE_BY_NAME["alert_center"]
        obs = np.zeros(interface.input_dim, dtype=float)
        action_dim = len(LOCOMOTION_ACTIONS)
        defaults = dict(
            interface=interface,
            name="alert_center",
            observation_key=interface.observation_key,
            observation=obs,
            logits=np.zeros(action_dim, dtype=float),
            probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
            active=True,
        )
        defaults.update(kwargs)
        return ModuleResult(**defaults)

    def test_neural_logits_defaults_to_none(self) -> None:
        result = self._make_result()
        self.assertIsNone(result.neural_logits)

    def test_reflex_delta_logits_defaults_to_none(self) -> None:
        result = self._make_result()
        self.assertIsNone(result.reflex_delta_logits)

    def test_post_reflex_logits_defaults_to_none(self) -> None:
        result = self._make_result()
        self.assertIsNone(result.post_reflex_logits)

    def test_reflex_applied_defaults_to_false(self) -> None:
        result = self._make_result()
        self.assertFalse(result.reflex_applied)

    def test_effective_reflex_scale_defaults_to_zero(self) -> None:
        result = self._make_result()
        self.assertAlmostEqual(result.effective_reflex_scale, 0.0)

    def test_module_reflex_override_defaults_to_false(self) -> None:
        result = self._make_result()
        self.assertFalse(result.module_reflex_override)

    def test_module_reflex_dominance_defaults_to_zero(self) -> None:
        result = self._make_result()
        self.assertAlmostEqual(result.module_reflex_dominance, 0.0)

    def test_can_set_neural_logits(self) -> None:
        logits = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = self._make_result(neural_logits=logits.copy())
        np.testing.assert_allclose(result.neural_logits, logits)

    def test_can_set_reflex_applied_true(self) -> None:
        result = self._make_result(reflex_applied=True)
        self.assertTrue(result.reflex_applied)

    def test_can_set_effective_reflex_scale(self) -> None:
        result = self._make_result(effective_reflex_scale=0.5)
        self.assertAlmostEqual(result.effective_reflex_scale, 0.5)

    def test_can_set_module_reflex_override_true(self) -> None:
        result = self._make_result(module_reflex_override=True)
        self.assertTrue(result.module_reflex_override)

    def test_can_set_module_reflex_dominance(self) -> None:
        result = self._make_result(module_reflex_dominance=0.75)
        self.assertAlmostEqual(result.module_reflex_dominance, 0.75)


class CorticalModuleBankResultFieldsTest(unittest.TestCase):
    """Tests that CorticalModuleBank.forward initializes new reflex fields on ModuleResult."""

    def _make_bank(self) -> CorticalModuleBank:
        rng = np.random.default_rng(99)
        return CorticalModuleBank(action_dim=5, rng=rng, module_dropout=0.0)

    def _blank_observation(self, bank: CorticalModuleBank) -> dict:
        obs = {}
        for spec in bank.specs:
            obs[spec.observation_key] = np.zeros(spec.input_dim, dtype=float)
        return obs

    def test_forward_sets_neural_logits_to_copy_of_logits(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.neural_logits)
                np.testing.assert_allclose(result.neural_logits, result.logits)

    def test_forward_sets_reflex_delta_logits_to_zeros(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.reflex_delta_logits)
                np.testing.assert_allclose(
                    result.reflex_delta_logits,
                    np.zeros_like(result.logits),
                )

    def test_forward_sets_post_reflex_logits_to_copy_of_logits(self) -> None:
        bank = self._make_bank()
        obs = self._blank_observation(bank)
        results = bank.forward(obs, store_cache=False, training=False)
        for result in results:
            if result.active:
                self.assertIsNotNone(result.post_reflex_logits)
                np.testing.assert_allclose(result.post_reflex_logits, result.logits)


class SpiderBrainMonolithicTest(unittest.TestCase):
    """Tests for the new monolithic architecture branch in SpiderBrain."""

    def _monolithic_config(self) -> BrainAblationConfig:
        return BrainAblationConfig(
            name="monolithic_policy",
            architecture="monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            disabled_modules=(),
        )

    def _build_observation(self) -> dict:
        """
        Builds a baseline observation dictionary containing zeroed input vectors for every module and context.
        
        The returned dictionary contains entries for each interface in MODULE_INTERFACES plus ACTION_CONTEXT_INTERFACE and MOTOR_CONTEXT_INTERFACE. Each key is the interface's `observation_key` and each value is a NumPy array of zeros with length equal to that interface's `input_dim`.
        
        Returns:
            dict: Mapping from observation_key to a zeroed NumPy array sized to the corresponding interface's input_dim.
        """
        obs = {}
        for interface in MODULE_INTERFACES:
            obs[interface.observation_key] = np.zeros(interface.input_dim, dtype=float)
        obs[ACTION_CONTEXT_INTERFACE.observation_key] = np.zeros(
            ACTION_CONTEXT_INTERFACE.input_dim, dtype=float
        )
        obs[MOTOR_CONTEXT_INTERFACE.observation_key] = np.zeros(
            MOTOR_CONTEXT_INTERFACE.input_dim, dtype=float
        )
        return obs

    def test_monolithic_brain_has_no_module_bank(self) -> None:
        brain = SpiderBrain(seed=1, config=self._monolithic_config())
        self.assertIsNone(brain.module_bank)

    def test_monolithic_brain_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=1, config=self._monolithic_config())
        self.assertIsNotNone(brain.monolithic_policy)

    def test_modular_brain_has_module_bank(self) -> None:
        brain = SpiderBrain(seed=1)
        self.assertIsNotNone(brain.module_bank)

    def test_modular_brain_has_no_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=1)
        self.assertIsNone(brain.monolithic_policy)

    def test_monolithic_act_returns_single_module_result(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertEqual(len(step.module_results), 1)
        self.assertEqual(step.module_results[0].name, SpiderBrain.MONOLITHIC_POLICY_NAME)

    def test_monolithic_act_result_has_none_interface(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertIsNone(step.module_results[0].interface)

    def test_monolithic_act_result_is_active(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertTrue(step.module_results[0].active)

    def test_monolithic_module_names_includes_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=5, config=self._monolithic_config())
        names = brain._module_names()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)

    def test_modular_module_names_includes_each_module(self) -> None:
        brain = SpiderBrain(seed=5)
        names = brain._module_names()
        self.assertIn("visual_cortex", names)
        self.assertIn("alert_center", names)
        self.assertIn("action_center", names)
        self.assertIn("motor_cortex", names)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, names)

    def test_monolithic_parameter_norms_has_monolithic_policy(self) -> None:
        brain = SpiderBrain(seed=3, config=self._monolithic_config())
        norms = brain.parameter_norms()
        self.assertIn(SpiderBrain.MONOLITHIC_POLICY_NAME, norms)
        self.assertIn("action_center", norms)
        self.assertIn("motor_cortex", norms)

    def test_modular_parameter_norms_has_module_names(self) -> None:
        brain = SpiderBrain(seed=3)
        norms = brain.parameter_norms()
        self.assertIn("visual_cortex", norms)
        self.assertIn("alert_center", norms)
        self.assertIn("action_center", norms)
        self.assertIn("motor_cortex", norms)
        self.assertNotIn(SpiderBrain.MONOLITHIC_POLICY_NAME, norms)

    def test_monolithic_estimate_value_returns_finite_float(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        value = brain.estimate_value(obs)
        self.assertIsInstance(value, float)
        self.assertTrue(np.isfinite(value))

    def test_monolithic_action_idx_is_valid(self) -> None:
        brain = SpiderBrain(seed=7, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        self.assertGreaterEqual(step.action_idx, 0)
        self.assertLess(step.action_idx, len(LOCOMOTION_ACTIONS))

    def test_monolithic_save_and_load_round_trip(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)
        obs = self._build_observation()
        step_before = brain.act(obs, sample=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            brain2 = SpiderBrain(seed=999, config=config)
            brain2.load(save_path)
            step_after = brain2.act(obs, sample=False)

        np.testing.assert_allclose(step_before.policy, step_after.policy, atol=1e-5)
        np.testing.assert_allclose(step_before.value, step_after.value, atol=1e-5)

    def test_loading_mismatched_architecture_version_raises_value_error(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["architecture_version"] = SpiderBrain.ARCHITECTURE_VERSION - 1
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaises(ValueError):
                brain2.load(save_path)

    def test_save_metadata_contains_registry_and_fingerprint(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertIn("interface_registry", metadata)
            self.assertIn("architecture_fingerprint", metadata)
            self.assertEqual(
                metadata["architecture_fingerprint"],
                metadata["architecture"]["fingerprint"],
            )

    def test_loading_mismatched_interface_registry_raises_value_error(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=11, config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain.save(save_path)

            metadata_path = save_path / SpiderBrain._METADATA_FILE
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["interface_registry"]["interfaces"]["motor_cortex_context"]["version"] = 999
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            brain2 = SpiderBrain(seed=999, config=config)
            with self.assertRaisesRegex(ValueError, "registry de interfaces incompatível"):
                brain2.load(save_path)

    def test_loading_monolithic_into_modular_raises_value_error(self) -> None:
        mono_config = self._monolithic_config()
        brain_mono = SpiderBrain(seed=13, config=mono_config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "mono_brain"
            brain_mono.save(save_path)

            brain_modular = SpiderBrain(seed=13)
            with self.assertRaises(ValueError):
                brain_modular.load(save_path)

    def test_loading_different_modular_ablation_config_raises_value_error(self) -> None:
        saved_brain = SpiderBrain(seed=19, config=default_brain_config(module_dropout=0.0))
        different_config = BrainAblationConfig(
            name="no_module_reflexes",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            disabled_modules=(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "modular_brain"
            saved_brain.save(save_path)

            mismatched_brain = SpiderBrain(seed=19, config=different_config)
            with self.assertRaises(ValueError):
                mismatched_brain.load(save_path)

    def test_loading_different_operational_profile_raises_value_error(self) -> None:
        saved_brain = SpiderBrain(seed=23)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "profiled_brain"
            saved_brain.save(save_path)

            mismatched_brain = SpiderBrain(
                seed=23,
                operational_profile=_profile_with_updates(predator_threat_smell_threshold=0.01),
            )
            with self.assertRaises(ValueError):
                mismatched_brain.load(save_path)

    def test_monolithic_config_stored_on_brain(self) -> None:
        config = self._monolithic_config()
        brain = SpiderBrain(seed=7, config=config)
        self.assertEqual(brain.config.architecture, "monolithic")
        self.assertEqual(brain.config.name, "monolithic_policy")

    def test_monolithic_learn_does_not_raise(self) -> None:
        brain = SpiderBrain(seed=17, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)
        next_obs = self._build_observation()
        stats = brain.learn(step, reward=0.5, next_observation=next_obs, done=False)
        self.assertIn("reward", stats)
        self.assertTrue(np.isfinite(stats["td_error"]))

    def test_modular_learn_routes_action_center_input_grads_to_module_aux_paths(self) -> None:
        config = BrainAblationConfig(
            name="modular_no_reflex_aux",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
        )
        brain = SpiderBrain(seed=7, config=config)
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        proposal_grads = np.arange(
            len(step.module_results) * len(LOCOMOTION_ACTIONS),
            dtype=float,
        ).reshape(len(step.module_results), len(LOCOMOTION_ACTIONS))
        upstream = np.concatenate(
            [
                proposal_grads.reshape(-1),
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            captured["grad_value"] = float(grad_value)
            captured["lr"] = float(lr)
            return upstream

        def fake_module_bank_backward(grad_logits, lr, aux_grads=None):
            captured["bank_grad"] = np.asarray(grad_logits, dtype=float).copy()
            captured["bank_lr"] = float(lr)
            captured["aux_grads"] = {
                name: np.asarray(value, dtype=float).copy()
                for name, value in (aux_grads or {}).items()
            }

        brain.action_center.backward = fake_action_center_backward
        brain.module_bank.backward = fake_module_bank_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(captured["bank_grad"], captured["shared_grad"])
        for idx, result in enumerate(step.module_results):
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                proposal_grads[idx],
            )

    def test_monolithic_learn_adds_action_center_input_grads_to_policy_update(self) -> None:
        brain = SpiderBrain(seed=11, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=False)

        captured: dict[str, object] = {}
        extra_grad = np.linspace(0.1, 0.5, len(LOCOMOTION_ACTIONS), dtype=float)
        upstream = np.concatenate(
            [
                extra_grad,
                np.zeros(ACTION_CONTEXT_INTERFACE.input_dim, dtype=float),
            ]
        )

        def fake_action_center_backward(*, grad_policy_logits, grad_value, lr):
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_monolithic_backward(grad_logits, lr):
            captured["mono_grad"] = np.asarray(grad_logits, dtype=float).copy()
            captured["mono_lr"] = float(lr)

        def fake_motor_backward(grad_logits, lr):
            captured["motor_grad"] = np.asarray(grad_logits, dtype=float).copy()
            captured["motor_lr"] = float(lr)

        brain.action_center.backward = fake_action_center_backward
        brain.monolithic_policy.backward = fake_monolithic_backward
        brain.motor_cortex.backward = fake_motor_backward
        brain.estimate_value = lambda _: 0.0

        brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(
            captured["mono_grad"],
            captured["shared_grad"] + extra_grad,
        )
        np.testing.assert_allclose(
            captured["motor_grad"],
            captured["shared_grad"],
        )


class SpiderSimulationBrainConfigTest(unittest.TestCase):
    """Tests for the brain_config parameter added to SpiderSimulation."""

    def test_default_brain_config_is_modular_full(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        self.assertEqual(sim.brain_config.name, "modular_full")
        self.assertEqual(sim.brain_config.architecture, "modular")

    def test_custom_brain_config_stored(self) -> None:
        config = BrainAblationConfig(name="no_module_dropout", architecture="modular", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        self.assertEqual(sim.brain_config.name, "no_module_dropout")

    def test_module_dropout_comes_from_brain_config(self) -> None:
        config = BrainAblationConfig(name="no_module_dropout", architecture="modular", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        self.assertEqual(sim.module_dropout, 0.0)

    def test_brain_config_propagated_to_brain(self) -> None:
        config = BrainAblationConfig(
            name="drop_alert_center",
            architecture="modular",
            module_dropout=0.0,
            disabled_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        self.assertEqual(sim.brain.config.name, "drop_alert_center")
        self.assertIn("alert_center", sim.brain.config.disabled_modules)

    def test_annotate_behavior_rows_uses_effective_eval_reflex_scale(self) -> None:
        config = BrainAblationConfig(
            name="no_module_reflexes",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            reflex_scale=0.0,
        )
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)

        rows = sim._annotate_behavior_rows([{"scenario": "night_rest"}], eval_reflex_scale=1.0)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(float(rows[0]["eval_reflex_scale"]), 0.0)


class AnnotateBehaviorRowsTest(unittest.TestCase):
    """Tests for SpiderSimulation._annotate_behavior_rows (new in this PR)."""

    def test_annotate_adds_ablation_variant(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertIn("ablation_variant", annotated[0])

    def test_annotate_adds_ablation_architecture(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertIn("ablation_architecture", annotated[0])

    def test_annotate_adds_operational_profile_metadata(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["operational_profile"], "default_v1")
        self.assertEqual(annotated[0]["operational_profile_version"], 1)
        self.assertEqual(annotated[0]["noise_profile"], "none")
        self.assertIn("noise_profile_config", annotated[0])

    def test_annotate_variant_matches_config_name(self) -> None:
        config = BrainAblationConfig(name="drop_sleep_center", architecture="modular", module_dropout=0.0, disabled_modules=("sleep_center",))
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        rows = [{"scenario": "night_rest", "success": 0}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["ablation_variant"], "drop_sleep_center")

    def test_annotate_architecture_matches_config(self) -> None:
        config = BrainAblationConfig(name="monolithic_policy", architecture="monolithic", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        rows = [{"scenario": "night_rest", "success": 0}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["ablation_architecture"], "monolithic")

    def test_annotate_preserves_original_row_content(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest", "success": 1, "failures": []}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["scenario"], "night_rest")
        self.assertEqual(annotated[0]["success"], 1)

    def test_annotate_does_not_mutate_original_rows(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest"}]
        sim._annotate_behavior_rows(rows)
        self.assertNotIn("ablation_variant", rows[0])

    def test_annotate_empty_rows_returns_empty(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        annotated = sim._annotate_behavior_rows([])
        self.assertEqual(annotated, [])

    def test_annotate_multiple_rows(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        rows = [{"scenario": "night_rest"}, {"scenario": "predator_edge"}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(len(annotated), 2)
        for row in annotated:
            self.assertIn("ablation_variant", row)


class BuildAblationDeltasTest(unittest.TestCase):
    """Tests for SpiderSimulation._build_ablation_deltas (new in this PR)."""

    def _make_variants(self) -> dict:
        def _variant(scenario_sr: float, episode_sr: float, scenario_success: float):
            return {
                "summary": {
                    "scenario_success_rate": scenario_sr,
                    "episode_success_rate": episode_sr,
                },
                "suite": {
                    "night_rest": {"success_rate": scenario_success},
                },
            }
        return {
            "modular_full": _variant(0.50, 0.60, 0.50),
            "no_module_reflexes": _variant(0.30, 0.40, 0.20),
        }

    def test_reference_variant_delta_is_zero(self) -> None:
        variants = self._make_variants()
        deltas = SpiderSimulation._build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        ref_delta = deltas["modular_full"]["summary"]
        self.assertAlmostEqual(ref_delta["scenario_success_rate_delta"], 0.0)
        self.assertAlmostEqual(ref_delta["episode_success_rate_delta"], 0.0)

    def test_nonreference_variant_delta_is_computed_correctly(self) -> None:
        variants = self._make_variants()
        deltas = SpiderSimulation._build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        d = deltas["no_module_reflexes"]["summary"]
        self.assertAlmostEqual(d["scenario_success_rate_delta"], 0.30 - 0.50, places=5)
        self.assertAlmostEqual(d["episode_success_rate_delta"], 0.40 - 0.60, places=5)

    def test_scenario_level_delta_computed(self) -> None:
        variants = self._make_variants()
        deltas = SpiderSimulation._build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        scenario_delta = deltas["no_module_reflexes"]["scenarios"]["night_rest"]["success_rate_delta"]
        self.assertAlmostEqual(scenario_delta, 0.20 - 0.50, places=5)

    def test_reference_scenario_delta_is_zero(self) -> None:
        variants = self._make_variants()
        deltas = SpiderSimulation._build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        self.assertAlmostEqual(
            deltas["modular_full"]["scenarios"]["night_rest"]["success_rate_delta"], 0.0
        )

    def test_all_variants_present_in_deltas(self) -> None:
        variants = self._make_variants()
        deltas = SpiderSimulation._build_ablation_deltas(
            variants, reference_variant="modular_full", scenario_names=["night_rest"]
        )
        self.assertIn("modular_full", deltas)
        self.assertIn("no_module_reflexes", deltas)


class BuildSummaryBrainConfigTest(unittest.TestCase):
    """Tests that _build_summary now includes brain config info."""

    def test_summary_config_has_brain_key(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIn("brain", summary["config"])

    def test_summary_brain_config_architecture_matches(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["brain"]["architecture"], "modular")

    def test_summary_config_has_operational_profile(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIn("operational_profile", summary["config"])
        self.assertEqual(summary["config"]["operational_profile"]["name"], "default_v1")
        self.assertIn("version", summary["config"]["operational_profile"])
        self.assertEqual(summary["config"]["operational_profile"]["version"], 1)
        self.assertIn("noise_profile", summary["config"])
        self.assertEqual(summary["config"]["noise_profile"]["name"], "none")

    def test_summary_brain_config_name_matches(self) -> None:
        """
        Verify that SpiderSimulation._build_summary includes the default brain config name "modular_full".
        """
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["brain"]["name"], "modular_full")

    def test_summary_has_architecture_traceability(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["architecture_version"], SpiderBrain.ARCHITECTURE_VERSION)
        self.assertTrue(summary["config"]["architecture_fingerprint"])

    def test_summary_monolithic_brain_config_in_summary(self) -> None:
        config = BrainAblationConfig(name="monolithic_policy", architecture="monolithic", module_dropout=0.0)
        sim = SpiderSimulation(seed=7, max_steps=5, brain_config=config)
        summary = sim._build_summary([], [])
        self.assertEqual(summary["config"]["brain"]["architecture"], "monolithic")


class CompareAblationSuiteAlwaysIncludesReferenceTest(unittest.TestCase):
    """Tests that compare_ablation_suite always includes modular_full as reference."""

    def test_reference_always_present_when_only_variant_requested(self) -> None:
        payload, _ = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["no_module_dropout"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertIn("modular_full", payload["variants"])
        self.assertEqual(payload["reference_variant"], "modular_full")

    def test_reference_not_duplicated_when_explicitly_requested(self) -> None:
        payload, _ = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["modular_full", "monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        variant_keys = list(payload["variants"].keys())
        self.assertEqual(variant_keys.count("modular_full"), 1)

    def test_deltas_always_has_reference_entry(self) -> None:
        payload, _ = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["no_module_reflexes"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertIn("modular_full", payload["deltas_vs_reference"])

    def test_scenario_names_in_payload(self) -> None:
        payload, _ = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest", "predator_edge"),
            seeds=(7,),
        )
        self.assertEqual(payload["scenario_names"], ["night_rest", "predator_edge"])

    def test_seeds_in_payload(self) -> None:
        payload, _ = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7, 17),
        )
        self.assertEqual(payload["seeds"], [7, 17])

    def test_rows_contain_ablation_columns(self) -> None:
        _, rows = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("ablation_variant", row)
            self.assertIn("ablation_architecture", row)

    def test_rows_contain_operational_profile_columns(self) -> None:
        _, rows = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("operational_profile", row)
            self.assertIn("operational_profile_version", row)
            self.assertIn("noise_profile", row)
            self.assertIn("noise_profile_config", row)

    def test_rows_contain_architecture_traceability_columns(self) -> None:
        _, rows = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("architecture_version", row)
            self.assertIn("architecture_fingerprint", row)

    def test_payload_includes_without_reflex_support_and_rows_track_eval_scale(self) -> None:
        payload, rows = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["reflex_scale_0_50"],
            names=("night_rest",),
            seeds=(7,),
        )

        self.assertIn("without_reflex_support", payload["variants"]["modular_full"])
        self.assertIn("without_reflex_support", payload["variants"]["reflex_scale_0_50"])
        self.assertAlmostEqual(
            float(payload["variants"]["reflex_scale_0_50"]["config"]["reflex_scale"]),
            0.5,
        )
        reflex_scale_rows = [
            row for row in rows if row["ablation_variant"] == "reflex_scale_0_50"
        ]
        self.assertTrue(reflex_scale_rows)
        self.assertTrue(
            any(float(row["eval_reflex_scale"]) == 0.0 for row in reflex_scale_rows)
        )
        self.assertTrue(
            any(float(row["eval_reflex_scale"]) > 0.0 for row in reflex_scale_rows)
        )

    def test_without_reflex_support_reuses_behavior_base_index(self) -> None:
        original_execute = SpiderSimulation._execute_behavior_suite
        recorded_calls: list[tuple[str, float, int]] = []

        def wrapped_execute(
            self,
            *,
            names,
            episodes_per_scenario,
            capture_trace,
            debug_trace,
            base_index=100_000,
        ):
            recorded_calls.append(
                (
                    self.brain.config.name,
                    float(self.brain.current_reflex_scale),
                    int(base_index),
                )
            )
            return original_execute(
                self,
                names=names,
                episodes_per_scenario=episodes_per_scenario,
                capture_trace=capture_trace,
                debug_trace=debug_trace,
                base_index=base_index,
            )

        SpiderSimulation._execute_behavior_suite = wrapped_execute  # type: ignore[method-assign]
        try:
            SpiderSimulation.compare_ablation_suite(
                episodes=0,
                evaluation_episodes=0,
                variant_names=["reflex_scale_0_50"],
                names=("night_rest",),
                seeds=(7,),
            )
        finally:
            SpiderSimulation._execute_behavior_suite = original_execute  # type: ignore[method-assign]

        self.assertEqual(
            recorded_calls,
            [
                ("modular_full", 1.0, 300_000),
                ("modular_full", 0.0, 300_000),
                ("reflex_scale_0_50", 0.5, 300_000),
                ("reflex_scale_0_50", 0.0, 300_000),
            ],
        )

    def test_rows_operational_profile_value_is_default(self) -> None:
        _, rows = SpiderSimulation.compare_ablation_suite(
            episodes=0,
            evaluation_episodes=0,
            variant_names=["monolithic_policy"],
            names=("night_rest",),
            seeds=(7,),
        )
        for row in rows:
            self.assertEqual(row["operational_profile"], "default_v1")
            self.assertEqual(row["operational_profile_version"], 1)
            self.assertEqual(row["noise_profile"], "none")
            self.assertIn("noise_profile_config", row)


class SpiderSimulationOperationalProfileTest(unittest.TestCase):
    """Tests for operational_profile propagation through SpiderSimulation."""

    def test_simulation_stores_resolved_operational_profile(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        self.assertEqual(sim.operational_profile.name, "default_v1")

    def test_simulation_accepts_profile_by_name(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile="default_v1")
        self.assertEqual(sim.operational_profile.name, "default_v1")

    def test_simulation_accepts_profile_instance(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        self.assertIs(sim.operational_profile, profile)

    def test_simulation_propagates_profile_to_world(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        self.assertIs(sim.world.operational_profile, profile)

    def test_simulation_propagates_profile_to_brain(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        self.assertIs(sim.brain.operational_profile, profile)

    def test_summary_operational_profile_version_is_int(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=5)
        summary = sim._build_summary([], [])
        self.assertIsInstance(summary["config"]["operational_profile"]["version"], int)

    def test_annotate_behavior_rows_with_custom_profile(self) -> None:
        profile = _profile_with_updates(predator_threat_smell_threshold=0.01)
        sim = SpiderSimulation(seed=7, max_steps=5, operational_profile=profile)
        rows = [{"scenario": "night_rest"}]
        annotated = sim._annotate_behavior_rows(rows)
        self.assertEqual(annotated[0]["operational_profile"], "ablation_test_profile")
        self.assertEqual(annotated[0]["operational_profile_version"], 21)
        self.assertEqual(annotated[0]["noise_profile"], "none")


if __name__ == "__main__":
    unittest.main()
