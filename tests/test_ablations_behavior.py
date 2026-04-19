from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    MODULE_NAMES,
    MULTI_PREDATOR_SCENARIOS,
    MULTI_PREDATOR_SCENARIO_GROUPS,
    VISUAL_PREDATOR_SCENARIOS,
    OLFACTORY_PREDATOR_SCENARIOS,
    MONOLITHIC_POLICY_NAME,
    canonical_ablation_configs,
    canonical_ablation_scenario_groups,
    compare_predator_type_ablation_performance,
    canonical_ablation_variant_names,
    default_brain_config,
    resolve_ablation_scenario_group,
    resolve_ablation_configs,
    _safe_float,
    _mean,
    _scenario_success_rate,
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
from spider_cortex_sim.nn import RecurrentProposalNetwork
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.simulation import SpiderSimulation

from tests.fixtures.ablations import _blank_mapping, _build_observation, _profile_with_updates

class AblationScenarioGroupTest(unittest.TestCase):
    def test_multi_predator_group_resolves_expected_scenarios(self) -> None:
        self.assertEqual(
            resolve_ablation_scenario_group("multi_predator_ecology"),
            MULTI_PREDATOR_SCENARIOS,
        )

    def test_compare_predator_type_ablation_performance_summarizes_visual_vs_olfactory_groups(self) -> None:
        """
        Verifies that compare_predator_type_ablation_performance summarizes mean success rates for visual and olfactory predator scenario groups and produces the expected sign change in visual_minus_olfactory_success_rate between visual and sensory cortex drops.
        
        Constructs a payload with per-variant scenario success rates and deltas, calls compare_predator_type_ablation_performance, and asserts that the result is available, that the grouped mean success rates for visual and olfactory predator scenarios match expected values, and that the visual-minus-olfactory difference is negative for the visual-cortex drop and positive for the sensory-cortex drop.
        """
        payload = {
            "variants": {
                "drop_visual_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.2},
                        "visual_hunter_open_field": {"success_rate": 0.1},
                        "olfactory_ambush": {"success_rate": 0.8},
                    }
                },
                "drop_sensory_cortex": {
                    "suite": {
                        "visual_olfactory_pincer": {"success_rate": 0.3},
                        "visual_hunter_open_field": {"success_rate": 0.9},
                        "olfactory_ambush": {"success_rate": 0.2},
                    }
                },
            },
            "deltas_vs_reference": {
                "drop_visual_cortex": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"success_rate_delta": -0.3},
                        "visual_hunter_open_field": {"success_rate_delta": -0.6},
                        "olfactory_ambush": {"success_rate_delta": -0.1},
                    }
                },
                "drop_sensory_cortex": {
                    "scenarios": {
                        "visual_olfactory_pincer": {"success_rate_delta": -0.2},
                        "visual_hunter_open_field": {"success_rate_delta": -0.1},
                        "olfactory_ambush": {"success_rate_delta": -0.7},
                    }
                },
            },
        }

        result = compare_predator_type_ablation_performance(payload)

        self.assertTrue(result["available"])
        visual_drop = result["comparisons"]["drop_visual_cortex"]
        sensory_drop = result["comparisons"]["drop_sensory_cortex"]
        self.assertAlmostEqual(
            visual_drop["visual_predator_scenarios"]["mean_success_rate"],
            0.15,
        )
        self.assertAlmostEqual(
            visual_drop["olfactory_predator_scenarios"]["mean_success_rate"],
            0.5,
        )
        self.assertLess(
            visual_drop["visual_minus_olfactory_success_rate"],
            0.0,
        )
        self.assertGreater(
            sensory_drop["visual_minus_olfactory_success_rate"],
            0.0,
        )

class BrainAblationBehaviorTest(unittest.TestCase):
    def _module_result(self, module_name: str, **signals: float) -> ModuleResult:
        """
        Create a synthetic ModuleResult for testing with specified observation signal values.
        
        Parameters:
            module_name (str): Name of the module whose interface and observation vector are used.
            **signals (float): Per-signal overrides for the interface's observation mapping; keys are signal names and values coerced to float.
        
        Returns:
            ModuleResult: Active ModuleResult for the named module with zero logits and uniform action probabilities, using the interface's observation_key and the constructed observation vector.
        """
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
                "predator_motion_salience": 1.0,
            },
            action_context={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
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
        """
        Verify that disabling auxiliary targets prevents any auxiliary-module gradients from being produced.
        
        Creates a modular SpiderBrain with reflexes enabled but auxiliary targets disabled, constructs an alert_center ModuleResult with a forced reflex decision, and asserts that _auxiliary_module_gradients returns an empty dict.
        """
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

    def test_fixed_arbitration_baseline_matches_fixed_formula_behavior(self) -> None:
        configs = canonical_ablation_configs(module_dropout=0.0)
        canonical_fixed = configs["fixed_arbitration_baseline"]
        manual_fixed = BrainAblationConfig(
            name="manual_fixed_arbitration",
            module_dropout=0.0,
            use_learned_arbitration=False,
        )
        observation = _build_observation(
            action_context={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
                "hunger": 0.95,
                "day": 1.0,
            },
            alert={
                "predator_visible": 1.0,
                "predator_certainty": 0.9,
                "predator_motion_salience": 0.8,
                "predator_smell_strength": 0.7,
            },
            hunger={
                "hunger": 0.95,
                "food_visible": 1.0,
                "food_certainty": 0.9,
                "food_smell_strength": 0.8,
                "food_memory_age": 0.0,
            },
        )
        canonical_brain = SpiderBrain(seed=19, module_dropout=0.0, config=canonical_fixed)
        manual_brain = SpiderBrain(seed=19, module_dropout=0.0, config=manual_fixed)

        canonical_step = canonical_brain.act(observation, sample=False)
        manual_step = manual_brain.act(observation, sample=False)

        self.assertFalse(canonical_step.arbitration_decision.learned_adjustment)
        self.assertFalse(manual_step.arbitration_decision.learned_adjustment)
        self.assertEqual(
            canonical_step.arbitration_decision.winning_valence,
            manual_step.arbitration_decision.winning_valence,
        )
        self.assertEqual(
            canonical_step.arbitration_decision.module_gates,
            manual_step.arbitration_decision.module_gates,
        )
        np.testing.assert_allclose(canonical_step.action_center_policy, manual_step.action_center_policy)

class CorticalModuleBankDisabledModulesTest(unittest.TestCase):
    """Tests for CorticalModuleBank.disabled_modules feature (new in this PR)."""

    def _make_bank(self, disabled: tuple[str, ...] = ()) -> CorticalModuleBank:
        """
        Constructs a CorticalModuleBank with a fixed RNG, zero module dropout, and the specified disabled modules.
        
        Parameters:
            disabled (tuple[str, ...]): Names of modules to disable in the returned bank.
        
        Returns:
            CorticalModuleBank: Bank with action_dim equal to the number of locomotion actions, RNG seeded with 42, module_dropout set to 0.0, and disabled_modules set to `disabled`.
        """
        rng = np.random.default_rng(42)
        return CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=0.0,
            disabled_modules=disabled,
        )

    def _blank_observation(self) -> dict:
        """
        Constructs a blank observation mapping for the brain's module interfaces.
        
        Returns:
            dict: Mapping from each interface's `observation_key` to a NumPy float array of zeros with length equal to that interface's `input_dim`.
        """
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
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                disabled_modules=("nonexistent_module",),
            )

    def test_disabled_module_produces_zero_logits(self) -> None:
        bank = self._make_bank(("hunger_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        hunger_result = next(r for r in results if r.name == "hunger_center")
        np.testing.assert_allclose(hunger_result.logits, np.zeros(len(LOCOMOTION_ACTIONS), dtype=float))

    def test_disabled_module_skips_forward_pass(self) -> None:
        """
        Verifies that a disabled module's forward pass is skipped and its outputs are zeroed.
        
        Asserts the module's forward method is not invoked, the corresponding ModuleResult is marked inactive, and its logits are all zeros with length equal to the locomotion action space.
        """
        bank = self._make_bank(("hunger_center",))
        obs = self._blank_observation()

        def _disabled_forward(*args: object, **kwargs: object) -> np.ndarray:
            raise AssertionError("disabled module forward should not run")

        bank.modules["hunger_center"].forward = _disabled_forward  # type: ignore[assignment]
        results = bank.forward(obs, store_cache=False, training=False)

        hunger_result = next(r for r in results if r.name == "hunger_center")
        self.assertFalse(hunger_result.active)
        np.testing.assert_allclose(hunger_result.logits, np.zeros(len(LOCOMOTION_ACTIONS), dtype=float))

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
            action_dim=len(LOCOMOTION_ACTIONS),
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

class CorticalModuleBankRecurrentModulesTest(unittest.TestCase):
    """Tests for selecting and resetting recurrent module proposers."""

    def _make_bank(self, recurrent: tuple[str, ...] = ()) -> CorticalModuleBank:
        rng = np.random.default_rng(44)
        return CorticalModuleBank(
            action_dim=len(LOCOMOTION_ACTIONS),
            rng=rng,
            module_dropout=0.0,
            recurrent_modules=recurrent,
        )

    def test_recurrent_module_uses_recurrent_network(self) -> None:
        bank = self._make_bank(("alert_center",))
        self.assertIsInstance(bank.modules["alert_center"], RecurrentProposalNetwork)
        self.assertNotIsInstance(bank.modules["hunger_center"], RecurrentProposalNetwork)

    def test_has_recurrent_modules_reflects_configuration(self) -> None:
        self.assertFalse(self._make_bank(()).has_recurrent_modules)
        self.assertTrue(self._make_bank(("alert_center",)).has_recurrent_modules)

    def test_unknown_recurrent_module_raises_value_error(self) -> None:
        rng = np.random.default_rng(44)
        with self.assertRaisesRegex(ValueError, "nonexistent_module"):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                recurrent_modules=("nonexistent_module",),
            )

    def test_reset_hidden_states_clears_recurrent_modules(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 1.0
        bank.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_snapshot_and_restore_hidden_states_round_trip(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 0.25
        snapshot = bank.snapshot_hidden_states()
        recurrent.hidden_state[:] = 0.75
        bank.restore_hidden_states(snapshot)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.full(recurrent.hidden_dim, 0.25, dtype=float),
        )

    def test_spider_brain_reset_hidden_states_delegates_to_module_bank(self) -> None:
        brain = SpiderBrain(seed=5, module_dropout=0.0)
        brain.module_bank = self._make_bank(("alert_center",))
        recurrent = brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 1.0
        brain.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_spider_brain_estimate_value_does_not_commit_hidden_state(self) -> None:
        brain = SpiderBrain(seed=5, module_dropout=0.0)
        brain.module_bank = self._make_bank(("alert_center",))
        recurrent = brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 0.5
        observation = _build_observation()
        brain.estimate_value(observation)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.full(recurrent.hidden_dim, 0.5, dtype=float),
        )

    def test_recurrent_hidden_state_accumulates_within_episode_and_resets_between_episodes(self) -> None:
        config = BrainAblationConfig(
            name="recurrent_integration",
            architecture="modular",
            module_dropout=0.0,
            recurrent_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=17, max_steps=6, brain_config=config)
        observation = sim.world.reset(seed=17)
        sim.brain.reset_hidden_states()
        recurrent = sim.brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

        first_step = sim.brain.act(observation, sample=False)
        hidden_after_first = recurrent.hidden_state.copy()
        self.assertGreater(float(np.linalg.norm(hidden_after_first)), 0.0)

        next_observation, _, _, _ = sim.world.step(first_step.action_idx)
        sim.brain.act(next_observation, sample=False)
        hidden_after_second = recurrent.hidden_state.copy()
        self.assertGreater(float(np.linalg.norm(hidden_after_second)), 0.0)
        self.assertFalse(np.allclose(hidden_after_second, hidden_after_first))

        sim.world.reset(seed=18)
        sim.brain.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_snapshot_returns_empty_dict_when_no_recurrent_modules(self) -> None:
        bank = self._make_bank(())
        snapshot = bank.snapshot_hidden_states()
        self.assertEqual(snapshot, {})

    def test_restore_raises_key_error_for_unknown_module(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        snapshot = {"nonexistent_module": np.zeros(recurrent.hidden_dim)}
        with self.assertRaises(KeyError):
            bank.restore_hidden_states(snapshot)

    def test_restore_raises_value_error_for_non_recurrent_module(self) -> None:
        bank = self._make_bank(("alert_center",))
        # hunger_center is feed-forward; it has no hidden_state attribute
        non_recurrent = bank.modules["hunger_center"]
        self.assertNotIsInstance(non_recurrent, RecurrentProposalNetwork)
        snapshot = {"hunger_center": np.zeros(10)}
        with self.assertRaises(ValueError):
            bank.restore_hidden_states(snapshot)

    def test_multiple_recurrent_modules_all_get_recurrent_networks(self) -> None:
        bank = self._make_bank(("alert_center", "sleep_center", "hunger_center"))
        for name in ("alert_center", "sleep_center", "hunger_center"):
            self.assertIsInstance(bank.modules[name], RecurrentProposalNetwork,
                                  f"{name} should be RecurrentProposalNetwork")
        # Feed-forward modules should remain feed-forward
        for name in ("visual_cortex", "sensory_cortex"):
            self.assertNotIsInstance(bank.modules[name], RecurrentProposalNetwork,
                                     f"{name} should not be RecurrentProposalNetwork")

    def test_snapshot_returns_copies_not_live_references(self) -> None:
        bank = self._make_bank(("alert_center",))
        recurrent = bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        recurrent.hidden_state[:] = 0.3
        snapshot = bank.snapshot_hidden_states()
        # Mutate live hidden state after snapshot
        recurrent.hidden_state[:] = 0.9
        # Snapshot should retain the value at snapshot time
        np.testing.assert_allclose(
            snapshot["alert_center"],
            np.full(recurrent.hidden_dim, 0.3, dtype=float),
        )

    def test_reset_hidden_states_does_not_affect_feedforward_module_identity(self) -> None:
        # Resetting hidden states should not raise and must leave feed-forward modules intact
        bank = self._make_bank(("alert_center",))
        ff_before = bank.modules["hunger_center"]
        bank.reset_hidden_states()
        ff_after = bank.modules["hunger_center"]
        self.assertIs(ff_before, ff_after)

    def test_spider_brain_reset_hidden_states_is_noop_with_no_module_bank(self) -> None:
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        brain.module_bank = None
        # Should not raise
        brain.reset_hidden_states()

    def test_run_episode_resets_recurrent_hidden_state_at_start(self) -> None:
        config = BrainAblationConfig(
            name="recurrent_episode_reset",
            architecture="modular",
            module_dropout=0.0,
            recurrent_modules=("alert_center",),
        )
        sim = SpiderSimulation(seed=19, max_steps=3, brain_config=config)
        recurrent = sim.brain.module_bank.modules["alert_center"]
        self.assertIsInstance(recurrent, RecurrentProposalNetwork)
        dirty_state = np.full(recurrent.hidden_dim, 99.0, dtype=float)
        recurrent.hidden_state[:] = dirty_state
        reset_snapshots: list[np.ndarray] = []
        original_reset_hidden_states = sim.brain.reset_hidden_states

        def wrapped_reset_hidden_states() -> None:
            original_reset_hidden_states()
            reset_snapshots.append(recurrent.get_hidden_state())

        with mock.patch.object(
            sim.brain,
            "reset_hidden_states",
            side_effect=wrapped_reset_hidden_states,
        ) as reset_mock:
            sim.run_episode(0, training=False, sample=False)

        self.assertEqual(reset_mock.call_count, 1)
        self.assertEqual(len(reset_snapshots), 1)
        np.testing.assert_allclose(
            reset_snapshots[0],
            np.zeros(recurrent.hidden_dim, dtype=float),
        )
        self.assertFalse(np.allclose(reset_snapshots[0], dirty_state))
        self.assertTrue(np.all(np.isfinite(recurrent.hidden_state)))

class ModuleResultNamedObservationTest(unittest.TestCase):
    """Tests for ModuleResult.named_observation (new None-guard added in this PR)."""

    def test_named_observation_returns_empty_dict_for_none_interface(self) -> None:
        result = ModuleResult(
            interface=None,
            name="monolithic_policy",
            observation_key="monolithic_policy",
            observation=np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            logits=np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            probs=np.full(
                len(LOCOMOTION_ACTIONS),
                1.0 / len(LOCOMOTION_ACTIONS),
                dtype=float,
            ),
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
            logits=np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            probs=np.full(
                len(LOCOMOTION_ACTIONS),
                1.0 / len(LOCOMOTION_ACTIONS),
                dtype=float,
            ),
            active=True,
        )
        named = result.named_observation()
        self.assertIsInstance(named, dict)
        self.assertIn("predator_visible", named)

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

    def test_training_td_update_publishes_credit_diagnostics(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=2)

        sim.run_episode(
            episode_index=0,
            training=True,
            sample=False,
        )

        td_updates = [
            message.payload
            for message in sim.bus.history()
            if message.topic == "td_update"
        ]
        self.assertGreater(len(td_updates), 0)
        payload = td_updates[0]
        self.assertIn("module_credit_weights", payload)
        self.assertIn("module_gradient_norms", payload)
        self.assertEqual(payload["credit_strategy"], "broadcast")

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
