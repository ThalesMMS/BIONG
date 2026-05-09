from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    COARSE_ROLLUP_MODULES,
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
from spider_cortex_sim.direct_policy_affordances import (
    AFFORDANCE_SHELTER_POSITION_NAMES,
    DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM,
    DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM,
    DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,
    DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,
    DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM,
    DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM,
    DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM,
)
from spider_cortex_sim.direct_policy_options import OPTION_NAMES
from spider_cortex_sim.interfaces import (
    ACTION_TO_INDEX,
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import (
    RecurrentEventAttentionTrueMonolithicNetwork,
    RecurrentOptionAffordanceFeedbackTrueMonolithicNetwork,
    RecurrentOptionAffordanceGeometryFeedbackTrueMonolithicNetwork,
    RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
    RecurrentOptionAffordanceTopologyFeedbackTrueMonolithicNetwork,
    RecurrentOptionAffordanceTrueMonolithicNetwork,
    RecurrentOptionTrueMonolithicNetwork,
    RecurrentProposalNetwork,
    RecurrentTrueMonolithicNetwork,
)
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.phase import PHASE_TO_INDEX
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

    def test_disabled_module_is_removed_from_runtime_results(self) -> None:
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
        self.assertTrue(decision.module_results)
        self.assertNotIn(
            "alert_center",
            [result.name for result in decision.module_results],
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
            disabled_modules=COARSE_ROLLUP_MODULES,
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

    def test_unknown_hidden_dim_override_raises_value_error(self) -> None:
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(ValueError, "nonexistent_module"):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                hidden_dims={"nonexistent_module": 16},
            )

    def test_non_positive_hidden_dim_override_raises_value_error(self) -> None:
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(ValueError, "int > 0"):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                hidden_dims={"visual_cortex": 0},
            )

    def test_all_disabled_modules_raises_value_error(self) -> None:
        rng = np.random.default_rng(42)
        with self.assertRaisesRegex(
            ValueError,
            "all modules were disabled; at least one module must be enabled",
        ):
            CorticalModuleBank(
                action_dim=len(LOCOMOTION_ACTIONS),
                rng=rng,
                module_dropout=0.0,
                disabled_modules=MODULE_NAMES,
            )

    def test_disabled_module_preserves_canonical_specs_and_prunes_enabled_specs(self) -> None:
        bank = self._make_bank(("hunger_center",))
        self.assertIn("hunger_center", [spec.name for spec in bank.specs])
        self.assertNotIn("hunger_center", [spec.name for spec in bank.enabled_specs])

    def test_disabled_module_is_absent_from_networks_and_results(self) -> None:
        """
        Verifies that a disabled module is removed from the bank rather than kept as an inactive zero-logit slot.
        
        Asserts the disabled module is absent from both the instantiated network map and the
        returned ModuleResult list.
        """
        bank = self._make_bank(("hunger_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        self.assertNotIn("hunger_center", bank.modules)
        self.assertNotIn("hunger_center", [result.name for result in results])

    def test_disabled_module_not_returned(self) -> None:
        bank = self._make_bank(("sleep_center",))
        obs = self._blank_observation()
        results = bank.forward(obs, store_cache=False, training=False)
        self.assertNotIn("sleep_center", [result.name for result in results])

    def test_load_state_dict_ignores_modules_absent_from_bank(self) -> None:
        source_bank = self._make_bank(())
        target_bank = self._make_bank(("hunger_center",))
        loaded = target_bank.load_state_dict(
            source_bank.state_dict(),
            modules=list(source_bank.state_dict().keys()),
        )
        expected_loaded = set(source_bank.state_dict().keys()) & set(target_bank.state_dict().keys())
        self.assertSetEqual(set(loaded), expected_loaded)

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
        self.assertEqual(len(active_names), 1)
        self.assertNotIn("sensory_cortex", active_names)
        self.assertNotIn("hunger_center", active_names)
        self.assertNotIn("sleep_center", active_names)
        self.assertNotIn("alert_center", active_names)

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
        brain = SpiderBrain(
            seed=5,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="full_modular_recurrent_test",
                module_dropout=0.0,
                disabled_modules=(),
            ),
        )
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

    def test_true_monolithic_recurrent_policy_uses_recurrent_direct_network(self) -> None:
        brain = SpiderBrain(
            seed=23,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_recurrent_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
            ),
        )
        self.assertIsInstance(brain.true_monolithic_policy, RecurrentTrueMonolithicNetwork)

    def test_true_monolithic_recurrent_reset_hidden_states_clears_hidden_state(self) -> None:
        brain = SpiderBrain(
            seed=24,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_recurrent_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentTrueMonolithicNetwork)
        recurrent.hidden_state[:] = 1.0
        brain.reset_hidden_states()
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.zeros(recurrent.hidden_dim, dtype=float),
        )

    def test_true_monolithic_recurrent_estimate_value_does_not_commit_hidden_state(self) -> None:
        brain = SpiderBrain(
            seed=25,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_recurrent_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentTrueMonolithicNetwork)
        recurrent.hidden_state[:] = 0.5
        observation = _build_observation()
        brain.estimate_value(observation)
        np.testing.assert_allclose(
            recurrent.hidden_state,
            np.full(recurrent.hidden_dim, 0.5, dtype=float),
        )

    def test_true_monolithic_recurrent_inference_is_deterministic_given_same_state(self) -> None:
        config = BrainAblationConfig(
            name="true_monolithic_recurrent_policy",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
        )
        brain = SpiderBrain(seed=26, module_dropout=0.0, config=config)
        observation = _build_observation()
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentTrueMonolithicNetwork)
        seeded_state = np.linspace(-0.2, 0.2, recurrent.hidden_dim, dtype=float)
        recurrent.set_hidden_state(seeded_state)
        first = brain.act_inference(observation, sample=False)
        hidden_after_first = recurrent.get_hidden_state()
        recurrent.set_hidden_state(seeded_state)
        second = brain.act_inference(observation, sample=False)
        hidden_after_second = recurrent.get_hidden_state()
        np.testing.assert_allclose(first.total_logits, second.total_logits)
        self.assertEqual(first.action_idx, second.action_idx)
        np.testing.assert_allclose(hidden_after_first, hidden_after_second)

    def test_true_monolithic_recurrent_phase_policy_exposes_phase_predictions(self) -> None:
        brain = SpiderBrain(
            seed=27,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_recurrent_phase_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
            ),
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentTrueMonolithicNetwork)
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)
        self.assertGreaterEqual(decision.phase_prediction_confidence, 0.0)
        self.assertLessEqual(decision.phase_prediction_confidence, 1.0)

    def test_true_monolithic_event_attention_policy_exposes_top_event(self) -> None:
        brain = SpiderBrain(
            seed=28,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_event_attention_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentEventAttentionTrueMonolithicNetwork,
        )
        brain.set_direct_policy_event_clock(4)
        brain.record_direct_policy_event(
            "REST_STARTED",
            features=np.array([1.0, 0.4, 0.1, 1.0, 0.0], dtype=float),
            tick=4,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.event_attention_top_type, "REST_STARTED")
        self.assertEqual(decision.event_attention_top_age, 0)
        self.assertGreaterEqual(decision.event_attention_entropy, 0.0)

    def test_true_monolithic_option_policy_exposes_option_commitment(self) -> None:
        brain = SpiderBrain(
            seed=29,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(recurrent, RecurrentOptionTrueMonolithicNetwork)
        brain.set_direct_policy_event_clock(6)
        brain.record_direct_policy_event(
            "RECOVERY_COMPLETED",
            features=np.array([1.0, 0.6, 0.1, 1.0, 0.0], dtype=float),
            tick=6,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertIn(decision.selected_option, OPTION_NAMES)
        self.assertEqual(decision.option_age, 0)
        self.assertEqual(decision.option_termination_reason, "initial_selection")
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(
            recurrent.last_option_summary["selected_option"],
            decision.selected_option,
        )

    def test_true_monolithic_option_affordance_policy_exposes_affordance_logits(self) -> None:
        brain = SpiderBrain(
            seed=30,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordanceTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(
            decision.affordance_blocked_logits.shape,
            (len(LOCOMOTION_ACTIONS),),
        )
        self.assertEqual(
            decision.affordance_role_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 4,),
        )

    def test_true_monolithic_option_affordance_feedback_policy_exposes_affordance_logits(self) -> None:
        brain = SpiderBrain(
            seed=31,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_feedback_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordanceFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertIn(decision.selected_option, OPTION_NAMES)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(
            decision.affordance_blocked_logits.shape,
            (len(LOCOMOTION_ACTIONS),),
        )
        self.assertEqual(
            decision.affordance_role_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 4,),
        )

    def test_true_monolithic_option_affordance_geometry_policy_exposes_geometry_logits(self) -> None:
        brain = SpiderBrain(
            seed=32,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_geometry_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordanceGeometryFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(
            decision.geometry_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 3,),
        )

    def test_true_monolithic_option_affordance_topology_policy_exposes_shelter_column_logits(self) -> None:
        brain = SpiderBrain(
            seed=33,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_topology_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_column_head=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordanceTopologyFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(
            decision.shelter_column_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 4,),
        )

    def test_true_monolithic_option_affordance_position_policy_exposes_shelter_position_logits(self) -> None:
        brain = SpiderBrain(
            seed=34,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(
            decision.shelter_position_logits.shape,
            (len(LOCOMOTION_ACTIONS) * 10,),
        )

    def test_true_monolithic_option_affordance_position_teacher_policy_uses_position_network(self) -> None:
        brain = SpiderBrain(
            seed=35,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_teacher_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )

    def test_true_monolithic_option_affordance_position_teacher_option_policy_uses_position_network(self) -> None:
        brain = SpiderBrain(
            seed=36,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_teacher_option_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )

    def test_true_monolithic_executive_option_guarded_policy_skips_food_direction_bias(self) -> None:
        common_fields = dict(
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_phase_head=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_phase_option_feedback=True,
            direct_policy_option_transition_feedback=True,
            direct_policy_option_termination_cooldown=True,
            direct_policy_option_decoder_state=True,
            direct_policy_option_recurrent_dynamics=True,
            direct_policy_option_decoder_recurrent_state=True,
            direct_policy_option_action_recurrent_core=True,
            direct_policy_option_action_separate_recurrent_head=True,
            direct_policy_option_action_separate_policy_path=True,
            direct_policy_option_action_separate_backbone=True,
        )
        control_brain = SpiderBrain(
            seed=301,
            module_dropout=0.0,
            config=BrainAblationConfig(name="control_executive_bias_on", **common_fields),
        )
        guarded_brain = SpiderBrain(
            seed=301,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_guarded_policy",
                **common_fields,
            ),
        )
        observation = _build_observation()
        control_net = control_brain.true_monolithic_policy
        guarded_net = guarded_brain.true_monolithic_policy
        self.assertIsNotNone(control_net)
        self.assertIsNotNone(guarded_net)

        def _fixed_forward(net):
            policy_logits = np.array(
                [-2.0, -2.0, -2.0, -2.0, 0.5, -2.0, -2.0, -2.0, -2.0],
                dtype=float,
            )
            option_logits = np.zeros(net.option_dim, dtype=float)
            phase_logits = np.zeros(net.phase_output_dim, dtype=float)
            return policy_logits, 0.0, option_logits, phase_logits

        control_net.forward = lambda *_args, **_kwargs: _fixed_forward(control_net)
        guarded_net.forward = lambda *_args, **_kwargs: _fixed_forward(guarded_net)
        control_brain._threat_escape_bias_action = lambda _obs: None
        control_brain._sleep_rest_bias_action = lambda _obs: None
        control_brain._food_direction_bias_action = lambda _obs: "MOVE_RIGHT"
        guarded_brain._threat_escape_bias_action = lambda _obs: None
        guarded_brain._sleep_rest_bias_action = lambda _obs: None
        guarded_brain._food_direction_bias_action = lambda _obs: "MOVE_RIGHT"

        control_decision = control_brain.act(observation, sample=False)
        guarded_decision = guarded_brain.act(observation, sample=False)

        self.assertEqual(LOCOMOTION_ACTIONS[control_decision.action_idx], "MOVE_RIGHT")
        self.assertEqual(LOCOMOTION_ACTIONS[guarded_decision.action_idx], "STAY")

    def test_executive_physiology_option_gating_prefers_rest_inside_shelter(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)

        def _build_monolithic_input() -> np.ndarray:
            vectors = []
            for spec in MODULE_INTERFACES:
                mapping = {name: 0.0 for name in spec.signal_names}
                if spec.name == "sleep_center":
                    mapping.update(
                        {
                            "fatigue": 0.35,
                            "hunger": 0.1,
                            "on_shelter": 1.0,
                            "night": 1.0,
                            "sleep_phase_level": 0.4,
                            "rest_streak_norm": 0.6,
                            "sleep_debt": 0.3,
                            "shelter_role_level": 1.0,
                            "shelter_memory_age": 0.0,
                        }
                    )
                vectors.append(spec.vector_from_mapping(mapping))
            return np.concatenate(vectors, axis=0)

        def _build_network(*, gated: bool):
            net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
                input_dim=input_dim,
                hidden_dim=8,
                output_dim=len(LOCOMOTION_ACTIONS),
                rng=np.random.default_rng(7),
                event_buffer_size=2,
                option_ttl=4,
                executive_physiology_option_gating=gated,
                name="test_true_monolithic_policy",
            )
            net.W_xh.fill(0.0)
            net.W_hh.fill(0.0)
            net.b_h.fill(0.0)
            net.W2_option.fill(0.0)
            net.b2_option.fill(0.0)
            net.W2_option_feedback.fill(0.0)
            net.b2_option_feedback.fill(0.0)
            net.W_affordance_feedback.fill(0.0)
            net.b_affordance_feedback.fill(0.0)
            net.b2_option[OPTION_NAMES.index("ESCAPE")] = 4.0
            return net

        x = _build_monolithic_input()
        ungated = _build_network(gated=False)
        gated = _build_network(gated=True)
        ungated.forward(x, store_cache=False)
        gated.forward(x, store_cache=False)

        self.assertEqual(ungated.last_option_summary["selected_option"], "ESCAPE")
        self.assertEqual(gated.last_option_summary["selected_option"], "REST")

    def test_executive_release_action_gating_prefers_move_up_after_rest(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)

        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.05,
                        "hunger": 0.28,
                        "on_shelter": 1.0,
                        "night": 0.0,
                        "sleep_phase_level": 0.0,
                        "rest_streak_norm": 0.5,
                        "sleep_debt": 0.04,
                        "shelter_role_level": 1.0,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(8),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            name="test_true_monolithic_policy",
        )
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_DOWN"]] = 5.0
        net.b2_option[OPTION_NAMES.index("POST_REST_REACTIVATE")] = 6.0
        net.W2_shelter_position.fill(0.0)
        net.b2_shelter_position.fill(0.0)
        move_up_idx = ACTION_TO_INDEX["MOVE_UP"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        offset = move_up_idx * len(AFFORDANCE_SHELTER_POSITION_NAMES)
        net.b2_shelter_position[offset + outside_idx] = 6.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(net.last_option_summary["selected_option"], "POST_REST_REACTIVATE")
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "MOVE_UP",
        )

    def test_executive_masked_release_action_gating_prefers_stay_for_rest(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.35,
                        "hunger": 0.1,
                        "on_shelter": 1.0,
                        "night": 1.0,
                        "sleep_phase_level": 0.4,
                        "rest_streak_norm": 0.6,
                        "sleep_debt": 0.3,
                        "shelter_role_level": 1.0,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(9),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            executive_option_action_masking=True,
            name="test_true_monolithic_policy",
        )
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_UP"]] = 12.0
        net.b2_option[OPTION_NAMES.index("REST")] = 6.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(net.last_option_summary["selected_option"], "REST")
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "STAY",
        )

    def test_executive_event_release_latch_promotes_post_rest_move_up(self) -> None:
        brain = SpiderBrain(
            seed=47,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_event_release_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("DEEPEN_IN_SHELTER")
        recurrent.current_option_age = 2
        recurrent.current_option_steps_remaining = 1
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["ORIENT_DOWN"]] = 6.0
        recurrent.b2_option[OPTION_NAMES.index("DEEPEN_IN_SHELTER")] = 8.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        move_up_idx = ACTION_TO_INDEX["MOVE_UP"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        offset = move_up_idx * len(AFFORDANCE_SHELTER_POSITION_NAMES)
        recurrent.b2_shelter_position[offset + outside_idx] = 6.0
        brain.set_direct_policy_event_clock(7)
        brain.record_direct_policy_event(
            "DEEP_SLEEP_REACHED",
            features=np.array([1.0, 0.4, 0.0, 1.0, 0.0], dtype=float),
            tick=7,
        )
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.22,
                    "on_shelter": 1.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 1.0,
                    "shelter_memory_age": 0.0,
                }
            ),
            sample=False,
        )
        self.assertEqual(decision.option_termination_reason, "deep_shelter_reached")
        self.assertEqual(decision.selected_option, "POST_REST_REACTIVATE")
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_transition_release_commitment_forces_move_up(self) -> None:
        brain = SpiderBrain(
            seed=49,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_transition_release_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.executive_release_steps_remaining = 2
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 9.0
        recurrent.b2_option[OPTION_NAMES.index("RETURN_TO_SHELTER")] = 8.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        move_up_idx = ACTION_TO_INDEX["MOVE_UP"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        offset = move_up_idx * len(AFFORDANCE_SHELTER_POSITION_NAMES)
        recurrent.b2_shelter_position[offset + outside_idx] = 6.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.22,
                    "on_shelter": 1.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 1.0,
                    "shelter_memory_age": 0.0,
                }
            ),
            sample=False,
        )
        self.assertIn(
            decision.selected_option,
            {"RETURN_TO_SHELTER", "POST_REST_REACTIVATE"},
        )
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_release_phase_state_arms_move_up_at_start(self) -> None:
        brain = SpiderBrain(
            seed=50,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_release_phase_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["ORIENT_UP"]] = 9.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        move_up_idx = ACTION_TO_INDEX["MOVE_UP"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        offset = move_up_idx * len(AFFORDANCE_SHELTER_POSITION_NAMES)
        recurrent.b2_shelter_position[offset + outside_idx] = 6.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 1.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.0,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 1.0,
                    "shelter_memory_age": 0.0,
                }
            ),
            sample=False,
        )
        self.assertGreater(recurrent.executive_release_steps_remaining, 0)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_release_progression_prefers_move_right_at_entrance(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.05,
                        "hunger": 0.18,
                        "on_shelter": 1.0,
                        "night": 0.0,
                        "sleep_phase_level": 0.0,
                        "rest_streak_norm": 0.0,
                        "sleep_debt": 0.04,
                        "shelter_role_level": 0.33,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(10),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            executive_option_action_masking=True,
            executive_event_release_latching=True,
            executive_event_release_action_commitment=True,
            executive_release_phase_state=True,
            executive_release_progression=True,
            name="test_true_monolithic_policy",
        )
        net.executive_release_steps_remaining = 2
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_RIGHT"]] = 12.0
        net.W2_shelter_position.fill(0.0)
        net.b2_shelter_position.fill(0.0)
        stay_idx = ACTION_TO_INDEX["STAY"]
        entrance_center_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("entrance_center")
        move_right_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        pos_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
        net.b2_shelter_position[stay_idx * pos_dim + entrance_center_idx] = 6.0
        net.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 6.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "MOVE_RIGHT",
        )

    def test_executive_release_exit_contract_prefers_move_left_at_entrance(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.05,
                        "hunger": 0.18,
                        "on_shelter": 1.0,
                        "night": 0.0,
                        "sleep_phase_level": 0.0,
                        "rest_streak_norm": 0.0,
                        "sleep_debt": 0.04,
                        "shelter_role_level": 0.33,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(11),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            executive_option_action_masking=True,
            executive_event_release_latching=True,
            executive_event_release_action_commitment=True,
            executive_release_phase_state=True,
            executive_release_exit_contract=True,
            name="test_true_monolithic_policy",
        )
        net.executive_release_steps_remaining = 2
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_RIGHT"]] = 12.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "MOVE_LEFT",
        )

    def test_executive_release_substate_progression_keeps_move_left_at_entrance_with_last_step(self) -> None:
        input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
        vectors = []
        for spec in MODULE_INTERFACES:
            mapping = {name: 0.0 for name in spec.signal_names}
            if spec.name == "sleep_center":
                mapping.update(
                    {
                        "fatigue": 0.05,
                        "hunger": 0.18,
                        "on_shelter": 1.0,
                        "night": 0.0,
                        "sleep_phase_level": 0.0,
                        "rest_streak_norm": 0.0,
                        "sleep_debt": 0.04,
                        "shelter_role_level": 1.0 / 3.0,
                        "shelter_memory_age": 0.0,
                    }
                )
            vectors.append(spec.vector_from_mapping(mapping))
        x = np.concatenate(vectors, axis=0)

        net = RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork(
            input_dim=input_dim,
            hidden_dim=8,
            output_dim=len(LOCOMOTION_ACTIONS),
            rng=np.random.default_rng(12),
            event_buffer_size=2,
            option_ttl=4,
            executive_physiology_option_gating=True,
            executive_affordance_action_gating=True,
            executive_option_action_masking=True,
            executive_event_release_latching=True,
            executive_event_release_action_commitment=True,
            executive_release_phase_state=True,
            executive_release_exit_contract=True,
            executive_release_substate_progression=True,
            name="test_true_monolithic_policy",
        )
        net.executive_release_steps_remaining = 1
        net.W_xh.fill(0.0)
        net.W_hh.fill(0.0)
        net.b_h.fill(0.0)
        net.W2_option.fill(0.0)
        net.b2_option.fill(0.0)
        net.W2_option_feedback.fill(0.0)
        net.b2_option_feedback.fill(0.0)
        net.W_affordance_feedback.fill(0.0)
        net.b_affordance_feedback.fill(0.0)
        net.W2_policy.fill(0.0)
        net.b2_policy.fill(0.0)
        net.b2_policy[ACTION_TO_INDEX["ORIENT_RIGHT"]] = 12.0
        policy_logits = net.forward(x, store_cache=False)[0]
        self.assertEqual(
            LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))],
            "MOVE_LEFT",
        )

    def test_executive_post_exit_continuation_preserves_post_rest_option_after_exit(self) -> None:
        brain = SpiderBrain(
            seed=112,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_continuation_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 3
        recurrent.current_option_steps_remaining = 1
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 8.0
        recurrent.record_event(
            "SHELTER_EXIT",
            features=np.zeros(recurrent.event_feature_dim, dtype=float),
            tick=0,
        )
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                }
            ),
            sample=False,
        )
        self.assertEqual(decision.selected_option, "POST_REST_REACTIVATE")
        self.assertGreater(recurrent.executive_post_exit_steps_remaining, 0)

    def test_executive_post_exit_food_guidance_prefers_move_right_from_food_memory(self) -> None:
        brain = SpiderBrain(
            seed=115,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_food_guidance_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.executive_post_exit_steps_remaining = 2
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 10.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                },
                hunger={
                    "food_memory_dx": 0.8,
                    "food_memory_dy": 0.1,
                    "food_memory_age": 0.1,
                },
            ),
            sample=False,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_RIGHT")

    def test_executive_post_exit_food_commitment_extends_outside_window(self) -> None:
        brain = SpiderBrain(
            seed=118,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_food_commitment_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
                direct_policy_executive_post_exit_food_commitment=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 3
        recurrent.current_option_steps_remaining = 0
        recurrent.executive_post_exit_steps_remaining = 1
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 10.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                },
                hunger={
                    "food_memory_dx": 0.8,
                    "food_memory_dy": 0.1,
                    "food_memory_age": 0.1,
                },
            ),
            sample=False,
        )
        self.assertEqual(decision.selected_option, "POST_REST_REACTIVATE")
        self.assertGreaterEqual(recurrent.executive_post_exit_steps_remaining, 1)

    def test_executive_post_exit_food_progression_prefers_move_up_over_memory_reversal(self) -> None:
        brain = SpiderBrain(
            seed=121,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_food_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
                direct_policy_executive_post_exit_food_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.executive_post_exit_steps_remaining = 2
        recurrent.previous_action_idx = ACTION_TO_INDEX["MOVE_LEFT"]
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 10.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                },
                hunger={
                    "food_memory_dx": 0.8,
                    "food_memory_dy": 0.1,
                    "food_memory_age": 0.1,
                },
            ),
            sample=False,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_post_exit_food_heading_progression_prefers_move_up_after_right_recenter(self) -> None:
        brain = SpiderBrain(
            seed=124,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_food_heading_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
                direct_policy_executive_post_exit_food_heading_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.executive_post_exit_steps_remaining = 2
        recurrent.previous_action_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["ORIENT_UP"]] = 10.0
        decision = brain.act_inference(
            _build_observation(
                sleep={
                    "fatigue": 0.05,
                    "hunger": 0.18,
                    "on_shelter": 0.0,
                    "night": 0.0,
                    "sleep_phase_level": 0.0,
                    "rest_streak_norm": 0.6,
                    "sleep_debt": 0.04,
                    "shelter_role_level": 0.0,
                    "shelter_memory_age": 0.0,
                },
                hunger={
                    "food_memory_dx": 0.8,
                    "food_memory_dy": 0.1,
                    "food_memory_age": 0.1,
                },
            ),
            sample=False,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_UP")

    def test_executive_post_exit_smell_progression_extends_outside_window(self) -> None:
        brain = SpiderBrain(
            seed=127,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_smell_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_food_guidance=True,
                direct_policy_executive_post_exit_smell_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.18,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.6,
                "sleep_debt": 0.04,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.0,
            },
            hunger={
                "food_smell_strength": 0.4,
                "food_smell_dx": 0.8,
                "food_smell_dy": 0.1,
                "food_memory_age": 0.95,
            },
        )
        monolithic_observation = brain._build_monolithic_observation(observation)
        termination_reason = recurrent._apply_executive_post_exit_continuation(
            monolithic_observation,
            "shelter_exited",
        )
        self.assertIsNone(termination_reason)
        self.assertEqual(recurrent.executive_post_exit_steps_remaining, 3)

    def test_executive_post_exit_corridor_progression_prefers_blind_rightward_advance(self) -> None:
        brain = SpiderBrain(
            seed=128,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_corridor_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_exit_steps_remaining = 7
        recurrent.executive_post_exit_corridor_steps_remaining = 7
        observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.18,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.6,
                "sleep_debt": 0.04,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.0,
            },
            hunger={
                "food_smell_strength": 0.0,
                "food_memory_age": 1.0,
            },
        )
        decision = brain.act_inference(observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_RIGHT")

    def test_executive_post_exit_corridor_affordance_progression_delays_drop_until_outside(self) -> None:
        brain = SpiderBrain(
            seed=129,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_exit_corridor_affordance_progression_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
                direct_policy_executive_post_exit_corridor_affordance_progression=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_exit_steps_remaining = 7
        recurrent.executive_post_exit_corridor_steps_remaining = 7
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 10.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        pos_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
        move_down_idx = ACTION_TO_INDEX["MOVE_DOWN"]
        move_right_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        entrance_center_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("entrance_center")
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        recurrent.b2_shelter_position[move_down_idx * pos_dim + entrance_center_idx] = 6.0
        recurrent.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 6.0

        shelter_observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.18,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.6,
                "sleep_debt": 0.04,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.0,
            },
            hunger={
                "food_smell_strength": 0.0,
                "food_memory_age": 1.0,
            },
        )
        decision = brain.act_inference(shelter_observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_RIGHT")

        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("POST_REST_REACTIVATE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_exit_steps_remaining = 4
        recurrent.executive_post_exit_corridor_steps_remaining = 4
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 10.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position[move_down_idx * pos_dim + outside_idx] = 6.0
        recurrent.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 6.0
        outside_observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.18,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.6,
                "sleep_debt": 0.04,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.0,
            },
            hunger={
                "food_smell_strength": 0.0,
                "food_memory_age": 1.0,
            },
        )
        decision = brain.act_inference(outside_observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_DOWN")

    def test_executive_post_food_return_prefers_shelterward_recovery_and_deep_stay(self) -> None:
        brain = SpiderBrain(
            seed=130,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_food_return_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
                direct_policy_executive_post_exit_corridor_affordance_progression=True,
                direct_policy_executive_post_food_return=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("RETURN_TO_SHELTER")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_food_return_steps_remaining = 8
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 10.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        pos_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
        move_left_idx = ACTION_TO_INDEX["MOVE_LEFT"]
        move_right_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        stay_idx = ACTION_TO_INDEX["STAY"]
        deep_center_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("deep_center")
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        recurrent.b2_shelter_position[move_left_idx * pos_dim + deep_center_idx] = 8.0
        recurrent.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 8.0

        outside_observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.12,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.2,
                "sleep_debt": 0.05,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.1,
            },
            predator={
                "predator_visible": 0.0,
                "predator_proximity": 0.0,
                "predator_motion": 0.0,
                "predator_heading_alignment": 0.0,
                "predator_recent_contact": 0.0,
                "recent_pain": 0.0,
            },
        )
        decision = brain.act_inference(outside_observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[decision.action_idx], "MOVE_LEFT")

        recurrent.current_option_idx = OPTION_NAMES.index("DEEPEN_IN_SHELTER")
        recurrent.current_option_steps_remaining = 2
        recurrent.executive_post_food_return_steps_remaining = 4
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position[stay_idx * pos_dim + deep_center_idx] = 8.0
        deep_observation = _build_observation(
            sleep={
                "fatigue": 0.08,
                "hunger": 0.08,
                "on_shelter": 1.0,
                "night": 1.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.4,
                "sleep_debt": 0.08,
                "shelter_role_level": 1.0,
                "shelter_memory_age": 0.0,
            },
            predator={
                "predator_visible": 0.0,
                "predator_proximity": 0.0,
                "predator_motion": 0.0,
                "predator_heading_alignment": 0.0,
                "predator_recent_contact": 0.0,
                "recent_pain": 0.0,
            },
        )
        deep_decision = brain.act_inference(deep_observation, sample=False)
        self.assertEqual(LOCOMOTION_ACTIONS[deep_decision.action_idx], "STAY")

    def test_executive_post_food_vector_return_uses_shelter_memory_direction(self) -> None:
        brain = SpiderBrain(
            seed=131,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_food_vector_return_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
                direct_policy_executive_post_exit_corridor_affordance_progression=True,
                direct_policy_executive_post_food_return=True,
                direct_policy_executive_post_food_vector_return=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("RETURN_TO_SHELTER")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_food_return_steps_remaining = 8
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_RIGHT"]] = 10.0
        recurrent.W2_shelter_position.fill(0.0)
        recurrent.b2_shelter_position.fill(0.0)
        pos_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
        move_right_idx = ACTION_TO_INDEX["MOVE_RIGHT"]
        outside_idx = AFFORDANCE_SHELTER_POSITION_NAMES.index("outside")
        recurrent.b2_shelter_position[move_right_idx * pos_dim + outside_idx] = 8.0
        recurrent.set_runtime_observation_meta(
            {
                "memory_vectors": {
                    "shelter": {"dx": -0.8, "dy": 0.0, "age": 0.0, "ttl": 20}
                }
            }
        )

        observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.12,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.2,
                "sleep_debt": 0.05,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.1,
            },
            predator={
                "predator_visible": 0.0,
                "predator_proximity": 0.0,
                "predator_motion": 0.0,
                "predator_heading_alignment": 0.0,
                "predator_recent_contact": 0.0,
                "recent_pain": 0.0,
            },
        )
        monolithic_observation = brain._build_monolithic_observation(observation)
        policy_logits, *_ = recurrent.forward(monolithic_observation, store_cache=False)
        self.assertEqual(LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))], "MOVE_LEFT")

    def test_executive_post_food_path_return_replays_inverse_outbound_action(self) -> None:
        brain = SpiderBrain(
            seed=132,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_executive_option_post_food_path_return_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=False,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_executive_physiology_option_gating=True,
                direct_policy_executive_affordance_action_gating=True,
                direct_policy_executive_option_action_masking=True,
                direct_policy_executive_event_release_latching=True,
                direct_policy_executive_event_release_action_commitment=True,
                direct_policy_executive_release_phase_state=True,
                direct_policy_executive_release_exit_contract=True,
                direct_policy_executive_release_substate_progression=True,
                direct_policy_executive_post_exit_continuation=True,
                direct_policy_executive_post_exit_corridor_progression=True,
                direct_policy_executive_post_exit_corridor_affordance_progression=True,
                direct_policy_executive_post_food_return=True,
                direct_policy_executive_post_food_path_return=True,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("RETURN_TO_SHELTER")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 3
        recurrent.executive_post_food_return_steps_remaining = 8
        recurrent.executive_post_food_return_queue = [
            ACTION_TO_INDEX["MOVE_LEFT"]
        ]
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.b2_policy[ACTION_TO_INDEX["MOVE_DOWN"]] = 10.0

        observation = _build_observation(
            sleep={
                "fatigue": 0.05,
                "hunger": 0.12,
                "on_shelter": 0.0,
                "night": 0.0,
                "sleep_phase_level": 0.0,
                "rest_streak_norm": 0.2,
                "sleep_debt": 0.05,
                "shelter_role_level": 0.0,
                "shelter_memory_age": 0.1,
            },
            predator={
                "predator_visible": 0.0,
                "predator_proximity": 0.0,
                "predator_motion": 0.0,
                "predator_heading_alignment": 0.0,
                "predator_recent_contact": 0.0,
                "recent_pain": 0.0,
            },
        )
        monolithic_observation = brain._build_monolithic_observation(observation)
        policy_logits, *_ = recurrent.forward(monolithic_observation, store_cache=False)
        self.assertEqual(LOCOMOTION_ACTIONS[int(np.argmax(policy_logits))], "MOVE_LEFT")

    def test_true_monolithic_option_affordance_position_teacher_option_replay_policy_uses_position_network(self) -> None:
        brain = SpiderBrain(
            seed=37,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )

    def test_true_monolithic_option_affordance_position_phase_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=38,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_teacher_option_margin_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=39,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_teacher_option_margin_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
                direct_policy_continuation_margin_weight=1.0,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_feedback_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=40,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_feedback_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=42,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_cooldown_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=45,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_cooldown_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_action_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=47,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_action_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_action_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_decoder_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=49,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_decoder_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_decoder_feedback_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=51,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_decoder_feedback_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=52,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_action_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=54,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_action_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_action_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_sequence_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=56,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_sequence_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_sequence_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=60,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_transition_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=62,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_transition_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_transition_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_controller_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=66,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_controller_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_controller_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_token_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=70,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_token_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_token_decoder=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_core_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=74,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_core_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_recurrent_head_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=78,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_recurrent_head_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_policy_path_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=82,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_policy_path_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_teacher_option_replay_policy_exposes_phase_and_option_logits(self) -> None:
        brain = SpiderBrain(
            seed=86,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        self.assertIsInstance(
            brain.true_monolithic_policy,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_logits.shape, (len(OPTION_NAMES),))
        self.assertEqual(decision.phase_logits.shape, (8,))
        self.assertIsNotNone(decision.phase_prediction)

    def test_phase_option_feedback_affects_selected_option(self) -> None:
        brain = SpiderBrain(
            seed=41,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_feedback_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_phase_option_feedback=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W2_phase.fill(0.0)
        recurrent.b2_phase.fill(0.0)
        recurrent.W2_phase_option_feedback.fill(0.0)
        recurrent.b2_phase_option_feedback.fill(0.0)
        recurrent.b2_phase[int(PHASE_TO_INDEX["POST_REST_REACTIVATE"])] = 8.0
        recurrent.W2_phase_option_feedback[
            OPTION_NAMES.index("FORAGE"),
            int(PHASE_TO_INDEX["POST_REST_REACTIVATE"]),
        ] = 6.0
        recurrent.W2_phase_option_feedback[
            OPTION_NAMES.index("RETURN_TO_SHELTER"),
            int(PHASE_TO_INDEX["POST_REST_REACTIVATE"]),
        ] = -6.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.phase_prediction, "POST_REST_REACTIVATE")
        self.assertEqual(decision.selected_option, "FORAGE")

    def test_option_transition_feedback_affects_selected_option(self) -> None:
        brain = SpiderBrain(
            seed=43,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("RETURN_TO_SHELTER")
        recurrent.current_option_age = 2
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W2_option_transition_feedback.fill(0.0)
        recurrent.b2_option_transition_feedback.fill(0.0)
        recurrent.W2_option_transition_feedback[
            OPTION_NAMES.index("FORAGE"),
            OPTION_NAMES.index("RETURN_TO_SHELTER"),
        ] = 6.0
        recurrent.W2_option_transition_feedback[
            OPTION_NAMES.index("RETURN_TO_SHELTER"),
            OPTION_NAMES.index("RETURN_TO_SHELTER"),
        ] = -6.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")

    def test_option_transition_branch_holds_option_until_ttl_expires(self) -> None:
        brain = SpiderBrain(
            seed=44,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_transition_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_transition_feedback=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        decisions = [
            brain.act_inference(_build_observation(), sample=False)
            for _ in range(5)
        ]
        selected_option = decisions[0].selected_option
        self.assertEqual(decisions[0].option_age, 0)
        self.assertEqual(decisions[0].option_termination_reason, "initial_selection")
        for expected_age, decision in enumerate(decisions[1:4], start=1):
            self.assertEqual(decision.selected_option, selected_option)
            self.assertEqual(decision.option_age, expected_age)
            self.assertEqual(decision.option_termination_reason, "active")
        self.assertEqual(decisions[4].option_age, 0)
        self.assertEqual(decisions[4].option_termination_reason, "ttl_expired")

    def test_option_termination_cooldown_blocks_immediate_reselection(self) -> None:
        brain = SpiderBrain(
            seed=46,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_cooldown_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_termination_cooldown=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.current_option_idx = OPTION_NAMES.index("REST")
        recurrent.current_option_age = 2
        recurrent.current_option_steps_remaining = 1
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.b2_option[OPTION_NAMES.index("REST")] = 8.0
        recurrent.b2_option[OPTION_NAMES.index("FORAGE")] = 6.0
        brain.set_direct_policy_event_clock(9)
        brain.record_direct_policy_event(
            "RECOVERY_COMPLETED",
            features=np.array([1.0, 0.5, 0.1, 1.0, 0.0], dtype=float),
            tick=9,
        )
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.option_termination_reason, "recovery_completed")
        self.assertEqual(decision.selected_option, "FORAGE")

    def test_option_action_head_affects_selected_action(self) -> None:
        brain = SpiderBrain(
            seed=48,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_action_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_action_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.b2_option[OPTION_NAMES.index("FORAGE")] = 8.0
        recurrent.W2_option_action_head.fill(0.0)
        recurrent.b2_option_action_head.fill(0.0)
        recurrent.b2_option_action_head[
            OPTION_NAMES.index("FORAGE"),
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
        ] = 6.0
        recurrent.b2_option_action_head[
            OPTION_NAMES.index("FORAGE"),
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = -6.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_decoder_state_affects_selected_action(self) -> None:
        brain = SpiderBrain(
            seed=50,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_decoder_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.b_h[0] = 1.0
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.b2_option[OPTION_NAMES.index("FORAGE")] = 8.0
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.b_option_decoder_state[OPTION_NAMES.index("FORAGE"), 0] = 4.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_recurrent_dynamics_affects_selected_action(self) -> None:
        brain = SpiderBrain(
            seed=53,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.hidden_state[0] = 1.0
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_option_recurrent_dynamics[
            OPTION_NAMES.index("FORAGE"),
            0,
            0,
        ] = 4.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_sequence_head_affects_selected_action_by_option_age(self) -> None:
        brain = SpiderBrain(
            seed=57,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_sequence_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_sequence_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.hidden_state[0] = 1.0
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W2_option_sequence_head.fill(0.0)
        recurrent.b2_option_sequence_head.fill(0.0)
        recurrent.b2_option_sequence_head[
            OPTION_NAMES.index("FORAGE"),
            2,
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
        ] = 3.0
        recurrent.b2_option_sequence_head[
            OPTION_NAMES.index("FORAGE"),
            2,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = -3.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.option_age, 2)
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_decoder_recurrent_state_affects_selected_action(self) -> None:
        brain = SpiderBrain(
            seed=61,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 1
        recurrent.current_option_steps_remaining = 2
        recurrent.decoder_action_state.fill(0.0)
        recurrent.decoder_action_state[0] = 1.0
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_policy[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_option_decoder_recurrent_state[
            OPTION_NAMES.index("FORAGE"),
            0,
            0,
        ] = 4.0
        decision = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(decision.selected_option, "FORAGE")
        self.assertEqual(decision.option_age, 2)
        self.assertEqual(decision.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_transition_state_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=63,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_transition_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_transition_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_option_action_transition_state.fill(0.0)
        recurrent.b_option_action_transition_state.fill(0.0)
        recurrent.b2_policy[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_policy[LOCOMOTION_ACTIONS.index("MOVE_RIGHT"), 0] = 1.0
        recurrent.W2_policy[LOCOMOTION_ACTIONS.index("STAY"), 0] = -1.0
        recurrent.W_option_action_transition_state[
            OPTION_NAMES.index("FORAGE"),
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_controller_state_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=67,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_controller_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_controller_state=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.action_controller_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_option_action_controller_decoder.fill(0.0)
        recurrent.W_option_action_controller_prev.fill(0.0)
        recurrent.W_option_action_controller_action.fill(0.0)
        recurrent.b_option_action_controller.fill(0.0)
        recurrent.W2_option_action_controller_head.fill(0.0)
        recurrent.b2_option_action_controller_head.fill(0.0)
        recurrent.b2_policy[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W_option_action_controller_action[
            OPTION_NAMES.index("FORAGE"),
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        recurrent.W2_option_action_controller_head[
            OPTION_NAMES.index("FORAGE"),
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_option_action_controller_head[
            OPTION_NAMES.index("FORAGE"),
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_token_decoder_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=71,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_token_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_token_decoder=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.action_token_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_option_action_token_decoder.fill(0.0)
        recurrent.W_option_action_token_prev.fill(0.0)
        recurrent.W_option_action_token_action.fill(0.0)
        recurrent.b_option_action_token.fill(0.0)
        recurrent.b2_policy[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_policy[LOCOMOTION_ACTIONS.index("MOVE_RIGHT"), 0] = 1.0
        recurrent.W2_policy[LOCOMOTION_ACTIONS.index("STAY"), 0] = -1.0
        recurrent.W_option_action_token_action[
            OPTION_NAMES.index("FORAGE"),
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_recurrent_core_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=75,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_decoder_recurrent_action_core_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.action_policy_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.W2_action_policy_core.fill(0.0)
        recurrent.b2_action_policy_core.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_option_action_policy_decoder.fill(0.0)
        recurrent.W_option_action_policy_prev.fill(0.0)
        recurrent.W_option_action_policy_action.fill(0.0)
        recurrent.b_option_action_policy.fill(0.0)
        recurrent.b2_action_policy_core[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_action_policy_core[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_action_policy_core[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_option_action_policy_action[
            OPTION_NAMES.index("FORAGE"),
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_separate_recurrent_head_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=79,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_recurrent_head_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.action_policy_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.W2_action_policy_core.fill(0.0)
        recurrent.b2_action_policy_core.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_option_action_policy_decoder.fill(0.0)
        recurrent.W_option_action_policy_prev.fill(0.0)
        recurrent.W_option_action_policy_action.fill(0.0)
        recurrent.b_option_action_policy.fill(0.0)
        recurrent.b2_action_policy_core[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_action_policy_core[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_action_policy_core[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_option_action_policy_action[
            OPTION_NAMES.index("FORAGE"),
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_separate_policy_path_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=83,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_policy_path_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.action_policy_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.W2_action_policy_path.fill(0.0)
        recurrent.b2_action_policy_path.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_action_policy_path_input.fill(0.0)
        recurrent.W_action_policy_path_prev.fill(0.0)
        recurrent.W_action_policy_path_action.fill(0.0)
        recurrent.b_action_policy_path.fill(0.0)
        recurrent.b2_action_policy_path[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_action_policy_path[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_action_policy_path[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_action_policy_path_action[
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_option_action_separate_backbone_uses_previous_executed_action(self) -> None:
        brain = SpiderBrain(
            seed=87,
            module_dropout=0.0,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.reset_hidden_state()
        recurrent.reset_event_memory()
        recurrent.hidden_state.fill(0.0)
        recurrent.current_option_idx = OPTION_NAMES.index("FORAGE")
        recurrent.current_option_age = 0
        recurrent.current_option_steps_remaining = 3
        recurrent.decoder_action_state.fill(0.0)
        recurrent.action_backbone_state.fill(0.0)
        recurrent.W_xh.fill(0.0)
        recurrent.W_hh.fill(0.0)
        recurrent.b_h.fill(0.0)
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.W2_action_backbone.fill(0.0)
        recurrent.b2_action_backbone.fill(0.0)
        recurrent.option_action_bias.fill(0.0)
        recurrent.W2_option.fill(0.0)
        recurrent.b2_option.fill(0.0)
        recurrent.W2_option_feedback.fill(0.0)
        recurrent.b2_option_feedback.fill(0.0)
        recurrent.W_option_decoder_state.fill(0.0)
        recurrent.b_option_decoder_state.fill(0.0)
        recurrent.W_option_recurrent_dynamics.fill(0.0)
        recurrent.b_option_recurrent_dynamics.fill(0.0)
        recurrent.W_option_decoder_recurrent_state.fill(0.0)
        recurrent.b_option_decoder_recurrent_state.fill(0.0)
        recurrent.W_action_backbone_input.fill(0.0)
        recurrent.W_action_backbone_prev.fill(0.0)
        recurrent.W_action_backbone_action.fill(0.0)
        recurrent.b_action_backbone.fill(0.0)
        recurrent.b2_action_backbone[LOCOMOTION_ACTIONS.index("STAY")] = 0.5
        recurrent.W2_action_backbone[
            LOCOMOTION_ACTIONS.index("MOVE_RIGHT"),
            0,
        ] = 1.0
        recurrent.W2_action_backbone[
            LOCOMOTION_ACTIONS.index("STAY"),
            0,
        ] = -1.0
        recurrent.W_action_backbone_action[
            0,
            LOCOMOTION_ACTIONS.index("STAY"),
        ] = 4.0
        first = brain.act_inference(_build_observation(), sample=False)
        second = brain.act_inference(_build_observation(), sample=False)
        self.assertEqual(first.selected_option, "FORAGE")
        self.assertEqual(first.action_idx, LOCOMOTION_ACTIONS.index("STAY"))
        self.assertEqual(second.selected_option, "FORAGE")
        self.assertEqual(second.action_idx, LOCOMOTION_ACTIONS.index("MOVE_RIGHT"))

    def test_direct_policy_handoff_teacher_requires_position_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_handoff_teacher requires direct_policy_shelter_position_head=True",
        ):
            BrainAblationConfig(
                name="invalid_teacher_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_handoff_teacher=True,
            )

    def test_direct_policy_handoff_option_teacher_requires_handoff_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_handoff_option_teacher requires direct_policy_handoff_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_teacher_option_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_option_teacher=True,
            )

    def test_direct_policy_continuation_replay_requires_handoff_option_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_continuation_replay_passes requires direct_policy_handoff_option_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_teacher_replay_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            )

    def test_direct_policy_phase_head_with_option_head_requires_position_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_phase_head with direct_policy_option_head requires direct_policy_shelter_position_head=True",
        ):
            BrainAblationConfig(
                name="invalid_phase_option_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
            )

    def test_direct_policy_continuation_margin_requires_phase_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_continuation_margin_weight requires direct_policy_phase_head=True",
        ):
            BrainAblationConfig(
                name="invalid_continuation_margin_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_continuation_margin_weight=1.0,
            )

    def test_direct_policy_phase_option_feedback_requires_phase_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_phase_option_feedback requires direct_policy_phase_head=True",
        ):
            BrainAblationConfig(
                name="invalid_phase_option_feedback_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_phase_option_feedback=True,
            )

    def test_direct_policy_option_transition_feedback_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_transition_feedback requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_transition_feedback_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_transition_feedback=True,
            )

    def test_direct_policy_option_termination_cooldown_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_termination_cooldown requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_termination_cooldown_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_termination_cooldown=True,
            )

    def test_direct_policy_option_action_head_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_head requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_head=True,
            )

    def test_direct_policy_option_decoder_state_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_decoder_state requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_decoder_state_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_decoder_state=True,
            )

    def test_direct_policy_option_recurrent_dynamics_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_recurrent_dynamics requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_recurrent_dynamics_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_recurrent_dynamics=True,
            )

    def test_direct_policy_option_sequence_head_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_sequence_head requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_sequence_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_sequence_head=True,
            )

    def test_direct_policy_option_decoder_recurrent_state_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_decoder_recurrent_state requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_decoder_recurrent_state_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_decoder_recurrent_state=True,
            )

    def test_direct_policy_option_action_transition_state_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_transition_state requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_transition_state_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_transition_state=True,
            )

    def test_direct_policy_option_action_controller_state_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_controller_state requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_controller_state_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_controller_state=True,
            )

    def test_direct_policy_option_action_token_decoder_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_token_decoder requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_token_decoder_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_token_decoder=True,
            )

    def test_direct_policy_option_action_recurrent_core_requires_option_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_recurrent_core requires direct_policy_option_head=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_recurrent_core_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_option_action_recurrent_core=True,
            )

    def test_direct_policy_option_action_separate_recurrent_head_requires_action_recurrent_core(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_separate_recurrent_head requires direct_policy_option_action_recurrent_core=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_separate_recurrent_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_option_action_separate_recurrent_head=True,
            )

    def test_direct_policy_option_action_separate_policy_path_requires_action_recurrent_core(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_separate_policy_path requires direct_policy_option_action_recurrent_core=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_separate_policy_path_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_option_action_separate_policy_path=True,
            )

    def test_direct_policy_option_action_separate_backbone_requires_separate_policy_path(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_option_action_separate_backbone requires direct_policy_option_action_separate_policy_path=True",
        ):
            BrainAblationConfig(
                name="invalid_option_action_separate_backbone_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_backbone=True,
            )

    def test_direct_policy_post_rest_action_teacher_requires_separate_backbone(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_action_teacher requires direct_policy_option_action_separate_backbone=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_action_teacher_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_post_rest_action_teacher=True,
            )

    def test_direct_policy_post_rest_release_sequence_teacher_requires_post_rest_action_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_release_sequence_teacher requires direct_policy_post_rest_action_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_release_sequence_teacher_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_release_sequence_teacher=True,
            )

    def test_direct_policy_post_rest_release_sequence_replay_boost_requires_release_sequence_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_release_sequence_replay_boost requires direct_policy_post_rest_release_sequence_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_release_sequence_replay_boost_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_replay_boost=True,
            )

    def test_direct_policy_post_rest_release_sequence_distill_requires_release_sequence_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_release_sequence_distill requires direct_policy_post_rest_release_sequence_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_release_sequence_distill_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_distill=True,
            )

    def test_direct_policy_post_rest_probe_distillation_requires_release_sequence_teacher(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_distillation requires direct_policy_post_rest_release_sequence_teacher=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
            )

    def test_direct_policy_post_rest_probe_sequence_distillation_requires_probe_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_sequence_distillation requires direct_policy_post_rest_probe_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_sequence_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
            )

    def test_direct_policy_post_rest_probe_family_distillation_requires_probe_sequence_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_family_distillation requires direct_policy_post_rest_probe_sequence_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_family_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
            )

    def test_direct_policy_post_rest_probe_handoff_distillation_requires_probe_family_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_handoff_distillation requires direct_policy_post_rest_probe_family_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_handoff_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
            )

    def test_direct_policy_post_rest_probe_trajectory_distillation_requires_probe_handoff_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_trajectory_distillation requires direct_policy_post_rest_probe_handoff_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_trajectory_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
            )

    def test_direct_policy_post_rest_probe_cycle_distillation_requires_probe_trajectory_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_cycle_distillation requires direct_policy_post_rest_probe_trajectory_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_cycle_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_cycle_distillation=True,
            )

    def test_direct_policy_post_rest_probe_trace_distillation_requires_probe_cycle_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_trace_distillation requires direct_policy_post_rest_probe_cycle_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_trace_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
                direct_policy_post_rest_probe_trace_distillation=True,
            )

    def test_direct_policy_post_rest_probe_rollout_distillation_requires_probe_trace_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_rollout_distillation requires direct_policy_post_rest_probe_trace_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_rollout_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
                direct_policy_post_rest_probe_cycle_distillation=True,
                direct_policy_post_rest_probe_rollout_distillation=True,
            )

    def test_direct_policy_post_rest_probe_frontier_teacher_distillation_requires_probe_rollout_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_frontier_teacher_distillation requires direct_policy_post_rest_probe_rollout_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_frontier_teacher_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
                direct_policy_post_rest_probe_cycle_distillation=True,
                direct_policy_post_rest_probe_trace_distillation=True,
                direct_policy_post_rest_probe_frontier_teacher_distillation=True,
            )

    def test_direct_policy_post_rest_probe_replayable_teacher_distillation_requires_probe_rollout_distillation(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_post_rest_probe_replayable_teacher_distillation requires direct_policy_post_rest_probe_rollout_distillation=True",
        ):
            BrainAblationConfig(
                name="invalid_post_rest_probe_replayable_teacher_distillation_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_post_rest_probe_distillation=True,
                direct_policy_post_rest_probe_sequence_distillation=True,
                direct_policy_post_rest_probe_family_distillation=True,
                direct_policy_post_rest_probe_handoff_distillation=True,
                direct_policy_post_rest_probe_trajectory_distillation=True,
                direct_policy_post_rest_probe_cycle_distillation=True,
                direct_policy_post_rest_probe_trace_distillation=True,
                direct_policy_post_rest_probe_replayable_teacher_distillation=True,
            )

    def test_direct_policy_local_affordance_inputs_require_position_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_affordance_inputs requires direct_policy_shelter_position_head=True",
        ):
            BrainAblationConfig(
                name="invalid_local_affordance_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_local_affordance_inputs=True,
            )

    def test_direct_policy_local_affordance_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_affordance_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
        )
        brain = SpiderBrain(seed=13, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM,
        )
        affordance_tail = vector[-DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM:]
        np.testing.assert_allclose(affordance_tail[:4], np.array([0.0, 0.0, 1.0, 0.0]))

    def test_direct_policy_local_spatial_inputs_require_affordance_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_spatial_inputs requires direct_policy_local_affordance_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_local_spatial_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_spatial_inputs=True,
            )

    def test_direct_policy_local_spatial_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_spatial_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
        )
        brain = SpiderBrain(seed=17, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
            "local_spatial_patch": {
                "blocked": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "shelter_role_level": [0.0, 0.25, 0.5, 0.0, 0.66, 0.75, 1.0, 1.0, 1.0],
                "food": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
            + DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM,
        )
        spatial_tail = vector[-DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM:]
        np.testing.assert_allclose(
            spatial_tail[:9],
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        )

    def test_direct_policy_local_transition_inputs_require_spatial_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_transition_inputs requires direct_policy_local_spatial_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_local_transition_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_transition_inputs=True,
            )

    def test_direct_policy_local_transition_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_transition_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
        )
        brain = SpiderBrain(seed=19, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
            "local_spatial_patch": {
                "blocked": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "shelter_role_level": [0.0, 0.25, 0.5, 0.0, 0.66, 0.75, 1.0, 1.0, 1.0],
                "food": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            "local_transition_consequences": {
                "STAY": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
                "MOVE_UP": {"food_dist_delta": 1.0, "shelter_dist_delta": 1.0, "predator_dist_delta": -1.0, "next_cell_has_food": False},
                "MOVE_DOWN": {"food_dist_delta": -1.0, "shelter_dist_delta": -1.0, "predator_dist_delta": 1.0, "next_cell_has_food": False},
                "MOVE_LEFT": {"food_dist_delta": 0.0, "shelter_dist_delta": 1.0, "predator_dist_delta": 0.0, "next_cell_has_food": True},
                "MOVE_RIGHT": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
            + DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM,
        )
        transition_tail = vector[-DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM:]
        np.testing.assert_allclose(
            transition_tail[:8],
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0]),
        )

    def test_direct_policy_local_transition_rollout_inputs_require_transition_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_transition_rollout_inputs requires direct_policy_local_transition_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_local_transition_rollout_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_rollout_inputs=True,
            )

    def test_direct_policy_local_transition_rollout_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_transition_rollout_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
        )
        brain = SpiderBrain(seed=23, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
            "local_spatial_patch": {
                "blocked": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "shelter_role_level": [0.0, 0.25, 0.5, 0.0, 0.66, 0.75, 1.0, 1.0, 1.0],
                "food": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            "local_transition_consequences": {
                "STAY": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
                "MOVE_UP": {"food_dist_delta": 1.0, "shelter_dist_delta": 1.0, "predator_dist_delta": -1.0, "next_cell_has_food": False},
                "MOVE_DOWN": {"food_dist_delta": -1.0, "shelter_dist_delta": -1.0, "predator_dist_delta": 1.0, "next_cell_has_food": False},
                "MOVE_LEFT": {"food_dist_delta": 0.0, "shelter_dist_delta": 1.0, "predator_dist_delta": 0.0, "next_cell_has_food": True},
                "MOVE_RIGHT": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
            },
            "local_transition_rollouts": {
                "STAY": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": False},
                "MOVE_UP": {"best_food_dist_delta": 1.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": -1.0, "food_reachable_within_two_steps": True},
                "MOVE_DOWN": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": -1.0, "best_predator_dist_delta": 1.0, "food_reachable_within_two_steps": False},
                "MOVE_LEFT": {"best_food_dist_delta": 1.0, "best_shelter_dist_delta": 1.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": True},
                "MOVE_RIGHT": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": False},
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
            + DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM,
        )
        rollout_tail = vector[-DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM:]
        np.testing.assert_allclose(
            rollout_tail[:8],
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 1.0]),
        )

    def test_direct_policy_transition_prediction_head_requires_transition_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_transition_prediction_head requires direct_policy_local_transition_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_transition_prediction_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_transition_prediction_head=True,
            )

    def test_transition_prediction_feedback_requires_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_transition_prediction_feedback requires direct_policy_transition_prediction_head=True",
        ):
            BrainAblationConfig(
                name="invalid_transition_prediction_feedback_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_inputs=True,
                direct_policy_transition_prediction_feedback=True,
            )

    def test_transition_prediction_branch_exposes_transition_logits(self) -> None:
        config = BrainAblationConfig(
            name="transition_prediction_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_transition_prediction_head=True,
            direct_policy_transition_prediction_feedback=True,
        )
        brain = SpiderBrain(seed=24, module_dropout=0.0, config=config)
        decision = brain.act_inference(_build_observation(), sample=False)
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        self.assertEqual(
            decision.transition_prediction_logits.shape,
            (DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,),
        )
        self.assertEqual(
            len(
                recurrent.last_affordance_summary.get(
                    "transition_prediction_logits",
                    [],
                )
            ),
            DIRECT_POLICY_TRANSITION_PREDICTION_FEATURE_DIM,
        )

    def test_transition_rollout_prediction_head_requires_rollout_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_transition_rollout_prediction_head requires direct_policy_local_transition_rollout_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_transition_rollout_prediction_head_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_inputs=True,
                direct_policy_transition_rollout_prediction_head=True,
            )

    def test_transition_rollout_prediction_feedback_requires_head(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_transition_rollout_prediction_feedback requires direct_policy_transition_rollout_prediction_head=True",
        ):
            BrainAblationConfig(
                name="invalid_transition_rollout_prediction_feedback_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_inputs=True,
                direct_policy_local_transition_rollout_inputs=True,
                direct_policy_transition_rollout_prediction_feedback=True,
            )

    def test_transition_rollout_prediction_branch_exposes_transition_logits(self) -> None:
        config = BrainAblationConfig(
            name="transition_rollout_prediction_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_transition_rollout_prediction_head=True,
            direct_policy_transition_rollout_prediction_feedback=True,
        )
        brain = SpiderBrain(seed=25, module_dropout=0.0, config=config)
        decision = brain.act_inference(_build_observation(), sample=False)
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        self.assertEqual(
            decision.transition_rollout_prediction_logits.shape,
            (DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,),
        )
        self.assertEqual(
            len(
                recurrent.last_affordance_summary.get(
                    "transition_rollout_prediction_logits",
                    [],
                )
            ),
            DIRECT_POLICY_TRANSITION_ROLLOUT_PREDICTION_FEATURE_DIM,
        )

    def test_direct_policy_local_geodesic_inputs_require_rollout_inputs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "direct_policy_local_geodesic_inputs requires direct_policy_local_transition_rollout_inputs=True",
        ):
            BrainAblationConfig(
                name="invalid_local_geodesic_inputs_branch",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_local_affordance_inputs=True,
                direct_policy_local_spatial_inputs=True,
                direct_policy_local_transition_inputs=True,
                direct_policy_local_geodesic_inputs=True,
            )

    def test_direct_policy_local_geodesic_inputs_extend_observation_vector(self) -> None:
        config = BrainAblationConfig(
            name="local_geodesic_inputs_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=True,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_local_geodesic_inputs=True,
        )
        brain = SpiderBrain(seed=29, module_dropout=0.0, config=config)
        observation = _build_observation()
        observation["meta"] = {
            "shelter_role": "inside",
            "shelter_role_level": 0.66,
            "local_affordances": {
                "STAY": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_UP": {"blocked": False, "next_role": "entrance", "next_role_level": 0.33},
                "MOVE_DOWN": {"blocked": False, "next_role": "deep", "next_role_level": 1.0},
                "MOVE_LEFT": {"blocked": False, "next_role": "inside", "next_role_level": 0.66},
                "MOVE_RIGHT": {"blocked": True, "next_role": "inside", "next_role_level": 0.66},
            },
            "local_spatial_patch": {
                "blocked": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "shelter_role_level": [0.0, 0.25, 0.5, 0.0, 0.66, 0.75, 1.0, 1.0, 1.0],
                "food": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            "local_transition_consequences": {
                "STAY": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
                "MOVE_UP": {"food_dist_delta": 1.0, "shelter_dist_delta": 1.0, "predator_dist_delta": -1.0, "next_cell_has_food": False},
                "MOVE_DOWN": {"food_dist_delta": -1.0, "shelter_dist_delta": -1.0, "predator_dist_delta": 1.0, "next_cell_has_food": False},
                "MOVE_LEFT": {"food_dist_delta": 0.0, "shelter_dist_delta": 1.0, "predator_dist_delta": 0.0, "next_cell_has_food": True},
                "MOVE_RIGHT": {"food_dist_delta": 0.0, "shelter_dist_delta": 0.0, "predator_dist_delta": 0.0, "next_cell_has_food": False},
            },
            "local_transition_rollouts": {
                "STAY": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": False},
                "MOVE_UP": {"best_food_dist_delta": 1.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": -1.0, "food_reachable_within_two_steps": True},
                "MOVE_DOWN": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": -1.0, "best_predator_dist_delta": 1.0, "food_reachable_within_two_steps": False},
                "MOVE_LEFT": {"best_food_dist_delta": 1.0, "best_shelter_dist_delta": 1.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": True},
                "MOVE_RIGHT": {"best_food_dist_delta": 0.0, "best_shelter_dist_delta": 0.0, "best_predator_dist_delta": 0.0, "food_reachable_within_two_steps": False},
            },
            "local_geodesic_consequences": {
                "STAY": {"exit_geodesic_delta": 0.0, "deep_geodesic_delta": 0.0, "next_on_exit_target": False, "next_on_deep_target": False},
                "MOVE_UP": {"exit_geodesic_delta": 1.0, "deep_geodesic_delta": -1.0, "next_on_exit_target": True, "next_on_deep_target": False},
                "MOVE_DOWN": {"exit_geodesic_delta": -1.0, "deep_geodesic_delta": 1.0, "next_on_exit_target": False, "next_on_deep_target": True},
                "MOVE_LEFT": {"exit_geodesic_delta": 0.0, "deep_geodesic_delta": 0.0, "next_on_exit_target": False, "next_on_deep_target": False},
                "MOVE_RIGHT": {"exit_geodesic_delta": 0.0, "deep_geodesic_delta": 0.0, "next_on_exit_target": False, "next_on_deep_target": False},
            },
        }
        vector = brain._build_monolithic_observation(observation)

        self.assertEqual(
            vector.shape[0],
            sum(spec.input_dim for spec in MODULE_INTERFACES)
            + DIRECT_POLICY_LOCAL_AFFORDANCE_INPUT_DIM
            + DIRECT_POLICY_LOCAL_SPATIAL_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_INPUT_DIM
            + DIRECT_POLICY_LOCAL_TRANSITION_ROLLOUT_INPUT_DIM
            + DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM,
        )
        geodesic_tail = vector[-DIRECT_POLICY_LOCAL_GEODESIC_INPUT_DIM:]
        np.testing.assert_allclose(
            geodesic_tail[:8],
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0]),
        )

    def test_transition_rollout_prediction_feedback_affects_selected_action(self) -> None:
        config = BrainAblationConfig(
            name="transition_rollout_prediction_feedback_action_branch",
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            enable_food_direction_bias=False,
            use_learned_arbitration=False,
            warm_start_scale=0.0,
            direct_policy_hidden_dims=(32,),
            direct_policy_recurrent=True,
            direct_policy_event_attention=True,
            direct_policy_event_buffer_size=8,
            direct_policy_option_head=True,
            direct_policy_option_ttl=4,
            direct_policy_affordance_head=True,
            direct_policy_affordance_feedback=True,
            direct_policy_geometry_head=True,
            direct_policy_shelter_position_head=True,
            direct_policy_local_affordance_inputs=True,
            direct_policy_local_spatial_inputs=True,
            direct_policy_local_transition_inputs=True,
            direct_policy_local_transition_rollout_inputs=True,
            direct_policy_transition_rollout_prediction_head=True,
            direct_policy_transition_rollout_prediction_feedback=True,
        )
        brain = SpiderBrain(seed=26, module_dropout=0.0, config=config)
        recurrent = brain.true_monolithic_policy
        self.assertIsInstance(
            recurrent,
            RecurrentOptionAffordancePositionFeedbackTrueMonolithicNetwork,
        )
        recurrent.W2_policy.fill(0.0)
        recurrent.b2_policy.fill(0.0)
        recurrent.W_affordance_feedback.fill(0.0)
        recurrent.b_affordance_feedback.fill(0.0)
        recurrent.W2_policy_feedback.fill(0.0)
        recurrent.b2_policy_feedback.fill(0.0)
        recurrent.W2_transition_rollout_prediction.fill(0.0)
        recurrent.b2_transition_rollout_prediction.fill(0.0)
        recurrent.W_transition_rollout_prediction_feedback.fill(0.0)
        recurrent.b_transition_rollout_prediction_feedback.fill(0.0)
        recurrent.b2_transition_rollout_prediction[0] = 10.0
        recurrent.W_transition_rollout_prediction_feedback[0, 0] = 5.0
        recurrent.W2_policy_feedback[ACTION_TO_INDEX["MOVE_UP"], 0] = 5.0
        observation = _build_observation()
        monolithic_observation = brain._build_monolithic_observation(observation)
        policy_logits = recurrent.forward(monolithic_observation, store_cache=False)[0]
        self.assertEqual(
            int(np.argmax(policy_logits)),
            ACTION_TO_INDEX["MOVE_UP"],
        )

    def test_collect_direct_policy_probe_trace_distillation_rollout_collects_trace_samples(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_trace_distill_option_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        sim = SpiderSimulation(seed=11, max_steps=20, brain_config=config)
        dataset = sim.collect_direct_policy_probe_trace_distillation_rollout(
            episodes=1,
            episode_start=13,
        )
        summary = dataset.to_summary()
        self.assertGreater(len(dataset), 40)
        self.assertEqual(summary["episode_ids"], [13, 14, 15, 16])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "redirected_direct_policy_post_rest_probe_trace",
        )
        action_stages = set(dataset.teacher_metadata.get("action_stages", []))
        self.assertIn("trace_release", action_stages)
        self.assertIn("trace_realign", action_stages)

    def test_collect_direct_policy_probe_rollout_distillation_rollout_collects_post_handoff_samples(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_rollout_distill_option_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        sim = SpiderSimulation(seed=11, max_steps=20, brain_config=config)
        dataset = sim.collect_direct_policy_probe_rollout_distillation_rollout(
            episodes=1,
            episode_start=17,
        )
        summary = dataset.to_summary()
        self.assertGreater(len(dataset), 10)
        self.assertEqual(summary["episode_ids"], [17, 18])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "redirected_direct_policy_post_rest_probe_rollout",
        )
        action_stages = set(dataset.teacher_metadata.get("action_stages", []))
        self.assertIn("rollout_release", action_stages)
        self.assertIn("rollout_live", action_stages)

    def test_direct_policy_handoff_teacher_targets_follow_probe_sequence(self) -> None:
        sim = SpiderSimulation(seed=7, max_steps=2)
        sim._reset_direct_policy_handoff_teacher_state()
        center_x = sim._teacher_center_column()
        release_state = {
            "x": center_x,
            "y": 6,
            "shelter_role": "inside",
            "sleep_phase": "RESTING",
            "rest_streak": 3,
            "hunger": 0.18,
            "food_memory": {"target": [center_x + 4, 7], "age": 1, "ttl": 12},
            "predator_positions": [[0, 0]],
            "recent_contact": 0.0,
            "recent_pain": 0.0,
            "predator_motion_salience": 0.0,
            "predator_trace": {"strength": 0.0},
        }
        entrance_state = {
            **release_state,
            "y": 5,
            "shelter_role": "entrance",
            "sleep_phase": "AWAKE",
        }
        inside_state = {
            **release_state,
            "y": 6,
            "shelter_role": "inside",
            "sleep_phase": "AWAKE",
        }
        deep_state = {
            **release_state,
            "y": 7,
            "shelter_role": "deep",
            "sleep_phase": "AWAKE",
        }
        post_rest_inside_state = {
            **inside_state,
            "sleep_events": 1,
        }
        post_rest_outside_state = {
            **inside_state,
            "y": 4,
            "shelter_role": "outside",
            "sleep_events": 1,
        }

        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=release_state,
            food_direction_action="MOVE_RIGHT",
            tick=12,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[target_idx], "MOVE_UP")
        self.assertEqual(stage, "handoff_release")

        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=entrance_state,
            food_direction_action="MOVE_RIGHT",
            tick=13,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[target_idx], "STAY")
        self.assertEqual(stage, "handoff_hold")

        for tick, state in ((14, entrance_state), (15, inside_state), (16, deep_state)):
            target_idx, stage = sim._direct_policy_handoff_teacher_target(
                current_state=state,
                food_direction_action="MOVE_RIGHT",
                tick=tick,
            )
            self.assertEqual(LOCOMOTION_ACTIONS[target_idx], "MOVE_DOWN")
            self.assertEqual(stage, "handoff_continue")

        option_idx, option_stage = sim._direct_policy_handoff_option_teacher_target(
            current_state=post_rest_inside_state,
            food_direction_action="MOVE_RIGHT",
            tick=17,
        )
        self.assertEqual(OPTION_NAMES[option_idx], "FORAGE")
        self.assertEqual(option_stage, "option_forage_window")

        sim.brain = SpiderBrain(
            seed=99,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_action_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        sim._reset_direct_policy_handoff_teacher_state()
        sim._direct_policy_handoff_teacher_target(
            current_state=release_state,
            food_direction_action="MOVE_RIGHT",
            tick=12,
        )
        sim._direct_policy_handoff_teacher_target(
            current_state=entrance_state,
            food_direction_action="MOVE_RIGHT",
            tick=13,
        )
        for tick, state in ((14, entrance_state), (15, inside_state), (16, deep_state)):
            sim._direct_policy_handoff_teacher_target(
                current_state=state,
                food_direction_action="MOVE_RIGHT",
                tick=tick,
            )
        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=post_rest_outside_state,
            food_direction_action="MOVE_RIGHT",
            tick=17,
        )
        self.assertEqual(target_idx, int(ACTION_TO_INDEX["MOVE_RIGHT"]))
        self.assertEqual(stage, "handoff_forage_window")

        sim._reset_direct_policy_handoff_teacher_state()
        option_idx, option_stage = sim._direct_policy_handoff_option_teacher_target(
            current_state=post_rest_inside_state,
            food_direction_action="MOVE_RIGHT",
            tick=18,
        )
        self.assertEqual(OPTION_NAMES[option_idx], "POST_REST_REACTIVATE")
        self.assertEqual(option_stage, "option_post_rest_inside")

        threatened_outside_state = {
            **post_rest_outside_state,
            "predator_positions": [[center_x, 4]],
            "recent_contact": 0.0,
            "recent_pain": 0.0,
            "predator_motion_salience": 1.0,
            "predator_trace": {"strength": 1.0},
        }
        option_idx, option_stage = sim._direct_policy_handoff_option_teacher_target(
            current_state=threatened_outside_state,
            food_direction_action="MOVE_RIGHT",
            tick=19,
        )
        self.assertEqual(OPTION_NAMES[option_idx], "RETURN_TO_SHELTER")
        self.assertEqual(option_stage, "option_threatened_return")

        sim._reset_direct_policy_handoff_teacher_state()
        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=post_rest_outside_state,
            food_direction_action="MOVE_RIGHT",
            tick=20,
        )
        self.assertEqual(target_idx, int(ACTION_TO_INDEX["MOVE_RIGHT"]))
        self.assertEqual(stage, "handoff_post_rest_forage")

        sim.brain = SpiderBrain(
            seed=101,
            config=BrainAblationConfig(
                name="true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_release_sequence_teacher_option_replay_policy",
                architecture="true_monolithic",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                enable_food_direction_bias=True,
                use_learned_arbitration=False,
                warm_start_scale=0.0,
                direct_policy_hidden_dims=(32,),
                direct_policy_recurrent=True,
                direct_policy_phase_head=True,
                direct_policy_event_attention=True,
                direct_policy_event_buffer_size=8,
                direct_policy_option_head=True,
                direct_policy_option_ttl=4,
                direct_policy_affordance_head=True,
                direct_policy_affordance_feedback=True,
                direct_policy_geometry_head=True,
                direct_policy_shelter_position_head=True,
                direct_policy_handoff_teacher=True,
                direct_policy_handoff_option_teacher=True,
                direct_policy_post_rest_action_teacher=True,
                direct_policy_post_rest_release_sequence_teacher=True,
                direct_policy_option_decoder_state=True,
                direct_policy_option_recurrent_dynamics=True,
                direct_policy_option_decoder_recurrent_state=True,
                direct_policy_option_action_recurrent_core=True,
                direct_policy_option_action_separate_recurrent_head=True,
                direct_policy_option_action_separate_policy_path=True,
                direct_policy_option_action_separate_backbone=True,
                direct_policy_continuation_replay_passes=2,
                direct_policy_continuation_replay_lr_scale=0.5,
            ),
        )
        sim._reset_direct_policy_handoff_teacher_state()
        target_idx, stage = sim._direct_policy_handoff_teacher_target(
            current_state=post_rest_inside_state,
            food_direction_action="MOVE_RIGHT",
            tick=21,
        )
        self.assertEqual(LOCOMOTION_ACTIONS[target_idx], "MOVE_UP")
        self.assertEqual(stage, "handoff_release")

    def test_collect_direct_policy_probe_distillation_rollout_collects_teacher_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_distillation_rollout(
            scenario_name="continuous_survival_post_rest_inside_v1",
            episodes=1,
            episode_start=3,
        )
        self.assertGreater(len(dataset), 0)
        self.assertEqual(dataset.to_summary()["episode_ids"], [3])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "scripted_direct_policy_post_rest_probe",
        )
        self.assertIn("handoff_release", dataset.teacher_metadata["stages"])

    def test_collect_direct_policy_probe_sequence_distillation_rollout_collects_multistep_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_sequence_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_sequence_distillation_rollout(
            scenario_name="continuous_survival_return_after_late_forage_v1",
            episodes=1,
            episode_start=4,
        )
        self.assertGreater(len(dataset), 3)
        self.assertEqual(dataset.to_summary()["episode_ids"], [4])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "scripted_direct_policy_post_rest_probe_sequence",
        )
        self.assertIn("MOVE_LEFT", dataset.teacher_metadata["scripted_sequence"])

    def test_collect_direct_policy_probe_family_distillation_rollout_collects_global_teacher_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_family_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_family_distillation_rollout(
            episodes=1,
            episode_start=5,
        )
        self.assertGreater(len(dataset), 20)
        self.assertEqual(dataset.to_summary()["episode_ids"], [5, 6])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "scripted_direct_policy_post_rest_probe_family",
        )
        self.assertEqual(
            dataset.teacher_metadata["teacher_scenarios"],
            ["continuous_survival_canonical", "continuous_survival_easy_v1"],
        )

    def test_collect_direct_policy_probe_handoff_distillation_rollout_collects_handoff_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_handoff_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_handoff_distillation_rollout(
            episodes=1,
            episode_start=7,
        )
        self.assertGreater(len(dataset), 20)
        self.assertEqual(dataset.to_summary()["episode_ids"], [7, 8])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "scripted_direct_policy_post_rest_probe_handoff",
        )

    def test_collect_direct_policy_probe_trajectory_distillation_rollout_collects_redirected_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_trajectory_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_trajectory_distillation_rollout(
            episodes=1,
            episode_start=9,
        )
        self.assertGreater(len(dataset), 40)
        self.assertEqual(dataset.to_summary()["episode_ids"], [9, 10])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "redirected_direct_policy_post_rest_probe_trajectory",
        )
        self.assertIn(
            "trajectory_release",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )

    def test_collect_direct_policy_probe_cycle_distillation_rollout_collects_cycle_samples(self) -> None:
        sim = SpiderSimulation(
            seed=11,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_cycle_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_cycle_distillation_rollout(
            episodes=1,
            episode_start=11,
        )
        self.assertGreater(len(dataset), 40)
        self.assertEqual(dataset.to_summary()["episode_ids"], [11, 12])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "redirected_direct_policy_post_rest_probe_cycle",
        )
        self.assertIn(
            "cycle_release",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )

    def test_collect_direct_policy_probe_frontier_teacher_distillation_rollout_collects_fixed_teacher_samples(self) -> None:
        sim = SpiderSimulation(
            seed=17,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_frontier_teacher_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_frontier_teacher_distillation_rollout(
            episodes=1,
            episode_start=17,
        )
        self.assertGreater(len(dataset), 20)
        self.assertEqual(dataset.to_summary()["episode_ids"], [17, 18])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "checkpoint_direct_policy_post_rest_probe_frontier_teacher",
        )
        self.assertIn(
            "architecture version is incompatible",
            str(dataset.teacher_metadata["teacher_checkpoint_error"]),
        )
        self.assertIn(
            "rollout_release",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )
        self.assertIn(
            "rollout_live",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )

    def test_collect_direct_policy_probe_replayable_teacher_distillation_rollout_collects_probe_samples(self) -> None:
        sim = SpiderSimulation(
            seed=19,
            max_steps=20,
            brain_config=resolve_ablation_configs(
                [
                    "true_monolithic_option_affordance_position_phase_option_dynamics_separate_action_backbone_post_rest_probe_replayable_teacher_distill_option_replay_policy"
                ],
                module_dropout=0.0,
            )[0],
        )
        dataset = sim.collect_direct_policy_probe_replayable_teacher_distillation_rollout(
            episodes=1,
            episode_start=19,
        )
        self.assertGreater(len(dataset), 20)
        self.assertEqual(dataset.to_summary()["episode_ids"], [19])
        self.assertEqual(
            dataset.teacher_metadata["source"],
            "stateful_direct_policy_post_rest_probe_replayable_teacher",
        )
        self.assertGreaterEqual(
            int(dataset.teacher_metadata.get("first_up_redirects", 0)),
            1,
        )
        self.assertGreaterEqual(
            int(dataset.teacher_metadata.get("down_redirects", 0)),
            1,
        )
        self.assertIn(
            "probe_release",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )
        self.assertIn(
            "probe_down_continue",
            set(dataset.teacher_metadata.get("action_stages", [])),
        )

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
        """
        Runs a short training episode and asserts that temporal-difference update diagnostics containing module credit information are published to the simulation bus.
        
        Verifies at least one "td_update" message is present and that its payload includes:
        - "module_credit_weights" and "module_gradient_norms" keys,
        - "credit_strategy" equal to "route_mask",
        - "route_mask_enabled" set to True.
        """
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
        self.assertEqual(payload["credit_strategy"], "route_mask")
        self.assertTrue(payload["route_mask_enabled"])

    def test_true_monolithic_phase_replay_training_episode_applies_replay_passes(self) -> None:
        config = resolve_ablation_configs(
            [
                "true_monolithic_option_affordance_position_phase_teacher_option_replay_policy"
            ],
            module_dropout=0.0,
        )[0]
        sim = SpiderSimulation(seed=7, max_steps=18, brain_config=config)

        sim.run_episode(
            episode_index=0,
            training=True,
            sample=False,
            scenario_name="continuous_survival_post_rest_inside_v1",
        )

        td_updates = [
            message.payload
            for message in sim.bus.history()
            if message.topic == "td_update"
        ]
        self.assertGreater(len(td_updates), 0)
        self.assertTrue(
            any(
                int(payload.get("continuation_replay_passes_applied", 0)) > 0
                for payload in td_updates
            )
        )

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
