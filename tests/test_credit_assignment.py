from __future__ import annotations

import unittest

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    ModuleInterface,
    MOTOR_CONTEXT_INTERFACE,
)
from spider_cortex_sim.modules import ModuleResult


def _blank_mapping(interface: ModuleInterface) -> dict[str, float]:
    """Return a dict[str, float] blank signal mapping for an interface."""
    return {
        signal.name: (1.0 if signal.name.endswith("_age") else 0.0)
        for signal in interface.inputs
    }


def _blank_obs() -> dict[str, np.ndarray]:
    """Return a dict[str, np.ndarray] blank observation for all interfaces."""
    observation: dict[str, np.ndarray] = {}
    interfaces: tuple[ModuleInterface, ...] = (
        *MODULE_INTERFACES,
        ACTION_CONTEXT_INTERFACE,
        MOTOR_CONTEXT_INTERFACE,
    )
    for interface in interfaces:
        observation[interface.observation_key] = interface.vector_from_mapping(
            _blank_mapping(interface)
        )
    return observation


class CounterfactualCreditTest(unittest.TestCase):
    def _module_result(self, interface: ModuleInterface, logits: np.ndarray) -> ModuleResult:
        """Return a ModuleResult for an interface and logits."""
        action_dim = len(LOCOMOTION_ACTIONS)
        return ModuleResult(
            interface=interface,
            name=interface.name,
            observation_key=interface.observation_key,
            observation=np.zeros(interface.input_dim, dtype=float),
            logits=np.asarray(logits, dtype=float).copy(),
            probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
            active=True,
        )

    def _module_results(self, logits_by_name: dict[str, np.ndarray]) -> list[ModuleResult]:
        """Return one ModuleResult per module interface."""
        action_dim = len(LOCOMOTION_ACTIONS)
        default_logits = np.zeros(action_dim, dtype=float)
        return [
            self._module_result(
                interface,
                logits_by_name.get(interface.name, default_logits),
            )
            for interface in MODULE_INTERFACES
        ]

    def _zero_action_center_correction(self, brain: SpiderBrain) -> None:
        """Patch action_center.forward to return zero correction."""
        brain.action_center.forward = lambda _x, *, store_cache=False: (
            np.zeros(brain.action_dim, dtype=float),
            0.0,
        )

    def test_identical_module_outputs_produce_equal_weights(self) -> None:
        brain = SpiderBrain(seed=3, module_dropout=0.0)
        self._zero_action_center_correction(brain)
        shared_logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        shared_logits[0] = 2.0
        module_results = self._module_results(
            {
                interface.name: shared_logits
                for interface in MODULE_INTERFACES
            }
        )

        weights = brain._compute_counterfactual_credit(
            module_results,
            _blank_obs(),
            action_idx=0,
        )

        expected_weight = 1.0 / float(len(MODULE_INTERFACES))
        self.assertEqual(set(weights), {interface.name for interface in MODULE_INTERFACES})
        for weight in weights.values():
            self.assertAlmostEqual(weight, expected_weight)

    def test_greater_action_probability_contribution_receives_higher_weight(self) -> None:
        brain = SpiderBrain(seed=5, module_dropout=0.0)
        self._zero_action_center_correction(brain)
        action_logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        strongest_logits = action_logits.copy()
        strongest_logits[0] = 3.0
        weaker_logits = action_logits.copy()
        weaker_logits[0] = 1.0
        module_results = self._module_results(
            {
                "visual_cortex": strongest_logits,
                "sensory_cortex": weaker_logits,
            }
        )

        weights = brain._compute_counterfactual_credit(
            module_results,
            _blank_obs(),
            action_idx=0,
        )

        self.assertGreater(weights["visual_cortex"], weights["sensory_cortex"])
        for module_name, weight in weights.items():
            if module_name not in {"visual_cortex", "sensory_cortex"}:
                self.assertAlmostEqual(weight, 0.0)

    def test_counterfactual_weights_are_normalized(self) -> None:
        """
        Verify that counterfactual credit weights are non-negative and sum to 1.
        
        Constructs a SpiderBrain with a disabled action-center correction, provides distinct per-module logits for the chosen action, computes counterfactual credit weights for action index 0, and asserts the weights are all >= 0.0 and their sum is approximately 1.0.
        """
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        self._zero_action_center_correction(brain)
        visual_logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        sensory_logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        hunger_logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        visual_logits[0] = 3.0
        sensory_logits[0] = -1.5
        hunger_logits[0] = 0.75

        weights = brain._compute_counterfactual_credit(
            self._module_results(
                {
                    "visual_cortex": visual_logits,
                    "sensory_cortex": sensory_logits,
                    "hunger_center": hunger_logits,
                }
            ),
            _blank_obs(),
            action_idx=0,
        )

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(weight >= 0.0 for weight in weights.values()))

    def test_learn_with_counterfactual_credit_returns_diagnostics_and_updates_parameters(self) -> None:
        config = BrainAblationConfig(
            name="counterfactual_credit",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="counterfactual",
        )
        brain = SpiderBrain(seed=11, module_dropout=0.0, config=config)
        obs = _blank_obs()
        decision = brain.act(obs, sample=True)
        decision.action_idx = 0
        decision.policy = np.full(brain.action_dim, 1.0 / brain.action_dim, dtype=float)
        for result in decision.module_results:
            result.logits = np.zeros(brain.action_dim, dtype=float)
        decision.module_results[0].logits[0] = 3.0
        decision.module_results[1].logits[0] = 1.0
        self._zero_action_center_correction(brain)
        before_motor_w1 = brain.motor_cortex.W1.copy()

        learn_stats = brain.learn(
            decision,
            reward=1.0,
            next_observation=obs,
            done=True,
        )

        self.assertIn("module_credit_weights", learn_stats)
        self.assertIn("module_gradient_norms", learn_stats)
        self.assertEqual(learn_stats["credit_strategy"], "counterfactual")
        weights = learn_stats["module_credit_weights"]
        self.assertGreater(weights[decision.module_results[0].name], weights[decision.module_results[1].name])
        self.assertGreater(max(weights.values()), min(weights.values()))
        self.assertFalse(np.allclose(before_motor_w1, brain.motor_cortex.W1))


class CounterfactualCreditEdgeCasesTest(unittest.TestCase):
    """Edge-case and boundary tests for _compute_counterfactual_credit."""

    def _zero_action_center_correction(self, brain: SpiderBrain) -> None:
        brain.action_center.forward = lambda _x, *, store_cache=False: (
            np.zeros(brain.action_dim, dtype=float),
            0.0,
        )

    def _module_result(
        self,
        interface: ModuleInterface,
        logits: np.ndarray,
        *,
        active: bool = True,
    ) -> ModuleResult:
        action_dim = len(LOCOMOTION_ACTIONS)
        return ModuleResult(
            interface=interface,
            name=interface.name,
            observation_key=interface.observation_key,
            observation=np.zeros(interface.input_dim, dtype=float),
            logits=np.asarray(logits, dtype=float).copy(),
            probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
            active=active,
        )

    def test_empty_module_results_returns_empty_dict(self) -> None:
        """_compute_counterfactual_credit returns {} for empty module list."""
        brain = SpiderBrain(seed=1, module_dropout=0.0)
        self._zero_action_center_correction(brain)
        weights = brain._compute_counterfactual_credit([], _blank_obs(), action_idx=0)
        self.assertEqual(weights, {})

    def test_inactive_module_receives_weight_zero(self) -> None:
        """An inactive module gets credit weight 0.0 even if others are active."""
        brain = SpiderBrain(seed=2, module_dropout=0.0)
        self._zero_action_center_correction(brain)

        results = []
        for interface in MODULE_INTERFACES:
            is_active = interface.name == "visual_cortex"
            logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
            if is_active:
                logits[0] = 2.0
            results.append(self._module_result(interface, logits, active=is_active))

        weights = brain._compute_counterfactual_credit(results, _blank_obs(), action_idx=0)

        for interface in MODULE_INTERFACES:
            if interface.name == "visual_cortex":
                self.assertGreaterEqual(weights[interface.name], 0.0)
            else:
                self.assertAlmostEqual(weights[interface.name], 0.0)

    def test_all_importances_zero_uniform_fallback_active_modules(self) -> None:
        """When all magnitudes are ~0, uniform weight is distributed across active modules."""
        brain = SpiderBrain(seed=3, module_dropout=0.0)
        # Zero correction from action center
        self._zero_action_center_correction(brain)

        # All modules with identical zero logits -> all importances will be zero
        zero_logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        results = [
            self._module_result(interface, zero_logits)
            for interface in MODULE_INTERFACES
        ]

        weights = brain._compute_counterfactual_credit(results, _blank_obs(), action_idx=0)

        # All should be equal with sum = 1.0
        self.assertEqual(set(weights), {interface.name for interface in MODULE_INTERFACES})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        n = len(MODULE_INTERFACES)
        expected = 1.0 / float(n)
        for w in weights.values():
            self.assertAlmostEqual(w, expected)

    def test_all_modules_inactive_uniform_across_all(self) -> None:
        """When all modules are inactive and importances are zero, uniform weight over all."""
        brain = SpiderBrain(seed=4, module_dropout=0.0)
        self._zero_action_center_correction(brain)

        zero_logits = np.zeros(len(LOCOMOTION_ACTIONS), dtype=float)
        results = [
            self._module_result(interface, zero_logits, active=False)
            for interface in MODULE_INTERFACES
        ]

        weights = brain._compute_counterfactual_credit(results, _blank_obs(), action_idx=0)

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        n = len(MODULE_INTERFACES)
        expected = 1.0 / float(n)
        for w in weights.values():
            self.assertAlmostEqual(w, expected)

    def test_action_idx_selects_correct_action(self) -> None:
        """Credit for action_idx differs when two modules specialize in different actions."""
        brain = SpiderBrain(seed=5, module_dropout=0.0)
        self._zero_action_center_correction(brain)

        action_dim = len(LOCOMOTION_ACTIONS)
        # visual_cortex contributes strongly to action 1
        # sensory_cortex contributes strongly to action 0
        # All other modules are zero
        logits_by_name = {}
        for interface in MODULE_INTERFACES:
            logits = np.zeros(action_dim, dtype=float)
            if interface.name == "visual_cortex":
                logits[1] = 5.0
            elif interface.name == "sensory_cortex":
                logits[0] = 5.0
            logits_by_name[interface.name] = logits

        results = [
            self._module_result(interface, logits_by_name[interface.name])
            for interface in MODULE_INTERFACES
        ]

        weights_for_action_1 = brain._compute_counterfactual_credit(
            results, _blank_obs(), action_idx=1
        )
        weights_for_action_0 = brain._compute_counterfactual_credit(
            results, _blank_obs(), action_idx=0
        )

        # visual_cortex should get more credit for action 1 than for action 0
        self.assertGreater(
            weights_for_action_1["visual_cortex"],
            weights_for_action_0["visual_cortex"],
        )
        # sensory_cortex should get more credit for action 0 than for action 1
        self.assertGreater(
            weights_for_action_0["sensory_cortex"],
            weights_for_action_1["sensory_cortex"],
        )

    def test_all_weights_non_negative(self) -> None:
        """All returned weights must be non-negative even when a module interferes."""
        brain = SpiderBrain(seed=6, module_dropout=0.0)
        self._zero_action_center_correction(brain)

        action_dim = len(LOCOMOTION_ACTIONS)
        results = []
        for i, interface in enumerate(MODULE_INTERFACES):
            logits = np.zeros(action_dim, dtype=float)
            # Alternate: some modules help action 0, some oppose it
            logits[0] = 2.0 if i % 2 == 0 else -2.0
            results.append(self._module_result(interface, logits))

        weights = brain._compute_counterfactual_credit(results, _blank_obs(), action_idx=0)

        for name, w in weights.items():
            self.assertGreaterEqual(w, 0.0, f"weight for {name!r} must be non-negative")
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_return_keys_match_module_names(self) -> None:
        """Returned dict keys match exactly the names of the provided module results."""
        brain = SpiderBrain(seed=7, module_dropout=0.0)
        self._zero_action_center_correction(brain)

        action_dim = len(LOCOMOTION_ACTIONS)
        results = [
            self._module_result(interface, np.zeros(action_dim, dtype=float))
            for interface in MODULE_INTERFACES
        ]
        weights = brain._compute_counterfactual_credit(results, _blank_obs(), action_idx=0)

        self.assertEqual(
            set(weights.keys()),
            {interface.name for interface in MODULE_INTERFACES},
        )

    def test_learn_counterfactual_raises_value_error_when_observation_empty(self) -> None:
        """learn() raises ValueError when counterfactual credit is used but step.observation is empty."""
        config = BrainAblationConfig(
            name="counterfactual_credit",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="counterfactual",
        )
        brain = SpiderBrain(seed=8, module_dropout=0.0, config=config)
        obs = _blank_obs()
        step = brain.act(obs, sample=False)
        # Clear the observation snapshot so learn() cannot compute counterfactual credit
        step.observation = {}

        with self.assertRaises(ValueError):
            brain.learn(step, reward=0.0, next_observation=obs, done=True)

    def test_act_populates_brain_step_observation_for_counterfactual_credit(self) -> None:
        """BrainStep.observation is populated only for counterfactual credit."""
        config = BrainAblationConfig(
            name="counterfactual_credit",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="counterfactual",
        )
        brain = SpiderBrain(seed=9, module_dropout=0.0, config=config)
        obs = _blank_obs()
        step = brain.act(obs, sample=False)
        self.assertGreater(len(step.observation), 0)
        for value in step.observation.values():
            self.assertIsInstance(value, np.ndarray)

    def test_act_omits_brain_step_observation_for_broadcast_credit(self) -> None:
        """BrainStep.observation stays empty when counterfactual credit is off."""
        brain = SpiderBrain(seed=9, module_dropout=0.0)
        step = brain.act(_blank_obs(), sample=False)
        self.assertEqual(step.observation, {})

    def test_broadcast_learn_returns_ones_credit_weights(self) -> None:
        """broadcast strategy: all module_credit_weights equal 1.0."""
        config = BrainAblationConfig(
            name="test_broadcast",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="broadcast",
        )
        brain = SpiderBrain(seed=10, module_dropout=0.0, config=config)
        obs = _blank_obs()
        # sample=True so training mode is active and motor_cortex caches state for backward
        step = brain.act(obs, sample=True)

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        self.assertEqual(stats["credit_strategy"], "broadcast")
        for name, weight in stats["module_credit_weights"].items():
            self.assertAlmostEqual(weight, 1.0, msg=f"Module {name!r} weight should be 1.0 for broadcast")
        # counterfactual_credit_weights should be empty for broadcast
        self.assertEqual(stats["counterfactual_credit_weights"], {})

    def test_local_only_learn_returns_zero_credit_weights(self) -> None:
        """local_only strategy: all module_credit_weights equal 0.0."""
        config = BrainAblationConfig(
            name="test_local_only",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="local_only",
        )
        brain = SpiderBrain(seed=11, module_dropout=0.0, config=config)
        obs = _blank_obs()
        # sample=True so training mode is active and motor_cortex caches state for backward
        step = brain.act(obs, sample=True)

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        self.assertEqual(stats["credit_strategy"], "local_only")
        for name, weight in stats["module_credit_weights"].items():
            self.assertAlmostEqual(weight, 0.0, msg=f"Module {name!r} weight should be 0.0 for local_only")
        # counterfactual_credit_weights should be empty for local_only
        self.assertEqual(stats["counterfactual_credit_weights"], {})

    def test_counterfactual_learn_populates_counterfactual_credit_weights(self) -> None:
        """counterfactual strategy: counterfactual_credit_weights is non-empty and matches module_credit_weights."""
        config = BrainAblationConfig(
            name="counterfactual_credit",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="counterfactual",
        )
        brain = SpiderBrain(seed=12, module_dropout=0.0, config=config)
        obs = _blank_obs()
        # sample=True so training mode is active and motor_cortex caches state for backward
        step = brain.act(obs, sample=True)

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        self.assertEqual(stats["credit_strategy"], "counterfactual")
        self.assertGreater(len(stats["counterfactual_credit_weights"]), 0)
        # The counterfactual weights should match module_credit_weights
        self.assertEqual(stats["counterfactual_credit_weights"], stats["module_credit_weights"])

    def test_route_mask_learn_masks_non_route_proposers(self) -> None:
        config = BrainAblationConfig(
            name="test_route_mask",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="route_mask",
            route_mask_threshold=0.05,
        )
        brain = SpiderBrain(seed=16, module_dropout=0.0, config=config)
        obs = _blank_obs()
        step = brain.act(obs, sample=True, training=True)

        route_owner = "alert_center"
        for result in step.module_results:
            result.gate_weight = 0.0
            result.contribution_share = 0.0
        for result in step.module_results:
            if result.name == route_owner:
                result.gate_weight = 1.0
                result.contribution_share = 1.0
                break

        stats = brain.learn(step, reward=1.0, next_observation=obs, done=True)

        self.assertEqual(stats["credit_strategy"], "route_mask")
        self.assertTrue(stats["route_mask_enabled"])
        self.assertEqual(stats["route_active_modules"], [route_owner])
        self.assertEqual(stats["module_credit_weights"][route_owner], 1.0)
        for name in stats["module_credit_weights"]:
            expected_weight = 1.0 if name == route_owner else 0.0
            self.assertAlmostEqual(
                stats["route_credit_weights"][name],
                expected_weight,
            )
        for name, norm in stats["module_gradient_norms"].items():
            if name == route_owner:
                self.assertGreater(float(norm), 0.0)
            else:
                self.assertEqual(float(norm), 0.0)

    def test_module_gradient_norms_present_for_all_strategies(self) -> None:
        """module_gradient_norms is present in learn() output for all credit strategies."""
        for strategy in ("broadcast", "local_only", "counterfactual", "route_mask"):
            config = BrainAblationConfig(
                name=f"test_{strategy}",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                credit_strategy=strategy,
            )
            brain = SpiderBrain(seed=13, module_dropout=0.0, config=config)
            obs = _blank_obs()
            # sample=True so training mode is active and motor_cortex caches state for backward
            step = brain.act(obs, sample=True)
            stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)
            with self.subTest(strategy=strategy):
                self.assertIn("module_gradient_norms", stats)
                self.assertGreater(len(stats["module_gradient_norms"]), 0)
                for norm in stats["module_gradient_norms"].values():
                    self.assertGreaterEqual(float(norm), 0.0)

    def test_module_gradient_norms_are_non_negative_floats(self) -> None:
        """Gradient norms returned from learn() are all non-negative floats."""
        brain = SpiderBrain(seed=14, module_dropout=0.0)
        obs = _blank_obs()
        # sample=True so training mode is active and motor_cortex caches state for backward
        step = brain.act(obs, sample=True)
        stats = brain.learn(step, reward=-1.0, next_observation=obs, done=False)

        for name, norm in stats["module_gradient_norms"].items():
            self.assertIsInstance(float(norm), float)
            self.assertGreaterEqual(float(norm), 0.0, f"Norm for {name!r} should be >= 0")

    def test_frozen_proposer_skips_module_update_and_reports_zero_gradient_norm(self) -> None:
        brain = SpiderBrain(seed=15, module_dropout=0.0)
        obs = _blank_obs()
        brain.freeze_proposers(("alert_center",))
        frozen_before = brain.module_bank.modules["alert_center"].W1.copy()
        trainable_before = brain.module_bank.modules["hunger_center"].W1.copy()

        step = brain.act(obs, sample=False, training=True)
        stats = brain.learn(step, reward=1.0, next_observation=obs, done=True)

        np.testing.assert_allclose(
            brain.module_bank.modules["alert_center"].W1,
            frozen_before,
        )
        self.assertEqual(stats["module_gradient_norms"]["alert_center"], 0.0)
        self.assertFalse(
            np.allclose(
                brain.module_bank.modules["hunger_center"].W1,
                trainable_before,
            )
        )


if __name__ == "__main__":
    unittest.main()
