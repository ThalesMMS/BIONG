from __future__ import annotations

from .shared import *


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
