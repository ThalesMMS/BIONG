import unittest
from types import SimpleNamespace

import numpy as np

from spider_cortex_sim.ablations import BrainAblationConfig
from spider_cortex_sim.agent import SpiderBrain
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_TO_INDEX,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MODULE_INTERFACE_BY_NAME,
    MOTOR_CONTEXT_INTERFACE,
    ActionContextObservation,
    MotorContextObservation,
    interface_registry,
)
from spider_cortex_sim.modules import ModuleResult
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.reflexes import (
    _alert_reflex_decision,
    _apply_reflex_path,
    _hunger_reflex_decision,
    _module_reflex_decision,
    _sensory_reflex_decision,
    _sleep_reflex_decision,
    _visual_reflex_decision,
)

class SpiderReflexDecisionTestBase(unittest.TestCase):
    def setUp(self) -> None:
        """
        Prepare the test fixture by creating a deterministic SpiderBrain instance and storing it on self.brain.
        
        Initializes a SpiderBrain with seed=3 and module_dropout=0.0 so tests run with a reproducible brain configuration.
        """
        self.brain = SpiderBrain(seed=3, module_dropout=0.0)

    def _reflex_decision(
        self,
        result: ModuleResult,
        *,
        brain: SpiderBrain | None = None,
    ):
        """
        Selects a SpiderBrain (defaults to the test instance's brain) and obtains the reflex decision for the given module result.
        
        Parameters:
            result (ModuleResult): The module result whose named observation will be evaluated for reflexes.
            brain (SpiderBrain | None): Optional brain to use for operational profile and interface registry; if None, uses self.brain.
        
        Returns:
            ModuleReflex | None: The reflex decision for the module, or `None` if no reflex applies.
        """
        brain = self.brain if brain is None else brain
        return _module_reflex_decision(
            result.name,
            result.named_observation(),
            operational_profile=brain.operational_profile,
            interface_registry=brain._interface_registry(),
        )

    def _module_result(self, module_name: str, **signals: float) -> ModuleResult:
        """
        Builds a ModuleResult for the given module using a default observation with optional signal overrides.
        
        Parameters:
            module_name (str): Name of the module whose interface provides input names and observation key.
            **signals (float): Mapping of input names to values that override the default signals. By default inputs whose names end with "_age" are set to 1.0 and all other inputs are set to 0.0 before applying overrides.
        
        Returns:
            ModuleResult: A result populated with the module's interface, name, observation_key, the constructed observation vector, zeroed `logits`, a uniform `probs` distribution over locomotion actions, and `active=True`.
        """
        interface = MODULE_INTERFACE_BY_NAME[module_name]
        signal_mapping = {
            spec.name: 1.0 if spec.name.endswith("_age") else 0.0
            for spec in interface.inputs
        }
        signal_mapping.update({name: float(value) for name, value in signals.items()})
        observation = interface.vector_from_mapping(signal_mapping)
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

    def _build_observation(self, **overrides: dict[str, float]) -> dict[str, np.ndarray]:
        """
        Builds a complete observation dictionary for all module, action-context, and motor-context interfaces.

        Constructs default input vectors for each interface where inputs ending in "_age" are set to 1.0 and all other inputs to 0.0, then applies any numeric overrides supplied for an interface's observation key. The resulting vectors are produced via each interface's `vector_from_mapping` and stored under that interface's observation key.

        Parameters:
            overrides (dict[str, dict[str, float]]): Optional per-interface override mappings keyed by interface observation key; each mapping provides input-name -> numeric value to override the defaults.

        Returns:
            dict[str, np.ndarray]: A mapping from each interface's observation key to its validated observation vector.
        """
        observation: dict[str, np.ndarray] = {}
        for interface in MODULE_INTERFACES:
            signal_mapping = {
                spec.name: 1.0 if spec.name.endswith("_age") else 0.0
                for spec in interface.inputs
            }
            signal_mapping.update({
                name: float(value)
                for name, value in overrides.get(interface.observation_key, {}).items()
            })
            observation[interface.observation_key] = interface.vector_from_mapping(signal_mapping)
        action_mapping = {
            spec.name: 1.0 if spec.name.endswith("_age") else 0.0
            for spec in ACTION_CONTEXT_INTERFACE.inputs
        }
        action_mapping.update({
            name: float(value)
            for name, value in overrides.get(ACTION_CONTEXT_INTERFACE.observation_key, {}).items()
        })
        observation[ACTION_CONTEXT_INTERFACE.observation_key] = ACTION_CONTEXT_INTERFACE.vector_from_mapping(
            action_mapping
        )
        motor_mapping = {
            spec.name: 1.0 if spec.name.endswith("_age") else 0.0
            for spec in MOTOR_CONTEXT_INTERFACE.inputs
        }
        motor_mapping.update({
            name: float(value)
            for name, value in overrides.get(MOTOR_CONTEXT_INTERFACE.observation_key, {}).items()
        })
        observation[MOTOR_CONTEXT_INTERFACE.observation_key] = MOTOR_CONTEXT_INTERFACE.vector_from_mapping(
            motor_mapping
        )
        return observation

    def _profile_with_reflex_config_updates(
        self,
        *,
        alert_aux_weight: float | None = None,
        visual_predator_certainty: float | None = None,
    ) -> OperationalProfile:
        """
        Create an OperationalProfile derived from DEFAULT_OPERATIONAL_PROFILE with optional reflex-specific overrides for testing.
        
        Parameters:
            alert_aux_weight (float | None): If provided, sets the `brain.aux_weights.alert_center` value in the profile.
            visual_predator_certainty (float | None): If provided, sets the `brain.reflex_thresholds.visual_cortex.predator_certainty` value in the profile.
        
        Returns:
            OperationalProfile: An OperationalProfile constructed from the modified profile summary.
        """
        summary = DEFAULT_OPERATIONAL_PROFILE.to_summary()
        summary["name"] = "test_profile"
        summary["version"] = 99
        if alert_aux_weight is not None:
            summary["brain"]["aux_weights"]["alert_center"] = alert_aux_weight
        if visual_predator_certainty is not None:
            summary["brain"]["reflex_thresholds"]["visual_cortex"]["predator_certainty"] = visual_predator_certainty
        return OperationalProfile.from_summary(summary)

    def _reordered_action_registry(self) -> dict[str, object]:
        registry = interface_registry()
        actions = list(registry["actions"])
        registry["actions"] = actions[1:] + actions[:1]
        return registry

    def _assert_registry_local_target(self, reflex, registry: dict[str, object]) -> None:
        self.assertIsNotNone(reflex)
        actions = list(registry["actions"])
        action_index = actions.index(reflex.action)
        self.assertEqual(reflex.target_probs.shape, (len(actions),))
        self.assertEqual(int(np.argmax(reflex.target_probs)), action_index)
        self.assertAlmostEqual(float(reflex.target_probs[action_index]), 1.0)

    def _assert_reordered_reflex_decision(
        self,
        decision_fn,
        observations: dict[str, float],
        *,
        expected_action: str,
        expected_reason: str,
    ) -> None:
        registry = self._reordered_action_registry()
        reflex = decision_fn(
            observations,
            operational_profile=DEFAULT_OPERATIONAL_PROFILE,
            interface_registry=registry,
        )

        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.action, expected_action)
        self.assertEqual(reflex.reason, expected_reason)
        self._assert_registry_local_target(reflex, registry)
