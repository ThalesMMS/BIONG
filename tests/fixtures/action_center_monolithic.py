"""Comprehensive tests for the action_center addition introduced in this PR.

Covers:
- architecture_signature() new structure (action_center, motor_cortex keys)
- SpiderBrain.ARCHITECTURE_VERSION = 12
- SpiderBrain.action_center / motor_cortex network types
- BrainStep new fields (action_center_logits, action_center_policy,
  action_intent_idx, motor_override, action_center_input)
- SpiderBrain.estimate_value uses action_center, not motor_cortex alone
- _build_motor_input NaN / inf sanitization
- ActionContextObservation and MotorContextObservation field composition
- ACTION_CONTEXT_INTERFACE / MOTOR_CONTEXT_INTERFACE properties
- OBSERVATION_DIMS updated entries
- build_action_context_observation / build_motor_context_observation
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from spider_cortex_sim.ablations import (
    BrainAblationConfig,
    MONOLITHIC_POLICY_NAME,
    TRUE_MONOLITHIC_POLICY_NAME,
    canonical_ablation_configs,
    default_brain_config,
)
from spider_cortex_sim.agent import BrainStep, SpiderBrain
from spider_cortex_sim.arbitration import ArbitrationDecision
from spider_cortex_sim.bus import MessageBus
from spider_cortex_sim.interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    LOCOMOTION_ACTIONS,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    MODULE_INTERFACE_BY_NAME,
    OBSERVATION_DIMS,
    ActionContextObservation,
    MotorContextObservation,
    architecture_signature,
)
from spider_cortex_sim.maps import CLUTTER, NARROW, OPEN
from spider_cortex_sim.modules import CorticalModuleBank, ModuleResult
from spider_cortex_sim.nn import ArbitrationNetwork, MotorNetwork, ProposalNetwork
from spider_cortex_sim.noise import compute_execution_difficulty
from spider_cortex_sim.operational_profiles import DEFAULT_OPERATIONAL_PROFILE, OperationalProfile
from spider_cortex_sim.perception import (
    PerceivedTarget,
    build_action_context_observation,
    build_motor_context_observation,
)
from spider_cortex_sim.world import SpiderWorld


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from tests.fixtures.action_center import (
    _null_percept,
    _blank_obs,
    _module_interface,
    _vector_for,
    _valence_vector,
    _profile_with_updates,
)

class MonolithicArchitectureFixtures:
    def _monolithic_config(
        self,
        *,
        credit_strategy: str = "broadcast",
    ) -> BrainAblationConfig:
        """
        Create a BrainAblationConfig preconfigured for the monolithic architecture.
        
        Parameters:
            credit_strategy (str): Credit assignment strategy to use for the configuration (e.g., "broadcast").
        
        Returns:
            BrainAblationConfig: Configuration with name set to the monolithic policy, architecture "monolithic",
            module_dropout 0.0, reflexes and auxiliary targets disabled, the provided credit strategy, and no disabled modules.
        """
        return BrainAblationConfig(
            name=MONOLITHIC_POLICY_NAME,
            architecture="monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy=credit_strategy,
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

    def _numeric_state(self, network) -> dict[str, np.ndarray]:
        """
        Extract a mapping of numeric NumPy arrays from a network's state dictionary.
        
        Parameters:
            network: An object exposing a `state_dict()` mapping of parameter names to values.
        
        Returns:
            dict[str, np.ndarray]: A dictionary mapping state keys to copies of their `np.ndarray` values.
        """
        return {
            key: value.copy()
            for key, value in network.state_dict().items()
            if isinstance(value, np.ndarray)
        }

    def _state_changed(
        self,
        before: dict[str, np.ndarray],
        after: dict[str, np.ndarray],
    ) -> bool:
        """
        Check whether any numeric state array changed between two network state snapshots.
        
        Parameters:
            before (dict[str, np.ndarray]): Mapping of parameter names to their numeric arrays from the earlier snapshot.
            after (dict[str, np.ndarray]): Mapping of parameter names to their numeric arrays from the later snapshot; keys should include those present in `before`.
        
        Returns:
            `True` if any array in `after` differs from the corresponding array in `before` according to `np.allclose`, `False` otherwise.
        """
        return any(not np.allclose(before[key], after[key]) for key in before)

    def _true_monolithic_config(
        self,
        *,
        credit_strategy: str = "broadcast",
    ) -> BrainAblationConfig:
        """
        Create a BrainAblationConfig preconfigured for the "true_monolithic" architecture.
        
        Parameters:
            credit_strategy (str): Credit assignment strategy to use for the config (e.g., "broadcast").
        
        Returns:
            BrainAblationConfig: Configuration with name set to TRUE_MONOLITHIC_POLICY_NAME, architecture "true_monolithic", module_dropout 0.0, reflexes and auxiliary targets disabled, learned arbitration disabled, reflex_scale 0.0, and the provided credit strategy.
        """
        return BrainAblationConfig(
            name=TRUE_MONOLITHIC_POLICY_NAME,
            architecture="true_monolithic",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            use_learned_arbitration=False,
            credit_strategy=credit_strategy,
            reflex_scale=0.0,
            disabled_modules=(),
        )

    def _assert_monolithic_non_broadcast_credit_coerces_to_broadcast(
        self,
        credit_strategy: str,
    ) -> None:
        """
        Assert that a non-broadcast monolithic credit strategy is coerced to "broadcast" and that the monolithic policy gradient equals the shared action-center gradient plus locomotion gradients.
        
        Parameters:
            credit_strategy (str): The credit strategy to test (e.g., "individual", "proportional"); the helper verifies it is coerced to "broadcast" for the monolithic architecture.
        
        Raises:
            AssertionError: If the gradient composition, reported credit strategy, module/counterfactual credit weights, or reported module gradient norm do not match the expected values.
        """
        brain = SpiderBrain(
            seed=13,
            config=self._monolithic_config(credit_strategy=credit_strategy),
        )
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
            """
            Create a fake backward function for the action center that captures the gradient passed for policy logits and returns a predefined upstream gradient.
            
            Parameters:
                grad_policy_logits (array-like): Gradient w.r.t. action-center policy logits; this value is copied and stored into the outer-scope `captured["shared_grad"]`.
                grad_value: Unused in this fake; provided to match the real backward signature.
                lr: Unused learning-rate parameter; provided to match the real backward signature.
            
            Returns:
                np.ndarray: The precomputed upstream gradient vector (`upstream`) to be propagated further.
            """
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_monolithic_backward(grad_logits, lr):
            """
            Store a copy of `grad_logits` into `captured["mono_grad"]`.
            
            Parameters:
                grad_logits (array-like): Gradient logits to store; converted to a float numpy array and copied.
                lr (float): Learning rate provided for signature compatibility; ignored.
            """
            captured["mono_grad"] = np.asarray(grad_logits, dtype=float).copy()

        brain.action_center.backward = fake_action_center_backward
        brain.monolithic_policy.backward = fake_monolithic_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(
            captured["mono_grad"],
            captured["shared_grad"] + extra_grad,
        )
        self.assertEqual(stats["credit_strategy"], "broadcast")
        self.assertEqual(
            stats["module_credit_weights"],
            {SpiderBrain.MONOLITHIC_POLICY_NAME: 1.0},
        )
        self.assertEqual(stats["counterfactual_credit_weights"], {})
        self.assertAlmostEqual(
            stats["module_gradient_norms"][SpiderBrain.MONOLITHIC_POLICY_NAME],
            float(np.linalg.norm(captured["mono_grad"])),
        )
