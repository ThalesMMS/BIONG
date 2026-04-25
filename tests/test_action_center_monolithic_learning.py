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
from tests.fixtures.action_center_monolithic import MonolithicArchitectureFixtures


class SpiderBrainMonolithicLearningTest(MonolithicArchitectureFixtures, unittest.TestCase):
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

        expected_reflex_aux = brain._auxiliary_module_gradients(step.module_results)
        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        np.testing.assert_allclose(
            captured["bank_grad"],
            np.zeros_like(captured["shared_grad"]),
        )
        for idx, result in enumerate(step.module_results):
            expected_total_grad = result.gate_weight * (
                captured["shared_grad"] + proposal_grads[idx]
            )
            expected_total_grad += result.gate_weight * expected_reflex_aux.get(
                result.name,
                np.zeros(len(LOCOMOTION_ACTIONS), dtype=float),
            )
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                expected_total_grad,
            )
            self.assertAlmostEqual(
                stats["module_credit_weights"][result.name],
                1.0,
            )
            self.assertAlmostEqual(
                stats["module_gradient_norms"][result.name],
                float(np.linalg.norm(expected_total_grad)),
            )
        self.assertEqual(stats["aux_modules"], float(len(expected_reflex_aux)))
        self.assertEqual(stats["credit_strategy"], "broadcast")

    def test_learn_projects_action_center_input_grads_to_arbitration_gates(self) -> None:
        config = BrainAblationConfig(
            name="modular_no_reflex_aux",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
        )
        brain = SpiderBrain(
            seed=7,
            config=config,
            arbitration_regularization_weight=0.25,
            arbitration_valence_regularization_weight=0.4,
        )
        brain.arbitration_network.b2_gate[:] = np.log(0.7 / 0.3)
        obs = self._build_observation()
        step = brain.act(obs, sample=False, training=True)
        arbitration = step.arbitration_decision
        self.assertIsNotNone(arbitration)

        captured: dict[str, object] = {}
        proposal_grads = np.linspace(
            -0.2,
            0.3,
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
            """
            Capture a copy of the policy logits gradient into the outer `captured` mapping and return the upstream gradient unchanged.

            Parameters:
                grad_policy_logits (array-like): Gradient of the policy logits to capture; converted to a float NumPy array and stored as `captured["policy_grad"]`.
                grad_value: Ignored.
                lr: Ignored.

            Returns:
                The upstream gradient value unchanged.
            """
            captured["policy_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_arbitration_backward(*, grad_valence_logits, grad_gate_adjustments, grad_value, lr):
            """
            Record provided arbitration gradients and learning rate into the `captured` mapping and return a zeroed input-gradient vector matching the arbitration network input dimension.

            Parameters:
                grad_valence_logits (array-like): Gradients for arbitration valence logits; stored as a float NumPy array under key `"arb_valence_grad"`.
                grad_gate_adjustments (array-like): Gradients for arbitration gate adjustments; stored as a float NumPy array under key `"arb_gate_grad"`.
                grad_value (float): Gradient for the arbitration value output; stored as a float under key `"arb_value_grad"`.
                lr (float): Learning rate used for this backward pass; stored as a float under key `"arb_lr"`.

            Returns:
                numpy.ndarray: A zero-filled float array whose length equals the arbitration network input dimension.
            """
            captured["arb_valence_grad"] = np.asarray(grad_valence_logits, dtype=float).copy()
            captured["arb_gate_grad"] = np.asarray(grad_gate_adjustments, dtype=float).copy()
            captured["arb_value_grad"] = float(grad_value)
            captured["arb_lr"] = float(lr)
            return np.zeros(brain.arbitration_network.input_dim, dtype=float)

        brain.action_center.backward = fake_action_center_backward
        brain.arbitration_network.backward = fake_arbitration_backward
        brain.module_bank.backward = lambda grad_logits, lr, aux_grads=None: None
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        td_target = 0.5
        advantage = float(np.clip(td_target - step.value, -4.0, 4.0))
        valence_probs = np.array(
            [
                float(arbitration.valence_scores[name])
                for name in SpiderBrain.VALENCE_ORDER
            ],
            dtype=float,
        )
        fixed_targets = brain._fixed_formula_valence_scores_from_evidence(arbitration.evidence)
        expected_valence_grad = advantage * (
            valence_probs
            - np.eye(len(SpiderBrain.VALENCE_ORDER))[SpiderBrain.VALENCE_ORDER.index(arbitration.winning_valence)]
        )
        expected_valence_grad += brain.arbitration_valence_regularization_weight * (
            valence_probs - fixed_targets
        )

        gate_indices = {
            name: index
            for index, name in enumerate(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER)
        }
        expected_final_gate_grad = np.zeros(
            len(SpiderBrain.ARBITRATION_GATE_MODULE_ORDER),
            dtype=float,
        )
        expected_gate_reg = np.zeros_like(expected_final_gate_grad)
        expected_base = np.zeros_like(expected_final_gate_grad)
        for idx, result in enumerate(step.module_results):
            gate_index = gate_indices[result.name]
            raw_logits = result.post_reflex_logits
            self.assertIsNotNone(raw_logits)
            expected_base[gate_index] = arbitration.base_gates[result.name]
            expected_final_gate_grad[gate_index] += float(
                np.dot(proposal_grads[idx], raw_logits)
            )
            expected_gate_reg[gate_index] += (
                brain.arbitration_regularization_weight
                * (arbitration.module_gates[result.name] - arbitration.base_gates[result.name])
            )
        expected_gate_grad = expected_base * (
            expected_final_gate_grad + expected_gate_reg
        )

        np.testing.assert_allclose(captured["arb_valence_grad"], expected_valence_grad)
        np.testing.assert_allclose(captured["arb_gate_grad"], expected_gate_grad)
        self.assertAlmostEqual(captured["arb_value_grad"], arbitration.arbitration_value - td_target)
        self.assertEqual(captured["arb_lr"], brain.arbitration_lr)
        self.assertGreater(stats["arbitration_gate_regularization_norm"], 0.0)
        self.assertGreater(stats["arbitration_valence_regularization_norm"], 0.0)
        self.assertIn("arbitration_loss", stats)
        self.assertIn("gate_adjustment_magnitude", stats)
        self.assertIn("regularization_loss", stats)
        self.assertGreater(stats["gate_adjustment_magnitude"], 0.0)
        self.assertGreater(stats["regularization_loss"], 0.0)

    def test_local_credit_only_learn_omits_shared_policy_broadcast(self) -> None:
        """
        Verifies that the "local_credit_only" ablation prevents shared policy gradients from being broadcast to module auxiliary gradient paths.

        Sets up a modular brain configured for the local-credit-only variant, stubs the action-center and module-bank backward methods to capture gradients, invokes learning on a single step, and asserts each module's auxiliary gradient equals the module's gate weight multiplied by its proposal gradients (i.e., only local gate-weighted proposal gradients are applied to module auxiliary paths).
        """
        config = BrainAblationConfig(
            name="local_credit_only",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="local_only",
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
            return upstream

        def fake_module_bank_backward(grad_logits, lr, aux_grads=None):
            captured["aux_grads"] = {
                name: np.asarray(value, dtype=float).copy()
                for name, value in (aux_grads or {}).items()
            }

        brain.action_center.backward = fake_action_center_backward
        brain.module_bank.backward = fake_module_bank_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        for idx, result in enumerate(step.module_results):
            expected_total_grad = result.gate_weight * proposal_grads[idx]
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                expected_total_grad,
            )
            self.assertAlmostEqual(
                stats["module_credit_weights"][result.name],
                0.0,
            )
            self.assertAlmostEqual(
                stats["module_gradient_norms"][result.name],
                float(np.linalg.norm(expected_total_grad)),
            )
        self.assertEqual(stats["credit_strategy"], "local_only")

    def test_counterfactual_credit_normalizes_absolute_interference(self) -> None:
        """
        Verifies that counterfactual credit weights normalize absolute interference between modules.

        Constructs two module results with opposing effects and asserts the counterfactual credit
        computation:
        - returns positive weights for each contributing module,
        - normalizes weights to sum to 1.0,
        - assigns different weights when modules have different absolute contributions.
        """
        brain = SpiderBrain(
            seed=7,
            config=BrainAblationConfig(
                name="counterfactual_credit",
                architecture="modular",
                module_dropout=0.0,
                enable_reflexes=False,
                enable_auxiliary_targets=False,
                credit_strategy="counterfactual",
            ),
        )
        action_dim = len(LOCOMOTION_ACTIONS)

        def module_result(module_name: str, logits: np.ndarray) -> ModuleResult:
            """
            Create a ready-to-use active ModuleResult for the given module name using the provided logits.

            Parameters:
                module_name (str): Module identifier; its interface is looked up to set `interface` and `observation_key`.
                logits (np.ndarray): Logits for the module's action outputs; length must match the action dimension used by the brain.

            Returns:
                ModuleResult: A ModuleResult with the resolved interface and observation_key, a zeroed observation vector sized to the interface's input dimension, the supplied `logits`, a uniform probability vector over actions, and `active=True`.
            """
            interface = MODULE_INTERFACE_BY_NAME[module_name]
            return ModuleResult(
                interface=interface,
                name=module_name,
                observation_key=interface.observation_key,
                observation=np.zeros(interface.input_dim, dtype=float),
                logits=logits,
                probs=np.full(action_dim, 1.0 / action_dim, dtype=float),
                active=True,
            )

        visual_logits = np.zeros(action_dim, dtype=float)
        sensory_logits = np.zeros(action_dim, dtype=float)
        visual_logits[0] = 1.0
        sensory_logits[0] = -1.0
        visual = module_result("visual_cortex", visual_logits)
        sensory = module_result("sensory_cortex", sensory_logits)
        def fake_action_center_forward(x, *, store_cache=False):
            """
            Compute a small corrective action logits vector from a concatenated visual+sensory input and return it with a zero baseline value.

            Parameters:
                x (array-like): Flattened input whose first `action_dim` entries are the visual segment and the next `action_dim` entries are the sensory segment; length must be at least 2 * action_dim.

            Returns:
                tuple:
                    correction (numpy.ndarray): 1D float array of length `action_dim` containing the corrective logits.
                    baseline (float): A scalar baseline value (always 0.0).
            """
            action_input = np.asarray(x, dtype=float)
            visual_segment = action_input[:action_dim]
            sensory_segment = action_input[action_dim : 2 * action_dim]
            correction = np.zeros(action_dim, dtype=float)
            correction[0] = (
                0.75 * visual_segment[0]
                - 0.50 * sensory_segment[0]
            )
            correction[1] = (
                -0.25 * visual_segment[0]
                + 0.25 * sensory_segment[0]
            )
            return correction, 0.0

        brain.action_center.forward = fake_action_center_forward

        weights = brain._compute_counterfactual_credit(
            [visual, sensory],
            self._build_observation(),
            action_idx=0,
        )

        self.assertEqual(set(weights), {"visual_cortex", "sensory_cortex"})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["visual_cortex"], 0.0)
        self.assertGreater(weights["sensory_cortex"], 0.0)
        self.assertNotAlmostEqual(
            weights["visual_cortex"],
            weights["sensory_cortex"],
        )

    def test_counterfactual_learn_scales_shared_policy_broadcast(self) -> None:
        config = BrainAblationConfig(
            name="counterfactual_credit",
            architecture="modular",
            module_dropout=0.0,
            enable_reflexes=False,
            enable_auxiliary_targets=False,
            credit_strategy="counterfactual",
        )
        brain = SpiderBrain(seed=7, config=config)
        obs = self._build_observation()
        step = brain.act(obs, sample=False)
        weights = {
            result.name: float(index + 1)
            for index, result in enumerate(step.module_results)
        }
        weight_total = float(sum(weights.values()))
        weights = {
            name: value / weight_total
            for name, value in weights.items()
        }

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
            """
            Stub backward function used in tests to capture the shared policy gradient.

            Stores a copy of `grad_policy_logits` (as a float numpy array) into `captured["shared_grad"]` and returns the incoming `upstream` value unchanged.

            Parameters:
                grad_policy_logits: array-like
                    Gradients for the policy logits to be captured.
                grad_value:
                    Ignored by this stub.
                lr:
                    Ignored by this stub.

            Returns:
                upstream: the unchanged upstream value passed through by the stub.
            """
            captured["shared_grad"] = np.asarray(grad_policy_logits, dtype=float).copy()
            return upstream

        def fake_module_bank_backward(grad_logits, lr, aux_grads=None):
            """
            Capture and store a copy of auxiliary gradients into the outer `captured` mapping.

            Parameters:
                grad_logits: Unused in this helper; present to match the real backward signature.
                lr: Unused learning-rate parameter; present to match the real backward signature.
                aux_grads (dict[str, array_like] | None): Mapping from module name to gradient array. Each value is converted to a NumPy float array and stored as a copy in `captured["aux_grads"]`.
            """
            captured["aux_grads"] = {
                name: np.asarray(value, dtype=float).copy()
                for name, value in (aux_grads or {}).items()
            }

        brain._compute_counterfactual_credit = lambda module_results, observation, action_idx: dict(weights)
        brain.action_center.backward = fake_action_center_backward
        brain.module_bank.backward = fake_module_bank_backward
        brain.motor_cortex.backward = lambda grad_logits, lr: None
        brain.estimate_value = lambda _: 0.0

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        shared_grad = np.asarray(captured["shared_grad"], dtype=float)
        for idx, result in enumerate(step.module_results):
            expected_total_grad = result.gate_weight * (
                weights[result.name] * shared_grad + proposal_grads[idx]
            )
            np.testing.assert_allclose(
                captured["aux_grads"][result.name],
                expected_total_grad,
            )
            self.assertAlmostEqual(
                stats["module_gradient_norms"][result.name],
                float(np.linalg.norm(expected_total_grad)),
            )
        self.assertEqual(stats["credit_strategy"], "counterfactual")
        self.assertEqual(stats["module_credit_weights"], weights)
        self.assertEqual(stats["counterfactual_credit_weights"], weights)

    def test_monolithic_learn_adds_action_center_input_grads_to_policy_update(self) -> None:
        """
        Test that monolithic learning adds action-center input gradients to the monolithic policy update and records credit diagnostics.

        Sets up a monolithic SpiderBrain with stubbed backward methods to capture gradients, runs a single learn step, and asserts:
        - the monolithic policy backward receives the elementwise sum of the shared policy gradient and action-center input gradients,
        - the motor cortex receives only the shared policy gradient,
        - learning stats include `credit_strategy == "broadcast"`,
        - `module_credit_weights` contains the monolithic policy with weight 1.0,
        - `module_gradient_norms` contains the norm of the monolithic policy gradient.
        """
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
        self.assertAlmostEqual(
            stats["module_gradient_norms"][SpiderBrain.MONOLITHIC_POLICY_NAME],
            float(np.linalg.norm(captured["mono_grad"])),
        )
        np.testing.assert_allclose(
            captured["motor_grad"],
            captured["shared_grad"],
        )

    def test_monolithic_frozen_proposer_skips_weight_update_and_reports_zero_norm(self) -> None:
        brain = SpiderBrain(seed=11, config=self._monolithic_config())
        obs = self._build_observation()
        step = brain.act(obs, sample=True)

        brain.freeze_proposers([SpiderBrain.MONOLITHIC_POLICY_NAME])
        monolithic_before = self._numeric_state(brain.monolithic_policy)
        action_center_before = self._numeric_state(brain.action_center)
        motor_cortex_before = self._numeric_state(brain.motor_cortex)

        stats = brain.learn(step, reward=0.5, next_observation=obs, done=False)

        self.assertEqual(
            brain.frozen_module_names(),
            [SpiderBrain.MONOLITHIC_POLICY_NAME],
        )
        self.assertFalse(
            self._state_changed(
                monolithic_before,
                self._numeric_state(brain.monolithic_policy),
            )
        )
        self.assertEqual(
            stats["module_gradient_norms"][SpiderBrain.MONOLITHIC_POLICY_NAME],
            0.0,
        )
        self.assertTrue(
            self._state_changed(
                action_center_before,
                self._numeric_state(brain.action_center),
            )
        )
        self.assertTrue(
            self._state_changed(
                motor_cortex_before,
                self._numeric_state(brain.motor_cortex),
            )
        )

    def test_monolithic_local_only_credit_reports_effective_broadcast(self) -> None:
        self._assert_monolithic_non_broadcast_credit_coerces_to_broadcast("local_only")

    def test_monolithic_counterfactual_credit_reports_effective_broadcast(self) -> None:
        self._assert_monolithic_non_broadcast_credit_coerces_to_broadcast(
            "counterfactual"
        )
