from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Sequence

import numpy as np

from ..ablations import BrainAblationConfig, default_brain_config
from ..arbitration import (
    ARBITRATION_EVIDENCE_FIELDS as DEFAULT_ARBITRATION_EVIDENCE_FIELDS,
    ARBITRATION_GATE_MODULE_ORDER as DEFAULT_ARBITRATION_GATE_MODULE_ORDER,
    ARBITRATION_NETWORK_NAME as DEFAULT_ARBITRATION_NETWORK_NAME,
    MONOLITHIC_POLICY_NAME as DEFAULT_MONOLITHIC_POLICY_NAME,
    PRIORITY_GATING_WEIGHTS as DEFAULT_PRIORITY_GATING_WEIGHTS,
    VALENCE_EVIDENCE_WEIGHTS as DEFAULT_VALENCE_EVIDENCE_WEIGHTS,
    VALENCE_ORDER as DEFAULT_VALENCE_ORDER,
    ArbitrationDecision,
    ValenceScore,
    apply_priority_gating,
    arbitration_evidence_input_dim,
    arbitration_gate_weight_for,
    clamp_unit,
    compute_arbitration,
    fixed_formula_valence_scores_from_evidence,
    warm_start_arbitration_network,
)
from ..bus import MessageBus
from ..interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    architecture_signature,
    interface_registry,
)
from ..modules import MODULE_HIDDEN_DIMS, CorticalModuleBank, ModuleResult, ReflexDecision
from ..nn import (
    ArbitrationNetwork,
    MotorNetwork,
    ProposalNetwork,
    RecurrentProposalNetwork,
    one_hot,
    softmax,
)
from ..noise import _compute_execution_difficulty_core
from ..operational_profiles import OperationalProfile, runtime_operational_profile
from ..reflexes import (
    _apply_reflex_path as apply_reflex_path,
    _direction_action as direction_action,
    _module_reflex_decision as module_reflex_decision,
)
from ..world import ACTIONS

from .types import BrainStep


class BrainRuntimeMixin:
    @staticmethod
    def _network_forward_macs(network: object) -> int:
        if isinstance(network, RecurrentProposalNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
            )
        if isinstance(network, ArbitrationNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.valence_dim
                + network.hidden_dim * network.gate_dim
                + network.hidden_dim
            )
        if isinstance(network, MotorNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
            )
        if isinstance(network, ProposalNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
            )
        return 0

    def estimate_compute_cost(self) -> Dict[str, object]:
        per_network: Dict[str, int] = {}
        if self.module_bank is not None:
            per_network.update(
                {
                    name: self._network_forward_macs(network)
                    for name, network in self.module_bank.modules.items()
                }
            )
        if self.monolithic_policy is not None:
            per_network[self.MONOLITHIC_POLICY_NAME] = self._network_forward_macs(
                self.monolithic_policy
            )
        if self.true_monolithic_policy is not None:
            per_network[self.TRUE_MONOLITHIC_POLICY_NAME] = self._network_forward_macs(
                self.true_monolithic_policy
            )
        if self.arbitration_network is not None:
            per_network[self.ARBITRATION_NETWORK_NAME] = self._network_forward_macs(
                self.arbitration_network
            )
        if self.action_center is not None:
            per_network["action_center"] = self._network_forward_macs(
                self.action_center
            )
        if self.motor_cortex is not None:
            per_network["motor_cortex"] = self._network_forward_macs(
                self.motor_cortex
            )
        return {
            "unit": "approx_forward_macs",
            "per_network": per_network,
            "total": int(sum(per_network.values())),
        }

    def _true_monolithic_arbitration_decision(
        self,
        *,
        module_name: str,
        action_idx: int,
    ) -> ArbitrationDecision:
        """Build a trivial arbitration payload for the direct-control baseline."""
        return ArbitrationDecision(
            strategy="direct_control",
            winning_valence="exploration",
            valence_scores={
                "threat": 0.0,
                "hunger": 0.0,
                "sleep": 0.0,
                "exploration": 1.0,
            },
            module_gates={module_name: 1.0},
            suppressed_modules=[],
            evidence={},
            intent_before_gating_idx=int(action_idx),
            intent_after_gating_idx=int(action_idx),
            valence_logits={
                "threat": 0.0,
                "hunger": 0.0,
                "sleep": 0.0,
                "exploration": 1.0,
            },
            base_gates={module_name: 1.0},
            gate_adjustments={module_name: 1.0},
            arbitration_value=0.0,
            learned_adjustment=False,
            module_contribution_share={module_name: 1.0},
            dominant_module=module_name,
            dominant_module_share=1.0,
            effective_module_count=1.0,
            module_agreement_rate=1.0,
            module_disagreement_rate=0.0,
        )

    def set_runtime_reflex_scale(self, scale: float) -> None:
        """
        Set the runtime multiplier applied to reflex strengths.
        
        Parameters:
            scale (float): Desired reflex scale; must be finite. Negative values are clamped to 0.0 and the value is stored as a float in `self.current_reflex_scale`.
        """
        value = float(scale)
        if not math.isfinite(value):
            raise ValueError("non-finite reflex scale")
        self.current_reflex_scale = max(0.0, value)

    def reset_runtime_reflex_scale(self) -> None:
        """
        Restore the runtime reflex scale to the configured default.
        
        Clamps the configured reflex scale to a minimum of 0.0 and sets self.current_reflex_scale accordingly. Raises ValueError if the configured reflex scale is not finite.
        """
        value = float(self.config.reflex_scale)
        if not math.isfinite(value):
            raise ValueError("non-finite reflex scale")
        self.current_reflex_scale = max(0.0, value)

    def _effective_reflex_scale(self, module_name: str) -> float:
        """
        Compute the non-negative reflex scaling factor for the given module.
        
        If reflexes are disabled or the architecture is not modular, this returns 0.0. Otherwise
        returns the product of the current runtime reflex scale and the module's configured
        multiplier, clamped to be greater than or equal to 0.0.
        
        Returns:
            float: Effective scale applied to reflex strengths for the module (>= 0.0).
        """
        if not self.config.enable_reflexes or not self.config.is_modular:
            return 0.0
        return max(
            0.0,
            float(self.current_reflex_scale)
            * float(self.config.module_reflex_scales.get(module_name, 1.0)),
        )

    def act(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = True,
        policy_mode: str = "normal",
        training: bool | None = None,
    ) -> BrainStep:
        """
        Choose and execute an action for the provided observation.
        
        Builds per-module proposals, optionally applies reflexes, computes arbitration and priority gating, runs action-center and motor-cortex corrections (unless `policy_mode == "reflex_only"`), and returns a populated BrainStep describing module results, logits/policies, selected intent/action, overrides, controller inputs, and the arbitration decision.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Mapping of interface observation arrays consumed by proposal modules and context interfaces.
            bus (MessageBus | None): Optional message bus for publishing per-module proposal diagnostics and final selection/execution diagnostics; pass None to disable publishing.
            sample (bool): If True, sample the executed action from the final policy distribution; if False, select the greedy argmax action.
            policy_mode (str): Execution mode, either "normal" to apply learned action-center and motor-cortex corrections, or "reflex_only" to select directly from post-reflex modular proposals. "reflex_only" requires a modular architecture with reflexes enabled.
            training (bool | None): If provided, forces training mode on/off for internal network cache and learned-arbitration behavior; if None, training mode is inferred from `sample` or an internal override.
        
        Returns:
            BrainStep: Decision container populated with per-module ModuleResult entries, action-center and motor-cortex logits/policies, combined logits with and without reflexes, the final policy and value estimate, selected intent and action indices, override flags, controller input vectors, the active `policy_mode`, and the computed `arbitration_decision`.
        
        Raises:
            ValueError: If `policy_mode` is not "normal" or "reflex_only", or if `policy_mode == "reflex_only"` is requested but the brain is not modular or reflexes are disabled.
        """
        if policy_mode not in {"normal", "reflex_only"}:
            raise ValueError(
                "Invalid policy_mode. Use 'normal' or 'reflex_only'."
            )
        if policy_mode == "reflex_only" and not self.config.is_modular:
            raise ValueError(
                "policy_mode='reflex_only' requires the modular architecture."
            )
        if policy_mode == "reflex_only" and not self.config.enable_reflexes:
            raise ValueError(
                "policy_mode='reflex_only' requires reflexes to be enabled."
            )

        runtime_training = getattr(self, "_act_training_override", None)
        if training is None:
            training_mode = bool(sample if runtime_training is None else runtime_training)
        else:
            training_mode = bool(training)
        store_cache = training_mode and policy_mode == "normal"
        proposal_sum = np.zeros(self.action_dim, dtype=float)
        action_center_input = np.zeros(0, dtype=float)
        motor_input = np.zeros(0, dtype=float)
        action_center_correction_logits = np.zeros(self.action_dim, dtype=float)
        motor_correction_logits = np.zeros(self.action_dim, dtype=float)
        value = 0.0
        arbitration = None
        if self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            policy_logits, value = self.true_monolithic_policy.forward(
                monolithic_observation,
                store_cache=store_cache,
            )
            direct_result = ModuleResult(
                interface=None,
                name=self.TRUE_MONOLITHIC_POLICY_NAME,
                observation_key=self.TRUE_MONOLITHIC_POLICY_NAME,
                observation=monolithic_observation.copy(),
                logits=policy_logits.copy(),
                probs=softmax(policy_logits),
                active=True,
                reflex=None,
                neural_logits=policy_logits.copy(),
                reflex_delta_logits=np.zeros_like(policy_logits),
                post_reflex_logits=policy_logits.copy(),
            )
            module_results = [direct_result]
            direct_result.valence_role = "integrated_policy"
            direct_result.gate_weight = 1.0
            direct_result.gated_logits = direct_result.logits.copy()
            direct_result.contribution_share = 1.0
            direct_result.intent_before_gating = ACTIONS[int(np.argmax(direct_result.logits))]
            direct_result.intent_after_gating = direct_result.intent_before_gating
            total_logits_without_reflex = direct_result.logits.copy()
            total_logits = direct_result.logits.copy()
            action_center_logits = total_logits.copy()
            action_center_policy = softmax(action_center_logits)
            proposal_sum = total_logits.copy()
            policy = softmax(total_logits)
            action_intent_idx = int(np.argmax(action_center_policy))
            action_intent_without_reflex_idx = action_intent_idx
            action_without_reflex_idx = action_intent_idx
            motor_action_idx = action_intent_idx
            arbitration = self._true_monolithic_arbitration_decision(
                module_name=self.TRUE_MONOLITHIC_POLICY_NAME,
                action_idx=action_intent_idx,
            )
            if sample:
                action_idx = int(self.rng.choice(self.action_dim, p=policy))
            else:
                action_idx = motor_action_idx
            motor_override = False
            final_reflex_override = False
        else:
            module_results = self._proposal_results(
                observation,
                store_cache=store_cache,
                training=training_mode,
            )
            arbitration_without_reflex = self._compute_arbitration(
                module_results,
                observation,
                training=False,
                store_cache=False,
            )
            gated_logits_without_reflex = [
                arbitration_gate_weight_for(arbitration_without_reflex, result.name) * result.logits
                for result in module_results
            ]
            proposal_sum_without_reflex = np.sum(
                np.stack(gated_logits_without_reflex, axis=0),
                axis=0,
            )
            if policy_mode == "reflex_only":
                action_center_logits_without_reflex = proposal_sum_without_reflex.copy()
                action_intent_without_reflex_idx = int(
                    np.argmax(action_center_logits_without_reflex)
                )
                total_logits_without_reflex = proposal_sum_without_reflex.copy()
            else:
                action_context_mapping = self._bound_action_context(observation)
                action_context = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_context_mapping)
                action_input_without_reflex = np.concatenate(
                    [np.concatenate(gated_logits_without_reflex, axis=0), action_context],
                    axis=0,
                )
                if self.action_center is None:
                    raise RuntimeError("Action center unavailable for the configured architecture.")
                action_center_correction_without_reflex, _ = self.action_center.forward(
                    action_input_without_reflex,
                    store_cache=False,
                )
                action_center_logits_without_reflex = (
                    proposal_sum_without_reflex + action_center_correction_without_reflex
                )
                action_intent_without_reflex_idx = int(
                    np.argmax(action_center_logits_without_reflex)
                )
                motor_input_without_reflex = self._build_motor_input(
                    one_hot(action_intent_without_reflex_idx, self.action_dim),
                    observation,
                )
                if self.motor_cortex is None:
                    raise RuntimeError("Motor cortex unavailable for the configured architecture.")
                motor_correction_without_reflex = self.motor_cortex.forward(
                    motor_input_without_reflex,
                    store_cache=False,
                )
                total_logits_without_reflex = (
                    action_center_logits_without_reflex + motor_correction_without_reflex
                )

            apply_reflex_path(
                module_results,
                ablation_config=self.config,
                operational_profile=self.operational_profile,
                interface_registry=self._interface_registry(),
                current_reflex_scale=self.current_reflex_scale,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )
            arbitration = self._compute_arbitration(
                module_results,
                observation,
                training=training_mode,
                store_cache=store_cache,
            )
            apply_priority_gating(
                module_results,
                arbitration,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )

        if bus is not None:
            for result in module_results:
                bus.publish(
                    sender=result.name,
                    topic="action.proposal",
                    payload={
                        "active": bool(result.active),
                        "action_logits": result.logits.round(6).tolist(),
                        "action_probs": result.probs.round(6).tolist(),
                        "neural_logits": result.neural_logits.round(6).tolist() if result.neural_logits is not None else None,
                        "reflex_delta_logits": result.reflex_delta_logits.round(6).tolist() if result.reflex_delta_logits is not None else None,
                        "post_reflex_logits": result.post_reflex_logits.round(6).tolist() if result.post_reflex_logits is not None else None,
                        "reflex_applied": bool(result.reflex_applied),
                        "effective_reflex_scale": round(float(result.effective_reflex_scale), 6),
                        "module_reflex_override": bool(result.module_reflex_override),
                        "module_reflex_dominance": round(float(result.module_reflex_dominance), 6),
                        "reflex": result.reflex.to_payload() if result.reflex is not None else None,
                        "valence_role": result.valence_role,
                        "gate_weight": round(float(result.gate_weight), 6),
                        "contribution_share": round(float(result.contribution_share), 6),
                        "gated_logits": result.gated_logits.round(6).tolist() if result.gated_logits is not None else None,
                        "intent_before_gating": result.intent_before_gating,
                        "intent_after_gating": result.intent_after_gating,
                    },
                )

        if not self.config.is_true_monolithic:
            proposal_sum = np.sum(
                np.stack(
                    [
                        result.gated_logits if result.gated_logits is not None else result.logits
                        for result in module_results
                    ],
                    axis=0,
                ),
                axis=0,
            )
            action_center_input = self._build_action_input(module_results, observation)
            if policy_mode == "reflex_only":
                action_center_logits = proposal_sum.copy()
                action_center_policy = softmax(action_center_logits)
                action_intent_idx = int(np.argmax(action_center_policy))
                motor_input = self._build_motor_input(
                    one_hot(action_intent_idx, self.action_dim),
                    observation,
                )
                total_logits = action_center_logits.copy()
                policy = softmax(total_logits)
                action_without_reflex_idx = int(np.argmax(total_logits_without_reflex))
                motor_action_idx = int(np.argmax(total_logits))
                if sample:
                    action_idx = int(self.rng.choice(self.action_dim, p=policy))
                else:
                    action_idx = motor_action_idx
                motor_override = False
                final_reflex_override = action_without_reflex_idx != motor_action_idx
            else:
                if self.action_center is None or self.motor_cortex is None:
                    raise RuntimeError(
                        "Action/motor pipeline unavailable for the configured architecture."
                    )
                action_center_correction_logits, value = self.action_center.forward(
                    action_center_input,
                    store_cache=store_cache,
                )
                action_center_logits = proposal_sum + action_center_correction_logits
                action_center_policy = softmax(action_center_logits)
                action_intent_idx = int(np.argmax(action_center_policy))
                motor_input = self._build_motor_input(
                    one_hot(action_intent_idx, self.action_dim),
                    observation,
                )
                motor_correction_logits = self.motor_cortex.forward(
                    motor_input,
                    store_cache=store_cache,
                )
                total_logits = action_center_logits + motor_correction_logits
                motor_action_idx_before_food_bias = int(np.argmax(total_logits))
                if (
                    self.config.enable_food_direction_bias
                    and not training_mode
                    and not sample
                    and arbitration is not None
                    and arbitration.winning_valence == "hunger"
                ):
                    hunger_obs = self._bound_observation("hunger_center", observation)
                    food_dx = 0.0
                    food_dy = 0.0
                    if hunger_obs["food_visible"] > 0.0 and hunger_obs["food_certainty"] > 0.0:
                        food_dx = hunger_obs["food_dx"]
                        food_dy = hunger_obs["food_dy"]
                    elif hunger_obs["food_trace_strength"] > 0.0:
                        food_dx = hunger_obs["food_trace_dx"]
                        food_dy = hunger_obs["food_trace_dy"]
                    elif hunger_obs["food_memory_age"] < 1.0:
                        food_dx = hunger_obs["food_memory_dx"]
                        food_dy = hunger_obs["food_memory_dy"]
                    elif hunger_obs["food_smell_strength"] > 0.0:
                        food_dx = hunger_obs["food_smell_dx"]
                        food_dy = hunger_obs["food_smell_dy"]
                    food_bias_action = direction_action(food_dx, food_dy)
                    if food_bias_action != "STAY":
                        total_logits = total_logits.copy()
                        total_logits[ACTION_TO_INDEX[food_bias_action]] += 3.0
                        arbitration = replace(
                            arbitration,
                            food_bias_applied=True,
                            food_bias_action=food_bias_action,
                        )
                policy = softmax(total_logits)
                action_without_reflex_idx = int(np.argmax(total_logits_without_reflex))
                motor_action_idx = int(np.argmax(total_logits))
                if sample:
                    action_idx = int(self.rng.choice(self.action_dim, p=policy))
                else:
                    action_idx = motor_action_idx
                motor_override = action_intent_idx != motor_action_idx
                final_reflex_override = (
                    action_intent_without_reflex_idx != action_intent_idx
                    or action_without_reflex_idx != motor_action_idx_before_food_bias
                )

        execution_diagnostics = self._motor_execution_diagnostics(
            observation,
            action_idx,
        )
        orientation_alignment = float(execution_diagnostics["orientation_alignment"])
        terrain_difficulty = float(execution_diagnostics["terrain_difficulty"])
        momentum = float(execution_diagnostics["momentum"])
        execution_difficulty = float(execution_diagnostics["execution_difficulty"])

        if bus is not None:
            if self.config.is_true_monolithic:
                bus.publish(
                    sender=self.TRUE_MONOLITHIC_POLICY_NAME,
                    topic="action.selection",
                    payload={
                        "policy_mode": policy_mode,
                        "direct_policy_logits": total_logits.round(6).tolist(),
                        "policy": policy.round(6).tolist(),
                        "selected_action": ACTIONS[motor_action_idx],
                        "executed_action": ACTIONS[action_idx],
                        "value_estimate": round(float(value), 6),
                    },
                )
            else:
                arbitration_payload = arbitration.to_payload() if arbitration is not None else {}
                bus.publish(
                    sender="action_center",
                    topic="action.selection",
                    payload={
                        "policy_mode": policy_mode,
                        "proposal_sum_logits": proposal_sum.round(6).tolist(),
                        "action_center_correction_logits": action_center_correction_logits.round(6).tolist(),
                        "action_center_logits": action_center_logits.round(6).tolist(),
                        "action_center_policy": action_center_policy.round(6).tolist(),
                        "selected_intent": ACTIONS[action_intent_idx],
                        "selected_intent_without_reflex": ACTIONS[action_intent_without_reflex_idx],
                        "value_estimate": round(float(value), 6),
                        **arbitration_payload,
                    },
                )
                bus.publish(
                    sender="motor_cortex",
                    topic="action.execution",
                    payload={
                        "policy_mode": policy_mode,
                        "motor_correction_logits": motor_correction_logits.round(6).tolist(),
                        "total_logits_without_reflex": total_logits_without_reflex.round(6).tolist(),
                        "total_logits": total_logits.round(6).tolist(),
                        "policy": policy.round(6).tolist(),
                        "selected_intent": ACTIONS[action_intent_idx],
                        "selected_action": ACTIONS[motor_action_idx],
                        "executed_action": ACTIONS[action_idx],
                        "selected_action_without_reflex": ACTIONS[action_without_reflex_idx],
                        "motor_override": bool(motor_override),
                        "final_reflex_override": bool(final_reflex_override),
                        "orientation_alignment": round(float(orientation_alignment), 6),
                        "terrain_difficulty": round(float(terrain_difficulty), 6),
                        "momentum": round(float(momentum), 6),
                        "execution_difficulty": round(float(execution_difficulty), 6),
                        "execution_slip_occurred": False,
                        "slip_reason": "none",
                    },
                )

        step_observation: Dict[str, np.ndarray] = {}
        if (
            policy_mode == "normal"
            and self.config.is_modular
            and self.config.uses_counterfactual_credit
        ):
            brain_observation_keys = {
                spec.observation_key
                for spec in MODULE_INTERFACES
            }
            brain_observation_keys.update(
                {
                    ACTION_CONTEXT_INTERFACE.observation_key,
                    MOTOR_CONTEXT_INTERFACE.observation_key,
                }
            )
            step_observation = {
                key: np.asarray(observation[key], dtype=float).copy()
                for key in brain_observation_keys
                if key in observation
            }

        return BrainStep(
            module_results=module_results,
            action_center_logits=action_center_logits,
            action_center_policy=action_center_policy,
            motor_correction_logits=motor_correction_logits,
            observation=step_observation,
            total_logits_without_reflex=total_logits_without_reflex,
            total_logits=total_logits,
            policy=policy,
            value=float(value),
            action_intent_idx=action_intent_idx,
            motor_action_idx=motor_action_idx,
            action_idx=action_idx,
            orientation_alignment=orientation_alignment,
            terrain_difficulty=terrain_difficulty,
            momentum=momentum,
            execution_difficulty=execution_difficulty,
            execution_slip_occurred=False,
            motor_slip_occurred=False,
            motor_noise_applied=False,
            slip_reason="none",
            motor_override=bool(motor_override),
            final_reflex_override=bool(final_reflex_override),
            action_center_input=action_center_input,
            motor_input=motor_input,
            policy_mode=policy_mode,
            arbitration_decision=arbitration,
        )

    def estimate_value(self, observation: Dict[str, np.ndarray]) -> float:
        """
        Estimate the action-center state value for a single observation.
        
        Builds proposals, applies reflex and arbitration gating as used at inference time, and returns the scalar value produced by the action-center. If the brain contains recurrent modules, their hidden states are snapshot and restored so this call does not mutate recurrent state.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Observation arrays keyed by interface names used to produce proposals and construct the action-center input.
        
        Returns:
            float: Scalar value estimate for the provided observation.
        """
        if self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            _, value = self.true_monolithic_policy.forward(
                monolithic_observation,
                store_cache=False,
            )
            return float(value)
        hidden_state_snapshot: Dict[str, np.ndarray] | None = None
        if self.module_bank is not None and self.module_bank.has_recurrent_modules:
            hidden_state_snapshot = self.module_bank.snapshot_hidden_states()
        try:
            module_results = self._proposal_results(
                observation,
                store_cache=False,
                training=False,
            )
            apply_reflex_path(
                module_results,
                ablation_config=self.config,
                operational_profile=self.operational_profile,
                interface_registry=self._interface_registry(),
                current_reflex_scale=self.current_reflex_scale,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )
            arbitration = self._compute_arbitration(
                module_results,
                observation,
                training=False,
                store_cache=False,
            )
            apply_priority_gating(
                module_results,
                arbitration,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )
            action_input = self._build_action_input(module_results, observation)
            if self.action_center is None:
                raise RuntimeError("Action center unavailable for the configured architecture.")
            _, value = self.action_center.forward(action_input, store_cache=False)
            return float(value)
        finally:
            if hidden_state_snapshot is not None and self.module_bank is not None:
                self.module_bank.restore_hidden_states(hidden_state_snapshot)

    def _proposal_stage_names(self) -> List[str]:
        """
        Return the ordered proposal sources that feed the action-center input.
        """
        if self.module_bank is not None:
            return [spec.name for spec in self.module_bank.enabled_specs]
        if self.true_monolithic_policy is not None:
            return [self.TRUE_MONOLITHIC_POLICY_NAME]
        return [self.MONOLITHIC_POLICY_NAME]

    def _architecture_signature(self) -> dict[str, object]:
        """
        Compute the runtime architecture signature for the active proposal backend.
        
        Returns:
            signature (dict): A mapping describing architecture identifiers and configuration used for compatibility/fingerprinting (includes proposal backend name and order, whether learned arbitration is enabled, and arbitration network input/hidden dims and regularization weight).
        """
        arbitration_input_dim = (
            self.arbitration_network.input_dim
            if self.arbitration_network is not None
            else 0
        )
        arbitration_hidden_dim = (
            self.arbitration_network.hidden_dim
            if self.arbitration_network is not None
            else 0
        )
        return architecture_signature(
            proposal_backend=self.config.architecture,
            proposal_order=self._proposal_stage_names(),
            learned_arbitration=(
                self.config.use_learned_arbitration and self.arbitration_network is not None
            ),
            arbitration_input_dim=arbitration_input_dim,
            arbitration_hidden_dim=arbitration_hidden_dim,
            arbitration_regularization_weight=self.arbitration_regularization_weight,
            capacity_profile_name=self.config.capacity_profile_name,
            module_hidden_dims=self.config.module_hidden_dims,
            integration_hidden_dim=self.config.integration_hidden_dim,
            monolithic_hidden_dim=self.config.monolithic_hidden_dim,
            capacity_profile=self.config.capacity_profile.to_summary(),
        )

    def _interface_registry(self) -> dict[str, object]:
        """
        Retrieve the runtime-governed interface registry used by this brain.
        
        Returns:
            dict[str, object]: Mapping of interface names to their interface objects as provided by the active runtime registry.
        """
        return interface_registry()

    def _architecture_fingerprint(self) -> str:
        """
        Get the stable fingerprint of the brain's runtime architecture signature.
        
        Returns:
            fingerprint (str): String representation of the architecture signature's `fingerprint` field.
        """
        return str(self._architecture_signature()["fingerprint"])

    def parameter_norms(self) -> Dict[str, float]:
        """
        Compute the L2 norm of parameters for each trainable network component.
        
        Returns:
            Mapping from component name to its L2 parameter norm for each active
            trainable network in the configured topology.
        """
        norms: Dict[str, float] = {}
        if self.module_bank is not None:
            norms.update(self.module_bank.parameter_norms())
        if self.monolithic_policy is not None:
            norms[self.MONOLITHIC_POLICY_NAME] = self.monolithic_policy.parameter_norm()
        if self.true_monolithic_policy is not None:
            norms[self.TRUE_MONOLITHIC_POLICY_NAME] = (
                self.true_monolithic_policy.parameter_norm()
            )
        if self.arbitration_network is not None:
            norms[self.ARBITRATION_NETWORK_NAME] = self.arbitration_network.parameter_norm()
        if self.action_center is not None:
            norms["action_center"] = self.action_center.parameter_norm()
        if self.motor_cortex is not None:
            norms["motor_cortex"] = self.motor_cortex.parameter_norm()
        return norms

    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable parameters for each trainable network component.

        Returns:
            Mapping from component name to trainable parameter count for each
            active trainable network in the configured topology.
        """
        counts: Dict[str, int] = {}
        if self.module_bank is not None:
            counts.update(self.module_bank.parameter_counts())
        if self.monolithic_policy is not None:
            counts[self.MONOLITHIC_POLICY_NAME] = self.monolithic_policy.count_parameters()
        if self.true_monolithic_policy is not None:
            counts[self.TRUE_MONOLITHIC_POLICY_NAME] = (
                self.true_monolithic_policy.count_parameters()
            )
        if self.arbitration_network is not None:
            counts[self.ARBITRATION_NETWORK_NAME] = self.arbitration_network.count_parameters()
        if self.action_center is not None:
            counts["action_center"] = self.action_center.count_parameters()
        if self.motor_cortex is not None:
            counts["motor_cortex"] = self.motor_cortex.count_parameters()
        return counts

    def _module_names(self) -> List[str]:
        """
        List module names present in the brain for inspection.
        
        When the brain is modular, returns the module spec names in their
        configured order followed by the active downstream controllers.
        Monolithic variants return only the networks present in that topology.
        
        Returns:
            names (List[str]): Ordered list of module names and the two controller component names.
        """
        if self.module_bank is not None:
            return [
                spec.name for spec in self.module_bank.enabled_specs
            ] + [self.ARBITRATION_NETWORK_NAME, "action_center", "motor_cortex"]
        if self.true_monolithic_policy is not None:
            return [self.TRUE_MONOLITHIC_POLICY_NAME]
        return [self.MONOLITHIC_POLICY_NAME, self.ARBITRATION_NETWORK_NAME, "action_center", "motor_cortex"]
