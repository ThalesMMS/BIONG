from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Sequence

import numpy as np

from ..ablations import BrainAblationConfig, default_brain_config
from ..direct_policy_affordances import AFFORDANCE_SHELTER_ROLE_NAMES
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
from ..b_series import (
    B_CURRENT_BRIDGE_EFFECTIVE_LEVEL,
    B_CURRENT_BRIDGE_SELECTION_SOURCE,
    B_SERIES_POLICY_NAME,
    B_SEMANTIC_ACTIONS,
    B_SEMANTIC_ACTION_TO_INDEX,
    bridge_b_semantic_action,
)
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
    RecurrentEventAttentionTrueMonolithicNetwork,
    RecurrentOptionAffordanceTrueMonolithicNetwork,
    RecurrentOptionTrueMonolithicNetwork,
    RecurrentProposalNetwork,
    one_hot,
    softmax,
)
from ..noise import _compute_execution_difficulty_core
from ..operational_profiles import OperationalProfile, runtime_operational_profile
from ..phase import PHASE_LABELS
from ..reflexes import (
    _apply_reflex_path as apply_reflex_path,
    _direction_action as direction_action,
    _module_reflex_decision as module_reflex_decision,
)
from ..world import ACTIONS

from .types import BrainStep

TRUE_MONOLITHIC_NO_FOOD_DIRECTION_VARIANTS = {
    "true_monolithic_executive_option_guarded_policy",
}

TRUE_MONOLITHIC_DIRECTION_BIAS_LOGIT = 3.0
MODULAR_DIRECTION_BIAS_LOGIT = 3.0
TRUE_MONOLITHIC_THREAT_ESCAPE_BIAS_LOGIT = 6.0
TRUE_MONOLITHIC_THREAT_ESCAPE_SMELL_THRESHOLD = 0.43
TRUE_MONOLITHIC_SLEEP_REST_BIAS_LOGIT = 6.0


class BrainRuntimeMixin:
    @staticmethod
    def _b_series_float(mapping: MappingProxyType | Dict[str, float], key: str) -> float:
        try:
            value = float(mapping.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    def _b0_current_simple_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int]:
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        hunger = self._b_series_float(hunger_obs, "hunger")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        health = self._b_series_float(sleep_obs, "health")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        on_shelter = self._b_series_float(sleep_obs, "on_shelter") > 0.5 or bool(
            meta.get("on_shelter", False)
        )
        night = self._b_series_float(sleep_obs, "night") > 0.5 or bool(
            meta.get("night", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        shelter_role_level = max(
            self._b_series_float(sleep_obs, "shelter_role_level"),
            self._b_series_float(meta, "shelter_role_level"),
        )
        food_memory_signal = (
            1.0 - self._b_series_float(hunger_obs, "food_memory_age")
            if (
                abs(float(hunger_obs.get("food_memory_dx", 0.0)))
                + abs(float(hunger_obs.get("food_memory_dy", 0.0)))
            )
            > 0.05
            else 0.0
        )
        food_signal = max(
            self._b_series_float(hunger_obs, "food_visible"),
            self._b_series_float(hunger_obs, "food_certainty"),
            self._b_series_float(hunger_obs, "food_smell_strength"),
            self._b_series_float(hunger_obs, "food_trace_strength"),
            food_memory_signal,
        )
        acute_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
        )
        threat_pressure = max(
            acute_threat,
            self._b_series_float(threat_obs, "predator_smell_strength"),
        )

        if on_food and hunger >= 0.10:
            semantic_action = "EAT"
            reason = "b0_current_eat_on_food"
        elif on_shelter:
            rest_pressure = bool(night or fatigue >= 0.25 or sleep_debt >= 0.25)
            if health <= 0.65 and hunger < 0.55:
                if shelter_role_level < 0.75:
                    semantic_action = "MOVE_TO_SHELTER"
                    reason = "b0_current_low_health_deepen"
                elif rest_pressure:
                    semantic_action = "SLEEP"
                    reason = "b0_current_low_health_rest"
                else:
                    semantic_action = "STAY"
                    reason = "b0_current_low_health_hold"
            elif threat_pressure >= 0.55 and hunger < 0.48:
                if shelter_role_level < 0.75:
                    semantic_action = "MOVE_TO_SHELTER"
                    reason = "b0_current_threat_hold_deepen"
                elif rest_pressure:
                    semantic_action = "SLEEP"
                    reason = "b0_current_threat_hold_rest"
                else:
                    semantic_action = "STAY"
                    reason = "b0_current_threat_hold_shelter"
            elif rest_pressure and hunger < 0.55:
                if shelter_role_level < 0.75:
                    semantic_action = "MOVE_TO_SHELTER"
                    reason = "b0_current_deepen_before_rest"
                else:
                    semantic_action = "SLEEP"
                    reason = "b0_current_rest_in_shelter"
            elif hunger >= 0.50 or (food_signal >= 0.35 and not rest_pressure):
                semantic_action = "MOVE_TO_FOOD"
                reason = "b0_current_forage_from_shelter"
            else:
                semantic_action = "STAY"
                reason = "b0_current_shelter_hold"
        elif (
            (hunger < 0.40 and (night or fatigue >= 0.25 or sleep_debt >= 0.25))
            or acute_threat >= 0.85
            or (threat_pressure >= 0.55 and hunger < 0.55)
            or (health <= 0.65 and hunger < 0.55)
            or health <= 0.35
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b0_current_recover_return"
        elif hunger >= 0.45 or food_signal >= 0.15:
            semantic_action = "MOVE_TO_FOOD"
            reason = "b0_current_forage"
        elif night or fatigue >= 0.52 or sleep_debt >= 0.52:
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b0_current_rest_return"
        else:
            semantic_action = "EXPLORE"
            reason = "b0_current_explore"

        return (
            semantic_action,
            B_CURRENT_BRIDGE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
        )

    @staticmethod
    def _network_forward_macs(network: object) -> int:
        if isinstance(network, RecurrentProposalNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
            )
        if isinstance(network, RecurrentEventAttentionTrueMonolithicNetwork):
            recurrent_input_dim = (
                network.input_dim + network.event_context_dim
            )
            event_raw_dim = (
                network.event_embedding_dim + network.event_feature_dim + 1
            )
            return int(
                recurrent_input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.event_context_dim * (network.input_dim + network.hidden_dim)
                + 2 * (network.event_context_dim * event_raw_dim)
            )
        if isinstance(network, RecurrentOptionAffordanceTrueMonolithicNetwork):
            recurrent_input_dim = (
                network.input_dim + network.event_context_dim + network.option_dim
            )
            event_raw_dim = (
                network.event_embedding_dim + network.event_feature_dim + 1
            )
            return int(
                recurrent_input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.hidden_dim * network.option_dim
                + network.option_dim * network.output_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.hidden_dim * network.output_dim * network.affordance_role_dim
                + network.output_dim * network.affordance_role_dim
                + network.event_context_dim * (network.input_dim + network.hidden_dim)
                + 2 * (network.event_context_dim * event_raw_dim)
            )
        if isinstance(network, RecurrentOptionTrueMonolithicNetwork):
            recurrent_input_dim = (
                network.input_dim + network.event_context_dim + network.option_dim
            )
            event_raw_dim = (
                network.event_embedding_dim + network.event_feature_dim + 1
            )
            return int(
                recurrent_input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.hidden_dim * network.option_dim
                + network.option_dim * network.output_dim
                + network.event_context_dim * (network.input_dim + network.hidden_dim)
                + 2 * (network.event_context_dim * event_raw_dim)
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
        if getattr(self, "b_series_policy", None) is not None:
            per_network[B_SERIES_POLICY_NAME] = self._network_forward_macs(
                self.b_series_policy
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
            gate_entropy=0.0,
            dominance_rate=1.0,
            effective_proposer_count=1.0,
            module_counts={module_name: 1},
            module_agreement_rate=1.0,
            module_disagreement_rate=0.0,
        )

    def _food_direction_bias_action(
        self,
        observation: Dict[str, np.ndarray],
    ) -> str | None:
        """
        Return a locomotion action that follows the strongest available food cue.

        Preference order matches the modular inference path: visible food,
        then food trace, then fresh food memory, then smell.
        """
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
        action = direction_action(food_dx, food_dy)
        if action == "STAY":
            return None
        return action

    def _threat_escape_bias_action(
        self,
        observation: Dict[str, np.ndarray],
    ) -> str | None:
        """
        Return a locomotion action that biases the agent back toward shelter under threat.

        For the direct-control baseline this provides a lightweight escape prior:
        when recent contact/pain, a predator trace, or fresh predator memory is
        active, prefer the freshest shelter-memory direction and otherwise move
        away from the predator cue.
        """
        threat_obs = self._bound_observation("threat_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        sleep_rest_action = self._sleep_rest_bias_action(observation)
        smell_signal = threat_obs["predator_smell_strength"]
        predator_signal = max(
            threat_obs["predator_trace_strength"],
            max(0.0, 1.0 - threat_obs["predator_memory_age"]),
            threat_obs["predator_visible"] * threat_obs["predator_certainty"],
            threat_obs["recent_contact"],
            threat_obs["recent_pain"],
        )
        if predator_signal <= 0.0 and smell_signal < TRUE_MONOLITHIC_THREAT_ESCAPE_SMELL_THRESHOLD:
            return None
        if sleep_rest_action == "STAY" and predator_signal <= 0.0:
            return None
        if sleep_obs["shelter_memory_age"] < 1.0:
            shelter_action = direction_action(
                sleep_obs["shelter_memory_dx"],
                sleep_obs["shelter_memory_dy"],
            )
            if shelter_action != "STAY":
                return shelter_action
        predator_dx = 0.0
        predator_dy = 0.0
        if threat_obs["predator_trace_strength"] > 0.0:
            predator_dx = threat_obs["predator_trace_dx"]
            predator_dy = threat_obs["predator_trace_dy"]
        elif threat_obs["predator_memory_age"] < 1.0:
            predator_dx = threat_obs["predator_memory_dx"]
            predator_dy = threat_obs["predator_memory_dy"]
        elif threat_obs["predator_visible"] > 0.0 and threat_obs["predator_certainty"] > 0.0:
            predator_dx = threat_obs["predator_dx"]
            predator_dy = threat_obs["predator_dy"]
        elif smell_signal >= TRUE_MONOLITHIC_THREAT_ESCAPE_SMELL_THRESHOLD:
            predator_dx = threat_obs["predator_smell_dx"]
            predator_dy = threat_obs["predator_smell_dy"]
        escape_action = direction_action(-predator_dx, -predator_dy)
        if escape_action == "STAY":
            return None
        return escape_action

    def _sleep_rest_bias_action(
        self,
        observation: Dict[str, np.ndarray],
    ) -> str | None:
        """
        Return STAY when the spider is already sheltered and should rest in place.

        This mirrors the existing sleep-center reflex thresholds so the
        true-monolithic baseline can hold shelter and accumulate rest instead of
        pacing inside the burrow after a successful return.
        """
        sleep_obs = self._bound_observation("sleep_center", observation)
        thresholds = self.operational_profile.brain_reflex_thresholds["sleep_center"]
        if (
            sleep_obs["on_shelter"] > thresholds["on_shelter"]
            and sleep_obs["sleep_phase_level"] > thresholds["sleep_phase"]
            and sleep_obs["hunger"] < thresholds["rest_hunger"]
        ):
            return "STAY"
        if (
            sleep_obs["on_shelter"] > thresholds["on_shelter"]
            and sleep_obs["shelter_role_level"] > thresholds["deep_shelter_level"]
            and sleep_obs["rest_streak_norm"] > thresholds["rest_streak"]
            and sleep_obs["hunger"] < thresholds["rest_hunger"]
        ):
            return "STAY"
        if (
            sleep_obs["on_shelter"] > thresholds["on_shelter"]
            and (
                sleep_obs["night"] > thresholds["on_shelter"]
                or sleep_obs["fatigue"] > thresholds["fatigue_to_hold"]
                or sleep_obs["sleep_debt"] > thresholds["sleep_debt_to_hold"]
            )
            and sleep_obs["hunger"] < thresholds["rest_hunger"]
        ):
            return "STAY"
        return None

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

    def _act_with_training(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None,
        *,
        sample: bool,
        policy_mode: str,
        training: bool,
    ) -> BrainStep:
        try:
            return self.act(
                observation,
                bus,
                sample=sample,
                policy_mode=policy_mode,
                training=training,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'training'" not in str(exc):
                raise
            return self.act(
                observation,
                bus,
                sample=sample,
                policy_mode=policy_mode,
            )

    def act_exploration(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        policy_mode: str = "normal",
    ) -> BrainStep:
        return self._act_with_training(
            observation, bus, sample=True, policy_mode=policy_mode, training=False
        )

    def act_inference(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = False,
        policy_mode: str = "normal",
    ) -> BrainStep:
        return self._act_with_training(
            observation, bus, sample=sample, policy_mode=policy_mode, training=False
        )

    def act_train(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = True,
        policy_mode: str = "normal",
    ) -> BrainStep:
        return self._act_with_training(
            observation, bus, sample=sample, policy_mode=policy_mode, training=True
        )

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

    def _true_monolithic_allows_food_direction_bias(self) -> bool:
        return self.config.name not in TRUE_MONOLITHIC_NO_FOOD_DIRECTION_VARIANTS

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
        direct_policy_trace_payload: Dict[str, object] = {}
        phase_logits = np.zeros(0, dtype=float)
        phase_prediction: str | None = None
        phase_prediction_confidence = 0.0
        event_attention_top_type: str | None = None
        event_attention_top_age = -1
        event_attention_entropy = 0.0
        selected_option: str | None = None
        option_age = -1
        option_termination_reason = "none"
        option_logits = np.zeros(0, dtype=float)
        option_leaf_logits = np.zeros(0, dtype=float)
        option_owned_action: str | None = None
        safety_mask_applied = False
        safety_masked_actions: list[str] = []
        external_override_count = 0
        affordance_blocked_logits = np.zeros(0, dtype=float)
        affordance_role_logits = np.zeros(0, dtype=float)
        geometry_logits = np.zeros(0, dtype=float)
        shelter_column_logits = np.zeros(0, dtype=float)
        shelter_position_logits = np.zeros(0, dtype=float)
        transition_prediction_logits = np.zeros(0, dtype=float)
        transition_rollout_prediction_logits = np.zeros(0, dtype=float)
        if self.config.is_b_series:
            if self.b_series_policy is None:
                raise RuntimeError(
                    "B-series policy unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            semantic_logits, value = self.b_series_policy.forward(
                monolithic_observation,
                store_cache=store_cache,
            )
            semantic_policy = softmax(semantic_logits)
            if sample:
                learned_semantic_action_idx = int(
                    self.rng.choice(len(B_SEMANTIC_ACTIONS), p=semantic_policy)
                )
            else:
                learned_semantic_action_idx = int(np.argmax(semantic_policy))
            learned_semantic_action = B_SEMANTIC_ACTIONS[learned_semantic_action_idx]
            semantic_action = learned_semantic_action
            semantic_action_source = "network_policy"
            semantic_action_reason = "network_argmax_or_sample"
            semantic_override_count = 0
            b_effective_level = "B0"
            if (
                int(getattr(self.config, "b_level", 0)) == 0
                and str(getattr(self.config, "b_mode", "")) == "current_bridge"
            ):
                (
                    semantic_action,
                    semantic_action_source,
                    semantic_action_reason,
                    semantic_override_count,
                ) = self._b0_current_simple_semantic_action(
                    observation,
                    learned_semantic_action=learned_semantic_action,
                )
                b_effective_level = B_CURRENT_BRIDGE_EFFECTIVE_LEVEL
            semantic_action_idx = int(B_SEMANTIC_ACTION_TO_INDEX[semantic_action])
            bridge_decision = bridge_b_semantic_action(
                semantic_action,
                observation,
                rng=self.rng,
                sample=bool(sample),
            )
            action_idx = int(bridge_decision.primitive_action_idx)
            motor_action_idx = action_idx
            action_intent_idx = action_idx
            action_intent_without_reflex_idx = action_idx
            action_without_reflex_idx = action_idx
            primitive_logits = np.zeros(self.action_dim, dtype=float)
            primitive_logits[action_idx] = 6.0
            total_logits_without_reflex = primitive_logits.copy()
            total_logits = primitive_logits.copy()
            action_center_logits = primitive_logits.copy()
            action_center_policy = softmax(action_center_logits)
            policy = softmax(total_logits)
            proposal_sum = total_logits.copy()
            module_results = [
                ModuleResult(
                    interface=None,
                    name=B_SERIES_POLICY_NAME,
                    observation_key=B_SERIES_POLICY_NAME,
                    observation=monolithic_observation.copy(),
                    logits=primitive_logits.copy(),
                    probs=policy.copy(),
                    active=True,
                    reflex=None,
                    neural_logits=primitive_logits.copy(),
                    reflex_delta_logits=np.zeros_like(primitive_logits),
                    post_reflex_logits=primitive_logits.copy(),
                )
            ]
            module_results[0].valence_role = "semantic_bridge"
            module_results[0].gate_weight = 1.0
            module_results[0].gated_logits = primitive_logits.copy()
            module_results[0].contribution_share = 1.0
            module_results[0].intent_before_gating = bridge_decision.semantic_action
            module_results[0].intent_after_gating = bridge_decision.primitive_action
            arbitration = self._true_monolithic_arbitration_decision(
                module_name=B_SERIES_POLICY_NAME,
                action_idx=action_idx,
            )
            direct_policy_trace_payload = {
                "b_level": int(self.config.b_level),
                "b_effective_level": b_effective_level,
                "b_mode": str(self.config.b_mode),
                "semantic_action": semantic_action,
                "semantic_action_idx": int(semantic_action_idx),
                "learned_semantic_action": learned_semantic_action,
                "learned_semantic_action_idx": int(learned_semantic_action_idx),
                "semantic_action_source": semantic_action_source,
                "semantic_action_reason": semantic_action_reason,
                "semantic_override_count": int(semantic_override_count),
                "semantic_logits": np.asarray(semantic_logits, dtype=float)
                .round(6)
                .tolist(),
                "semantic_policy": np.asarray(semantic_policy, dtype=float)
                .round(6)
                .tolist(),
                "bridge_primitive_action": bridge_decision.primitive_action,
                "bridge_reason": bridge_decision.reason,
                "blocked_mask": dict(bridge_decision.blocked_mask),
                "food_delta_used": round(float(bridge_decision.food_delta_used), 6),
                "shelter_delta_used": round(
                    float(bridge_decision.shelter_delta_used),
                    6,
                ),
                "external_override_count": int(
                    bridge_decision.external_override_count
                ),
            }
            external_override_count = int(bridge_decision.external_override_count)
            motor_override = False
            final_reflex_override = False
        elif self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
            )
            monolithic_observation = self._build_monolithic_observation(observation)
            if hasattr(self.true_monolithic_policy, "set_runtime_observation_meta"):
                self.true_monolithic_policy.set_runtime_observation_meta(
                    observation.get("meta", {})
                )
            hidden_before = self.snapshot_direct_policy_hidden_state()
            direct_forward = self.true_monolithic_policy.forward(
                monolithic_observation,
                store_cache=store_cache,
            )
            if len(direct_forward) == 4:
                policy_logits, value, option_logits_raw, phase_logits_raw = direct_forward
                option_logits = np.asarray(option_logits_raw, dtype=float).copy()
            elif len(direct_forward) == 3:
                policy_logits, value, aux_logits_raw = direct_forward
                phase_logits_raw = (
                    aux_logits_raw if self.config.direct_policy_phase_head else None
                )
                if self.config.direct_policy_option_head:
                    option_logits = np.asarray(aux_logits_raw, dtype=float).copy()
            else:
                policy_logits, value = direct_forward
                phase_logits_raw = None
            if phase_logits_raw is not None:
                phase_logits = np.asarray(phase_logits_raw, dtype=float).copy()
                phase_probs = softmax(phase_logits)
                phase_prediction_idx = int(np.argmax(phase_probs))
                phase_prediction = PHASE_LABELS[phase_prediction_idx]
                phase_prediction_confidence = float(phase_probs[phase_prediction_idx])
            hidden_after = self.snapshot_direct_policy_hidden_state()
            if hidden_after is not None:
                hidden_before_array = (
                    hidden_before
                    if hidden_before is not None
                    else np.zeros_like(hidden_after, dtype=float)
                )
                direct_policy_trace_payload = {
                    "recurrent_hidden_norm": round(float(np.linalg.norm(hidden_after)), 6),
                    "recurrent_hidden_delta_norm": round(
                        float(np.linalg.norm(hidden_after - hidden_before_array)),
                        6,
                    ),
                    "hidden_reset_event": bool(self._direct_policy_hidden_reset_pending),
                    "architecture_metadata": {
                        "direct_policy_recurrent": bool(self.config.direct_policy_recurrent),
                        "direct_policy_hidden_dims": list(self.config.direct_policy_hidden_dims),
                        "direct_policy_phase_head": bool(self.config.direct_policy_phase_head),
                        "direct_policy_event_attention": bool(
                            self.config.direct_policy_event_attention
                        ),
                        "direct_policy_event_buffer_size": int(
                            self.config.direct_policy_event_buffer_size
                        ),
                        "direct_policy_option_head": bool(
                            self.config.direct_policy_option_head
                        ),
                        "direct_policy_owned_option_controller": bool(
                            self.config.direct_policy_owned_option_controller
                        ),
                        "direct_policy_option_ttl": int(
                            self.config.direct_policy_option_ttl
                        ),
                        "direct_policy_affordance_head": bool(
                            self.config.direct_policy_affordance_head
                        ),
                        "direct_policy_affordance_feedback": bool(
                            self.config.direct_policy_affordance_feedback
                        ),
                        "direct_policy_geometry_head": bool(
                            self.config.direct_policy_geometry_head
                        ),
                        "direct_policy_shelter_column_head": bool(
                            self.config.direct_policy_shelter_column_head
                        ),
                        "direct_policy_shelter_position_head": bool(
                            self.config.direct_policy_shelter_position_head
                        ),
                        "direct_policy_local_affordance_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_affordance_inputs",
                                False,
                            )
                        ),
                        "direct_policy_local_spatial_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_spatial_inputs",
                                False,
                            )
                        ),
                        "direct_policy_local_transition_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_transition_inputs",
                                False,
                            )
                        ),
                        "direct_policy_local_transition_rollout_inputs": bool(
                            getattr(
                                self.config,
                                "direct_policy_local_transition_rollout_inputs",
                                False,
                            )
                        ),
                        "direct_policy_transition_prediction_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_prediction_head",
                                False,
                            )
                        ),
                        "direct_policy_transition_prediction_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_prediction_feedback",
                                False,
                            )
                        ),
                        "direct_policy_transition_rollout_prediction_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_rollout_prediction_head",
                                False,
                            )
                        ),
                        "direct_policy_transition_rollout_prediction_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_transition_rollout_prediction_feedback",
                                False,
                            )
                        ),
                        "direct_policy_handoff_teacher": bool(
                            self.config.direct_policy_handoff_teacher
                        ),
                        "direct_policy_handoff_option_teacher": bool(
                            getattr(
                                self.config,
                                "direct_policy_handoff_option_teacher",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_action_teacher": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_action_teacher",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_release_sequence_teacher": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_release_sequence_teacher",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_release_sequence_replay_boost": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_release_sequence_replay_boost",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_release_sequence_distill": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_release_sequence_distill",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_sequence_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_sequence_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_family_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_family_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_handoff_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_handoff_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_trajectory_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_trajectory_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_cycle_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_cycle_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_trace_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_trace_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_rollout_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_rollout_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_frontier_teacher_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_frontier_teacher_distillation",
                                False,
                            )
                        ),
                        "direct_policy_post_rest_probe_replayable_teacher_distillation": bool(
                            getattr(
                                self.config,
                                "direct_policy_post_rest_probe_replayable_teacher_distillation",
                                False,
                            )
                        ),
                        "direct_policy_continuation_replay_passes": int(
                            getattr(
                                self.config,
                                "direct_policy_continuation_replay_passes",
                                0,
                            )
                        ),
                        "direct_policy_continuation_replay_lr_scale": float(
                            getattr(
                                self.config,
                                "direct_policy_continuation_replay_lr_scale",
                                0.0,
                            )
                        ),
                        "direct_policy_continuation_margin_weight": float(
                            getattr(
                                self.config,
                                "direct_policy_continuation_margin_weight",
                                0.0,
                            )
                        ),
                        "direct_policy_phase_option_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_phase_option_feedback",
                                False,
                            )
                        ),
                        "direct_policy_option_transition_feedback": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_transition_feedback",
                                False,
                            )
                        ),
                        "direct_policy_option_termination_cooldown": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_termination_cooldown",
                                False,
                            )
                        ),
                        "direct_policy_option_action_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_head",
                                False,
                            )
                        ),
                        "direct_policy_option_decoder_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_decoder_state",
                                False,
                            )
                        ),
                        "direct_policy_option_recurrent_dynamics": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_recurrent_dynamics",
                                False,
                            )
                        ),
                        "direct_policy_option_sequence_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_sequence_head",
                                False,
                            )
                        ),
                        "direct_policy_option_decoder_recurrent_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_decoder_recurrent_state",
                                False,
                            )
                        ),
                        "direct_policy_option_action_transition_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_transition_state",
                                False,
                            )
                        ),
                        "direct_policy_option_action_controller_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_controller_state",
                                False,
                            )
                        ),
                        "direct_policy_option_action_token_decoder": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_token_decoder",
                                False,
                            )
                        ),
                        "direct_policy_option_action_recurrent_core": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_recurrent_core",
                                False,
                            )
                        ),
                        "direct_policy_option_action_separate_recurrent_head": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_separate_recurrent_head",
                                False,
                            )
                        ),
                        "direct_policy_option_action_separate_policy_path": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_separate_policy_path",
                                False,
                            )
                        ),
                        "direct_policy_option_action_separate_backbone": bool(
                            getattr(
                                self.config,
                                "direct_policy_option_action_separate_backbone",
                                False,
                            )
                        ),
                        "direct_policy_executive_physiology_option_gating": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_physiology_option_gating",
                                False,
                            )
                        ),
                        "direct_policy_executive_affordance_action_gating": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_affordance_action_gating",
                                False,
                            )
                        ),
                        "direct_policy_executive_option_action_masking": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_option_action_masking",
                                False,
                            )
                        ),
                        "direct_policy_executive_event_release_latching": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_event_release_latching",
                                False,
                            )
                        ),
                        "direct_policy_executive_event_release_action_commitment": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_event_release_action_commitment",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_phase_state": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_phase_state",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_exit_contract": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_exit_contract",
                                False,
                            )
                        ),
                        "direct_policy_executive_release_substate_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_release_substate_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_continuation": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_continuation",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_guidance": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_guidance",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_commitment": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_commitment",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_food_heading_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_food_heading_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_smell_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_smell_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_corridor_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_corridor_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_exit_corridor_affordance_progression": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_exit_corridor_affordance_progression",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_food_return": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_food_return",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_food_vector_return": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_food_vector_return",
                                False,
                            )
                        ),
                        "direct_policy_executive_post_food_path_return": bool(
                            getattr(
                                self.config,
                                "direct_policy_executive_post_food_path_return",
                                False,
                            )
                        ),
                    },
                }
            if phase_prediction is not None:
                direct_policy_trace_payload.update(
                    {
                        "phase_prediction": phase_prediction,
                        "phase_prediction_confidence": round(
                            float(phase_prediction_confidence),
                            6,
                        ),
                    }
                )
            attention_summary = getattr(
                self.true_monolithic_policy,
                "last_attention_summary",
                None,
            )
            if isinstance(attention_summary, dict):
                event_attention_top_type = attention_summary.get(
                    "event_attention_top_type"
                )
                event_attention_top_age = int(
                    attention_summary.get("event_attention_top_age", -1)
                )
                event_attention_entropy = float(
                    attention_summary.get("event_attention_entropy", 0.0)
                )
                if event_attention_top_type is not None:
                    direct_policy_trace_payload.update(
                        {
                            "event_attention_top_type": str(
                                event_attention_top_type
                            ),
                            "event_attention_top_age": int(
                                event_attention_top_age
                            ),
                            "event_attention_entropy": round(
                                float(event_attention_entropy),
                                6,
                            ),
                        }
                    )
            option_summary = getattr(self.true_monolithic_policy, "last_option_summary", None)
            if isinstance(option_summary, dict):
                selected_option_raw = option_summary.get("selected_option")
                if selected_option_raw is not None:
                    selected_option = str(selected_option_raw)
                    option_age = int(option_summary.get("option_age", -1))
                    option_termination_reason = str(
                        option_summary.get("option_termination_reason", "none")
                    )
                    option_logits = np.asarray(
                        option_summary.get("option_logits", []),
                        dtype=float,
                    ).copy()
                    direct_policy_trace_payload.update(
                        {
                            "selected_option": selected_option,
                            "option_age": int(option_age),
                            "option_termination_reason": option_termination_reason,
                            "option_logits": option_logits.round(6).tolist(),
                        }
                    )
                    option_leaf_logits = np.asarray(
                        option_summary.get("option_leaf_logits", []),
                        dtype=float,
                    ).copy()
                    option_owned_action_raw = option_summary.get("option_owned_action")
                    option_owned_action = (
                        None
                        if option_owned_action_raw is None
                        else str(option_owned_action_raw)
                    )
                    safety_mask_applied = bool(
                        option_summary.get("safety_mask_applied", False)
                    )
                    safety_masked_actions = [
                        str(action)
                        for action in option_summary.get(
                            "safety_masked_actions",
                            [],
                        )
                    ]
                    external_override_count = int(
                        option_summary.get("external_override_count", 0)
                    )
                    if option_leaf_logits.size > 0:
                        direct_policy_trace_payload["option_leaf_logits"] = (
                            option_leaf_logits.round(6).tolist()
                        )
                    if option_owned_action is not None:
                        direct_policy_trace_payload["option_owned_action"] = (
                            option_owned_action
                        )
                    direct_policy_trace_payload["safety_mask_applied"] = bool(
                        safety_mask_applied
                    )
                    direct_policy_trace_payload["safety_masked_actions"] = list(
                        safety_masked_actions
                    )
                    direct_policy_trace_payload["external_override_count"] = int(
                        external_override_count
                    )
            affordance_summary = getattr(
                self.true_monolithic_policy,
                "last_affordance_summary",
                None,
            )
            if isinstance(affordance_summary, dict):
                affordance_blocked_logits = np.asarray(
                    affordance_summary.get("blocked_logits", []),
                    dtype=float,
                ).copy()
                affordance_role_logits = np.asarray(
                    affordance_summary.get("role_logits", []),
                    dtype=float,
                ).copy()
                if affordance_blocked_logits.size > 0:
                    direct_policy_trace_payload["affordance_blocked_logits"] = (
                        affordance_blocked_logits.round(6).tolist()
                    )
                if affordance_role_logits.size > 0:
                    direct_policy_trace_payload["affordance_role_logits"] = (
                        affordance_role_logits.round(6).tolist()
                    )
                geometry_logits = np.asarray(
                    affordance_summary.get("geometry_logits", []),
                    dtype=float,
                ).copy()
                if geometry_logits.size > 0:
                    direct_policy_trace_payload["geometry_logits"] = (
                        geometry_logits.round(6).tolist()
                    )
                shelter_column_logits = np.asarray(
                    affordance_summary.get("shelter_column_logits", []),
                    dtype=float,
                ).copy()
                if shelter_column_logits.size > 0:
                    direct_policy_trace_payload["shelter_column_logits"] = (
                        shelter_column_logits.round(6).tolist()
                    )
                shelter_position_logits = np.asarray(
                    affordance_summary.get("shelter_position_logits", []),
                    dtype=float,
                ).copy()
                if shelter_position_logits.size > 0:
                    direct_policy_trace_payload["shelter_position_logits"] = (
                        shelter_position_logits.round(6).tolist()
                    )
                transition_prediction_logits = np.asarray(
                    affordance_summary.get("transition_prediction_logits", []),
                    dtype=float,
                ).copy()
                if transition_prediction_logits.size > 0:
                    direct_policy_trace_payload["transition_prediction_logits"] = (
                        transition_prediction_logits.round(6).tolist()
                    )
                transition_rollout_prediction_logits = np.asarray(
                    affordance_summary.get(
                        "transition_rollout_prediction_logits",
                        [],
                    ),
                    dtype=float,
                ).copy()
                if transition_rollout_prediction_logits.size > 0:
                    direct_policy_trace_payload[
                        "transition_rollout_prediction_logits"
                    ] = transition_rollout_prediction_logits.round(6).tolist()
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
            if (
                self.config.enable_food_direction_bias
                and not self.config.direct_policy_owned_option_controller
                and not training_mode
                and not sample
            ):
                bias_action = self._threat_escape_bias_action(observation)
                if bias_action is None:
                    bias_action = self._sleep_rest_bias_action(observation)
                if (
                    bias_action is None
                    and self._true_monolithic_allows_food_direction_bias()
                ):
                    bias_action = self._food_direction_bias_action(observation)
                if bias_action is not None:
                    total_logits = total_logits.copy()
                    threat_bias_action = self._threat_escape_bias_action(observation)
                    sleep_bias_action = self._sleep_rest_bias_action(observation)
                    if threat_bias_action is not None and bias_action == threat_bias_action:
                        bias_bonus = TRUE_MONOLITHIC_THREAT_ESCAPE_BIAS_LOGIT
                    elif sleep_bias_action is not None and bias_action == sleep_bias_action:
                        bias_bonus = TRUE_MONOLITHIC_SLEEP_REST_BIAS_LOGIT
                    else:
                        bias_bonus = TRUE_MONOLITHIC_DIRECTION_BIAS_LOGIT
                    total_logits[ACTION_TO_INDEX[bias_action]] += bias_bonus
                    policy = softmax(total_logits)
                    action_intent_idx = int(np.argmax(policy))
                    action_intent_without_reflex_idx = action_intent_idx
                    action_without_reflex_idx = action_intent_idx
                    motor_action_idx = action_intent_idx
                    action_center_logits = total_logits.copy()
                    action_center_policy = policy.copy()
                    proposal_sum = total_logits.copy()
                    arbitration = replace(
                        arbitration,
                        food_bias_applied=True,
                        food_bias_action=bias_action,
                        intent_before_gating_idx=int(np.argmax(total_logits_without_reflex)),
                        intent_after_gating_idx=action_intent_idx,
                    )
                    if hasattr(self.true_monolithic_policy, "record_external_override"):
                        self.true_monolithic_policy.record_external_override("final_bias")
                        external_override_count = int(
                            getattr(
                                self.true_monolithic_policy,
                                "external_override_count",
                                external_override_count,
                            )
                        )
                        direct_policy_trace_payload["external_override_count"] = int(
                            external_override_count
                        )
            if sample:
                action_idx = int(self.rng.choice(self.action_dim, p=policy))
            else:
                action_idx = motor_action_idx
            if hasattr(self.true_monolithic_policy, "record_executed_action"):
                self.true_monolithic_policy.record_executed_action(action_idx)
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
                        **(
                            direct_policy_trace_payload
                            if result.name
                            in {self.TRUE_MONOLITHIC_POLICY_NAME, B_SERIES_POLICY_NAME}
                            else {}
                        ),
                    },
                )

        if not (self.config.is_true_monolithic or self.config.is_b_series):
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
                    food_bias_action = self._food_direction_bias_action(observation)
                    if food_bias_action is not None:
                        total_logits = total_logits.copy()
                        total_logits[ACTION_TO_INDEX[food_bias_action]] += (
                            MODULAR_DIRECTION_BIAS_LOGIT
                        )
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
            if self.config.is_b_series:
                bus.publish(
                    sender=B_SERIES_POLICY_NAME,
                    topic="action.selection",
                    payload={
                        "policy_mode": policy_mode,
                        "direct_policy_logits": total_logits.round(6).tolist(),
                        "policy": policy.round(6).tolist(),
                        "selected_action": ACTIONS[motor_action_idx],
                        "executed_action": ACTIONS[action_idx],
                        "value_estimate": round(float(value), 6),
                        **direct_policy_trace_payload,
                    },
                )
            elif self.config.is_true_monolithic:
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
                        **direct_policy_trace_payload,
                    },
                )
                self._direct_policy_hidden_reset_pending = False
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
        if policy_mode == "normal" and (
            self.config.is_true_monolithic or self.config.is_b_series
        ):
            brain_observation_keys = set(observation.keys())
            step_observation = {
                key: np.asarray(observation[key], dtype=float).copy()
                for key in brain_observation_keys
                if key in observation and key != "meta"
            }
        elif (
            policy_mode == "normal"
            and self.config.is_modular
            and self.config.uses_counterfactual_credit
        ):
            brain_observation_keys = {
                spec.observation_key for spec in MODULE_INTERFACES
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
            phase_logits=phase_logits,
            phase_prediction=phase_prediction,
            phase_prediction_confidence=float(phase_prediction_confidence),
            event_attention_top_type=event_attention_top_type,
            event_attention_top_age=int(event_attention_top_age),
            event_attention_entropy=float(event_attention_entropy),
            selected_option=selected_option,
            option_age=int(option_age),
            option_termination_reason=option_termination_reason,
            option_logits=np.asarray(option_logits, dtype=float).copy(),
            option_leaf_logits=np.asarray(option_leaf_logits, dtype=float).copy(),
            option_owned_action=option_owned_action,
            safety_mask_applied=bool(safety_mask_applied),
            safety_masked_actions=tuple(safety_masked_actions),
            external_override_count=int(external_override_count),
            affordance_blocked_logits=np.asarray(
                affordance_blocked_logits,
                dtype=float,
            ).copy(),
            affordance_role_logits=np.asarray(
                affordance_role_logits,
                dtype=float,
            ).copy(),
            geometry_logits=np.asarray(geometry_logits, dtype=float).copy(),
            shelter_column_logits=np.asarray(
                shelter_column_logits,
                dtype=float,
            ).copy(),
            shelter_position_logits=np.asarray(
                shelter_position_logits,
                dtype=float,
            ).copy(),
            transition_prediction_logits=np.asarray(
                transition_prediction_logits,
                dtype=float,
            ).copy(),
            transition_rollout_prediction_logits=np.asarray(
                transition_rollout_prediction_logits,
                dtype=float,
            ).copy(),
            b_level=int(self.config.b_level) if self.config.is_b_series else -1,
            b_effective_level=(
                str(direct_policy_trace_payload.get("b_effective_level"))
                if self.config.is_b_series
                and direct_policy_trace_payload.get("b_effective_level") is not None
                else None
            ),
            b_mode=str(self.config.b_mode) if self.config.is_b_series else None,
            semantic_action=(
                direct_policy_trace_payload.get("semantic_action")
                if self.config.is_b_series
                else None
            ),
            semantic_action_idx=(
                int(direct_policy_trace_payload.get("semantic_action_idx", -1))
                if self.config.is_b_series
                else -1
            ),
            learned_semantic_action=(
                direct_policy_trace_payload.get("learned_semantic_action")
                if self.config.is_b_series
                else None
            ),
            learned_semantic_action_idx=(
                int(direct_policy_trace_payload.get("learned_semantic_action_idx", -1))
                if self.config.is_b_series
                else -1
            ),
            semantic_action_source=(
                direct_policy_trace_payload.get("semantic_action_source")
                if self.config.is_b_series
                else None
            ),
            semantic_action_reason=(
                direct_policy_trace_payload.get("semantic_action_reason")
                if self.config.is_b_series
                else None
            ),
            semantic_override_count=(
                int(direct_policy_trace_payload.get("semantic_override_count", 0))
                if self.config.is_b_series
                else 0
            ),
            semantic_logits=(
                np.asarray(semantic_logits, dtype=float).copy()
                if self.config.is_b_series
                else np.zeros(0, dtype=float)
            ),
            semantic_policy=(
                np.asarray(semantic_policy, dtype=float).copy()
                if self.config.is_b_series
                else np.zeros(0, dtype=float)
            ),
            bridge_primitive_action=(
                direct_policy_trace_payload.get("bridge_primitive_action")
                if self.config.is_b_series
                else None
            ),
            bridge_reason=(
                direct_policy_trace_payload.get("bridge_reason")
                if self.config.is_b_series
                else None
            ),
            blocked_mask=(
                dict(direct_policy_trace_payload.get("blocked_mask", {}))
                if self.config.is_b_series
                else {}
            ),
            food_delta_used=(
                float(direct_policy_trace_payload.get("food_delta_used", 0.0))
                if self.config.is_b_series
                else 0.0
            ),
            shelter_delta_used=(
                float(direct_policy_trace_payload.get("shelter_delta_used", 0.0))
                if self.config.is_b_series
                else 0.0
            ),
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
        if self.config.is_b_series:
            if self.b_series_policy is None:
                raise RuntimeError(
                    "B-series policy unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            return float(
                self.b_series_policy.value_only(monolithic_observation)
            )
        if self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
                )
            monolithic_observation = self._build_monolithic_observation(observation)
            if hasattr(self.true_monolithic_policy, "set_runtime_observation_meta"):
                self.true_monolithic_policy.set_runtime_observation_meta(
                    observation.get("meta", {})
                )
            runtime_state_snapshot = self.snapshot_direct_policy_runtime_state()
            try:
                direct_forward = self.true_monolithic_policy.forward(
                    monolithic_observation,
                    store_cache=False,
                )
                if len(direct_forward) >= 3:
                    _, value, *_ = direct_forward
                else:
                    _, value = direct_forward
                return float(value)
            finally:
                self.restore_direct_policy_runtime_state(runtime_state_snapshot)
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
        if getattr(self, "b_series_policy", None) is not None:
            return [B_SERIES_POLICY_NAME]
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
            module_variants=getattr(self.config, "module_variants", None),
            learned_arbitration=(
                self.config.use_learned_arbitration and self.arbitration_network is not None
            ),
            arbitration_input_dim=arbitration_input_dim,
            arbitration_hidden_dim=arbitration_hidden_dim,
            arbitration_regularization_weight=self.arbitration_regularization_weight,
            capacity_profile_name=self.config.capacity_profile_name,
            module_hidden_dims=self.config.module_hidden_dims,
            action_center_hidden_dim=self.config.action_center_hidden_dim,
            motor_hidden_dim=self.config.motor_hidden_dim,
            integration_hidden_dim=self.config.integration_hidden_dim,
            monolithic_hidden_dim=self.config.monolithic_hidden_dim,
            b_level=getattr(self.config, "b_level", 0),
            b_mode=getattr(self.config, "b_mode", "current_bridge"),
            direct_policy_hidden_dims=self.config.direct_policy_hidden_dims or None,
            direct_policy_recurrent=bool(self.config.direct_policy_recurrent),
            direct_policy_phase_head=bool(self.config.direct_policy_phase_head),
            direct_policy_event_attention=bool(
                self.config.direct_policy_event_attention
            ),
            direct_policy_event_buffer_size=int(
                self.config.direct_policy_event_buffer_size
            ),
            direct_policy_option_head=bool(self.config.direct_policy_option_head),
            direct_policy_owned_option_controller=bool(
                self.config.direct_policy_owned_option_controller
            ),
            direct_policy_option_ttl=int(self.config.direct_policy_option_ttl),
            direct_policy_affordance_head=bool(
                self.config.direct_policy_affordance_head
            ),
            direct_policy_affordance_feedback=bool(
                self.config.direct_policy_affordance_feedback
            ),
            direct_policy_geometry_head=bool(
                self.config.direct_policy_geometry_head
            ),
            direct_policy_shelter_column_head=bool(
                self.config.direct_policy_shelter_column_head
            ),
            direct_policy_shelter_position_head=bool(
                self.config.direct_policy_shelter_position_head
            ),
            direct_policy_local_affordance_inputs=bool(
                getattr(self.config, "direct_policy_local_affordance_inputs", False)
            ),
            direct_policy_local_spatial_inputs=bool(
                getattr(self.config, "direct_policy_local_spatial_inputs", False)
            ),
            direct_policy_local_transition_inputs=bool(
                getattr(self.config, "direct_policy_local_transition_inputs", False)
            ),
            direct_policy_local_transition_rollout_inputs=bool(
                getattr(
                    self.config,
                    "direct_policy_local_transition_rollout_inputs",
                    False,
                )
            ),
            direct_policy_transition_prediction_head=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_prediction_head",
                    False,
                )
            ),
            direct_policy_transition_prediction_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_prediction_feedback",
                    False,
                )
            ),
            direct_policy_transition_rollout_prediction_head=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_rollout_prediction_head",
                    False,
                )
            ),
            direct_policy_transition_rollout_prediction_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_transition_rollout_prediction_feedback",
                    False,
                )
            ),
            direct_policy_handoff_teacher=bool(
                self.config.direct_policy_handoff_teacher
            ),
            direct_policy_handoff_option_teacher=bool(
                getattr(self.config, "direct_policy_handoff_option_teacher", False)
            ),
            direct_policy_post_rest_action_teacher=bool(
                getattr(self.config, "direct_policy_post_rest_action_teacher", False)
            ),
            direct_policy_post_rest_release_sequence_teacher=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_teacher",
                    False,
                )
            ),
            direct_policy_post_rest_release_sequence_replay_boost=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_replay_boost",
                    False,
                )
            ),
            direct_policy_post_rest_release_sequence_distill=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_distill",
                    False,
                )
            ),
            direct_policy_post_rest_probe_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_sequence_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_sequence_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_family_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_family_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_handoff_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_handoff_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_trajectory_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_trajectory_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_cycle_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_cycle_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_trace_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_trace_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_rollout_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_rollout_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_frontier_teacher_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_frontier_teacher_distillation",
                    False,
                )
            ),
            direct_policy_post_rest_probe_replayable_teacher_distillation=bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_probe_replayable_teacher_distillation",
                    False,
                )
            ),
            direct_policy_continuation_replay_passes=int(
                getattr(
                    self.config,
                    "direct_policy_continuation_replay_passes",
                    0,
                )
            ),
            direct_policy_continuation_replay_lr_scale=float(
                getattr(
                    self.config,
                    "direct_policy_continuation_replay_lr_scale",
                    0.0,
                )
            ),
            direct_policy_continuation_margin_weight=float(
                getattr(
                    self.config,
                    "direct_policy_continuation_margin_weight",
                    0.0,
                )
            ),
            direct_policy_phase_option_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_phase_option_feedback",
                    False,
                )
            ),
            direct_policy_option_transition_feedback=bool(
                getattr(
                    self.config,
                    "direct_policy_option_transition_feedback",
                    False,
                )
            ),
            direct_policy_option_termination_cooldown=bool(
                getattr(
                    self.config,
                    "direct_policy_option_termination_cooldown",
                    False,
                )
            ),
            direct_policy_option_action_head=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_head",
                    False,
                )
            ),
            direct_policy_option_decoder_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_decoder_state",
                    False,
                )
            ),
            direct_policy_option_recurrent_dynamics=bool(
                getattr(
                    self.config,
                    "direct_policy_option_recurrent_dynamics",
                    False,
                )
            ),
            direct_policy_option_sequence_head=bool(
                getattr(
                    self.config,
                    "direct_policy_option_sequence_head",
                    False,
                )
            ),
            direct_policy_option_decoder_recurrent_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_decoder_recurrent_state",
                    False,
                )
            ),
            direct_policy_option_action_transition_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_transition_state",
                    False,
                )
            ),
            direct_policy_option_action_controller_state=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_controller_state",
                    False,
                )
            ),
            direct_policy_option_action_token_decoder=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_token_decoder",
                    False,
                )
            ),
            direct_policy_option_action_recurrent_core=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_recurrent_core",
                    False,
                )
            ),
            direct_policy_option_action_separate_recurrent_head=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_separate_recurrent_head",
                    False,
                )
            ),
            direct_policy_option_action_separate_policy_path=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_separate_policy_path",
                    False,
                )
            ),
            direct_policy_option_action_separate_backbone=bool(
                getattr(
                    self.config,
                    "direct_policy_option_action_separate_backbone",
                    False,
                )
            ),
            direct_policy_executive_physiology_option_gating=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_physiology_option_gating",
                    False,
                )
            ),
            direct_policy_executive_affordance_action_gating=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_affordance_action_gating",
                    False,
                )
            ),
            direct_policy_executive_option_action_masking=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_option_action_masking",
                    False,
                )
            ),
            direct_policy_executive_event_release_latching=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_event_release_latching",
                    False,
                )
            ),
            direct_policy_executive_event_release_action_commitment=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_event_release_action_commitment",
                    False,
                )
            ),
            direct_policy_executive_release_phase_state=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_phase_state",
                    False,
                )
            ),
            direct_policy_executive_release_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_progression",
                    False,
                )
            ),
            direct_policy_executive_release_exit_contract=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_exit_contract",
                    False,
                )
            ),
            direct_policy_executive_release_substate_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_release_substate_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_continuation=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_continuation",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_guidance=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_guidance",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_commitment=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_commitment",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_food_heading_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_food_heading_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_smell_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_smell_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_corridor_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_corridor_progression",
                    False,
                )
            ),
            direct_policy_executive_post_exit_corridor_affordance_progression=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_exit_corridor_affordance_progression",
                    False,
                )
            ),
            direct_policy_executive_post_food_return=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_food_return",
                    False,
                )
            ),
            direct_policy_executive_post_food_vector_return=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_food_vector_return",
                    False,
                )
            ),
            direct_policy_executive_post_food_path_return=bool(
                getattr(
                    self.config,
                    "direct_policy_executive_post_food_path_return",
                    False,
                )
            ),
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
        if getattr(self, "b_series_policy", None) is not None:
            norms[B_SERIES_POLICY_NAME] = self.b_series_policy.parameter_norm()
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
        if getattr(self, "b_series_policy", None) is not None:
            counts[B_SERIES_POLICY_NAME] = self.b_series_policy.count_parameters()
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
        if getattr(self, "b_series_policy", None) is not None:
            return [B_SERIES_POLICY_NAME]
        return [self.MONOLITHIC_POLICY_NAME, self.ARBITRATION_NETWORK_NAME, "action_center", "motor_cortex"]
