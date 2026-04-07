from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .ablations import BrainAblationConfig, default_brain_config
from .bus import MessageBus
from .interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_TO_INDEX,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    architecture_signature,
    interface_registry,
)
from .modules import MODULE_HIDDEN_DIMS, CorticalModuleBank, ModuleResult, ReflexDecision
from .nn import MotorNetwork, ProposalNetwork, one_hot, softmax
from .operational_profiles import OperationalProfile, resolve_operational_profile
from .world import ACTIONS


@dataclass
class BrainStep:
    module_results: List[ModuleResult]
    action_center_logits: np.ndarray
    action_center_policy: np.ndarray
    motor_correction_logits: np.ndarray
    total_logits_without_reflex: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    total_logits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    policy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    value: float = 0.0
    action_intent_idx: int = 0
    motor_action_idx: int = 0
    action_idx: int = 0
    motor_override: bool = False
    final_reflex_override: bool = False
    action_center_input: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    motor_input: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    policy_mode: str = "normal"


class SpiderBrain:
    """Cérebro neuro-modular com propostas locomotoras padronizadas por interface.

    A memória explícita permanece no mundo e chega aqui apenas como observação nomeada.
    O mundo continua dono do estado ecológico e da memória explícita; `interfaces.py`
    define os contratos nomeados; `modules.py` executa apenas os propositores neurais.
    Aqui ficam os reflexos locais interpretáveis e a arbitragem motora final.
    """

    ARCHITECTURE_VERSION = 8
    MONOLITHIC_POLICY_NAME = "monolithic_policy"
    MONOLITHIC_HIDDEN_DIM = sum(MODULE_HIDDEN_DIMS.values())

    def __init__(
        self,
        seed: int = 0,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        config: BrainAblationConfig | None = None,
        operational_profile: str | OperationalProfile | None = None,
    ) -> None:
        """
        Create and configure a SpiderBrain, building either a modular or monolithic proposal stage and the motor/value network.
        
        The constructor resolves the operational profile (which supplies reflex auxiliary weights, logit strengths, and thresholds), applies the provided or default ablation/configuration, and instantiates proposal and motor networks according to the selected architecture.
        
        Parameters:
            seed (int): RNG seed for reproducible initialization.
            gamma (float): Discount factor used by learning updates.
            module_lr (float): Learning rate for proposal modules or the monolithic proposal.
            motor_lr (float): Learning rate for the motor/value network.
            module_dropout (float): Default module dropout probability used when no config is provided.
            config (BrainAblationConfig | None): Optional ablation/config that selects modular vs monolithic mode and may override module_dropout; when omitted a default config is created.
            operational_profile (str | OperationalProfile | None): Operational profile or its name; resolved into runtime reflex parameters (auxiliary weights, logit strengths, and thresholds) used by reflex decision logic.
        """
        self.rng = np.random.default_rng(seed)
        self.action_dim = len(ACTIONS)
        self.config = config if config is not None else default_brain_config(module_dropout=module_dropout)
        self.operational_profile = resolve_operational_profile(operational_profile)
        self.reflex_aux_weights = self.operational_profile.brain_aux_weights
        self.reflex_logit_strengths = self.operational_profile.brain_reflex_logit_strengths
        self.reflex_thresholds = self.operational_profile.brain_reflex_thresholds
        self.current_reflex_scale = float(self.config.reflex_scale)
        self.module_bank: CorticalModuleBank | None = None
        self.monolithic_policy: ProposalNetwork | None = None
        if self.config.is_modular:
            self.module_bank = CorticalModuleBank(
                action_dim=self.action_dim,
                rng=self.rng,
                module_dropout=self.config.module_dropout,
                disabled_modules=self.config.disabled_modules,
            )
            action_input_dim = self.action_dim * len(self.module_bank.specs) + ACTION_CONTEXT_INTERFACE.input_dim
        else:
            monolithic_input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
            self.monolithic_policy = ProposalNetwork(
                input_dim=monolithic_input_dim,
                hidden_dim=self.MONOLITHIC_HIDDEN_DIM,
                output_dim=self.action_dim,
                rng=self.rng,
                name=self.MONOLITHIC_POLICY_NAME,
            )
            action_input_dim = self.action_dim + ACTION_CONTEXT_INTERFACE.input_dim
        motor_input_dim = self.action_dim + MOTOR_CONTEXT_INTERFACE.input_dim
        self.action_center = MotorNetwork(
            input_dim=action_input_dim,
            hidden_dim=32,
            output_dim=self.action_dim,
            rng=self.rng,
            name="action_center",
        )
        self.motor_cortex = ProposalNetwork(
            input_dim=motor_input_dim,
            hidden_dim=32,
            output_dim=self.action_dim,
            rng=self.rng,
            name="motor_cortex",
        )
        self.gamma = gamma
        self.module_lr = module_lr
        self.motor_lr = motor_lr
        self.module_dropout = self.config.module_dropout

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
        Restore the runtime reflex scaling factor to the configured default.
        
        Sets self.current_reflex_scale to the configured default reflex scale after validating it is finite.
        """
        value = float(self.config.reflex_scale)
        if not math.isfinite(value):
            raise ValueError("non-finite reflex scale")
        self.current_reflex_scale = max(0.0, value)

    def _effective_reflex_scale(self, module_name: str) -> float:
        """
        Compute the effective reflex scaling factor for a given module.
        
        Parameters:
            module_name (str): Module identifier used to look up a configured per-module reflex multiplier.
        
        Returns:
            float: Effective scale (>= 0.0) applied to reflex strengths for the module. Returns 0.0 if reflexes are disabled or the architecture is not modular; otherwise the product of the current runtime reflex scale and the module's configured multiplier (clamped to be non-negative).
        """
        if not self.config.enable_reflexes or not self.config.is_modular:
            return 0.0
        return max(
            0.0,
            float(self.current_reflex_scale)
            * float(self.config.module_reflex_scales.get(module_name, 1.0)),
        )

    def _build_monolithic_observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build a flat numeric observation vector for the monolithic proposal network.
        
        Converts each module interface's observation array to float, replaces NaN with 0.0,
        +inf with 1.0 and -inf with -1.0, then concatenates them in the order of MODULE_INTERFACES.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Mapping of observation keys to arrays; keys must include
                each spec.observation_key from MODULE_INTERFACES.
        
        Returns:
            np.ndarray: 1-D concatenated float vector containing the sanitized observations for all interfaces.
        """
        return np.concatenate(
            [
                np.nan_to_num(
                    np.asarray(observation[spec.observation_key], dtype=float),
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                )
                for spec in MODULE_INTERFACES
            ],
            axis=0,
        )

    def _build_action_input(self, module_results: List[ModuleResult], observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Construct the flattened input vector for the action center by concatenating proposal logits and bound action context.

        Parameters:
        	module_results (List[ModuleResult]): Per-module proposal results whose `logits` arrays are concatenated in proposal order.
        	observation (Dict[str, np.ndarray]): Full observation mapping; the action context is taken from the key defined by ACTION_CONTEXT_INTERFACE.observation_key and bound/flattened via ACTION_CONTEXT_INTERFACE.

        Returns:
        	np.ndarray: 1-D array formed by concatenating all module logits followed by the action context vector.
        """
        logits_flat = np.concatenate([result.logits for result in module_results], axis=0)
        action_context_mapping = ACTION_CONTEXT_INTERFACE.bind_values(
            observation[ACTION_CONTEXT_INTERFACE.observation_key]
        )
        action_context = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_context_mapping)
        return np.concatenate([logits_flat, action_context], axis=0)

    def _build_motor_input(self, action_intent: np.ndarray, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Build the flattened input vector for the motor cortex by combining a sanitized action intent and the bound motor context.
        
        The intent vector is converted to a finite float array (NaN -> 0.0, +inf -> 1.0, -inf -> -1.0) and must match shape (action_dim,), otherwise a ValueError is raised. The motor context is obtained from the observation via MOTOR_CONTEXT_INTERFACE and concatenated to the intent.
        
        Parameters:
            action_intent (np.ndarray): One-hot or real-valued intent vector for the chosen action.
            observation (Dict[str, np.ndarray]): Raw observation dict used to bind motor context.
        
        Returns:
            np.ndarray: Concatenation of the sanitized intent and the motor-context vector.
        """
        intent = np.nan_to_num(np.asarray(action_intent, dtype=float), nan=0.0, posinf=1.0, neginf=-1.0)
        if intent.shape != (self.action_dim,):
            raise ValueError(
                f"action_intent esperado com shape {(self.action_dim,)}, recebido {intent.shape}."
            )
        motor_context_mapping = MOTOR_CONTEXT_INTERFACE.bind_values(
            observation[MOTOR_CONTEXT_INTERFACE.observation_key]
        )
        motor_context = MOTOR_CONTEXT_INTERFACE.vector_from_mapping(motor_context_mapping)
        return np.concatenate([intent, motor_context], axis=0)

    def _proposal_results(
        self,
        observation: Dict[str, np.ndarray],
        *,
        store_cache: bool,
        training: bool,
    ) -> List[ModuleResult]:
        """
        Produce per-module proposal outputs for the current architecture (modular or monolithic).
        
        For the modular configuration this delegates to the module bank and returns its per-module
        ModuleResult list. For the monolithic configuration this constructs a single
        ModuleResult named "monolithic_policy" from the sanitized monolithic observation,
        with `logits`, `probs`, and reflex-related diagnostic fields initialized (no reflex applied).
        
        Parameters:
            observation (Dict[str, np.ndarray]): Raw observation mapping keyed by module/interface names.
            store_cache (bool): Allow proposal components to cache intermediate state for later learning.
            training (bool): Run proposal components in training mode (e.g., with dropout enabled).
        
        Returns:
            List[ModuleResult]: Per-proposal-module results including logits, probabilities and reflex/diagnostic fields.
        
        Raises:
            RuntimeError: If the configured proposal backend (module bank for modular, monolithic policy for monolithic) is not available.
        """
        if self.config.is_modular:
            if self.module_bank is None:
                raise RuntimeError("Banco de módulos indisponível para arquitetura modular.")
            return self.module_bank.forward(
                observation,
                store_cache=store_cache,
                training=training,
            )

        if self.monolithic_policy is None:
            raise RuntimeError("Rede monolítica indisponível para arquitetura configurada.")
        monolithic_observation = self._build_monolithic_observation(observation)
        logits = self.monolithic_policy.forward(monolithic_observation, store_cache=store_cache)
        return [
            ModuleResult(
                interface=None,
                name=self.MONOLITHIC_POLICY_NAME,
                observation_key=self.MONOLITHIC_POLICY_NAME,
                observation=monolithic_observation.copy(),
                logits=logits,
                probs=softmax(logits),
                active=True,
                reflex=None,
                neural_logits=logits.copy(),
                reflex_delta_logits=np.zeros_like(logits),
                post_reflex_logits=logits.copy(),
            )
        ]

    def _apply_reflex_path(self, module_results: List[ModuleResult]) -> None:
        """
        Apply per-module reflex adjustments in-place so proposal outputs match the act() decision path.
        """
        should_compute_reflexes = self.config.is_modular and self.config.enable_reflexes
        if should_compute_reflexes:
            for result in module_results:
                result.neural_logits = result.logits.copy()
                result.reflex_delta_logits = np.zeros_like(result.logits)
                result.post_reflex_logits = result.logits.copy()
                result.reflex_applied = False
                result.effective_reflex_scale = 0.0
                result.module_reflex_override = False
                result.module_reflex_dominance = 0.0
                if not result.active:
                    continue
                result.reflex = self._module_reflex_decision(result)
                reflex = result.reflex
                if reflex is None:
                    continue
                effective_scale = self._effective_reflex_scale(result.name)
                result.effective_reflex_scale = float(effective_scale)
                effective_logit_strength = float(reflex.logit_strength) * effective_scale
                reflex.auxiliary_weight = float(reflex.auxiliary_weight) * effective_scale
                reflex.logit_strength = effective_logit_strength
                if effective_logit_strength <= 0.0:
                    continue
                result.reflex_delta_logits = effective_logit_strength * reflex.target_probs
                result.post_reflex_logits = result.neural_logits + result.reflex_delta_logits
                result.logits = result.post_reflex_logits.copy()
                result.probs = softmax(result.logits)
                result.reflex_applied = bool(np.any(np.abs(result.reflex_delta_logits) > 1e-12))
                neural_argmax = int(np.argmax(result.neural_logits))
                post_argmax = int(np.argmax(result.post_reflex_logits))
                result.module_reflex_override = neural_argmax != post_argmax
                denom = (
                    float(np.sum(np.abs(result.neural_logits)))
                    + float(np.sum(np.abs(result.reflex_delta_logits)))
                    + 1e-8
                )
                result.module_reflex_dominance = float(
                    np.sum(np.abs(result.reflex_delta_logits)) / denom
                )
        else:
            for result in module_results:
                result.neural_logits = result.logits.copy()
                result.reflex_delta_logits = np.zeros_like(result.logits)
                result.post_reflex_logits = result.logits.copy()
                result.reflex_applied = False
                result.effective_reflex_scale = 0.0
                result.module_reflex_override = False
                result.module_reflex_dominance = 0.0

    def act(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = True,
        policy_mode: str = "normal",
    ) -> BrainStep:
        """
        Select an action for the given observation and return a populated BrainStep.
        
        Runs the proposal stage and optionally applies per-module reflex adjustments. In the default `"normal"` mode it then computes action-center and motor-cortex corrections before selecting a final action. In `"reflex_only"` mode it skips `action_center` and `motor_cortex`, sums only the post-reflex proposal logits, and selects directly from that policy. The method can also publish per-module and final-selection diagnostics to a MessageBus.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Mapping of sensory/motor arrays consumed by proposal modules and context interfaces.
            bus (MessageBus | None): Optional message bus for publishing module proposals and final selection; omit to disable publishing.
            sample (bool): If True, sample an action from the final policy distribution; if False, choose the greedy argmax action.
            policy_mode (str): `"normal"` for the standard learned action-center/motor path, or `"reflex_only"` to execute only the modular reflex-adjusted proposal path.
        
        Returns:
            BrainStep: Decision container populated with per-module ModuleResult entries and diagnostics, action-center logits and policy, motor-correction logits, combined logits (with and without reflexes), final policy, scalar value estimate, selected intent and action indices, override flags, and the action-center and motor-cortex input vectors.
        """
        if policy_mode not in {"normal", "reflex_only"}:
            raise ValueError(
                "policy_mode inválido. Use 'normal' ou 'reflex_only'."
            )
        if policy_mode == "reflex_only" and not self.config.is_modular:
            raise ValueError(
                "policy_mode='reflex_only' requer a arquitetura modular."
            )
        if policy_mode == "reflex_only" and not self.config.enable_reflexes:
            raise ValueError(
                "policy_mode='reflex_only' requer reflexos habilitados."
            )

        store_cache = sample and policy_mode == "normal"
        module_results = self._proposal_results(
            observation,
            store_cache=store_cache,
            training=sample,
        )
        proposal_sum_without_reflex = np.sum(
            np.stack([result.logits for result in module_results], axis=0),
            axis=0,
        )
        if policy_mode == "reflex_only":
            action_center_logits_without_reflex = proposal_sum_without_reflex.copy()
            action_intent_without_reflex_idx = int(
                np.argmax(action_center_logits_without_reflex)
            )
            total_logits_without_reflex = proposal_sum_without_reflex.copy()
        else:
            action_input_without_reflex = self._build_action_input(
                module_results,
                observation,
            )
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
            motor_correction_without_reflex = self.motor_cortex.forward(
                motor_input_without_reflex,
                store_cache=False,
            )
            total_logits_without_reflex = (
                action_center_logits_without_reflex + motor_correction_without_reflex
            )

        self._apply_reflex_path(module_results)

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
                    },
                )

        proposal_sum = np.sum(
            np.stack([result.logits for result in module_results], axis=0),
            axis=0,
        )
        action_center_input = self._build_action_input(module_results, observation)
        value = 0.0
        action_center_correction_logits = np.zeros(self.action_dim, dtype=float)
        if policy_mode == "reflex_only":
            action_center_logits = proposal_sum.copy()
            action_center_policy = softmax(action_center_logits)
            action_intent_idx = int(np.argmax(action_center_policy))
            motor_input = self._build_motor_input(
                one_hot(action_intent_idx, self.action_dim),
                observation,
            )
            motor_correction_logits = np.zeros(self.action_dim, dtype=float)
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
            action_center_correction_logits, value = self.action_center.forward(
                action_center_input,
                store_cache=True,
            )
            action_center_logits = proposal_sum + action_center_correction_logits
            action_center_policy = softmax(action_center_logits)
            action_intent_idx = int(np.argmax(action_center_policy))
            motor_input = self._build_motor_input(
                one_hot(action_intent_idx, self.action_dim),
                observation,
            )
            motor_correction_logits = self.motor_cortex.forward(motor_input, store_cache=True)
            total_logits = action_center_logits + motor_correction_logits
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
                or action_without_reflex_idx != motor_action_idx
            )

        if bus is not None:
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
                },
            )

        return BrainStep(
            module_results=module_results,
            action_center_logits=action_center_logits,
            action_center_policy=action_center_policy,
            motor_correction_logits=motor_correction_logits,
            total_logits_without_reflex=total_logits_without_reflex,
            total_logits=total_logits,
            policy=policy,
            value=float(value),
            action_intent_idx=action_intent_idx,
            motor_action_idx=motor_action_idx,
            action_idx=action_idx,
            motor_override=bool(motor_override),
            final_reflex_override=bool(final_reflex_override),
            action_center_input=action_center_input,
            motor_input=motor_input,
            policy_mode=policy_mode,
        )

    def estimate_value(self, observation: Dict[str, np.ndarray]) -> float:
        """
        Estimate the action-center value for a given observation.
        
        Builds module proposals from the observation, applies any reflex modifications, constructs the action-center input, and returns the scalar state-value produced by the action-center.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Observation arrays keyed by interface names, used to produce module proposals and to build the action-center input.
        
        Returns:
            float: Scalar value estimate for the provided observation.
        """
        module_results = self._proposal_results(
            observation,
            store_cache=False,
            training=False,
        )
        self._apply_reflex_path(module_results)
        action_input = self._build_action_input(module_results, observation)
        _, value = self.action_center.forward(action_input, store_cache=False)
        return float(value)

    def _action_target(self, action: str) -> np.ndarray:
        """
        Return a one-hot action vector for the given action name.
        
        Parameters:
            action (str): Action name key present in ACTION_TO_INDEX.
        
        Returns:
            np.ndarray: One-hot vector of length self.action_dim with `1.0` at the action's index and `0.0` elsewhere.
        """
        return one_hot(ACTION_TO_INDEX[action], self.action_dim)

    def _direction_action(self, dx: float, dy: float, *, away: bool = False) -> str:
        dx = float(-dx if away else dx)
        dy = float(-dy if away else dy)
        if abs(dx) < 0.05 and abs(dy) < 0.05:
            return "STAY"
        if abs(dx) >= abs(dy):
            return "MOVE_RIGHT" if dx > 0 else "MOVE_LEFT"
        return "MOVE_DOWN" if dy > 0 else "MOVE_UP"

    def _signal_subset(self, signals: Dict[str, float], *names: str) -> Dict[str, float]:
        return {
            name: float(signals[name])
            for name in names
        }

    def _reflex_decision(
        self,
        module_name: str,
        *,
        action: str,
        reason: str,
        triggers: Dict[str, float],
    ) -> ReflexDecision:
        """
        Constructs a ReflexDecision for a named proposal module using the brain's configured auxiliary weight and logit strength.
        
        Parameters:
            module_name (str): Name of the proposal module used to look up auxiliary weight and logit strength in the brain's operational profile.
            action (str): Action name to target; used to build a one-hot target probability vector.
            reason (str): Short human-readable explanation for the reflex decision.
            triggers (Dict[str, float]): Mapping of trigger names to their observed magnitudes that caused this reflex.
        
        Returns:
            ReflexDecision: Decision object containing the action, one-hot `target_probs` for the action, `reason`, `triggers`, and the module-specific `auxiliary_weight` and `logit_strength`.
        """
        return ReflexDecision(
            action=action,
            target_probs=self._action_target(action),
            reason=reason,
            triggers=triggers,
            auxiliary_weight=self.reflex_aux_weights.get(module_name, 0.0),
            logit_strength=self.reflex_logit_strengths.get(module_name, 0.0),
        )

    def _stay_reflex(
        self,
        module_name: str,
        *,
        reason: str,
        triggers: Dict[str, float],
    ) -> ReflexDecision:
        """
        Produce a ReflexDecision that targets the 'STAY' action with the given reason and trigger magnitudes.
        
        Parameters:
            module_name (str): Name of the module producing the reflex.
            reason (str): Short explanation for why the reflex was created.
            triggers (Dict[str, float]): Mapping of observed trigger names to their measured magnitudes.
        
        Returns:
            ReflexDecision: A decision object that directs the agent to remain in place (`"STAY"`) and includes the provided reason and triggers.
        """
        return self._reflex_decision(
            module_name,
            action="STAY",
            reason=reason,
            triggers=triggers,
        )

    def _direction_reflex(
        self,
        module_name: str,
        *,
        dx: float,
        dy: float,
        away: bool = False,
        reason: str,
        triggers: Dict[str, float],
    ) -> ReflexDecision:
        """
        Create a ReflexDecision that targets a directional action derived from a displacement vector.
        
        Parameters:
            module_name (str): Name of the module the reflex applies to.
            dx (float): Horizontal displacement used to choose the directional action (positive is right).
            dy (float): Vertical displacement used to choose the directional action (positive is down).
            away (bool): If True, invert the direction to move away from the (dx, dy) vector.
            reason (str): Short description of why the reflex was created.
            triggers (Dict[str, float]): Numeric evidence values that triggered the reflex.
        
        Returns:
            ReflexDecision: A reflex that targets a directional (or "STAY") action with associated metadata.
        """
        return self._reflex_decision(
            module_name,
            action=self._direction_action(dx, dy, away=away),
            reason=reason,
            triggers=triggers,
        )

    def _visual_reflex_decision(self, signals: Dict[str, float]) -> ReflexDecision | None:
        """
        Evaluate visual sensory signals and return a direction-based reflex when a predator, shelter-at-night, or visible food should be acted on.
        
        Parameters:
            signals (Dict[str, float]): Mapping of visual-related signal names to numeric values. Expected keys include:
                - predator_visible, predator_certainty, predator_dx, predator_dy
                - night, shelter_visible, shelter_certainty, shelter_dx, shelter_dy
                - food_visible, food_certainty, food_dx, food_dy
        
        Returns:
            ReflexDecision | None: A direction-based `ReflexDecision` that directs retreat from a visible predator, return to shelter at night, or approach visible food when the corresponding visual thresholds (from `self.reflex_thresholds["visual_cortex"]`) are exceeded; `None` if no visual reflex is triggered.
        """
        thresholds = self.reflex_thresholds["visual_cortex"]
        if (
            signals["predator_visible"] > thresholds["visible"]
            and signals["predator_certainty"] >= thresholds["predator_certainty"]
        ):
            return self._direction_reflex(
                "visual_cortex",
                dx=signals["predator_dx"],
                dy=signals["predator_dy"],
                away=True,
                reason="retreat_from_visible_predator",
                triggers=self._signal_subset(
                    signals,
                    "predator_visible",
                    "predator_certainty",
                    "predator_dx",
                    "predator_dy",
                ),
            )
        if (
            signals["night"] > thresholds["visible"]
            and signals["shelter_visible"] > thresholds["visible"]
            and signals["shelter_certainty"] >= thresholds["shelter_certainty"]
        ):
            return self._direction_reflex(
                "visual_cortex",
                dx=signals["shelter_dx"],
                dy=signals["shelter_dy"],
                reason="return_to_shelter_at_night",
                triggers=self._signal_subset(
                    signals,
                    "night",
                    "shelter_visible",
                    "shelter_certainty",
                    "shelter_dx",
                    "shelter_dy",
                ),
            )
        if (
            signals["food_visible"] > thresholds["visible"]
            and signals["food_certainty"] >= thresholds["food_certainty"]
        ):
            return self._direction_reflex(
                "visual_cortex",
                dx=signals["food_dx"],
                dy=signals["food_dy"],
                reason="approach_visible_food",
                triggers=self._signal_subset(
                    signals,
                    "food_visible",
                    "food_certainty",
                    "food_dx",
                    "food_dy",
                ),
            )
        return None

    def _sensory_reflex_decision(self, signals: Dict[str, float]) -> ReflexDecision | None:
        """
        Evaluate sensory signals and produce a reflex to retreat from immediate threats or to follow a food smell when thresholds are exceeded.
        
        Parameters:
        	signals (Dict[str, float]): Sensory values keyed by signal name. Expected keys used by this method include:
        		"recent_contact", "recent_pain", "predator_smell_strength", "predator_smell_dx", "predator_smell_dy",
        		"hunger", "food_smell_strength", "food_smell_dx", "food_smell_dy".
        
        Returns:
        	ReflexDecision | None: A retreat-oriented `ReflexDecision` (with `away=True`) if contact, pain, or predator smell exceed configured thresholds; a food-following `ReflexDecision` if hunger and food smell exceed thresholds; `None` if no reflex conditions are met.
        """
        thresholds = self.reflex_thresholds["sensory_cortex"]
        if (
            signals["recent_contact"] > thresholds["contact"]
            or signals["recent_pain"] > thresholds["pain"]
            or signals["predator_smell_strength"] > thresholds["predator_smell"]
        ):
            return self._direction_reflex(
                "sensory_cortex",
                dx=signals["predator_smell_dx"],
                dy=signals["predator_smell_dy"],
                away=True,
                reason="retreat_from_immediate_threat",
                triggers=self._signal_subset(
                    signals,
                    "recent_contact",
                    "recent_pain",
                    "predator_smell_strength",
                    "predator_smell_dx",
                    "predator_smell_dy",
                ),
            )
        if (
            signals["hunger"] > thresholds["hunger"]
            and signals["food_smell_strength"] > thresholds["food_smell"]
        ):
            return self._direction_reflex(
                "sensory_cortex",
                dx=signals["food_smell_dx"],
                dy=signals["food_smell_dy"],
                reason="follow_food_smell_when_hungry",
                triggers=self._signal_subset(
                    signals,
                    "hunger",
                    "food_smell_strength",
                    "food_smell_dx",
                    "food_smell_dy",
                ),
            )
        return None

    def _hunger_reflex_decision(self, signals: Dict[str, float]) -> ReflexDecision | None:
        """
        Evaluate hunger-related signals and return a hunger reflex directing the agent to stay or move toward food when configured thresholds are exceeded.
        
        Checks signals in priority order: stay on food, approach visible food, follow occluded food smell, follow food smell, and follow food memory.
        
        Parameters:
            signals (Dict[str, float]): Mapping of sensory and internal signals used by the hunger reflex. Expected keys:
                - "on_food": indicator of being on a food tile
                - "hunger": internal hunger level
                - "food_visible": strength of visible food cue
                - "food_certainty": certainty of visible food detection
                - "food_dx", "food_dy": direction vector to visible food
                - "food_occluded": indicator of occluded (seen but not reachable) food
                - "food_smell_strength": strength of food odor
                - "food_smell_dx", "food_smell_dy": direction vector for food smell
                - "food_memory_dx", "food_memory_dy": direction vector to remembered food location
                - "food_memory_age": age of the food memory
        
        Returns:
            ReflexDecision | None: A ReflexDecision describing the chosen hunger reflex (stay or a directional target) when a threshold is met, or `None` if no hunger reflex applies.
        """
        thresholds = self.reflex_thresholds["hunger_center"]
        if (
            signals["on_food"] > thresholds["on_food"]
            and signals["hunger"] > thresholds["stay_hunger"]
        ):
            return self._stay_reflex(
                "hunger_center",
                reason="stay_on_food",
                triggers=self._signal_subset(signals, "on_food", "hunger"),
            )
        if (
            signals["hunger"] > thresholds["visible_hunger"]
            and signals["food_visible"] > thresholds["visible_food"]
            and signals["food_certainty"] >= thresholds["visible_food_certainty"]
        ):
            return self._direction_reflex(
                "hunger_center",
                dx=signals["food_dx"],
                dy=signals["food_dy"],
                reason="approach_visible_food",
                triggers=self._signal_subset(
                    signals,
                    "hunger",
                    "food_visible",
                    "food_certainty",
                    "food_dx",
                    "food_dy",
                ),
            )
        if (
            signals["hunger"] > thresholds["occluded_hunger"]
            and signals["food_occluded"] > thresholds["occluded_food"]
            and signals["food_smell_strength"] > thresholds["occluded_food_smell"]
        ):
            return self._direction_reflex(
                "hunger_center",
                dx=signals["food_smell_dx"],
                dy=signals["food_smell_dy"],
                reason="follow_occluded_food_smell",
                triggers=self._signal_subset(
                    signals,
                    "hunger",
                    "food_occluded",
                    "food_smell_strength",
                    "food_smell_dx",
                    "food_smell_dy",
                ),
            )
        if (
            signals["hunger"] > thresholds["smell_hunger"]
            and signals["food_smell_strength"] > thresholds["food_smell"]
        ):
            return self._direction_reflex(
                "hunger_center",
                dx=signals["food_smell_dx"],
                dy=signals["food_smell_dy"],
                reason="follow_food_smell",
                triggers=self._signal_subset(
                    signals,
                    "hunger",
                    "food_smell_strength",
                    "food_smell_dx",
                    "food_smell_dy",
                ),
            )
        if (
            signals["hunger"] > thresholds["memory_hunger"]
            and signals["food_memory_age"] < thresholds["memory_age"]
        ):
            return self._direction_reflex(
                "hunger_center",
                dx=signals["food_memory_dx"],
                dy=signals["food_memory_dy"],
                reason="follow_food_memory",
                triggers=self._signal_subset(
                    signals,
                    "hunger",
                    "food_memory_dx",
                    "food_memory_dy",
                    "food_memory_age",
                ),
            )
        return None

    def _sleep_reflex_decision(self, signals: Dict[str, float]) -> ReflexDecision | None:
        """
        Decides whether the agent should stay in place or move to shelter/home for rest based on sleep-, shelter-, fatigue-, and hunger-related signals.
        
        Parameters:
            signals (Dict[str, float]): Sensor and internal-state values used to evaluate sleep reflexes. Expected keys include:
                - "on_shelter", "sleep_phase_level", "shelter_role_level", "rest_streak_norm"
                - "night", "fatigue", "sleep_debt", "hunger"
                - "shelter_memory_age", "shelter_memory_dx", "shelter_memory_dy"
                - "home_dx", "home_dy"
        
        Returns:
            ReflexDecision | None: A ReflexDecision that either forces staying in shelter or directs movement toward a shelter/home location when sleep-related conditions are met, or `None` if no sleep reflex is triggered.
        """
        thresholds = self.reflex_thresholds["sleep_center"]
        if (
            signals["on_shelter"] > thresholds["on_shelter"]
            and signals["sleep_phase_level"] > thresholds["sleep_phase"]
            and signals["hunger"] < thresholds["rest_hunger"]
        ):
            return self._stay_reflex(
                "sleep_center",
                reason="stay_while_sleeping",
                triggers=self._signal_subset(
                    signals,
                    "on_shelter",
                    "sleep_phase_level",
                    "hunger",
                ),
            )
        if (
            signals["on_shelter"] > thresholds["on_shelter"]
            and signals["shelter_role_level"] > thresholds["deep_shelter_level"]
            and signals["rest_streak_norm"] > thresholds["rest_streak"]
            and signals["hunger"] < thresholds["rest_hunger"]
        ):
            return self._stay_reflex(
                "sleep_center",
                reason="stay_in_deep_shelter",
                triggers=self._signal_subset(
                    signals,
                    "on_shelter",
                    "shelter_role_level",
                    "rest_streak_norm",
                    "hunger",
                ),
            )
        if (
            signals["on_shelter"] > thresholds["on_shelter"]
            and (
                signals["night"] > thresholds["on_shelter"]
                or signals["fatigue"] > thresholds["fatigue_to_hold"]
                or signals["sleep_debt"] > thresholds["sleep_debt_to_hold"]
            )
            and signals["hunger"] < thresholds["rest_hunger"]
        ):
            return self._stay_reflex(
                "sleep_center",
                reason="stay_in_shelter_to_rest",
                triggers=self._signal_subset(
                    signals,
                    "on_shelter",
                    "night",
                    "fatigue",
                    "sleep_debt",
                    "hunger",
                ),
            )
        if (
            signals["shelter_memory_age"] < thresholds["memory_age"]
            and (
                signals["fatigue"] > thresholds["memory_fatigue"]
                or signals["sleep_debt"] > thresholds["sleep_debt_to_seek"]
                or signals["night"] > thresholds["on_shelter"]
            )
            and signals["hunger"] < thresholds["memory_rest_hunger"]
        ):
            return self._direction_reflex(
                "sleep_center",
                dx=signals["shelter_memory_dx"],
                dy=signals["shelter_memory_dy"],
                reason="return_to_safe_shelter_memory",
                triggers=self._signal_subset(
                    signals,
                    "fatigue",
                    "sleep_debt",
                    "night",
                    "hunger",
                    "shelter_memory_dx",
                    "shelter_memory_dy",
                    "shelter_memory_age",
                ),
            )
        if (
            (
                signals["fatigue"] > thresholds["fatigue_to_seek"]
                or signals["sleep_debt"] > thresholds["sleep_debt_to_seek"]
                or signals["night"] > thresholds["on_shelter"]
            )
            and signals["hunger"] < thresholds["memory_rest_hunger"]
        ):
            return self._direction_reflex(
                "sleep_center",
                dx=signals["home_dx"],
                dy=signals["home_dy"],
                reason="return_home_to_rest",
                triggers=self._signal_subset(
                    signals,
                    "fatigue",
                    "sleep_debt",
                    "night",
                    "hunger",
                    "home_dx",
                    "home_dy",
                ),
            )
        return None

    def _alert_reflex_decision(self, signals: Dict[str, float]) -> ReflexDecision | None:
        """
        Determine whether an alert-related reflex should be triggered based on sensed signals.
        
        Evaluates alert-center thresholds to decide one of:
        - freeze/stay in shelter when on-shelter and a predator/threat is detected,
        - retreat away from a visible predator,
        - retreat away from a recent predator memory when outside shelter,
        - return home when non-visual threat cues (occluded predator, smell, contact, or pain) are present,
        - repeat a recent escape route when an escape memory is fresh and the agent is outside shelter.
        
        Parameters:
            signals (Dict[str, float]): Sensor and memory values referenced by keys such as
                "on_shelter", "predator_visible", "predator_occluded", "predator_smell_strength",
                "recent_contact", "predator_certainty", "predator_dx", "predator_dy",
                "predator_memory_age", "predator_memory_dx", "predator_memory_dy",
                "recent_pain", "home_dx", "home_dy", "escape_memory_age",
                "escape_memory_dx", "escape_memory_dy".
        
        Returns:
            ReflexDecision | None: A ReflexDecision describing the chosen reflex (stay or directional)
            when a threshold condition is met, otherwise `None`.
        """
        thresholds = self.reflex_thresholds["alert_center"]
        if (
            signals["on_shelter"] > thresholds["on_shelter"]
            and (
                signals["predator_visible"] > thresholds["predator_visible"]
                or signals["predator_occluded"] > thresholds["predator_occluded"]
                or signals["predator_smell_strength"] > thresholds["predator_smell"]
                or signals["recent_contact"] > thresholds["contact_any"]
            )
        ):
            return self._stay_reflex(
                "alert_center",
                reason="freeze_in_shelter_under_threat",
                triggers=self._signal_subset(
                    signals,
                    "on_shelter",
                    "predator_visible",
                    "predator_occluded",
                    "predator_smell_strength",
                    "recent_contact",
                ),
            )
        if (
            signals["predator_visible"] > thresholds["predator_visible"]
            and signals["predator_certainty"] >= thresholds["predator_certainty"]
        ):
            return self._direction_reflex(
                "alert_center",
                dx=signals["predator_dx"],
                dy=signals["predator_dy"],
                away=True,
                reason="retreat_from_visible_predator",
                triggers=self._signal_subset(
                    signals,
                    "predator_visible",
                    "predator_certainty",
                    "predator_dx",
                    "predator_dy",
                ),
            )
        if (
            signals["predator_memory_age"] < thresholds["predator_memory_age"]
            and signals["on_shelter"] < thresholds["on_shelter"]
        ):
            return self._direction_reflex(
                "alert_center",
                dx=signals["predator_memory_dx"],
                dy=signals["predator_memory_dy"],
                away=True,
                reason="retreat_from_predator_memory",
                triggers=self._signal_subset(
                    signals,
                    "on_shelter",
                    "predator_memory_dx",
                    "predator_memory_dy",
                    "predator_memory_age",
                ),
            )
        if (
            signals["predator_occluded"] > thresholds["predator_occluded"]
            or signals["predator_smell_strength"] > thresholds["predator_smell"]
            or signals["recent_contact"] > thresholds["contact_threat"]
            or signals["recent_pain"] > thresholds["pain"]
        ):
            return self._direction_reflex(
                "alert_center",
                dx=signals["home_dx"],
                dy=signals["home_dy"],
                reason="return_home_under_threat",
                triggers=self._signal_subset(
                    signals,
                    "predator_occluded",
                    "predator_smell_strength",
                    "recent_contact",
                    "recent_pain",
                    "home_dx",
                    "home_dy",
                ),
            )
        if (
            signals["escape_memory_age"] < thresholds["escape_memory_age"]
            and signals["on_shelter"] < thresholds["on_shelter"]
        ):
            return self._direction_reflex(
                "alert_center",
                dx=signals["escape_memory_dx"],
                dy=signals["escape_memory_dy"],
                reason="repeat_recent_escape_route",
                triggers=self._signal_subset(
                    signals,
                    "on_shelter",
                    "escape_memory_dx",
                    "escape_memory_dy",
                    "escape_memory_age",
                ),
            )
        return None

    def _module_reflex_decision(self, result: ModuleResult) -> ReflexDecision | None:
        """
        Determine whether a module's latest observation warrants an immediate reflex action and produce the corresponding ReflexDecision.
        
        Parameters:
            result (ModuleResult): The module's output container; its named observation will be evaluated for reflex triggers.
        
        Returns:
            ReflexDecision: Decision describing the reflex target distribution and metadata if a reflex is triggered for this module, `None` otherwise.
        """
        signals = result.named_observation()
        if result.name == "visual_cortex":
            return self._visual_reflex_decision(signals)
        if result.name == "sensory_cortex":
            return self._sensory_reflex_decision(signals)
        if result.name == "hunger_center":
            return self._hunger_reflex_decision(signals)
        if result.name == "sleep_center":
            return self._sleep_reflex_decision(signals)
        if result.name == "alert_center":
            return self._alert_reflex_decision(signals)
        return None

    def _proposal_stage_names(self) -> List[str]:
        """
        Return the ordered proposal sources that feed the action-center input.
        """
        if self.module_bank is not None:
            return [spec.name for spec in self.module_bank.specs]
        return [self.MONOLITHIC_POLICY_NAME]

    def _architecture_signature(self) -> dict[str, object]:
        """
        Return the runtime architecture signature for this brain's active proposal backend.
        """
        return architecture_signature(
            proposal_backend=self.config.architecture,
            proposal_order=self._proposal_stage_names(),
        )

    def _interface_registry(self) -> dict[str, object]:
        """
        Return the runtime-governed interface registry used by this brain.
        """
        return interface_registry()

    def _architecture_fingerprint(self) -> str:
        """
        Return the stable fingerprint for this brain's runtime architecture signature.
        """
        return str(self._architecture_signature()["fingerprint"])

    def _auxiliary_module_gradients(self, module_results: List[ModuleResult]) -> Dict[str, np.ndarray]:
        """
        Compute auxiliary gradient targets for proposal modules based on reflex decisions.
        
        If auxiliary targets are disabled via the brain configuration, returns an empty dict.
        For each module result that is active, has a non-None reflex, and whose reflex specifies
        a positive auxiliary_weight, produces a gradient array equal to
        `auxiliary_weight * (probs - reflex.target_probs)`.
        
        Parameters:
            module_results (List[ModuleResult]): List of per-module proposal outputs; each item must
                expose `name`, `active`, `probs`, and `reflex.target_probs`/`reflex.auxiliary_weight`.
        
        Returns:
            aux_grads (Dict[str, np.ndarray]): Mapping from module name to the auxiliary gradient array
            for modules that contribute auxiliary targets. Modules that do not meet the conditions are
            omitted.
        """
        if not self.config.enable_auxiliary_targets:
            return {}
        aux_grads: Dict[str, np.ndarray] = {}
        for result in module_results:
            if not result.active or result.reflex is None:
                continue
            weight = result.reflex.auxiliary_weight
            if weight <= 0.0:
                continue
            aux_grads[result.name] = weight * (result.probs - result.reflex.target_probs)
        return aux_grads

    def learn(self, decision: BrainStep, reward: float, next_observation: Dict[str, np.ndarray], done: bool) -> Dict[str, float]:
        """
        Update model parameters using a temporal-difference policy-gradient update and return training metrics.
        
        Performs a TD(0)-style update: computes TD target and clipped advantage, forms policy-gradient logits, applies optional auxiliary gradients for proposal modules (when enabled), updates either the modular module bank or the monolithic proposal network, and updates the motor/value network.
        
        Parameters:
            decision (BrainStep): Recorded action decision containing module results, motor/value outputs, selected action index, and policy.
            reward (float): Observed scalar reward following the decision.
            next_observation (Dict[str, np.ndarray]): Environment observation after taking the action, used to estimate the next state's value.
            done (bool): Whether the episode terminated after the action.
        
        Returns:
            metrics (Dict[str, float]): Scalar training diagnostics including:
                - "reward": the provided reward.
                - "td_target": computed TD target (reward + gamma * next_value).
                - "td_error": clipped advantage used for the policy update.
                - "value": the value estimate from the provided decision.
                - "next_value": the estimated value of the next observation (0.0 if done).
                - "entropy": policy entropy computed from the decision.policy.
                - "aux_modules": number of modules that received auxiliary gradients.
        """
        if decision.policy_mode != "normal":
            raise ValueError(
                "learn() só suporta decisões produzidas com policy_mode='normal'."
            )
        next_value = 0.0 if done else self.estimate_value(next_observation)
        td_target = reward + self.gamma * next_value
        advantage = float(np.clip(td_target - decision.value, -4.0, 4.0))
        grad_policy_logits = advantage * (decision.policy - one_hot(decision.action_idx, self.action_dim))
        aux_grads = {
            name: np.asarray(grad, dtype=float).copy()
            for name, grad in self._auxiliary_module_gradients(decision.module_results).items()
        }
        grad_value = decision.value - td_target
        action_center_input_grads = self.action_center.backward(
            grad_policy_logits=grad_policy_logits,
            grad_value=grad_value,
            lr=self.motor_lr,
        )
        proposal_grad_width = self.action_dim * len(decision.module_results)
        proposal_input_grads = np.asarray(
            action_center_input_grads[:proposal_grad_width],
            dtype=float,
        )

        if self.config.is_modular:
            if self.module_bank is None:
                raise RuntimeError("Banco de módulos indisponível para arquitetura modular.")
            per_module_input_grads = proposal_input_grads.reshape(
                len(decision.module_results),
                self.action_dim,
            )
            for result, extra_grad in zip(
                decision.module_results,
                per_module_input_grads,
                strict=True,
            ):
                aux_grads[result.name] = aux_grads.get(
                    result.name,
                    np.zeros(self.action_dim, dtype=float),
                ) + extra_grad
            self.module_bank.backward(grad_policy_logits, lr=self.module_lr, aux_grads=aux_grads)
        else:
            if self.monolithic_policy is None:
                raise RuntimeError("Rede monolítica indisponível para arquitetura configurada.")
            grad_for_monolithic = grad_policy_logits + proposal_input_grads
            self.monolithic_policy.backward(grad_for_monolithic, lr=self.module_lr)
        self.motor_cortex.backward(grad_policy_logits, lr=self.motor_lr)

        entropy = -float(np.sum(decision.policy * np.log(decision.policy + 1e-8)))
        return {
            "reward": float(reward),
            "td_target": float(td_target),
            "td_error": float(advantage),
            "value": float(decision.value),
            "next_value": float(next_value),
            "entropy": entropy,
            "aux_modules": float(len(aux_grads)),
        }

    def parameter_norms(self) -> Dict[str, float]:
        """
        Return the L2 norms of parameters for each trainable network component.
        
        Returns:
            norms (Dict[str, float]): Mapping from component name to its parameter norm. Includes per-module norms if a modular bank is present, `"monolithic_policy"` when a monolithic policy is present, and always includes `"action_center"` and `"motor_cortex"`.
        """
        norms: Dict[str, float] = {}
        if self.module_bank is not None:
            norms.update(self.module_bank.parameter_norms())
        if self.monolithic_policy is not None:
            norms[self.MONOLITHIC_POLICY_NAME] = self.monolithic_policy.parameter_norm()
        norms["action_center"] = self.action_center.parameter_norm()
        norms["motor_cortex"] = self.motor_cortex.parameter_norm()
        return norms

    _METADATA_FILE = "metadata.json"

    def _module_names(self) -> List[str]:
        """
        List module names present in the brain for inspection.
        
        When the brain is modular, returns the module spec names in their configured order followed by "action_center" and "motor_cortex". When monolithic, returns ["monolithic_policy", "action_center", "motor_cortex"].
        
        Returns:
            names (List[str]): Ordered list of module names and the two controller component names.
        """
        if self.module_bank is not None:
            return [spec.name for spec in self.module_bank.specs] + ["action_center", "motor_cortex"]
        return [self.MONOLITHIC_POLICY_NAME, "action_center", "motor_cortex"]

    def save(self, directory: str | Path) -> Path:
        """
        Save the brain's trainable weights and configuration metadata to the specified directory.
        
        Saves proposal module weights (one file per module or a single monolithic policy file) and the motor cortex weights as NumPy .npz files, and writes a JSON metadata file ("metadata.json") containing architecture/version, operational profile summary, ablation summary, learning rates, gamma, dropout, action dimensionality, and per-module shape metadata.
        
        Parameters:
            directory (str | Path): Destination directory for saved files; created if it does not exist.
        
        Returns:
            Path: Path to the directory where files were written.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        architecture = self._architecture_signature()
        interface_registry = self._interface_registry()
        architecture_fingerprint = str(architecture["fingerprint"])

        metadata: Dict[str, object] = {
            "architecture_version": self.ARCHITECTURE_VERSION,
            "architecture": architecture,
            "interface_registry": interface_registry,
            "architecture_fingerprint": architecture_fingerprint,
            "ablation_config": self.config.to_summary(),
            "operational_profile": self.operational_profile.to_summary(),
            "gamma": self.gamma,
            "module_lr": self.module_lr,
            "motor_lr": self.motor_lr,
            "module_dropout": self.module_dropout,
            "action_dim": self.action_dim,
            "modules": {},
        }

        if self.module_bank is not None:
            bank_state = self.module_bank.state_dict()
            for name, sd in bank_state.items():
                arrays = {k: v for k, v in sd.items() if isinstance(v, np.ndarray)}
                np.savez(directory / f"{name}.npz", **arrays)
                metadata["modules"][name] = {
                    "type": "proposal",
                    "input_dim": sd["input_dim"],
                    "hidden_dim": sd["hidden_dim"],
                    "output_dim": sd["output_dim"],
                }
        elif self.monolithic_policy is not None:
            mono_sd = self.monolithic_policy.state_dict()
            mono_arrays = {k: v for k, v in mono_sd.items() if isinstance(v, np.ndarray)}
            np.savez(directory / f"{self.MONOLITHIC_POLICY_NAME}.npz", **mono_arrays)
            metadata["modules"][self.MONOLITHIC_POLICY_NAME] = {
                "type": "proposal",
                "input_dim": mono_sd["input_dim"],
                "hidden_dim": mono_sd["hidden_dim"],
                "output_dim": mono_sd["output_dim"],
            }

        action_sd = self.action_center.state_dict()
        action_arrays = {k: v for k, v in action_sd.items() if isinstance(v, np.ndarray)}
        np.savez(directory / "action_center.npz", **action_arrays)
        metadata["modules"]["action_center"] = {
            "type": "action",
            "input_dim": action_sd["input_dim"],
            "hidden_dim": action_sd["hidden_dim"],
            "output_dim": action_sd["output_dim"],
        }

        motor_sd = self.motor_cortex.state_dict()
        motor_arrays = {k: v for k, v in motor_sd.items() if isinstance(v, np.ndarray)}
        np.savez(directory / "motor_cortex.npz", **motor_arrays)
        metadata["modules"]["motor_cortex"] = {
            "type": "motor",
            "input_dim": motor_sd["input_dim"],
            "hidden_dim": motor_sd["hidden_dim"],
            "output_dim": motor_sd["output_dim"],
        }

        (directory / self._METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    def load(
        self,
        directory: str | Path,
        *,
        modules: Sequence[str] | None = None,
    ) -> List[str]:
        """
        Load saved network weights into this brain and validate the save's metadata for compatibility.
        
        Parameters:
            directory (str | Path): Path to a directory containing metadata.json and per-module `.npz` weight files.
            modules (Sequence[str] | None): Optional subset of module names to load; when `None`, load all modules listed in the metadata.
        
        Returns:
            List[str]: Names of modules whose weights were loaded.
        
        Raises:
            FileNotFoundError: If metadata.json or any required `.npz` weight file is missing.
            ValueError: If the saved architecture signature, ablation configuration, or operational profile does not match this brain's configuration.
            KeyError: If a requested module is not present in the save, or if monolithic weights are required but missing.
        """
        directory = Path(directory)
        meta_path = directory / self._METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Arquivo de metadados não encontrado: {meta_path}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        saved_architecture_version = metadata.get("architecture_version")
        if saved_architecture_version != self.ARCHITECTURE_VERSION:
            raise ValueError(
                "O cérebro salvo usa uma versão de arquitetura incompatível com a versão atual "
                f"do simulador (save={saved_architecture_version}, atual={self.ARCHITECTURE_VERSION})."
            )
        saved_architecture = metadata.get("architecture")
        expected_architecture = self._architecture_signature()
        saved_registry = metadata.get("interface_registry")
        expected_registry = self._interface_registry()
        if saved_registry != expected_registry:
            saved_registry_fingerprint = None
            if isinstance(saved_architecture, dict):
                saved_registry_fingerprint = saved_architecture.get("registry_fingerprint")
            raise ValueError(
                "O cérebro salvo usa um registry de interfaces incompatível com o registry atual. "
                f"(save_registry_fingerprint={saved_registry_fingerprint}, "
                f"atual={expected_architecture.get('registry_fingerprint')}). "
                "Não há migração automática de checkpoints para mudanças de contrato."
            )
        if saved_architecture != expected_architecture:
            raise ValueError(
                "O cérebro salvo usa uma assinatura arquitetural incompatível com a topologia atual "
                f"(save_fingerprint={metadata.get('architecture_fingerprint')}, "
                f"atual={self._architecture_fingerprint()}). "
                "Não há migração automática de checkpoints para esta mudança; treine ou carregue "
                "um cérebro compatível com a versão atual do simulador."
            )
        saved_config = metadata.get("ablation_config")
        expected_config = self.config.to_summary()
        if saved_config != expected_config:
            raise ValueError(
                "O cérebro salvo usa uma configuração de ablação diferente da configuração atual."
            )
        saved_operational_profile = metadata.get("operational_profile")
        expected_operational_profile = self.operational_profile.to_summary()
        if saved_operational_profile != expected_operational_profile:
            raise ValueError(
                "O cérebro salvo usa um perfil operacional diferente do perfil operacional atual."
            )

        available = list(metadata.get("modules", {}).keys())
        targets = list(modules) if modules is not None else available

        loaded: List[str] = []
        bank_state: Dict[str, dict] = {}

        for name in targets:
            if name not in available:
                raise KeyError(
                    f"Módulo '{name}' não encontrado no save. "
                    f"Disponíveis: {available}"
                )
            npz_path = directory / f"{name}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Arquivo de pesos não encontrado: {npz_path}")

            mod_meta = metadata["modules"][name]
            arrays = dict(np.load(npz_path))

            if mod_meta["type"] == "action":
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["output_dim"] = mod_meta["output_dim"]
                self.action_center.load_state_dict(arrays)
            elif mod_meta["type"] == "motor":
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["output_dim"] = mod_meta["output_dim"]
                self.motor_cortex.load_state_dict(arrays)
            else:
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["output_dim"] = mod_meta["output_dim"]
                bank_state[name] = arrays

            loaded.append(name)

        if bank_state and self.module_bank is not None:
            self.module_bank.load_state_dict(bank_state, modules=list(bank_state.keys()))
        elif bank_state and self.monolithic_policy is not None:
            monolithic_name = self.MONOLITHIC_POLICY_NAME
            if monolithic_name not in bank_state:
                raise KeyError("Save incompatível: pesos monolíticos ausentes.")
            self.monolithic_policy.load_state_dict(bank_state[monolithic_name])

        return loaded
