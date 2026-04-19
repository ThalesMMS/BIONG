from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Sequence

import numpy as np

from .ablations import BrainAblationConfig, default_brain_config
from .arbitration import (
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
from .bus import MessageBus
from .interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_DELTAS,
    ACTION_TO_INDEX,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    architecture_signature,
    interface_registry,
)
from .modules import MODULE_HIDDEN_DIMS, CorticalModuleBank, ModuleResult, ReflexDecision
from .nn import ArbitrationNetwork, MotorNetwork, ProposalNetwork, one_hot, softmax
from .noise import _compute_execution_difficulty_core
from .operational_profiles import OperationalProfile, runtime_operational_profile
from .reflexes import (
    _apply_reflex_path as apply_reflex_path,
    _direction_action as direction_action,
    _module_reflex_decision as module_reflex_decision,
)
from .world import ACTIONS


@dataclass
class BrainStep:
    module_results: List[ModuleResult]
    action_center_logits: np.ndarray
    action_center_policy: np.ndarray
    motor_correction_logits: np.ndarray
    observation: Dict[str, np.ndarray] = field(default_factory=dict)
    total_logits_without_reflex: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    total_logits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    policy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    value: float = 0.0
    action_intent_idx: int = 0
    motor_action_idx: int = 0
    action_idx: int = 0
    orientation_alignment: float = 1.0
    terrain_difficulty: float = 0.0
    momentum: float = 0.0
    execution_difficulty: float = 0.0
    execution_slip_occurred: bool = False
    motor_slip_occurred: bool = False
    motor_noise_applied: bool = False
    slip_reason: str = "none"
    motor_override: bool = False
    final_reflex_override: bool = False
    action_center_input: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    motor_input: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    policy_mode: str = "normal"
    arbitration_decision: ArbitrationDecision | None = None


class SpiderBrain:
    """Neuro-modular brain with interface-standardized locomotion proposals.

    Explicit memory remains in the world and arrives here only as named observations.
    The world remains the owner of ecological state and explicit memory; `interfaces.py`
    defines the named contracts, while `modules.py` runs only the neural proposers.
    This is where the interpretable local reflexes and final motor arbitration live.
    """

    ARCHITECTURE_VERSION = 12
    MONOLITHIC_POLICY_NAME = DEFAULT_MONOLITHIC_POLICY_NAME
    MONOLITHIC_HIDDEN_DIM = sum(MODULE_HIDDEN_DIMS.values())
    VALENCE_ORDER = DEFAULT_VALENCE_ORDER
    ARBITRATION_NETWORK_NAME = DEFAULT_ARBITRATION_NETWORK_NAME
    MODULE_VALENCE_ROLES = MappingProxyType(
        {
            "alert_center": "threat",
            "hunger_center": "hunger",
            "sleep_center": "sleep",
            "visual_cortex": "support",
            "sensory_cortex": "support",
            MONOLITHIC_POLICY_NAME: "integrated_policy",
        }
    )
    PRIORITY_GATING_WEIGHTS = DEFAULT_PRIORITY_GATING_WEIGHTS
    ARBITRATION_EVIDENCE_FIELDS = DEFAULT_ARBITRATION_EVIDENCE_FIELDS
    ARBITRATION_GATE_MODULE_ORDER = DEFAULT_ARBITRATION_GATE_MODULE_ORDER
    # Linear weights for each valence's evidence signals used by the fixed-formula
    # arbitration baseline and by the warm-start initialization of the learned network.
    VALENCE_EVIDENCE_WEIGHTS: MappingProxyType = DEFAULT_VALENCE_EVIDENCE_WEIGHTS

    def __init__(
        self,
        seed: int = 0,
        gamma: float = 0.96,
        module_lr: float = 0.010,
        motor_lr: float = 0.012,
        module_dropout: float = 0.05,
        arbitration_lr: float | None = None,
        arbitration_regularization_weight: float = 0.1,
        arbitration_valence_regularization_weight: float = 0.1,
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
            arbitration_lr (float | None): Learning rate for the learned arbitration network; defaults to `module_lr` when omitted.
            arbitration_regularization_weight (float): Strength of the gate regularizer that keeps learned final gates near the fixed priority-gate baseline. This can be annealed over training to progressively allow more deviation from the fixed baseline.
            arbitration_valence_regularization_weight (float): Optional strength of a valence-logit regularizer toward the fixed-formula valence distribution. Set to 0.0 to disable.
            config (BrainAblationConfig | None): Optional ablation/config that selects modular vs monolithic mode and may override module_dropout; when omitted a default config is created.
            operational_profile (str | OperationalProfile | None): Operational profile or its name; resolved into runtime reflex parameters (auxiliary weights, logit strengths, and thresholds) used by reflex decision logic.
        """
        self.rng = np.random.default_rng(seed)
        self.arbitration_rng = np.random.default_rng(int(seed) + 104729)
        self.action_dim = len(ACTIONS)
        self.config = config if config is not None else default_brain_config(module_dropout=module_dropout)
        self.operational_profile = runtime_operational_profile(operational_profile)
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
                recurrent_modules=self.config.recurrent_modules,
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
        arbitration_input_dim = arbitration_evidence_input_dim(
            arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
        )
        self.arbitration_network = ArbitrationNetwork(
            input_dim=arbitration_input_dim,
            hidden_dim=32,
            rng=self.arbitration_rng,
            name=self.ARBITRATION_NETWORK_NAME,
            gate_adjustment_min=self.config.gate_adjustment_bounds[0],
            gate_adjustment_max=self.config.gate_adjustment_bounds[1],
        )
        warm_start_arbitration_network(
            self.arbitration_network,
            ablation_config=self.config,
            warm_start_scale=self.config.warm_start_scale,
            valence_order=self.VALENCE_ORDER,
            arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
            valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
        )
        if self.config.name == "learned_arbitration_no_regularization":
            arbitration_regularization_weight = 0.0
            arbitration_valence_regularization_weight = 0.0
        if not self.config.use_learned_arbitration:
            arbitration_regularization_weight = 0.0
            arbitration_valence_regularization_weight = 0.0
        self.gamma = gamma
        self.module_lr = module_lr
        self.arbitration_lr = module_lr if arbitration_lr is None else float(arbitration_lr)
        self.arbitration_regularization_weight = float(arbitration_regularization_weight)
        self.arbitration_valence_regularization_weight = float(arbitration_valence_regularization_weight)
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
        Restore the runtime reflex scale to the configured default.
        
        Clamps the configured reflex scale to a minimum of 0.0 and sets self.current_reflex_scale accordingly. Raises ValueError if the configured reflex scale is not finite.
        """
        value = float(self.config.reflex_scale)
        if not math.isfinite(value):
            raise ValueError("non-finite reflex scale")
        self.current_reflex_scale = max(0.0, value)

    def reset_hidden_states(self) -> None:
        """
        Reset recurrent hidden state for all modules owned by this brain.
        
        If a module bank is present, delegates to its reset_hidden_states(); otherwise this is a no-op.
        """
        if self.module_bank is not None:
            self.module_bank.reset_hidden_states()

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
            module_results (List[ModuleResult]): Per-module proposal results whose gated logits, when present, are concatenated in proposal order.
            observation (Dict[str, np.ndarray]): Full observation mapping; the action context is taken from the key defined by ACTION_CONTEXT_INTERFACE.observation_key and bound/flattened via ACTION_CONTEXT_INTERFACE.

        Returns:
            np.ndarray: 1-D array formed by concatenating all action-path module logits followed by the action context vector.
        """
        logits_flat = np.concatenate(
            [
                result.gated_logits if result.gated_logits is not None else result.logits
                for result in module_results
            ],
            axis=0,
        )
        action_context_mapping = self._bound_action_context(observation)
        action_context = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_context_mapping)
        return np.concatenate([logits_flat, action_context], axis=0)

    def _compute_counterfactual_credit(
        self,
        module_results: List[ModuleResult],
        observation: Dict[str, np.ndarray],
        action_idx: int,
    ) -> Dict[str, float]:
        """
        Estimate per-module credit by masking each proposal out of the action-center path.

        Each counterfactual keeps all proposal logits except one module's slice,
        which is replaced with zeros before the action center is evaluated
        without touching its training cache. Importance is the signed change in
        selected-action probability; absolute magnitudes are normalized so
        interfering modules still receive proportional credit.
        """
        if not module_results:
            return {}
        action_idx = int(action_idx)
        action_context_mapping = self._bound_action_context(observation)
        action_context = ACTION_CONTEXT_INTERFACE.vector_from_mapping(action_context_mapping)
        logits_by_module = [
            np.asarray(
                result.gated_logits if result.gated_logits is not None else result.logits,
                dtype=float,
            )
            for result in module_results
        ]
        logits_flat = np.concatenate(logits_by_module, axis=0)
        actual_proposal_sum = np.sum(np.stack(logits_by_module, axis=0), axis=0)
        actual_correction_logits, _ = self.action_center.forward(
            np.concatenate([logits_flat, action_context], axis=0),
            store_cache=False,
        )
        actual_policy = softmax(actual_proposal_sum + actual_correction_logits)

        raw_importance: Dict[str, float] = {}
        for module_index, result in enumerate(module_results):
            if not result.active:
                raw_importance[result.name] = 0.0
                continue
            counterfactual_logits_flat = logits_flat.copy()
            start = module_index * self.action_dim
            stop = start + self.action_dim
            counterfactual_logits_flat[start:stop] = 0.0
            counterfactual_correction_logits, _ = self.action_center.forward(
                np.concatenate([counterfactual_logits_flat, action_context], axis=0),
                store_cache=False,
            )
            counterfactual_policy = softmax(
                actual_proposal_sum
                - logits_by_module[module_index]
                + counterfactual_correction_logits
            )
            raw_importance[result.name] = float(
                actual_policy[action_idx] - counterfactual_policy[action_idx]
            )

        total = sum(abs(v) for v in raw_importance.values())
        if total > 1e-8:
            return {name: abs(value) / total for name, value in raw_importance.items()}
        active_results = [r for r in module_results if r.active]
        pool = active_results or module_results
        uniform = 1.0 / len(pool)
        pool_names = {r.name for r in pool}
        return {r.name: (uniform if r.name in pool_names else 0.0) for r in module_results}

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
                f"action_intent expected shape {(self.action_dim,)}, received {intent.shape}."
            )
        motor_context_mapping = MOTOR_CONTEXT_INTERFACE.bind_values(
            observation[MOTOR_CONTEXT_INTERFACE.observation_key]
        )
        motor_context = MOTOR_CONTEXT_INTERFACE.vector_from_mapping(motor_context_mapping)
        return np.concatenate([intent, motor_context], axis=0)

    def _motor_execution_diagnostics(
        self,
        observation: Dict[str, np.ndarray],
        action_idx: int,
    ) -> Dict[str, float]:
        """Preview physical execution difficulty from motor context and the selected action."""
        sanitized_obs = np.nan_to_num(
            np.asarray(observation[MOTOR_CONTEXT_INTERFACE.observation_key], dtype=float),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        motor_context = MOTOR_CONTEXT_INTERFACE.bind_values(sanitized_obs)
        action_name = ACTIONS[int(action_idx)]
        move_dx, move_dy = ACTION_DELTAS.get(action_name, (0, 0))
        heading_dx = float(motor_context.get("heading_dx", 0.0))
        heading_dy = float(motor_context.get("heading_dy", 0.0))
        terrain_difficulty = float(
            np.clip(float(motor_context.get("terrain_difficulty", 0.0)), 0.0, 1.0)
        )
        fatigue = float(np.clip(float(motor_context.get("fatigue", 0.0)), 0.0, 1.0))
        momentum = float(np.clip(float(motor_context.get("momentum", 0.0)), 0.0, 1.0))
        execution_difficulty, components = _compute_execution_difficulty_core(
            (heading_dx, heading_dy),
            (float(move_dx), float(move_dy)),
            terrain_difficulty=terrain_difficulty,
            fatigue=fatigue,
            momentum=momentum,
        )
        return {
            "orientation_alignment": float(components["orientation_alignment"]),
            "terrain_difficulty": float(components["terrain_difficulty"]),
            "momentum": float(components["momentum"]),
            "execution_difficulty": execution_difficulty,
        }

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
                raise RuntimeError("Module bank unavailable for modular architecture.")
            return self.module_bank.forward(
                observation,
                store_cache=store_cache,
                training=training,
            )

        if self.monolithic_policy is None:
            raise RuntimeError("Monolithic network unavailable for the configured architecture.")
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

    def _bound_observation(
        self,
        interface_name: str,
        observation: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Bind the named interface's raw observation and sanitize its values into bounded scalar evidence.
        
        Parameters:
            interface_name (str): Interface identifier present in MODULE_INTERFACE_BY_NAME; the function looks up that interface and uses its observation_key to fetch data from `observation`.
            observation (Dict[str, np.ndarray]): Full observation mapping from observation keys to arrays.
        
        Returns:
            Dict[str, float]: The interface's bound observation fields as finite float values; NaN values are converted to 0.0, +inf to 1.0, and -inf to -1.0 before binding.
        
        Raises:
            KeyError: If no interface with the given name exists.
        """
        interface = MODULE_INTERFACE_BY_NAME.get(interface_name)
        if interface is None:
            raise KeyError(f"Unknown interface: {interface_name}")
        sanitized_obs = np.nan_to_num(
            np.asarray(observation[interface.observation_key], dtype=float),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return interface.bind_values(sanitized_obs)

    def _bound_action_context(self, observation: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Bind and sanitize the action context observation into finite scalar values."""
        sanitized_obs = np.nan_to_num(
            np.asarray(observation[ACTION_CONTEXT_INTERFACE.observation_key], dtype=float),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return ACTION_CONTEXT_INTERFACE.bind_values(sanitized_obs)

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
        module_results = self._proposal_results(
            observation,
            store_cache=store_cache,
            training=training_mode,
        )
        arbitration_without_reflex = compute_arbitration(
            observation,
            module_results,
            arbitration_network=self.arbitration_network,
            ablation_config=self.config,
            arbitration_rng=self.arbitration_rng,
            operational_profile=self.operational_profile,
            training=False,
            store_cache=False,
            clamp_fn=clamp_unit,
            valence_order=self.VALENCE_ORDER,
            arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
            valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
            arbitration_gate_module_order=self.ARBITRATION_GATE_MODULE_ORDER,
            priority_gating_weights=self.PRIORITY_GATING_WEIGHTS,
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

        apply_reflex_path(
            module_results,
            ablation_config=self.config,
            operational_profile=self.operational_profile,
            interface_registry=self._interface_registry(),
            current_reflex_scale=self.current_reflex_scale,
            module_valence_roles=self.MODULE_VALENCE_ROLES,
        )
        arbitration = compute_arbitration(
            observation,
            module_results,
            arbitration_network=self.arbitration_network,
            ablation_config=self.config,
            arbitration_rng=self.arbitration_rng,
            operational_profile=self.operational_profile,
            training=training_mode,
            store_cache=store_cache,
            clamp_fn=clamp_unit,
            valence_order=self.VALENCE_ORDER,
            arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
            valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
            arbitration_gate_module_order=self.ARBITRATION_GATE_MODULE_ORDER,
            priority_gating_weights=self.PRIORITY_GATING_WEIGHTS,
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
            arbitration_payload = arbitration.to_payload()
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
            arbitration = compute_arbitration(
                observation,
                module_results,
                arbitration_network=self.arbitration_network,
                ablation_config=self.config,
                arbitration_rng=self.arbitration_rng,
                operational_profile=self.operational_profile,
                training=False,
                store_cache=False,
                clamp_fn=clamp_unit,
                valence_order=self.VALENCE_ORDER,
                arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
                valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
                arbitration_gate_module_order=self.ARBITRATION_GATE_MODULE_ORDER,
                priority_gating_weights=self.PRIORITY_GATING_WEIGHTS,
            )
            apply_priority_gating(
                module_results,
                arbitration,
                module_valence_roles=self.MODULE_VALENCE_ROLES,
            )
            action_input = self._build_action_input(module_results, observation)
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
            return [spec.name for spec in self.module_bank.specs]
        return [self.MONOLITHIC_POLICY_NAME]

    def _architecture_signature(self) -> dict[str, object]:
        """
        Compute the runtime architecture signature for the active proposal backend.
        
        Returns:
            signature (dict): A mapping describing architecture identifiers and configuration used for compatibility/fingerprinting (includes proposal backend name and order, whether learned arbitration is enabled, and arbitration network input/hidden dims and regularization weight).
        """
        return architecture_signature(
            proposal_backend=self.config.architecture,
            proposal_order=self._proposal_stage_names(),
            learned_arbitration=self.config.use_learned_arbitration,
            arbitration_input_dim=self.arbitration_network.input_dim,
            arbitration_hidden_dim=self.arbitration_network.hidden_dim,
            arbitration_regularization_weight=self.arbitration_regularization_weight,
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

    def learn(self, decision: BrainStep, reward: float, next_observation: Dict[str, np.ndarray], done: bool) -> Dict[str, object]:
        """
        Perform a temporal-difference policy-gradient update that trains the policy, proposal modules (or monolithic proposal), and motor/value networks.
        
        Parameters:
            decision (BrainStep): Recorded action decision containing module results, selected action index, policy, and controller outputs.
            reward (float): Observed scalar reward following the decision.
            next_observation (Dict[str, np.ndarray]): Observation after the action used to estimate the next state's value.
            done (bool): Whether the episode terminated after the action.
        
        Returns:
            metrics (Dict[str, object]): Training diagnostics with the following keys:
                - "reward": the provided reward.
                - "td_target": computed TD target (reward + gamma * next_value).
                - "td_error": clipped advantage used for the policy update.
                - "value": the value estimate from the provided decision.
                - "next_value": the estimated value of the next observation (0.0 if done).
                - "entropy": policy entropy computed from decision.policy.
                - "aux_modules": number of modules that received auxiliary gradients.
                - "credit_strategy": configured module-credit strategy.
                - "module_credit_weights": per-module policy-credit weights.
                - "module_gradient_norms": per-module total-gradient L2 norms.
                - "counterfactual_credit_weights": counterfactual weights when
                  enabled, otherwise an empty dict.
        """
        if decision.policy_mode != "normal":
            raise ValueError(
                "learn() only supports decisions produced with policy_mode='normal'."
            )
        next_value = 0.0 if done else self.estimate_value(next_observation)
        td_target = reward + self.gamma * next_value
        advantage = float(np.clip(td_target - decision.value, -4.0, 4.0))
        grad_policy_logits = advantage * (decision.policy - one_hot(decision.action_idx, self.action_dim))
        effective_credit_strategy = self.config.credit_strategy
        uses_counterfactual_credit = self.config.uses_counterfactual_credit
        uses_local_credit_only = self.config.uses_local_credit_only
        if self.config.is_monolithic and (uses_counterfactual_credit or uses_local_credit_only):
            # A monolithic policy has no per-module proposal path, so diagnostics
            # follow the broadcast-style update that is actually applied below.
            effective_credit_strategy = "broadcast"
            uses_counterfactual_credit = False
            uses_local_credit_only = False
        module_credit_weights = {
            result.name: (1.0 if result.active else 0.0)
            for result in decision.module_results
        }
        counterfactual_credit_weights: Dict[str, float] = {}
        if uses_counterfactual_credit:
            if not decision.observation:
                raise ValueError(
                    "counterfactual credit requires BrainStep.observation to be populated."
                )
            counterfactual_credit_weights = self._compute_counterfactual_credit(
                decision.module_results,
                decision.observation,
                decision.action_idx,
            )
            module_credit_weights = dict(counterfactual_credit_weights)
        elif uses_local_credit_only:
            module_credit_weights = {
                result.name: 0.0
                for result in decision.module_results
            }
        reflex_aux_grads = {
            name: np.asarray(grad, dtype=float).copy()
            for name, grad in self._auxiliary_module_gradients(decision.module_results).items()
        }
        action_center_value_grad = decision.value - td_target
        action_center_input_grads = self.action_center.backward(
            grad_policy_logits=grad_policy_logits,
            grad_value=action_center_value_grad,
            lr=self.motor_lr,
        )
        proposal_grad_width = self.action_dim * len(decision.module_results)
        proposal_input_grads = np.asarray(
            action_center_input_grads[:proposal_grad_width],
            dtype=float,
        )
        per_result_input_grads = proposal_input_grads.reshape(
            len(decision.module_results),
            self.action_dim,
        )

        arbitration = decision.arbitration_decision
        arbitration_grad_valence_norm = 0.0
        arbitration_grad_gate_norm = 0.0
        arbitration_gate_regularization_norm = 0.0
        arbitration_valence_regularization_norm = 0.0
        arbitration_value_grad = 0.0
        arbitration_loss = 0.0
        regularization_loss = 0.0
        gate_adjustment_magnitude = 0.0
        if arbitration is not None and arbitration.gate_adjustments:
            gate_adjustment_magnitude = float(
                np.mean(
                    [
                        abs(float(adjustment) - 1.0)
                        for adjustment in arbitration.gate_adjustments.values()
                    ]
                )
            )
        if (
            arbitration is not None
            and arbitration.learned_adjustment
            and self.config.use_learned_arbitration
            and self.arbitration_network.cache is not None
        ):
            valence_probs = np.array(
                [
                    float(arbitration.valence_scores.get(name, 0.0))
                    for name in self.VALENCE_ORDER
                ],
                dtype=float,
            )
            winning_valence_idx = self.VALENCE_ORDER.index(arbitration.winning_valence)
            grad_valence = advantage * (
                valence_probs - one_hot(winning_valence_idx, len(self.VALENCE_ORDER))
            )

            valence_reg = np.zeros_like(grad_valence)
            valence_regularization_loss = 0.0
            if self.arbitration_valence_regularization_weight > 0.0:
                fixed_valence_targets = fixed_formula_valence_scores_from_evidence(
                    arbitration.evidence,
                    valence_order=self.VALENCE_ORDER,
                    valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
                    clamp_fn=clamp_unit,
                )
                valence_reg = (
                    self.arbitration_valence_regularization_weight
                    * (valence_probs - fixed_valence_targets)
                )
                grad_valence += valence_reg
                valence_regularization_loss = float(
                    0.5
                    * self.arbitration_valence_regularization_weight
                    * np.sum((valence_probs - fixed_valence_targets) ** 2)
                )

            grad_final_gates = np.zeros(
                len(self.ARBITRATION_GATE_MODULE_ORDER),
                dtype=float,
            )
            gate_regularization = np.zeros_like(grad_final_gates)
            gate_deviation = np.zeros_like(grad_final_gates)
            base_gate_by_index = np.zeros_like(grad_final_gates)
            module_gate_indices = {
                name: index
                for index, name in enumerate(self.ARBITRATION_GATE_MODULE_ORDER)
            }
            for result, extra_grad in zip(decision.module_results, per_result_input_grads):
                gate_index = module_gate_indices.get(result.name)
                if gate_index is None:
                    continue
                pre_gate_logits = (
                    result.post_reflex_logits
                    if result.post_reflex_logits is not None
                    else result.gated_logits
                )
                if pre_gate_logits is None:
                    continue
                base_gate = float(arbitration.base_gates.get(result.name, 0.0))
                learned_gate = float(arbitration.module_gates.get(result.name, base_gate))
                base_gate_by_index[gate_index] = base_gate
                gate_deviation[gate_index] = learned_gate - base_gate
                grad_final_gates[gate_index] += float(
                    np.dot(
                        np.asarray(extra_grad, dtype=float),
                        np.asarray(pre_gate_logits, dtype=float),
                    )
                )
                gate_regularization[gate_index] += float(
                    self.arbitration_regularization_weight
                    * (learned_gate - base_gate)
                )

            # action_center_input_grads are gradients with respect to gated
            # logits. Project through gated_logits_i = gate_i * raw_logits_i,
            # then through gate_i ~= base_gate_i * learned_adjustment_i. This
            # uses the requested first-order approximation and ignores only the
            # final diagnostic [0, 1] clamp.
            grad_gates = base_gate_by_index * (grad_final_gates + gate_regularization)
            arbitration_value_grad = float(arbitration.arbitration_value - td_target)
            gate_regularization_loss = float(
                0.5
                * self.arbitration_regularization_weight
                * np.sum(gate_deviation**2)
            )
            regularization_loss = gate_regularization_loss + valence_regularization_loss
            selected_valence_prob = max(float(valence_probs[winning_valence_idx]), 1e-8)
            arbitration_policy_loss = float(-advantage * np.log(selected_valence_prob))
            arbitration_value_loss = float(0.5 * arbitration_value_grad * arbitration_value_grad)
            arbitration_loss = arbitration_policy_loss + arbitration_value_loss + regularization_loss
            # Keep the warm-start valence and gate behavior stable in proportion
            # to the baseline regularizer without fully freezing the default.
            anchor = float(np.clip(self.arbitration_regularization_weight, 0.0, 1.0))
            _anchored_param_names = ("W1", "b1", "W2_valence", "b2_valence", "W2_gate", "b2_gate")
            anchored_params: dict[str, np.ndarray] = (
                {name: getattr(self.arbitration_network, name).copy() for name in _anchored_param_names}
                if anchor > 0.0
                else {}
            )
            self.arbitration_network.backward(
                grad_valence_logits=grad_valence,
                grad_gate_adjustments=grad_gates,
                grad_value=arbitration_value_grad,
                lr=self.arbitration_lr,
            )
            for param_name, pre_update in anchored_params.items():
                post_update = getattr(self.arbitration_network, param_name)
                setattr(
                    self.arbitration_network,
                    param_name,
                    anchor * pre_update + (1.0 - anchor) * post_update,
                )
            arbitration_grad_valence_norm = float(np.linalg.norm(grad_valence))
            arbitration_grad_gate_norm = float(np.linalg.norm(grad_gates))
            arbitration_gate_regularization_norm = float(np.linalg.norm(gate_regularization))
            arbitration_valence_regularization_norm = float(np.linalg.norm(valence_reg))

        module_total_grads: Dict[str, np.ndarray] = {}
        module_gradient_norms: Dict[str, float] = {}
        if self.config.is_modular:
            if self.module_bank is None:
                raise RuntimeError("Module bank unavailable for modular architecture.")
            for result, extra_grad in zip(decision.module_results, per_result_input_grads):
                if not result.active:
                    continue
                gate_weight = float(result.gate_weight)
                total_grad = gate_weight * np.asarray(
                    reflex_aux_grads.get(
                        result.name,
                        np.zeros(self.action_dim, dtype=float),
                    ),
                    dtype=float,
                )
                if uses_counterfactual_credit:
                    cf_weight = float(counterfactual_credit_weights.get(result.name, 0.0))
                    total_grad += gate_weight * cf_weight * grad_policy_logits
                elif not uses_local_credit_only:
                    total_grad += gate_weight * grad_policy_logits
                total_grad += gate_weight * np.asarray(extra_grad, dtype=float)
                module_total_grads[result.name] = module_total_grads.get(
                    result.name,
                    np.zeros(self.action_dim, dtype=float),
                ) + total_grad
                module_gradient_norms[result.name] = float(np.linalg.norm(total_grad))
            self.module_bank.backward(
                np.zeros(self.action_dim, dtype=float),
                lr=self.module_lr,
                aux_grads=module_total_grads,
            )
        else:
            if self.monolithic_policy is None:
                raise RuntimeError("Monolithic network unavailable for the configured architecture.")
            grad_for_monolithic = grad_policy_logits + proposal_input_grads
            module_gradient_norms[self.MONOLITHIC_POLICY_NAME] = float(
                np.linalg.norm(grad_for_monolithic)
            )
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
            "aux_modules": float(len(reflex_aux_grads)),
            "arbitration_value": float(arbitration.arbitration_value) if arbitration is not None else 0.0,
            "arbitration_value_grad": float(arbitration_value_grad),
            "arbitration_grad_valence_norm": arbitration_grad_valence_norm,
            "arbitration_grad_gate_norm": arbitration_grad_gate_norm,
            "arbitration_gate_regularization_norm": arbitration_gate_regularization_norm,
            "arbitration_valence_regularization_norm": arbitration_valence_regularization_norm,
            "arbitration_loss": float(arbitration_loss),
            "gate_adjustment_magnitude": float(gate_adjustment_magnitude),
            "regularization_loss": float(regularization_loss),
            "credit_strategy": effective_credit_strategy,
            "module_credit_weights": {
                name: float(value)
                for name, value in sorted(module_credit_weights.items())
            },
            "module_gradient_norms": {
                name: float(value)
                for name, value in sorted(module_gradient_norms.items())
            },
            "counterfactual_credit_weights": {
                name: float(value)
                for name, value in sorted(counterfactual_credit_weights.items())
            },
        }

    def parameter_norms(self) -> Dict[str, float]:
        """
        Compute L2 norms of parameters for each trainable network component.
        
        Returns:
            norms (Dict[str, float]): Mapping from component name to its L2 parameter norm. Includes per-module norms when a modular bank is present, `"monolithic_policy"` when a monolithic policy is present, and always includes `"arbitration_network"`, `"action_center"`, and `"motor_cortex"`.
        """
        norms: Dict[str, float] = {}
        if self.module_bank is not None:
            norms.update(self.module_bank.parameter_norms())
        if self.monolithic_policy is not None:
            norms[self.MONOLITHIC_POLICY_NAME] = self.monolithic_policy.parameter_norm()
        norms[self.ARBITRATION_NETWORK_NAME] = self.arbitration_network.parameter_norm()
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
            return [
                spec.name for spec in self.module_bank.specs
            ] + [self.ARBITRATION_NETWORK_NAME, "action_center", "motor_cortex"]
        return [self.MONOLITHIC_POLICY_NAME, self.ARBITRATION_NETWORK_NAME, "action_center", "motor_cortex"]

    def save(self, directory: str | Path) -> Path:
        """
        Persist the brain's trainable weights and configuration metadata to the specified directory.
        
        Saves per-module proposal weights (one .npz per module or a single monolithic policy .npz), the arbitration network .npz, action_center and motor_cortex .npz files, and a JSON metadata file ("metadata.json") containing architecture/version/fingerprint, interface registry, ablation and operational profile summaries, learning rates and related hyperparameters, action dimensionality, and per-module shape/type metadata.
        
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
            "arbitration_lr": self.arbitration_lr,
            "arbitration_regularization_weight": self.arbitration_regularization_weight,
            "arbitration_valence_regularization_weight": self.arbitration_valence_regularization_weight,
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

        arbitration_sd = self.arbitration_network.state_dict()
        arbitration_arrays = {
            k: (v if isinstance(v, np.ndarray) else np.asarray(v))
            for k, v in arbitration_sd.items()
        }
        np.savez(directory / f"{self.ARBITRATION_NETWORK_NAME}.npz", **arbitration_arrays)
        metadata["modules"][self.ARBITRATION_NETWORK_NAME] = {
            "type": "arbitration",
            "input_dim": arbitration_sd["input_dim"],
            "hidden_dim": arbitration_sd["hidden_dim"],
            "valence_dim": arbitration_sd["valence_dim"],
            "gate_dim": arbitration_sd["gate_dim"],
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
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        saved_architecture_version = metadata.get("architecture_version")
        if saved_architecture_version != self.ARCHITECTURE_VERSION:
            raise ValueError(
                "The saved brain uses an architecture version incompatible with the current "
                f"simulator version (save={saved_architecture_version}, current={self.ARCHITECTURE_VERSION})."
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
                "The saved brain uses an interface registry incompatible with the current registry. "
                f"(save_registry_fingerprint={saved_registry_fingerprint}, "
                f"current={expected_architecture.get('registry_fingerprint')}). "
                "There is no automatic checkpoint migration for contract changes."
            )
        if saved_architecture != expected_architecture:
            raise ValueError(
                "The saved brain uses an architecture signature incompatible with the current topology "
                f"(save_fingerprint={metadata.get('architecture_fingerprint')}, "
                f"current={self._architecture_fingerprint()}). "
                "There is no automatic checkpoint migration for this change; train or load "
                "a brain compatible with the current simulator version."
            )
        saved_config = metadata.get("ablation_config")
        expected_config = self.config.to_summary()
        if saved_config != expected_config:
            raise ValueError(
                "The saved brain uses an ablation configuration different from the current configuration."
            )
        saved_operational_profile = metadata.get("operational_profile")
        expected_operational_profile = self.operational_profile.to_summary()
        if saved_operational_profile != expected_operational_profile:
            raise ValueError(
                "The saved brain uses an operational profile different from the current operational profile."
            )

        saved_modules = metadata.get("modules", {})
        if not isinstance(saved_modules, dict):
            raise ValueError("The saved brain metadata has an invalid modules section.")
        requested_module_set = (
            None
            if modules is None
            else {str(name) for name in modules}
        )
        needs_arbitration_weights = (
            requested_module_set is None
            or self.ARBITRATION_NETWORK_NAME in requested_module_set
        )
        arbitration_npz_path = directory / f"{self.ARBITRATION_NETWORK_NAME}.npz"
        if needs_arbitration_weights:
            if self.ARBITRATION_NETWORK_NAME not in saved_modules:
                raise FileNotFoundError(
                    "Checkpoint is missing arbitration_network metadata. "
                    "Older checkpoints must be migrated or retrained for this architecture."
                )
            if not arbitration_npz_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint is missing arbitration weights: {arbitration_npz_path}"
                )

        available = list(saved_modules.keys())
        targets = list(modules) if modules is not None else available

        loaded: List[str] = []
        bank_state: Dict[str, dict] = {}

        for name in targets:
            if name not in available:
                raise KeyError(
                    f"Module '{name}' not found in the save. "
                    f"Available: {available}"
                )
            npz_path = directory / f"{name}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Weight file not found: {npz_path}")

            mod_meta = saved_modules[name]
            arrays = dict(np.load(npz_path))

            if mod_meta["type"] == "action":
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["output_dim"] = mod_meta["output_dim"]
                self.action_center.load_state_dict(arrays)
            elif mod_meta["type"] == "arbitration":
                arrays["name"] = name
                arrays["input_dim"] = mod_meta["input_dim"]
                arrays["hidden_dim"] = mod_meta["hidden_dim"]
                arrays["valence_dim"] = mod_meta["valence_dim"]
                arrays["gate_dim"] = mod_meta["gate_dim"]
                self.arbitration_network.load_state_dict(arrays)
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
                raise KeyError("Incompatible save: monolithic weights are missing.")
            self.monolithic_policy.load_state_dict(bank_state[monolithic_name])

        return loaded


def _spiderbrain_clamp_unit(value: float) -> float:
    return clamp_unit(value)


def _spiderbrain_warm_start_arbitration_network(
    self: SpiderBrain,
    warm_start_scale: float | None = None,
) -> None:
    warm_start_arbitration_network(
        self.arbitration_network,
        ablation_config=self.config,
        warm_start_scale=warm_start_scale,
        valence_order=self.VALENCE_ORDER,
        arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
        valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
    )


def _spiderbrain_fixed_formula_valence_scores_from_evidence(
    self: SpiderBrain,
    evidence: Dict[str, Dict[str, float]],
) -> np.ndarray:
    return fixed_formula_valence_scores_from_evidence(
        evidence,
        valence_order=self.VALENCE_ORDER,
        valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
        clamp_fn=clamp_unit,
    )


def _spiderbrain_compute_arbitration(
    self: SpiderBrain,
    module_results: List[ModuleResult],
    observation: Dict[str, np.ndarray],
    *,
    training: bool = False,
    store_cache: bool = True,
) -> ArbitrationDecision:
    return compute_arbitration(
        observation,
        module_results,
        arbitration_network=self.arbitration_network,
        ablation_config=self.config,
        arbitration_rng=self.arbitration_rng,
        operational_profile=self.operational_profile,
        training=training,
        store_cache=store_cache,
        clamp_fn=clamp_unit,
        valence_order=self.VALENCE_ORDER,
        arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
        valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
        arbitration_gate_module_order=self.ARBITRATION_GATE_MODULE_ORDER,
        priority_gating_weights=self.PRIORITY_GATING_WEIGHTS,
    )


def _spiderbrain_module_reflex_decision(
    self: SpiderBrain,
    result: ModuleResult,
) -> ReflexDecision | None:
    return module_reflex_decision(
        result.name,
        result.named_observation(),
        operational_profile=self.operational_profile,
        interface_registry=self._interface_registry(),
    )


SpiderBrain._clamp_unit = staticmethod(_spiderbrain_clamp_unit)
SpiderBrain._warm_start_arbitration_network = _spiderbrain_warm_start_arbitration_network
SpiderBrain._fixed_formula_valence_scores_from_evidence = (
    _spiderbrain_fixed_formula_valence_scores_from_evidence
)
SpiderBrain._compute_arbitration = _spiderbrain_compute_arbitration
SpiderBrain._module_reflex_decision = _spiderbrain_module_reflex_decision
