from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .ablations import BrainAblationConfig, default_brain_config
from .bus import MessageBus
from .interfaces import (
    ACTION_CONTEXT_INTERFACE,
    ACTION_TO_INDEX,
    MODULE_INTERFACE_BY_NAME,
    MODULE_INTERFACES,
    MOTOR_CONTEXT_INTERFACE,
    architecture_signature,
    interface_registry,
)
from .modules import MODULE_HIDDEN_DIMS, CorticalModuleBank, ModuleResult, ReflexDecision
from .nn import MotorNetwork, ProposalNetwork, one_hot, softmax
from .operational_profiles import OperationalProfile, resolve_operational_profile
from .world import ACTIONS


@dataclass(frozen=True)
class ValenceScore:
    name: str
    score: float
    evidence: Dict[str, float]

    def to_payload(self) -> Dict[str, object]:
        """
        Serialize the valence score into a JSON-serializable payload with rounded numeric values.
        
        Returns:
            payload (Dict[str, object]): Dictionary with keys:
                - "name": the valence name (str).
                - "score": the score rounded to six decimal places (float).
                - "evidence": a dict mapping each evidence key (sorted) to its value rounded to six decimal places (float).
        """
        return {
            "name": self.name,
            "score": round(float(self.score), 6),
            "evidence": {
                key: round(float(value), 6)
                for key, value in sorted(self.evidence.items())
            },
        }


@dataclass(frozen=True)
class ArbitrationDecision:
    strategy: str
    winning_valence: str
    valence_scores: Dict[str, float]
    module_gates: Dict[str, float]
    suppressed_modules: List[str]
    evidence: Dict[str, Dict[str, float]]
    intent_before_gating_idx: int
    intent_after_gating_idx: int
    module_contribution_share: Dict[str, float] = field(default_factory=dict)
    dominant_module: str = ""
    dominant_module_share: float = 0.0
    effective_module_count: float = 0.0
    module_agreement_rate: float = 0.0
    module_disagreement_rate: float = 0.0

    def to_payload(self) -> Dict[str, object]:
        """
        Convert the arbitration decision into a JSON-serializable dictionary suitable for telemetry or storage.

        The returned payload includes the arbitration strategy and winning valence, per-valence normalized scores and per-module gate weights (both rounded to six decimal places), the list of suppressed module names, structured evidence dictionaries with values rounded to six decimal places, and the action names corresponding to the intent indices before and after gating.
        
        Returns:
            payload (Dict[str, object]): Dictionary with keys:
                - "strategy": arbitration strategy name.
                - "winning_valence": name of the selected valence.
                - "valence_scores": mapping of valence name -> score (rounded to 6 decimals).
                - "module_gates": mapping of module name -> gate weight (rounded to 6 decimals).
                - "module_contribution_share": mapping of module name -> normalized contribution share.
                - "dominant_module": name of the dominant proposal source.
                - "dominant_module_share": dominant proposal contribution share.
                - "effective_module_count": effective number of active proposal sources.
                - "module_agreement_rate": fraction of active modules whose gated argmax matches the final intent.
                - "module_disagreement_rate": complement of module_agreement_rate.
                - "suppressed_modules": list of suppressed module names.
                - "evidence": mapping of valence name -> evidence dict (each value rounded to 6 decimals).
                - "intent_before_gating": action name for the intent index before gating.
                - "intent_after_gating": action name for the intent index after gating.
        """
        return {
            "strategy": self.strategy,
            "winning_valence": self.winning_valence,
            "valence_scores": {
                key: round(float(value), 6)
                for key, value in sorted(self.valence_scores.items())
            },
            "module_gates": {
                key: round(float(value), 6)
                for key, value in sorted(self.module_gates.items())
            },
            "module_contribution_share": {
                key: round(float(value), 6)
                for key, value in sorted(self.module_contribution_share.items())
            },
            "dominant_module": self.dominant_module,
            "dominant_module_share": round(float(self.dominant_module_share), 6),
            "effective_module_count": round(float(self.effective_module_count), 6),
            "module_agreement_rate": round(float(self.module_agreement_rate), 6),
            "module_disagreement_rate": round(float(self.module_disagreement_rate), 6),
            "suppressed_modules": list(self.suppressed_modules),
            "evidence": {
                key: {
                    inner_key: round(float(inner_value), 6)
                    for inner_key, inner_value in sorted(values.items())
                }
                for key, values in sorted(self.evidence.items())
            },
            "intent_before_gating": ACTIONS[self.intent_before_gating_idx],
            "intent_after_gating": ACTIONS[self.intent_after_gating_idx],
        }


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
    arbitration_decision: ArbitrationDecision | None = None


class SpiderBrain:
    """Neuro-modular brain with interface-standardized locomotion proposals.

    Explicit memory remains in the world and arrives here only as named observations.
    The world remains the owner of ecological state and explicit memory; `interfaces.py`
    defines the named contracts, while `modules.py` runs only the neural proposers.
    This is where the interpretable local reflexes and final motor arbitration live.
    """

    ARCHITECTURE_VERSION = 9
    MONOLITHIC_POLICY_NAME = "monolithic_policy"
    MONOLITHIC_HIDDEN_DIM = sum(MODULE_HIDDEN_DIMS.values())
    VALENCE_ORDER = ("threat", "hunger", "sleep", "exploration")
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
    PRIORITY_GATING_WEIGHTS = MappingProxyType(
        {
            "threat": MappingProxyType(
                {
                    "alert_center": 1.0,
                    "hunger_center": 0.18,
                    "sleep_center": 0.14,
                    "visual_cortex": 0.58,
                    "sensory_cortex": 0.68,
                    MONOLITHIC_POLICY_NAME: 1.0,
                }
            ),
            "hunger": MappingProxyType(
                {
                    "alert_center": 0.32,
                    "hunger_center": 1.0,
                    "sleep_center": 0.34,
                    "visual_cortex": 0.74,
                    "sensory_cortex": 0.7,
                    MONOLITHIC_POLICY_NAME: 1.0,
                }
            ),
            "sleep": MappingProxyType(
                {
                    "alert_center": 0.3,
                    "hunger_center": 0.24,
                    "sleep_center": 1.0,
                    "visual_cortex": 0.48,
                    "sensory_cortex": 0.56,
                    MONOLITHIC_POLICY_NAME: 1.0,
                }
            ),
            "exploration": MappingProxyType(
                {
                    "alert_center": 0.42,
                    "hunger_center": 0.55,
                    "sleep_center": 0.55,
                    "visual_cortex": 0.96,
                    "sensory_cortex": 0.92,
                    MONOLITHIC_POLICY_NAME: 1.0,
                }
            ),
        }
    )

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
        action_context_mapping = self._bound_action_context(observation)
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
                f"action_intent expected shape {(self.action_dim,)}, received {intent.shape}."
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

    def _apply_reflex_path(self, module_results: List[ModuleResult]) -> None:
        """
        Apply per-module reflex decisions to the provided ModuleResult objects in place.
        
        If reflexes are enabled and the brain is modular, compute each active module's reflex decision (if any), scale its strengths, and apply reflex-induced changes to that module's logits and derived diagnostics. If reflexes are disabled or the architecture is not modular, initialize the same reflex-related bookkeeping fields without changing logits.
        
        The following ModuleResult fields are written/updated:
        - neural_logits: copy of logits before any reflex is applied
        - reflex_delta_logits: reflex-induced additive logit delta
        - post_reflex_logits: neural_logits + reflex_delta_logits
        - logits, probs: updated to reflect post_reflex_logits when a reflex is applied (otherwise unchanged)
        - reflex: the ReflexDecision object or None
        - reflex_applied: `true` if a non-trivial reflex delta was applied
        - effective_reflex_scale: scalar applied to reflex strengths for this module
        - module_reflex_override: `true` if reflex changed the module argmax
        - module_reflex_dominance: relative L1 magnitude of reflex delta versus total absolute logits (small epsilon added)
        - valence_role: semantic role for arbitration (from MODULE_VALENCE_ROLES or "support")
        - gate_weight: default gate weight (left as 1.0 here)
        - gated_logits: copy of logits after reflex application (or original logits when no reflex)
        
        No value is returned; the function mutates the ModuleResult instances in the input list.
        """
        for result in module_results:
            result.neural_logits = result.logits.copy()
            result.reflex_delta_logits = np.zeros_like(result.logits)
            result.post_reflex_logits = result.logits.copy()
            result.reflex = None
            result.reflex_applied = False
            result.effective_reflex_scale = 0.0
            result.module_reflex_override = False
            result.module_reflex_dominance = 0.0
            result.valence_role = self.MODULE_VALENCE_ROLES.get(result.name, "support")
            result.gate_weight = 1.0
            result.contribution_share = 0.0
            result.gated_logits = result.logits.copy()
            result.intent_before_gating = None
            result.intent_after_gating = None

        should_compute_reflexes = self.config.is_modular and self.config.enable_reflexes
        if should_compute_reflexes:
            for result in module_results:
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
                result.gated_logits = result.logits.copy()

    @staticmethod
    def _clamp_unit(value: float) -> float:
        """
        Clamp a numeric value into the unit interval [0.0, 1.0].
        
        Returns:
            A float equal to the input value constrained to be no less than 0.0 and no greater than 1.0.
        """
        return float(min(1.0, max(0.0, value)))

    def _bound_observation(
        self,
        interface_name: str,
        observation: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        Bind and sanitize an interface's raw observation into a dictionary of bounded scalar evidence.

        Parameters:
            interface_name (str): Name of the interface to bind (must match one of MODULE_INTERFACES).
            observation (Dict[str, np.ndarray]): Full observation mapping; the interface is looked up by its observation_key.

        Returns:
            Dict[str, float]: A mapping of the interface's bound observation fields to finite float values.

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

    @staticmethod
    def _proposal_contribution_share(
        module_results: List[ModuleResult],
        gated_logits: Sequence[np.ndarray],
    ) -> Dict[str, float]:
        """
        Compute normalized proposal-contribution shares from gated logits.

        Uses the L1 magnitude of each active module's gated logits. When every
        active proposal is exactly zero, the share falls back to a uniform split
        across active proposal sources so the metric remains well-defined.
        """
        active_results = [result for result in module_results if result.active]
        if not active_results:
            return {
                result.name: 0.0
                for result in module_results
            }
        magnitudes = {
            result.name: float(np.sum(np.abs(gated_logit)))
            for result, gated_logit in zip(module_results, gated_logits, strict=True)
            if result.active
        }
        total = float(sum(magnitudes.values()))
        if total <= 1e-8:
            uniform = 1.0 / float(len(active_results))
            return {
                result.name: (uniform if result.active else 0.0)
                for result in module_results
            }
        return {
            result.name: (
                float(magnitudes.get(result.name, 0.0) / total)
                if result.active
                else 0.0
            )
            for result in module_results
        }

    def _compute_arbitration(
        self,
        module_results: List[ModuleResult],
        observation: Dict[str, np.ndarray],
    ) -> ArbitrationDecision:
        """
        Compute an arbitration decision that selects a dominant valence and per-module gate weights based on module proposals and the current observation.
        
        Parameters:
            module_results (List[ModuleResult]): Ordered proposal outputs from each proposal source; each entry provides logits, activity state, and bound observations used for arbitration.
            observation (Dict[str, np.ndarray]): Raw observation mapping interface keys to arrays; used to derive contextual evidence (visual, sensory, hunger, sleep, alert, and action context).
        
        Returns:
            ArbitrationDecision: An object containing:
                - strategy: arbitration strategy name ("priority_gating").
                - winning_valence: the chosen valence category (e.g., "threat", "hunger", "sleep", "exploration").
                - valence_scores: normalized valence weights that sum to 1.
                - module_gates: per-module gate weights to be applied to proposal logits.
                - suppressed_modules: list of active modules whose gate weight is effectively < 1.0.
                - evidence: per-valence evidence dictionaries used to compute raw scores.
                - intent_before_gating_idx: index of the highest-scoring action intent from raw (ungated) module logits.
                - intent_after_gating_idx: index of the highest-scoring action intent after applying module gate weights.
        """
        action_context = self._bound_action_context(observation)
        visual = self._bound_observation("visual_cortex", observation)
        sensory = self._bound_observation("sensory_cortex", observation)
        hunger = self._bound_observation("hunger_center", observation)
        sleep = self._bound_observation("sleep_center", observation)
        alert = self._bound_observation("alert_center", observation)

        threat_evidence = {
            "predator_visible": action_context["predator_visible"],
            "predator_certainty": action_context["predator_certainty"],
            "predator_proximity": self._clamp_unit(1.0 - action_context["predator_dist"]),
            "recent_contact": action_context["recent_contact"],
            "recent_pain": action_context["recent_pain"],
            "predator_smell_strength": alert["predator_smell_strength"],
        }
        threat_raw = self._clamp_unit(
            0.28 * threat_evidence["predator_visible"]
            + 0.2 * threat_evidence["predator_certainty"]
            + 0.2 * threat_evidence["predator_proximity"]
            + 0.16 * threat_evidence["recent_contact"]
            + 0.08 * threat_evidence["recent_pain"]
            + 0.08 * threat_evidence["predator_smell_strength"]
        )

        hunger_memory_freshness = self._clamp_unit(1.0 - hunger["food_memory_age"])
        hunger_evidence = {
            "hunger": action_context["hunger"],
            "on_food": action_context["on_food"],
            "food_visible": hunger["food_visible"],
            "food_certainty": hunger["food_certainty"],
            "food_smell_strength": hunger["food_smell_strength"],
            "food_memory_freshness": hunger_memory_freshness,
        }
        hunger_raw = self._clamp_unit(
            0.38 * hunger_evidence["hunger"]
            + 0.14 * hunger_evidence["on_food"]
            + 0.16 * hunger_evidence["food_visible"]
            + 0.1 * hunger_evidence["food_certainty"]
            + 0.12 * hunger_evidence["food_smell_strength"]
            + 0.1 * hunger_evidence["food_memory_freshness"]
        )

        sleep_home_pressure = self._clamp_unit(1.0 - sleep["home_dist"])
        sleep_evidence = {
            "fatigue": action_context["fatigue"],
            "sleep_debt": action_context["sleep_debt"],
            "night": action_context["night"],
            "on_shelter": action_context["on_shelter"],
            "shelter_role_level": action_context["shelter_role_level"],
            "home_pressure": sleep_home_pressure,
        }
        sleep_raw = self._clamp_unit(
            0.26 * sleep_evidence["fatigue"]
            + 0.24 * sleep_evidence["sleep_debt"]
            + 0.14 * sleep_evidence["night"]
            + 0.12 * sleep_evidence["on_shelter"]
            + 0.12 * sleep_evidence["shelter_role_level"]
            + 0.12 * sleep_evidence["home_pressure"]
        )

        exploration_safety = self._clamp_unit(1.0 - threat_raw)
        exploration_residual = self._clamp_unit(1.0 - max(threat_raw, hunger_raw, sleep_raw))
        exploration_evidence = {
            "safety_margin": exploration_safety,
            "residual_drive": exploration_residual,
            "day": action_context["day"],
            "off_shelter": self._clamp_unit(1.0 - action_context["on_shelter"]),
            "visual_openness": max(
                visual["food_visible"],
                visual["shelter_visible"],
                visual["predator_visible"],
            ),
            "food_smell_directionality": self._clamp_unit(
                abs(sensory["food_smell_dx"]) + abs(sensory["food_smell_dy"])
            ),
        }
        exploration_raw = self._clamp_unit(
            0.46 * exploration_evidence["residual_drive"]
            + 0.18 * exploration_evidence["safety_margin"]
            + 0.14 * exploration_evidence["day"]
            + 0.1 * exploration_evidence["off_shelter"]
            + 0.06 * exploration_evidence["visual_openness"]
            + 0.06 * exploration_evidence["food_smell_directionality"]
        )

        raw_scores = {
            "threat": threat_raw,
            "hunger": hunger_raw,
            "sleep": sleep_raw,
            "exploration": exploration_raw,
        }
        total = float(sum(raw_scores.values()))
        if total <= 1e-8:
            normalized_scores = {
                "threat": 0.0,
                "hunger": 0.0,
                "sleep": 0.0,
                "exploration": 1.0,
            }
        else:
            normalized_scores = {
                name: float(score / total)
                for name, score in raw_scores.items()
            }
        # Earlier entries in self.VALENCE_ORDER win ties in normalized_scores, which
        # keeps the arbitration survival-first: threat > hunger > sleep > exploration.
        winning_valence = max(
            self.VALENCE_ORDER,
            key=lambda name: (normalized_scores[name], -self.VALENCE_ORDER.index(name)),
        )

        pre_gating_logits = np.sum(
            np.stack([result.logits for result in module_results], axis=0),
            axis=0,
        )
        intent_before_gating_idx = int(np.argmax(pre_gating_logits))
        module_gates: Dict[str, float] = {}
        for result in module_results:
            module_gates[result.name] = self._priority_gate_weight_for(
                winning_valence,
                result.name,
            )
        suppressed_modules = [
            result.name
            for result in module_results
            if result.active and module_gates[result.name] < 0.999
        ]
        gated_logits = [
            module_gates[result.name] * result.logits
            for result in module_results
        ]
        intent_after_gating_idx = int(
            np.argmax(np.sum(np.stack(gated_logits, axis=0), axis=0))
        )
        module_contribution_share = self._proposal_contribution_share(
            module_results,
            gated_logits,
        )
        dominant_module = max(
            module_results,
            key=lambda result: (
                module_contribution_share[result.name],
                -module_results.index(result),
            ),
        ).name
        dominant_module_share = float(module_contribution_share[dominant_module])
        effective_module_count = 0.0
        positive_shares = [
            share
            for share in module_contribution_share.values()
            if share > 1e-8
        ]
        if positive_shares:
            effective_module_count = float(
                1.0 / sum(share * share for share in positive_shares)
            )
        active_results = [result for result in module_results if result.active]
        if active_results:
            agreeing_modules = sum(
                1
                for result, gated_logit in zip(module_results, gated_logits, strict=True)
                if result.active and int(np.argmax(gated_logit)) == intent_after_gating_idx
            )
            module_agreement_rate = float(agreeing_modules / len(active_results))
        else:
            module_agreement_rate = 0.0
        module_disagreement_rate = float(1.0 - module_agreement_rate) if active_results else 0.0
        return ArbitrationDecision(
            strategy="priority_gating",
            winning_valence=winning_valence,
            valence_scores=normalized_scores,
            module_gates=module_gates,
            module_contribution_share=module_contribution_share,
            dominant_module=dominant_module,
            dominant_module_share=dominant_module_share,
            effective_module_count=effective_module_count,
            module_agreement_rate=module_agreement_rate,
            module_disagreement_rate=module_disagreement_rate,
            suppressed_modules=suppressed_modules,
            evidence={
                "threat": threat_evidence,
                "hunger": hunger_evidence,
                "sleep": sleep_evidence,
                "exploration": exploration_evidence,
            },
            intent_before_gating_idx=intent_before_gating_idx,
            intent_after_gating_idx=intent_after_gating_idx,
        )

    def _priority_gate_weight_for(self, winning_valence: str, module_name: str) -> float:
        """Return the configured gate weight for a module under the winning valence."""
        valence_weights = self.PRIORITY_GATING_WEIGHTS.get(winning_valence)
        if valence_weights is None:
            raise ValueError(
                f"Priority gating weights missing winning valence '{winning_valence}'."
            )
        if module_name not in valence_weights:
            raise ValueError(
                f"Priority gating weights missing module '{module_name}' "
                f"for winning valence '{winning_valence}'."
            )
        return float(valence_weights[module_name])

    def _arbitration_gate_weight_for(
        self,
        arbitration: ArbitrationDecision,
        module_name: str,
    ) -> float:
        """Return the gate weight exported by an arbitration decision for one module."""
        if module_name not in arbitration.module_gates:
            raise ValueError(
                f"Arbitration decision missing gate weight for module '{module_name}' "
                f"under winning valence '{arbitration.winning_valence}'."
            )
        return float(arbitration.module_gates[module_name])

    def _apply_priority_gating(
        self,
        module_results: List[ModuleResult],
        arbitration: ArbitrationDecision,
    ) -> None:
        """
        Apply per-module priority gating from an arbitration decision to each module's logits and probabilities.
        
        This updates each ModuleResult in place: assigns its valence role, gate weight,
        computes gated logits (gate weight multiplied by the module's logits), replaces the
        module's logits with the gated logits, recomputes the module's softmax probabilities,
        and stamps the intents before and after gating for downstream debug export.
        
        Parameters:
            module_results (List[ModuleResult]): Ordered list of module outputs to update.
            arbitration (ArbitrationDecision): ArbitrationDecision providing per-module gate weights and winning valence.
        """
        intent_before_gating = ACTIONS[arbitration.intent_before_gating_idx]
        intent_after_gating = ACTIONS[arbitration.intent_after_gating_idx]
        for result in module_results:
            result.valence_role = self.MODULE_VALENCE_ROLES.get(result.name, "support")
            gate_weight = self._arbitration_gate_weight_for(arbitration, result.name)
            result.gate_weight = gate_weight
            result.contribution_share = float(
                arbitration.module_contribution_share.get(result.name, 0.0)
            )
            result.gated_logits = gate_weight * result.logits
            result.logits = result.gated_logits.copy()
            result.probs = softmax(result.logits)
            result.intent_before_gating = intent_before_gating
            result.intent_after_gating = intent_after_gating

    def act(
        self,
        observation: Dict[str, np.ndarray],
        bus: MessageBus | None = None,
        *,
        sample: bool = True,
        policy_mode: str = "normal",
    ) -> BrainStep:
        """
        Choose and execute an action for the provided observation and return a populated BrainStep.
        
        Parameters:
            observation (Dict[str, np.ndarray]): Mapping of interface observation arrays consumed by proposal modules and context interfaces.
            bus (MessageBus | None): Optional message bus for publishing per-module proposal diagnostics and final selection/execution diagnostics; pass None to disable publishing.
            sample (bool): If True, sample the executed action from the final policy distribution; if False, select the greedy argmax action.
            policy_mode (str): Execution mode, either "normal" to run the learned action-center and motor-cortex corrections, or "reflex_only" to skip learned controllers and select directly from the post-reflex modular proposals. "reflex_only" requires a modular architecture and reflexes enabled in the configuration.
        
        Returns:
            BrainStep: Decision container filled with per-module ModuleResult entries, action-center and motor-cortex logits/policies, combined logits with and without reflexes, the final policy and value estimate, selected intent and action indices, override flags, controller input vectors, the active policy_mode, and the computed arbitration_decision.
        
        Raises:
            ValueError: If policy_mode is invalid, or if "reflex_only" is requested but the brain is not modular or reflexes are disabled.
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

        store_cache = sample and policy_mode == "normal"
        module_results = self._proposal_results(
            observation,
            store_cache=store_cache,
            training=sample,
        )
        arbitration_without_reflex = self._compute_arbitration(module_results, observation)
        gated_logits_without_reflex = [
            self._arbitration_gate_weight_for(arbitration_without_reflex, result.name) * result.logits
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

        self._apply_reflex_path(module_results)
        arbitration = self._compute_arbitration(module_results, observation)
        self._apply_priority_gating(module_results, arbitration)

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
            arbitration_decision=arbitration,
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
        arbitration = self._compute_arbitration(module_results, observation)
        self._apply_priority_gating(module_results, arbitration)
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
        Perform a temporal-difference policy-gradient update that trains the policy, proposal modules (or monolithic proposal), and motor/value networks.
        
        Parameters:
            decision (BrainStep): Recorded action decision containing module results, selected action index, policy, and controller outputs.
            reward (float): Observed scalar reward following the decision.
            next_observation (Dict[str, np.ndarray]): Observation after the action used to estimate the next state's value.
            done (bool): Whether the episode terminated after the action.
        
        Returns:
            metrics (Dict[str, float]): Training diagnostics with the following keys:
                - "reward": the provided reward.
                - "td_target": computed TD target (reward + gamma * next_value).
                - "td_error": clipped advantage used for the policy update.
                - "value": the value estimate from the provided decision.
                - "next_value": the estimated value of the next observation (0.0 if done).
                - "entropy": policy entropy computed from decision.policy.
                - "aux_modules": number of modules that received auxiliary gradients.
        """
        if decision.policy_mode != "normal":
            raise ValueError(
                "learn() only supports decisions produced with policy_mode='normal'."
            )
        next_value = 0.0 if done else self.estimate_value(next_observation)
        td_target = reward + self.gamma * next_value
        advantage = float(np.clip(td_target - decision.value, -4.0, 4.0))
        grad_policy_logits = advantage * (decision.policy - one_hot(decision.action_idx, self.action_dim))
        reflex_aux_grads = {
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

        module_total_grads: Dict[str, np.ndarray] = {}
        if self.config.is_modular:
            if self.module_bank is None:
                raise RuntimeError("Module bank unavailable for modular architecture.")
            per_module_input_grads = proposal_input_grads.reshape(
                len(decision.module_results),
                self.action_dim,
            )
            for result, extra_grad in zip(
                decision.module_results,
                per_module_input_grads,
                strict=True,
            ):
                gate_weight = float(result.gate_weight)
                total_grad = gate_weight * np.asarray(
                    reflex_aux_grads.get(
                        result.name,
                        np.zeros(self.action_dim, dtype=float),
                    ),
                    dtype=float,
                )
                if not self.config.uses_local_credit_only:
                    total_grad += gate_weight * grad_policy_logits
                total_grad += gate_weight * np.asarray(extra_grad, dtype=float)
                module_total_grads[result.name] = module_total_grads.get(
                    result.name,
                    np.zeros(self.action_dim, dtype=float),
                ) + total_grad
            self.module_bank.backward(
                np.zeros(self.action_dim, dtype=float),
                lr=self.module_lr,
                aux_grads=module_total_grads,
            )
        else:
            if self.monolithic_policy is None:
                raise RuntimeError("Monolithic network unavailable for the configured architecture.")
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
            "aux_modules": float(len(reflex_aux_grads)),
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

        available = list(metadata.get("modules", {}).keys())
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
                raise KeyError("Incompatible save: monolithic weights are missing.")
            self.monolithic_policy.load_state_dict(bank_state[monolithic_name])

        return loaded
