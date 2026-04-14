from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .ablations import BrainAblationConfig, default_brain_config
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
    valence_logits: Dict[str, float] = field(default_factory=dict)
    base_gates: Dict[str, float] = field(default_factory=dict)
    gate_adjustments: Dict[str, float] = field(default_factory=dict)
    arbitration_value: float = 0.0
    learned_adjustment: bool = False
    guards_applied: bool = False
    food_bias_applied: bool = False
    food_bias_action: str | None = None
    module_contribution_share: Dict[str, float] = field(default_factory=dict)
    dominant_module: str = ""
    dominant_module_share: float = 0.0
    effective_module_count: float = 0.0
    module_agreement_rate: float = 0.0
    module_disagreement_rate: float = 0.0

    def to_payload(self) -> Dict[str, object]:
        """
        Serialize the arbitration decision to a JSON-safe dictionary for telemetry or persistence.
        
        This payload rounds numeric values to six decimal places, sorts mapping keys for determinism, and injects a legacy
        `threat.predator_proximity = 0.0` entry when missing for backward compatibility.
        
        Returns:
            Dict[str, object]: Dictionary containing:
                - "strategy": arbitration strategy name.
                - "winning_valence": selected valence name.
                - "valence_scores": mapping valence -> score (rounded to 6 decimals).
                - "valence_logits": mapping valence -> raw logits (rounded to 6 decimals).
                - "base_gates": mapping module -> base gate weight (rounded to 6 decimals).
                - "gate_adjustments": mapping module -> learned gate adjustment (rounded to 6 decimals).
                - "module_gates": mapping module -> final gate weight (rounded to 6 decimals).
                - "arbitration_value": arbitration scalar value (rounded to 6 decimals).
                - "learned_adjustment": boolean indicating whether learned adjustments were applied.
                - "guards_applied": boolean indicating whether deterministic arbitration guards overrode selection.
                - "food_bias_applied": boolean indicating whether deterministic food-direction bias was added.
                - "food_bias_action": biased action name when a food-direction bias was added, otherwise None.
                - "module_contribution_share": mapping module -> normalized contribution share (rounded to 6 decimals).
                - "dominant_module": name of the dominant proposal source.
                - "dominant_module_share": dominant module contribution share (rounded to 6 decimals).
                - "effective_module_count": effective number of contributing modules (rounded to 6 decimals).
                - "module_agreement_rate": fraction of modules agreeing on the final intent (rounded to 6 decimals).
                - "module_disagreement_rate": complement of agreement (rounded to 6 decimals).
                - "suppressed_modules": list of suppressed module names.
                - "evidence": mapping valence -> evidence dict with each numeric value rounded to 6 decimals.
                - "intent_before_gating": action name corresponding to intent index before gating.
                - "intent_after_gating": action name corresponding to intent index after gating.
        """
        evidence_payload = {
            key: {
                inner_key: round(float(inner_value), 6)
                for inner_key, inner_value in sorted(values.items())
            }
            for key, values in sorted(self.evidence.items())
        }
        threat_payload = evidence_payload.get("threat")
        if threat_payload is not None and "predator_proximity" not in threat_payload:
            # Legacy conflict scorers key on predator_proximity. The learned
            # arbitration evidence no longer consumes distance, so traces export
            # a conservative compatibility value and let certainty carry danger.
            threat_payload["predator_proximity"] = 0.0
            evidence_payload["threat"] = {
                key: threat_payload[key]
                for key in sorted(threat_payload)
            }
        return {
            "strategy": self.strategy,
            "winning_valence": self.winning_valence,
            "valence_scores": {
                key: round(float(value), 6)
                for key, value in sorted(self.valence_scores.items())
            },
            "valence_logits": {
                key: round(float(value), 6)
                for key, value in sorted(self.valence_logits.items())
            },
            "base_gates": {
                key: round(float(value), 6)
                for key, value in sorted(self.base_gates.items())
            },
            "gate_adjustments": {
                key: round(float(value), 6)
                for key, value in sorted(self.gate_adjustments.items())
            },
            "module_gates": {
                key: round(float(value), 6)
                for key, value in sorted(self.module_gates.items())
            },
            "arbitration_value": round(float(self.arbitration_value), 6),
            "learned_adjustment": bool(self.learned_adjustment),
            "guards_applied": bool(self.guards_applied),
            "food_bias_applied": bool(self.food_bias_applied),
            "food_bias_action": self.food_bias_action,
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
            "evidence": evidence_payload,
            "intent_before_gating": ACTIONS[self.intent_before_gating_idx],
            "intent_after_gating": ACTIONS[self.intent_after_gating_idx],
        }


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
    MONOLITHIC_POLICY_NAME = "monolithic_policy"
    MONOLITHIC_HIDDEN_DIM = sum(MODULE_HIDDEN_DIMS.values())
    VALENCE_ORDER = ("threat", "hunger", "sleep", "exploration")
    ARBITRATION_NETWORK_NAME = "arbitration_network"
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
    ARBITRATION_EVIDENCE_FIELDS = MappingProxyType(
        {
            "threat": (
                "predator_visible",
                "predator_certainty",
                "predator_motion_salience",
                "recent_contact",
                "recent_pain",
                "predator_smell_strength",
            ),
            "hunger": (
                "hunger",
                "on_food",
                "food_visible",
                "food_certainty",
                "food_smell_strength",
                "food_memory_freshness",
            ),
            "sleep": (
                "fatigue",
                "sleep_debt",
                "night",
                "on_shelter",
                "shelter_role_level",
                "shelter_path_confidence",
            ),
            "exploration": (
                "safety_margin",
                "residual_drive",
                "day",
                "off_shelter",
                "visual_openness",
                "food_smell_directionality",
            ),
        }
    )
    ARBITRATION_GATE_MODULE_ORDER = (
        "alert_center",
        "hunger_center",
        "sleep_center",
        "visual_cortex",
        "sensory_cortex",
        MONOLITHIC_POLICY_NAME,
    )
    # Linear weights for each valence's evidence signals used by the fixed-formula
    # arbitration baseline and by the warm-start initialization of the learned network.
    VALENCE_EVIDENCE_WEIGHTS: MappingProxyType = MappingProxyType(
        {
            "threat": {
                "predator_visible": 0.3,
                "predator_certainty": 0.22,
                "predator_motion_salience": 0.12,
                "recent_contact": 0.14,
                "recent_pain": 0.1,
                "predator_smell_strength": 0.12,
            },
            "hunger": {
                "hunger": 0.38,
                "on_food": 0.14,
                "food_visible": 0.16,
                "food_certainty": 0.1,
                "food_smell_strength": 0.12,
                "food_memory_freshness": 0.1,
            },
            "sleep": {
                "fatigue": 0.26,
                "sleep_debt": 0.24,
                "night": 0.14,
                "on_shelter": 0.12,
                "shelter_role_level": 0.12,
                "shelter_path_confidence": 0.12,
            },
            "exploration": {
                "residual_drive": 0.46,
                "safety_margin": 0.18,
                "day": 0.14,
                "off_shelter": 0.1,
                "visual_openness": 0.06,
                "food_smell_directionality": 0.06,
            },
        }
    )

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
        arbitration_input_dim = self._arbitration_evidence_input_dim()
        self.arbitration_network = ArbitrationNetwork(
            input_dim=arbitration_input_dim,
            hidden_dim=32,
            rng=self.arbitration_rng,
            name=self.ARBITRATION_NETWORK_NAME,
            gate_adjustment_min=self.config.gate_adjustment_bounds[0],
            gate_adjustment_max=self.config.gate_adjustment_bounds[1],
        )
        self._warm_start_arbitration_network(
            warm_start_scale=self.config.warm_start_scale,
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
        	module_results (List[ModuleResult]): Per-module proposal results whose `logits` arrays are concatenated in proposal order.
        	observation (Dict[str, np.ndarray]): Full observation mapping; the action context is taken from the key defined by ACTION_CONTEXT_INTERFACE.observation_key and bound/flattened via ACTION_CONTEXT_INTERFACE.

        Returns:
        	np.ndarray: 1-D array formed by concatenating all module logits followed by the action context vector.
        """
        logits_flat = np.concatenate([result.logits for result in module_results], axis=0)
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
            np.asarray(result.logits, dtype=float)
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
        heading_norm = math.hypot(heading_dx, heading_dy)
        move_norm = math.hypot(float(move_dx), float(move_dy))
        if heading_norm <= 1e-8 or move_norm <= 1e-8:
            orientation_alignment = 1.0
        else:
            cosine = (
                heading_dx * float(move_dx) + heading_dy * float(move_dy)
            ) / (heading_norm * move_norm)
            orientation_alignment = float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))
        terrain_difficulty = float(
            np.clip(float(motor_context.get("terrain_difficulty", 0.0)), 0.0, 1.0)
        )
        fatigue = float(np.clip(float(motor_context.get("fatigue", 0.0)), 0.0, 1.0))
        execution_difficulty = float(
            np.clip(
                terrain_difficulty * (1.0 - orientation_alignment) * (1.0 + fatigue),
                0.0,
                1.0,
            )
        )
        return {
            "orientation_alignment": orientation_alignment,
            "terrain_difficulty": terrain_difficulty,
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

    def _deterministic_valence_winner(self, normalized_scores: Dict[str, float]) -> str:
        """Return the highest-scoring valence, with earlier VALENCE_ORDER entries winning ties."""
        return max(
            self.VALENCE_ORDER,
            key=lambda name: (normalized_scores[name], -self.VALENCE_ORDER.index(name)),
        )

    @classmethod
    def _arbitration_evidence_input_dim(cls) -> int:
        """
        Compute the total number of scalar signals in the flattened arbitration evidence vector.
        
        This is the sum of the lengths of each sequence in `ARBITRATION_EVIDENCE_FIELDS`.
        
        Returns:
            input_dim (int): Total size of the concatenated evidence vector consumed by the arbitration network.
        """
        return sum(len(fields) for fields in cls.ARBITRATION_EVIDENCE_FIELDS.values())

    @classmethod
    def _arbitration_evidence_signal_names(cls) -> tuple[str, ...]:
        """
        Return the flattened arbitration evidence signal names in the exact order expected by the arbitration network.
        
        The returned tuple contains strings formatted as "<valence>.<field>" for each field listed in ARBITRATION_EVIDENCE_FIELDS, iterating valences in VALENCE_ORDER and fields in their declared order.
        
        Returns:
            tuple[str, ...]: Ordered signal names used to construct the arbitration evidence vector.
        """
        return tuple(
            f"{valence}.{evidence_field}"
            for valence in cls.VALENCE_ORDER
            for evidence_field in cls.ARBITRATION_EVIDENCE_FIELDS[valence]
        )

    def _arbitration_evidence_vector(self, evidence: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        Flatten arbitration evidence into a 1D numpy array matching the exact input order expected by the ArbitrationNetwork.
        
        Parameters:
            evidence (Dict[str, Dict[str, float]]): Mapping from valence name to a mapping of evidence field name to its numeric value.
        
        Returns:
            np.ndarray: 1D float array containing evidence values in the order:
                for each valence in self.VALENCE_ORDER, iterate fields in self.ARBITRATION_EVIDENCE_FIELDS[valence].
        
        Raises:
            RuntimeError: If the computed signal order does not match ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES.
        """
        signal_names = self._arbitration_evidence_signal_names()
        if signal_names != ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES:
            raise RuntimeError("Arbitration evidence order does not match ArbitrationNetwork input order.")
        return np.array(
            [
                float(evidence[valence][evidence_field])
                for valence in self.VALENCE_ORDER
                for evidence_field in self.ARBITRATION_EVIDENCE_FIELDS[valence]
            ],
            dtype=float,
        )

    def _warm_start_arbitration_network(
        self,
        warm_start_scale: float | None = None,
    ) -> None:
        """
        Initialize the arbitration network so its initial valence logits approximate the legacy fixed-formula scores, scaled by a warm-start factor.
        
        Parameters:
            warm_start_scale (float | None): Optional scale in [0.0, 1.0] that interpolates between the random initialization (0.0) and the legacy-like warm-start (1.0). If None, the brain's configured warm_start_scale is used.
        
        Raises:
            ValueError: If `warm_start_scale` (or the configured scale when None) is not finite or not in [0.0, 1.0], or if the network's hidden dimension is smaller than its input dimension.
            RuntimeError: If the computed arbitration evidence signal ordering does not match ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES.
        """
        net = self.arbitration_network
        scale = (
            float(self.config.warm_start_scale)
            if warm_start_scale is None
            else float(warm_start_scale)
        )
        if not math.isfinite(scale):
            raise ValueError("warm_start_scale must be finite.")
        if scale < 0.0 or scale > 1.0:
            raise ValueError("warm_start_scale must be in [0.0, 1.0].")
        if scale == 0.0:
            return
        signal_names = self._arbitration_evidence_signal_names()
        if signal_names != ArbitrationNetwork.EVIDENCE_SIGNAL_NAMES:
            raise RuntimeError("Arbitration evidence order does not match ArbitrationNetwork input order.")
        if net.hidden_dim < net.input_dim:
            raise ValueError("Arbitration warm start requires hidden_dim >= input_dim.")

        net.W1.fill(0.0)
        net.b1.fill(0.0)
        net.W2_valence.fill(0.0)
        net.b2_valence.fill(0.0)
        net.W2_gate.fill(0.0)
        net.b2_gate.fill(0.0)
        net.W2_value.fill(0.0)
        net.b2_value.fill(0.0)

        copy_scale = 0.1
        valence_logit_scale = 6.0
        per_side_scale = math.sqrt(scale)
        for index in range(net.input_dim):
            net.W1[index, index] = copy_scale * per_side_scale

        input_index = 0
        for evidence_valence in self.VALENCE_ORDER:
            weights = self.VALENCE_EVIDENCE_WEIGHTS[evidence_valence]
            for evidence_field in self.ARBITRATION_EVIDENCE_FIELDS[evidence_valence]:
                for output_index, output_valence in enumerate(self.VALENCE_ORDER):
                    net.W2_valence[output_index, input_index] = (
                        per_side_scale
                        * valence_logit_scale
                        * weights.get(evidence_field, 0.0)
                        / copy_scale
                        if output_valence == evidence_valence
                        else 0.0
                    )
                input_index += 1
        net.cache = None

    def _fixed_formula_valence_scores_from_evidence(
        self,
        evidence: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """
        Compute a deterministic, fixed-formula valence distribution from structured evidence for arbitration regularization.
        
        This uses a hard-coded linear weighting of evidence signals to produce clamped per-valence scores for the valences "threat", "hunger", "sleep", and "exploration", then normalizes them to sum to 1. If the summed raw scores are effectively zero, returns a fallback distribution that places all mass on "exploration".
        
        Parameters:
            evidence (Dict[str, Dict[str, float]]): Nested mapping from valence name to signal name to scalar value.
                Expected top-level keys: "threat", "hunger", "sleep", "exploration". Each inner mapping should contain
                the signals referenced by the fixed formula (e.g., "predator_visible", "hunger", "fatigue", "residual_drive", etc.).
        
        Returns:
            np.ndarray: 1-D array of four floats ordered according to self.VALENCE_ORDER representing the normalized
            valence scores. Returns [0.0, 0.0, 0.0, 1.0] if the raw total is <= 1e-8.
        """
        raw_scores = {
            valence: self._clamp_unit(
                sum(
                    weight * evidence[valence][evidence_field]
                    for evidence_field, weight in self.VALENCE_EVIDENCE_WEIGHTS[valence].items()
                )
            )
            for valence in self.VALENCE_ORDER
        }
        total = float(sum(raw_scores.values()))
        if total <= 1e-8:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        return np.array(
            [
                float(raw_scores[name] / total)
                for name in self.VALENCE_ORDER
            ],
            dtype=float,
        )

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
        *,
        training: bool = False,
        store_cache: bool = True,
    ) -> ArbitrationDecision:
        """
        Compute an arbitration decision: choose a valence and produce per-module gate weights and diagnostics from module proposals and the current observation.
        
        Builds per-valence evidence (threat, hunger, sleep, exploration), scores valences either with the learned arbitration network or a fixed-formula fallback, optionally applies deterministic inference-time guards, computes multiplicative priority gates for each proposal module, applies those gates to module logits, and returns contribution, agreement, and intent diagnostics used downstream for gating and action selection.
        
        Parameters:
            module_results (List[ModuleResult]): Ordered proposal outputs for each proposal source; each entry provides logits and activity flags used to compute gates and contributions.
            observation (Dict[str, np.ndarray]): Raw observation mapping interface keys to arrays; bound/sanitized values are used to construct per-valence evidence.
            training (bool, optional): If True and learned arbitration is enabled, select the winning valence stochastically from learned probabilities; if False selection is deterministic. Defaults to False.
            store_cache (bool, optional): When using the learned arbitration network, controls whether the network may store/reuse its forward cache. Defaults to True.
        
        Returns:
            ArbitrationDecision: Populated arbitration decision including:
                - strategy: arbitration strategy name.
                - winning_valence, valence_scores, valence_logits.
                - arbitration_value and learned_adjustment flag.
                - module_gates, base_gates, gate_adjustments.
                - module_contribution_share, dominant_module and dominant_module_share, effective_module_count.
                - module_agreement_rate and module_disagreement_rate.
                - suppressed_modules.
                - evidence (per-valence evidence dicts).
                - intent_before_gating_idx and intent_after_gating_idx.
                - guards_applied flag indicating whether deterministic guards overrode the chosen valence.
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
            "predator_motion_salience": alert["predator_motion_salience"],
            "recent_contact": action_context["recent_contact"],
            "recent_pain": action_context["recent_pain"],
            "predator_smell_strength": alert["predator_smell_strength"],
        }
        threat_raw = self._clamp_unit(
            0.3 * threat_evidence["predator_visible"]
            + 0.22 * threat_evidence["predator_certainty"]
            + 0.12 * threat_evidence["predator_motion_salience"]
            + 0.14 * threat_evidence["recent_contact"]
            + 0.1 * threat_evidence["recent_pain"]
            + 0.12 * threat_evidence["predator_smell_strength"]
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

        sleep_path_confidence = self._clamp_unit(
            max(
                sleep["shelter_trace_strength"],
                1.0 - sleep["shelter_memory_age"],
            )
        )
        sleep_evidence = {
            "fatigue": action_context["fatigue"],
            "sleep_debt": action_context["sleep_debt"],
            "night": action_context["night"],
            "on_shelter": action_context["on_shelter"],
            "shelter_role_level": action_context["shelter_role_level"],
            "shelter_path_confidence": sleep_path_confidence,
        }
        sleep_raw = self._clamp_unit(
            0.26 * sleep_evidence["fatigue"]
            + 0.24 * sleep_evidence["sleep_debt"]
            + 0.14 * sleep_evidence["night"]
            + 0.12 * sleep_evidence["on_shelter"]
            + 0.12 * sleep_evidence["shelter_role_level"]
            + 0.12 * sleep_evidence["shelter_path_confidence"]
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

        evidence = {
            "threat": threat_evidence,
            "hunger": hunger_evidence,
            "sleep": sleep_evidence,
            "exploration": exploration_evidence,
        }
        if self.config.use_learned_arbitration:
            evidence_vector = self._arbitration_evidence_vector(evidence)
            valence_logits_array, gate_adjustments_array, arbitration_value = self.arbitration_network.forward(
                evidence_vector,
                store_cache=store_cache,
            )
            valence_probs = softmax(valence_logits_array)
            normalized_scores = {
                name: float(valence_probs[index])
                for index, name in enumerate(self.VALENCE_ORDER)
            }
            valence_logits = {
                name: float(valence_logits_array[index])
                for index, name in enumerate(self.VALENCE_ORDER)
            }
            if training:
                winning_index = int(self.arbitration_rng.choice(len(self.VALENCE_ORDER), p=valence_probs))
                winning_valence = self.VALENCE_ORDER[winning_index]
            else:
                winning_valence = self._deterministic_valence_winner(normalized_scores)

            adjustment_by_module = {
                name: float(gate_adjustments_array[index])
                for index, name in enumerate(self.ARBITRATION_GATE_MODULE_ORDER)
            }
        else:
            self.arbitration_network.cache = None
            valence_probs = self._fixed_formula_valence_scores_from_evidence(evidence)
            normalized_scores = {
                name: float(valence_probs[index])
                for index, name in enumerate(self.VALENCE_ORDER)
            }
            valence_logits = {
                "threat": float(threat_raw),
                "hunger": float(hunger_raw),
                "sleep": float(sleep_raw),
                "exploration": float(exploration_raw),
            }
            winning_valence = self._deterministic_valence_winner(normalized_scores)
            adjustment_by_module = {
                name: 1.0
                for name in self.ARBITRATION_GATE_MODULE_ORDER
            }
            arbitration_value = 0.0
        guards_applied = False
        if self.config.enable_deterministic_guards:
            if (
                not training
                and threat_evidence["predator_visible"] >= 0.5
                and threat_evidence["predator_certainty"] >= 0.35
            ):
                winning_valence = "threat"
                guards_applied = True
            elif (
                not training
                and threat_raw < 0.25
                and hunger_evidence["hunger"] >= 0.5
                and sleep_evidence["night"] < 0.5
            ):
                winning_valence = "hunger"
                guards_applied = True

        pre_gating_logits = np.sum(
            np.stack([result.logits for result in module_results], axis=0),
            axis=0,
        )
        intent_before_gating_idx = int(np.argmax(pre_gating_logits))
        module_gates: Dict[str, float] = {}
        base_gates: Dict[str, float] = {}
        gate_adjustments: Dict[str, float] = {}
        for result in module_results:
            base_gate = self._priority_gate_weight_for(
                winning_valence,
                result.name,
            )
            adjustment = adjustment_by_module.get(result.name, 1.0)
            base_gates[result.name] = base_gate
            gate_adjustments[result.name] = adjustment
            module_gates[result.name] = float(np.clip(base_gate * adjustment, 0.0, 1.0))
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
            evidence=evidence,
            intent_before_gating_idx=intent_before_gating_idx,
            intent_after_gating_idx=intent_after_gating_idx,
            valence_logits=valence_logits,
            base_gates=base_gates,
            gate_adjustments=gate_adjustments,
            arbitration_value=float(arbitration_value),
            learned_adjustment=bool(self.config.use_learned_arbitration),
            guards_applied=guards_applied,
        )

    def _priority_gate_weight_for(self, winning_valence: str, module_name: str) -> float:
        """
        Get the configured priority gate weight for a module for the specified winning valence.
        
        Parameters:
            winning_valence (str): Valence name used to look up priority gating weights.
            module_name (str): Module name whose gate weight is requested.
        
        Returns:
            float: The gate weight for the module.
        
        Raises:
            ValueError: If `winning_valence` is not present in PRIORITY_GATING_WEIGHTS or if
                        the specified `module_name` is not defined for that valence.
        """
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
        arbitration_without_reflex = self._compute_arbitration(
            module_results,
            observation,
            training=False,
            store_cache=False,
        )
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
        arbitration = self._compute_arbitration(
            module_results,
            observation,
            training=training_mode,
            store_cache=store_cache,
        )
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
                food_bias_action = self._direction_action(food_dx, food_dy)
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
            self._apply_reflex_path(module_results)
            arbitration = self._compute_arbitration(
                module_results,
                observation,
                training=False,
                store_cache=False,
            )
            self._apply_priority_gating(module_results, arbitration)
            action_input = self._build_action_input(module_results, observation)
            _, value = self.action_center.forward(action_input, store_cache=False)
            return float(value)
        finally:
            if hidden_state_snapshot is not None and self.module_bank is not None:
                self.module_bank.restore_hidden_states(hidden_state_snapshot)

    def _action_target(self, action: str) -> np.ndarray:
        """
        Get the one-hot action vector for a named action.
        
        Parameters:
            action (str): Action name key looked up in ACTION_TO_INDEX.
        
        Returns:
            np.ndarray: Float one-hot vector of length self.action_dim with 1.0 at the action's index and 0.0 elsewhere.
        
        Raises:
            KeyError: If `action` is not present in ACTION_TO_INDEX.
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
                - "shelter_trace_strength", "shelter_trace_dx", "shelter_trace_dy"
        
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
            and signals["shelter_trace_strength"] > 0.0
        ):
            return self._direction_reflex(
                "sleep_center",
                dx=signals["shelter_trace_dx"],
                dy=signals["shelter_trace_dy"],
                reason="follow_shelter_trace_to_rest",
                triggers=self._signal_subset(
                    signals,
                    "fatigue",
                    "sleep_debt",
                    "night",
                    "hunger",
                    "shelter_trace_dx",
                    "shelter_trace_dy",
                    "shelter_trace_strength",
                ),
            )
        return None

    def _alert_reflex_decision(self, signals: Dict[str, float]) -> ReflexDecision | None:
        """
        Decides whether an alert-related reflex should trigger and, if so, which reflex to issue.
        
        Examines alert-center thresholds against provided sensor and memory signals to select one of:
        - a stay/freeze reflex when on shelter and a threat is detected,
        - an away-directed retreat from a visible predator,
        - an away-directed retreat from a recent predator memory when outside shelter,
        - an away-directed retreat from a predator trace when non-visual threat cues are present and a predator trace exists,
        - a directional repeat of a recent escape route when an escape memory is fresh and the agent is outside shelter.
        
        Parameters:
            signals (dict[str, float]): Scalar sensor and memory values referenced by keys such as
                "on_shelter", "predator_visible", "predator_occluded", "predator_smell_strength",
                "recent_contact", "predator_certainty", "predator_dx", "predator_dy",
                "predator_memory_age", "predator_memory_dx", "predator_memory_dy",
                "recent_pain", "predator_trace_strength", "predator_trace_dx", "predator_trace_dy",
                "escape_memory_age", "escape_memory_dx", "escape_memory_dy".
        
        Returns:
            ReflexDecision: A decision describing the chosen reflex when a threshold condition is met.
            `None` if no alert reflex is triggered.
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
            (
                signals["predator_occluded"] > thresholds["predator_occluded"]
                or signals["predator_smell_strength"] > thresholds["predator_smell"]
                or signals["recent_contact"] > thresholds["contact_threat"]
                or signals["recent_pain"] > thresholds["pain"]
            )
            and signals["predator_trace_strength"] > 0.0
        ):
            return self._direction_reflex(
                "alert_center",
                dx=signals["predator_trace_dx"],
                dy=signals["predator_trace_dy"],
                away=True,
                reason="retreat_from_predator_trace",
                triggers=self._signal_subset(
                    signals,
                    "predator_occluded",
                    "predator_smell_strength",
                    "recent_contact",
                    "recent_pain",
                    "predator_trace_dx",
                    "predator_trace_dy",
                    "predator_trace_strength",
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
                fixed_valence_targets = self._fixed_formula_valence_scores_from_evidence(
                    arbitration.evidence
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
            for result, extra_grad in zip(
                decision.module_results,
                per_result_input_grads,
                strict=True,
            ):
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
            for result, extra_grad in zip(
                decision.module_results,
                per_result_input_grads,
                strict=True,
            ):
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
