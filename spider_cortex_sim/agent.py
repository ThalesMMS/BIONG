from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Sequence, Set

import numpy as np

from .ablations import (
    BrainAblationConfig,
    TRUE_MONOLITHIC_POLICY_NAME as DEFAULT_TRUE_MONOLITHIC_POLICY_NAME,
    default_brain_config,
)
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
from .capacity_profiles import CapacityProfile, resolve_capacity_profile
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
from .nn import (
    ArbitrationNetwork,
    MotorNetwork,
    ProposalNetwork,
    TrueMonolithicNetwork,
    one_hot,
    softmax,
)
from .noise import _compute_execution_difficulty_core
from .operational_profiles import OperationalProfile, runtime_operational_profile
from .reflexes import (
    _apply_reflex_path as apply_reflex_path,
    _direction_action as direction_action,
    _module_reflex_decision as module_reflex_decision,
)
from .world import ACTIONS

from .brain.inputs import BrainInputMixin
from .brain.learning import BrainLearningMixin
from .brain.persistence import BrainPersistenceMixin
from .brain.runtime import BrainRuntimeMixin
from .brain.types import BrainStep


class SpiderBrain(BrainInputMixin, BrainRuntimeMixin, BrainLearningMixin, BrainPersistenceMixin):
    """Neuro-modular brain with interface-standardized locomotion proposals.

    Explicit memory remains in the world and arrives here only as named observations.
    The world remains the owner of ecological state and explicit memory; `interfaces.py`
    defines the named contracts, while `modules.py` runs only the neural proposers.
    This is where the interpretable local reflexes and final motor arbitration live.
    """

    ARCHITECTURE_VERSION = 14
    _METADATA_FILE = "metadata.json"
    MONOLITHIC_POLICY_NAME = DEFAULT_MONOLITHIC_POLICY_NAME
    TRUE_MONOLITHIC_POLICY_NAME = DEFAULT_TRUE_MONOLITHIC_POLICY_NAME
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
            "perception_center": "support",
            "homeostasis_center": "hunger",
            "threat_center": "threat",
            MONOLITHIC_POLICY_NAME: "integrated_policy",
            TRUE_MONOLITHIC_POLICY_NAME: "integrated_policy",
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
        capacity_profile: str | CapacityProfile | None = None,
        operational_profile: str | OperationalProfile | None = None,
    ) -> None:
        """
        Initialize a SpiderBrain, setting up RNGs, resolving the operational profile and ablation config, and constructing the proposal (modular or monolithic), motor/value, and arbitration networks.
        
        Parameters:
            seed (int): RNG seed for reproducible initialization.
            gamma (float): Discount factor used by learning updates.
            module_lr (float): Learning rate for proposal modules or the monolithic proposal.
            motor_lr (float): Learning rate for the motor/value network.
            module_dropout (float): Default module dropout probability when no config is provided.
            arbitration_lr (float | None): Learning rate for the learned arbitration network; defaults to `module_lr` when omitted.
            arbitration_regularization_weight (float): Strength of the regularizer that pulls learned arbitration gates toward the fixed priority-gate baseline; set to `0.0` to disable.
            arbitration_valence_regularization_weight (float): Strength of the regularizer that pulls learned valence logits toward the fixed-formula valence distribution; set to `0.0` to disable.
            config (BrainAblationConfig | None): Optional ablation/config that selects modular vs monolithic mode and may override defaults (e.g., module dropout); when omitted a default config is created.
            operational_profile (str | OperationalProfile | None): Operational profile or its name; resolved into runtime reflex parameters (auxiliary weights, logit strengths, thresholds) used by reflex decision logic.
        """
        self.rng = np.random.default_rng(seed)
        self.arbitration_rng = np.random.default_rng(int(seed) + 104729)
        self.action_dim = len(ACTIONS)
        resolved_capacity_profile = resolve_capacity_profile(
            capacity_profile
            if capacity_profile is not None
            else (
                config.capacity_profile
                if config is not None and config.capacity_profile is not None
                else (
                    config.capacity_profile_name
                    if config is not None
                    else None
                )
            )
        )
        if config is not None:
            source_capacity_profile = resolve_capacity_profile(
                config.capacity_profile
                if config.capacity_profile is not None
                else config.capacity_profile_name
            )
            source_module_hidden_dims = dict(source_capacity_profile.module_hidden_dims)
            config_module_hidden_dims = dict(config.module_hidden_dims)
            explicit_module_hidden_dims = {
                name: hidden_dim
                for name, hidden_dim in config_module_hidden_dims.items()
                if source_module_hidden_dims.get(name) != hidden_dim
            }
            module_hidden_dims = {
                **dict(resolved_capacity_profile.module_hidden_dims),
                **explicit_module_hidden_dims,
            }

            def _config_dim_or_profile(
                field_name: str,
                source_default: int,
                profile_default: int,
            ) -> int:
                value = getattr(config, field_name)
                if value is not None and int(value) != int(source_default):
                    return int(value)
                return int(profile_default)

            action_center_hidden_dim = _config_dim_or_profile(
                "action_center_hidden_dim",
                source_capacity_profile.action_center_hidden_dim,
                resolved_capacity_profile.action_center_hidden_dim,
            )
            arbitration_hidden_dim = _config_dim_or_profile(
                "arbitration_hidden_dim",
                source_capacity_profile.arbitration_hidden_dim,
                resolved_capacity_profile.arbitration_hidden_dim,
            )
            motor_hidden_dim = _config_dim_or_profile(
                "motor_hidden_dim",
                source_capacity_profile.motor_hidden_dim,
                resolved_capacity_profile.motor_hidden_dim,
            )
            self.config = replace(
                config,
                capacity_profile=resolved_capacity_profile,
                capacity_profile_name=resolved_capacity_profile.name,
                module_hidden_dims=module_hidden_dims,
                action_center_hidden_dim=action_center_hidden_dim,
                arbitration_hidden_dim=arbitration_hidden_dim,
                motor_hidden_dim=motor_hidden_dim,
                integration_hidden_dim=action_center_hidden_dim,
            )
        else:
            self.config = default_brain_config(
                module_dropout=module_dropout,
                capacity_profile=resolved_capacity_profile,
            )
        self.operational_profile = runtime_operational_profile(operational_profile)
        self.reflex_aux_weights = self.operational_profile.brain_aux_weights
        self.reflex_logit_strengths = self.operational_profile.brain_reflex_logit_strengths
        self.reflex_thresholds = self.operational_profile.brain_reflex_thresholds
        self.current_reflex_scale = float(self.config.reflex_scale)
        self.module_bank: CorticalModuleBank | None = None
        self.monolithic_policy: ProposalNetwork | None = None
        self.true_monolithic_policy: TrueMonolithicNetwork | None = None
        self.action_center: MotorNetwork | None = None
        self.motor_cortex: ProposalNetwork | None = None
        self.arbitration_network: ArbitrationNetwork | None = None
        self._frozen_modules: Set[str] = set()
        module_hidden_dims = dict(self.config.module_hidden_dims)
        action_center_hidden_dim = int(self.config.action_center_hidden_dim)
        arbitration_hidden_dim = int(self.config.arbitration_hidden_dim)
        motor_hidden_dim = int(self.config.motor_hidden_dim)
        monolithic_hidden_dim = int(sum(module_hidden_dims.values()))
        if self.config.is_modular:
            self.module_bank = CorticalModuleBank(
                action_dim=self.action_dim,
                rng=self.rng,
                module_dropout=self.config.module_dropout,
                disabled_modules=self.config.disabled_modules,
                recurrent_modules=self.config.recurrent_modules,
                hidden_dims=module_hidden_dims,
            )
            action_input_dim = self.action_dim * len(self.module_bank.enabled_specs) + ACTION_CONTEXT_INTERFACE.input_dim
            motor_input_dim = self.action_dim + MOTOR_CONTEXT_INTERFACE.input_dim
            self.action_center = MotorNetwork(
                input_dim=action_input_dim,
                hidden_dim=action_center_hidden_dim,
                output_dim=self.action_dim,
                rng=self.rng,
                name="action_center",
            )
            self.motor_cortex = ProposalNetwork(
                input_dim=motor_input_dim,
                hidden_dim=motor_hidden_dim,
                output_dim=self.action_dim,
                rng=self.rng,
                name="motor_cortex",
            )
        elif self.config.is_monolithic:
            monolithic_input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
            self.monolithic_policy = ProposalNetwork(
                input_dim=monolithic_input_dim,
                hidden_dim=monolithic_hidden_dim,
                output_dim=self.action_dim,
                rng=self.rng,
                name=self.MONOLITHIC_POLICY_NAME,
            )
            action_input_dim = self.action_dim + ACTION_CONTEXT_INTERFACE.input_dim
            motor_input_dim = self.action_dim + MOTOR_CONTEXT_INTERFACE.input_dim
            self.action_center = MotorNetwork(
                input_dim=action_input_dim,
                hidden_dim=action_center_hidden_dim,
                output_dim=self.action_dim,
                rng=self.rng,
                name="action_center",
            )
            self.motor_cortex = ProposalNetwork(
                input_dim=motor_input_dim,
                hidden_dim=motor_hidden_dim,
                output_dim=self.action_dim,
                rng=self.rng,
                name="motor_cortex",
            )
        else:
            monolithic_input_dim = sum(spec.input_dim for spec in MODULE_INTERFACES)
            self.true_monolithic_policy = TrueMonolithicNetwork(
                input_dim=monolithic_input_dim,
                hidden_dim=monolithic_hidden_dim,
                output_dim=self.action_dim,
                rng=self.rng,
                name=self.TRUE_MONOLITHIC_POLICY_NAME,
            )
        if not self.config.is_true_monolithic:
            arbitration_input_dim = arbitration_evidence_input_dim(
                arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
            )
            self.arbitration_network = ArbitrationNetwork(
                input_dim=arbitration_input_dim,
                hidden_dim=arbitration_hidden_dim,
                rng=self.arbitration_rng,
                name=self.ARBITRATION_NETWORK_NAME,
                gate_adjustment_min=self.config.gate_adjustment_bounds[0],
                gate_adjustment_max=self.config.gate_adjustment_bounds[1],
            )
            self._warm_start_arbitration_network(self.config.warm_start_scale)
        if self.config.name == "learned_arbitration_no_regularization":
            arbitration_regularization_weight = 0.0
            arbitration_valence_regularization_weight = 0.0
        if not self.config.use_learned_arbitration or self.config.is_true_monolithic:
            arbitration_regularization_weight = 0.0
            arbitration_valence_regularization_weight = 0.0
        self.gamma = gamma
        self.module_lr = module_lr
        self.arbitration_lr = module_lr if arbitration_lr is None else float(arbitration_lr)
        self.arbitration_regularization_weight = float(arbitration_regularization_weight)
        self.arbitration_valence_regularization_weight = float(arbitration_valence_regularization_weight)
        self.motor_lr = motor_lr
        self.module_dropout = self.config.module_dropout

    def freeze_proposers(
        self,
        module_names: Sequence[str] | None = None,
    ) -> None:
        """
        Freeze the active proposer stage so it remains available for inference but is excluded from learning updates.

        Parameters:
            module_names (Sequence[str] | None): Specific proposer names to freeze. When omitted, freezes every
                currently active proposer for this architecture.
        """
        available = set(self._proposal_stage_names())
        target_names = available if module_names is None else {str(name) for name in module_names}
        invalid = sorted(name for name in target_names if name not in available)
        if invalid:
            raise ValueError(
                f"Cannot freeze unknown proposer modules for this architecture: {invalid}."
            )
        self._frozen_modules.update(target_names)

    def unfreeze_proposers(
        self,
        module_names: Sequence[str] | None = None,
    ) -> None:
        """
        Remove proposer modules from the frozen set.

        Parameters:
            module_names (Sequence[str] | None): Specific proposer names to unfreeze. When omitted, clears the
                entire frozen-proposer set.
        """
        if module_names is None:
            self._frozen_modules.clear()
            return
        for name in module_names:
            self._frozen_modules.discard(str(name))

    def frozen_module_names(self) -> List[str]:
        """
        Return the sorted names of proposer modules currently frozen for learning.
        """
        return sorted(self._frozen_modules)

    def is_module_frozen(self, name: str) -> bool:
        """
        Report whether a proposer module is currently frozen for learning.
        """
        return str(name) in self._frozen_modules

    @staticmethod
    def _clamp_unit(value: float) -> float:
        """
        Clamp a numeric value to the unit interval.

        Returns:
            The input constrained to the range 0.0 through 1.0.
        """
        return clamp_unit(value)

    def _warm_start_arbitration_network(
        self,
        warm_start_scale: float | None = None,
    ) -> None:
        """
        Reinitializes the arbitration network weights using the configured warm-start procedure.

        Calls the shared warm-start helper to set arbitration network parameters based on the brain's
        ablation configuration, class-level valence/evidence ordering and weights, and an optional
        scale override.

        Parameters:
            warm_start_scale (float | None): Optional multiplier to scale the warm-start initialization.
                If `None`, the value from the brain configuration is used.
        """
        if self.arbitration_network is None:
            return
        warm_start_arbitration_network(
            self.arbitration_network,
            ablation_config=self.config,
            warm_start_scale=warm_start_scale,
            valence_order=self.VALENCE_ORDER,
            arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
            valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
        )

    def _fixed_formula_valence_scores_from_evidence(
        self,
        evidence: Dict[str, Dict[str, float]],
    ) -> np.ndarray:
        """
        Compute valence scores for each valence role from module evidence using the class's fixed-formula weights and ordering.

        Parameters:
            evidence (dict): Mapping from module name to a mapping of evidence-field names to numeric values.

        Returns:
            numpy.ndarray: Array of valence scores in the order defined by `self.VALENCE_ORDER`; each score is clamped to the unit interval [0, 1].
        """
        return fixed_formula_valence_scores_from_evidence(
            evidence,
            valence_order=self.VALENCE_ORDER,
            valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
            clamp_fn=self._clamp_unit,
        )

    def _compute_arbitration(
        self,
        module_results: List[ModuleResult],
        observation: Dict[str, np.ndarray],
        *,
        training: bool = False,
        store_cache: bool = True,
    ) -> ArbitrationDecision:
        """
        Combine module proposals and the current observation to produce an arbitration decision using this brain's arbitration network and configuration.

        Parameters:
            module_results (List[ModuleResult]): Results produced by proposal modules to be evaluated by arbitration.
            observation (Dict[str, np.ndarray]): Mapping from interface names to observation arrays used to compute arbitration evidence.
            training (bool): If true, run the arbitration network in training mode (e.g., enabling stochastic behavior or dropout).
            store_cache (bool): If true, store intermediate arbitration evidence/diagnostics in the arbitration network's cache.

        Returns:
            ArbitrationDecision: Decision object containing arbitration outputs (gate values/logits), computed evidence, and related diagnostics.
        """
        return compute_arbitration(
            observation,
            module_results,
            arbitration_network=self.arbitration_network,
            ablation_config=self.config,
            arbitration_rng=self.arbitration_rng,
            operational_profile=self.operational_profile,
            training=training,
            store_cache=store_cache,
            clamp_fn=self._clamp_unit,
            valence_order=self.VALENCE_ORDER,
            arbitration_evidence_fields=self.ARBITRATION_EVIDENCE_FIELDS,
            valence_evidence_weights=self.VALENCE_EVIDENCE_WEIGHTS,
            arbitration_gate_module_order=self.ARBITRATION_GATE_MODULE_ORDER,
            priority_gating_weights=self.PRIORITY_GATING_WEIGHTS,
        )

    def _module_reflex_decision(
        self,
        result: ModuleResult,
    ) -> ReflexDecision | None:
        """
        Compute a reflex decision for a module's result using the current operational profile.

        Parameters:
            result (ModuleResult): The module's result (including its name and observation) to evaluate for a reflex.

        Returns:
            ReflexDecision | None: `ReflexDecision` if a reflex applies to the module result, `None` otherwise.
        """
        return module_reflex_decision(
            result.name,
            result.named_observation(),
            operational_profile=self.operational_profile,
            interface_registry=self._interface_registry(),
        )
