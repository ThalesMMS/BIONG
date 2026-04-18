"""Episode metric accumulation and representation-specialization helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Sequence

from ..ablations import PROPOSAL_SOURCE_NAMES, REFLEX_MODULE_NAMES
from .types import (
    EpisodeStats,
    PREDATOR_RESPONSE_END_THRESHOLD,
    PREDATOR_TYPE_NAMES,
    PRIMARY_REPRESENTATION_READOUT_MODULES,
    PROPOSER_REPRESENTATION_LOGIT_FIELD,
    SHELTER_ROLES,
)

def _clamp_unit_interval(value: float) -> float:
    """
    Clamp a numeric value to the closed interval [0.0, 1.0].
    
    Returns:
        float: The input coerced to float and clamped to the range 0.0 through 1.0.
    """
    return float(max(0.0, min(1.0, float(value))))


def _softmax_probabilities(logits: Sequence[float]) -> List[float]:
    """Convert logits into a probability distribution with a stable softmax."""
    values = [float(value) for value in logits]
    if not values:
        return []
    max_logit = max(values)
    exp_values = [math.exp(value - max_logit) for value in values]
    total = sum(exp_values)
    if total <= 0.0:
        return [0.0 for _ in values]
    return [float(value / total) for value in exp_values]


def jensen_shannon_divergence(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """
    Compute the base-2 Jensen-Shannon divergence between two numeric vectors.
    
    Inputs may be raw counts or already-normalized probabilities; each input must be a sequence of non-negative numbers of equal length. The inputs are normalized to probability distributions before computing the divergence. If either input is empty or sums to a value <= 0.0, the function returns 0.0.
    
    Parameters:
        left: Sequence[float] — Non-negative numeric values for the first distribution.
        right: Sequence[float] — Non-negative numeric values for the second distribution (must have the same length as `left`).
    
    Returns:
        float: Divergence clamped to the interval [0.0, 1.0]; `0.0` when distributions are identical, `1.0` when they are maximally different on the shared support.
    
    Raises:
        ValueError: If the sequences have different lengths or contain negative values.
    """
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values):
        raise ValueError("Probability distributions must have the same length.")
    if any(value < 0.0 for value in left_values + right_values):
        raise ValueError("Probability distributions cannot contain negative values.")
    if not left_values:
        return 0.0
    left_total = sum(left_values)
    right_total = sum(right_values)
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0
    left_probs = [value / left_total for value in left_values]
    right_probs = [value / right_total for value in right_values]
    midpoint = [
        0.5 * (left_value + right_value)
        for left_value, right_value in zip(left_probs, right_probs, strict=True)
    ]

    def _kl_divergence(source: Sequence[float], target: Sequence[float]) -> float:
        total = 0.0
        for source_value, target_value in zip(source, target, strict=True):
            if source_value <= 0.0 or target_value <= 0.0:
                continue
            total += source_value * math.log2(source_value / target_value)
        return float(total)

    divergence = 0.5 * _kl_divergence(left_probs, midpoint) + 0.5 * _kl_divergence(
        right_probs,
        midpoint,
    )
    return _clamp_unit_interval(divergence)


@dataclass
class EpisodeMetricAccumulator:
    reward_component_names: Sequence[str]
    predator_states: Sequence[str]
    night_ticks: int = 0
    night_shelter_ticks: int = 0
    night_still_ticks: int = 0
    night_role_ticks: Dict[str, int] = field(default_factory=dict)
    predator_response_latencies: List[int] = field(default_factory=list)
    active_predator_response: Dict[str, int] | None = None
    predator_contacts_by_type: Dict[str, int] = field(default_factory=dict)
    predator_escapes_by_type: Dict[str, int] = field(default_factory=dict)
    predator_response_latencies_by_type: Dict[str, List[int]] = field(default_factory=dict)
    active_predator_responses_by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reward_component_totals: Dict[str, float] = field(default_factory=dict)
    predator_state_ticks: Dict[str, int] = field(default_factory=dict)
    sleep_debt_samples: List[float] = field(default_factory=list)
    predator_mode_transitions: int = 0
    initial_food_dist: int | None = None
    final_food_dist: int | None = None
    initial_shelter_dist: int | None = None
    final_shelter_dist: int | None = None
    decision_steps: int = 0
    reflex_steps: int = 0
    final_reflex_override_steps: int = 0
    module_reflex_usage_steps: Dict[str, int] = field(default_factory=dict)
    module_reflex_override_steps: Dict[str, int] = field(default_factory=dict)
    module_reflex_dominance_sums: Dict[str, float] = field(default_factory=dict)
    module_contribution_share_sums: Dict[str, float] = field(default_factory=dict)
    module_response_by_predator_type_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    proposer_logits_by_predator_type: Dict[str, Dict[str, List[float]]] = field(
        default_factory=dict
    )
    proposer_probs_by_predator_type: Dict[str, Dict[str, List[float]]] = field(
        default_factory=dict
    )
    proposer_steps_by_predator_type: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    action_center_gate_sums_by_predator_type: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    action_center_gate_counts_by_predator_type: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    action_center_contribution_sums_by_predator_type: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    action_center_contribution_counts_by_predator_type: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    dominant_module_counts: Dict[str, int] = field(default_factory=dict)
    current_dominant_module: str = ""
    current_proposer_post_reflex_logits: Dict[str, List[float]] = field(default_factory=dict)
    current_proposer_probs: Dict[str, List[float]] = field(default_factory=dict)
    current_action_center_gates: Dict[str, float] = field(default_factory=dict)
    current_action_center_contribution_share: Dict[str, float] = field(default_factory=dict)
    dominant_module_share_sum: float = 0.0
    effective_module_count_sum: float = 0.0
    module_agreement_rate_sum: float = 0.0
    module_disagreement_rate_sum: float = 0.0
    learning_steps: int = 0
    module_credit_weight_sums: Dict[str, float] = field(default_factory=dict)
    module_gradient_norm_sums: Dict[str, float] = field(default_factory=dict)
    motor_execution_steps: int = 0
    motor_slip_steps: int = 0
    orientation_alignment_samples: List[float] = field(default_factory=list)
    terrain_difficulty_samples: List[float] = field(default_factory=list)
    terrain_execution_counts: Dict[str, int] = field(default_factory=dict)
    terrain_slip_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize missing accumulator maps with zeroed defaults.
        
        If a map is empty, populate it with keys taken from the corresponding configuration attributes and zero values:
        - `reward_component_totals` ← keys from `reward_component_names` with `0.0`
        - `predator_state_ticks` ← keys from `predator_states` with `0`
        - `night_role_ticks` ← keys from `SHELTER_ROLES` with `0`
        - `module_reflex_usage_steps` ← keys from `REFLEX_MODULE_NAMES` with `0`
        - `module_reflex_override_steps` ← keys from `REFLEX_MODULE_NAMES` with `0`
        - `module_reflex_dominance_sums` ← keys from `REFLEX_MODULE_NAMES` with `0.0`
        """
        if not self.reward_component_totals:
            self.reward_component_totals = {
                name: 0.0 for name in self.reward_component_names
            }
        if not self.predator_state_ticks:
            self.predator_state_ticks = {
                name: 0 for name in self.predator_states
            }
        if not self.night_role_ticks:
            self.night_role_ticks = {
                name: 0 for name in SHELTER_ROLES
            }
        for predator_type in PREDATOR_TYPE_NAMES:
            self.predator_contacts_by_type.setdefault(predator_type, 0)
            self.predator_escapes_by_type.setdefault(predator_type, 0)
            self.predator_response_latencies_by_type.setdefault(predator_type, [])
            self.module_response_by_predator_type_counts.setdefault(
                predator_type,
                {name: 0 for name in PROPOSAL_SOURCE_NAMES},
            )
            self.proposer_logits_by_predator_type.setdefault(predator_type, {})
            self.proposer_probs_by_predator_type.setdefault(predator_type, {})
            self.proposer_steps_by_predator_type.setdefault(predator_type, {})
            self.action_center_gate_sums_by_predator_type.setdefault(
                predator_type,
                {name: 0.0 for name in PROPOSAL_SOURCE_NAMES},
            )
            self.action_center_gate_counts_by_predator_type.setdefault(
                predator_type,
                {name: 0 for name in PROPOSAL_SOURCE_NAMES},
            )
            self.action_center_contribution_sums_by_predator_type.setdefault(
                predator_type,
                {name: 0.0 for name in PROPOSAL_SOURCE_NAMES},
            )
            self.action_center_contribution_counts_by_predator_type.setdefault(
                predator_type,
                {name: 0 for name in PROPOSAL_SOURCE_NAMES},
            )
        for name in REFLEX_MODULE_NAMES:
            self.module_reflex_usage_steps.setdefault(name, 0)
            self.module_reflex_override_steps.setdefault(name, 0)
            self.module_reflex_dominance_sums.setdefault(name, 0.0)
        for name in PROPOSAL_SOURCE_NAMES:
            self.module_contribution_share_sums.setdefault(name, 0.0)
            self.dominant_module_counts.setdefault(name, 0)
            self.module_credit_weight_sums.setdefault(name, 0.0)
            self.module_gradient_norm_sums.setdefault(name, 0.0)
            for predator_type in PREDATOR_TYPE_NAMES:
                self.module_response_by_predator_type_counts[predator_type].setdefault(
                    name,
                    0,
                )
                self.proposer_steps_by_predator_type[predator_type].setdefault(name, 0)
                self.action_center_gate_sums_by_predator_type[predator_type].setdefault(
                    name,
                    0.0,
                )
                self.action_center_gate_counts_by_predator_type[predator_type].setdefault(
                    name,
                    0,
                )
                self.action_center_contribution_sums_by_predator_type[predator_type].setdefault(
                    name,
                    0.0,
                )
                self.action_center_contribution_counts_by_predator_type[predator_type].setdefault(
                    name,
                    0,
                )

    @staticmethod
    def _coerce_vector(values: object) -> List[float]:
        """Best-effort conversion of an array-like object into a float list."""
        if values is None:
            return []
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError):
            return []

    @staticmethod
    def _accumulate_vector_sum(
        target: Dict[str, List[float]],
        *,
        name: str,
        values: Sequence[float],
    ) -> None:
        """Accumulate a vector into a named running sum."""
        vector = [float(value) for value in values]
        if not vector:
            return
        existing = target.get(name)
        if existing is None or len(existing) != len(vector):
            target[name] = [0.0 for _ in vector]
            existing = target[name]
        for index, value in enumerate(vector):
            existing[index] += value

    def _clear_current_representation_readouts(self) -> None:
        """Drop cached per-step representation readouts after transition accounting."""
        self.current_proposer_post_reflex_logits = {}
        self.current_proposer_probs = {}
        self.current_action_center_gates = {}
        self.current_action_center_contribution_share = {}

    def _record_representation_readouts_for_type(self, predator_type: str) -> None:
        """Accumulate proposer and action-center readouts for one predator type."""
        proposer_logit_sums = self.proposer_logits_by_predator_type.setdefault(
            predator_type,
            {},
        )
        proposer_prob_sums = self.proposer_probs_by_predator_type.setdefault(
            predator_type,
            {},
        )
        proposer_counts = self.proposer_steps_by_predator_type.setdefault(
            predator_type,
            {},
        )
        for module_name, logits in self.current_proposer_post_reflex_logits.items():
            probabilities = self.current_proposer_probs.get(
                module_name,
                _softmax_probabilities(logits),
            )
            self._accumulate_vector_sum(
                proposer_logit_sums,
                name=module_name,
                values=logits,
            )
            self._accumulate_vector_sum(
                proposer_prob_sums,
                name=module_name,
                values=probabilities,
            )
            proposer_counts[module_name] = int(proposer_counts.get(module_name, 0)) + 1

        gate_sums = self.action_center_gate_sums_by_predator_type.setdefault(
            predator_type,
            {},
        )
        gate_counts = self.action_center_gate_counts_by_predator_type.setdefault(
            predator_type,
            {},
        )
        for module_name, value in self.current_action_center_gates.items():
            gate_sums[module_name] = float(gate_sums.get(module_name, 0.0) + float(value))
            gate_counts[module_name] = int(gate_counts.get(module_name, 0)) + 1

        contribution_sums = self.action_center_contribution_sums_by_predator_type.setdefault(
            predator_type,
            {},
        )
        contribution_counts = self.action_center_contribution_counts_by_predator_type.setdefault(
            predator_type,
            {},
        )
        for module_name, value in self.current_action_center_contribution_share.items():
            contribution_sums[module_name] = float(
                contribution_sums.get(module_name, 0.0) + float(value)
            )
            contribution_counts[module_name] = int(
                contribution_counts.get(module_name, 0) + 1
            )

    def record_decision(self, decision: object) -> None:
        """
        Update accumulator counts based on a single decision, recording per-decision and per-module reflex, override, dominance, and arbitration statistics.
        
        Parameters:
            decision (object): Decision-like object with the following expected attributes:
                - module_results: iterable of result objects each exposing `name`, `reflex_applied`, `module_reflex_override`, and `module_reflex_dominance`.
                - final_reflex_override: optional boolean flag.
                - arbitration_decision: optional object that may provide `module_contribution_share` (mapping), `dominant_module`, `dominant_module_share`, `effective_module_count`, `module_agreement_rate`, and `module_disagreement_rate`.
            Only modules whose names appear in the accumulator's module maps are updated.
        """
        self.decision_steps += 1
        module_results = list(getattr(decision, "module_results", []))
        self.current_proposer_post_reflex_logits = {}
        self.current_proposer_probs = {}
        any_reflex = False
        for result in module_results:
            module_name = str(getattr(result, "name", ""))
            if module_name:
                post_reflex_logits = self._coerce_vector(
                    getattr(result, PROPOSER_REPRESENTATION_LOGIT_FIELD, None)
                )
                if post_reflex_logits:
                    self.current_proposer_post_reflex_logits[module_name] = (
                        post_reflex_logits
                    )
                    self.current_proposer_probs[module_name] = _softmax_probabilities(
                        post_reflex_logits
                    )
            if module_name not in self.module_reflex_usage_steps:
                continue
            reflex_applied = bool(getattr(result, "reflex_applied", False))
            module_override = bool(getattr(result, "module_reflex_override", False))
            dominance = float(getattr(result, "module_reflex_dominance", 0.0))
            if reflex_applied:
                any_reflex = True
                self.module_reflex_usage_steps[module_name] += 1
            if module_override:
                self.module_reflex_override_steps[module_name] += 1
            self.module_reflex_dominance_sums[module_name] += dominance
        if any_reflex:
            self.reflex_steps += 1
        if bool(getattr(decision, "final_reflex_override", False)):
            self.final_reflex_override_steps += 1
        arbitration = getattr(decision, "arbitration_decision", None)
        self.current_action_center_gates = {}
        self.current_action_center_contribution_share = {}
        if arbitration is None:
            self.current_dominant_module = ""
            return
        module_gates = getattr(arbitration, "module_gates", {})
        if isinstance(module_gates, Mapping):
            self.current_action_center_gates = {
                str(module_name): float(value)
                for module_name, value in module_gates.items()
            }
        contribution_share = getattr(arbitration, "module_contribution_share", {})
        if isinstance(contribution_share, Mapping):
            self.current_action_center_contribution_share = {
                str(module_name): float(value)
                for module_name, value in contribution_share.items()
            }
            for module_name in self.module_contribution_share_sums:
                self.module_contribution_share_sums[module_name] += float(
                    contribution_share.get(module_name, 0.0)
                )
        dominant_module = str(getattr(arbitration, "dominant_module", ""))
        if dominant_module in self.dominant_module_counts:
            self.dominant_module_counts[dominant_module] += 1
            self.current_dominant_module = dominant_module
        else:
            self.current_dominant_module = ""
        self.dominant_module_share_sum += float(
            getattr(arbitration, "dominant_module_share", 0.0)
        )
        self.effective_module_count_sum += float(
            getattr(arbitration, "effective_module_count", 0.0)
        )
        self.module_agreement_rate_sum += float(
            getattr(arbitration, "module_agreement_rate", 0.0)
        )
        self.module_disagreement_rate_sum += float(
            getattr(arbitration, "module_disagreement_rate", 0.0)
        )

    def record_learning(self, learn_stats: Mapping[str, object]) -> None:
        """
        Accumulate per-module learning-credit diagnostics from one learning step.

        Missing module entries count as zero for the step, so strategy-specific
        sparse maps still average cleanly over the whole episode.
        """
        self.learning_steps += 1
        for stats_key, target in [
            ("module_credit_weights", self.module_credit_weight_sums),
            ("module_gradient_norms", self.module_gradient_norm_sums),
        ]:
            values = learn_stats.get(stats_key, {})
            if isinstance(values, Mapping):
                for name, value in values.items():
                    module_name = str(name)
                    target.setdefault(module_name, 0.0)
                    target[module_name] += float(value)

    def record_transition(
        self,
        *,
        step: int,
        observation_meta: Dict[str, object],
        next_meta: Dict[str, object],
        info: Dict[str, object],
        state: object,
        predator_state_before: str,
        predator_state: str,
    ) -> None:
        """
        Update the accumulator with metrics observed during a single environment transition.
        
        This records reward-component totals; motor execution and slip metrics (orientation alignment, terrain difficulty, terrain slip counts); initial and final food/shelter distances; attribution of module responses to the current dominant module for a dominant predator type; predator mode transition counts and per-mode occupancy ticks; sleep-debt samples; night-related ticks (total night ticks, per-role ticks, shelter occupancy ticks, and stillness ticks); global and per-predator-type predator response event windowing and latencies; predator contacts and escapes by predator type; and related per-type module response counts.
        
        Parameters:
            step (int): Current step index in the episode.
            observation_meta (Dict[str, object]): Metadata for the current observation (e.g., "food_dist", "shelter_dist", "predator_visible").
            next_meta (Dict[str, object]): Metadata for the next observation (e.g., "food_dist", "shelter_dist", "predator_visible", "diagnostic", "night", "shelter_role", "on_shelter").
            info (Dict[str, object]): Transition info (e.g., "reward_components", "predator_contact", "motor_slip", "motor_noise_applied", "predator_escape").
            state (object): Agent state object containing at least `sleep_debt` and optional `last_move_dx`, `last_move_dy`.
            predator_state_before (str): Predator mode label before this transition.
            predator_state (str): Predator mode label after this transition.
        
        Raises:
            ValueError: If `next_meta["diagnostic"]` is present but not a mapping when a predator-response window needs diagnostic data.
        """
        reward_components = info["reward_components"]
        for name, value in reward_components.items():
            self.reward_component_totals[name] += float(value)

        motor_slip = info.get("motor_slip", {})
        has_motor_execution_info = (
            "motor_slip" in info
            or "motor_execution" in info
            or "motor_execution_components" in info
        )
        if has_motor_execution_info and isinstance(motor_slip, Mapping):
            self.motor_execution_steps += 1
            motor_slip_occurred = bool(
                motor_slip.get("occurred", info.get("motor_noise_applied", False))
            )
            if motor_slip_occurred:
                self.motor_slip_steps += 1
            components = info.get("motor_execution_components", {})
            if not isinstance(components, Mapping):
                components = motor_slip.get("components", {})
            if isinstance(components, Mapping):
                if "orientation_alignment" in components:
                    self.orientation_alignment_samples.append(
                        float(components["orientation_alignment"])
                    )
                if "terrain_difficulty" in components:
                    self.terrain_difficulty_samples.append(
                        float(components["terrain_difficulty"])
                    )
            terrain = str(motor_slip.get("terrain", "unknown"))
            self.terrain_execution_counts[terrain] = (
                self.terrain_execution_counts.get(terrain, 0) + 1
            )
            if motor_slip_occurred:
                self.terrain_slip_counts[terrain] = (
                    self.terrain_slip_counts.get(terrain, 0) + 1
                )

        if self.initial_food_dist is None:
            self.initial_food_dist = int(observation_meta["food_dist"])
        if self.initial_shelter_dist is None:
            self.initial_shelter_dist = int(observation_meta["shelter_dist"])
        self.final_food_dist = int(next_meta["food_dist"])
        self.final_shelter_dist = int(next_meta["shelter_dist"])
        representation_predator_type = _dominant_predator_type(observation_meta)
        representation_threat = _predator_type_threat(
            observation_meta,
            representation_predator_type,
        )
        if representation_predator_type and (
            representation_threat > 0.0 or bool(info.get("predator_contact"))
        ):
            self._record_representation_readouts_for_type(representation_predator_type)
        dominant_predator_type = _dominant_predator_type(next_meta)
        if dominant_predator_type and self.current_dominant_module in PROPOSAL_SOURCE_NAMES:
            threat_before = _predator_type_threat(observation_meta, dominant_predator_type)
            threat_after = _predator_type_threat(next_meta, dominant_predator_type)
            if threat_before > 0.0 or threat_after > 0.0 or bool(info["predator_contact"]):
                self.module_response_by_predator_type_counts[dominant_predator_type][
                    self.current_dominant_module
                ] += 1

        if predator_state_before != predator_state:
            self.predator_mode_transitions += 1

        if predator_state not in self.predator_state_ticks:
            self.predator_state_ticks[predator_state] = 0
        self.predator_state_ticks[predator_state] += 1
        self.sleep_debt_samples.append(float(state.sleep_debt))

        if bool(next_meta["night"]):
            self.night_ticks += 1
            role = str(next_meta["shelter_role"])
            if role not in self.night_role_ticks:
                self.night_role_ticks[role] = 0
            self.night_role_ticks[role] += 1
            if bool(next_meta["on_shelter"]):
                self.night_shelter_ticks += 1
            if getattr(state, "last_move_dx", 0) == 0 and getattr(state, "last_move_dy", 0) == 0:
                self.night_still_ticks += 1

        if self.active_predator_response is None and (
            (bool(next_meta["predator_visible"]) and not bool(observation_meta["predator_visible"]))
            or bool(info["predator_contact"])
        ):
            next_diagnostic = next_meta["diagnostic"]
            if not isinstance(next_diagnostic, Mapping):
                raise ValueError("next_meta['diagnostic'] must be a mapping")
            predator_dist = int(next_diagnostic["diagnostic_predator_dist"])
            self.active_predator_response = {
                "start_step": step,
                "start_distance": predator_dist,
            }

        for predator_type in PREDATOR_TYPE_NAMES:
            threat_before = _predator_type_threat(observation_meta, predator_type)
            threat_after = _predator_type_threat(next_meta, predator_type)
            start_triggered = (
                bool(info["predator_contact"]) and dominant_predator_type == predator_type
            ) or (threat_after > 0.0 and threat_before <= 0.0)
            if predator_type not in self.active_predator_responses_by_type and start_triggered:
                self.active_predator_responses_by_type[predator_type] = {
                    "start_step": int(step),
                    "start_distance": int(
                        _diagnostic_predator_distance_for_type(
                            next_meta,
                            predator_type,
                            state=state,
                        )
                    ),
                    "start_threat": float(threat_after),
                }

        if self.active_predator_response is not None:
            next_diagnostic = next_meta["diagnostic"]
            if not isinstance(next_diagnostic, Mapping):
                raise ValueError("next_meta['diagnostic'] must be a mapping")
            predator_dist = int(next_diagnostic["diagnostic_predator_dist"])
            if (
                str(next_meta["shelter_role"]) in {"inside", "deep"}
                or predator_dist >= self.active_predator_response["start_distance"] + 2
            ):
                self.predator_response_latencies.append(step - self.active_predator_response["start_step"])
                self.active_predator_response = None

        preserved_escape_type = ""
        for predator_type, active_response in list(self.active_predator_responses_by_type.items()):
            threat_after = _predator_type_threat(next_meta, predator_type)
            predator_dist = _diagnostic_predator_distance_for_type(
                next_meta,
                predator_type,
                state=state,
            )
            if (
                str(next_meta["shelter_role"]) in {"inside", "deep"}
                or threat_after <= PREDATOR_RESPONSE_END_THRESHOLD
                or predator_dist >= int(active_response["start_distance"]) + 2
            ):
                if bool(info.get("predator_escape")) and not preserved_escape_type:
                    preserved_escape_type = dominant_predator_type or _first_active_predator_type(
                        self.active_predator_responses_by_type
                    )
                latency = int(step - int(active_response["start_step"]))
                self.predator_response_latencies_by_type[predator_type].append(latency)
                self.active_predator_responses_by_type.pop(predator_type, None)

        if bool(info["predator_contact"]):
            for predator_type in _contact_predator_types(next_meta, state=state):
                self.predator_contacts_by_type[predator_type] += 1

        if bool(info.get("predator_escape")):
            escape_type = preserved_escape_type or dominant_predator_type or _first_active_predator_type(
                self.active_predator_responses_by_type
            )
            if escape_type:
                self.predator_escapes_by_type[escape_type] += 1
        self._clear_current_representation_readouts()

    def snapshot(self) -> Dict[str, object]:
        """
        Compute derived episode metrics and aggregated counts as a snapshot dictionary.
        
        Returns:
            snapshot (Dict[str, object]): Mapping of derived episode statistics including:
                - night_ticks: total number of night timesteps recorded.
                - night_shelter_ticks: number of night timesteps spent on a shelter.
                - night_still_ticks: number of night timesteps with no movement.
                - night_role_ticks: per-role integer counts of night timesteps.
                - night_role_distribution: per-role fraction of night timesteps (sums to <= 1.0).
                - predator_state_ticks: integer counts of timesteps spent in each predator state.
                - predator_state_occupancy: per-state fraction of predator-state occupancy.
                - predator_mode_transitions: count of predator-mode changes observed.
                - dominant_predator_state: predator state with the highest observed occupancy (defaults to "PATROL" if none).
                - predator_response_events: number of recorded predator response events.
                - mean_predator_response_latency: mean latency (in timesteps) for completed predator response events, 0.0 if none.
                - food_distance_delta: initial minus final food distance (float; uses 0 when missing).
                - shelter_distance_delta: initial minus final shelter distance (float; uses 0 when missing).
                - mean_sleep_debt: mean of collected sleep debt samples, 0.0 if none.
                - reward_component_totals: mapping of configured reward component names to their accumulated totals (floats).
        """
        night_role_distribution = _normalize_counts(self.night_role_ticks, total=self.night_ticks)
        predator_total = sum(self.predator_state_ticks.values())
        predator_state_occupancy = _normalize_counts(
            self.predator_state_ticks,
            total=predator_total,
        )
        dominant_predator_state = (
            max(self.predator_state_ticks, key=self.predator_state_ticks.get)
            if any(count > 0 for count in self.predator_state_ticks.values())
            else "PATROL"
        )
        decision_total = max(1, int(self.decision_steps))
        module_reflex_usage_rates = {
            name: float(self.module_reflex_usage_steps[name] / decision_total)
            for name in self.module_reflex_usage_steps
        }
        module_reflex_override_rates = {
            name: float(self.module_reflex_override_steps[name] / decision_total)
            for name in self.module_reflex_override_steps
        }
        module_reflex_dominance = {
            name: float(self.module_reflex_dominance_sums[name] / decision_total)
            for name in self.module_reflex_dominance_sums
        }
        module_contribution_share = {
            name: float(self.module_contribution_share_sums[name] / decision_total)
            for name in self.module_contribution_share_sums
        }
        learning_total = max(1, int(self.learning_steps))
        mean_module_credit_weights = {
            name: float(self.module_credit_weight_sums[name] / learning_total)
            for name in self.module_credit_weight_sums
        }
        module_gradient_norm_means = {
            name: float(self.module_gradient_norm_sums[name] / learning_total)
            for name in self.module_gradient_norm_sums
        }
        dominant_module_distribution = {
            name: float(self.dominant_module_counts[name] / decision_total)
            for name in self.dominant_module_counts
        }
        dominant_module = (
            max(self.dominant_module_counts, key=self.dominant_module_counts.get)
            if any(count > 0 for count in self.dominant_module_counts.values())
            else ""
        )
        terrain_slip_rates = {
            terrain: float(
                self.terrain_slip_counts.get(terrain, 0) / count
                if count
                else 0.0
            )
            for terrain, count in sorted(self.terrain_execution_counts.items())
        }
        predator_response_latency_by_type = {
            predator_type: (
                float(mean(self.predator_response_latencies_by_type.get(predator_type, [])))
                if self.predator_response_latencies_by_type.get(predator_type)
                else 0.0
            )
            for predator_type in PREDATOR_TYPE_NAMES
        }
        module_response_by_predator_type = {
            predator_type: _normalize_distribution(
                self.module_response_by_predator_type_counts.get(predator_type, {})
            )
            for predator_type in PREDATOR_TYPE_NAMES
        }
        mean_proposer_probs_by_predator_type: Dict[str, Dict[str, List[float]]] = {}
        for predator_type in PREDATOR_TYPE_NAMES:
            module_probs: Dict[str, List[float]] = {}
            prob_sums = self.proposer_probs_by_predator_type.get(predator_type, {})
            counts = self.proposer_steps_by_predator_type.get(predator_type, {})
            for module_name, summed_probs in prob_sums.items():
                step_count = int(counts.get(module_name, 0))
                if step_count <= 0:
                    continue
                module_probs[str(module_name)] = [
                    float(value / step_count) for value in summed_probs
                ]
            if module_probs:
                mean_proposer_probs_by_predator_type[predator_type] = module_probs

        proposer_divergence_by_module: Dict[str, float] = {}
        visual_probs_by_module = mean_proposer_probs_by_predator_type.get("visual", {})
        olfactory_probs_by_module = mean_proposer_probs_by_predator_type.get(
            "olfactory",
            {},
        )
        if visual_probs_by_module and olfactory_probs_by_module:
            for module_name in PRIMARY_REPRESENTATION_READOUT_MODULES:
                visual_probs = visual_probs_by_module.get(module_name)
                olfactory_probs = olfactory_probs_by_module.get(module_name)
                if visual_probs and olfactory_probs:
                    proposer_divergence_by_module[module_name] = (
                        jensen_shannon_divergence(
                            visual_probs,
                            olfactory_probs,
                        )
                    )
                else:
                    proposer_divergence_by_module[module_name] = 0.0

        action_center_gate_differential: Dict[str, float] = {}
        action_center_contribution_differential: Dict[str, float] = {}
        visual_gate_counts = self.action_center_gate_counts_by_predator_type.get(
            "visual",
            {},
        )
        olfactory_gate_counts = self.action_center_gate_counts_by_predator_type.get(
            "olfactory",
            {},
        )
        visual_contribution_counts = self.action_center_contribution_counts_by_predator_type.get(
            "visual",
            {},
        )
        olfactory_contribution_counts = self.action_center_contribution_counts_by_predator_type.get(
            "olfactory",
            {},
        )
        for module_name in PROPOSAL_SOURCE_NAMES:
            visual_gate_count = int(visual_gate_counts.get(module_name, 0))
            olfactory_gate_count = int(olfactory_gate_counts.get(module_name, 0))
            if visual_gate_count > 0 and olfactory_gate_count > 0:
                visual_mean_gate = (
                    self.action_center_gate_sums_by_predator_type["visual"].get(
                        module_name,
                        0.0,
                    )
                    / visual_gate_count
                )
                olfactory_mean_gate = (
                    self.action_center_gate_sums_by_predator_type["olfactory"].get(
                        module_name,
                        0.0,
                    )
                    / olfactory_gate_count
                )
                action_center_gate_differential[module_name] = float(
                    visual_mean_gate - olfactory_mean_gate
                )
            visual_contribution_count = int(visual_contribution_counts.get(module_name, 0))
            olfactory_contribution_count = int(
                olfactory_contribution_counts.get(module_name, 0)
            )
            if visual_contribution_count > 0 and olfactory_contribution_count > 0:
                visual_mean_contribution = (
                    self.action_center_contribution_sums_by_predator_type["visual"].get(
                        module_name,
                        0.0,
                    )
                    / visual_contribution_count
                )
                olfactory_mean_contribution = (
                    self.action_center_contribution_sums_by_predator_type["olfactory"].get(
                        module_name,
                        0.0,
                    )
                    / olfactory_contribution_count
                )
                action_center_contribution_differential[module_name] = float(
                    visual_mean_contribution - olfactory_mean_contribution
                )
        representation_specialization_score = _clamp_unit_interval(
            mean(proposer_divergence_by_module.values())
            if proposer_divergence_by_module
            else 0.0
        )
        return {
            "night_ticks": int(self.night_ticks),
            "night_shelter_ticks": int(self.night_shelter_ticks),
            "night_still_ticks": int(self.night_still_ticks),
            "night_role_ticks": {k: int(v) for k, v in self.night_role_ticks.items()},
            "night_role_distribution": night_role_distribution,
            "predator_state_ticks": {k: int(v) for k, v in self.predator_state_ticks.items()},
            "predator_state_occupancy": predator_state_occupancy,
            "predator_mode_transitions": int(self.predator_mode_transitions),
            "dominant_predator_state": dominant_predator_state,
            "predator_response_events": len(self.predator_response_latencies),
            "mean_predator_response_latency": (
                float(mean(self.predator_response_latencies))
                if self.predator_response_latencies
                else 0.0
            ),
            "predator_contacts_by_type": {
                predator_type: int(self.predator_contacts_by_type.get(predator_type, 0))
                for predator_type in PREDATOR_TYPE_NAMES
            },
            "predator_escapes_by_type": {
                predator_type: int(self.predator_escapes_by_type.get(predator_type, 0))
                for predator_type in PREDATOR_TYPE_NAMES
            },
            "predator_response_latency_by_type": predator_response_latency_by_type,
            "module_response_by_predator_type": module_response_by_predator_type,
            "proposer_divergence_by_module": proposer_divergence_by_module,
            "action_center_gate_differential": action_center_gate_differential,
            "action_center_contribution_differential": (
                action_center_contribution_differential
            ),
            "representation_specialization_score": representation_specialization_score,
            "mean_proposer_probs_by_predator_type": mean_proposer_probs_by_predator_type,
            "food_distance_delta": float(
                (self.initial_food_dist or 0) - (self.final_food_dist or 0)
            ),
            "shelter_distance_delta": float(
                (self.initial_shelter_dist or 0) - (self.final_shelter_dist or 0)
            ),
            "mean_sleep_debt": (
                float(mean(self.sleep_debt_samples))
                if self.sleep_debt_samples
                else 0.0
            ),
            "reward_component_totals": {
                name: float(self.reward_component_totals[name])
                for name in self.reward_component_names
            },
            "reflex_usage_rate": float(self.reflex_steps / decision_total),
            "final_reflex_override_rate": float(
                self.final_reflex_override_steps / decision_total
            ),
            "module_reflex_usage_rates": module_reflex_usage_rates,
            "module_reflex_override_rates": module_reflex_override_rates,
            "module_reflex_dominance": module_reflex_dominance,
            "mean_reflex_dominance": (
                float(mean(module_reflex_dominance.values()))
                if module_reflex_dominance
                else 0.0
            ),
            "module_contribution_share": module_contribution_share,
            "mean_module_credit_weights": mean_module_credit_weights,
            "module_gradient_norm_means": module_gradient_norm_means,
            "motor_slip_rate": float(
                self.motor_slip_steps / self.motor_execution_steps
                if self.motor_execution_steps
                else 0.0
            ),
            "mean_orientation_alignment": (
                float(mean(self.orientation_alignment_samples))
                if self.orientation_alignment_samples
                else 0.0
            ),
            "mean_terrain_difficulty": (
                float(mean(self.terrain_difficulty_samples))
                if self.terrain_difficulty_samples
                else 0.0
            ),
            "terrain_slip_rates": terrain_slip_rates,
            "dominant_module": dominant_module,
            "dominant_module_distribution": dominant_module_distribution,
            "dominant_module_share": float(
                self.dominant_module_share_sum / decision_total
            ),
            "effective_module_count": float(
                self.effective_module_count_sum / decision_total
            ),
            "module_agreement_rate": float(
                self.module_agreement_rate_sum / decision_total
            ),
            "module_disagreement_rate": float(
                self.module_disagreement_rate_sum / decision_total
            ),
        }

    def finalize(
        self,
        *,
        episode: int,
        seed: int,
        training: bool,
        scenario: str | None,
        total_reward: float,
        state: object,
    ) -> EpisodeStats:
        """
        Create an EpisodeStats containing consolidated metrics and final-state fields for an episode.
        
        Parameters:
            episode (int): Episode index.
            seed (int): Random seed used for the episode.
            training (bool): Whether the episode was collected in training mode.
            scenario (str | None): Optional scenario identifier.
            total_reward (float): Cumulative reward for the episode.
            state (object): Final environment state exposing numeric attributes used for summary:
                steps_alive, food_eaten, sleep_events, shelter_entries,
                alert_events, predator_contacts, predator_sightings, predator_escapes,
                hunger, fatigue, sleep_debt, and health.
        
        Returns:
            EpisodeStats: Consolidated episode summary including episode metadata (episode, seed, training, scenario, total_reward), event counters (steps, food/sleep/shelter/predator counts, alert events), night summaries (ticks, per-role ticks and distribution, shelter occupancy and stillness rates), predator statistics (state tick counts, mode transition count, dominant state, per-type contacts/escapes, response event counts and mean latencies, module-response-by-predator-type distributions), distance deltas, mean sleep debt, reward component totals, reflex and per-module metrics (usage, override, dominance, contribution shares, learning/gradient summaries), motor/terrain slip metrics, dominant module statistics, final physiological metrics (hunger, fatigue, sleep_debt, health) and an alive flag.
        """
        snapshot = self.snapshot()
        night_ticks = max(0, int(snapshot["night_ticks"]))
        night_shelter_occupancy_rate = (
            self.night_shelter_ticks / night_ticks if night_ticks else 0.0
        )
        night_stillness_rate = (
            self.night_still_ticks / night_ticks if night_ticks else 0.0
        )
        return EpisodeStats(
            episode=episode,
            seed=seed,
            training=training,
            scenario=scenario,
            total_reward=float(total_reward),
            steps=int(state.steps_alive),
            food_eaten=int(state.food_eaten),
            sleep_events=int(state.sleep_events),
            shelter_entries=int(state.shelter_entries),
            alert_events=int(state.alert_events),
            predator_contacts=int(state.predator_contacts),
            predator_sightings=int(state.predator_sightings),
            predator_escapes=int(state.predator_escapes),
            night_ticks=night_ticks,
            night_shelter_ticks=int(self.night_shelter_ticks),
            night_still_ticks=int(self.night_still_ticks),
            night_role_ticks={role: int(self.night_role_ticks[role]) for role in SHELTER_ROLES},
            night_shelter_occupancy_rate=float(night_shelter_occupancy_rate),
            night_stillness_rate=float(night_stillness_rate),
            night_role_distribution={
                role: float(snapshot["night_role_distribution"][role])
                for role in SHELTER_ROLES
            },
            predator_response_events=len(self.predator_response_latencies),
            mean_predator_response_latency=float(snapshot["mean_predator_response_latency"]),
            mean_sleep_debt=float(snapshot["mean_sleep_debt"]),
            food_distance_delta=float(snapshot["food_distance_delta"]),
            shelter_distance_delta=float(snapshot["shelter_distance_delta"]),
            final_hunger=float(state.hunger),
            final_fatigue=float(state.fatigue),
            final_sleep_debt=float(state.sleep_debt),
            final_health=float(state.health),
            alive=bool(state.health > 0.0),
            reward_component_totals={
                name: float(self.reward_component_totals[name])
                for name in self.reward_component_names
            },
            predator_state_ticks={
                name: int(self.predator_state_ticks[name])
                for name in self.predator_states
            },
            predator_mode_transitions=int(snapshot["predator_mode_transitions"]),
            dominant_predator_state=str(snapshot["dominant_predator_state"]),
            predator_contacts_by_type={
                predator_type: int(
                    snapshot["predator_contacts_by_type"].get(predator_type, 0)
                )
                for predator_type in PREDATOR_TYPE_NAMES
            },
            predator_escapes_by_type={
                predator_type: int(
                    snapshot["predator_escapes_by_type"].get(predator_type, 0)
                )
                for predator_type in PREDATOR_TYPE_NAMES
            },
            predator_response_latency_by_type={
                predator_type: float(
                    snapshot["predator_response_latency_by_type"].get(predator_type, 0.0)
                )
                for predator_type in PREDATOR_TYPE_NAMES
            },
            module_response_by_predator_type={
                predator_type: {
                    name: float(value)
                    for name, value in (
                        snapshot["module_response_by_predator_type"]
                        .get(predator_type, {})
                        .items()
                    )
                }
                for predator_type in PREDATOR_TYPE_NAMES
            },
            proposer_divergence_by_module={
                str(name): float(value)
                for name, value in snapshot["proposer_divergence_by_module"].items()
            },
            action_center_gate_differential={
                str(name): float(value)
                for name, value in snapshot["action_center_gate_differential"].items()
            },
            action_center_contribution_differential={
                str(name): float(value)
                for name, value in snapshot[
                    "action_center_contribution_differential"
                ].items()
            },
            representation_specialization_score=float(
                snapshot["representation_specialization_score"]
            ),
            mean_proposer_probs_by_predator_type={
                predator_type: {
                    str(module_name): [float(value) for value in probs]
                    for module_name, probs in (
                        snapshot["mean_proposer_probs_by_predator_type"]
                        .get(predator_type, {})
                        .items()
                    )
                }
                for predator_type in PREDATOR_TYPE_NAMES
                if snapshot["mean_proposer_probs_by_predator_type"].get(predator_type)
            },
            reflex_usage_rate=float(snapshot["reflex_usage_rate"]),
            final_reflex_override_rate=float(snapshot["final_reflex_override_rate"]),
            mean_reflex_dominance=float(snapshot["mean_reflex_dominance"]),
            module_reflex_usage_rates={
                name: float(snapshot["module_reflex_usage_rates"][name])
                for name in REFLEX_MODULE_NAMES
            },
            module_reflex_override_rates={
                name: float(snapshot["module_reflex_override_rates"][name])
                for name in REFLEX_MODULE_NAMES
            },
            module_reflex_dominance={
                name: float(snapshot["module_reflex_dominance"][name])
                for name in REFLEX_MODULE_NAMES
            },
            module_contribution_share={
                name: float(snapshot["module_contribution_share"][name])
                for name in PROPOSAL_SOURCE_NAMES
            },
            mean_module_credit_weights={
                name: float(value)
                for name, value in snapshot["mean_module_credit_weights"].items()
            },
            module_gradient_norm_means={
                name: float(value)
                for name, value in snapshot["module_gradient_norm_means"].items()
            },
            motor_slip_rate=float(snapshot["motor_slip_rate"]),
            mean_orientation_alignment=float(snapshot["mean_orientation_alignment"]),
            mean_terrain_difficulty=float(snapshot["mean_terrain_difficulty"]),
            terrain_slip_rates={
                str(name): float(value)
                for name, value in snapshot["terrain_slip_rates"].items()
            },
            dominant_module=str(snapshot["dominant_module"]),
            dominant_module_share=float(snapshot["dominant_module_share"]),
            effective_module_count=float(snapshot["effective_module_count"]),
            module_agreement_rate=float(snapshot["module_agreement_rate"]),
            module_disagreement_rate=float(snapshot["module_disagreement_rate"]),
        )


def _normalize_counts(counts: Dict[str, int], *, total: int) -> Dict[str, float]:
    """
    Normalize integer counts into fractional proportions using the provided total.
    
    Parameters:
        counts (Dict[str, int]): Mapping of names to integer counts.
        total (int): Denominator used to normalize counts.
    
    Returns:
        Dict[str, float]: Mapping of the same names to their normalized fraction (count / total). If `total` is less than or equal to 0, returns all zeros.
    """
    if total <= 0:
        return {name: 0.0 for name in counts}
    return {
        name: float(value / total)
        for name, value in counts.items()
    }


def _normalize_distribution(values: Mapping[str, int]) -> Dict[str, float]:
    """
    Normalize an integer-count mapping into a distribution of floats that sum to 1.0 (or all zeros if total is zero).
    
    Parameters:
        values (Mapping[str, int]): Mapping from keys to integer counts to normalize.
    
    Returns:
        Dict[str, float]: Mapping with the same keys and values as floats equal to count / total.
            If the sum of counts is less than or equal to zero, returns all keys mapped to 0.0.
    """
    total = sum(int(value) for value in values.values())
    if total <= 0:
        return {str(name): 0.0 for name in values}
    return {
        str(name): float(int(value) / total)
        for name, value in values.items()
    }


def _predator_type_threat(meta: Mapping[str, object], predator_type: str) -> float:
    """
    Retrieve the predator threat value for a given predator type from metadata.
    
    Parameters:
        meta (Mapping[str, object]): Metadata mapping that may contain keys of the form "<predator_type>_predator_threat".
        predator_type (str): Predator type label (e.g., "visual" or "olfactory") used to form the metadata key.
    
    Returns:
        float: The threat value coerced to float; returns 0.0 if the key is missing or the value is falsy.
    """
    return float(meta.get(f"{predator_type}_predator_threat", 0.0) or 0.0)


def _dominant_predator_type(meta: Mapping[str, object]) -> str:
    """
    Determine the dominant predator type label from diagnostic metadata.
    
    Examines `meta["dominant_predator_type_label"]` (case-insensitive) and returns it if it matches a known predator type. If no valid label is present, compares numeric threat values for "visual" and "olfactory" (read via keys like `"visual_predator_threat"` / `"olfactory_predator_threat"`) and returns the type with the larger threat. Returns an empty string when both threats are zero or absent.
    
    Parameters:
        meta (Mapping[str, object]): Diagnostic metadata that may include
            - "dominant_predator_type_label": a preferred label (string)
            - "<type>_predator_threat": numeric threat values for "visual" and "olfactory"
    
    Returns:
        The dominant predator type: "visual" or "olfactory", or an empty string if none is dominant.
    """
    label = str(meta.get("dominant_predator_type_label") or "").strip().lower()
    if label in PREDATOR_TYPE_NAMES:
        return label
    visual_threat = _predator_type_threat(meta, "visual")
    olfactory_threat = _predator_type_threat(meta, "olfactory")
    if visual_threat <= 0.0 and olfactory_threat <= 0.0:
        return ""
    return "olfactory" if olfactory_threat > visual_threat else "visual"


def _diagnostic_predator_distance(meta: Mapping[str, object]) -> int:
    """
    Extract the diagnostic predator distance from a metadata mapping.
    
    Reads the "diagnostic" mapping from `meta` and returns the value of
    "diagnostic_predator_dist" coerced to an integer. If the diagnostic
    mapping or the distance value is missing or not a number, returns 0.
    
    Parameters:
        meta (Mapping[str, object]): Metadata that may contain a "diagnostic" mapping.
    
    Returns:
        int: The diagnostic predator distance, or 0 when unavailable or invalid.
    """
    diagnostic = meta.get("diagnostic", {})
    if not isinstance(diagnostic, Mapping):
        return 0
    value = diagnostic.get("diagnostic_predator_dist", 0)
    if value in (None, ""):
        return 0
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(numeric_value):
        return 0
    return int(numeric_value)


def _diagnostic_predator_distance_for_type(
    meta: Mapping[str, object],
    predator_type: str,
    *,
    state: object,
) -> int:
    """Return the nearest diagnostic predator distance for a specific predator type."""
    spider_x = getattr(state, "x", None)
    spider_y = getattr(state, "y", None)
    if spider_x is None or spider_y is None:
        return _diagnostic_predator_distance(meta)
    distances: list[int] = []
    predators = meta.get("predators", [])
    if isinstance(predators, list):
        for predator in predators:
            if not isinstance(predator, Mapping):
                continue
            profile = predator.get("profile", {})
            if not isinstance(profile, Mapping):
                continue
            detection_style = str(profile.get("detection_style") or "").strip().lower()
            if detection_style != predator_type:
                continue
            try:
                predator_x = int(predator["x"])
                predator_y = int(predator["y"])
            except (KeyError, TypeError, ValueError):
                continue
            distances.append(abs(int(spider_x) - predator_x) + abs(int(spider_y) - predator_y))
    if distances:
        return min(distances)
    return _diagnostic_predator_distance(meta)


def _first_active_predator_type(active: Mapping[str, object]) -> str:
    """
    Selects the first predator type key from the active mapping that matches the known predator type order.
    
    Parameters:
        active (Mapping[str, object]): Mapping whose keys may include predator type names.
    
    Returns:
        str: The first predator type (one of PREDATOR_TYPE_NAMES) found in `active`, or an empty string if none are present.
    """
    for predator_type in PREDATOR_TYPE_NAMES:
        if predator_type in active:
            return predator_type
    return ""


def _coerce_grid_coordinate(value: object) -> int | None:
    """Return an integer grid coordinate, or None when the value is malformed."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _contact_predator_types(
    meta: Mapping[str, object],
    *,
    state: object,
) -> list[str]:
    """
    Identify predator detection styles present at the agent's integer grid location.
    
    Scans meta["predators"] for predators whose integer (x, y) coordinates match the agent state's x and y and returns the unique detection-style labels (normalized to lowercase) that are members of PREDATOR_TYPE_NAMES. If the state does not expose x/y, or no predators occupy the agent's cell, falls back to the dominant predator type from meta; returns a single-element list with that type when available, otherwise returns an empty list.
    
    Parameters:
        meta (Mapping[str, object]): Metadata potentially containing a "predators" list and fields used to determine the dominant predator type.
        state (object): Agent state expected to expose numeric attributes `x` and `y`.
    
    Returns:
        list[str]: List of predator type labels present at the agent's integer grid cell, or a single-element list containing the dominant predator type, or an empty list if no type can be determined.
    """
    spider_x = getattr(state, "x", None)
    spider_y = getattr(state, "y", None)
    spider_grid_x = _coerce_grid_coordinate(spider_x)
    spider_grid_y = _coerce_grid_coordinate(spider_y)
    if spider_grid_x is None or spider_grid_y is None:
        dominant_type = _dominant_predator_type(meta)
        return [dominant_type] if dominant_type else []
    contact_types: list[str] = []
    predators = meta.get("predators", [])
    if isinstance(predators, list):
        for predator in predators:
            if not isinstance(predator, Mapping):
                continue
            predator_x = _coerce_grid_coordinate(predator.get("x"))
            predator_y = _coerce_grid_coordinate(predator.get("y"))
            if predator_x is None or predator_y is None:
                continue
            if predator_x != spider_grid_x or predator_y != spider_grid_y:
                continue
            profile = predator.get("profile", {})
            if not isinstance(profile, Mapping):
                continue
            detection_style = str(profile.get("detection_style") or "").strip().lower()
            if detection_style in PREDATOR_TYPE_NAMES and detection_style not in contact_types:
                contact_types.append(detection_style)
    if contact_types:
        return contact_types
    dominant_type = _dominant_predator_type(meta)
    return [dominant_type] if dominant_type else []
