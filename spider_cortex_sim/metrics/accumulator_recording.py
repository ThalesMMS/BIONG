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

from .accumulator_math import _clamp_unit_interval, _softmax_probabilities, jensen_shannon_divergence, _normalize_counts, _normalize_distribution, _predator_type_threat, _dominant_predator_type, _diagnostic_predator_distance, _diagnostic_predator_distance_for_type, _first_active_predator_type, _coerce_grid_coordinate, _contact_predator_types


class AccumulatorRecordingMixin:
    def __post_init__(self) -> None:
        """
        Initialize missing accumulator maps with zeroed defaults.
        
        Ensure maps contain keys taken from the corresponding configuration attributes and zero values:
        - `reward_component_totals` ← keys from `reward_component_names` with `0.0`
        - `predator_state_ticks` ← keys from `predator_states` with `0`
        - `night_role_ticks` ← keys from `SHELTER_ROLES` with `0`
        - `module_reflex_usage_steps` ← keys from `REFLEX_MODULE_NAMES` with `0`
        - `module_reflex_override_steps` ← keys from `REFLEX_MODULE_NAMES` with `0`
        - `module_reflex_dominance_sums` ← keys from `REFLEX_MODULE_NAMES` with `0.0`
        """
        for name in self.reward_component_names:
            self.reward_component_totals.setdefault(name, 0.0)
        for name in self.predator_states:
            self.predator_state_ticks.setdefault(name, 0)
        for name in SHELTER_ROLES:
            self.night_role_ticks.setdefault(name, 0)
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
        """
        Convert an array-like input into a list of floats if possible.
        
        If `values` is None, returns an empty list. If `values` is iterable, attempts to convert each element to `float` and returns the resulting list. If iteration or conversion fails (e.g., `values` is non-iterable or elements cannot be cast to float), returns an empty list.
        
        Parameters:
            values (object): An array-like object or None.
        
        Returns:
            List[float]: A list of floats parsed from `values`, or an empty list on failure or when `values` is None.
        """
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
        """
        Add a numeric vector into the named running sum stored in `target`.
        
        Parameters:
            target (Dict[str, List[float]]): Mapping of names to accumulated float vectors; the entry
                for `name` will be created or replaced with a zero vector if missing or of a different
                length than `values`.
            name (str): Key in `target` under which to accumulate the vector.
            values (Sequence[float]): Sequence of numeric values to add elementwise; values are cast
                to `float`. If `values` is empty or cannot be converted into a non-empty float list,
                no change is made.
        
        Notes:
            - If an existing vector for `name` has a different length than `values`, it is replaced
              with a zero vector of matching length before accumulation.
        """
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
        """
        Accumulates the cached proposer logits/probabilities and action-center gate/contribution readouts into the running per-predator-type aggregates.
        
        This records, for the given predator type:
        - Per-module logit and probability vectors are added to the corresponding running sums and the per-module step counts are incremented. If a probability vector for a module is not cached, it is derived from the module's logits before accumulation.
        - Per-module action-center gate values are added to gate sums and gate counts are incremented.
        - Per-module action-center contribution-share values are added to contribution sums and contribution counts are incremented.
        
        The method mutates the mixin's per-predator-type maps (sums and counts) based on the current per-step cached readouts.
        """
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
        Accumulate per-module learning diagnostics for the current episode.
        
        Reads optional mappings `module_credit_weights` and `module_gradient_norms` from `learn_stats` and adds their numeric values into the accumulator's running sums; missing module entries are treated as zero (i.e., absent names are created with an initial value of 0.0).
        
        Parameters:
            learn_stats (Mapping[str, object]): A mapping that may contain `module_credit_weights`
                and/or `module_gradient_norms`, each itself a mapping from module name to a numeric value.
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
        Accumulate episode metrics observed during a single environment transition.
        
        Updates reward-component totals; motor execution and slip samples and terrain counts; initial and final food and shelter distances; representation readouts attribution for a dominant predator type; per-type counts of module responses when a dominant module is present; predator mode transition and per-mode occupancy ticks; sleep-debt samples; night-related ticks (total, per-role, on-shelter, and stillness); global and per-predator-type predator-response windowing and recorded latencies; predator contacts and escapes counted per predator type; and clears per-step cached representation readouts.
        
        Parameters:
            step (int): Transition index within the episode.
            observation_meta (Dict[str, object]): Metadata for the current observation (e.g., "food_dist", "shelter_dist", "predator_visible", diagnostic fields may be present).
            next_meta (Dict[str, object]): Metadata for the next observation (e.g., "food_dist", "shelter_dist", "predator_visible", "diagnostic", "night", "shelter_role", "on_shelter").
            info (Dict[str, object]): Transition info (e.g., "reward_components", "predator_contact", "motor_slip", "motor_noise_applied", "predator_escape", "motor_execution_components").
            state (object): Agent state containing at least the attribute `sleep_debt` and optionally `last_move_dx`, `last_move_dy`.
            predator_state_before (str): Predator mode label before the transition.
            predator_state (str): Predator mode label after the transition.
        
        Raises:
            ValueError: If a predator-response window must read diagnostic distance but `next_meta["diagnostic"]` exists and is not a mapping.
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
            predator_dist = _diagnostic_predator_distance(next_meta)
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
            predator_dist = _diagnostic_predator_distance(next_meta)
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
                    preserved_escape_type = predator_type
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
