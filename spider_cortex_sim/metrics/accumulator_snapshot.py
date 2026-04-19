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


class AccumulatorSnapshotMixin:
    def snapshot(self) -> Dict[str, object]:
        """
        Compute a comprehensive snapshot of derived episode metrics and aggregated counters.
        
        The returned dictionary contains normalized distributions, per-type and per-module summaries, scalar rates, and final-summary statistics derived from accumulated counters on the accumulator. Major groups included are:
        - night summaries (ticks, per-role counts and distribution, shelter/still counts),
        - predator summaries (state ticks, occupancy distribution, mode transitions, response event counts and latencies, per-type contacts/escapes and response latencies),
        - proposer/module summaries (proposer probability means by predator type, Jensen–Shannon divergences between visual and olfactory proposer distributions for primary modules, module response distributions, action-center gate and contribution differentials, representation specialization score),
        - reflex/module usage and dominance rates, module contribution shares, and learning/gradient means,
        - terrain and motor statistics (terrain slip rates, motor slip rate, terrain difficulty, orientation alignment),
        - dominant module statistics (distribution, selected dominant module, effective module count, agreement/disagreement rates),
        - distance deltas (food and shelter), mean sleep debt, and reward component totals.
        
        Returns:
            Dict[str, object]: A mapping of derived episode statistics and aggregated scalar values keyed by descriptive names (e.g., "night_ticks", "predator_state_occupancy", "module_reflex_usage_rates", "proposer_divergence_by_module", "representation_specialization_score", "reward_component_totals", etc.). Values are numeric scalars, distributions, or nested mappings as appropriate for each metric.
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
