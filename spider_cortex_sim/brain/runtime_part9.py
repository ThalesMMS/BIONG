from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart9Mixin:
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

    def _select_b_series_semantic_action(
        self,
        *,
        observation: Dict[str, np.ndarray],
        learned_semantic_action: str,
        semantic_action: str,
        semantic_action_source: str,
        semantic_action_reason: str,
        semantic_override_count: int,
        b_temporal_threat_trace: dict[str, object],
        b_level: int,
        b_effective_level: str,
    ) -> tuple[str, str, str, int, dict[str, object], str]:
        if (
            b_level == 0
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
        elif (
            b_level == 1
            and str(getattr(self.config, "name", "")) == B1_THREAT_GUARD_POLICY_NAME
        ):
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
            ) = self._b1_threat_guard_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B1_THREAT_GUARD_EFFECTIVE_LEVEL
        elif b_level == 2 and str(getattr(self.config, "name", "")) in {
            B2_TEMPORAL_THREAT_H48_POLICY_NAME,
            B2_TEMPORAL_THREAT_H56_POLICY_NAME,
            B2_TEMPORAL_THREAT_H64_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b2_temporal_threat_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B2_TEMPORAL_THREAT_EFFECTIVE_LEVEL
        elif b_level == 3 and str(getattr(self.config, "name", "")) in {
            B3_CONTACT_MEMORY_H48_POLICY_NAME,
            B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME,
            B3_CONTACT_MEMORY_H56_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b3_contact_memory_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B3_CONTACT_MEMORY_EFFECTIVE_LEVEL
        elif b_level == 3 and str(getattr(self.config, "name", "")) == (
            B3_RECURRENT_GUARD_H48_POLICY_NAME
        ):
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b3_recurrent_guard_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B3_RECURRENT_GUARD_EFFECTIVE_LEVEL
        elif b_level == 4 and str(getattr(self.config, "name", "")) in {
            B4_RECOVERY_BALANCE_H48_POLICY_NAME,
            B4_PREDATOR_EXIT_MEMORY_H48_POLICY_NAME,
            B4_RECOVERY_BALANCE_H56_POLICY_NAME,
            B4_GENETIC_RECOVERY_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b4_recovery_balance_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B4_RECOVERY_BALANCE_EFFECTIVE_LEVEL
        elif b_level == 5 and str(getattr(self.config, "name", "")) in {
            B5_HOMEOSTATIC_ARBITER_H48_POLICY_NAME,
            B5_CIRCADIAN_RECOVERY_H48_POLICY_NAME,
            B5_HOMEOSTATIC_ARBITER_H56_POLICY_NAME,
            B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b5_homeostatic_arbiter_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B5_HOMEOSTATIC_ARBITER_EFFECTIVE_LEVEL
        elif b_level == 6 and str(getattr(self.config, "name", "")) in {
            B6_RISK_FORAGE_ARBITER_H48_POLICY_NAME,
            B6_CORRIDOR_SURVIVAL_GUARD_H48_POLICY_NAME,
            B6_THREAT_PRIORITY_MEMORY_H48_POLICY_NAME,
            B6_RISK_CORRIDOR_H56_POLICY_NAME,
            B6_GENETIC_RISK_CORRIDOR_H48_POLICY_NAME,
            B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
            B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME,
            B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME,
            B6_RECURRENT_CONTEXT_H56_POLICY_NAME,
            B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME,
            B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b6_risk_corridor_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            if semantic_action_source == B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE:
                b_effective_level = B6_FUSED_RISK_RECURRENT_EFFECTIVE_LEVEL
            elif semantic_action_source == B6_RECURRENT_MEMORY_SELECTION_SOURCE:
                b_effective_level = B6_RECURRENT_MEMORY_EFFECTIVE_LEVEL
            else:
                b_effective_level = B6_RISK_CORRIDOR_EFFECTIVE_LEVEL
        elif b_level == 7 and str(getattr(self.config, "name", "")) in {
            B7_AFFORDANCE_BUDGET_H48_POLICY_NAME,
            B7_ENERGY_BUDGET_CORRIDOR_H48_POLICY_NAME,
            B7_RECURRENT_AFFORDANCE_H48_POLICY_NAME,
            B7_AFFORDANCE_BUDGET_H56_POLICY_NAME,
            B7_GENETIC_AFFORDANCE_BUDGET_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b7_affordance_budget_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B7_AFFORDANCE_BUDGET_EFFECTIVE_LEVEL
        elif b_level == 8 and str(getattr(self.config, "name", "")) in {
            B8_SPATIAL_AFFORDANCE_MAP_H48_POLICY_NAME,
            B8_RETURN_VECTOR_H48_POLICY_NAME,
            B8_CORRIDOR_PLACE_MEMORY_H48_POLICY_NAME,
            B8_SPATIAL_AFFORDANCE_MAP_H56_POLICY_NAME,
            B8_GENETIC_SPATIAL_AFFORDANCE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b8_spatial_affordance_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B8_SPATIAL_AFFORDANCE_EFFECTIVE_LEVEL
        elif b_level == 9 and str(getattr(self.config, "name", "")) in {
            B9_WAYPOINT_PLANNER_H48_POLICY_NAME,
            B9_PATH_INTEGRATION_H48_POLICY_NAME,
            B9_ROUTE_MEMORY_H48_POLICY_NAME,
            B9_WAYPOINT_PLANNER_H56_POLICY_NAME,
            B9_GENETIC_WAYPOINT_PLANNER_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b9_waypoint_planner_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B9_WAYPOINT_PLANNER_EFFECTIVE_LEVEL
        elif b_level == 10 and str(getattr(self.config, "name", "")) in {
            B10_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
            B10_VALUE_ROUTE_EVALUATOR_H48_POLICY_NAME,
            B10_REPLAY_PLANNER_H48_POLICY_NAME,
            B10_PROSPECTIVE_REPLAY_H56_POLICY_NAME,
            B10_GENETIC_REPLAY_PLANNER_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b10_prospective_replay_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B10_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL
        elif b_level == 11 and str(getattr(self.config, "name", "")) in {
            B11_CONFIDENCE_ARBITER_H48_POLICY_NAME,
            B11_UNCERTAINTY_GATE_H48_POLICY_NAME,
            B11_NEUROMODULATED_REPLAY_H48_POLICY_NAME,
            B11_CONFIDENCE_ARBITER_H56_POLICY_NAME,
            B11_GENETIC_CONFIDENCE_GATE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b11_confidence_arbiter_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B11_CONFIDENCE_ARBITER_EFFECTIVE_LEVEL
        elif b_level == 12 and str(getattr(self.config, "name", "")) in {
            B12_PREDICTIVE_ATTENTION_H48_POLICY_NAME,
            B12_ACTIVE_INFERENCE_GATE_H48_POLICY_NAME,
            B12_AFFORDANCE_ATTENTION_H48_POLICY_NAME,
            B12_PREDICTIVE_ATTENTION_H56_POLICY_NAME,
            B12_GENETIC_ATTENTION_GATE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b12_predictive_attention_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B12_PREDICTIVE_ATTENTION_EFFECTIVE_LEVEL
        elif b_level == 13 and str(getattr(self.config, "name", "")) in {
            B13_LOCAL_AFFORDANCE_SEARCH_H48_POLICY_NAME,
            B13_COUNTERFACTUAL_ROUTE_H48_POLICY_NAME,
            B13_AFFORDANCE_SAMPLER_H48_POLICY_NAME,
            B13_LOCAL_AFFORDANCE_SEARCH_H56_POLICY_NAME,
            B13_GENETIC_LOCAL_SEARCH_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b13_local_affordance_search_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B13_LOCAL_SEARCH_EFFECTIVE_LEVEL
        elif b_level == 14 and str(getattr(self.config, "name", "")) in {
            B14_AFFORDANCE_UNCERTAINTY_H48_POLICY_NAME,
            B14_RISK_CALIBRATED_SEARCH_H48_POLICY_NAME,
            B14_CONFIDENCE_WEIGHTED_ROUTE_H48_POLICY_NAME,
            B14_AFFORDANCE_UNCERTAINTY_H56_POLICY_NAME,
            B14_GENETIC_UNCERTAINTY_SEARCH_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b14_affordance_uncertainty_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B14_AFFORDANCE_UNCERTAINTY_EFFECTIVE_LEVEL
        elif b_level == 15 and str(getattr(self.config, "name", "")) in {
            B15_OPTION_CRITIC_H48_POLICY_NAME,
            B15_PERSISTENCE_GATE_H48_POLICY_NAME,
            B15_VALUE_GATED_OPTION_H48_POLICY_NAME,
            B15_OPTION_CRITIC_H56_POLICY_NAME,
            B15_GENETIC_OPTION_CRITIC_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b15_option_critic_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B15_OPTION_CRITIC_EFFECTIVE_LEVEL
        elif b_level == 16 and str(getattr(self.config, "name", "")) in {
            B16_OPTION_ENSEMBLE_H48_POLICY_NAME,
            B16_COMPETING_OPTIONS_H48_POLICY_NAME,
            B16_ACTION_SET_VOTER_H48_POLICY_NAME,
            B16_OPTION_ENSEMBLE_H56_POLICY_NAME,
            B16_GENETIC_OPTION_ENSEMBLE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b16_option_ensemble_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B16_OPTION_ENSEMBLE_EFFECTIVE_LEVEL
        elif b_level == 17 and str(getattr(self.config, "name", "")) in {
            B17_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
            B17_AROUSAL_GATED_OPTIONS_H48_POLICY_NAME,
            B17_HOMEOSTATIC_MODULATOR_H48_POLICY_NAME,
            B17_NEUROMODULATED_ENSEMBLE_H56_POLICY_NAME,
            B17_GENETIC_NEUROMODULATED_ENSEMBLE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b17_neuromodulated_ensemble_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B17_NEUROMODULATED_ENSEMBLE_EFFECTIVE_LEVEL
        elif b_level == 18 and str(getattr(self.config, "name", "")) in {
            B18_ELIGIBILITY_TRACE_H48_POLICY_NAME,
            B18_METASTABLE_AROUSAL_H48_POLICY_NAME,
            B18_SYNAPTIC_TRACE_MODULATOR_H48_POLICY_NAME,
            B18_ELIGIBILITY_TRACE_H56_POLICY_NAME,
            B18_GENETIC_ELIGIBILITY_TRACE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b18_eligibility_trace_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B18_ELIGIBILITY_TRACE_EFFECTIVE_LEVEL
        elif b_level == 19 and str(getattr(self.config, "name", "")) in {
            B19_EPISODIC_META_MEMORY_H48_POLICY_NAME,
            B19_STABILITY_MEMORY_H48_POLICY_NAME,
            B19_SWITCH_SUPPRESSION_H48_POLICY_NAME,
            B19_EPISODIC_META_MEMORY_H56_POLICY_NAME,
            B19_GENETIC_META_MEMORY_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b19_episodic_meta_memory_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B19_EPISODIC_META_MEMORY_EFFECTIVE_LEVEL
        elif b_level == 20 and str(getattr(self.config, "name", "")) in {
            B20_WORKING_MEMORY_GATE_H48_POLICY_NAME,
            B20_CONTEXT_BINDING_H48_POLICY_NAME,
            B20_STABILITY_BUFFER_H48_POLICY_NAME,
            B20_WORKING_MEMORY_GATE_H56_POLICY_NAME,
            B20_GENETIC_WORKING_MEMORY_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b20_working_memory_gate_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B20_WORKING_MEMORY_GATE_EFFECTIVE_LEVEL
        elif b_level == 21 and str(getattr(self.config, "name", "")) in {
            B21_HIPPOCAMPAL_REPLAY_H48_POLICY_NAME,
            B21_SEQUENCE_BINDING_H48_POLICY_NAME,
            B21_ROUTE_REHEARSAL_H48_POLICY_NAME,
            B21_HIPPOCAMPAL_REPLAY_H56_POLICY_NAME,
            B21_GENETIC_REPLAY_GATE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b21_hippocampal_replay_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B21_HIPPOCAMPAL_REPLAY_EFFECTIVE_LEVEL
        elif b_level == 22 and str(getattr(self.config, "name", "")) in {
            B22_PROSPECTIVE_MAP_REPLAY_H48_POLICY_NAME,
            B22_FORWARD_MODEL_GATE_H48_POLICY_NAME,
            B22_ROUTE_VIABILITY_SIM_H48_POLICY_NAME,
            B22_PROSPECTIVE_MAP_REPLAY_H56_POLICY_NAME,
            B22_GENETIC_PROSPECTIVE_REPLAY_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b22_prospective_replay_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B22_PROSPECTIVE_REPLAY_EFFECTIVE_LEVEL
        elif b_level == 23 and str(getattr(self.config, "name", "")) in {
            B23_CONFLICT_MONITOR_H48_POLICY_NAME,
            B23_ERROR_GATED_REPLAY_H48_POLICY_NAME,
            B23_ABORT_CONFLICT_ARBITER_H48_POLICY_NAME,
            B23_CONFLICT_MONITOR_H56_POLICY_NAME,
            B23_GENETIC_CONFLICT_MONITOR_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b23_conflict_monitor_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B23_CONFLICT_MONITOR_EFFECTIVE_LEVEL
        elif b_level == 24 and str(getattr(self.config, "name", "")) in {
            B24_PRECISION_CONFLICT_H48_POLICY_NAME,
            B24_PREDICTION_PRECISION_GATE_H48_POLICY_NAME,
            B24_RELIABILITY_ABORT_H48_POLICY_NAME,
            B24_PRECISION_CONFLICT_H56_POLICY_NAME,
            B24_GENETIC_PRECISION_CONFLICT_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b24_precision_conflict_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B24_PRECISION_CONFLICT_EFFECTIVE_LEVEL
        elif b_level == 25 and str(getattr(self.config, "name", "")) in {
            B25_METACOGNITIVE_CONFIDENCE_H48_POLICY_NAME,
            B25_CONFIDENCE_CALIBRATION_H48_POLICY_NAME,
            B25_UNCERTAINTY_INTEGRATOR_H48_POLICY_NAME,
            B25_METACOGNITIVE_CONFIDENCE_H56_POLICY_NAME,
            B25_GENETIC_METACOGNITION_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b25_metacognitive_confidence_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B25_METACOGNITIVE_CONFIDENCE_EFFECTIVE_LEVEL
        elif b_level == 26 and str(getattr(self.config, "name", "")) in {
            B26_ALLOSTATIC_PREDICTION_H48_POLICY_NAME,
            B26_SETPOINT_DRIFT_H48_POLICY_NAME,
            B26_ERROR_SUPPRESSION_H48_POLICY_NAME,
            B26_ALLOSTATIC_PREDICTION_H56_POLICY_NAME,
            B26_GENETIC_ALLOSTASIS_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b26_allostatic_prediction_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B26_ALLOSTATIC_PREDICTION_EFFECTIVE_LEVEL
        elif b_level == 27 and str(getattr(self.config, "name", "")) in {
            B27_AROUSAL_GAIN_H48_POLICY_NAME,
            B27_STRESS_MODULATION_H48_POLICY_NAME,
            B27_ENERGY_AROUSAL_H48_POLICY_NAME,
            B27_AROUSAL_GAIN_H56_POLICY_NAME,
            B27_GENETIC_AROUSAL_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b27_arousal_gain_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B27_AROUSAL_GAIN_EFFECTIVE_LEVEL
        elif b_level == 28 and str(getattr(self.config, "name", "")) in {
            B28_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
            B28_THREAT_FOCUS_ATTENTION_H48_POLICY_NAME,
            B28_HOMEOSTATIC_ATTENTION_H48_POLICY_NAME,
            B28_INTEROCEPTIVE_ATTENTION_H56_POLICY_NAME,
            B28_GENETIC_ATTENTION_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b28_interoceptive_attention_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B28_INTEROCEPTIVE_ATTENTION_EFFECTIVE_LEVEL
        elif b_level == 29 and str(getattr(self.config, "name", "")) in {
            B29_SALIENCE_COMPETITION_H48_POLICY_NAME,
            B29_THREAT_SALIENCE_GATE_H48_POLICY_NAME,
            B29_HOMEOSTATIC_SALIENCE_GATE_H48_POLICY_NAME,
            B29_SALIENCE_COMPETITION_H56_POLICY_NAME,
            B29_GENETIC_SALIENCE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b29_salience_competition_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B29_SALIENCE_COMPETITION_EFFECTIVE_LEVEL
        elif b_level == 30 and str(getattr(self.config, "name", "")) in {
            B30_BASAL_GANGLIA_GATE_H48_POLICY_NAME,
            B30_GO_NOGO_BALANCE_H48_POLICY_NAME,
            B30_THREAT_INHIBITION_GATE_H48_POLICY_NAME,
            B30_BASAL_GANGLIA_GATE_H56_POLICY_NAME,
            B30_GENETIC_ACTION_GATE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b30_basal_ganglia_gate_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B30_BASAL_GANGLIA_GATE_EFFECTIVE_LEVEL
        elif b_level == 31 and str(getattr(self.config, "name", "")) in {
            B31_DOPAMINE_PREDICTION_ERROR_H48_POLICY_NAME,
            B31_TONIC_DOPAMINE_GATE_H48_POLICY_NAME,
            B31_PHASIC_DOPAMINE_GATE_H48_POLICY_NAME,
            B31_DOPAMINE_PREDICTION_ERROR_H56_POLICY_NAME,
            B31_GENETIC_DOPAMINE_GATE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b31_dopamine_prediction_error_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B31_DOPAMINE_PREDICTION_ERROR_EFFECTIVE_LEVEL
        elif b_level == 32 and str(getattr(self.config, "name", "")) in {
            B32_ACTOR_CRITIC_VALUE_H48_POLICY_NAME,
            B32_ADVANTAGE_VALUE_GATE_H48_POLICY_NAME,
            B32_CRITIC_STABILITY_H48_POLICY_NAME,
            B32_ACTOR_CRITIC_VALUE_H56_POLICY_NAME,
            B32_GENETIC_ACTOR_CRITIC_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b32_actor_critic_value_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B32_ACTOR_CRITIC_VALUE_EFFECTIVE_LEVEL
        elif b_level == 33 and str(getattr(self.config, "name", "")) in {
            B33_TD_ERROR_DECOMPOSITION_H48_POLICY_NAME,
            B33_BOOTSTRAPPED_VALUE_GATE_H48_POLICY_NAME,
            B33_REWARD_TRACE_CRITIC_H48_POLICY_NAME,
            B33_TD_ERROR_DECOMPOSITION_H56_POLICY_NAME,
            B33_GENETIC_TD_VALUE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b33_td_error_decomposition_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B33_TD_ERROR_DECOMPOSITION_EFFECTIVE_LEVEL
        elif b_level == 34 and str(getattr(self.config, "name", "")) in {
            B34_ELIGIBILITY_CREDIT_H48_POLICY_NAME,
            B34_DELAYED_CREDIT_GATE_H48_POLICY_NAME,
            B34_SYNAPTIC_TAGGING_H48_POLICY_NAME,
            B34_ELIGIBILITY_CREDIT_H56_POLICY_NAME,
            B34_GENETIC_ELIGIBILITY_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b34_eligibility_credit_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B34_ELIGIBILITY_CREDIT_EFFECTIVE_LEVEL
        elif b_level == 35 and str(getattr(self.config, "name", "")) in {
            B35_FORWARD_MODEL_VALUE_H48_POLICY_NAME,
            B35_TRANSITION_ERROR_GATE_H48_POLICY_NAME,
            B35_MODEL_CONFIDENCE_H48_POLICY_NAME,
            B35_FORWARD_MODEL_VALUE_H56_POLICY_NAME,
            B35_GENETIC_FORWARD_MODEL_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b35_forward_model_value_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B35_FORWARD_MODEL_VALUE_EFFECTIVE_LEVEL
        elif b_level == 36 and str(getattr(self.config, "name", "")) in {
            B36_LATENT_BELIEF_STATE_H48_POLICY_NAME,
            B36_BELIEF_ERROR_GATE_H48_POLICY_NAME,
            B36_CONTEXT_INFERENCE_H48_POLICY_NAME,
            B36_LATENT_BELIEF_STATE_H56_POLICY_NAME,
            B36_GENETIC_BELIEF_STATE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b36_latent_belief_state_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B36_LATENT_BELIEF_STATE_EFFECTIVE_LEVEL
        elif b_level == 37 and str(getattr(self.config, "name", "")) in {
            B37_STATE_FACTOR_GATE_H48_POLICY_NAME,
            B37_INTERO_EXTERO_FACTOR_H48_POLICY_NAME,
            B37_FACTOR_CONFIDENCE_H48_POLICY_NAME,
            B37_STATE_FACTOR_GATE_H56_POLICY_NAME,
            B37_GENETIC_STATE_FACTOR_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b37_state_factor_gate_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B37_STATE_FACTOR_GATE_EFFECTIVE_LEVEL
        elif b_level == 38 and str(getattr(self.config, "name", "")) in {
            B38_FACTOR_ATTENTION_H48_POLICY_NAME,
            B38_INTEROCEPTIVE_ATTENTION_H48_POLICY_NAME,
            B38_CONFIDENCE_ATTENTION_H48_POLICY_NAME,
            B38_FACTOR_ATTENTION_H56_POLICY_NAME,
            B38_GENETIC_FACTOR_ATTENTION_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b38_factor_attention_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B38_FACTOR_ATTENTION_EFFECTIVE_LEVEL
        elif b_level == 39 and str(getattr(self.config, "name", "")) in {
            B39_ATTENTION_BINDING_H48_POLICY_NAME,
            B39_CROSS_FACTOR_BINDING_H48_POLICY_NAME,
            B39_CONTEXT_BINDING_ATTENTION_H48_POLICY_NAME,
            B39_ATTENTION_BINDING_H56_POLICY_NAME,
            B39_GENETIC_ATTENTION_BINDING_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b39_attention_binding_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B39_ATTENTION_BINDING_EFFECTIVE_LEVEL
        elif b_level == 40 and str(getattr(self.config, "name", "")) in {
            B40_GLOBAL_WORKSPACE_H48_POLICY_NAME,
            B40_SENSORY_WORKSPACE_H48_POLICY_NAME,
            B40_CONTEXT_WORKSPACE_H48_POLICY_NAME,
            B40_GLOBAL_WORKSPACE_H56_POLICY_NAME,
            B40_GENETIC_GLOBAL_WORKSPACE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b40_global_workspace_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B40_GLOBAL_WORKSPACE_EFFECTIVE_LEVEL
        elif b_level == 41 and str(getattr(self.config, "name", "")) in {
            B41_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
            B41_INHIBITORY_CONTROL_H48_POLICY_NAME,
            B41_GOAL_CONTEXT_SELECTOR_H48_POLICY_NAME,
            B41_EXECUTIVE_WORKSPACE_H56_POLICY_NAME,
            B41_GENETIC_EXECUTIVE_WORKSPACE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b41_executive_workspace_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B41_EXECUTIVE_WORKSPACE_EFFECTIVE_LEVEL
        elif b_level == 42 and str(getattr(self.config, "name", "")) in {
            B42_ERROR_MONITOR_H48_POLICY_NAME,
            B42_CONFLICT_MONITOR_H48_POLICY_NAME,
            B42_PERFORMANCE_MONITOR_H48_POLICY_NAME,
            B42_ERROR_MONITOR_H56_POLICY_NAME,
            B42_GENETIC_ERROR_MONITOR_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b42_error_monitor_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B42_ERROR_MONITOR_EFFECTIVE_LEVEL
        elif b_level == 43 and str(getattr(self.config, "name", "")) in {
            B43_ADAPTIVE_PRECISION_H48_POLICY_NAME,
            B43_AROUSAL_PRECISION_H48_POLICY_NAME,
            B43_THRESHOLD_ADAPTATION_H48_POLICY_NAME,
            B43_ADAPTIVE_PRECISION_H56_POLICY_NAME,
            B43_GENETIC_ADAPTIVE_PRECISION_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b43_adaptive_precision_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B43_ADAPTIVE_PRECISION_EFFECTIVE_LEVEL
        elif b_level == 44 and str(getattr(self.config, "name", "")) in {
            B44_THALAMIC_RELAY_H48_POLICY_NAME,
            B44_SENSORY_RELAY_H48_POLICY_NAME,
            B44_CONTEXT_RELAY_H48_POLICY_NAME,
            B44_THALAMIC_RELAY_H56_POLICY_NAME,
            B44_GENETIC_THALAMIC_RELAY_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b44_thalamic_relay_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B44_THALAMIC_RELAY_EFFECTIVE_LEVEL
        elif b_level == 45 and str(getattr(self.config, "name", "")) in {
            B45_RETICULAR_INHIBITION_H48_POLICY_NAME,
            B45_SENSORY_INHIBITION_H48_POLICY_NAME,
            B45_CONTEXT_INHIBITION_H48_POLICY_NAME,
            B45_RETICULAR_INHIBITION_H56_POLICY_NAME,
            B45_GENETIC_RETICULAR_INHIBITION_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b45_reticular_inhibition_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B45_RETICULAR_INHIBITION_EFFECTIVE_LEVEL
        elif b_level == 46 and str(getattr(self.config, "name", "")) in {
            B46_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
            B46_FEEDBACK_GAIN_H48_POLICY_NAME,
            B46_CONTEXT_FEEDBACK_H48_POLICY_NAME,
            B46_CORTICOTHALAMIC_FEEDBACK_H56_POLICY_NAME,
            B46_GENETIC_CORTICOTHALAMIC_FEEDBACK_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b46_corticothalamic_feedback_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B46_CORTICOTHALAMIC_FEEDBACK_EFFECTIVE_LEVEL
        elif b_level == 47 and str(getattr(self.config, "name", "")) in {
            B47_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
            B47_PHASE_LOCKING_H48_POLICY_NAME,
            B47_COHERENCE_GATE_H48_POLICY_NAME,
            B47_OSCILLATORY_SYNCHRONY_H56_POLICY_NAME,
            B47_GENETIC_OSCILLATORY_SYNCHRONY_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b47_oscillatory_synchrony_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B47_OSCILLATORY_SYNCHRONY_EFFECTIVE_LEVEL
        elif b_level == 48 and str(getattr(self.config, "name", "")) in {
            B48_CEREBELLAR_TIMING_H48_POLICY_NAME,
            B48_TIMING_ERROR_CORRECTION_H48_POLICY_NAME,
            B48_PREDICTIVE_TIMING_H48_POLICY_NAME,
            B48_CEREBELLAR_TIMING_H56_POLICY_NAME,
            B48_GENETIC_CEREBELLAR_TIMING_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b48_cerebellar_timing_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B48_CEREBELLAR_TIMING_EFFECTIVE_LEVEL
        elif b_level == 49 and str(getattr(self.config, "name", "")) in {
            B49_STRIATAL_ACTION_GATE_H48_POLICY_NAME,
            B49_DIRECT_PATH_FACILITATION_H48_POLICY_NAME,
            B49_INDIRECT_PATH_SUPPRESSION_H48_POLICY_NAME,
            B49_STRIATAL_ACTION_GATE_H56_POLICY_NAME,
            B49_GENETIC_STRIATAL_GATE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b49_striatal_action_gate_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B49_STRIATAL_ACTION_GATE_EFFECTIVE_LEVEL
        elif b_level == 50 and str(getattr(self.config, "name", "")) in {
            B50_HABIT_CHUNKING_H48_POLICY_NAME,
            B50_ACTION_CHUNK_VALUE_H48_POLICY_NAME,
            B50_HABIT_STABILITY_H48_POLICY_NAME,
            B50_HABIT_CHUNKING_H56_POLICY_NAME,
            B50_GENETIC_HABIT_CHUNKING_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b50_habit_chunking_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B50_HABIT_CHUNKING_EFFECTIVE_LEVEL
        elif b_level == 51 and str(getattr(self.config, "name", "")) in {
            B51_DOPAMINERGIC_HABIT_MODULATION_H48_POLICY_NAME,
            B51_REWARD_PREDICTION_GAIN_H48_POLICY_NAME,
            B51_NOVELTY_MODULATED_HABIT_H48_POLICY_NAME,
            B51_DOPAMINERGIC_HABIT_MODULATION_H56_POLICY_NAME,
            B51_GENETIC_DOPAMINE_HABIT_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b51_dopaminergic_habit_modulation_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B51_DOPAMINERGIC_HABIT_MODULATION_EFFECTIVE_LEVEL
        elif b_level == 52 and str(getattr(self.config, "name", "")) in {
            B52_CHOLINERGIC_PRECISION_GATE_H48_POLICY_NAME,
            B52_ATTENTION_GAIN_H48_POLICY_NAME,
            B52_UNCERTAINTY_RELEASE_H48_POLICY_NAME,
            B52_CHOLINERGIC_PRECISION_GATE_H56_POLICY_NAME,
            B52_GENETIC_CHOLINERGIC_PRECISION_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b52_cholinergic_precision_gate_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B52_CHOLINERGIC_PRECISION_GATE_EFFECTIVE_LEVEL
        elif b_level == 53 and str(getattr(self.config, "name", "")) in {
            B53_NORADRENERGIC_AROUSAL_GAIN_H48_POLICY_NAME,
            B53_SURPRISE_GAIN_H48_POLICY_NAME,
            B53_STRESS_PRECISION_H48_POLICY_NAME,
            B53_NORADRENERGIC_AROUSAL_GAIN_H56_POLICY_NAME,
            B53_GENETIC_AROUSAL_PRECISION_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b53_noradrenergic_arousal_gain_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B53_NORADRENERGIC_AROUSAL_GAIN_EFFECTIVE_LEVEL
        elif b_level == 54 and str(getattr(self.config, "name", "")) in {
            B54_SEROTONERGIC_PATIENCE_GATE_H48_POLICY_NAME,
            B54_IMPULSE_SUPPRESSION_H48_POLICY_NAME,
            B54_PATIENCE_BALANCE_H48_POLICY_NAME,
            B54_SEROTONERGIC_PATIENCE_GATE_H56_POLICY_NAME,
            B54_GENETIC_SEROTONIN_PATIENCE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b54_serotonergic_patience_gate_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B54_SEROTONERGIC_PATIENCE_GATE_EFFECTIVE_LEVEL
        elif b_level == 55 and str(getattr(self.config, "name", "")) in {
            B55_HYPOTHALAMIC_DRIVE_COUPLING_H48_POLICY_NAME,
            B55_SATIETY_RECOVERY_BALANCE_H48_POLICY_NAME,
            B55_SLEEP_HUNGER_ARBITER_H48_POLICY_NAME,
            B55_HYPOTHALAMIC_DRIVE_COUPLING_H56_POLICY_NAME,
            B55_GENETIC_HYPOTHALAMIC_DRIVE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b55_hypothalamic_drive_coupling_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B55_HYPOTHALAMIC_DRIVE_COUPLING_EFFECTIVE_LEVEL
        elif b_level == 56 and str(getattr(self.config, "name", "")) in {
            B56_HPA_STRESS_AXIS_H48_POLICY_NAME,
            B56_CORTISOL_RECOVERY_BALANCE_H48_POLICY_NAME,
            B56_STRESS_LOAD_GATE_H48_POLICY_NAME,
            B56_HPA_STRESS_AXIS_H56_POLICY_NAME,
            B56_GENETIC_HPA_STRESS_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b56_hpa_stress_axis_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B56_HPA_STRESS_AXIS_EFFECTIVE_LEVEL
        elif b_level == 57 and str(getattr(self.config, "name", "")) in {
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
            B57_VISCERAL_SALIENCE_GATE_H48_POLICY_NAME,
            B57_STRESS_DRIVE_AWARENESS_H48_POLICY_NAME,
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_H56_POLICY_NAME,
            B57_GENETIC_INTEROCEPTIVE_AWARENESS_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b57_insular_interoceptive_awareness_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B57_INSULAR_INTEROCEPTIVE_AWARENESS_EFFECTIVE_LEVEL
        elif b_level == 58 and str(getattr(self.config, "name", "")) in {
            B58_ACC_CONFLICT_MONITOR_H48_POLICY_NAME,
            B58_ERROR_SALIENCE_GATE_H48_POLICY_NAME,
            B58_CONFLICT_RESOLUTION_BALANCE_H48_POLICY_NAME,
            B58_ACC_CONFLICT_MONITOR_H56_POLICY_NAME,
            B58_GENETIC_ACC_CONFLICT_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b58_acc_conflict_monitor_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B58_ACC_CONFLICT_MONITOR_EFFECTIVE_LEVEL
        elif b_level == 59 and str(getattr(self.config, "name", "")) in {
            B59_PREFRONTAL_GOAL_CONTEXT_H48_POLICY_NAME,
            B59_WORKING_SET_STABILITY_H48_POLICY_NAME,
            B59_EXECUTIVE_TASK_SET_H48_POLICY_NAME,
            B59_PREFRONTAL_GOAL_CONTEXT_H56_POLICY_NAME,
            B59_GENETIC_PREFRONTAL_CONTROL_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b59_prefrontal_goal_context_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B59_PREFRONTAL_GOAL_CONTEXT_EFFECTIVE_LEVEL
        elif b_level == 60 and str(getattr(self.config, "name", "")) in {
            B60_ORBITOFRONTAL_OUTCOME_VALUE_H48_POLICY_NAME,
            B60_REVERSAL_VALUE_GATE_H48_POLICY_NAME,
            B60_GOAL_OUTCOME_PREDICTION_H48_POLICY_NAME,
            B60_ORBITOFRONTAL_OUTCOME_VALUE_H56_POLICY_NAME,
            B60_GENETIC_ORBITOFRONTAL_VALUE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b60_orbitofrontal_outcome_value_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B60_ORBITOFRONTAL_OUTCOME_VALUE_EFFECTIVE_LEVEL
        elif b_level == 61 and str(getattr(self.config, "name", "")) in {
            B61_AMYGDALA_SAFETY_VALUE_H48_POLICY_NAME,
            B61_THREAT_VALUE_TAG_H48_POLICY_NAME,
            B61_SAFETY_PREDICTION_GATE_H48_POLICY_NAME,
            B61_AMYGDALA_SAFETY_VALUE_H56_POLICY_NAME,
            B61_GENETIC_AMYGDALA_SAFETY_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b61_amygdala_safety_value_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B61_AMYGDALA_SAFETY_VALUE_EFFECTIVE_LEVEL
        elif b_level == 62 and str(getattr(self.config, "name", "")) in {
            B62_DEFENSIVE_MODE_SELECTOR_H48_POLICY_NAME,
            B62_FREEZE_FLEE_BALANCE_H48_POLICY_NAME,
            B62_SHELTER_DEFENSE_GATE_H48_POLICY_NAME,
            B62_DEFENSIVE_MODE_SELECTOR_H56_POLICY_NAME,
            B62_GENETIC_DEFENSIVE_MODE_H48_POLICY_NAME,
        }:
            (
                semantic_action,
                semantic_action_source,
                semantic_action_reason,
                semantic_override_count,
                b_temporal_threat_trace,
            ) = self._b62_defensive_mode_selector_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            b_effective_level = B62_DEFENSIVE_MODE_SELECTOR_EFFECTIVE_LEVEL
        return (
            semantic_action,
            semantic_action_source,
            semantic_action_reason,
            semantic_override_count,
            b_temporal_threat_trace,
            b_effective_level,
        )
