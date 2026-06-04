from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart3Mixin:
    def _b14_affordance_uncertainty_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b13_local_affordance_search_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b14_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "affordance_uncertainty"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b14_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b13_decision = str(trace_payload.get("b13_decision", "preserve_b13"))
        route_score = float(trace_payload.get("b13_local_route_score", 0.0) or 0.0)
        affordance_samples = float(trace_payload.get("b13_affordance_samples", 0.0) or 0.0)
        dead_end_score = float(trace_payload.get("b13_dead_end_score", 0.0) or 0.0)
        attention_gain = float(trace_payload.get("b12_attention_gain", 0.0) or 0.0)
        uncertainty_signal = float(
            np.clip(
                abs(route_score - affordance_samples)
                + dead_end_score * float(params["b14_risk_gain"])
                + max(0.0, 0.25 - attention_gain),
                0.0,
                1.0,
            )
        )
        uncertainty = max(
            float(getattr(self, "_b14_uncertainty_memory", 0.0))
            * float(params["b14_uncertainty_decay"]),
            uncertainty_signal,
        )
        confidence = float(np.clip(route_score + affordance_samples * 0.35 - uncertainty * 0.40, 0.0, 1.0))
        risk_adjusted_score = float(np.clip(confidence - dead_end_score * float(params["b14_risk_gain"]), 0.0, 1.0))
        commitment_lock = int(getattr(self, "_b14_commitment_lock", 0))
        uncertainty_state = "non_corridor"
        decision_label = "preserve_b13"

        if corridor_map:
            if (
                b13_decision in {"commit_local_affordance_search", "continue_local_search_lock"}
                and confidence >= float(params["b14_confidence_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                commitment_lock = max(commitment_lock, int(params["b14_commit_ticks"]))
                uncertainty_state = "confidence_calibrated_route"
                decision_label = "commit_confident_affordance"
                reason = "b14_commit_confident_affordance"
            elif (
                uncertainty >= float(params["b14_uncertainty_threshold"])
                and dead_end_score >= 0.35
            ):
                semantic_action = "MOVE_TO_SHELTER"
                uncertainty_state = "uncertain_dead_end_return"
                decision_label = "return_on_affordance_uncertainty"
                reason = "b14_return_on_affordance_uncertainty"
            elif commitment_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                uncertainty_state = "confidence_lock_continues"
                decision_label = "continue_confidence_lock"
                reason = "b14_continue_confidence_lock"

        trace_payload.update(
            {
                "b14_controller_profile": profile,
                "b14_uncertainty_state": uncertainty_state,
                "b14_affordance_confidence": round(float(confidence), 6),
                "b14_uncertainty": round(float(uncertainty), 6),
                "b14_risk_adjusted_score": round(float(risk_adjusted_score), 6),
                "b14_commitment_lock": int(commitment_lock),
                "b14_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b14_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b14_genetic_candidate"] = int(params["ga_candidate"])

        self._b14_uncertainty_memory = float(uncertainty)
        self._b14_commitment_lock = max(0, int(commitment_lock) - 1)
        self._b14_last_tick = int(tick)
        return (
            semantic_action,
            B14_AFFORDANCE_UNCERTAINTY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b15_controller_params(self) -> dict[str, float]:
        params = self._b14_controller_params()
        defaults = {
            "b15_option_memory_decay": 0.84,
            "b15_option_value_threshold": 0.28,
            "b15_termination_threshold": 0.62,
            "b15_persistence_gain": 0.45,
            "b15_option_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "option_critic"
        )
        if profile == "persistence_gate":
            defaults.update({"b15_persistence_gain": 0.62, "b15_option_commit_ticks": 7.0})
        elif profile == "value_gated_option":
            defaults.update({"b15_option_value_threshold": 0.22, "b15_termination_threshold": 0.68})
        elif profile == "option_critic_h56":
            defaults.update({"b15_option_memory_decay": 0.88, "b15_option_commit_ticks": 6.0})
        elif profile == "genetic_option_critic":
            defaults.update({"b15_option_value_threshold": 0.24, "b15_persistence_gain": 0.56})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b15_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b15_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b15_option_memory = 0.0
        self._b15_option_lock = 0
        self._b15_last_tick = int(tick)

    def _b15_option_critic_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b14_affordance_uncertainty_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b15_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "option_critic"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b15_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b14_decision = str(trace_payload.get("b14_decision", "preserve_b14"))
        confidence = float(trace_payload.get("b14_affordance_confidence", 0.0) or 0.0)
        uncertainty = float(trace_payload.get("b14_uncertainty", 0.0) or 0.0)
        risk_adjusted_score = float(trace_payload.get("b14_risk_adjusted_score", 0.0) or 0.0)
        route_confidence = float(trace_payload.get("b9_route_confidence", 0.0) or 0.0)
        prospective_value = float(trace_payload.get("b10_prospective_value", 0.0) or 0.0)
        abort_signal = float(trace_payload.get("b10_abort_signal", 0.0) or 0.0)
        dead_end_score = float(trace_payload.get("b13_dead_end_score", 0.0) or 0.0)
        option_value_raw = float(
            np.clip(
                risk_adjusted_score * 0.50
                + confidence * 0.25
                + max(0.0, route_confidence) * 0.15
                + max(0.0, prospective_value) * 0.10,
                0.0,
                1.0,
            )
        )
        option_memory = max(
            float(getattr(self, "_b15_option_memory", 0.0))
            * float(params["b15_option_memory_decay"]),
            option_value_raw,
        )
        termination_pressure = float(
            np.clip(
                uncertainty * 0.45
                + abort_signal * 0.30
                + dead_end_score * 0.25,
                0.0,
                1.0,
            )
        )
        persistence_score = float(
            np.clip(
                option_memory
                + confidence * float(params["b15_persistence_gain"])
                - termination_pressure * 0.35,
                0.0,
                1.0,
            )
        )
        option_lock = int(getattr(self, "_b15_option_lock", 0))
        option_state = "non_corridor"
        decision_label = "preserve_b14"

        if corridor_map:
            if (
                b14_decision in {"commit_confident_affordance", "continue_confidence_lock"}
                and persistence_score >= float(params["b15_option_value_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                option_lock = max(option_lock, int(params["b15_option_commit_ticks"]))
                option_state = "option_persist_food_route"
                decision_label = "persist_food_option"
                reason = "b15_persist_food_option"
            elif termination_pressure >= float(params["b15_termination_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                option_state = "option_terminated_return"
                decision_label = "terminate_uncertain_option"
                reason = "b15_terminate_uncertain_option"
            elif option_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                option_state = "option_lock_continues"
                decision_label = "continue_option_lock"
                reason = "b15_continue_option_lock"

        trace_payload.update(
            {
                "b15_controller_profile": profile,
                "b15_option_state": option_state,
                "b15_option_value": round(float(option_memory), 6),
                "b15_termination_pressure": round(float(termination_pressure), 6),
                "b15_persistence_score": round(float(persistence_score), 6),
                "b15_option_lock": int(option_lock),
                "b15_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b15_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b15_genetic_candidate"] = int(params["ga_candidate"])

        self._b15_option_memory = float(option_memory)
        self._b15_option_lock = max(0, int(option_lock) - 1)
        self._b15_last_tick = int(tick)
        return (
            semantic_action,
            B15_OPTION_CRITIC_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b16_controller_params(self) -> dict[str, float]:
        params = self._b15_controller_params()
        defaults = {
            "b16_ensemble_decay": 0.82,
            "b16_consensus_threshold": 0.30,
            "b16_conflict_threshold": 0.56,
            "b16_vote_gain": 0.50,
            "b16_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "option_ensemble"
        )
        if profile == "competing_options":
            defaults.update({"b16_conflict_threshold": 0.48, "b16_vote_gain": 0.62})
        elif profile == "action_set_voter":
            defaults.update({"b16_consensus_threshold": 0.24, "b16_commit_ticks": 6.0})
        elif profile == "option_ensemble_h56":
            defaults.update({"b16_ensemble_decay": 0.86, "b16_commit_ticks": 6.0})
        elif profile == "genetic_option_ensemble":
            defaults.update({"b16_consensus_threshold": 0.26, "b16_vote_gain": 0.58})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b16_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b16_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b16_ensemble_memory = 0.0
        self._b16_ensemble_lock = 0
        self._b16_last_tick = int(tick)

    def _b16_option_ensemble_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b15_option_critic_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b16_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "option_ensemble"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b16_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b15_decision = str(trace_payload.get("b15_decision", "preserve_b15"))
        option_value = float(trace_payload.get("b15_option_value", 0.0) or 0.0)
        termination_pressure = float(
            trace_payload.get("b15_termination_pressure", 0.0) or 0.0
        )
        persistence_score = float(
            trace_payload.get("b15_persistence_score", 0.0) or 0.0
        )
        confidence = float(trace_payload.get("b14_affordance_confidence", 0.0) or 0.0)
        uncertainty = float(trace_payload.get("b14_uncertainty", 0.0) or 0.0)
        dead_end_score = float(trace_payload.get("b13_dead_end_score", 0.0) or 0.0)
        vote_gain = float(params["b16_vote_gain"])
        continue_vote = float(
            np.clip(
                persistence_score * 0.45
                + option_value * 0.35
                + confidence * 0.20
                + vote_gain * 0.10,
                0.0,
                1.0,
            )
        )
        return_vote = float(
            np.clip(
                termination_pressure * 0.50
                + uncertainty * 0.30
                + dead_end_score * 0.20,
                0.0,
                1.0,
            )
        )
        consensus_raw = float(
            np.clip(max(continue_vote, return_vote) - min(continue_vote, return_vote) * 0.25, 0.0, 1.0)
        )
        ensemble_memory = max(
            float(getattr(self, "_b16_ensemble_memory", 0.0))
            * float(params["b16_ensemble_decay"]),
            consensus_raw,
        )
        conflict_score = float(np.clip(abs(continue_vote - return_vote), 0.0, 1.0))
        ensemble_lock = int(getattr(self, "_b16_ensemble_lock", 0))
        ensemble_state = "non_corridor"
        decision_label = "preserve_b15"

        if corridor_map:
            if (
                b15_decision in {"persist_food_option", "continue_option_lock"}
                and continue_vote >= float(params["b16_consensus_threshold"])
                and continue_vote >= return_vote
            ):
                semantic_action = "MOVE_TO_FOOD"
                ensemble_lock = max(ensemble_lock, int(params["b16_commit_ticks"]))
                ensemble_state = "ensemble_continue_consensus"
                decision_label = "ensemble_continue_option"
                reason = "b16_ensemble_continue_option"
            elif (
                return_vote >= float(params["b16_conflict_threshold"])
                and return_vote > continue_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                ensemble_state = "ensemble_return_consensus"
                decision_label = "ensemble_return_option"
                reason = "b16_ensemble_return_option"
            elif ensemble_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                ensemble_state = "ensemble_lock_continues"
                decision_label = "continue_ensemble_lock"
                reason = "b16_continue_ensemble_lock"

        trace_payload.update(
            {
                "b16_controller_profile": profile,
                "b16_ensemble_state": ensemble_state,
                "b16_continue_vote": round(float(continue_vote), 6),
                "b16_return_vote": round(float(return_vote), 6),
                "b16_option_votes": round(float(continue_vote - return_vote), 6),
                "b16_consensus_score": round(float(ensemble_memory), 6),
                "b16_conflict_score": round(float(conflict_score), 6),
                "b16_ensemble_lock": int(ensemble_lock),
                "b16_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b16_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b16_genetic_candidate"] = int(params["ga_candidate"])

        self._b16_ensemble_memory = float(ensemble_memory)
        self._b16_ensemble_lock = max(0, int(ensemble_lock) - 1)
        self._b16_last_tick = int(tick)
        return (
            semantic_action,
            B16_OPTION_ENSEMBLE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b17_controller_params(self) -> dict[str, float]:
        params = self._b16_controller_params()
        defaults = {
            "b17_arousal_decay": 0.84,
            "b17_gain_threshold": 0.34,
            "b17_conflict_release": 0.52,
            "b17_homeostasis_gain": 0.44,
            "b17_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "neuromodulated_ensemble"
        )
        if profile == "arousal_gated_options":
            defaults.update({"b17_gain_threshold": 0.28, "b17_commit_ticks": 6.0})
        elif profile == "homeostatic_modulator":
            defaults.update({"b17_homeostasis_gain": 0.58, "b17_gain_threshold": 0.30})
        elif profile == "neuromodulated_ensemble_h56":
            defaults.update({"b17_arousal_decay": 0.88, "b17_commit_ticks": 6.0})
        elif profile == "genetic_neuromodulated_ensemble":
            defaults.update({"b17_gain_threshold": 0.30, "b17_homeostasis_gain": 0.52})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b17_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b17_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b17_arousal_memory = 0.0
        self._b17_modulation_lock = 0
        self._b17_last_tick = int(tick)

    def _b17_neuromodulated_ensemble_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b16_option_ensemble_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b17_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "neuromodulated_ensemble"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b17_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b16_decision = str(trace_payload.get("b16_decision", "preserve_b16"))
        continue_vote = float(trace_payload.get("b16_continue_vote", 0.0) or 0.0)
        return_vote = float(trace_payload.get("b16_return_vote", 0.0) or 0.0)
        consensus = float(trace_payload.get("b16_consensus_score", 0.0) or 0.0)
        conflict = float(trace_payload.get("b16_conflict_score", 0.0) or 0.0)
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        hunger = self._b_series_float(hunger_obs, "hunger")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        health = self._b_series_float(sleep_obs, "health")
        threat = max(
            float(trace_payload.get("b_current_threat_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_temporal_threat_pressure", 0.0) or 0.0),
            float(trace_payload.get("b6_risk_pressure", 0.0) or 0.0),
        )
        homeostatic_gain = float(
            np.clip(
                hunger * 0.40
                + sleep_debt * 0.25
                + max(0.0, 1.0 - health) * 0.35,
                0.0,
                1.0,
            )
        )
        arousal_raw = float(
            np.clip(
                threat * 0.35
                + conflict * 0.25
                + consensus * 0.20
                + homeostatic_gain * float(params["b17_homeostasis_gain"]),
                0.0,
                1.0,
            )
        )
        arousal_signal = max(
            float(getattr(self, "_b17_arousal_memory", 0.0))
            * float(params["b17_arousal_decay"]),
            arousal_raw,
        )
        option_gain = float(
            np.clip(continue_vote + arousal_signal * 0.25 - return_vote * 0.20, 0.0, 1.0)
        )
        conflict_release = float(
            np.clip(return_vote + conflict * 0.35 + threat * 0.20, 0.0, 1.0)
        )
        modulation_lock = int(getattr(self, "_b17_modulation_lock", 0))
        modulator_state = "non_corridor"
        decision_label = "preserve_b16"

        if corridor_map:
            if (
                b16_decision in {"ensemble_continue_option", "continue_ensemble_lock"}
                and option_gain >= float(params["b17_gain_threshold"])
                and option_gain >= conflict_release
            ):
                semantic_action = "MOVE_TO_FOOD"
                modulation_lock = max(modulation_lock, int(params["b17_commit_ticks"]))
                modulator_state = "modulated_continue"
                decision_label = "neuromodulated_continue"
                reason = "b17_neuromodulated_continue"
            elif (
                conflict_release >= float(params["b17_conflict_release"])
                and conflict_release > option_gain
            ):
                semantic_action = "MOVE_TO_SHELTER"
                modulator_state = "modulated_return"
                decision_label = "neuromodulated_return"
                reason = "b17_neuromodulated_return"
            elif modulation_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                modulator_state = "modulation_lock_continues"
                decision_label = "continue_modulation_lock"
                reason = "b17_continue_modulation_lock"

        trace_payload.update(
            {
                "b17_controller_profile": profile,
                "b17_modulator_state": modulator_state,
                "b17_arousal_signal": round(float(arousal_signal), 6),
                "b17_homeostatic_gain": round(float(homeostatic_gain), 6),
                "b17_option_gain": round(float(option_gain), 6),
                "b17_conflict_release": round(float(conflict_release), 6),
                "b17_modulation_lock": int(modulation_lock),
                "b17_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b17_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b17_genetic_candidate"] = int(params["ga_candidate"])

        self._b17_arousal_memory = float(arousal_signal)
        self._b17_modulation_lock = max(0, int(modulation_lock) - 1)
        self._b17_last_tick = int(tick)
        return (
            semantic_action,
            B17_NEUROMODULATED_ENSEMBLE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b18_controller_params(self) -> dict[str, float]:
        params = self._b17_controller_params()
        defaults = {
            "b18_trace_decay": 0.86,
            "b18_stability_threshold": 0.30,
            "b18_switch_threshold": 0.58,
            "b18_prediction_gain": 0.42,
            "b18_trace_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "eligibility_trace"
        )
        if profile == "metastable_arousal":
            defaults.update({"b18_stability_threshold": 0.25, "b18_trace_commit_ticks": 6.0})
        elif profile == "synaptic_trace_modulator":
            defaults.update({"b18_prediction_gain": 0.56, "b18_switch_threshold": 0.62})
        elif profile == "eligibility_trace_h56":
            defaults.update({"b18_trace_decay": 0.88, "b18_trace_commit_ticks": 6.0})
        elif profile == "genetic_eligibility_trace":
            defaults.update({"b18_stability_threshold": 0.27, "b18_prediction_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b18_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b18_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b18_eligibility_trace = 0.0
        self._b18_trace_lock = 0
        self._b18_last_tick = int(tick)

    def _b18_eligibility_trace_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b17_neuromodulated_ensemble_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b18_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "eligibility_trace"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b18_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b17_decision = str(trace_payload.get("b17_decision", "preserve_b17"))
        arousal_signal = float(trace_payload.get("b17_arousal_signal", 0.0) or 0.0)
        homeostatic_gain = float(trace_payload.get("b17_homeostatic_gain", 0.0) or 0.0)
        option_gain = float(trace_payload.get("b17_option_gain", 0.0) or 0.0)
        conflict_release = float(trace_payload.get("b17_conflict_release", 0.0) or 0.0)
        consensus = float(trace_payload.get("b16_consensus_score", 0.0) or 0.0)
        option_value = float(trace_payload.get("b15_option_value", 0.0) or 0.0)
        prediction_proxy = float(
            np.clip(
                option_gain * 0.35
                + consensus * 0.25
                + option_value * 0.20
                + arousal_signal * 0.20,
                0.0,
                1.0,
            )
        )
        eligibility_raw = float(
            np.clip(
                prediction_proxy * float(params["b18_prediction_gain"])
                + max(0.0, option_gain - conflict_release) * 0.35
                + homeostatic_gain * 0.20,
                0.0,
                1.0,
            )
        )
        eligibility_trace = max(
            float(getattr(self, "_b18_eligibility_trace", 0.0))
            * float(params["b18_trace_decay"]),
            eligibility_raw,
        )
        stability_bias = float(
            np.clip(eligibility_trace + option_gain * 0.30 - conflict_release * 0.20, 0.0, 1.0)
        )
        switch_pressure = float(
            np.clip(conflict_release + max(0.0, conflict_release - option_gain) * 0.35, 0.0, 1.0)
        )
        trace_lock = int(getattr(self, "_b18_trace_lock", 0))
        trace_state = "non_corridor"
        decision_label = "preserve_b17"

        if corridor_map:
            if (
                b17_decision in {"neuromodulated_continue", "continue_modulation_lock"}
                and stability_bias >= float(params["b18_stability_threshold"])
                and stability_bias >= switch_pressure
            ):
                semantic_action = "MOVE_TO_FOOD"
                trace_lock = max(trace_lock, int(params["b18_trace_commit_ticks"]))
                trace_state = "trace_stabilizes_option"
                decision_label = "eligibility_stabilize_option"
                reason = "b18_eligibility_stabilize_option"
            elif (
                switch_pressure >= float(params["b18_switch_threshold"])
                and switch_pressure > stability_bias
            ):
                semantic_action = "MOVE_TO_SHELTER"
                trace_state = "trace_releases_option"
                decision_label = "eligibility_release_option"
                reason = "b18_eligibility_release_option"
            elif trace_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                trace_state = "trace_lock_continues"
                decision_label = "continue_trace_lock"
                reason = "b18_continue_trace_lock"

        trace_payload.update(
            {
                "b18_controller_profile": profile,
                "b18_trace_state": trace_state,
                "b18_eligibility_trace": round(float(eligibility_trace), 6),
                "b18_reward_prediction_proxy": round(float(prediction_proxy), 6),
                "b18_stability_bias": round(float(stability_bias), 6),
                "b18_switch_pressure": round(float(switch_pressure), 6),
                "b18_trace_lock": int(trace_lock),
                "b18_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b18_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b18_genetic_candidate"] = int(params["ga_candidate"])

        self._b18_eligibility_trace = float(eligibility_trace)
        self._b18_trace_lock = max(0, int(trace_lock) - 1)
        self._b18_last_tick = int(tick)
        return (
            semantic_action,
            B18_ELIGIBILITY_TRACE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b19_controller_params(self) -> dict[str, float]:
        params = self._b18_controller_params()
        defaults = {
            "b19_memory_decay": 0.88,
            "b19_consolidation_threshold": 0.30,
            "b19_switch_suppression_threshold": 0.58,
            "b19_stability_gain": 0.42,
            "b19_memory_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "episodic_meta_memory"
        )
        if profile == "stability_memory":
            defaults.update({"b19_consolidation_threshold": 0.25, "b19_memory_commit_ticks": 6.0})
        elif profile == "switch_suppression":
            defaults.update(
                {"b19_switch_suppression_threshold": 0.62, "b19_stability_gain": 0.56}
            )
        elif profile == "episodic_meta_memory_h56":
            defaults.update({"b19_memory_decay": 0.90, "b19_memory_commit_ticks": 6.0})
        elif profile == "genetic_meta_memory":
            defaults.update({"b19_consolidation_threshold": 0.27, "b19_stability_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b19_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b19_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b19_episode_memory = 0.0
        self._b19_memory_lock = 0
        self._b19_last_tick = int(tick)

    def _b19_episodic_meta_memory_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b18_eligibility_trace_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b19_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "episodic_meta_memory"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b19_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b18_decision = str(trace_payload.get("b18_decision", "preserve_b18"))
        eligibility_trace = float(trace_payload.get("b18_eligibility_trace", 0.0) or 0.0)
        prediction_proxy = float(trace_payload.get("b18_reward_prediction_proxy", 0.0) or 0.0)
        stability_bias = float(trace_payload.get("b18_stability_bias", 0.0) or 0.0)
        switch_pressure = float(trace_payload.get("b18_switch_pressure", 0.0) or 0.0)
        memory_input = float(
            np.clip(
                eligibility_trace * 0.35
                + prediction_proxy * 0.30
                + stability_bias * 0.25
                + max(0.0, stability_bias - switch_pressure) * 0.10,
                0.0,
                1.0,
            )
        )
        episode_memory = max(
            float(getattr(self, "_b19_episode_memory", 0.0))
            * float(params["b19_memory_decay"]),
            memory_input,
        )
        consolidation_score = float(
            np.clip(
                episode_memory * float(params["b19_stability_gain"])
                + stability_bias * 0.35,
                0.0,
                1.0,
            )
        )
        stability_vote = float(
            np.clip(
                consolidation_score + eligibility_trace * 0.20 - switch_pressure * 0.20,
                0.0,
                1.0,
            )
        )
        switch_suppression = float(
            np.clip(switch_pressure - episode_memory * 0.25, 0.0, 1.0)
        )
        memory_lock = int(getattr(self, "_b19_memory_lock", 0))
        memory_state = "non_corridor"
        decision_label = "preserve_b18"

        if corridor_map:
            if (
                b18_decision in {"eligibility_stabilize_option", "continue_trace_lock"}
                and stability_vote >= float(params["b19_consolidation_threshold"])
                and stability_vote >= switch_suppression
            ):
                semantic_action = "MOVE_TO_FOOD"
                memory_lock = max(memory_lock, int(params["b19_memory_commit_ticks"]))
                memory_state = "memory_consolidates_option"
                decision_label = "episodic_consolidate_option"
                reason = "b19_episodic_consolidate_option"
            elif (
                switch_suppression >= float(params["b19_switch_suppression_threshold"])
                and switch_suppression > stability_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                memory_state = "memory_releases_option"
                decision_label = "episodic_release_option"
                reason = "b19_episodic_release_option"
            elif memory_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                memory_state = "memory_lock_continues"
                decision_label = "continue_memory_lock"
                reason = "b19_continue_memory_lock"

        trace_payload.update(
            {
                "b19_controller_profile": profile,
                "b19_memory_state": memory_state,
                "b19_episode_memory": round(float(episode_memory), 6),
                "b19_consolidation_score": round(float(consolidation_score), 6),
                "b19_stability_vote": round(float(stability_vote), 6),
                "b19_switch_suppression": round(float(switch_suppression), 6),
                "b19_memory_lock": int(memory_lock),
                "b19_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b19_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b19_genetic_candidate"] = int(params["ga_candidate"])

        self._b19_episode_memory = float(episode_memory)
        self._b19_memory_lock = max(0, int(memory_lock) - 1)
        self._b19_last_tick = int(tick)
        return (
            semantic_action,
            B19_EPISODIC_META_MEMORY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b20_controller_params(self) -> dict[str, float]:
        params = self._b19_controller_params()
        defaults = {
            "b20_buffer_decay": 0.86,
            "b20_gate_threshold": 0.30,
            "b20_release_threshold": 0.58,
            "b20_context_gain": 0.44,
            "b20_buffer_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "working_memory_gate"
        )
        if profile == "context_binding":
            defaults.update({"b20_gate_threshold": 0.25, "b20_buffer_commit_ticks": 6.0})
        elif profile == "stability_buffer":
            defaults.update({"b20_context_gain": 0.56, "b20_release_threshold": 0.62})
        elif profile == "working_memory_gate_h56":
            defaults.update({"b20_buffer_decay": 0.88, "b20_buffer_commit_ticks": 6.0})
        elif profile == "genetic_working_memory":
            defaults.update({"b20_gate_threshold": 0.27, "b20_context_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b20_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b20_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b20_working_buffer = 0.0
        self._b20_buffer_lock = 0
        self._b20_last_tick = int(tick)

    def _b20_working_memory_gate_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b19_episodic_meta_memory_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b20_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "working_memory_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b20_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b19_decision = str(trace_payload.get("b19_decision", "preserve_b19"))
        episode_memory = float(trace_payload.get("b19_episode_memory", 0.0) or 0.0)
        consolidation_score = float(
            trace_payload.get("b19_consolidation_score", 0.0) or 0.0
        )
        stability_vote = float(trace_payload.get("b19_stability_vote", 0.0) or 0.0)
        switch_suppression = float(
            trace_payload.get("b19_switch_suppression", 0.0) or 0.0
        )
        buffer_input = float(
            np.clip(
                episode_memory * 0.35
                + consolidation_score * 0.25
                + stability_vote * 0.25
                + max(0.0, stability_vote - switch_suppression) * 0.15,
                0.0,
                1.0,
            )
        )
        working_buffer = max(
            float(getattr(self, "_b20_working_buffer", 0.0))
            * float(params["b20_buffer_decay"]),
            buffer_input,
        )
        context_binding = float(
            np.clip(
                working_buffer * float(params["b20_context_gain"])
                + stability_vote * 0.35,
                0.0,
                1.0,
            )
        )
        gate_vote = float(
            np.clip(
                context_binding + episode_memory * 0.15 - switch_suppression * 0.15,
                0.0,
                1.0,
            )
        )
        release_vote = float(
            np.clip(switch_suppression - working_buffer * 0.20, 0.0, 1.0)
        )
        buffer_lock = int(getattr(self, "_b20_buffer_lock", 0))
        buffer_state = "non_corridor"
        decision_label = "preserve_b19"

        if corridor_map:
            if (
                b19_decision in {"episodic_consolidate_option", "continue_memory_lock"}
                and gate_vote >= float(params["b20_gate_threshold"])
                and gate_vote >= release_vote
            ):
                semantic_action = "MOVE_TO_FOOD"
                buffer_lock = max(buffer_lock, int(params["b20_buffer_commit_ticks"]))
                buffer_state = "working_memory_holds_context"
                decision_label = "working_memory_gate_continue"
                reason = "b20_working_memory_gate_continue"
            elif (
                release_vote >= float(params["b20_release_threshold"])
                and release_vote > gate_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                buffer_state = "working_memory_releases_context"
                decision_label = "working_memory_gate_release"
                reason = "b20_working_memory_gate_release"
            elif buffer_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                buffer_state = "working_memory_lock_continues"
                decision_label = "continue_working_memory_lock"
                reason = "b20_continue_working_memory_lock"

        trace_payload.update(
            {
                "b20_controller_profile": profile,
                "b20_buffer_state": buffer_state,
                "b20_working_buffer": round(float(working_buffer), 6),
                "b20_context_binding": round(float(context_binding), 6),
                "b20_gate_vote": round(float(gate_vote), 6),
                "b20_release_vote": round(float(release_vote), 6),
                "b20_buffer_lock": int(buffer_lock),
                "b20_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b20_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b20_genetic_candidate"] = int(params["ga_candidate"])

        self._b20_working_buffer = float(working_buffer)
        self._b20_buffer_lock = max(0, int(buffer_lock) - 1)
        self._b20_last_tick = int(tick)
        return (
            semantic_action,
            B20_WORKING_MEMORY_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b21_controller_params(self) -> dict[str, float]:
        params = self._b20_controller_params()
        defaults = {
            "b21_replay_decay": 0.84,
            "b21_replay_threshold": 0.30,
            "b21_abort_threshold": 0.58,
            "b21_sequence_gain": 0.46,
            "b21_replay_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "hippocampal_replay"
        )
        if profile == "sequence_binding":
            defaults.update({"b21_replay_threshold": 0.25, "b21_replay_commit_ticks": 6.0})
        elif profile == "route_rehearsal":
            defaults.update({"b21_sequence_gain": 0.58, "b21_abort_threshold": 0.62})
        elif profile == "hippocampal_replay_h56":
            defaults.update({"b21_replay_decay": 0.86, "b21_replay_commit_ticks": 6.0})
        elif profile == "genetic_replay_gate":
            defaults.update({"b21_replay_threshold": 0.27, "b21_sequence_gain": 0.52})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b21_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b21_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b21_sequence_memory = 0.0
        self._b21_replay_lock = 0
        self._b21_last_tick = int(tick)

    def _b21_hippocampal_replay_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b20_working_memory_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b21_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "hippocampal_replay"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b21_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b20_decision = str(trace_payload.get("b20_decision", "preserve_b20"))
        working_buffer = float(trace_payload.get("b20_working_buffer", 0.0) or 0.0)
        context_binding = float(trace_payload.get("b20_context_binding", 0.0) or 0.0)
        gate_vote = float(trace_payload.get("b20_gate_vote", 0.0) or 0.0)
        release_vote = float(trace_payload.get("b20_release_vote", 0.0) or 0.0)
        sequence_input = float(
            np.clip(
                working_buffer * 0.30
                + context_binding * 0.30
                + gate_vote * 0.25
                + max(0.0, gate_vote - release_vote) * 0.15,
                0.0,
                1.0,
            )
        )
        sequence_memory = max(
            float(getattr(self, "_b21_sequence_memory", 0.0))
            * float(params["b21_replay_decay"]),
            sequence_input,
        )
        replay_score = float(
            np.clip(
                sequence_memory * float(params["b21_sequence_gain"])
                + context_binding * 0.35,
                0.0,
                1.0,
            )
        )
        route_commitment = float(
            np.clip(replay_score + working_buffer * 0.20 - release_vote * 0.15, 0.0, 1.0)
        )
        abort_prediction = float(
            np.clip(release_vote - sequence_memory * 0.20, 0.0, 1.0)
        )
        replay_lock = int(getattr(self, "_b21_replay_lock", 0))
        replay_state = "non_corridor"
        decision_label = "preserve_b20"

        if corridor_map:
            if (
                b20_decision in {"working_memory_gate_continue", "continue_working_memory_lock"}
                and route_commitment >= float(params["b21_replay_threshold"])
                and route_commitment >= abort_prediction
            ):
                semantic_action = "MOVE_TO_FOOD"
                replay_lock = max(replay_lock, int(params["b21_replay_commit_ticks"]))
                replay_state = "replay_rehearses_route"
                decision_label = "hippocampal_replay_continue"
                reason = "b21_hippocampal_replay_continue"
            elif (
                abort_prediction >= float(params["b21_abort_threshold"])
                and abort_prediction > route_commitment
            ):
                semantic_action = "MOVE_TO_SHELTER"
                replay_state = "replay_predicts_abort"
                decision_label = "hippocampal_replay_abort"
                reason = "b21_hippocampal_replay_abort"
            elif replay_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                replay_state = "replay_lock_continues"
                decision_label = "continue_replay_lock"
                reason = "b21_continue_replay_lock"

        trace_payload.update(
            {
                "b21_controller_profile": profile,
                "b21_replay_state": replay_state,
                "b21_sequence_memory": round(float(sequence_memory), 6),
                "b21_replay_score": round(float(replay_score), 6),
                "b21_route_commitment": round(float(route_commitment), 6),
                "b21_abort_prediction": round(float(abort_prediction), 6),
                "b21_replay_lock": int(replay_lock),
                "b21_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b21_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b21_genetic_candidate"] = int(params["ga_candidate"])

        self._b21_sequence_memory = float(sequence_memory)
        self._b21_replay_lock = max(0, int(replay_lock) - 1)
        self._b21_last_tick = int(tick)
        return (
            semantic_action,
            B21_HIPPOCAMPAL_REPLAY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b22_controller_params(self) -> dict[str, float]:
        params = self._b21_controller_params()
        defaults = {
            "b22_sim_decay": 0.84,
            "b22_viability_threshold": 0.30,
            "b22_abort_threshold": 0.58,
            "b22_forward_gain": 0.48,
            "b22_sim_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "prospective_map_replay"
        )
        if profile == "forward_model_gate":
            defaults.update(
                {"b22_viability_threshold": 0.25, "b22_sim_commit_ticks": 6.0}
            )
        elif profile == "route_viability_sim":
            defaults.update({"b22_forward_gain": 0.60, "b22_abort_threshold": 0.62})
        elif profile == "prospective_map_replay_h56":
            defaults.update({"b22_sim_decay": 0.86, "b22_sim_commit_ticks": 6.0})
        elif profile == "genetic_prospective_replay":
            defaults.update({"b22_viability_threshold": 0.27, "b22_forward_gain": 0.54})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b22_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b22_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b22_prospective_sim = 0.0
        self._b22_sim_lock = 0
        self._b22_last_tick = int(tick)

    def _b22_prospective_replay_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, object]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b21_hippocampal_replay_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b22_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "prospective_map_replay"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b22_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b21_decision = str(trace_payload.get("b21_decision", "preserve_b21"))
        sequence_memory = float(trace_payload.get("b21_sequence_memory", 0.0) or 0.0)
        replay_score = float(trace_payload.get("b21_replay_score", 0.0) or 0.0)
        route_commitment = float(
            trace_payload.get("b21_route_commitment", 0.0) or 0.0
        )
        abort_prediction = float(
            trace_payload.get("b21_abort_prediction", 0.0) or 0.0
        )
        sim_input = float(
            np.clip(
                sequence_memory * 0.30
                + replay_score * 0.30
                + route_commitment * 0.25
                + max(0.0, route_commitment - abort_prediction) * 0.15,
                0.0,
                1.0,
            )
        )
        prospective_sim = max(
            float(getattr(self, "_b22_prospective_sim", 0.0))
            * float(params["b22_sim_decay"]),
            sim_input,
        )
        forward_model_score = float(
            np.clip(
                prospective_sim * float(params["b22_forward_gain"])
                + route_commitment * 0.35,
                0.0,
                1.0,
            )
        )
        viability_projection = float(
            np.clip(
                forward_model_score + sequence_memory * 0.20 - abort_prediction * 0.15,
                0.0,
                1.0,
            )
        )
        abort_projection = float(
            np.clip(abort_prediction - prospective_sim * 0.20, 0.0, 1.0)
        )
        sim_lock = int(getattr(self, "_b22_sim_lock", 0))
        sim_state = "non_corridor"
        decision_label = "preserve_b21"

        if corridor_map:
            if (
                b21_decision
                in {"hippocampal_replay_continue", "continue_replay_lock"}
                and viability_projection >= float(params["b22_viability_threshold"])
                and viability_projection >= abort_projection
            ):
                semantic_action = "MOVE_TO_FOOD"
                sim_lock = max(sim_lock, int(params["b22_sim_commit_ticks"]))
                sim_state = "prospective_sim_commits_route"
                decision_label = "prospective_replay_continue"
                reason = "b22_prospective_replay_continue"
            elif (
                abort_projection >= float(params["b22_abort_threshold"])
                and abort_projection > viability_projection
            ):
                semantic_action = "MOVE_TO_SHELTER"
                sim_state = "prospective_sim_aborts_route"
                decision_label = "prospective_replay_abort"
                reason = "b22_prospective_replay_abort"
            elif sim_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                sim_state = "prospective_lock_continues"
                decision_label = "continue_prospective_lock"
                reason = "b22_continue_prospective_lock"

        trace_payload.update(
            {
                "b22_controller_profile": profile,
                "b22_sim_state": sim_state,
                "b22_prospective_sim": round(float(prospective_sim), 6),
                "b22_forward_model_score": round(float(forward_model_score), 6),
                "b22_viability_projection": round(float(viability_projection), 6),
                "b22_abort_projection": round(float(abort_projection), 6),
                "b22_sim_lock": int(sim_lock),
                "b22_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b22_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b22_genetic_candidate"] = int(params["ga_candidate"])

        self._b22_prospective_sim = float(prospective_sim)
        self._b22_sim_lock = max(0, int(sim_lock) - 1)
        self._b22_last_tick = int(tick)
        return (
            semantic_action,
            B22_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b23_controller_params(self) -> dict[str, float]:
        params = self._b22_controller_params()
        defaults = {
            "b23_conflict_decay": 0.86,
            "b23_conflict_threshold": 0.24,
            "b23_abort_bias_threshold": 0.62,
            "b23_error_gain": 0.46,
            "b23_monitor_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "conflict_monitor"
        )
        if profile == "error_gated_replay":
            defaults.update(
                {"b23_conflict_threshold": 0.20, "b23_monitor_commit_ticks": 6.0}
            )
        elif profile == "abort_conflict_arbiter":
            defaults.update({"b23_abort_bias_threshold": 0.58, "b23_error_gain": 0.54})
        elif profile == "conflict_monitor_h56":
            defaults.update({"b23_conflict_decay": 0.88, "b23_monitor_commit_ticks": 6.0})
        elif profile == "genetic_conflict_monitor":
            defaults.update({"b23_conflict_threshold": 0.22, "b23_error_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b23_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b23_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b23_conflict_memory = 0.0
        self._b23_monitor_lock = 0
        self._b23_last_tick = int(tick)
