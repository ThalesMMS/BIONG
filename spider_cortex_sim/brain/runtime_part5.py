from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart5Mixin:
    def _b32_controller_params(self) -> dict[str, float]:
        params = self._b31_controller_params()
        defaults = {
            "b32_value_decay": 0.88,
            "b32_advantage_threshold": 0.20,
            "b32_abort_threshold": -0.18,
            "b32_critic_gain": 0.46,
            "b32_actor_gain": 0.50,
            "b32_value_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "actor_critic_value"
        )
        if profile == "advantage_value_gate":
            defaults.update({"b32_advantage_threshold": 0.18, "b32_actor_gain": 0.56})
        elif profile == "critic_stability":
            defaults.update({"b32_value_decay": 0.90, "b32_value_commit_ticks": 6.0})
        elif profile == "actor_critic_value_h56":
            defaults.update({"b32_value_decay": 0.90, "b32_value_commit_ticks": 6.0})
        elif profile == "genetic_actor_critic":
            defaults.update({"b32_advantage_threshold": 0.18, "b32_actor_gain": 0.54})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b32_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b32_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b32_critic_value = 0.0
        self._b32_value_lock = 0
        self._b32_last_tick = int(tick)

    def _b32_actor_critic_value_semantic_action(
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
        ) = self._b31_dopamine_prediction_error_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b32_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "actor_critic_value"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b32_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        gate_bias = float(trace_payload.get("b31_gate_bias", 0.0) or 0.0)
        tonic_dopamine = float(trace_payload.get("b31_tonic_dopamine", 0.0) or 0.0)
        phasic_dopamine = float(trace_payload.get("b31_phasic_dopamine", 0.0) or 0.0)
        b31_decision = str(trace_payload.get("b31_decision", "preserve_b31"))
        no_go_signal = float(trace_payload.get("b30_no_go_signal", 0.0) or 0.0)
        go_signal = float(trace_payload.get("b30_go_signal", 0.0) or 0.0)

        current_value = float(getattr(self, "_b32_critic_value", 0.0))
        reward_proxy = float(
            np.clip(
                gate_bias * 0.34
                + tonic_dopamine * 0.24
                + phasic_dopamine * 0.20
                + go_signal * 0.16
                + max(0.0, health - 0.5) * 0.06
                - no_go_signal * 0.22
                - max(0.0, 0.72 - hunger) * 0.05,
                -1.0,
                1.0,
            )
        )
        value_error = reward_proxy - current_value
        critic_value = float(
            np.clip(
                current_value * float(params["b32_value_decay"])
                + reward_proxy * float(params["b32_critic_gain"]),
                -1.0,
                1.0,
            )
        )
        actor_advantage = float(
            np.clip(
                value_error * float(params["b32_actor_gain"])
                + gate_bias * 0.35
                + go_signal * 0.18
                - no_go_signal * 0.20,
                -1.0,
                1.0,
            )
        )
        policy_bias = float(np.clip(critic_value + actor_advantage, -1.0, 1.0))
        value_lock = int(getattr(self, "_b32_value_lock", 0))
        decision_label = "preserve_b31"

        if corridor_map:
            if (
                b31_decision in {"dopamine_gate_go", "continue_dopamine_lock"}
                and actor_advantage >= float(params["b32_advantage_threshold"])
                and policy_bias > 0.0
            ):
                semantic_action = "MOVE_TO_FOOD"
                value_lock = max(value_lock, int(params["b32_value_commit_ticks"]))
                decision_label = "actor_critic_commit"
                reason = "b32_actor_critic_commit"
            elif policy_bias <= float(params["b32_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "critic_abort_negative_value"
                reason = "b32_critic_abort_negative_value"
            elif value_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_value_lock"
                reason = "b32_continue_value_lock"

        trace_payload.update(
            {
                "b32_controller_profile": profile,
                "b32_critic_value": round(float(critic_value), 6),
                "b32_actor_advantage": round(float(actor_advantage), 6),
                "b32_value_error": round(float(value_error), 6),
                "b32_policy_bias": round(float(policy_bias), 6),
                "b32_value_lock": int(value_lock),
                "b32_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b32_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b32_genetic_candidate"] = int(params["ga_candidate"])

        self._b32_critic_value = float(critic_value)
        self._b32_value_lock = max(0, int(value_lock) - 1)
        self._b32_last_tick = int(tick)
        return (
            semantic_action,
            B32_ACTOR_CRITIC_VALUE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b33_controller_params(self) -> dict[str, float]:
        params = self._b32_controller_params()
        defaults = {
            "b33_td_decay": 0.88,
            "b33_bootstrap_gain": 0.42,
            "b33_reward_trace_gain": 0.36,
            "b33_td_commit_ticks": 5.0,
            "b33_td_threshold": 0.16,
            "b33_abort_threshold": -0.20,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "td_error_decomposition"
        )
        if profile == "bootstrapped_value_gate":
            defaults.update({"b33_bootstrap_gain": 0.48, "b33_td_commit_ticks": 6.0})
        elif profile == "reward_trace_critic":
            defaults.update({"b33_reward_trace_gain": 0.44, "b33_td_threshold": 0.14})
        elif profile == "td_error_decomposition_h56":
            defaults.update({"b33_td_decay": 0.90, "b33_td_commit_ticks": 6.0})
        elif profile == "genetic_td_value":
            defaults.update({"b33_bootstrap_gain": 0.46, "b33_reward_trace_gain": 0.40})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b33_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b33_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b33_bootstrap_value = 0.0
        self._b33_reward_trace = 0.0
        self._b33_td_lock = 0
        self._b33_last_tick = int(tick)

    def _b33_td_error_decomposition_semantic_action(
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
        ) = self._b32_actor_critic_value_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b33_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "td_error_decomposition"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b33_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        critic_value = float(trace_payload.get("b32_critic_value", 0.0) or 0.0)
        actor_advantage = float(trace_payload.get("b32_actor_advantage", 0.0) or 0.0)
        policy_bias = float(trace_payload.get("b32_policy_bias", 0.0) or 0.0)
        value_error = float(trace_payload.get("b32_value_error", 0.0) or 0.0)
        b32_decision = str(trace_payload.get("b32_decision", "preserve_b32"))
        dopamine_error = float(
            trace_payload.get("b31_reward_prediction_error", 0.0) or 0.0
        )
        gate_bias = float(trace_payload.get("b31_gate_bias", 0.0) or 0.0)
        previous_bootstrap = float(getattr(self, "_b33_bootstrap_value", 0.0))
        previous_trace = float(getattr(self, "_b33_reward_trace", 0.0))
        reward_trace = float(
            np.clip(
                previous_trace * float(params["b33_td_decay"])
                + max(0.0, actor_advantage) * float(params["b33_reward_trace_gain"]),
                -1.0,
                1.0,
            )
        )
        bootstrap_value = float(
            np.clip(
                previous_bootstrap * float(params["b33_td_decay"])
                + critic_value * float(params["b33_bootstrap_gain"])
                + policy_bias * 0.22,
                -1.0,
                1.0,
            )
        )
        td_error = float(
            np.clip(
                value_error
                + dopamine_error * 0.35
                + reward_trace * 0.28
                + bootstrap_value * 0.20
                - previous_bootstrap * 0.18,
                -1.0,
                1.0,
            )
        )
        actor_update = float(
            np.clip(
                actor_advantage + td_error * 0.45 + gate_bias * 0.12,
                -1.0,
                1.0,
            )
        )
        td_lock = int(getattr(self, "_b33_td_lock", 0))
        decision_label = "preserve_b32"

        if corridor_map:
            if (
                b32_decision in {"actor_critic_commit", "continue_value_lock"}
                and actor_update >= float(params["b33_td_threshold"])
                and td_error > float(params["b33_abort_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                td_lock = max(td_lock, int(params["b33_td_commit_ticks"]))
                decision_label = "td_error_commit"
                reason = "b33_td_error_commit"
            elif td_error <= float(params["b33_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "td_error_abort"
                reason = "b33_td_error_abort"
            elif td_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_td_lock"
                reason = "b33_continue_td_lock"

        trace_payload.update(
            {
                "b33_controller_profile": profile,
                "b33_td_error": round(float(td_error), 6),
                "b33_bootstrap_value": round(float(bootstrap_value), 6),
                "b33_reward_trace": round(float(reward_trace), 6),
                "b33_actor_update": round(float(actor_update), 6),
                "b33_td_lock": int(td_lock),
                "b33_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b33_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b33_genetic_candidate"] = int(params["ga_candidate"])

        self._b33_bootstrap_value = float(bootstrap_value)
        self._b33_reward_trace = float(reward_trace)
        self._b33_td_lock = max(0, int(td_lock) - 1)
        self._b33_last_tick = int(tick)
        return (
            semantic_action,
            B33_TD_ERROR_DECOMPOSITION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b34_controller_params(self) -> dict[str, float]:
        params = self._b33_controller_params()
        defaults = {
            "b34_eligibility_decay": 0.86,
            "b34_credit_gain": 0.38,
            "b34_tag_gain": 0.32,
            "b34_credit_threshold": 0.12,
            "b34_abort_threshold": -0.18,
            "b34_credit_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "eligibility_credit"
        )
        if profile == "delayed_credit_gate":
            defaults.update({"b34_credit_gain": 0.44, "b34_credit_lock_ticks": 6.0})
        elif profile == "synaptic_tagging":
            defaults.update({"b34_tag_gain": 0.42, "b34_credit_threshold": 0.10})
        elif profile == "eligibility_credit_h56":
            defaults.update({"b34_eligibility_decay": 0.89, "b34_credit_lock_ticks": 6.0})
        elif profile == "genetic_eligibility":
            defaults.update({"b34_credit_gain": 0.42, "b34_tag_gain": 0.38})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b34_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b34_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b34_eligibility_trace = 0.0
        self._b34_synaptic_tag = 0.0
        self._b34_credit_lock = 0
        self._b34_last_tick = int(tick)

    def _b34_eligibility_credit_semantic_action(
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
        ) = self._b33_td_error_decomposition_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b34_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "eligibility_credit"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b34_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        td_error = float(trace_payload.get("b33_td_error", 0.0) or 0.0)
        actor_update = float(trace_payload.get("b33_actor_update", 0.0) or 0.0)
        reward_trace = float(trace_payload.get("b33_reward_trace", 0.0) or 0.0)
        b33_decision = str(trace_payload.get("b33_decision", "preserve_b33"))
        previous_eligibility = float(getattr(self, "_b34_eligibility_trace", 0.0))
        previous_tag = float(getattr(self, "_b34_synaptic_tag", 0.0))
        decay = float(params["b34_eligibility_decay"])
        eligibility_trace = float(
            np.clip(previous_eligibility * decay + max(0.0, actor_update) * 0.42, -1.0, 1.0)
        )
        synaptic_tag = float(
            np.clip(previous_tag * decay + max(0.0, reward_trace) * float(params["b34_tag_gain"]), -1.0, 1.0)
        )
        credit_assignment = float(
            np.clip(
                eligibility_trace * float(params["b34_credit_gain"])
                + synaptic_tag * 0.30
                + td_error * 0.34,
                -1.0,
                1.0,
            )
        )
        decay_memory = float(np.clip(previous_eligibility * decay, -1.0, 1.0))
        credit_lock = int(getattr(self, "_b34_credit_lock", 0))
        decision_label = "preserve_b33"

        if corridor_map:
            if (
                b33_decision in {"td_error_commit", "continue_td_lock"}
                and credit_assignment >= float(params["b34_credit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                credit_lock = max(credit_lock, int(params["b34_credit_lock_ticks"]))
                decision_label = "eligibility_credit_commit"
                reason = "b34_eligibility_credit_commit"
            elif credit_assignment <= float(params["b34_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "eligibility_credit_abort"
                reason = "b34_eligibility_credit_abort"
            elif credit_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_eligibility_lock"
                reason = "b34_continue_eligibility_lock"

        trace_payload.update(
            {
                "b34_controller_profile": profile,
                "b34_eligibility_trace": round(float(eligibility_trace), 6),
                "b34_credit_assignment": round(float(credit_assignment), 6),
                "b34_synaptic_tag": round(float(synaptic_tag), 6),
                "b34_decay_memory": round(float(decay_memory), 6),
                "b34_credit_lock": int(credit_lock),
                "b34_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b34_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b34_genetic_candidate"] = int(params["ga_candidate"])

        self._b34_eligibility_trace = float(eligibility_trace)
        self._b34_synaptic_tag = float(synaptic_tag)
        self._b34_credit_lock = max(0, int(credit_lock) - 1)
        self._b34_last_tick = int(tick)
        return (
            semantic_action,
            B34_ELIGIBILITY_CREDIT_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b35_controller_params(self) -> dict[str, float]:
        params = self._b34_controller_params()
        defaults = {
            "b35_model_decay": 0.84,
            "b35_prediction_gain": 0.40,
            "b35_transition_gain": 0.34,
            "b35_confidence_gain": 0.30,
            "b35_commit_threshold": 0.14,
            "b35_abort_threshold": -0.16,
            "b35_model_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "forward_model_value"
        )
        if profile == "transition_error_gate":
            defaults.update({"b35_transition_gain": 0.42, "b35_commit_threshold": 0.12})
        elif profile == "model_confidence":
            defaults.update({"b35_confidence_gain": 0.38, "b35_model_lock_ticks": 6.0})
        elif profile == "forward_model_value_h56":
            defaults.update({"b35_model_decay": 0.87, "b35_model_lock_ticks": 6.0})
        elif profile == "genetic_forward_model":
            defaults.update({"b35_prediction_gain": 0.44, "b35_confidence_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b35_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b35_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b35_prediction_memory = 0.0
        self._b35_model_confidence = 0.0
        self._b35_model_lock = 0
        self._b35_last_tick = int(tick)

    def _b35_forward_model_value_semantic_action(
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
        ) = self._b34_eligibility_credit_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b35_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "forward_model_value"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b35_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        food_dist = float(meta.get("food_dist", 0.0) or 0.0)
        shelter_dist = float(meta.get("shelter_dist", 0.0) or 0.0)
        credit_assignment = float(trace_payload.get("b34_credit_assignment", 0.0) or 0.0)
        eligibility_trace = float(trace_payload.get("b34_eligibility_trace", 0.0) or 0.0)
        td_error = float(trace_payload.get("b33_td_error", 0.0) or 0.0)
        b34_decision = str(trace_payload.get("b34_decision", "preserve_b34"))
        previous_prediction = float(getattr(self, "_b35_prediction_memory", 0.0))
        previous_confidence = float(getattr(self, "_b35_model_confidence", 0.0))
        distance_advantage = float(np.clip((shelter_dist - food_dist) / 12.0, -1.0, 1.0))
        predicted_delta = float(
            np.clip(
                credit_assignment * float(params["b35_prediction_gain"])
                + eligibility_trace * 0.24
                + distance_advantage * 0.18,
                -1.0,
                1.0,
            )
        )
        prediction_memory = float(
            np.clip(
                previous_prediction * float(params["b35_model_decay"]) + predicted_delta,
                -1.0,
                1.0,
            )
        )
        transition_error = float(
            np.clip(
                predicted_delta
                - previous_prediction * 0.28
                + td_error * float(params["b35_transition_gain"]),
                -1.0,
                1.0,
            )
        )
        model_confidence = float(
            np.clip(
                previous_confidence * float(params["b35_model_decay"])
                + max(0.0, credit_assignment) * float(params["b35_confidence_gain"])
                - max(0.0, -transition_error) * 0.18,
                -1.0,
                1.0,
            )
        )
        forward_value = float(
            np.clip(prediction_memory * 0.48 + transition_error * 0.32 + model_confidence * 0.20, -1.0, 1.0)
        )
        model_lock = int(getattr(self, "_b35_model_lock", 0))
        decision_label = "preserve_b34"

        if corridor_map:
            if (
                b34_decision in {"eligibility_credit_commit", "continue_eligibility_lock"}
                and forward_value >= float(params["b35_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                model_lock = max(model_lock, int(params["b35_model_lock_ticks"]))
                decision_label = "forward_model_commit"
                reason = "b35_forward_model_commit"
            elif forward_value <= float(params["b35_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "forward_model_abort"
                reason = "b35_forward_model_abort"
            elif model_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_model_lock"
                reason = "b35_continue_model_lock"

        trace_payload.update(
            {
                "b35_controller_profile": profile,
                "b35_forward_value": round(float(forward_value), 6),
                "b35_transition_error": round(float(transition_error), 6),
                "b35_model_confidence": round(float(model_confidence), 6),
                "b35_prediction_memory": round(float(prediction_memory), 6),
                "b35_model_lock": int(model_lock),
                "b35_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b35_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b35_genetic_candidate"] = int(params["ga_candidate"])

        self._b35_prediction_memory = float(prediction_memory)
        self._b35_model_confidence = float(model_confidence)
        self._b35_model_lock = max(0, int(model_lock) - 1)
        self._b35_last_tick = int(tick)
        return (
            semantic_action,
            B35_FORWARD_MODEL_VALUE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b36_controller_params(self) -> dict[str, float]:
        params = self._b35_controller_params()
        defaults = {
            "b36_belief_decay": 0.86,
            "b36_latent_gain": 0.38,
            "b36_context_gain": 0.32,
            "b36_error_gain": 0.34,
            "b36_commit_threshold": 0.14,
            "b36_abort_threshold": -0.16,
            "b36_belief_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "latent_belief_state"
        )
        if profile == "belief_error_gate":
            defaults.update({"b36_error_gain": 0.42, "b36_commit_threshold": 0.12})
        elif profile == "context_inference":
            defaults.update({"b36_context_gain": 0.40, "b36_belief_lock_ticks": 6.0})
        elif profile == "latent_belief_state_h56":
            defaults.update({"b36_belief_decay": 0.89, "b36_belief_lock_ticks": 6.0})
        elif profile == "genetic_belief_state":
            defaults.update({"b36_latent_gain": 0.42, "b36_context_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b36_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b36_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b36_latent_state = 0.0
        self._b36_context_memory = 0.0
        self._b36_state_confidence = 0.0
        self._b36_belief_lock = 0
        self._b36_last_tick = int(tick)

    def _b36_latent_belief_state_semantic_action(
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
        ) = self._b35_forward_model_value_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b36_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "latent_belief_state"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b36_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        food_dist = float(meta.get("food_dist", 0.0) or 0.0)
        shelter_dist = float(meta.get("shelter_dist", 0.0) or 0.0)
        forward_value = float(trace_payload.get("b35_forward_value", 0.0) or 0.0)
        transition_error = float(trace_payload.get("b35_transition_error", 0.0) or 0.0)
        model_confidence = float(trace_payload.get("b35_model_confidence", 0.0) or 0.0)
        prediction_memory = float(trace_payload.get("b35_prediction_memory", 0.0) or 0.0)
        b35_decision = str(trace_payload.get("b35_decision", "preserve_b35"))
        previous_latent = float(getattr(self, "_b36_latent_state", 0.0))
        previous_context = float(getattr(self, "_b36_context_memory", 0.0))
        previous_confidence = float(getattr(self, "_b36_state_confidence", 0.0))
        corridor_context = float(np.clip((shelter_dist - food_dist) / 12.0, -1.0, 1.0))
        context_memory = float(
            np.clip(
                previous_context * float(params["b36_belief_decay"])
                + corridor_context * float(params["b36_context_gain"]),
                -1.0,
                1.0,
            )
        )
        latent_state = float(
            np.clip(
                previous_latent * float(params["b36_belief_decay"])
                + forward_value * float(params["b36_latent_gain"])
                + prediction_memory * 0.22
                + context_memory * 0.18,
                -1.0,
                1.0,
            )
        )
        belief_error = float(
            np.clip(
                transition_error * float(params["b36_error_gain"])
                + latent_state
                - previous_latent * 0.30,
                -1.0,
                1.0,
            )
        )
        state_confidence = float(
            np.clip(
                previous_confidence * float(params["b36_belief_decay"])
                + max(0.0, model_confidence) * 0.34
                + max(0.0, latent_state) * 0.24
                - max(0.0, -belief_error) * 0.18,
                -1.0,
                1.0,
            )
        )
        belief_score = float(
            np.clip(latent_state * 0.42 + belief_error * 0.30 + state_confidence * 0.28, -1.0, 1.0)
        )
        belief_lock = int(getattr(self, "_b36_belief_lock", 0))
        decision_label = "preserve_b35"

        if corridor_map:
            if (
                b35_decision in {"forward_model_commit", "continue_model_lock"}
                and belief_score >= float(params["b36_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                belief_lock = max(belief_lock, int(params["b36_belief_lock_ticks"]))
                decision_label = "latent_belief_commit"
                reason = "b36_latent_belief_commit"
            elif belief_score <= float(params["b36_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "latent_belief_abort"
                reason = "b36_latent_belief_abort"
            elif belief_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_belief_lock"
                reason = "b36_continue_belief_lock"

        trace_payload.update(
            {
                "b36_controller_profile": profile,
                "b36_latent_state": round(float(latent_state), 6),
                "b36_belief_error": round(float(belief_error), 6),
                "b36_state_confidence": round(float(state_confidence), 6),
                "b36_context_memory": round(float(context_memory), 6),
                "b36_belief_lock": int(belief_lock),
                "b36_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b36_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b36_genetic_candidate"] = int(params["ga_candidate"])

        self._b36_latent_state = float(latent_state)
        self._b36_context_memory = float(context_memory)
        self._b36_state_confidence = float(state_confidence)
        self._b36_belief_lock = max(0, int(belief_lock) - 1)
        self._b36_last_tick = int(tick)
        return (
            semantic_action,
            B36_LATENT_BELIEF_STATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b37_controller_params(self) -> dict[str, float]:
        params = self._b36_controller_params()
        defaults = {
            "b37_factor_decay": 0.86,
            "b37_external_gain": 0.36,
            "b37_internal_gain": 0.34,
            "b37_factor_balance_gain": 0.32,
            "b37_commit_threshold": 0.14,
            "b37_abort_threshold": -0.16,
            "b37_factor_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "state_factor_gate"
        )
        if profile == "intero_extero_factor":
            defaults.update({"b37_external_gain": 0.40, "b37_internal_gain": 0.38})
        elif profile == "factor_confidence":
            defaults.update(
                {"b37_factor_balance_gain": 0.40, "b37_factor_lock_ticks": 6.0}
            )
        elif profile == "state_factor_gate_h56":
            defaults.update({"b37_factor_decay": 0.89, "b37_factor_lock_ticks": 6.0})
        elif profile == "genetic_state_factor":
            defaults.update({"b37_external_gain": 0.38, "b37_internal_gain": 0.38})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b37_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b37_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b37_external_state_factor = 0.0
        self._b37_internal_state_factor = 0.0
        self._b37_factor_confidence = 0.0
        self._b37_factor_lock = 0
        self._b37_last_tick = int(tick)

    def _b37_state_factor_gate_semantic_action(
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
        ) = self._b36_latent_belief_state_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b37_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "state_factor_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b37_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        food_dist = float(meta.get("food_dist", 0.0) or 0.0)
        shelter_dist = float(meta.get("shelter_dist", 0.0) or 0.0)
        corridor_context = float(np.clip((shelter_dist - food_dist) / 12.0, -1.0, 1.0))
        latent_state = float(trace_payload.get("b36_latent_state", 0.0) or 0.0)
        belief_error = float(trace_payload.get("b36_belief_error", 0.0) or 0.0)
        state_confidence = float(trace_payload.get("b36_state_confidence", 0.0) or 0.0)
        context_memory = float(trace_payload.get("b36_context_memory", 0.0) or 0.0)
        b36_decision = str(trace_payload.get("b36_decision", "preserve_b35"))

        decay = float(params["b37_factor_decay"])
        previous_external = float(getattr(self, "_b37_external_state_factor", 0.0))
        previous_internal = float(getattr(self, "_b37_internal_state_factor", 0.0))
        previous_confidence = float(getattr(self, "_b37_factor_confidence", 0.0))
        external_factor = float(
            np.clip(
                previous_external * decay
                + context_memory * float(params["b37_external_gain"])
                + corridor_context * 0.18,
                -1.0,
                1.0,
            )
        )
        internal_factor = float(
            np.clip(
                previous_internal * decay
                + latent_state * float(params["b37_internal_gain"])
                + state_confidence * 0.22,
                -1.0,
                1.0,
            )
        )
        balance_gain = float(params.get("b37_factor_balance_gain", 0.32))
        factor_alignment = float(
            np.clip(
                (external_factor + internal_factor) * 0.5
                + belief_error * balance_gain,
                -1.0,
                1.0,
            )
        )
        factor_confidence = float(
            np.clip(
                previous_confidence * decay
                + max(0.0, state_confidence) * 0.30
                + max(0.0, factor_alignment) * 0.24
                - max(0.0, -factor_alignment) * 0.18,
                -1.0,
                1.0,
            )
        )
        factor_score = float(
            np.clip(
                factor_alignment * 0.48
                + factor_confidence * 0.32
                + internal_factor * 0.20,
                -1.0,
                1.0,
            )
        )
        factor_lock = int(getattr(self, "_b37_factor_lock", 0))
        decision_label = "preserve_b36"

        if corridor_map:
            if (
                b36_decision in {"latent_belief_commit", "continue_belief_lock"}
                and factor_score >= float(params["b37_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                factor_lock = max(factor_lock, int(params["b37_factor_lock_ticks"]))
                decision_label = "state_factor_commit"
                reason = "b37_state_factor_commit"
            elif factor_score <= float(params["b37_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "state_factor_abort"
                reason = "b37_state_factor_abort"
            elif factor_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_factor_lock"
                reason = "b37_continue_factor_lock"

        trace_payload.update(
            {
                "b37_controller_profile": profile,
                "b37_external_state_factor": round(float(external_factor), 6),
                "b37_internal_state_factor": round(float(internal_factor), 6),
                "b37_factor_alignment": round(float(factor_alignment), 6),
                "b37_factor_confidence": round(float(factor_confidence), 6),
                "b37_factor_lock": int(factor_lock),
                "b37_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b37_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b37_genetic_candidate"] = int(params["ga_candidate"])

        self._b37_external_state_factor = float(external_factor)
        self._b37_internal_state_factor = float(internal_factor)
        self._b37_factor_confidence = float(factor_confidence)
        self._b37_factor_lock = max(0, int(factor_lock) - 1)
        self._b37_last_tick = int(tick)
        return (
            semantic_action,
            B37_STATE_FACTOR_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b38_controller_params(self) -> dict[str, float]:
        params = self._b37_controller_params()
        defaults = {
            "b38_attention_decay": 0.86,
            "b38_external_attention_gain": 0.34,
            "b38_internal_attention_gain": 0.36,
            "b38_confidence_attention_gain": 0.32,
            "b38_attention_threshold": 0.13,
            "b38_abort_threshold": -0.17,
            "b38_attention_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "factor_attention"
        )
        if profile == "interoceptive_attention":
            defaults.update(
                {"b38_internal_attention_gain": 0.42, "b38_attention_lock_ticks": 6.0}
            )
        elif profile == "confidence_attention":
            defaults.update(
                {"b38_confidence_attention_gain": 0.42, "b38_attention_threshold": 0.11}
            )
        elif profile == "factor_attention_h56":
            defaults.update({"b38_attention_decay": 0.89, "b38_attention_lock_ticks": 6.0})
        elif profile == "genetic_factor_attention":
            defaults.update(
                {"b38_external_attention_gain": 0.36, "b38_internal_attention_gain": 0.38}
            )
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b38_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b38_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b38_external_attention = 0.0
        self._b38_internal_attention = 0.0
        self._b38_attention_gain = 0.0
        self._b38_attention_lock = 0
        self._b38_last_tick = int(tick)

    def _b38_factor_attention_semantic_action(
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
        ) = self._b37_state_factor_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b38_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "factor_attention"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b38_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        external_factor = float(trace_payload.get("b37_external_state_factor", 0.0) or 0.0)
        internal_factor = float(trace_payload.get("b37_internal_state_factor", 0.0) or 0.0)
        factor_alignment = float(trace_payload.get("b37_factor_alignment", 0.0) or 0.0)
        factor_confidence = float(trace_payload.get("b37_factor_confidence", 0.0) or 0.0)
        b37_decision = str(trace_payload.get("b37_decision", "preserve_b36"))

        decay = float(params["b38_attention_decay"])
        previous_external = float(getattr(self, "_b38_external_attention", 0.0))
        previous_internal = float(getattr(self, "_b38_internal_attention", 0.0))
        previous_gain = float(getattr(self, "_b38_attention_gain", 0.0))
        external_attention = float(
            np.clip(
                previous_external * decay
                + abs(external_factor) * float(params["b38_external_attention_gain"]),
                -1.0,
                1.0,
            )
        )
        internal_attention = float(
            np.clip(
                previous_internal * decay
                + max(0.0, internal_factor) * float(params["b38_internal_attention_gain"]),
                -1.0,
                1.0,
            )
        )
        attention_balance = float(
            np.clip(
                (internal_attention - external_attention) * 0.32
                + factor_alignment * 0.42
                + factor_confidence * 0.26,
                -1.0,
                1.0,
            )
        )
        attention_gain = float(
            np.clip(
                previous_gain * decay
                + max(0.0, factor_confidence)
                * float(params["b38_confidence_attention_gain"])
                + max(0.0, attention_balance) * 0.24
                - max(0.0, -attention_balance) * 0.18,
                -1.0,
                1.0,
            )
        )
        attention_score = float(
            np.clip(attention_balance * 0.46 + attention_gain * 0.34 + internal_attention * 0.20, -1.0, 1.0)
        )
        attention_lock = int(getattr(self, "_b38_attention_lock", 0))
        decision_label = "preserve_b37"

        if corridor_map:
            if (
                b37_decision in {"state_factor_commit", "continue_factor_lock"}
                and attention_score >= float(params["b38_attention_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                attention_lock = max(
                    attention_lock,
                    int(params["b38_attention_lock_ticks"]),
                )
                decision_label = "factor_attention_commit"
                reason = "b38_factor_attention_commit"
            elif attention_score <= float(params["b38_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "factor_attention_abort"
                reason = "b38_factor_attention_abort"
            elif attention_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_attention_lock"
                reason = "b38_continue_attention_lock"

        trace_payload.update(
            {
                "b38_controller_profile": profile,
                "b38_external_attention": round(float(external_attention), 6),
                "b38_internal_attention": round(float(internal_attention), 6),
                "b38_attention_balance": round(float(attention_balance), 6),
                "b38_attention_gain": round(float(attention_gain), 6),
                "b38_attention_lock": int(attention_lock),
                "b38_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b38_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b38_genetic_candidate"] = int(params["ga_candidate"])

        self._b38_external_attention = float(external_attention)
        self._b38_internal_attention = float(internal_attention)
        self._b38_attention_gain = float(attention_gain)
        self._b38_attention_lock = max(0, int(attention_lock) - 1)
        self._b38_last_tick = int(tick)
        return (
            semantic_action,
            B38_FACTOR_ATTENTION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b39_controller_params(self) -> dict[str, float]:
        params = self._b38_controller_params()
        defaults = {
            "b39_binding_decay": 0.86,
            "b39_external_binding_gain": 0.32,
            "b39_internal_binding_gain": 0.34,
            "b39_context_binding_gain": 0.36,
            "b39_binding_threshold": 0.12,
            "b39_abort_threshold": -0.18,
            "b39_binding_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "attention_binding"
        )
        if profile == "cross_factor_binding":
            defaults.update(
                {"b39_external_binding_gain": 0.38, "b39_internal_binding_gain": 0.38}
            )
        elif profile == "context_binding_attention":
            defaults.update(
                {"b39_context_binding_gain": 0.44, "b39_binding_threshold": 0.10}
            )
        elif profile == "attention_binding_h56":
            defaults.update({"b39_binding_decay": 0.89, "b39_binding_lock_ticks": 6.0})
        elif profile == "genetic_attention_binding":
            defaults.update(
                {"b39_internal_binding_gain": 0.38, "b39_context_binding_gain": 0.40}
            )
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b39_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b39_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b39_binding_strength = 0.0
        self._b39_bound_context = 0.0
        self._b39_binding_gain = 0.0
        self._b39_binding_lock = 0
        self._b39_last_tick = int(tick)

    def _b39_attention_binding_semantic_action(
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
        ) = self._b38_factor_attention_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b39_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "attention_binding"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b39_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        external_attention = float(trace_payload.get("b38_external_attention", 0.0) or 0.0)
        internal_attention = float(trace_payload.get("b38_internal_attention", 0.0) or 0.0)
        attention_balance = float(trace_payload.get("b38_attention_balance", 0.0) or 0.0)
        attention_gain = float(trace_payload.get("b38_attention_gain", 0.0) or 0.0)
        b38_decision = str(trace_payload.get("b38_decision", "preserve_b37"))

        decay = float(params["b39_binding_decay"])
        previous_strength = float(getattr(self, "_b39_binding_strength", 0.0))
        previous_context = float(getattr(self, "_b39_bound_context", 0.0))
        previous_gain = float(getattr(self, "_b39_binding_gain", 0.0))
        cross_factor_coherence = float(
            np.clip(
                (external_attention + internal_attention) * 0.5
                - abs(external_attention - internal_attention) * 0.24
                + attention_balance * 0.28,
                -1.0,
                1.0,
            )
        )
        binding_strength = float(
            np.clip(
                previous_strength * decay
                + max(0.0, external_attention)
                * float(params["b39_external_binding_gain"])
                + max(0.0, internal_attention)
                * float(params["b39_internal_binding_gain"])
                + max(0.0, cross_factor_coherence) * 0.18,
                -1.0,
                1.0,
            )
        )
        bound_context = float(
            np.clip(
                previous_context * decay
                + binding_strength * float(params["b39_context_binding_gain"])
                + attention_balance * 0.22,
                -1.0,
                1.0,
            )
        )
        binding_gain = float(
            np.clip(
                previous_gain * decay
                + max(0.0, attention_gain) * 0.32
                + max(0.0, bound_context) * 0.28
                - max(0.0, -bound_context) * 0.18,
                -1.0,
                1.0,
            )
        )
        binding_score = float(
            np.clip(
                binding_strength * 0.36
                + cross_factor_coherence * 0.28
                + bound_context * 0.20
                + binding_gain * 0.16,
                -1.0,
                1.0,
            )
        )
        binding_lock = int(getattr(self, "_b39_binding_lock", 0))
        decision_label = "preserve_b38"

        if corridor_map:
            if (
                b38_decision in {"factor_attention_commit", "continue_attention_lock"}
                and binding_score >= float(params["b39_binding_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                binding_lock = max(binding_lock, int(params["b39_binding_lock_ticks"]))
                decision_label = "attention_binding_commit"
                reason = "b39_attention_binding_commit"
            elif binding_score <= float(params["b39_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "attention_binding_abort"
                reason = "b39_attention_binding_abort"
            elif binding_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_binding_lock"
                reason = "b39_continue_binding_lock"

        trace_payload.update(
            {
                "b39_controller_profile": profile,
                "b39_binding_strength": round(float(binding_strength), 6),
                "b39_cross_factor_coherence": round(float(cross_factor_coherence), 6),
                "b39_bound_context": round(float(bound_context), 6),
                "b39_binding_gain": round(float(binding_gain), 6),
                "b39_binding_lock": int(binding_lock),
                "b39_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b39_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b39_genetic_candidate"] = int(params["ga_candidate"])

        self._b39_binding_strength = float(binding_strength)
        self._b39_bound_context = float(bound_context)
        self._b39_binding_gain = float(binding_gain)
        self._b39_binding_lock = max(0, int(binding_lock) - 1)
        self._b39_last_tick = int(tick)
        return (
            semantic_action,
            B39_ATTENTION_BINDING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b40_controller_params(self) -> dict[str, float]:
        params = self._b39_controller_params()
        defaults = {
            "b40_workspace_decay": 0.86,
            "b40_activation_gain": 0.34,
            "b40_broadcast_gain": 0.32,
            "b40_context_gain": 0.36,
            "b40_workspace_threshold": 0.12,
            "b40_abort_threshold": -0.18,
            "b40_workspace_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None) or "global_workspace"
        )
        if profile == "sensory_workspace":
            defaults.update({"b40_activation_gain": 0.40, "b40_broadcast_gain": 0.36})
        elif profile == "context_workspace":
            defaults.update({"b40_context_gain": 0.44, "b40_workspace_threshold": 0.10})
        elif profile == "global_workspace_h56":
            defaults.update({"b40_workspace_decay": 0.89, "b40_workspace_lock_ticks": 6.0})
        elif profile == "genetic_global_workspace":
            defaults.update({"b40_activation_gain": 0.38, "b40_context_gain": 0.40})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b40_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b40_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b40_workspace_activation = 0.0
        self._b40_context_availability = 0.0
        self._b40_workspace_stability = 0.0
        self._b40_workspace_lock = 0
        self._b40_last_tick = int(tick)
