from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8Mixin:
    def _b56_hpa_stress_axis_semantic_action(
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
        ) = self._b55_hypothalamic_drive_coupling_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b56_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "hpa_stress_axis")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b56_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b55_decision = str(trace_payload.get("b55_decision", "preserve_b54"))
        hypothalamic_drive = float(trace_payload.get("b55_hypothalamic_drive", 0.0) or 0.0)
        recovery_bias = float(trace_payload.get("b55_recovery_bias", 0.0) or 0.0)
        drive_balance_input = float(trace_payload.get("b55_drive_balance", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b56_endocrine_decay"])
        previous_cortisol = float(getattr(self, "_b56_cortisol_level", 0.0))
        previous_stress = float(getattr(self, "_b56_stress_load", 0.0))
        previous_recovery = float(getattr(self, "_b56_recovery_signal", 0.0))
        previous_balance = float(getattr(self, "_b56_endocrine_balance", 0.0))
        stress_load = float(
            np.clip(
                previous_stress * decay
                + current_threat * float(params["b56_stress_load_gain"])
                + max(0.0, hypothalamic_drive) * 0.10,
                0.0,
                1.0,
            )
        )
        cortisol_level = float(
            np.clip(
                previous_cortisol * decay
                + stress_load * float(params["b56_cortisol_gain"])
                + max(0.0, current_threat) * 0.08,
                0.0,
                1.0,
            )
        )
        recovery_signal = float(
            np.clip(
                previous_recovery * decay
                + max(0.0, recovery_bias) * float(params["b56_recovery_signal_gain"])
                + max(0.0, 1.0 - current_threat) * 0.04,
                0.0,
                1.0,
            )
        )
        endocrine_balance = float(
            np.clip(
                previous_balance * decay
                + drive_balance_input * 0.34
                + max(0.0, hypothalamic_drive) * float(params["b56_drive_mod_gain"])
                + recovery_signal * 0.08
                - cortisol_level * 0.08
                - stress_load * 0.05,
                -1.0,
                1.0,
            )
        )
        stress_lock = int(getattr(self, "_b56_stress_lock", 0))
        decision_label = "preserve_b55"

        if corridor_map:
            if (
                b55_decision
                in {"hypothalamic_drive_commit", "continue_drive_lock"}
                and endocrine_balance >= float(params["b56_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                stress_lock = max(stress_lock, int(params["b56_stress_lock_ticks"]))
                decision_label = "hpa_stress_axis_commit"
                reason = "b56_hpa_stress_axis_commit"
            elif endocrine_balance <= float(params["b56_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "hpa_stress_axis_abort"
                reason = "b56_hpa_stress_axis_abort"
            elif stress_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_stress_axis_lock"
                reason = "b56_continue_stress_axis_lock"

        trace_payload.update(
            {
                "b56_controller_profile": profile,
                "b56_cortisol_level": round(float(cortisol_level), 6),
                "b56_stress_load": round(float(stress_load), 6),
                "b56_recovery_signal": round(float(recovery_signal), 6),
                "b56_endocrine_balance": round(float(endocrine_balance), 6),
                "b56_stress_lock": int(stress_lock),
                "b56_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b56_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b56_genetic_candidate"] = int(params["ga_candidate"])

        self._b56_cortisol_level = float(cortisol_level)
        self._b56_stress_load = float(stress_load)
        self._b56_recovery_signal = float(recovery_signal)
        self._b56_endocrine_balance = float(endocrine_balance)
        self._b56_stress_lock = max(0, int(stress_lock) - 1)
        self._b56_last_tick = int(tick)
        return (
            semantic_action,
            B56_HPA_STRESS_AXIS_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b57_controller_params(self) -> dict[str, float]:
        params = self._b56_controller_params()
        defaults = {
            "b57_awareness_decay": 0.87,
            "b57_visceral_salience_gain": 0.32,
            "b57_body_confidence_gain": 0.30,
            "b57_stress_awareness_gain": 0.28,
            "b57_drive_awareness_gain": 0.28,
            "b57_commit_threshold": 0.08,
            "b57_abort_threshold": -0.20,
            "b57_awareness_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "insular_interoceptive_awareness"
        )
        if profile == "visceral_salience_gate":
            defaults.update(
                {"b57_visceral_salience_gain": 0.38, "b57_body_confidence_gain": 0.34}
            )
        elif profile == "stress_drive_awareness":
            defaults.update(
                {"b57_stress_awareness_gain": 0.34, "b57_drive_awareness_gain": 0.34}
            )
        elif profile == "insular_interoceptive_awareness_h56":
            defaults.update({"b57_awareness_decay": 0.90, "b57_awareness_lock_ticks": 6.0})
        elif profile == "genetic_interoceptive_awareness":
            defaults.update(
                {"b57_visceral_salience_gain": 0.34, "b57_drive_awareness_gain": 0.32}
            )
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b57_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b57_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b57_interoceptive_awareness = 0.0
        self._b57_visceral_salience = 0.0
        self._b57_body_state_confidence = 0.0
        self._b57_awareness_balance = 0.0
        self._b57_awareness_lock = 0
        self._b57_last_tick = int(tick)

    def _b57_insular_interoceptive_awareness_semantic_action(
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
        ) = self._b56_hpa_stress_axis_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b57_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "insular_interoceptive_awareness"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b57_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b56_decision = str(trace_payload.get("b56_decision", "preserve_b55"))
        cortisol_level = float(trace_payload.get("b56_cortisol_level", 0.0) or 0.0)
        stress_load = float(trace_payload.get("b56_stress_load", 0.0) or 0.0)
        recovery_signal = float(trace_payload.get("b56_recovery_signal", 0.0) or 0.0)
        endocrine_balance = float(trace_payload.get("b56_endocrine_balance", 0.0) or 0.0)
        hypothalamic_drive = float(trace_payload.get("b55_hypothalamic_drive", 0.0) or 0.0)
        hunger = self._b_series_float(meta, "hunger")
        sleep_debt = self._b_series_float(meta, "sleep_debt")
        health_deficit = max(0.0, 1.0 - self._b_series_float(meta, "health"))
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b57_awareness_decay"])
        previous_awareness = float(getattr(self, "_b57_interoceptive_awareness", 0.0))
        previous_salience = float(getattr(self, "_b57_visceral_salience", 0.0))
        previous_confidence = float(getattr(self, "_b57_body_state_confidence", 0.0))
        previous_balance = float(getattr(self, "_b57_awareness_balance", 0.0))
        body_load = max(hunger, sleep_debt, health_deficit)
        visceral_salience = float(
            np.clip(
                previous_salience * decay
                + body_load * float(params["b57_visceral_salience_gain"])
                + max(0.0, hypothalamic_drive) * 0.08,
                0.0,
                1.0,
            )
        )
        body_state_confidence = float(
            np.clip(
                previous_confidence * decay
                + recovery_signal * float(params["b57_body_confidence_gain"])
                + max(0.0, 1.0 - current_threat) * 0.06,
                0.0,
                1.0,
            )
        )
        interoceptive_awareness = float(
            np.clip(
                previous_awareness * decay
                + visceral_salience * 0.24
                + stress_load * float(params["b57_stress_awareness_gain"])
                + max(0.0, endocrine_balance) * float(params["b57_drive_awareness_gain"]),
                0.0,
                1.0,
            )
        )
        awareness_balance = float(
            np.clip(
                previous_balance * decay
                + endocrine_balance * 0.32
                + interoceptive_awareness * 0.10
                + body_state_confidence * 0.10
                - cortisol_level * 0.08
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        awareness_lock = int(getattr(self, "_b57_awareness_lock", 0))
        decision_label = "preserve_b56"

        if corridor_map:
            if (
                b56_decision in {"hpa_stress_axis_commit", "continue_stress_axis_lock"}
                and awareness_balance >= float(params["b57_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                awareness_lock = max(awareness_lock, int(params["b57_awareness_lock_ticks"]))
                decision_label = "interoceptive_awareness_commit"
                reason = "b57_interoceptive_awareness_commit"
            elif awareness_balance <= float(params["b57_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "interoceptive_awareness_abort"
                reason = "b57_interoceptive_awareness_abort"
            elif awareness_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_awareness_lock"
                reason = "b57_continue_awareness_lock"

        trace_payload.update(
            {
                "b57_controller_profile": profile,
                "b57_interoceptive_awareness": round(float(interoceptive_awareness), 6),
                "b57_visceral_salience": round(float(visceral_salience), 6),
                "b57_body_state_confidence": round(float(body_state_confidence), 6),
                "b57_awareness_balance": round(float(awareness_balance), 6),
                "b57_awareness_lock": int(awareness_lock),
                "b57_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b57_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b57_genetic_candidate"] = int(params["ga_candidate"])

        self._b57_interoceptive_awareness = float(interoceptive_awareness)
        self._b57_visceral_salience = float(visceral_salience)
        self._b57_body_state_confidence = float(body_state_confidence)
        self._b57_awareness_balance = float(awareness_balance)
        self._b57_awareness_lock = max(0, int(awareness_lock) - 1)
        self._b57_last_tick = int(tick)
        return (
            semantic_action,
            B57_INSULAR_INTEROCEPTIVE_AWARENESS_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b58_controller_params(self) -> dict[str, float]:
        params = self._b57_controller_params()
        defaults = {
            "b58_conflict_decay": 0.87,
            "b58_conflict_gain": 0.32,
            "b58_error_likelihood_gain": 0.30,
            "b58_control_allocation_gain": 0.30,
            "b58_awareness_mod_gain": 0.28,
            "b58_commit_threshold": 0.08,
            "b58_abort_threshold": -0.20,
            "b58_conflict_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "acc_conflict_monitor")
        if profile == "error_salience_gate":
            defaults.update({"b58_error_likelihood_gain": 0.36, "b58_conflict_gain": 0.34})
        elif profile == "conflict_resolution_balance":
            defaults.update(
                {"b58_control_allocation_gain": 0.36, "b58_awareness_mod_gain": 0.32}
            )
        elif profile == "acc_conflict_monitor_h56":
            defaults.update({"b58_conflict_decay": 0.90, "b58_conflict_lock_ticks": 6.0})
        elif profile == "genetic_acc_conflict":
            defaults.update({"b58_conflict_gain": 0.34, "b58_control_allocation_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b58_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b58_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b58_conflict_signal = 0.0
        self._b58_error_likelihood = 0.0
        self._b58_control_allocation = 0.0
        self._b58_resolution_balance = 0.0
        self._b58_conflict_lock = 0
        self._b58_last_tick = int(tick)

    def _b58_acc_conflict_monitor_semantic_action(
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
        ) = self._b57_insular_interoceptive_awareness_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b58_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "acc_conflict_monitor")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b58_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b57_decision = str(trace_payload.get("b57_decision", "preserve_b56"))
        awareness_balance = float(trace_payload.get("b57_awareness_balance", 0.0) or 0.0)
        interoceptive_awareness = float(
            trace_payload.get("b57_interoceptive_awareness", 0.0) or 0.0
        )
        visceral_salience = float(trace_payload.get("b57_visceral_salience", 0.0) or 0.0)
        body_confidence = float(trace_payload.get("b57_body_state_confidence", 0.0) or 0.0)
        cortisol_level = float(trace_payload.get("b56_cortisol_level", 0.0) or 0.0)
        stress_load = float(trace_payload.get("b56_stress_load", 0.0) or 0.0)
        food_dist = self._b_series_float(meta, "food_dist")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        route_conflict = float(np.clip((shelter_dist + 1.0) / (food_dist + 1.0), 0.0, 1.0))
        if food_dist <= 0.0:
            route_conflict = 0.0

        decay = float(params["b58_conflict_decay"])
        previous_conflict = float(getattr(self, "_b58_conflict_signal", 0.0))
        previous_error = float(getattr(self, "_b58_error_likelihood", 0.0))
        previous_control = float(getattr(self, "_b58_control_allocation", 0.0))
        previous_balance = float(getattr(self, "_b58_resolution_balance", 0.0))
        conflict_input = abs(awareness_balance) * 0.35 + current_threat * 0.25 + route_conflict * 0.20
        conflict_signal = float(
            np.clip(
                previous_conflict * decay
                + conflict_input * float(params["b58_conflict_gain"])
                + visceral_salience * 0.06,
                0.0,
                1.0,
            )
        )
        error_likelihood = float(
            np.clip(
                previous_error * decay
                + (current_threat + cortisol_level + stress_load) / 3.0
                * float(params["b58_error_likelihood_gain"])
                + route_conflict * 0.05,
                0.0,
                1.0,
            )
        )
        control_allocation = float(
            np.clip(
                previous_control * decay
                + body_confidence * float(params["b58_control_allocation_gain"])
                + interoceptive_awareness * float(params["b58_awareness_mod_gain"])
                + max(0.0, awareness_balance) * 0.08,
                0.0,
                1.0,
            )
        )
        resolution_balance = float(
            np.clip(
                previous_balance * decay
                + awareness_balance * 0.32
                + control_allocation * 0.14
                - error_likelihood * 0.08
                - conflict_signal * 0.06,
                -1.0,
                1.0,
            )
        )
        conflict_lock = int(getattr(self, "_b58_conflict_lock", 0))
        decision_label = "preserve_b57"

        if corridor_map:
            if (
                b57_decision
                in {"interoceptive_awareness_commit", "continue_awareness_lock"}
                and resolution_balance >= float(params["b58_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                conflict_lock = max(conflict_lock, int(params["b58_conflict_lock_ticks"]))
                decision_label = "acc_conflict_commit"
                reason = "b58_acc_conflict_commit"
            elif resolution_balance <= float(params["b58_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "acc_conflict_abort"
                reason = "b58_acc_conflict_abort"
            elif conflict_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_conflict_lock"
                reason = "b58_continue_conflict_lock"

        trace_payload.update(
            {
                "b58_controller_profile": profile,
                "b58_conflict_signal": round(float(conflict_signal), 6),
                "b58_error_likelihood": round(float(error_likelihood), 6),
                "b58_control_allocation": round(float(control_allocation), 6),
                "b58_resolution_balance": round(float(resolution_balance), 6),
                "b58_conflict_lock": int(conflict_lock),
                "b58_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b58_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b58_genetic_candidate"] = int(params["ga_candidate"])

        self._b58_conflict_signal = float(conflict_signal)
        self._b58_error_likelihood = float(error_likelihood)
        self._b58_control_allocation = float(control_allocation)
        self._b58_resolution_balance = float(resolution_balance)
        self._b58_conflict_lock = max(0, int(conflict_lock) - 1)
        self._b58_last_tick = int(tick)
        return (
            semantic_action,
            B58_ACC_CONFLICT_MONITOR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b59_controller_params(self) -> dict[str, float]:
        params = self._b58_controller_params()
        defaults = {
            "b59_context_decay": 0.87,
            "b59_goal_context_gain": 0.32,
            "b59_working_set_gain": 0.30,
            "b59_task_confidence_gain": 0.30,
            "b59_control_mod_gain": 0.28,
            "b59_commit_threshold": 0.08,
            "b59_abort_threshold": -0.20,
            "b59_executive_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "prefrontal_goal_context")
        if profile == "working_set_stability":
            defaults.update({"b59_working_set_gain": 0.36, "b59_task_confidence_gain": 0.34})
        elif profile == "executive_task_set":
            defaults.update({"b59_goal_context_gain": 0.34, "b59_control_mod_gain": 0.34})
        elif profile == "prefrontal_goal_context_h56":
            defaults.update({"b59_context_decay": 0.90, "b59_executive_lock_ticks": 6.0})
        elif profile == "genetic_prefrontal_control":
            defaults.update({"b59_goal_context_gain": 0.34, "b59_working_set_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b59_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b59_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b59_goal_context = 0.0
        self._b59_working_set_stability = 0.0
        self._b59_task_set_confidence = 0.0
        self._b59_executive_balance = 0.0
        self._b59_executive_lock = 0
        self._b59_last_tick = int(tick)

    def _b59_prefrontal_goal_context_semantic_action(
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
        ) = self._b58_acc_conflict_monitor_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b59_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "prefrontal_goal_context")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b59_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b58_decision = str(trace_payload.get("b58_decision", "preserve_b57"))
        resolution_balance = float(trace_payload.get("b58_resolution_balance", 0.0) or 0.0)
        control_allocation = float(trace_payload.get("b58_control_allocation", 0.0) or 0.0)
        conflict_signal = float(trace_payload.get("b58_conflict_signal", 0.0) or 0.0)
        error_likelihood = float(trace_payload.get("b58_error_likelihood", 0.0) or 0.0)
        awareness_balance = float(trace_payload.get("b57_awareness_balance", 0.0) or 0.0)
        body_confidence = float(trace_payload.get("b57_body_state_confidence", 0.0) or 0.0)
        food_dist = self._b_series_float(meta, "food_dist")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        corridor_goal = 1.0 if corridor_map and food_dist >= shelter_dist else 0.0

        decay = float(params["b59_context_decay"])
        previous_goal = float(getattr(self, "_b59_goal_context", 0.0))
        previous_stability = float(getattr(self, "_b59_working_set_stability", 0.0))
        previous_confidence = float(getattr(self, "_b59_task_set_confidence", 0.0))
        previous_balance = float(getattr(self, "_b59_executive_balance", 0.0))
        goal_context = float(
            np.clip(
                previous_goal * decay
                + corridor_goal * float(params["b59_goal_context_gain"])
                + max(0.0, resolution_balance) * 0.10,
                0.0,
                1.0,
            )
        )
        working_set_stability = float(
            np.clip(
                previous_stability * decay
                + control_allocation * float(params["b59_working_set_gain"])
                + body_confidence * 0.06,
                0.0,
                1.0,
            )
        )
        task_set_confidence = float(
            np.clip(
                previous_confidence * decay
                + max(0.0, awareness_balance) * float(params["b59_task_confidence_gain"])
                + max(0.0, 1.0 - current_threat) * 0.06,
                0.0,
                1.0,
            )
        )
        executive_balance = float(
            np.clip(
                previous_balance * decay
                + resolution_balance * 0.32
                + goal_context * 0.10
                + working_set_stability * 0.10
                + task_set_confidence * float(params["b59_control_mod_gain"])
                - conflict_signal * 0.05
                - error_likelihood * 0.07,
                -1.0,
                1.0,
            )
        )
        executive_lock = int(getattr(self, "_b59_executive_lock", 0))
        decision_label = "preserve_b58"

        if corridor_map:
            if (
                b58_decision in {"acc_conflict_commit", "continue_conflict_lock"}
                and executive_balance >= float(params["b59_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                executive_lock = max(executive_lock, int(params["b59_executive_lock_ticks"]))
                decision_label = "prefrontal_goal_commit"
                reason = "b59_prefrontal_goal_commit"
            elif executive_balance <= float(params["b59_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "prefrontal_goal_abort"
                reason = "b59_prefrontal_goal_abort"
            elif executive_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_executive_lock"
                reason = "b59_continue_executive_lock"

        trace_payload.update(
            {
                "b59_controller_profile": profile,
                "b59_goal_context": round(float(goal_context), 6),
                "b59_working_set_stability": round(float(working_set_stability), 6),
                "b59_task_set_confidence": round(float(task_set_confidence), 6),
                "b59_executive_balance": round(float(executive_balance), 6),
                "b59_executive_lock": int(executive_lock),
                "b59_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b59_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b59_genetic_candidate"] = int(params["ga_candidate"])

        self._b59_goal_context = float(goal_context)
        self._b59_working_set_stability = float(working_set_stability)
        self._b59_task_set_confidence = float(task_set_confidence)
        self._b59_executive_balance = float(executive_balance)
        self._b59_executive_lock = max(0, int(executive_lock) - 1)
        self._b59_last_tick = int(tick)
        return (
            semantic_action,
            B59_PREFRONTAL_GOAL_CONTEXT_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b60_controller_params(self) -> dict[str, float]:
        params = self._b59_controller_params()
        defaults = {
            "b60_value_decay": 0.88,
            "b60_outcome_value_gain": 0.32,
            "b60_reversal_signal_gain": 0.28,
            "b60_goal_value_confidence_gain": 0.30,
            "b60_prefrontal_mod_gain": 0.26,
            "b60_commit_threshold": 0.07,
            "b60_reversal_threshold": 0.24,
            "b60_value_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "orbitofrontal_outcome_value"
        )
        if profile == "reversal_value_gate":
            defaults.update({"b60_reversal_signal_gain": 0.34, "b60_reversal_threshold": 0.20})
        elif profile == "goal_outcome_prediction":
            defaults.update(
                {
                    "b60_outcome_value_gain": 0.36,
                    "b60_goal_value_confidence_gain": 0.34,
                }
            )
        elif profile == "orbitofrontal_outcome_value_h56":
            defaults.update({"b60_value_decay": 0.90, "b60_value_lock_ticks": 6.0})
        elif profile == "genetic_orbitofrontal_value":
            defaults.update({"b60_outcome_value_gain": 0.34, "b60_prefrontal_mod_gain": 0.30})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b60_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b60_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b60_outcome_value = 0.0
        self._b60_reversal_signal = 0.0
        self._b60_goal_value_confidence = 0.0
        self._b60_value_balance = 0.0
        self._b60_value_lock = 0
        self._b60_last_tick = int(tick)

    def _b60_orbitofrontal_outcome_value_semantic_action(
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
        ) = self._b59_prefrontal_goal_context_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b60_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "orbitofrontal_outcome_value"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b60_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b59_decision = str(trace_payload.get("b59_decision", "preserve_b58"))
        executive_balance = float(trace_payload.get("b59_executive_balance", 0.0) or 0.0)
        goal_context = float(trace_payload.get("b59_goal_context", 0.0) or 0.0)
        working_set = float(trace_payload.get("b59_working_set_stability", 0.0) or 0.0)
        task_confidence = float(trace_payload.get("b59_task_set_confidence", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        hunger = self._b_series_float(meta, "hunger")
        health = self._b_series_float(meta, "health")
        food_dist = self._b_series_float(meta, "food_dist")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        progress_value = 1.0 if food_dist >= shelter_dist else 0.0
        homeostatic_margin = float(np.clip((1.0 - hunger) * 0.45 + health * 0.55, 0.0, 1.0))
        threat_penalty = float(np.clip(current_threat + max(0.0, shelter_dist - food_dist) * 0.02, 0.0, 1.0))

        decay = float(params["b60_value_decay"])
        previous_value = float(getattr(self, "_b60_outcome_value", 0.0))
        previous_reversal = float(getattr(self, "_b60_reversal_signal", 0.0))
        previous_confidence = float(getattr(self, "_b60_goal_value_confidence", 0.0))
        previous_balance = float(getattr(self, "_b60_value_balance", 0.0))
        outcome_value = float(
            np.clip(
                previous_value * decay
                + executive_balance * float(params["b60_outcome_value_gain"])
                + progress_value * 0.08
                + homeostatic_margin * 0.06,
                -1.0,
                1.0,
            )
        )
        reversal_signal = float(
            np.clip(
                previous_reversal * decay
                + threat_penalty * float(params["b60_reversal_signal_gain"])
                + max(0.0, -executive_balance) * 0.12,
                0.0,
                1.0,
            )
        )
        goal_value_confidence = float(
            np.clip(
                previous_confidence * decay
                + task_confidence * float(params["b60_goal_value_confidence_gain"])
                + working_set * 0.08
                + goal_context * 0.07,
                0.0,
                1.0,
            )
        )
        value_balance = float(
            np.clip(
                previous_balance * decay
                + outcome_value * 0.36
                + goal_value_confidence * float(params["b60_prefrontal_mod_gain"])
                - reversal_signal * 0.30,
                -1.0,
                1.0,
            )
        )
        value_lock = int(getattr(self, "_b60_value_lock", 0))
        decision_label = "preserve_b59"

        if corridor_map:
            if (
                b59_decision in {"prefrontal_goal_commit", "continue_executive_lock"}
                and value_balance >= float(params["b60_commit_threshold"])
                and reversal_signal < float(params["b60_reversal_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                value_lock = max(value_lock, int(params["b60_value_lock_ticks"]))
                decision_label = "orbitofrontal_value_commit"
                reason = "b60_orbitofrontal_value_commit"
            elif reversal_signal >= float(params["b60_reversal_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "orbitofrontal_value_reversal"
                reason = "b60_orbitofrontal_value_reversal"
            elif value_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_value_lock"
                reason = "b60_continue_value_lock"

        trace_payload.update(
            {
                "b60_controller_profile": profile,
                "b60_outcome_value": round(float(outcome_value), 6),
                "b60_reversal_signal": round(float(reversal_signal), 6),
                "b60_goal_value_confidence": round(float(goal_value_confidence), 6),
                "b60_value_balance": round(float(value_balance), 6),
                "b60_value_lock": int(value_lock),
                "b60_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b60_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b60_genetic_candidate"] = int(params["ga_candidate"])

        self._b60_outcome_value = float(outcome_value)
        self._b60_reversal_signal = float(reversal_signal)
        self._b60_goal_value_confidence = float(goal_value_confidence)
        self._b60_value_balance = float(value_balance)
        self._b60_value_lock = max(0, int(value_lock) - 1)
        self._b60_last_tick = int(tick)
        return (
            semantic_action,
            B60_ORBITOFRONTAL_OUTCOME_VALUE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b61_controller_params(self) -> dict[str, float]:
        params = self._b60_controller_params()
        defaults = {
            "b61_affect_decay": 0.88,
            "b61_safety_value_gain": 0.32,
            "b61_threat_value_gain": 0.28,
            "b61_safety_confidence_gain": 0.30,
            "b61_ofc_mod_gain": 0.26,
            "b61_commit_threshold": 0.07,
            "b61_threat_threshold": 0.26,
            "b61_safety_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "amygdala_safety_value")
        if profile == "threat_value_tag":
            defaults.update({"b61_threat_value_gain": 0.34, "b61_threat_threshold": 0.22})
        elif profile == "safety_prediction_gate":
            defaults.update(
                {"b61_safety_value_gain": 0.36, "b61_safety_confidence_gain": 0.34}
            )
        elif profile == "amygdala_safety_value_h56":
            defaults.update({"b61_affect_decay": 0.90, "b61_safety_lock_ticks": 6.0})
        elif profile == "genetic_amygdala_safety":
            defaults.update({"b61_safety_value_gain": 0.34, "b61_ofc_mod_gain": 0.30})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b61_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b61_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b61_safety_value = 0.0
        self._b61_threat_value = 0.0
        self._b61_safety_confidence = 0.0
        self._b61_affective_balance = 0.0
        self._b61_safety_lock = 0
        self._b61_last_tick = int(tick)

    def _b61_amygdala_safety_value_semantic_action(
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
        ) = self._b60_orbitofrontal_outcome_value_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b61_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "amygdala_safety_value")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b61_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b60_decision = str(trace_payload.get("b60_decision", "preserve_b59"))
        outcome_value = float(trace_payload.get("b60_outcome_value", 0.0) or 0.0)
        reversal_signal = float(trace_payload.get("b60_reversal_signal", 0.0) or 0.0)
        goal_value_confidence = float(trace_payload.get("b60_goal_value_confidence", 0.0) or 0.0)
        value_balance = float(trace_payload.get("b60_value_balance", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        hunger = self._b_series_float(meta, "hunger")
        health = self._b_series_float(meta, "health")
        food_dist = self._b_series_float(meta, "food_dist")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        corridor_safety_cue = float(
            np.clip((1.0 - current_threat) * 0.65 + max(0.0, food_dist - shelter_dist) * 0.04, 0.0, 1.0)
        )
        vulnerability = float(np.clip(hunger * 0.35 + (1.0 - health) * 0.45 + reversal_signal * 0.20, 0.0, 1.0))

        decay = float(params["b61_affect_decay"])
        previous_safety = float(getattr(self, "_b61_safety_value", 0.0))
        previous_threat = float(getattr(self, "_b61_threat_value", 0.0))
        previous_confidence = float(getattr(self, "_b61_safety_confidence", 0.0))
        previous_balance = float(getattr(self, "_b61_affective_balance", 0.0))
        safety_value = float(
            np.clip(
                previous_safety * decay
                + max(0.0, outcome_value) * float(params["b61_safety_value_gain"])
                + corridor_safety_cue * 0.08,
                0.0,
                1.0,
            )
        )
        threat_value = float(
            np.clip(
                previous_threat * decay
                + current_threat * float(params["b61_threat_value_gain"])
                + vulnerability * 0.03,
                0.0,
                1.0,
            )
        )
        safety_confidence = float(
            np.clip(
                previous_confidence * decay
                + goal_value_confidence * float(params["b61_safety_confidence_gain"])
                + max(0.0, value_balance) * 0.08,
                0.0,
                1.0,
            )
        )
        affective_balance = float(
            np.clip(
                previous_balance * decay
                + safety_value * 0.32
                + safety_confidence * float(params["b61_ofc_mod_gain"])
                - threat_value * 0.30,
                -1.0,
                1.0,
            )
        )
        safety_lock = int(getattr(self, "_b61_safety_lock", 0))
        decision_label = "preserve_b60"

        if corridor_map:
            if (
                b60_decision in {"orbitofrontal_value_commit", "continue_value_lock"}
                and affective_balance >= float(params["b61_commit_threshold"])
                and threat_value < float(params["b61_threat_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                safety_lock = max(safety_lock, int(params["b61_safety_lock_ticks"]))
                decision_label = "amygdala_safety_commit"
                reason = "b61_amygdala_safety_commit"
            elif threat_value >= float(params["b61_threat_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "amygdala_threat_reversal"
                reason = "b61_amygdala_threat_reversal"
            elif safety_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_safety_lock"
                reason = "b61_continue_safety_lock"

        trace_payload.update(
            {
                "b61_controller_profile": profile,
                "b61_safety_value": round(float(safety_value), 6),
                "b61_threat_value": round(float(threat_value), 6),
                "b61_safety_confidence": round(float(safety_confidence), 6),
                "b61_affective_balance": round(float(affective_balance), 6),
                "b61_safety_lock": int(safety_lock),
                "b61_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b61_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b61_genetic_candidate"] = int(params["ga_candidate"])

        self._b61_safety_value = float(safety_value)
        self._b61_threat_value = float(threat_value)
        self._b61_safety_confidence = float(safety_confidence)
        self._b61_affective_balance = float(affective_balance)
        self._b61_safety_lock = max(0, int(safety_lock) - 1)
        self._b61_last_tick = int(tick)
        return (
            semantic_action,
            B61_AMYGDALA_SAFETY_VALUE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b62_controller_params(self) -> dict[str, float]:
        params = self._b61_controller_params()
        defaults = {
            "b62_defense_decay": 0.88,
            "b62_freeze_gain": 0.30,
            "b62_flee_gain": 0.34,
            "b62_shelter_bias_gain": 0.28,
            "b62_balance_gain": 0.26,
            "b62_freeze_threshold": 0.30,
            "b62_flee_threshold": 0.23,
            "b62_defense_lock_ticks": 4.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "defensive_mode_selector")
        if profile == "freeze_flee_balance":
            defaults.update({"b62_freeze_gain": 0.34, "b62_flee_gain": 0.36})
        elif profile == "shelter_defense_gate":
            defaults.update({"b62_shelter_bias_gain": 0.34, "b62_flee_threshold": 0.20})
        elif profile == "defensive_mode_selector_h56":
            defaults.update({"b62_defense_decay": 0.90, "b62_defense_lock_ticks": 5.0})
        elif profile == "genetic_defensive_mode":
            defaults.update({"b62_flee_gain": 0.36, "b62_balance_gain": 0.30})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b62_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b62_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b62_freeze_pressure = 0.0
        self._b62_flee_pressure = 0.0
        self._b62_shelter_bias = 0.0
        self._b62_defense_balance = 0.0
        self._b62_defense_lock = 0
        self._b62_last_tick = int(tick)

    def _b62_defensive_mode_selector_semantic_action(
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
        ) = self._b61_amygdala_safety_value_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b62_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "defensive_mode_selector")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b62_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        shelter_role = str(meta.get("shelter_role", "outside"))
        b61_decision = str(trace_payload.get("b61_decision", "preserve_b60"))
        safety_value = float(trace_payload.get("b61_safety_value", 0.0) or 0.0)
        threat_value = float(trace_payload.get("b61_threat_value", 0.0) or 0.0)
        safety_confidence = float(trace_payload.get("b61_safety_confidence", 0.0) or 0.0)
        affective_balance = float(trace_payload.get("b61_affective_balance", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        hunger = self._b_series_float(meta, "hunger")
        health = self._b_series_float(meta, "health")
        sleep_debt = self._b_series_float(meta, "sleep_debt")
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        food_dist = self._b_series_float(meta, "food_dist")
        near_shelter = 1.0 if shelter_dist <= 2.0 or shelter_role in {"at_shelter", "deep_shelter"} else 0.0
        homeostatic_vulnerability = float(
            np.clip((1.0 - health) * 0.45 + sleep_debt * 0.25 + hunger * 0.20, 0.0, 1.0)
        )
        shelter_advantage = float(np.clip((food_dist - shelter_dist) * 0.05 + near_shelter * 0.18, 0.0, 1.0))

        decay = float(params["b62_defense_decay"])
        previous_freeze = float(getattr(self, "_b62_freeze_pressure", 0.0))
        previous_flee = float(getattr(self, "_b62_flee_pressure", 0.0))
        previous_shelter = float(getattr(self, "_b62_shelter_bias", 0.0))
        previous_balance = float(getattr(self, "_b62_defense_balance", 0.0))
        freeze_pressure = float(
            np.clip(
                previous_freeze * decay
                + (current_threat + threat_value) * 0.5 * float(params["b62_freeze_gain"])
                + near_shelter * 0.06,
                0.0,
                1.0,
            )
        )
        flee_pressure = float(
            np.clip(
                previous_flee * decay
                + (threat_value + homeostatic_vulnerability) * float(params["b62_flee_gain"])
                + max(0.0, -affective_balance) * 0.08,
                0.0,
                1.0,
            )
        )
        shelter_bias = float(
            np.clip(
                previous_shelter * decay
                + shelter_advantage * float(params["b62_shelter_bias_gain"])
                + threat_value * 0.08,
                0.0,
                1.0,
            )
        )
        defense_balance = float(
            np.clip(
                previous_balance * decay
                + flee_pressure * float(params["b62_balance_gain"])
                + shelter_bias * 0.24
                - safety_value * 0.16
                - safety_confidence * 0.10,
                -1.0,
                1.0,
            )
        )
        defense_lock = int(getattr(self, "_b62_defense_lock", 0))
        decision_label = "preserve_b61"
        defensive_mode = "preserve"

        if corridor_map:
            if (
                flee_pressure >= float(params["b62_flee_threshold"])
                and defense_balance > 0.0
                and hunger < 0.96
            ):
                semantic_action = "MOVE_TO_SHELTER"
                defense_lock = max(defense_lock, int(params["b62_defense_lock_ticks"]))
                defensive_mode = "flee_to_shelter"
                decision_label = "defensive_flee_to_shelter"
                reason = "b62_defensive_flee_to_shelter"
            elif (
                freeze_pressure >= float(params["b62_freeze_threshold"])
                and near_shelter > 0.0
                and hunger < 0.92
            ):
                semantic_action = "STAY"
                defense_lock = max(defense_lock, int(params["b62_defense_lock_ticks"]))
                defensive_mode = "freeze_hold"
                decision_label = "defensive_freeze_hold"
                reason = "b62_defensive_freeze_hold"
            elif defense_lock > 0 and hunger < 0.96:
                semantic_action = "MOVE_TO_SHELTER"
                defensive_mode = "continue_defense_lock"
                decision_label = "continue_defense_lock"
                reason = "b62_continue_defense_lock"
            elif (
                b61_decision in {"amygdala_safety_commit", "continue_safety_lock"}
                and safety_value >= max(threat_value, 0.0)
                and safety_confidence > 0.0
            ):
                semantic_action = "MOVE_TO_FOOD"
                defensive_mode = "safe_advance"
                decision_label = "defensive_safe_advance"
                reason = "b62_defensive_safe_advance"

        trace_payload.update(
            {
                "b62_controller_profile": profile,
                "b62_defensive_mode": defensive_mode,
                "b62_freeze_pressure": round(float(freeze_pressure), 6),
                "b62_flee_pressure": round(float(flee_pressure), 6),
                "b62_shelter_bias": round(float(shelter_bias), 6),
                "b62_defense_balance": round(float(defense_balance), 6),
                "b62_defense_lock": int(defense_lock),
                "b62_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b62_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b62_genetic_candidate"] = int(params["ga_candidate"])

        self._b62_freeze_pressure = float(freeze_pressure)
        self._b62_flee_pressure = float(flee_pressure)
        self._b62_shelter_bias = float(shelter_bias)
        self._b62_defense_balance = float(defense_balance)
        self._b62_defense_lock = max(0, int(defense_lock) - 1)
        self._b62_last_tick = int(tick)
        return (
            semantic_action,
            B62_DEFENSIVE_MODE_SELECTOR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    @staticmethod
    def _network_forward_macs(network: object) -> int:
        if isinstance(network, RecurrentProposalNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
            )
        if isinstance(network, RecurrentEventAttentionTrueMonolithicNetwork):
            recurrent_input_dim = (
                network.input_dim + network.event_context_dim
            )
            event_raw_dim = (
                network.event_embedding_dim + network.event_feature_dim + 1
            )
            return int(
                recurrent_input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.event_context_dim * (network.input_dim + network.hidden_dim)
                + 2 * (network.event_context_dim * event_raw_dim)
            )
        if isinstance(network, RecurrentOptionAffordanceTrueMonolithicNetwork):
            recurrent_input_dim = (
                network.input_dim + network.event_context_dim + network.option_dim
            )
            event_raw_dim = (
                network.event_embedding_dim + network.event_feature_dim + 1
            )
            return int(
                recurrent_input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.hidden_dim * network.option_dim
                + network.option_dim * network.output_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.hidden_dim * network.output_dim * network.affordance_role_dim
                + network.output_dim * network.affordance_role_dim
                + network.event_context_dim * (network.input_dim + network.hidden_dim)
                + 2 * (network.event_context_dim * event_raw_dim)
            )
        if isinstance(network, RecurrentOptionTrueMonolithicNetwork):
            recurrent_input_dim = (
                network.input_dim + network.event_context_dim + network.option_dim
            )
            event_raw_dim = (
                network.event_embedding_dim + network.event_feature_dim + 1
            )
            return int(
                recurrent_input_dim * network.hidden_dim
                + network.hidden_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
                + network.hidden_dim * network.option_dim
                + network.option_dim * network.output_dim
                + network.event_context_dim * (network.input_dim + network.hidden_dim)
                + 2 * (network.event_context_dim * event_raw_dim)
            )
        if isinstance(network, ArbitrationNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.valence_dim
                + network.hidden_dim * network.gate_dim
                + network.hidden_dim
            )
        if isinstance(network, MotorNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
                + network.hidden_dim
            )
        if isinstance(network, ProposalNetwork):
            return int(
                network.input_dim * network.hidden_dim
                + network.hidden_dim * network.output_dim
            )
        return 0

    def estimate_compute_cost(self) -> Dict[str, object]:
        per_network: Dict[str, int] = {}
        if self.module_bank is not None:
            per_network.update(
                {
                    name: self._network_forward_macs(network)
                    for name, network in self.module_bank.modules.items()
                }
            )
        if self.monolithic_policy is not None:
            per_network[self.MONOLITHIC_POLICY_NAME] = self._network_forward_macs(
                self.monolithic_policy
            )
        if self.true_monolithic_policy is not None:
            per_network[self.TRUE_MONOLITHIC_POLICY_NAME] = self._network_forward_macs(
                self.true_monolithic_policy
            )
        if getattr(self, "b_series_policy", None) is not None:
            per_network[B_SERIES_POLICY_NAME] = self._network_forward_macs(
                self.b_series_policy
            )
        if self.arbitration_network is not None:
            per_network[self.ARBITRATION_NETWORK_NAME] = self._network_forward_macs(
                self.arbitration_network
            )
        if self.action_center is not None:
            per_network["action_center"] = self._network_forward_macs(
                self.action_center
            )
        if self.motor_cortex is not None:
            per_network["motor_cortex"] = self._network_forward_macs(
                self.motor_cortex
            )
        return {
            "unit": "approx_forward_macs",
            "per_network": per_network,
            "total": int(sum(per_network.values())),
        }

    def _true_monolithic_arbitration_decision(
        self,
        *,
        module_name: str,
        action_idx: int,
    ) -> ArbitrationDecision:
        """Build a trivial arbitration payload for the direct-control baseline."""
        return ArbitrationDecision(
            strategy="direct_control",
            winning_valence="exploration",
            valence_scores={
                "threat": 0.0,
                "hunger": 0.0,
                "sleep": 0.0,
                "exploration": 1.0,
            },
            module_gates={module_name: 1.0},
            suppressed_modules=[],
            evidence={},
            intent_before_gating_idx=int(action_idx),
            intent_after_gating_idx=int(action_idx),
            valence_logits={
                "threat": 0.0,
                "hunger": 0.0,
                "sleep": 0.0,
                "exploration": 1.0,
            },
            base_gates={module_name: 1.0},
            gate_adjustments={module_name: 1.0},
            arbitration_value=0.0,
            learned_adjustment=False,
            module_contribution_share={module_name: 1.0},
            dominant_module=module_name,
            dominant_module_share=1.0,
            effective_module_count=1.0,
            gate_entropy=0.0,
            dominance_rate=1.0,
            effective_proposer_count=1.0,
            module_counts={module_name: 1},
            module_agreement_rate=1.0,
            module_disagreement_rate=0.0,
        )

    def _food_direction_bias_action(
        self,
        observation: Dict[str, np.ndarray],
    ) -> str | None:
        """
        Return a locomotion action that follows the strongest available food cue.

        Preference order matches the modular inference path: visible food,
        then food trace, then fresh food memory, then smell.
        """
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
        action = direction_action(food_dx, food_dy)
        if action == "STAY":
            return None
        return action
