from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart7Mixin:
    def _b48_cerebellar_timing_semantic_action(
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
        ) = self._b47_oscillatory_synchrony_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b48_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cerebellar_timing"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b48_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        phase_alignment = float(trace_payload.get("b47_phase_alignment", 0.0) or 0.0)
        synchrony_gain = float(trace_payload.get("b47_synchrony_gain", 0.0) or 0.0)
        cross_loop_coherence = float(
            trace_payload.get("b47_cross_loop_coherence", 0.0) or 0.0
        )
        phase_lock = float(trace_payload.get("b47_phase_lock", 0.0) or 0.0)
        b47_decision = str(trace_payload.get("b47_decision", "preserve_b46"))

        decay = float(params["b48_timing_decay"])
        previous_error = float(getattr(self, "_b48_timing_error", 0.0))
        previous_prediction = float(getattr(self, "_b48_predictive_timing", 0.0))
        previous_correction = float(getattr(self, "_b48_corrective_gain", 0.0))
        target_phase = 0.55 + 0.08 * float(np.sin(max(0, tick) * 0.23))
        raw_error = abs(target_phase - max(0.0, phase_alignment))
        timing_error = float(
            np.clip(
                previous_error * decay
                + (1.0 - min(1.0, raw_error)) * float(params["b48_error_gain"])
                + max(0.0, cross_loop_coherence) * 0.18,
                -1.0,
                1.0,
            )
        )
        predictive_timing = float(
            np.clip(
                previous_prediction * decay
                + max(0.0, synchrony_gain) * float(params["b48_prediction_gain"])
                + max(0.0, phase_lock) * 0.04
                + max(0.0, timing_error) * 0.18,
                -1.0,
                1.0,
            )
        )
        corrective_gain = float(
            np.clip(
                previous_correction * decay
                + max(0.0, predictive_timing) * float(params["b48_corrective_gain"])
                + max(0.0, cross_loop_coherence) * 0.22
                + max(0.0, timing_error) * 0.16,
                -1.0,
                1.0,
            )
        )
        timing_score = float(
            np.clip(
                timing_error * 0.24
                + predictive_timing * 0.32
                + corrective_gain * 0.32
                + cross_loop_coherence * 0.12,
                -1.0,
                1.0,
            )
        )
        calibration_lock = int(getattr(self, "_b48_calibration_lock", 0))
        decision_label = "preserve_b47"

        if corridor_map:
            if (
                b47_decision
                in {"oscillatory_synchrony_commit", "continue_phase_lock"}
                and timing_score >= float(params["b48_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                calibration_lock = max(
                    calibration_lock,
                    int(params["b48_calibration_lock_ticks"]),
                )
                decision_label = "cerebellar_timing_commit"
                reason = "b48_cerebellar_timing_commit"
            elif timing_score <= float(params["b48_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "cerebellar_timing_abort"
                reason = "b48_cerebellar_timing_abort"
            elif calibration_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_calibration_lock"
                reason = "b48_continue_calibration_lock"

        trace_payload.update(
            {
                "b48_controller_profile": profile,
                "b48_timing_error": round(float(timing_error), 6),
                "b48_predictive_timing": round(float(predictive_timing), 6),
                "b48_corrective_gain": round(float(corrective_gain), 6),
                "b48_calibration_lock": int(calibration_lock),
                "b48_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b48_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b48_genetic_candidate"] = int(params["ga_candidate"])

        self._b48_timing_error = float(timing_error)
        self._b48_predictive_timing = float(predictive_timing)
        self._b48_corrective_gain = float(corrective_gain)
        self._b48_calibration_lock = max(0, int(calibration_lock) - 1)
        self._b48_last_tick = int(tick)
        return (
            semantic_action,
            B48_CEREBELLAR_TIMING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b49_controller_params(self) -> dict[str, float]:
        params = self._b48_controller_params()
        defaults = {
            "b49_gate_decay": 0.86,
            "b49_go_gain": 0.34,
            "b49_no_go_gain": 0.30,
            "b49_balance_gain": 0.32,
            "b49_commit_threshold": 0.08,
            "b49_abort_threshold": -0.18,
            "b49_selection_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "striatal_action_gate"
        )
        if profile == "direct_path_facilitation":
            defaults.update({"b49_go_gain": 0.42, "b49_commit_threshold": 0.06})
        elif profile == "indirect_path_suppression":
            defaults.update({"b49_no_go_gain": 0.40, "b49_abort_threshold": -0.14})
        elif profile == "striatal_action_gate_h56":
            defaults.update({"b49_gate_decay": 0.89, "b49_selection_lock_ticks": 6.0})
        elif profile == "genetic_striatal_gate":
            defaults.update({"b49_go_gain": 0.38, "b49_balance_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b49_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b49_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b49_go_signal = 0.0
        self._b49_no_go_signal = 0.0
        self._b49_action_gate_balance = 0.0
        self._b49_selection_lock = 0
        self._b49_last_tick = int(tick)

    def _b49_striatal_action_gate_semantic_action(
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
        ) = self._b48_cerebellar_timing_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b49_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "striatal_action_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b49_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        timing_error = float(trace_payload.get("b48_timing_error", 0.0) or 0.0)
        predictive_timing = float(trace_payload.get("b48_predictive_timing", 0.0) or 0.0)
        corrective_gain = float(trace_payload.get("b48_corrective_gain", 0.0) or 0.0)
        calibration_lock = float(trace_payload.get("b48_calibration_lock", 0.0) or 0.0)
        b48_decision = str(trace_payload.get("b48_decision", "preserve_b47"))

        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        hunger = self._b_series_float(meta, "hunger")
        decay = float(params["b49_gate_decay"])
        previous_go = float(getattr(self, "_b49_go_signal", 0.0))
        previous_no_go = float(getattr(self, "_b49_no_go_signal", 0.0))
        previous_balance = float(getattr(self, "_b49_action_gate_balance", 0.0))
        go_signal = float(
            np.clip(
                previous_go * decay
                + max(0.0, predictive_timing) * float(params["b49_go_gain"])
                + max(0.0, corrective_gain) * 0.22
                + max(0.0, calibration_lock) * 0.04
                + max(0.0, hunger - 0.45) * 0.10,
                -1.0,
                1.0,
            )
        )
        no_go_signal = float(
            np.clip(
                previous_no_go * decay
                + max(0.0, current_threat) * float(params["b49_no_go_gain"])
                + max(0.0, 0.35 - timing_error) * 0.12
                + 0.03,
                -1.0,
                1.0,
            )
        )
        action_gate_balance = float(
            np.clip(
                previous_balance * decay
                + (go_signal - no_go_signal) * float(params["b49_balance_gain"])
                + max(0.0, corrective_gain) * 0.18,
                -1.0,
                1.0,
            )
        )
        selection_lock = int(getattr(self, "_b49_selection_lock", 0))
        decision_label = "preserve_b48"

        if corridor_map:
            if (
                b48_decision
                in {"cerebellar_timing_commit", "continue_calibration_lock"}
                and action_gate_balance >= float(params["b49_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                selection_lock = max(
                    selection_lock,
                    int(params["b49_selection_lock_ticks"]),
                )
                decision_label = "striatal_gate_commit"
                reason = "b49_striatal_gate_commit"
            elif action_gate_balance <= float(params["b49_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "striatal_gate_abort"
                reason = "b49_striatal_gate_abort"
            elif selection_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_selection_lock"
                reason = "b49_continue_selection_lock"

        trace_payload.update(
            {
                "b49_controller_profile": profile,
                "b49_go_signal": round(float(go_signal), 6),
                "b49_no_go_signal": round(float(no_go_signal), 6),
                "b49_action_gate_balance": round(float(action_gate_balance), 6),
                "b49_selection_lock": int(selection_lock),
                "b49_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b49_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b49_genetic_candidate"] = int(params["ga_candidate"])

        self._b49_go_signal = float(go_signal)
        self._b49_no_go_signal = float(no_go_signal)
        self._b49_action_gate_balance = float(action_gate_balance)
        self._b49_selection_lock = max(0, int(selection_lock) - 1)
        self._b49_last_tick = int(tick)
        return (
            semantic_action,
            B49_STRIATAL_ACTION_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b50_controller_params(self) -> dict[str, float]:
        params = self._b49_controller_params()
        defaults = {
            "b50_habit_decay": 0.86,
            "b50_habit_gain": 0.34,
            "b50_chunk_value_gain": 0.30,
            "b50_stability_gain": 0.32,
            "b50_commit_threshold": 0.08,
            "b50_abort_threshold": -0.18,
            "b50_chunk_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "habit_chunking"
        )
        if profile == "action_chunk_value":
            defaults.update({"b50_chunk_value_gain": 0.40, "b50_commit_threshold": 0.07})
        elif profile == "habit_stability":
            defaults.update({"b50_stability_gain": 0.40, "b50_chunk_lock_ticks": 6.0})
        elif profile == "habit_chunking_h56":
            defaults.update({"b50_habit_decay": 0.89, "b50_chunk_lock_ticks": 6.0})
        elif profile == "genetic_habit_chunking":
            defaults.update({"b50_habit_gain": 0.38, "b50_stability_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b50_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b50_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b50_habit_strength = 0.0
        self._b50_chunk_value = 0.0
        self._b50_habit_stability = 0.0
        self._b50_chunk_lock = 0
        self._b50_last_tick = int(tick)

    def _b50_habit_chunking_semantic_action(
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
        ) = self._b49_striatal_action_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b50_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "habit_chunking"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b50_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        go_signal = float(trace_payload.get("b49_go_signal", 0.0) or 0.0)
        no_go_signal = float(trace_payload.get("b49_no_go_signal", 0.0) or 0.0)
        gate_balance = float(trace_payload.get("b49_action_gate_balance", 0.0) or 0.0)
        selection_lock = float(trace_payload.get("b49_selection_lock", 0.0) or 0.0)
        b49_decision = str(trace_payload.get("b49_decision", "preserve_b48"))

        decay = float(params["b50_habit_decay"])
        previous_habit = float(getattr(self, "_b50_habit_strength", 0.0))
        previous_chunk = float(getattr(self, "_b50_chunk_value", 0.0))
        previous_stability = float(getattr(self, "_b50_habit_stability", 0.0))
        habit_strength = float(
            np.clip(
                previous_habit * decay
                + max(0.0, go_signal - no_go_signal) * float(params["b50_habit_gain"])
                + max(0.0, selection_lock) * 0.04,
                -1.0,
                1.0,
            )
        )
        chunk_value = float(
            np.clip(
                previous_chunk * decay
                + max(0.0, gate_balance) * float(params["b50_chunk_value_gain"])
                + max(0.0, habit_strength) * 0.18,
                -1.0,
                1.0,
            )
        )
        habit_stability = float(
            np.clip(
                previous_stability * decay
                + max(0.0, chunk_value) * float(params["b50_stability_gain"])
                + max(0.0, habit_strength) * 0.18,
                -1.0,
                1.0,
            )
        )
        chunk_score = float(
            np.clip(
                habit_strength * 0.30
                + chunk_value * 0.34
                + habit_stability * 0.30
                - no_go_signal * 0.06,
                -1.0,
                1.0,
            )
        )
        chunk_lock = int(getattr(self, "_b50_chunk_lock", 0))
        decision_label = "preserve_b49"

        if corridor_map:
            if (
                b49_decision in {"striatal_gate_commit", "continue_selection_lock"}
                and chunk_score >= float(params["b50_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                chunk_lock = max(chunk_lock, int(params["b50_chunk_lock_ticks"]))
                decision_label = "habit_chunk_commit"
                reason = "b50_habit_chunk_commit"
            elif chunk_score <= float(params["b50_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "habit_chunk_abort"
                reason = "b50_habit_chunk_abort"
            elif chunk_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_habit_chunk"
                reason = "b50_continue_habit_chunk"

        trace_payload.update(
            {
                "b50_controller_profile": profile,
                "b50_habit_strength": round(float(habit_strength), 6),
                "b50_chunk_value": round(float(chunk_value), 6),
                "b50_habit_stability": round(float(habit_stability), 6),
                "b50_chunk_lock": int(chunk_lock),
                "b50_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b50_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b50_genetic_candidate"] = int(params["ga_candidate"])

        self._b50_habit_strength = float(habit_strength)
        self._b50_chunk_value = float(chunk_value)
        self._b50_habit_stability = float(habit_stability)
        self._b50_chunk_lock = max(0, int(chunk_lock) - 1)
        self._b50_last_tick = int(tick)
        return (
            semantic_action,
            B50_HABIT_CHUNKING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b51_controller_params(self) -> dict[str, float]:
        params = self._b50_controller_params()
        defaults = {
            "b51_dopamine_decay": 0.86,
            "b51_prediction_error_gain": 0.32,
            "b51_dopamine_gain": 0.34,
            "b51_habit_modulation_gain": 0.30,
            "b51_commit_threshold": 0.08,
            "b51_abort_threshold": -0.18,
            "b51_modulation_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "dopaminergic_habit_modulation"
        )
        if profile == "reward_prediction_gain":
            defaults.update({"b51_prediction_error_gain": 0.40, "b51_commit_threshold": 0.07})
        elif profile == "novelty_modulated_habit":
            defaults.update({"b51_dopamine_gain": 0.40, "b51_habit_modulation_gain": 0.34})
        elif profile == "dopaminergic_habit_modulation_h56":
            defaults.update({"b51_dopamine_decay": 0.89, "b51_modulation_lock_ticks": 6.0})
        elif profile == "genetic_dopamine_habit":
            defaults.update({"b51_dopamine_gain": 0.38, "b51_habit_modulation_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b51_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b51_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b51_prediction_error = 0.0
        self._b51_dopamine_gain = 0.0
        self._b51_habit_modulation = 0.0
        self._b51_modulation_lock = 0
        self._b51_last_tick = int(tick)

    def _b51_dopaminergic_habit_modulation_semantic_action(
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
        ) = self._b50_habit_chunking_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b51_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "dopaminergic_habit_modulation"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b51_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        habit_strength = float(trace_payload.get("b50_habit_strength", 0.0) or 0.0)
        chunk_value = float(trace_payload.get("b50_chunk_value", 0.0) or 0.0)
        habit_stability = float(trace_payload.get("b50_habit_stability", 0.0) or 0.0)
        chunk_lock = float(trace_payload.get("b50_chunk_lock", 0.0) or 0.0)
        b50_decision = str(trace_payload.get("b50_decision", "preserve_b49"))
        hunger = self._b_series_float(meta, "hunger")
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b51_dopamine_decay"])
        previous_error = float(getattr(self, "_b51_prediction_error", 0.0))
        previous_gain = float(getattr(self, "_b51_dopamine_gain", 0.0))
        previous_modulation = float(getattr(self, "_b51_habit_modulation", 0.0))
        expected_chunk_value = 0.42 + 0.18 * max(0.0, habit_stability)
        reward_proxy = (
            max(0.0, chunk_value)
            + max(0.0, hunger - 0.45) * 0.20
            - max(0.0, current_threat) * 0.14
        )
        prediction_error = float(
            np.clip(
                previous_error * decay
                + max(0.0, reward_proxy - expected_chunk_value)
                * float(params["b51_prediction_error_gain"])
                + max(0.0, chunk_lock) * 0.03,
                -1.0,
                1.0,
            )
        )
        dopamine_gain = float(
            np.clip(
                previous_gain * decay
                + max(0.0, prediction_error) * float(params["b51_dopamine_gain"])
                + max(0.0, habit_strength) * 0.16,
                -1.0,
                1.0,
            )
        )
        habit_modulation = float(
            np.clip(
                previous_modulation * decay
                + max(0.0, dopamine_gain) * float(params["b51_habit_modulation_gain"])
                + max(0.0, habit_stability) * 0.18,
                -1.0,
                1.0,
            )
        )
        modulation_score = float(
            np.clip(
                prediction_error * 0.24
                + dopamine_gain * 0.34
                + habit_modulation * 0.34
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        modulation_lock = int(getattr(self, "_b51_modulation_lock", 0))
        decision_label = "preserve_b50"

        if corridor_map:
            if (
                b50_decision in {"habit_chunk_commit", "continue_habit_chunk"}
                and modulation_score >= float(params["b51_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                modulation_lock = max(
                    modulation_lock,
                    int(params["b51_modulation_lock_ticks"]),
                )
                decision_label = "dopamine_habit_commit"
                reason = "b51_dopamine_habit_commit"
            elif modulation_score <= float(params["b51_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "dopamine_habit_abort"
                reason = "b51_dopamine_habit_abort"
            elif modulation_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_dopamine_modulation"
                reason = "b51_continue_dopamine_modulation"

        trace_payload.update(
            {
                "b51_controller_profile": profile,
                "b51_prediction_error": round(float(prediction_error), 6),
                "b51_dopamine_gain": round(float(dopamine_gain), 6),
                "b51_habit_modulation": round(float(habit_modulation), 6),
                "b51_modulation_lock": int(modulation_lock),
                "b51_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b51_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b51_genetic_candidate"] = int(params["ga_candidate"])

        self._b51_prediction_error = float(prediction_error)
        self._b51_dopamine_gain = float(dopamine_gain)
        self._b51_habit_modulation = float(habit_modulation)
        self._b51_modulation_lock = max(0, int(modulation_lock) - 1)
        self._b51_last_tick = int(tick)
        return (
            semantic_action,
            B51_DOPAMINERGIC_HABIT_MODULATION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b52_controller_params(self) -> dict[str, float]:
        params = self._b51_controller_params()
        defaults = {
            "b52_acetylcholine_decay": 0.86,
            "b52_uncertainty_gain": 0.30,
            "b52_precision_gain": 0.34,
            "b52_attention_gain": 0.32,
            "b52_commit_threshold": 0.08,
            "b52_abort_threshold": -0.20,
            "b52_attention_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cholinergic_precision_gate"
        )
        if profile == "attention_gain":
            defaults.update({"b52_attention_gain": 0.40, "b52_commit_threshold": 0.07})
        elif profile == "uncertainty_release":
            defaults.update({"b52_uncertainty_gain": 0.38, "b52_abort_threshold": -0.16})
        elif profile == "cholinergic_precision_gate_h56":
            defaults.update(
                {"b52_acetylcholine_decay": 0.89, "b52_attention_lock_ticks": 6.0}
            )
        elif profile == "genetic_cholinergic_precision":
            defaults.update({"b52_precision_gain": 0.38, "b52_attention_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b52_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b52_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b52_acetylcholine_level = 0.0
        self._b52_precision_gain = 0.0
        self._b52_uncertainty_signal = 0.0
        self._b52_attention_lock = 0
        self._b52_last_tick = int(tick)

    def _b52_cholinergic_precision_gate_semantic_action(
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
        ) = self._b51_dopaminergic_habit_modulation_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b52_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cholinergic_precision_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b52_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b51_decision = str(trace_payload.get("b51_decision", "preserve_b50"))
        prediction_error = float(trace_payload.get("b51_prediction_error", 0.0) or 0.0)
        dopamine_gain = float(trace_payload.get("b51_dopamine_gain", 0.0) or 0.0)
        habit_modulation = float(trace_payload.get("b51_habit_modulation", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b52_acetylcholine_decay"])
        previous_acetylcholine = float(
            getattr(self, "_b52_acetylcholine_level", 0.0)
        )
        previous_precision = float(getattr(self, "_b52_precision_gain", 0.0))
        previous_uncertainty = float(getattr(self, "_b52_uncertainty_signal", 0.0))
        uncertainty_signal = float(
            np.clip(
                previous_uncertainty * decay
                + max(0.0, abs(prediction_error)) * float(params["b52_uncertainty_gain"])
                + max(0.0, current_threat) * 0.16
                + max(0.0, 0.42 - habit_modulation) * 0.10,
                0.0,
                1.0,
            )
        )
        acetylcholine_level = float(
            np.clip(
                previous_acetylcholine * decay
                + uncertainty_signal * 0.34
                + max(0.0, dopamine_gain) * 0.16,
                0.0,
                1.0,
            )
        )
        precision_gain = float(
            np.clip(
                previous_precision * decay
                + acetylcholine_level * float(params["b52_precision_gain"])
                + max(0.0, habit_modulation) * 0.18,
                0.0,
                1.0,
            )
        )
        attention_score = float(
            np.clip(
                precision_gain * float(params["b52_attention_gain"])
                + max(0.0, dopamine_gain) * 0.22
                + max(0.0, habit_modulation) * 0.20
                - uncertainty_signal * 0.06
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        attention_lock = int(getattr(self, "_b52_attention_lock", 0))
        decision_label = "preserve_b51"

        if corridor_map:
            if (
                b51_decision
                in {"dopamine_habit_commit", "continue_dopamine_modulation"}
                and attention_score >= float(params["b52_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                attention_lock = max(
                    attention_lock,
                    int(params["b52_attention_lock_ticks"]),
                )
                decision_label = "cholinergic_precision_commit"
                reason = "b52_cholinergic_precision_commit"
            elif attention_score <= float(params["b52_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "cholinergic_precision_abort"
                reason = "b52_cholinergic_precision_abort"
            elif attention_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_precision_attention"
                reason = "b52_continue_precision_attention"

        trace_payload.update(
            {
                "b52_controller_profile": profile,
                "b52_acetylcholine_level": round(float(acetylcholine_level), 6),
                "b52_precision_gain": round(float(precision_gain), 6),
                "b52_uncertainty_signal": round(float(uncertainty_signal), 6),
                "b52_attention_lock": int(attention_lock),
                "b52_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b52_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b52_genetic_candidate"] = int(params["ga_candidate"])

        self._b52_acetylcholine_level = float(acetylcholine_level)
        self._b52_precision_gain = float(precision_gain)
        self._b52_uncertainty_signal = float(uncertainty_signal)
        self._b52_attention_lock = max(0, int(attention_lock) - 1)
        self._b52_last_tick = int(tick)
        return (
            semantic_action,
            B52_CHOLINERGIC_PRECISION_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b53_controller_params(self) -> dict[str, float]:
        params = self._b52_controller_params()
        defaults = {
            "b53_norepinephrine_decay": 0.86,
            "b53_surprise_gain": 0.30,
            "b53_arousal_gain": 0.34,
            "b53_precision_mod_gain": 0.32,
            "b53_commit_threshold": 0.08,
            "b53_abort_threshold": -0.20,
            "b53_gain_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "noradrenergic_arousal_gain"
        )
        if profile == "surprise_gain":
            defaults.update({"b53_surprise_gain": 0.40, "b53_commit_threshold": 0.07})
        elif profile == "stress_precision":
            defaults.update({"b53_arousal_gain": 0.40, "b53_abort_threshold": -0.16})
        elif profile == "noradrenergic_arousal_gain_h56":
            defaults.update({"b53_norepinephrine_decay": 0.89, "b53_gain_lock_ticks": 6.0})
        elif profile == "genetic_arousal_precision":
            defaults.update({"b53_arousal_gain": 0.38, "b53_precision_mod_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b53_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b53_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b53_norepinephrine_level = 0.0
        self._b53_arousal_gain = 0.0
        self._b53_surprise_signal = 0.0
        self._b53_gain_lock = 0
        self._b53_last_tick = int(tick)

    def _b53_noradrenergic_arousal_gain_semantic_action(
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
        ) = self._b52_cholinergic_precision_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b53_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "noradrenergic_arousal_gain"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b53_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b52_decision = str(trace_payload.get("b52_decision", "preserve_b51"))
        acetylcholine_level = float(
            trace_payload.get("b52_acetylcholine_level", 0.0) or 0.0
        )
        precision_gain = float(trace_payload.get("b52_precision_gain", 0.0) or 0.0)
        uncertainty_signal = float(
            trace_payload.get("b52_uncertainty_signal", 0.0) or 0.0
        )
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b53_norepinephrine_decay"])
        previous_ne = float(getattr(self, "_b53_norepinephrine_level", 0.0))
        previous_arousal = float(getattr(self, "_b53_arousal_gain", 0.0))
        previous_surprise = float(getattr(self, "_b53_surprise_signal", 0.0))
        surprise_signal = float(
            np.clip(
                previous_surprise * decay
                + max(0.0, uncertainty_signal) * float(params["b53_surprise_gain"])
                + max(0.0, current_threat) * 0.14
                + max(0.0, 0.36 - precision_gain) * 0.10,
                0.0,
                1.0,
            )
        )
        norepinephrine_level = float(
            np.clip(
                previous_ne * decay
                + surprise_signal * 0.32
                + max(0.0, acetylcholine_level) * 0.16,
                0.0,
                1.0,
            )
        )
        arousal_gain = float(
            np.clip(
                previous_arousal * decay
                + norepinephrine_level * float(params["b53_arousal_gain"])
                + max(0.0, precision_gain) * float(params["b53_precision_mod_gain"]),
                0.0,
                1.0,
            )
        )
        gain_score = float(
            np.clip(
                arousal_gain * 0.34
                + max(0.0, precision_gain) * 0.26
                + max(0.0, acetylcholine_level) * 0.18
                - surprise_signal * 0.04
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        gain_lock = int(getattr(self, "_b53_gain_lock", 0))
        decision_label = "preserve_b52"

        if corridor_map:
            if (
                b52_decision
                in {"cholinergic_precision_commit", "continue_precision_attention"}
                and gain_score >= float(params["b53_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                gain_lock = max(gain_lock, int(params["b53_gain_lock_ticks"]))
                decision_label = "noradrenergic_arousal_commit"
                reason = "b53_noradrenergic_arousal_commit"
            elif gain_score <= float(params["b53_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "noradrenergic_arousal_abort"
                reason = "b53_noradrenergic_arousal_abort"
            elif gain_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_arousal_gain"
                reason = "b53_continue_arousal_gain"

        trace_payload.update(
            {
                "b53_controller_profile": profile,
                "b53_norepinephrine_level": round(float(norepinephrine_level), 6),
                "b53_arousal_gain": round(float(arousal_gain), 6),
                "b53_surprise_signal": round(float(surprise_signal), 6),
                "b53_gain_lock": int(gain_lock),
                "b53_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b53_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b53_genetic_candidate"] = int(params["ga_candidate"])

        self._b53_norepinephrine_level = float(norepinephrine_level)
        self._b53_arousal_gain = float(arousal_gain)
        self._b53_surprise_signal = float(surprise_signal)
        self._b53_gain_lock = max(0, int(gain_lock) - 1)
        self._b53_last_tick = int(tick)
        return (
            semantic_action,
            B53_NORADRENERGIC_AROUSAL_GAIN_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b54_controller_params(self) -> dict[str, float]:
        params = self._b53_controller_params()
        defaults = {
            "b54_serotonin_decay": 0.86,
            "b54_patience_gain": 0.34,
            "b54_impulse_suppression_gain": 0.32,
            "b54_arousal_balance_gain": 0.30,
            "b54_commit_threshold": 0.08,
            "b54_abort_threshold": -0.20,
            "b54_patience_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "serotonergic_patience_gate"
        )
        if profile == "impulse_suppression":
            defaults.update({"b54_impulse_suppression_gain": 0.40, "b54_abort_threshold": -0.16})
        elif profile == "patience_balance":
            defaults.update({"b54_patience_gain": 0.40, "b54_commit_threshold": 0.07})
        elif profile == "serotonergic_patience_gate_h56":
            defaults.update({"b54_serotonin_decay": 0.89, "b54_patience_lock_ticks": 6.0})
        elif profile == "genetic_serotonin_patience":
            defaults.update({"b54_patience_gain": 0.38, "b54_arousal_balance_gain": 0.36})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b54_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b54_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b54_serotonin_level = 0.0
        self._b54_patience_signal = 0.0
        self._b54_impulse_suppression = 0.0
        self._b54_patience_lock = 0
        self._b54_last_tick = int(tick)

    def _b54_serotonergic_patience_gate_semantic_action(
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
        ) = self._b53_noradrenergic_arousal_gain_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b54_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "serotonergic_patience_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b54_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b53_decision = str(trace_payload.get("b53_decision", "preserve_b52"))
        norepinephrine_level = float(
            trace_payload.get("b53_norepinephrine_level", 0.0) or 0.0
        )
        arousal_gain = float(trace_payload.get("b53_arousal_gain", 0.0) or 0.0)
        surprise_signal = float(trace_payload.get("b53_surprise_signal", 0.0) or 0.0)
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b54_serotonin_decay"])
        previous_serotonin = float(getattr(self, "_b54_serotonin_level", 0.0))
        previous_patience = float(getattr(self, "_b54_patience_signal", 0.0))
        previous_suppression = float(
            getattr(self, "_b54_impulse_suppression", 0.0)
        )
        balanced_arousal = max(0.0, arousal_gain - surprise_signal * 0.25)
        serotonin_level = float(
            np.clip(
                previous_serotonin * decay
                + max(0.0, balanced_arousal) * 0.30
                + max(0.0, 1.0 - current_threat) * 0.08,
                0.0,
                1.0,
            )
        )
        patience_signal = float(
            np.clip(
                previous_patience * decay
                + serotonin_level * float(params["b54_patience_gain"])
                + max(0.0, norepinephrine_level) * 0.14,
                0.0,
                1.0,
            )
        )
        impulse_suppression = float(
            np.clip(
                previous_suppression * decay
                + max(0.0, surprise_signal) * float(params["b54_impulse_suppression_gain"])
                + max(0.0, current_threat) * 0.12,
                0.0,
                1.0,
            )
        )
        patience_score = float(
            np.clip(
                patience_signal * 0.34
                + serotonin_level * 0.28
                + max(0.0, arousal_gain) * float(params["b54_arousal_balance_gain"])
                - impulse_suppression * 0.08
                - current_threat * 0.08,
                -1.0,
                1.0,
            )
        )
        patience_lock = int(getattr(self, "_b54_patience_lock", 0))
        decision_label = "preserve_b53"

        if corridor_map:
            if (
                b53_decision
                in {"noradrenergic_arousal_commit", "continue_arousal_gain"}
                and patience_score >= float(params["b54_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                patience_lock = max(
                    patience_lock,
                    int(params["b54_patience_lock_ticks"]),
                )
                decision_label = "serotonergic_patience_commit"
                reason = "b54_serotonergic_patience_commit"
            elif patience_score <= float(params["b54_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "serotonergic_patience_abort"
                reason = "b54_serotonergic_patience_abort"
            elif patience_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_patience_lock"
                reason = "b54_continue_patience_lock"

        trace_payload.update(
            {
                "b54_controller_profile": profile,
                "b54_serotonin_level": round(float(serotonin_level), 6),
                "b54_patience_signal": round(float(patience_signal), 6),
                "b54_impulse_suppression": round(float(impulse_suppression), 6),
                "b54_patience_lock": int(patience_lock),
                "b54_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b54_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b54_genetic_candidate"] = int(params["ga_candidate"])

        self._b54_serotonin_level = float(serotonin_level)
        self._b54_patience_signal = float(patience_signal)
        self._b54_impulse_suppression = float(impulse_suppression)
        self._b54_patience_lock = max(0, int(patience_lock) - 1)
        self._b54_last_tick = int(tick)
        return (
            semantic_action,
            B54_SEROTONERGIC_PATIENCE_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b55_controller_params(self) -> dict[str, float]:
        params = self._b54_controller_params()
        defaults = {
            "b55_drive_decay": 0.86,
            "b55_hunger_gain": 0.34,
            "b55_satiety_gain": 0.28,
            "b55_recovery_gain": 0.30,
            "b55_threat_gate_gain": 0.26,
            "b55_commit_threshold": 0.08,
            "b55_abort_threshold": -0.20,
            "b55_drive_lock_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "hypothalamic_drive_coupling"
        )
        if profile == "satiety_recovery_balance":
            defaults.update({"b55_satiety_gain": 0.34, "b55_recovery_gain": 0.36})
        elif profile == "sleep_hunger_arbiter":
            defaults.update({"b55_hunger_gain": 0.38, "b55_threat_gate_gain": 0.30})
        elif profile == "hypothalamic_drive_coupling_h56":
            defaults.update({"b55_drive_decay": 0.89, "b55_drive_lock_ticks": 6.0})
        elif profile == "genetic_hypothalamic_drive":
            defaults.update({"b55_hunger_gain": 0.36, "b55_recovery_gain": 0.34})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b55_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b55_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b55_hypothalamic_drive = 0.0
        self._b55_satiety_signal = 0.0
        self._b55_recovery_bias = 0.0
        self._b55_drive_balance = 0.0
        self._b55_drive_lock = 0
        self._b55_last_tick = int(tick)

    def _b55_hypothalamic_drive_coupling_semantic_action(
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
        ) = self._b54_serotonergic_patience_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b55_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "hypothalamic_drive_coupling"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b55_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        b54_decision = str(trace_payload.get("b54_decision", "preserve_b53"))
        serotonin_level = float(trace_payload.get("b54_serotonin_level", 0.0) or 0.0)
        patience_signal = float(trace_payload.get("b54_patience_signal", 0.0) or 0.0)
        impulse_suppression = float(
            trace_payload.get("b54_impulse_suppression", 0.0) or 0.0
        )
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )

        decay = float(params["b55_drive_decay"])
        previous_drive = float(getattr(self, "_b55_hypothalamic_drive", 0.0))
        previous_satiety = float(getattr(self, "_b55_satiety_signal", 0.0))
        previous_recovery = float(getattr(self, "_b55_recovery_bias", 0.0))
        previous_balance = float(getattr(self, "_b55_drive_balance", 0.0))
        hypothalamic_drive = float(
            np.clip(
                previous_drive * decay
                + max(0.0, hunger) * float(params["b55_hunger_gain"])
                + max(0.0, patience_signal) * 0.14,
                0.0,
                1.0,
            )
        )
        satiety_signal = float(
            np.clip(
                previous_satiety * decay
                + max(0.0, 1.0 - hunger) * float(params["b55_satiety_gain"])
                + max(0.0, serotonin_level) * 0.10,
                0.0,
                1.0,
            )
        )
        recovery_bias = float(
            np.clip(
                previous_recovery * decay
                + max(0.0, sleep_debt) * float(params["b55_recovery_gain"])
                + max(0.0, 1.0 - health) * 0.18,
                0.0,
                1.0,
            )
        )
        drive_balance = float(
            np.clip(
                previous_balance * decay
                + hypothalamic_drive * 0.34
                + max(0.0, patience_signal) * 0.24
                - satiety_signal * 0.08
                - recovery_bias * 0.10
                - current_threat * float(params["b55_threat_gate_gain"])
                - impulse_suppression * 0.04,
                -1.0,
                1.0,
            )
        )
        drive_lock = int(getattr(self, "_b55_drive_lock", 0))
        decision_label = "preserve_b54"

        if corridor_map:
            if (
                b54_decision
                in {"serotonergic_patience_commit", "continue_patience_lock"}
                and drive_balance >= float(params["b55_commit_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                drive_lock = max(drive_lock, int(params["b55_drive_lock_ticks"]))
                decision_label = "hypothalamic_drive_commit"
                reason = "b55_hypothalamic_drive_commit"
            elif drive_balance <= float(params["b55_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "hypothalamic_drive_abort"
                reason = "b55_hypothalamic_drive_abort"
            elif drive_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "continue_drive_lock"
                reason = "b55_continue_drive_lock"

        trace_payload.update(
            {
                "b55_controller_profile": profile,
                "b55_hypothalamic_drive": round(float(hypothalamic_drive), 6),
                "b55_satiety_signal": round(float(satiety_signal), 6),
                "b55_recovery_bias": round(float(recovery_bias), 6),
                "b55_drive_balance": round(float(drive_balance), 6),
                "b55_drive_lock": int(drive_lock),
                "b55_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b55_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b55_genetic_candidate"] = int(params["ga_candidate"])

        self._b55_hypothalamic_drive = float(hypothalamic_drive)
        self._b55_satiety_signal = float(satiety_signal)
        self._b55_recovery_bias = float(recovery_bias)
        self._b55_drive_balance = float(drive_balance)
        self._b55_drive_lock = max(0, int(drive_lock) - 1)
        self._b55_last_tick = int(tick)
        return (
            semantic_action,
            B55_HYPOTHALAMIC_DRIVE_COUPLING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b56_controller_params(self) -> dict[str, float]:
        params = self._b55_controller_params()
        defaults = {
            "b56_endocrine_decay": 0.88,
            "b56_cortisol_gain": 0.30,
            "b56_stress_load_gain": 0.32,
            "b56_recovery_signal_gain": 0.30,
            "b56_drive_mod_gain": 0.28,
            "b56_commit_threshold": 0.08,
            "b56_abort_threshold": -0.20,
            "b56_stress_lock_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "hpa_stress_axis")
        if profile == "cortisol_recovery_balance":
            defaults.update({"b56_cortisol_gain": 0.34, "b56_recovery_signal_gain": 0.36})
        elif profile == "stress_load_gate":
            defaults.update({"b56_stress_load_gain": 0.38, "b56_drive_mod_gain": 0.32})
        elif profile == "hpa_stress_axis_h56":
            defaults.update({"b56_endocrine_decay": 0.90, "b56_stress_lock_ticks": 6.0})
        elif profile == "genetic_hpa_stress":
            defaults.update({"b56_cortisol_gain": 0.32, "b56_drive_mod_gain": 0.34})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b56_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b56_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b56_cortisol_level = 0.0
        self._b56_stress_load = 0.0
        self._b56_recovery_signal = 0.0
        self._b56_endocrine_balance = 0.0
        self._b56_stress_lock = 0
        self._b56_last_tick = int(tick)
