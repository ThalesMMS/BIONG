from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart4Mixin:
    def _b23_conflict_monitor_semantic_action(
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
        ) = self._b22_prospective_replay_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b23_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "conflict_monitor"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b23_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b22_decision = str(trace_payload.get("b22_decision", "preserve_b22"))
        prospective_sim = float(trace_payload.get("b22_prospective_sim", 0.0) or 0.0)
        forward_model_score = float(
            trace_payload.get("b22_forward_model_score", 0.0) or 0.0
        )
        viability_projection = float(
            trace_payload.get("b22_viability_projection", 0.0) or 0.0
        )
        abort_projection = float(trace_payload.get("b22_abort_projection", 0.0) or 0.0)
        prediction_error = float(
            np.clip(
                abs(viability_projection - forward_model_score)
                + abort_projection * float(params["b23_error_gain"]),
                0.0,
                1.0,
            )
        )
        conflict_input = float(
            np.clip(
                prediction_error
                + max(0.0, abort_projection - viability_projection) * 0.35,
                0.0,
                1.0,
            )
        )
        conflict_memory = max(
            float(getattr(self, "_b23_conflict_memory", 0.0))
            * float(params["b23_conflict_decay"]),
            conflict_input,
        )
        stability_vote = float(
            np.clip(
                viability_projection
                + prospective_sim * 0.20
                - conflict_memory * 0.25,
                0.0,
                1.0,
            )
        )
        abort_bias = float(
            np.clip(
                abort_projection + conflict_memory * 0.25 - stability_vote * 0.15,
                0.0,
                1.0,
            )
        )
        monitor_lock = int(getattr(self, "_b23_monitor_lock", 0))
        conflict_state = "non_corridor"
        decision_label = "preserve_b22"

        if corridor_map:
            if (
                b22_decision
                in {"prospective_replay_continue", "continue_prospective_lock"}
                and stability_vote >= float(params["b23_conflict_threshold"])
                and abort_bias < float(params["b23_abort_bias_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                monitor_lock = max(monitor_lock, int(params["b23_monitor_commit_ticks"]))
                conflict_state = "conflict_monitor_stabilizes_route"
                decision_label = "conflict_monitor_continue"
                reason = "b23_conflict_monitor_continue"
            elif (
                abort_bias >= float(params["b23_abort_bias_threshold"])
                and abort_bias > stability_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                conflict_state = "conflict_monitor_predicts_abort"
                decision_label = "conflict_monitor_abort"
                reason = "b23_conflict_monitor_abort"
            elif monitor_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                conflict_state = "conflict_lock_continues"
                decision_label = "continue_conflict_lock"
                reason = "b23_continue_conflict_lock"

        trace_payload.update(
            {
                "b23_controller_profile": profile,
                "b23_conflict_state": conflict_state,
                "b23_prediction_error": round(float(prediction_error), 6),
                "b23_conflict_memory": round(float(conflict_memory), 6),
                "b23_stability_vote": round(float(stability_vote), 6),
                "b23_abort_bias": round(float(abort_bias), 6),
                "b23_monitor_lock": int(monitor_lock),
                "b23_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b23_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b23_genetic_candidate"] = int(params["ga_candidate"])

        self._b23_conflict_memory = float(conflict_memory)
        self._b23_monitor_lock = max(0, int(monitor_lock) - 1)
        self._b23_last_tick = int(tick)
        return (
            semantic_action,
            B23_CONFLICT_MONITOR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b24_controller_params(self) -> dict[str, float]:
        params = self._b23_controller_params()
        defaults = {
            "b24_precision_decay": 0.88,
            "b24_precision_threshold": 0.26,
            "b24_uncertainty_threshold": 0.64,
            "b24_precision_gain": 0.45,
            "b24_precision_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "precision_conflict"
        )
        if profile == "prediction_precision_gate":
            defaults.update(
                {"b24_precision_threshold": 0.22, "b24_precision_commit_ticks": 6.0}
            )
        elif profile == "reliability_abort":
            defaults.update({"b24_uncertainty_threshold": 0.60, "b24_precision_gain": 0.52})
        elif profile == "precision_conflict_h56":
            defaults.update({"b24_precision_decay": 0.90, "b24_precision_commit_ticks": 6.0})
        elif profile == "genetic_precision_conflict":
            defaults.update({"b24_precision_threshold": 0.24, "b24_precision_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b24_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b24_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b24_precision_memory = 0.0
        self._b24_precision_lock = 0
        self._b24_last_tick = int(tick)

    def _b24_precision_conflict_semantic_action(
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
        ) = self._b23_conflict_monitor_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b24_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "precision_conflict"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b24_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b23_decision = str(trace_payload.get("b23_decision", "preserve_b23"))
        prediction_error = float(trace_payload.get("b23_prediction_error", 0.0) or 0.0)
        conflict_memory = float(trace_payload.get("b23_conflict_memory", 0.0) or 0.0)
        stability_vote = float(trace_payload.get("b23_stability_vote", 0.0) or 0.0)
        abort_bias = float(trace_payload.get("b23_abort_bias", 0.0) or 0.0)
        reliability_input = float(
            np.clip(
                stability_vote * max(0.0, 1.0 - prediction_error)
                + max(0.0, stability_vote - abort_bias) * 0.25,
                0.0,
                1.0,
            )
        )
        precision_memory = max(
            float(getattr(self, "_b24_precision_memory", 0.0))
            * float(params["b24_precision_decay"]),
            reliability_input,
        )
        uncertainty_pressure = float(
            np.clip(
                prediction_error
                + conflict_memory * 0.35
                + abort_bias * 0.25
                - precision_memory * 0.25,
                0.0,
                1.0,
            )
        )
        precision_vote = float(
            np.clip(
                precision_memory * float(params["b24_precision_gain"])
                + stability_vote * 0.35
                - uncertainty_pressure * 0.15,
                0.0,
                1.0,
            )
        )
        abort_precision = float(
            np.clip(abort_bias + uncertainty_pressure * 0.20 - precision_vote * 0.15, 0.0, 1.0)
        )
        precision_lock = int(getattr(self, "_b24_precision_lock", 0))
        precision_state = "non_corridor"
        decision_label = "preserve_b23"

        if corridor_map:
            if (
                b23_decision in {"conflict_monitor_continue", "continue_conflict_lock"}
                and precision_vote >= float(params["b24_precision_threshold"])
                and abort_precision < float(params["b24_uncertainty_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                precision_lock = max(
                    precision_lock, int(params["b24_precision_commit_ticks"])
                )
                precision_state = "precision_conflict_stabilizes_route"
                decision_label = "precision_conflict_continue"
                reason = "b24_precision_conflict_continue"
            elif (
                abort_precision >= float(params["b24_uncertainty_threshold"])
                and abort_precision > precision_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                precision_state = "precision_conflict_predicts_abort"
                decision_label = "precision_conflict_abort"
                reason = "b24_precision_conflict_abort"
            elif precision_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                precision_state = "precision_lock_continues"
                decision_label = "continue_precision_lock"
                reason = "b24_continue_precision_lock"

        trace_payload.update(
            {
                "b24_controller_profile": profile,
                "b24_precision_state": precision_state,
                "b24_precision_memory": round(float(precision_memory), 6),
                "b24_precision_vote": round(float(precision_vote), 6),
                "b24_uncertainty_pressure": round(float(uncertainty_pressure), 6),
                "b24_abort_precision": round(float(abort_precision), 6),
                "b24_precision_lock": int(precision_lock),
                "b24_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b24_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b24_genetic_candidate"] = int(params["ga_candidate"])

        self._b24_precision_memory = float(precision_memory)
        self._b24_precision_lock = max(0, int(precision_lock) - 1)
        self._b24_last_tick = int(tick)
        return (
            semantic_action,
            B24_PRECISION_CONFLICT_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b25_controller_params(self) -> dict[str, float]:
        params = self._b24_controller_params()
        defaults = {
            "b25_confidence_decay": 0.90,
            "b25_confidence_threshold": 0.28,
            "b25_doubt_threshold": 0.66,
            "b25_control_gain": 0.44,
            "b25_meta_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "metacognitive_confidence"
        )
        if profile == "confidence_calibration":
            defaults.update(
                {"b25_confidence_threshold": 0.24, "b25_meta_commit_ticks": 6.0}
            )
        elif profile == "uncertainty_integrator":
            defaults.update({"b25_doubt_threshold": 0.62, "b25_control_gain": 0.52})
        elif profile == "metacognitive_confidence_h56":
            defaults.update({"b25_confidence_decay": 0.92, "b25_meta_commit_ticks": 6.0})
        elif profile == "genetic_metacognition":
            defaults.update({"b25_confidence_threshold": 0.26, "b25_control_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b25_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b25_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b25_confidence_memory = 0.0
        self._b25_meta_lock = 0
        self._b25_last_tick = int(tick)

    def _b25_metacognitive_confidence_semantic_action(
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
        ) = self._b24_precision_conflict_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b25_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "metacognitive_confidence"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b25_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b24_decision = str(trace_payload.get("b24_decision", "preserve_b24"))
        precision_memory = float(trace_payload.get("b24_precision_memory", 0.0) or 0.0)
        precision_vote = float(trace_payload.get("b24_precision_vote", 0.0) or 0.0)
        uncertainty_pressure = float(
            trace_payload.get("b24_uncertainty_pressure", 0.0) or 0.0
        )
        abort_precision = float(trace_payload.get("b24_abort_precision", 0.0) or 0.0)
        confidence_input = float(
            np.clip(
                precision_vote * max(0.0, 1.0 - uncertainty_pressure)
                + precision_memory * 0.25
                + max(0.0, precision_vote - abort_precision) * 0.20,
                0.0,
                1.0,
            )
        )
        confidence_memory = max(
            float(getattr(self, "_b25_confidence_memory", 0.0))
            * float(params["b25_confidence_decay"]),
            confidence_input,
        )
        doubt_pressure = float(
            np.clip(
                uncertainty_pressure
                + abort_precision * 0.35
                - confidence_memory * 0.25,
                0.0,
                1.0,
            )
        )
        control_gain = float(
            np.clip(
                confidence_memory * float(params["b25_control_gain"])
                + precision_vote * 0.35
                - doubt_pressure * 0.15,
                0.0,
                1.0,
            )
        )
        confidence_vote = float(
            np.clip(control_gain + confidence_memory * 0.25 - doubt_pressure * 0.20, 0.0, 1.0)
        )
        meta_lock = int(getattr(self, "_b25_meta_lock", 0))
        metacognitive_state = "non_corridor"
        decision_label = "preserve_b24"

        if corridor_map:
            if (
                b24_decision in {"precision_conflict_continue", "continue_precision_lock"}
                and confidence_vote >= float(params["b25_confidence_threshold"])
                and doubt_pressure < float(params["b25_doubt_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                meta_lock = max(meta_lock, int(params["b25_meta_commit_ticks"]))
                metacognitive_state = "metacognition_confirms_route"
                decision_label = "metacognitive_confidence_continue"
                reason = "b25_metacognitive_confidence_continue"
            elif (
                doubt_pressure >= float(params["b25_doubt_threshold"])
                and doubt_pressure > confidence_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                metacognitive_state = "metacognition_withholds_route"
                decision_label = "metacognitive_confidence_abort"
                reason = "b25_metacognitive_confidence_abort"
            elif meta_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                metacognitive_state = "meta_lock_continues"
                decision_label = "continue_meta_lock"
                reason = "b25_continue_meta_lock"

        trace_payload.update(
            {
                "b25_controller_profile": profile,
                "b25_metacognitive_state": metacognitive_state,
                "b25_confidence_memory": round(float(confidence_memory), 6),
                "b25_confidence_vote": round(float(confidence_vote), 6),
                "b25_doubt_pressure": round(float(doubt_pressure), 6),
                "b25_control_gain": round(float(control_gain), 6),
                "b25_meta_lock": int(meta_lock),
                "b25_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b25_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b25_genetic_candidate"] = int(params["ga_candidate"])

        self._b25_confidence_memory = float(confidence_memory)
        self._b25_meta_lock = max(0, int(meta_lock) - 1)
        self._b25_last_tick = int(tick)
        return (
            semantic_action,
            B25_METACOGNITIVE_CONFIDENCE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b26_controller_params(self) -> dict[str, float]:
        params = self._b25_controller_params()
        defaults = {
            "b26_error_decay": 0.88,
            "b26_prediction_threshold": 0.24,
            "b26_abort_threshold": 0.68,
            "b26_control_gain": 0.46,
            "b26_stability_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "allostatic_prediction"
        )
        if profile == "setpoint_drift":
            defaults.update(
                {"b26_prediction_threshold": 0.22, "b26_stability_commit_ticks": 6.0}
            )
        elif profile == "error_suppression":
            defaults.update({"b26_abort_threshold": 0.64, "b26_control_gain": 0.52})
        elif profile == "allostatic_prediction_h56":
            defaults.update({"b26_error_decay": 0.90, "b26_stability_commit_ticks": 6.0})
        elif profile == "genetic_allostasis":
            defaults.update({"b26_prediction_threshold": 0.23, "b26_control_gain": 0.50})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b26_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b26_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b26_prediction_error_memory = 0.0
        self._b26_stability_lock = 0
        self._b26_last_tick = int(tick)

    def _b26_allostatic_prediction_semantic_action(
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
        ) = self._b25_metacognitive_confidence_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b26_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "allostatic_prediction"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b26_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        b25_decision = str(trace_payload.get("b25_decision", "preserve_b25"))
        confidence_vote = float(trace_payload.get("b25_confidence_vote", 0.0) or 0.0)
        doubt_pressure = float(trace_payload.get("b25_doubt_pressure", 0.0) or 0.0)
        control_gain = float(trace_payload.get("b25_control_gain", 0.0) or 0.0)
        homeostatic_pressure = float(
            np.clip(hunger * 0.45 + sleep_debt * 0.25 + max(0.0, 1.0 - health) * 0.30, 0.0, 1.0)
        )
        route_stability = float(
            np.clip(confidence_vote * 0.45 + control_gain * 0.35 - doubt_pressure * 0.20, 0.0, 1.0)
        )
        prediction_error_input = float(
            np.clip(homeostatic_pressure * 0.55 + doubt_pressure * 0.35 - route_stability * 0.30, 0.0, 1.0)
        )
        prediction_error = max(
            float(getattr(self, "_b26_prediction_error_memory", 0.0))
            * float(params["b26_error_decay"]),
            prediction_error_input,
        )
        setpoint_pressure = float(
            np.clip(homeostatic_pressure + prediction_error * 0.25 - route_stability * 0.15, 0.0, 1.0)
        )
        allostatic_vote = float(
            np.clip(
                route_stability * float(params["b26_control_gain"])
                + confidence_vote * 0.35
                - prediction_error * 0.18,
                0.0,
                1.0,
            )
        )
        stability_lock = int(getattr(self, "_b26_stability_lock", 0))
        allostatic_state = "non_corridor"
        decision_label = "preserve_b25"

        if corridor_map:
            if (
                b25_decision
                in {"metacognitive_confidence_continue", "continue_meta_lock"}
                and allostatic_vote >= float(params["b26_prediction_threshold"])
                and prediction_error < float(params["b26_abort_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                stability_lock = max(
                    stability_lock,
                    int(params["b26_stability_commit_ticks"]),
                )
                allostatic_state = "allostasis_confirms_route"
                decision_label = "allostatic_prediction_continue"
                reason = "b26_allostatic_prediction_continue"
            elif (
                prediction_error >= float(params["b26_abort_threshold"])
                and prediction_error > allostatic_vote
            ):
                semantic_action = "MOVE_TO_SHELTER"
                allostatic_state = "allostasis_predicts_abort"
                decision_label = "allostatic_prediction_abort"
                reason = "b26_allostatic_prediction_abort"
            elif stability_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                allostatic_state = "allostatic_lock_continues"
                decision_label = "continue_allostatic_lock"
                reason = "b26_continue_allostatic_lock"

        trace_payload.update(
            {
                "b26_controller_profile": profile,
                "b26_allostatic_state": allostatic_state,
                "b26_prediction_error": round(float(prediction_error), 6),
                "b26_setpoint_pressure": round(float(setpoint_pressure), 6),
                "b26_control_vote": round(float(allostatic_vote), 6),
                "b26_stability_lock": int(stability_lock),
                "b26_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b26_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b26_genetic_candidate"] = int(params["ga_candidate"])

        self._b26_prediction_error_memory = float(prediction_error)
        self._b26_stability_lock = max(0, int(stability_lock) - 1)
        self._b26_last_tick = int(tick)
        return (
            semantic_action,
            B26_ALLOSTATIC_PREDICTION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b27_controller_params(self) -> dict[str, float]:
        params = self._b26_controller_params()
        defaults = {
            "b27_arousal_decay": 0.86,
            "b27_gain_threshold": 0.24,
            "b27_stress_threshold": 0.70,
            "b27_modulation_gain": 0.48,
            "b27_arousal_commit_ticks": 5.0,
        }
        profile = str(getattr(self.config, "b_controller_profile", None) or "arousal_gain")
        if profile == "stress_modulation":
            defaults.update({"b27_stress_threshold": 0.66, "b27_modulation_gain": 0.54})
        elif profile == "energy_arousal":
            defaults.update({"b27_gain_threshold": 0.22, "b27_arousal_commit_ticks": 6.0})
        elif profile == "arousal_gain_h56":
            defaults.update({"b27_arousal_decay": 0.88, "b27_arousal_commit_ticks": 6.0})
        elif profile == "genetic_arousal":
            defaults.update({"b27_gain_threshold": 0.23, "b27_modulation_gain": 0.52})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b27_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b27_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b27_arousal_memory = 0.0
        self._b27_arousal_lock = 0
        self._b27_last_tick = int(tick)

    def _b27_arousal_gain_semantic_action(
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
        ) = self._b26_allostatic_prediction_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b27_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "arousal_gain")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b27_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        threat_arr = np.asarray(observation.get("threat", np.zeros(1)), dtype=float).ravel()
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        threat = float(threat_arr[0]) if threat_arr.size else 0.0
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        b26_decision = str(trace_payload.get("b26_decision", "preserve_b26"))
        prediction_error = float(trace_payload.get("b26_prediction_error", 0.0) or 0.0)
        setpoint_pressure = float(trace_payload.get("b26_setpoint_pressure", 0.0) or 0.0)
        control_vote = float(trace_payload.get("b26_control_vote", 0.0) or 0.0)
        arousal_input = float(
            np.clip(
                setpoint_pressure * 0.35
                + control_vote * 0.30
                + hunger * 0.15
                + max(0.0, 1.0 - health) * 0.10
                + sleep_debt * 0.10,
                0.0,
                1.0,
            )
        )
        arousal_level = max(
            float(getattr(self, "_b27_arousal_memory", 0.0))
            * float(params["b27_arousal_decay"]),
            arousal_input,
        )
        stress_pressure = float(
            np.clip(threat * 0.35 + prediction_error * 0.40 - arousal_level * 0.18, 0.0, 1.0)
        )
        gain_modulation = float(
            np.clip(
                arousal_level * float(params["b27_modulation_gain"])
                + control_vote * 0.38
                - stress_pressure * 0.16,
                0.0,
                1.0,
            )
        )
        arousal_lock = int(getattr(self, "_b27_arousal_lock", 0))
        arousal_state = "non_corridor"
        decision_label = "preserve_b26"

        if corridor_map:
            if (
                b26_decision
                in {"allostatic_prediction_continue", "continue_allostatic_lock"}
                and gain_modulation >= float(params["b27_gain_threshold"])
                and stress_pressure < float(params["b27_stress_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                arousal_lock = max(arousal_lock, int(params["b27_arousal_commit_ticks"]))
                arousal_state = "arousal_gain_stabilizes_route"
                decision_label = "arousal_gain_continue"
                reason = "b27_arousal_gain_continue"
            elif (
                stress_pressure >= float(params["b27_stress_threshold"])
                and stress_pressure > gain_modulation
            ):
                semantic_action = "MOVE_TO_SHELTER"
                arousal_state = "stress_modulation_aborts_route"
                decision_label = "arousal_gain_abort"
                reason = "b27_arousal_gain_abort"
            elif arousal_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                arousal_state = "arousal_lock_continues"
                decision_label = "continue_arousal_lock"
                reason = "b27_continue_arousal_lock"

        trace_payload.update(
            {
                "b27_controller_profile": profile,
                "b27_arousal_state": arousal_state,
                "b27_arousal_level": round(float(arousal_level), 6),
                "b27_gain_modulation": round(float(gain_modulation), 6),
                "b27_stress_pressure": round(float(stress_pressure), 6),
                "b27_arousal_lock": int(arousal_lock),
                "b27_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b27_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b27_genetic_candidate"] = int(params["ga_candidate"])

        self._b27_arousal_memory = float(arousal_level)
        self._b27_arousal_lock = max(0, int(arousal_lock) - 1)
        self._b27_last_tick = int(tick)
        return (
            semantic_action,
            B27_AROUSAL_GAIN_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b28_controller_params(self) -> dict[str, float]:
        params = self._b27_controller_params()
        defaults = {
            "b28_attention_decay": 0.86,
            "b28_focus_threshold": 0.24,
            "b28_distractor_threshold": 0.70,
            "b28_attention_gain": 0.48,
            "b28_attention_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "interoceptive_attention"
        )
        if profile == "threat_focus_attention":
            defaults.update(
                {"b28_distractor_threshold": 0.66, "b28_attention_gain": 0.54}
            )
        elif profile == "homeostatic_attention":
            defaults.update(
                {"b28_focus_threshold": 0.22, "b28_attention_commit_ticks": 6.0}
            )
        elif profile == "interoceptive_attention_h56":
            defaults.update(
                {"b28_attention_decay": 0.88, "b28_attention_commit_ticks": 6.0}
            )
        elif profile == "genetic_attention":
            defaults.update({"b28_focus_threshold": 0.23, "b28_attention_gain": 0.52})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b28_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b28_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b28_focus_memory = 0.0
        self._b28_attention_lock = 0
        self._b28_last_tick = int(tick)

    def _b28_interoceptive_attention_semantic_action(
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
        ) = self._b27_arousal_gain_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b28_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "interoceptive_attention"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b28_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        threat_arr = np.asarray(observation.get("threat", np.zeros(1)), dtype=float).ravel()
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        threat = float(threat_arr[0]) if threat_arr.size else 0.0
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        b27_decision = str(trace_payload.get("b27_decision", "preserve_b27"))
        arousal_level = float(trace_payload.get("b27_arousal_level", 0.0) or 0.0)
        gain_modulation = float(trace_payload.get("b27_gain_modulation", 0.0) or 0.0)
        stress_pressure = float(trace_payload.get("b27_stress_pressure", 0.0) or 0.0)
        interoceptive_focus = float(
            np.clip(
                hunger * 0.30
                + sleep_debt * 0.18
                + max(0.0, 1.0 - health) * 0.16
                + arousal_level * 0.18
                + gain_modulation * 0.18,
                0.0,
                1.0,
            )
        )
        attention_memory = max(
            float(getattr(self, "_b28_focus_memory", 0.0))
            * float(params["b28_attention_decay"]),
            interoceptive_focus,
        )
        distractor_pressure = float(
            np.clip(threat * 0.34 + stress_pressure * 0.40 - attention_memory * 0.18, 0.0, 1.0)
        )
        attention_gain = float(
            np.clip(
                attention_memory * float(params["b28_attention_gain"])
                + gain_modulation * 0.36
                - distractor_pressure * 0.16,
                0.0,
                1.0,
            )
        )
        attention_lock = int(getattr(self, "_b28_attention_lock", 0))
        attention_state = "non_corridor"
        decision_label = "preserve_b27"

        if corridor_map:
            if (
                b27_decision in {"arousal_gain_continue", "continue_arousal_lock"}
                and attention_gain >= float(params["b28_focus_threshold"])
                and distractor_pressure < float(params["b28_distractor_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                attention_lock = max(
                    attention_lock,
                    int(params["b28_attention_commit_ticks"]),
                )
                attention_state = "interoceptive_attention_stabilizes_route"
                decision_label = "interoceptive_attention_continue"
                reason = "b28_interoceptive_attention_continue"
            elif (
                distractor_pressure >= float(params["b28_distractor_threshold"])
                and distractor_pressure > attention_gain
            ):
                semantic_action = "MOVE_TO_SHELTER"
                attention_state = "attention_distractor_aborts_route"
                decision_label = "interoceptive_attention_abort"
                reason = "b28_interoceptive_attention_abort"
            elif attention_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                attention_state = "attention_lock_continues"
                decision_label = "continue_attention_lock"
                reason = "b28_continue_attention_lock"

        trace_payload.update(
            {
                "b28_controller_profile": profile,
                "b28_attention_state": attention_state,
                "b28_interoceptive_focus": round(float(interoceptive_focus), 6),
                "b28_attention_gain": round(float(attention_gain), 6),
                "b28_distractor_pressure": round(float(distractor_pressure), 6),
                "b28_attention_lock": int(attention_lock),
                "b28_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b28_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b28_genetic_candidate"] = int(params["ga_candidate"])

        self._b28_focus_memory = float(attention_memory)
        self._b28_attention_lock = max(0, int(attention_lock) - 1)
        self._b28_last_tick = int(tick)
        return (
            semantic_action,
            B28_INTEROCEPTIVE_ATTENTION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b29_controller_params(self) -> dict[str, float]:
        params = self._b28_controller_params()
        defaults = {
            "b29_salience_decay": 0.86,
            "b29_corridor_threshold": 0.24,
            "b29_threat_threshold": 0.70,
            "b29_homeostatic_gain": 0.40,
            "b29_competition_gain": 0.52,
            "b29_salience_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "salience_competition"
        )
        if profile == "threat_salience_gate":
            defaults.update(
                {"b29_threat_threshold": 0.66, "b29_competition_gain": 0.56}
            )
        elif profile == "homeostatic_salience_gate":
            defaults.update(
                {"b29_homeostatic_gain": 0.46, "b29_salience_commit_ticks": 6.0}
            )
        elif profile == "salience_competition_h56":
            defaults.update(
                {"b29_salience_decay": 0.88, "b29_salience_commit_ticks": 6.0}
            )
        elif profile == "genetic_salience":
            defaults.update({"b29_corridor_threshold": 0.23, "b29_competition_gain": 0.55})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b29_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b29_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b29_salience_memory = 0.0
        self._b29_salience_lock = 0
        self._b29_last_tick = int(tick)

    def _b29_salience_competition_semantic_action(
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
        ) = self._b28_interoceptive_attention_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b29_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "salience_competition"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b29_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        threat_arr = np.asarray(observation.get("threat", np.zeros(1)), dtype=float).ravel()
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        threat = float(threat_arr[0]) if threat_arr.size else 0.0
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        sleep_debt = float(sleep_arr[1]) if sleep_arr.size > 1 else 0.0
        attention_gain = float(trace_payload.get("b28_attention_gain", 0.0) or 0.0)
        distractor = float(trace_payload.get("b28_distractor_pressure", 0.0) or 0.0)
        focus = float(trace_payload.get("b28_interoceptive_focus", 0.0) or 0.0)
        b28_decision = str(trace_payload.get("b28_decision", "preserve_b28"))

        homeostatic_salience = float(
            np.clip(
                hunger * 0.36
                + sleep_debt * 0.22
                + max(0.0, 1.0 - health) * 0.24
                + focus * float(params["b29_homeostatic_gain"]),
                0.0,
                1.0,
            )
        )
        threat_salience = float(
            np.clip(threat * 0.52 + distractor * 0.34 - attention_gain * 0.14, 0.0, 1.0)
        )
        salience_memory = max(
            float(getattr(self, "_b29_salience_memory", 0.0))
            * float(params["b29_salience_decay"]),
            attention_gain,
        )
        corridor_salience = float(
            np.clip(
                salience_memory * float(params["b29_competition_gain"])
                + homeostatic_salience * 0.28
                - threat_salience * 0.18,
                0.0,
                1.0,
            )
        )
        salience_map = {
            "corridor": corridor_salience,
            "homeostasis": homeostatic_salience,
            "threat": threat_salience,
        }
        winner_channel = max(salience_map, key=salience_map.get)
        salience_lock = int(getattr(self, "_b29_salience_lock", 0))
        salience_state = "non_corridor"
        decision_label = "preserve_b28"

        if corridor_map:
            if (
                b28_decision in {
                    "interoceptive_attention_continue",
                    "continue_attention_lock",
                }
                and winner_channel in {"corridor", "homeostasis"}
                and corridor_salience >= float(params["b29_corridor_threshold"])
                and threat_salience < float(params["b29_threat_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                salience_lock = max(
                    salience_lock,
                    int(params["b29_salience_commit_ticks"]),
                )
                salience_state = "corridor_salience_wins"
                decision_label = "salience_competition_continue"
                reason = "b29_salience_competition_continue"
            elif (
                winner_channel == "threat"
                and threat_salience >= float(params["b29_threat_threshold"])
            ):
                semantic_action = "MOVE_TO_SHELTER"
                salience_state = "threat_salience_aborts"
                decision_label = "salience_competition_abort"
                reason = "b29_salience_competition_abort"
            elif salience_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                salience_state = "salience_lock_continues"
                decision_label = "continue_salience_lock"
                reason = "b29_continue_salience_lock"

        trace_payload.update(
            {
                "b29_controller_profile": profile,
                "b29_salience_state": salience_state,
                "b29_threat_salience": round(float(threat_salience), 6),
                "b29_homeostatic_salience": round(float(homeostatic_salience), 6),
                "b29_corridor_salience": round(float(corridor_salience), 6),
                "b29_winner_channel": winner_channel,
                "b29_salience_lock": int(salience_lock),
                "b29_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b29_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b29_genetic_candidate"] = int(params["ga_candidate"])

        self._b29_salience_memory = float(corridor_salience)
        self._b29_salience_lock = max(0, int(salience_lock) - 1)
        self._b29_last_tick = int(tick)
        return (
            semantic_action,
            B29_SALIENCE_COMPETITION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b30_controller_params(self) -> dict[str, float]:
        params = self._b29_controller_params()
        defaults = {
            "b30_gate_decay": 0.86,
            "b30_go_threshold": 0.24,
            "b30_nogo_threshold": 0.70,
            "b30_go_gain": 0.52,
            "b30_nogo_gain": 0.44,
            "b30_gate_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "basal_ganglia_gate"
        )
        if profile == "go_nogo_balance":
            defaults.update({"b30_go_threshold": 0.22, "b30_gate_commit_ticks": 6.0})
        elif profile == "threat_inhibition_gate":
            defaults.update({"b30_nogo_threshold": 0.66, "b30_nogo_gain": 0.50})
        elif profile == "basal_ganglia_gate_h56":
            defaults.update({"b30_gate_decay": 0.88, "b30_gate_commit_ticks": 6.0})
        elif profile == "genetic_action_gate":
            defaults.update({"b30_go_threshold": 0.23, "b30_go_gain": 0.55})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b30_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b30_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b30_gate_memory = 0.0
        self._b30_gate_lock = 0
        self._b30_last_tick = int(tick)

    def _b30_basal_ganglia_gate_semantic_action(
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
        ) = self._b29_salience_competition_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b30_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "basal_ganglia_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b30_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        threat_arr = np.asarray(observation.get("threat", np.zeros(1)), dtype=float).ravel()
        threat = float(threat_arr[0]) if threat_arr.size else 0.0
        attention_gain = float(trace_payload.get("b28_attention_gain", 0.0) or 0.0)
        distractor = float(trace_payload.get("b28_distractor_pressure", 0.0) or 0.0)
        corridor_salience = float(trace_payload.get("b29_corridor_salience", 0.0) or 0.0)
        homeostatic_salience = float(
            trace_payload.get("b29_homeostatic_salience", 0.0) or 0.0
        )
        threat_salience = float(trace_payload.get("b29_threat_salience", 0.0) or 0.0)
        b29_decision = str(trace_payload.get("b29_decision", "preserve_b29"))

        gate_memory = max(
            float(getattr(self, "_b30_gate_memory", 0.0))
            * float(params["b30_gate_decay"]),
            corridor_salience,
        )
        no_go_signal = float(
            np.clip(
                threat * 0.34
                + threat_salience * float(params["b30_nogo_gain"])
                + distractor * 0.22
                - gate_memory * 0.10,
                0.0,
                1.0,
            )
        )
        go_signal = float(
            np.clip(
                gate_memory * float(params["b30_go_gain"])
                + homeostatic_salience * 0.30
                + attention_gain * 0.24
                - no_go_signal * 0.18,
                0.0,
                1.0,
            )
        )
        action_gate = "go" if go_signal >= no_go_signal else "no_go"
        gate_lock = int(getattr(self, "_b30_gate_lock", 0))
        gate_state = "non_corridor"
        decision_label = "preserve_b29"

        if corridor_map:
            if (
                b29_decision in {"salience_competition_continue", "continue_salience_lock"}
                and go_signal >= float(params["b30_go_threshold"])
                and no_go_signal < float(params["b30_nogo_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                gate_lock = max(gate_lock, int(params["b30_gate_commit_ticks"]))
                gate_state = "basal_go_gate_opens"
                decision_label = "basal_gate_go"
                reason = "b30_basal_gate_go"
            elif no_go_signal >= float(params["b30_nogo_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                action_gate = "no_go"
                gate_state = "basal_nogo_gate_inhibits"
                decision_label = "basal_gate_no_go"
                reason = "b30_basal_gate_no_go"
            elif gate_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                action_gate = "go"
                gate_state = "basal_gate_lock_continues"
                decision_label = "continue_basal_gate_lock"
                reason = "b30_continue_basal_gate_lock"

        trace_payload.update(
            {
                "b30_controller_profile": profile,
                "b30_gate_state": gate_state,
                "b30_go_signal": round(float(go_signal), 6),
                "b30_no_go_signal": round(float(no_go_signal), 6),
                "b30_action_gate": action_gate,
                "b30_gate_lock": int(gate_lock),
                "b30_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b30_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b30_genetic_candidate"] = int(params["ga_candidate"])

        self._b30_gate_memory = float(go_signal)
        self._b30_gate_lock = max(0, int(gate_lock) - 1)
        self._b30_last_tick = int(tick)
        return (
            semantic_action,
            B30_BASAL_GANGLIA_GATE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b31_controller_params(self) -> dict[str, float]:
        params = self._b30_controller_params()
        defaults = {
            "b31_dopamine_decay": 0.86,
            "b31_go_threshold": 0.24,
            "b31_nogo_threshold": 0.70,
            "b31_prediction_gain": 0.46,
            "b31_phasic_gain": 0.52,
            "b31_tonic_gain": 0.42,
            "b31_dopamine_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "dopamine_prediction_error"
        )
        if profile == "tonic_dopamine_gate":
            defaults.update({"b31_tonic_gain": 0.48, "b31_dopamine_commit_ticks": 6.0})
        elif profile == "phasic_dopamine_gate":
            defaults.update({"b31_phasic_gain": 0.58, "b31_go_threshold": 0.22})
        elif profile == "dopamine_prediction_error_h56":
            defaults.update({"b31_dopamine_decay": 0.88, "b31_dopamine_commit_ticks": 6.0})
        elif profile == "genetic_dopamine_gate":
            defaults.update({"b31_prediction_gain": 0.50, "b31_phasic_gain": 0.55})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b31_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b31_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b31_value_prediction = 0.0
        self._b31_dopamine_memory = 0.0
        self._b31_dopamine_lock = 0
        self._b31_last_tick = int(tick)

    def _b31_dopamine_prediction_error_semantic_action(
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
        ) = self._b30_basal_ganglia_gate_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b31_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "dopamine_prediction_error"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b31_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        hunger_arr = np.asarray(observation.get("hunger", np.zeros(1)), dtype=float).ravel()
        sleep_arr = np.asarray(observation.get("sleep", np.zeros(2)), dtype=float).ravel()
        hunger = float(hunger_arr[0]) if hunger_arr.size else 0.0
        health = float(sleep_arr[0]) if sleep_arr.size else 1.0
        go_signal = float(trace_payload.get("b30_go_signal", 0.0) or 0.0)
        no_go_signal = float(trace_payload.get("b30_no_go_signal", 0.0) or 0.0)
        action_gate = str(trace_payload.get("b30_action_gate", "go"))
        b30_decision = str(trace_payload.get("b30_decision", "preserve_b30"))
        corridor_salience = float(trace_payload.get("b29_corridor_salience", 0.0) or 0.0)
        homeostatic_salience = float(
            trace_payload.get("b29_homeostatic_salience", 0.0) or 0.0
        )

        reward_proxy = float(
            np.clip(
                corridor_salience * 0.40
                + homeostatic_salience * 0.30
                + go_signal * 0.24
                + max(0.0, health - 0.5) * 0.06
                - no_go_signal * 0.18
                - max(0.0, 0.70 - hunger) * 0.04,
                0.0,
                1.0,
            )
        )
        prediction = float(getattr(self, "_b31_value_prediction", 0.0))
        reward_prediction_error = reward_proxy - prediction
        tonic_dopamine = max(
            float(getattr(self, "_b31_dopamine_memory", 0.0))
            * float(params["b31_dopamine_decay"]),
            reward_proxy * float(params["b31_tonic_gain"]),
        )
        phasic_dopamine = float(
            np.clip(
                max(0.0, reward_prediction_error) * float(params["b31_phasic_gain"])
                + go_signal * 0.20
                - no_go_signal * 0.12,
                0.0,
                1.0,
            )
        )
        gate_bias = float(
            np.clip(
                tonic_dopamine
                + phasic_dopamine
                + reward_prediction_error * float(params["b31_prediction_gain"]),
                -1.0,
                1.0,
            )
        )
        dopamine_lock = int(getattr(self, "_b31_dopamine_lock", 0))
        dopamine_state = "non_corridor"
        decision_label = "preserve_b30"

        if corridor_map:
            if (
                b30_decision in {"basal_gate_go", "continue_basal_gate_lock"}
                and action_gate == "go"
                and gate_bias >= float(params["b31_go_threshold"])
                and no_go_signal < float(params["b31_nogo_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                dopamine_lock = max(
                    dopamine_lock,
                    int(params["b31_dopamine_commit_ticks"]),
                )
                dopamine_state = "dopamine_go_bias_stabilizes"
                decision_label = "dopamine_gate_go"
                reason = "b31_dopamine_gate_go"
            elif no_go_signal >= float(params["b31_nogo_threshold"]) and gate_bias < 0.25:
                semantic_action = "MOVE_TO_SHELTER"
                dopamine_state = "dopamine_no_go_inhibits"
                decision_label = "dopamine_gate_no_go"
                reason = "b31_dopamine_gate_no_go"
            elif dopamine_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                dopamine_state = "dopamine_lock_continues"
                decision_label = "continue_dopamine_lock"
                reason = "b31_continue_dopamine_lock"

        trace_payload.update(
            {
                "b31_controller_profile": profile,
                "b31_dopamine_state": dopamine_state,
                "b31_reward_prediction_error": round(float(reward_prediction_error), 6),
                "b31_tonic_dopamine": round(float(tonic_dopamine), 6),
                "b31_phasic_dopamine": round(float(phasic_dopamine), 6),
                "b31_gate_bias": round(float(gate_bias), 6),
                "b31_dopamine_lock": int(dopamine_lock),
                "b31_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b31_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b31_genetic_candidate"] = int(params["ga_candidate"])

        self._b31_value_prediction = float(
            prediction + 0.35 * reward_prediction_error
        )
        self._b31_dopamine_memory = float(tonic_dopamine)
        self._b31_dopamine_lock = max(0, int(dopamine_lock) - 1)
        self._b31_last_tick = int(tick)
        return (
            semantic_action,
            B31_DOPAMINE_PREDICTION_ERROR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
