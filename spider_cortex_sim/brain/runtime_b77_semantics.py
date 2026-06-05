from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8nMixin:
    def _b77_controller_params(self) -> dict[str, float]:
        params = self._b76_controller_params()
        defaults = {
            "b77_error_decay": 0.90,
            "b77_prediction_gain": 0.34,
            "b77_climbing_gain": 0.32,
            "b77_correction_gain": 0.30,
            "b77_hold_threshold": 0.18,
            "b77_release_threshold": 0.30,
            "b77_error_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "olivary_error_correction"
        )
        if profile == "error_prediction_pacing":
            defaults.update({"b77_prediction_gain": 0.35, "b77_release_threshold": 0.28})
        elif profile == "climbing_fiber_recovery":
            defaults.update({"b77_climbing_gain": 0.35, "b77_error_lock_ticks": 5.0})
        elif profile == "olivary_error_correction_h56":
            defaults.update({"b77_error_decay": 0.92, "b77_error_lock_ticks": 5.0})
        elif profile == "genetic_olivary_error":
            defaults.update({"b77_prediction_gain": 0.35, "b77_correction_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b77_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b77_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b77_prediction_error = 0.0
        self._b77_climbing_fiber_drive = 0.0
        self._b77_corrective_timing = 0.0
        self._b77_stability_confidence = 0.0
        self._b77_error_lock = 0
        self._b77_last_tick = int(tick)

    def _b77_olivary_error_semantic_action(
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
        ) = self._b76_cerebellar_stride_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b77_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "olivary_error_correction"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b77_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        shelter_role = str(meta.get("shelter_role", "outside"))
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("alert_center", observation)
        hunger = max(
            self._b_series_float(meta, "hunger"),
            self._b_series_float(hunger_obs, "hunger"),
        )
        sleep_debt = max(
            self._b_series_float(meta, "sleep_debt"),
            self._b_series_float(sleep_obs, "sleep_debt"),
        )
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        near_shelter = (
            1.0
            if shelter_dist <= 1.0 or shelter_role in {"at_shelter", "deep_shelter"}
            else 0.0
        )
        predator_smell = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
        )
        predator_motion = max(
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
        )
        recent_contact = self._b_series_float(meta, "recent_contact")
        b76_timing = float(trace_payload.get("b76_stride_timing_signal", 0.0) or 0.0)
        b76_error = float(trace_payload.get("b76_error_correction", 0.0) or 0.0)
        b76_smoothing = float(trace_payload.get("b76_burst_smoothing", 0.0) or 0.0)
        b76_gate = float(trace_payload.get("b76_stride_gate", 0.0) or 0.0)
        b76_lock = int(trace_payload.get("b76_stride_lock", 0) or 0)
        error_context = 1.0 if (
            b76_lock > 0
            or b76_timing > 0.0
            or b76_error > 0.0
            or b76_gate > 0.0
        ) else 0.0

        decay = float(params["b77_error_decay"])
        previous_error = float(getattr(self, "_b77_prediction_error", 0.0))
        previous_climbing = float(getattr(self, "_b77_climbing_fiber_drive", 0.0))
        previous_timing = float(getattr(self, "_b77_corrective_timing", 0.0))
        previous_stability = float(getattr(self, "_b77_stability_confidence", 0.0))
        prediction_error = float(
            np.clip(
                previous_error * decay
                + b76_error * float(params["b77_prediction_gain"])
                + abs(b76_gate - b76_smoothing) * 0.10
                + predator_motion * 0.05
                + predator_smell * 0.04
                + recent_contact * 0.05
                + sleep_debt * 0.02
                + error_context * 0.03,
                0.0,
                1.0,
            )
        )
        climbing_fiber_drive = float(
            np.clip(
                previous_climbing * decay
                + prediction_error * float(params["b77_climbing_gain"])
                + b76_error * 0.12
                + max(0.0, b76_timing - b76_gate) * 0.08,
                0.0,
                1.0,
            )
        )
        corrective_timing = float(
            np.clip(
                previous_timing * decay
                + b76_timing * float(params["b77_correction_gain"])
                + b76_smoothing * 0.12
                + max(0.0, b76_gate - prediction_error) * 0.08
                + error_context * 0.02,
                0.0,
                1.0,
            )
        )
        stability_confidence = float(
            np.clip(
                previous_stability * decay
                + corrective_timing * 0.16
                + b76_gate * 0.12
                + max(0.0, b76_smoothing - prediction_error) * 0.10
                + error_context * 0.02,
                0.0,
                1.0,
            )
        )
        error_lock = int(getattr(self, "_b77_error_lock", 0))
        decision_label = "preserve_b76"

        if corridor_map:
            if error_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_error_lock"
                reason = "b77_continue_error_lock"
            elif (
                near_shelter > 0.0
                and prediction_error >= float(params["b77_hold_threshold"])
                and climbing_fiber_drive > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                error_lock = max(error_lock, int(params["b77_error_lock_ticks"]))
                decision_label = "olivary_error_hold"
                reason = "b77_olivary_error_hold"
            elif (
                corrective_timing >= float(params["b77_release_threshold"])
                and stability_confidence >= float(params["b77_release_threshold"])
                and prediction_error < float(params["b77_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "corrective_stride_release"
                reason = "b77_corrective_stride_release"
            elif stability_confidence > 0.0 and corrective_timing > 0.0:
                if hunger >= 0.68 and near_shelter > 0.0:
                    semantic_action = "MOVE_TO_FOOD"
                decision_label = "stabilized_stride_release"
                reason = "b77_stabilized_stride_release"

        trace_payload.update(
            {
                "b77_controller_profile": profile,
                "b77_prediction_error": round(float(prediction_error), 6),
                "b77_climbing_fiber_drive": round(float(climbing_fiber_drive), 6),
                "b77_corrective_timing": round(float(corrective_timing), 6),
                "b77_stability_confidence": round(float(stability_confidence), 6),
                "b77_error_lock": int(error_lock),
                "b77_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b77_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b77_genetic_candidate"] = int(params["ga_candidate"])

        self._b77_prediction_error = float(prediction_error)
        self._b77_climbing_fiber_drive = float(climbing_fiber_drive)
        self._b77_corrective_timing = float(corrective_timing)
        self._b77_stability_confidence = float(stability_confidence)
        self._b77_error_lock = max(0, int(error_lock) - 1)
        self._b77_last_tick = int(tick)
        return (
            semantic_action,
            B77_OLIVARY_ERROR_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
