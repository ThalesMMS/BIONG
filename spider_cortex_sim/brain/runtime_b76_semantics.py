from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8mMixin:
    def _b76_controller_params(self) -> dict[str, float]:
        params = self._b75_controller_params()
        defaults = {
            "b76_timing_decay": 0.90,
            "b76_stride_gain": 0.34,
            "b76_error_gain": 0.32,
            "b76_smoothing_gain": 0.30,
            "b76_hold_threshold": 0.18,
            "b76_stride_threshold": 0.30,
            "b76_stride_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cerebellar_stride_gate"
        )
        if profile == "stride_error_pacing":
            defaults.update({"b76_error_gain": 0.35, "b76_hold_threshold": 0.16})
        elif profile == "burst_smoothing_recovery":
            defaults.update({"b76_smoothing_gain": 0.34, "b76_stride_lock_ticks": 5.0})
        elif profile == "cerebellar_stride_gate_h56":
            defaults.update({"b76_timing_decay": 0.92, "b76_stride_lock_ticks": 5.0})
        elif profile == "genetic_cerebellar_stride":
            defaults.update({"b76_stride_gain": 0.35, "b76_smoothing_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b76_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b76_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b76_stride_timing_signal = 0.0
        self._b76_error_correction = 0.0
        self._b76_burst_smoothing = 0.0
        self._b76_stride_gate = 0.0
        self._b76_stride_lock = 0
        self._b76_last_tick = int(tick)

    def _b76_cerebellar_stride_semantic_action(
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
        ) = self._b75_basal_thalamic_release_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b76_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "cerebellar_stride_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b76_reset_state_if_needed(tick)

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
        b75_timing = float(trace_payload.get("b75_release_timing_drive", 0.0) or 0.0)
        b75_go = float(trace_payload.get("b75_go_disinhibition", 0.0) or 0.0)
        b75_nogo = float(trace_payload.get("b75_nogo_brake", 0.0) or 0.0)
        b75_burst = float(trace_payload.get("b75_burst_window", 0.0) or 0.0)
        b75_lock = int(trace_payload.get("b75_release_lock", 0) or 0)
        stride_context = 1.0 if (
            b75_lock > 0
            or b75_timing > 0.0
            or b75_go > 0.0
            or b75_burst > 0.0
        ) else 0.0

        decay = float(params["b76_timing_decay"])
        previous_timing = float(getattr(self, "_b76_stride_timing_signal", 0.0))
        previous_error = float(getattr(self, "_b76_error_correction", 0.0))
        previous_smoothing = float(getattr(self, "_b76_burst_smoothing", 0.0))
        previous_gate = float(getattr(self, "_b76_stride_gate", 0.0))
        stride_timing_signal = float(
            np.clip(
                previous_timing * decay
                + b75_timing * float(params["b76_stride_gain"])
                + b75_burst * 0.16
                + b75_go * 0.10
                + stride_context * 0.03,
                0.0,
                1.0,
            )
        )
        error_correction = float(
            np.clip(
                previous_error * decay
                + b75_nogo * float(params["b76_error_gain"])
                + abs(b75_go - b75_burst) * 0.08
                + predator_motion * 0.05
                + predator_smell * 0.04
                + recent_contact * 0.05
                + sleep_debt * 0.03,
                0.0,
                1.0,
            )
        )
        burst_smoothing = float(
            np.clip(
                previous_smoothing * decay
                + b75_burst * float(params["b76_smoothing_gain"])
                + max(0.0, b75_burst - error_correction) * 0.10
                + b75_timing * 0.08
                + stride_context * 0.02,
                0.0,
                1.0,
            )
        )
        stride_gate = float(
            np.clip(
                previous_gate * decay
                + stride_timing_signal * 0.14
                + burst_smoothing * 0.16
                + max(0.0, b75_go - error_correction) * 0.08
                + stride_context * 0.03,
                0.0,
                1.0,
            )
        )
        stride_lock = int(getattr(self, "_b76_stride_lock", 0))
        decision_label = "preserve_b75"

        if corridor_map:
            if stride_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_stride_lock"
                reason = "b76_continue_stride_lock"
            elif (
                near_shelter > 0.0
                and error_correction >= float(params["b76_hold_threshold"])
                and stride_timing_signal > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                stride_lock = max(stride_lock, int(params["b76_stride_lock_ticks"]))
                decision_label = "cerebellar_stride_hold"
                reason = "b76_cerebellar_stride_hold"
            elif (
                stride_gate >= float(params["b76_stride_threshold"])
                and burst_smoothing >= float(params["b76_stride_threshold"])
                and error_correction < float(params["b76_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "timed_stride_release"
                reason = "b76_timed_stride_release"
            elif stride_gate > 0.0 and burst_smoothing > 0.0:
                if hunger >= 0.68 and near_shelter > 0.0:
                    semantic_action = "MOVE_TO_FOOD"
                decision_label = "smoothed_burst_stride"
                reason = "b76_smoothed_burst_stride"

        trace_payload.update(
            {
                "b76_controller_profile": profile,
                "b76_stride_timing_signal": round(float(stride_timing_signal), 6),
                "b76_error_correction": round(float(error_correction), 6),
                "b76_burst_smoothing": round(float(burst_smoothing), 6),
                "b76_stride_gate": round(float(stride_gate), 6),
                "b76_stride_lock": int(stride_lock),
                "b76_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b76_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b76_genetic_candidate"] = int(params["ga_candidate"])

        self._b76_stride_timing_signal = float(stride_timing_signal)
        self._b76_error_correction = float(error_correction)
        self._b76_burst_smoothing = float(burst_smoothing)
        self._b76_stride_gate = float(stride_gate)
        self._b76_stride_lock = max(0, int(stride_lock) - 1)
        self._b76_last_tick = int(tick)
        return (
            semantic_action,
            B76_CEREBELLAR_STRIDE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
