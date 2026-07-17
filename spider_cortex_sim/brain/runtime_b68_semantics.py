from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8eMixin:
    def _b68_controller_params(self) -> dict[str, float]:
        params = self._b67_controller_params()
        defaults = {
            "b68_motor_decay": 0.90,
            "b68_motor_reserve_gain": 0.32,
            "b68_stride_pacing_gain": 0.30,
            "b68_overexertion_gain": 0.34,
            "b68_hold_threshold": 0.18,
            "b68_release_threshold": 0.30,
            "b68_pacing_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "proprioceptive_pacing_gate"
        )
        if profile == "stride_recovery_pacing":
            defaults.update({"b68_stride_pacing_gain": 0.34, "b68_hold_threshold": 0.16})
        elif profile == "overexertion_aware_release":
            defaults.update({"b68_overexertion_gain": 0.30, "b68_release_threshold": 0.26})
        elif profile == "proprioceptive_pacing_gate_h56":
            defaults.update({"b68_motor_decay": 0.92, "b68_pacing_lock_ticks": 5.0})
        elif profile == "genetic_motor_pacing":
            defaults.update({"b68_motor_reserve_gain": 0.34, "b68_overexertion_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b68_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b68_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b68_motor_reserve = 0.0
        self._b68_stride_pacing = 0.0
        self._b68_overexertion_risk = 0.0
        self._b68_pacing_lock = 0
        self._b68_last_tick = int(tick)

    def _b68_motor_pacing_semantic_action(
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
        ) = self._b67_glial_energy_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b68_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "proprioceptive_pacing_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b68_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        shelter_role = str(meta.get("shelter_role", "outside"))
        sleep_obs = self._bound_observation("sleep_center", observation)
        hunger_obs = self._bound_observation("hunger_center", observation)
        hunger = max(
            self._b_series_float(meta, "hunger"),
            self._b_series_float(hunger_obs, "hunger"),
        )
        obs_health = self._b_series_float(sleep_obs, "health")
        health = obs_health if obs_health > 0.0 else self._b_series_float(meta, "health")
        sleep_debt = max(
            self._b_series_float(meta, "sleep_debt"),
            self._b_series_float(sleep_obs, "sleep_debt"),
        )
        shelter_dist = self._b_series_float(meta, "shelter_dist")
        near_shelter = (
            1.0
            if shelter_dist <= 1.0 or shelter_role in {"inside", "deep"}
            else 0.0
        )
        b67_reserve = float(trace_payload.get("b67_glial_reserve", 0.0) or 0.0)
        b67_lactate = float(trace_payload.get("b67_lactate_support", 0.0) or 0.0)
        b67_fatigue = float(trace_payload.get("b67_fatigue_pressure", 0.0) or 0.0)
        b67_lock = int(trace_payload.get("b67_recovery_lock", 0) or 0)
        motor_context = 1.0 if (
            b67_lock > 0
            or b67_fatigue > 0.0
            or b67_lactate > 0.0
        ) else 0.0

        decay = float(params["b68_motor_decay"])
        previous_reserve = float(getattr(self, "_b68_motor_reserve", 0.0))
        previous_stride = float(getattr(self, "_b68_stride_pacing", 0.0))
        previous_risk = float(getattr(self, "_b68_overexertion_risk", 0.0))
        motor_reserve = float(
            np.clip(
                previous_reserve * decay
                + b67_reserve * float(params["b68_motor_reserve_gain"])
                + max(0.0, health) * 0.08
                + b67_lactate * 0.06
                - b67_fatigue * 0.04,
                0.0,
                1.0,
            )
        )
        stride_pacing = float(
            np.clip(
                previous_stride * decay
                + b67_lactate * float(params["b68_stride_pacing_gain"])
                + motor_context * 0.05
                + near_shelter * 0.03,
                0.0,
                1.0,
            )
        )
        overexertion_risk = float(
            np.clip(
                previous_risk * decay
                + b67_fatigue * float(params["b68_overexertion_gain"])
                + sleep_debt * 0.08
                + max(0.0, 1.0 - health) * 0.10
                - motor_reserve * 0.03,
                0.0,
                1.0,
            )
        )
        pacing_lock = int(getattr(self, "_b68_pacing_lock", 0))
        decision_label = "preserve_b67"

        if corridor_map:
            if pacing_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_motor_pacing_lock"
                reason = "b68_continue_motor_pacing_lock"
            elif (
                near_shelter > 0.0
                and overexertion_risk >= float(params["b68_hold_threshold"])
                and stride_pacing > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                pacing_lock = max(pacing_lock, int(params["b68_pacing_lock_ticks"]))
                decision_label = "motor_recovery_pace"
                reason = "b68_motor_recovery_pace"
            elif (
                motor_reserve >= float(params["b68_release_threshold"])
                and overexertion_risk < float(params["b68_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "motor_safe_stride"
                reason = "b68_motor_safe_stride"
            elif motor_context > 0.0:
                decision_label = "motor_safe_stride"

        trace_payload.update(
            {
                "b68_controller_profile": profile,
                "b68_motor_reserve": round(float(motor_reserve), 6),
                "b68_stride_pacing": round(float(stride_pacing), 6),
                "b68_overexertion_risk": round(float(overexertion_risk), 6),
                "b68_pacing_lock": int(pacing_lock),
                "b68_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b68_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b68_genetic_candidate"] = int(params["ga_candidate"])

        self._b68_motor_reserve = float(motor_reserve)
        self._b68_stride_pacing = float(stride_pacing)
        self._b68_overexertion_risk = float(overexertion_risk)
        self._b68_pacing_lock = max(0, int(pacing_lock) - 1)
        self._b68_last_tick = int(tick)
        return (
            semantic_action,
            B68_MOTOR_PACING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
