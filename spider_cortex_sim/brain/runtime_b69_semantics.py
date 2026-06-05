from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8fMixin:
    def _b69_controller_params(self) -> dict[str, float]:
        params = self._b68_controller_params()
        defaults = {
            "b69_orientation_decay": 0.90,
            "b69_heading_confidence_gain": 0.32,
            "b69_turn_error_gain": 0.30,
            "b69_stability_gain": 0.34,
            "b69_hold_threshold": 0.18,
            "b69_release_threshold": 0.30,
            "b69_orientation_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "vestibular_orientation_gate"
        )
        if profile == "heading_stability_pacing":
            defaults.update({"b69_heading_confidence_gain": 0.35, "b69_release_threshold": 0.28})
        elif profile == "turn_error_recovery":
            defaults.update({"b69_turn_error_gain": 0.34, "b69_hold_threshold": 0.16})
        elif profile == "vestibular_orientation_gate_h56":
            defaults.update({"b69_orientation_decay": 0.92, "b69_orientation_lock_ticks": 5.0})
        elif profile == "genetic_orientation_gate":
            defaults.update({"b69_heading_confidence_gain": 0.34, "b69_turn_error_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b69_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b69_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b69_heading_confidence = 0.0
        self._b69_turn_error = 0.0
        self._b69_orientation_stability = 0.0
        self._b69_orientation_lock = 0
        self._b69_last_tick = int(tick)

    def _b69_vestibular_orientation_semantic_action(
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
        ) = self._b68_motor_pacing_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b69_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "vestibular_orientation_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b69_reset_state_if_needed(tick)

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
        recent_pain = self._b_series_float(meta, "recent_pain")
        recent_contact = self._b_series_float(meta, "recent_contact")
        b68_reserve = float(trace_payload.get("b68_motor_reserve", 0.0) or 0.0)
        b68_stride = float(trace_payload.get("b68_stride_pacing", 0.0) or 0.0)
        b68_risk = float(trace_payload.get("b68_overexertion_risk", 0.0) or 0.0)
        b68_lock = int(trace_payload.get("b68_pacing_lock", 0) or 0)
        orientation_context = 1.0 if (
            b68_lock > 0
            or b68_stride > 0.0
            or b68_risk > 0.0
        ) else 0.0

        decay = float(params["b69_orientation_decay"])
        previous_confidence = float(getattr(self, "_b69_heading_confidence", 0.0))
        previous_error = float(getattr(self, "_b69_turn_error", 0.0))
        previous_stability = float(getattr(self, "_b69_orientation_stability", 0.0))
        heading_confidence = float(
            np.clip(
                previous_confidence * decay
                + b68_reserve * float(params["b69_heading_confidence_gain"])
                + b68_stride * 0.08
                + orientation_context * 0.04,
                0.0,
                1.0,
            )
        )
        turn_error = float(
            np.clip(
                previous_error * decay
                + b68_risk * float(params["b69_turn_error_gain"])
                + sleep_debt * 0.05
                + recent_pain * 0.08
                + recent_contact * 0.10
                + max(0.0, 1.0 - heading_confidence) * 0.03,
                0.0,
                1.0,
            )
        )
        orientation_stability = float(
            np.clip(
                previous_stability * decay
                + heading_confidence * float(params["b69_stability_gain"])
                + b68_stride * 0.06
                + near_shelter * 0.02
                - turn_error * 0.03,
                0.0,
                1.0,
            )
        )
        orientation_lock = int(getattr(self, "_b69_orientation_lock", 0))
        decision_label = "preserve_b68"

        if corridor_map:
            if orientation_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_orientation_lock"
                reason = "b69_continue_orientation_lock"
            elif (
                near_shelter > 0.0
                and turn_error >= float(params["b69_hold_threshold"])
                and orientation_stability > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                orientation_lock = max(
                    orientation_lock,
                    int(params["b69_orientation_lock_ticks"]),
                )
                decision_label = "vestibular_recenter"
                reason = "b69_vestibular_recenter"
            elif (
                heading_confidence >= float(params["b69_release_threshold"])
                and turn_error < float(params["b69_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "stable_heading_stride"
                reason = "b69_stable_heading_stride"
            elif orientation_context > 0.0:
                decision_label = "stable_heading_stride"

        trace_payload.update(
            {
                "b69_controller_profile": profile,
                "b69_heading_confidence": round(float(heading_confidence), 6),
                "b69_turn_error": round(float(turn_error), 6),
                "b69_orientation_stability": round(float(orientation_stability), 6),
                "b69_orientation_lock": int(orientation_lock),
                "b69_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b69_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b69_genetic_candidate"] = int(params["ga_candidate"])

        self._b69_heading_confidence = float(heading_confidence)
        self._b69_turn_error = float(turn_error)
        self._b69_orientation_stability = float(orientation_stability)
        self._b69_orientation_lock = max(0, int(orientation_lock) - 1)
        self._b69_last_tick = int(tick)
        return (
            semantic_action,
            B69_VESTIBULAR_ORIENTATION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
