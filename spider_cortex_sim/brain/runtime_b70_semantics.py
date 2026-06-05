from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8gMixin:
    def _b70_controller_params(self) -> dict[str, float]:
        params = self._b69_controller_params()
        defaults = {
            "b70_flow_decay": 0.90,
            "b70_flow_confidence_gain": 0.32,
            "b70_lateral_drift_gain": 0.30,
            "b70_looming_risk_gain": 0.34,
            "b70_hold_threshold": 0.18,
            "b70_release_threshold": 0.30,
            "b70_flow_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "optic_flow_stabilization_gate"
        )
        if profile == "lateral_flow_pacing":
            defaults.update({"b70_lateral_drift_gain": 0.34, "b70_hold_threshold": 0.16})
        elif profile == "looming_risk_recovery":
            defaults.update({"b70_looming_risk_gain": 0.32, "b70_release_threshold": 0.28})
        elif profile == "optic_flow_stabilization_gate_h56":
            defaults.update({"b70_flow_decay": 0.92, "b70_flow_lock_ticks": 5.0})
        elif profile == "genetic_optic_flow_gate":
            defaults.update({"b70_flow_confidence_gain": 0.34, "b70_looming_risk_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b70_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b70_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b70_flow_confidence = 0.0
        self._b70_lateral_drift = 0.0
        self._b70_looming_risk = 0.0
        self._b70_flow_lock = 0
        self._b70_last_tick = int(tick)

    def _b70_optic_flow_semantic_action(
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
        ) = self._b69_vestibular_orientation_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b70_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "optic_flow_stabilization_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b70_reset_state_if_needed(tick)

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
        recent_pain = self._b_series_float(meta, "recent_pain")
        recent_contact = self._b_series_float(meta, "recent_contact")
        predator_motion = max(
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
        )
        b69_heading = float(trace_payload.get("b69_heading_confidence", 0.0) or 0.0)
        b69_turn_error = float(trace_payload.get("b69_turn_error", 0.0) or 0.0)
        b69_stability = float(trace_payload.get("b69_orientation_stability", 0.0) or 0.0)
        b69_lock = int(trace_payload.get("b69_orientation_lock", 0) or 0)
        flow_context = 1.0 if (
            b69_lock > 0
            or b69_heading > 0.0
            or b69_turn_error > 0.0
        ) else 0.0

        decay = float(params["b70_flow_decay"])
        previous_confidence = float(getattr(self, "_b70_flow_confidence", 0.0))
        previous_drift = float(getattr(self, "_b70_lateral_drift", 0.0))
        previous_looming = float(getattr(self, "_b70_looming_risk", 0.0))
        flow_confidence = float(
            np.clip(
                previous_confidence * decay
                + b69_heading * float(params["b70_flow_confidence_gain"])
                + b69_stability * 0.08
                + flow_context * 0.04,
                0.0,
                1.0,
            )
        )
        lateral_drift = float(
            np.clip(
                previous_drift * decay
                + b69_turn_error * float(params["b70_lateral_drift_gain"])
                + max(0.0, 1.0 - b69_stability) * 0.04
                + recent_contact * 0.08,
                0.0,
                1.0,
            )
        )
        looming_risk = float(
            np.clip(
                previous_looming * decay
                + b69_turn_error * float(params["b70_looming_risk_gain"])
                + predator_motion * 0.08
                + recent_pain * 0.06
                + sleep_debt * 0.04
                - flow_confidence * 0.03,
                0.0,
                1.0,
            )
        )
        flow_lock = int(getattr(self, "_b70_flow_lock", 0))
        decision_label = "preserve_b69"

        if corridor_map:
            if flow_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_flow_lock"
                reason = "b70_continue_flow_lock"
            elif (
                near_shelter > 0.0
                and max(lateral_drift, looming_risk) >= float(params["b70_hold_threshold"])
                and flow_confidence > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                flow_lock = max(flow_lock, int(params["b70_flow_lock_ticks"]))
                decision_label = "optic_flow_recenter"
                reason = "b70_optic_flow_recenter"
            elif (
                flow_confidence >= float(params["b70_release_threshold"])
                and lateral_drift < float(params["b70_hold_threshold"])
                and looming_risk < float(params["b70_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "clear_path_stride"
                reason = "b70_clear_path_stride"
            elif flow_context > 0.0:
                decision_label = "clear_path_stride"

        trace_payload.update(
            {
                "b70_controller_profile": profile,
                "b70_flow_confidence": round(float(flow_confidence), 6),
                "b70_lateral_drift": round(float(lateral_drift), 6),
                "b70_looming_risk": round(float(looming_risk), 6),
                "b70_flow_lock": int(flow_lock),
                "b70_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b70_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b70_genetic_candidate"] = int(params["ga_candidate"])

        self._b70_flow_confidence = float(flow_confidence)
        self._b70_lateral_drift = float(lateral_drift)
        self._b70_looming_risk = float(looming_risk)
        self._b70_flow_lock = max(0, int(flow_lock) - 1)
        self._b70_last_tick = int(tick)
        return (
            semantic_action,
            B70_OPTIC_FLOW_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
