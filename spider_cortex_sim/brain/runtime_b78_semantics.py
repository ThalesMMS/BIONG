from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8oMixin:
    def _b78_controller_params(self) -> dict[str, float]:
        params = self._b77_controller_params()
        defaults = {
            "b78_balance_decay": 0.90,
            "b78_error_gain": 0.34,
            "b78_stabilization_gain": 0.32,
            "b78_confidence_gain": 0.30,
            "b78_hold_threshold": 0.18,
            "b78_release_threshold": 0.30,
            "b78_balance_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "vestibular_balance"
        )
        if profile == "head_stabilization":
            defaults.update({"b78_stabilization_gain": 0.35, "b78_release_threshold": 0.28})
        elif profile == "locomotor_confidence":
            defaults.update({"b78_confidence_gain": 0.35, "b78_balance_lock_ticks": 5.0})
        elif profile == "vestibular_balance_h56":
            defaults.update({"b78_balance_decay": 0.92, "b78_balance_lock_ticks": 5.0})
        elif profile == "genetic_vestibular_balance":
            defaults.update({"b78_error_gain": 0.35, "b78_confidence_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b78_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b78_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b78_balance_error = 0.0
        self._b78_head_stabilization = 0.0
        self._b78_locomotor_confidence = 0.0
        self._b78_slip_risk = 0.0
        self._b78_balance_lock = 0
        self._b78_last_tick = int(tick)

    def _b78_vestibular_balance_semantic_action(
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
        ) = self._b77_olivary_error_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b78_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "vestibular_balance"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b78_reset_state_if_needed(tick)

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
            if shelter_dist <= 1.0 or shelter_role in {"inside", "deep"}
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
        b77_prediction_error = float(
            trace_payload.get("b77_prediction_error", 0.0) or 0.0
        )
        b77_climbing = float(
            trace_payload.get("b77_climbing_fiber_drive", 0.0) or 0.0
        )
        b77_timing = float(trace_payload.get("b77_corrective_timing", 0.0) or 0.0)
        b77_stability = float(
            trace_payload.get("b77_stability_confidence", 0.0) or 0.0
        )
        b77_lock = int(trace_payload.get("b77_error_lock", 0) or 0)
        balance_context = 1.0 if (
            b77_lock > 0
            or b77_prediction_error > 0.0
            or b77_climbing > 0.0
            or b77_timing > 0.0
        ) else 0.0

        decay = float(params["b78_balance_decay"])
        previous_error = float(getattr(self, "_b78_balance_error", 0.0))
        previous_stabilization = float(getattr(self, "_b78_head_stabilization", 0.0))
        previous_confidence = float(getattr(self, "_b78_locomotor_confidence", 0.0))
        previous_slip = float(getattr(self, "_b78_slip_risk", 0.0))
        balance_error = float(
            np.clip(
                previous_error * decay
                + b77_prediction_error * float(params["b78_error_gain"])
                + b77_climbing * 0.12
                + abs(b77_timing - b77_stability) * 0.10
                + predator_motion * 0.05
                + predator_smell * 0.03
                + recent_contact * 0.05
                + sleep_debt * 0.02
                + balance_context * 0.03,
                0.0,
                1.0,
            )
        )
        head_stabilization = float(
            np.clip(
                previous_stabilization * decay
                + b77_timing * float(params["b78_stabilization_gain"])
                + b77_stability * 0.14
                + max(0.0, b77_climbing - balance_error) * 0.06
                + balance_context * 0.02,
                0.0,
                1.0,
            )
        )
        locomotor_confidence = float(
            np.clip(
                previous_confidence * decay
                + head_stabilization * float(params["b78_confidence_gain"])
                + b77_stability * 0.18
                + b77_timing * 0.08
                - balance_error * 0.04,
                0.0,
                1.0,
            )
        )
        slip_risk = float(
            np.clip(
                previous_slip * decay
                + balance_error * 0.20
                + max(0.0, b77_prediction_error - b77_stability) * 0.10
                + predator_motion * 0.05
                + recent_contact * 0.04
                + balance_context * 0.02,
                0.0,
                1.0,
            )
        )
        balance_lock = int(getattr(self, "_b78_balance_lock", 0))
        decision_label = "preserve_b77"

        if corridor_map:
            if balance_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_balance_lock"
                reason = "b78_continue_balance_lock"
            elif (
                near_shelter > 0.0
                and balance_error >= float(params["b78_hold_threshold"])
                and slip_risk > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                balance_lock = max(balance_lock, int(params["b78_balance_lock_ticks"]))
                decision_label = "vestibular_balance_hold"
                reason = "b78_vestibular_balance_hold"
            elif (
                head_stabilization >= float(params["b78_release_threshold"])
                and locomotor_confidence >= float(params["b78_release_threshold"])
                and balance_error < float(params["b78_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "stabilized_corridor_release"
                reason = "b78_stabilized_corridor_release"
            elif head_stabilization > 0.0 and locomotor_confidence > 0.0:
                if hunger >= 0.68 and near_shelter > 0.0:
                    semantic_action = "MOVE_TO_FOOD"
                decision_label = "balance_recovery_stride"
                reason = "b78_balance_recovery_stride"

        trace_payload.update(
            {
                "b78_controller_profile": profile,
                "b78_balance_error": round(float(balance_error), 6),
                "b78_head_stabilization": round(float(head_stabilization), 6),
                "b78_locomotor_confidence": round(float(locomotor_confidence), 6),
                "b78_slip_risk": round(float(slip_risk), 6),
                "b78_balance_lock": int(balance_lock),
                "b78_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b78_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b78_genetic_candidate"] = int(params["ga_candidate"])

        self._b78_balance_error = float(balance_error)
        self._b78_head_stabilization = float(head_stabilization)
        self._b78_locomotor_confidence = float(locomotor_confidence)
        self._b78_slip_risk = float(slip_risk)
        self._b78_balance_lock = max(0, int(balance_lock) - 1)
        self._b78_last_tick = int(tick)
        return (
            semantic_action,
            B78_VESTIBULAR_BALANCE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
