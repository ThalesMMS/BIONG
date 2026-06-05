from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8hMixin:
    def _b71_controller_params(self) -> dict[str, float]:
        params = self._b70_controller_params()
        defaults = {
            "b71_orienting_decay": 0.90,
            "b71_target_salience_gain": 0.32,
            "b71_orienting_gain": 0.30,
            "b71_collision_veto_gain": 0.34,
            "b71_hold_threshold": 0.18,
            "b71_release_threshold": 0.30,
            "b71_orienting_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "tectal_orienting_gate"
        )
        if profile == "salience_map_pacing":
            defaults.update({"b71_target_salience_gain": 0.35, "b71_release_threshold": 0.28})
        elif profile == "collision_veto_recovery":
            defaults.update({"b71_collision_veto_gain": 0.32, "b71_hold_threshold": 0.16})
        elif profile == "tectal_orienting_gate_h56":
            defaults.update({"b71_orienting_decay": 0.92, "b71_orienting_lock_ticks": 5.0})
        elif profile == "genetic_tectal_orienting":
            defaults.update({"b71_target_salience_gain": 0.34, "b71_orienting_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b71_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b71_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b71_target_salience = 0.0
        self._b71_orienting_gain = 0.0
        self._b71_collision_veto = 0.0
        self._b71_orienting_lock = 0
        self._b71_last_tick = int(tick)

    def _b71_tectal_orienting_semantic_action(
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
        ) = self._b70_optic_flow_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b71_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "tectal_orienting_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b71_reset_state_if_needed(tick)

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
        b70_flow = float(trace_payload.get("b70_flow_confidence", 0.0) or 0.0)
        b70_drift = float(trace_payload.get("b70_lateral_drift", 0.0) or 0.0)
        b70_looming = float(trace_payload.get("b70_looming_risk", 0.0) or 0.0)
        b70_lock = int(trace_payload.get("b70_flow_lock", 0) or 0)
        orienting_context = 1.0 if (
            b70_lock > 0
            or b70_flow > 0.0
            or b70_looming > 0.0
        ) else 0.0

        decay = float(params["b71_orienting_decay"])
        previous_salience = float(getattr(self, "_b71_target_salience", 0.0))
        previous_gain = float(getattr(self, "_b71_orienting_gain", 0.0))
        previous_veto = float(getattr(self, "_b71_collision_veto", 0.0))
        target_salience = float(
            np.clip(
                previous_salience * decay
                + b70_flow * float(params["b71_target_salience_gain"])
                + max(0.0, 1.0 - b70_drift) * 0.04
                + orienting_context * 0.04,
                0.0,
                1.0,
            )
        )
        orienting_gain = float(
            np.clip(
                previous_gain * decay
                + target_salience * float(params["b71_orienting_gain"])
                + b70_flow * 0.06
                + near_shelter * 0.02,
                0.0,
                1.0,
            )
        )
        collision_veto = float(
            np.clip(
                previous_veto * decay
                + b70_looming * float(params["b71_collision_veto_gain"])
                + b70_drift * 0.06
                + predator_motion * 0.06
                + predator_smell * 0.04
                + recent_contact * 0.08
                + sleep_debt * 0.03
                - orienting_gain * 0.03,
                0.0,
                1.0,
            )
        )
        orienting_lock = int(getattr(self, "_b71_orienting_lock", 0))
        decision_label = "preserve_b70"

        if corridor_map:
            if orienting_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_orienting_lock"
                reason = "b71_continue_orienting_lock"
            elif (
                near_shelter > 0.0
                and collision_veto >= float(params["b71_hold_threshold"])
                and target_salience > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                orienting_lock = max(orienting_lock, int(params["b71_orienting_lock_ticks"]))
                decision_label = "tectal_recenter"
                reason = "b71_tectal_recenter"
            elif (
                target_salience >= float(params["b71_release_threshold"])
                and orienting_gain >= float(params["b71_release_threshold"])
                and collision_veto < float(params["b71_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "salient_target_stride"
                reason = "b71_salient_target_stride"
            elif orienting_context > 0.0:
                decision_label = "salient_target_stride"

        trace_payload.update(
            {
                "b71_controller_profile": profile,
                "b71_target_salience": round(float(target_salience), 6),
                "b71_orienting_gain": round(float(orienting_gain), 6),
                "b71_collision_veto": round(float(collision_veto), 6),
                "b71_orienting_lock": int(orienting_lock),
                "b71_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b71_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b71_genetic_candidate"] = int(params["ga_candidate"])

        self._b71_target_salience = float(target_salience)
        self._b71_orienting_gain = float(orienting_gain)
        self._b71_collision_veto = float(collision_veto)
        self._b71_orienting_lock = max(0, int(orienting_lock) - 1)
        self._b71_last_tick = int(tick)
        return (
            semantic_action,
            B71_TECTAL_ORIENTING_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
