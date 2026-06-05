from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8iMixin:
    def _b72_controller_params(self) -> dict[str, float]:
        params = self._b71_controller_params()
        defaults = {
            "b72_attention_decay": 0.90,
            "b72_focus_gain": 0.34,
            "b72_distractor_gain": 0.32,
            "b72_filter_gain": 0.30,
            "b72_hold_threshold": 0.18,
            "b72_release_threshold": 0.30,
            "b72_attention_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "pulvinar_attention_gate"
        )
        if profile == "distractor_filter_pacing":
            defaults.update({"b72_distractor_gain": 0.35, "b72_hold_threshold": 0.16})
        elif profile == "focus_lock_recovery":
            defaults.update({"b72_focus_gain": 0.36, "b72_attention_lock_ticks": 5.0})
        elif profile == "pulvinar_attention_gate_h56":
            defaults.update({"b72_attention_decay": 0.92, "b72_attention_lock_ticks": 5.0})
        elif profile == "genetic_pulvinar_attention":
            defaults.update({"b72_focus_gain": 0.35, "b72_filter_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b72_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b72_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b72_focus_signal = 0.0
        self._b72_distractor_load = 0.0
        self._b72_filter_gain = 0.0
        self._b72_attention_lock = 0
        self._b72_last_tick = int(tick)

    def _b72_pulvinar_attention_semantic_action(
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
        ) = self._b71_tectal_orienting_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b72_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "pulvinar_attention_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b72_reset_state_if_needed(tick)

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
        b71_target = float(trace_payload.get("b71_target_salience", 0.0) or 0.0)
        b71_gain = float(trace_payload.get("b71_orienting_gain", 0.0) or 0.0)
        b71_veto = float(trace_payload.get("b71_collision_veto", 0.0) or 0.0)
        b71_lock = int(trace_payload.get("b71_orienting_lock", 0) or 0)
        attention_context = 1.0 if (
            b71_lock > 0
            or b71_target > 0.0
            or b71_gain > 0.0
            or b70_flow > 0.0
        ) else 0.0

        decay = float(params["b72_attention_decay"])
        previous_focus = float(getattr(self, "_b72_focus_signal", 0.0))
        previous_distractor = float(getattr(self, "_b72_distractor_load", 0.0))
        previous_filter = float(getattr(self, "_b72_filter_gain", 0.0))
        focus_signal = float(
            np.clip(
                previous_focus * decay
                + b71_target * float(params["b72_focus_gain"])
                + b71_gain * 0.14
                + b70_flow * 0.05
                + max(0.0, 1.0 - b70_drift) * 0.03
                + attention_context * 0.03,
                0.0,
                1.0,
            )
        )
        distractor_load = float(
            np.clip(
                previous_distractor * decay
                + b71_veto * float(params["b72_distractor_gain"])
                + b70_looming * 0.08
                + b70_drift * 0.05
                + predator_motion * 0.05
                + predator_smell * 0.04
                + recent_contact * 0.08
                + sleep_debt * 0.02,
                0.0,
                1.0,
            )
        )
        filter_gain = float(
            np.clip(
                previous_filter * decay
                + focus_signal * float(params["b72_filter_gain"])
                + max(0.0, focus_signal - distractor_load) * 0.12
                + attention_context * 0.04,
                0.0,
                1.0,
            )
        )
        attention_lock = int(getattr(self, "_b72_attention_lock", 0))
        decision_label = "preserve_b71"

        if corridor_map:
            if attention_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_attention_lock"
                reason = "b72_continue_attention_lock"
            elif (
                near_shelter > 0.0
                and distractor_load >= float(params["b72_hold_threshold"])
                and focus_signal > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                attention_lock = max(attention_lock, int(params["b72_attention_lock_ticks"]))
                decision_label = "pulvinar_filter_recenter"
                reason = "b72_pulvinar_filter_recenter"
            elif (
                focus_signal >= float(params["b72_release_threshold"])
                and filter_gain >= float(params["b72_release_threshold"])
                and distractor_load < float(params["b72_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "filtered_target_stride"
                reason = "b72_filtered_target_stride"
            elif attention_context > 0.0 and filter_gain > 0.0:
                decision_label = "selective_focus_stride"

        trace_payload.update(
            {
                "b72_controller_profile": profile,
                "b72_focus_signal": round(float(focus_signal), 6),
                "b72_distractor_load": round(float(distractor_load), 6),
                "b72_filter_gain": round(float(filter_gain), 6),
                "b72_attention_lock": int(attention_lock),
                "b72_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b72_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b72_genetic_candidate"] = int(params["ga_candidate"])

        self._b72_focus_signal = float(focus_signal)
        self._b72_distractor_load = float(distractor_load)
        self._b72_filter_gain = float(filter_gain)
        self._b72_attention_lock = max(0, int(attention_lock) - 1)
        self._b72_last_tick = int(tick)
        return (
            semantic_action,
            B72_PULVINAR_ATTENTION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
