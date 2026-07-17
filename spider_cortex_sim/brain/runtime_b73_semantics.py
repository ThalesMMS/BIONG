from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8jMixin:
    def _b73_controller_params(self) -> dict[str, float]:
        params = self._b72_controller_params()
        defaults = {
            "b73_inhibition_decay": 0.90,
            "b73_tone_gain": 0.34,
            "b73_suppression_gain": 0.32,
            "b73_release_gain": 0.30,
            "b73_hold_threshold": 0.18,
            "b73_release_threshold": 0.30,
            "b73_inhibition_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "reticular_inhibition_gate"
        )
        if profile == "surround_suppression_pacing":
            defaults.update({"b73_suppression_gain": 0.35, "b73_hold_threshold": 0.16})
        elif profile == "focus_release_recovery":
            defaults.update({"b73_release_gain": 0.34, "b73_inhibition_lock_ticks": 5.0})
        elif profile == "reticular_inhibition_gate_h56":
            defaults.update({"b73_inhibition_decay": 0.92, "b73_inhibition_lock_ticks": 5.0})
        elif profile == "genetic_reticular_inhibition":
            defaults.update({"b73_tone_gain": 0.35, "b73_release_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b73_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b73_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b73_inhibitory_tone = 0.0
        self._b73_surround_suppression = 0.0
        self._b73_release_drive = 0.0
        self._b73_inhibition_lock = 0
        self._b73_last_tick = int(tick)

    def _b73_reticular_inhibition_semantic_action(
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
        ) = self._b72_pulvinar_attention_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b73_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "reticular_inhibition_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b73_reset_state_if_needed(tick)

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
        b70_drift = float(trace_payload.get("b70_lateral_drift", 0.0) or 0.0)
        b70_looming = float(trace_payload.get("b70_looming_risk", 0.0) or 0.0)
        b72_focus = float(trace_payload.get("b72_focus_signal", 0.0) or 0.0)
        b72_distractor = float(trace_payload.get("b72_distractor_load", 0.0) or 0.0)
        b72_filter = float(trace_payload.get("b72_filter_gain", 0.0) or 0.0)
        b72_lock = int(trace_payload.get("b72_attention_lock", 0) or 0)
        inhibition_context = 1.0 if (
            b72_lock > 0
            or b72_focus > 0.0
            or b72_distractor > 0.0
            or b72_filter > 0.0
        ) else 0.0

        decay = float(params["b73_inhibition_decay"])
        previous_tone = float(getattr(self, "_b73_inhibitory_tone", 0.0))
        previous_suppression = float(getattr(self, "_b73_surround_suppression", 0.0))
        previous_release = float(getattr(self, "_b73_release_drive", 0.0))
        inhibitory_tone = float(
            np.clip(
                previous_tone * decay
                + b72_distractor * float(params["b73_tone_gain"])
                + b70_looming * 0.06
                + b72_focus * 0.03
                + inhibition_context * 0.03,
                0.0,
                1.0,
            )
        )
        surround_suppression = float(
            np.clip(
                previous_suppression * decay
                + inhibitory_tone * float(params["b73_suppression_gain"])
                + b72_distractor * 0.08
                + b70_drift * 0.04
                + predator_motion * 0.05
                + predator_smell * 0.04
                + recent_contact * 0.06
                + sleep_debt * 0.02,
                0.0,
                1.0,
            )
        )
        release_drive = float(
            np.clip(
                previous_release * decay
                + b72_focus * float(params["b73_release_gain"])
                + b72_filter * 0.12
                + max(0.0, b72_focus - surround_suppression) * 0.10
                + inhibition_context * 0.03,
                0.0,
                1.0,
            )
        )
        inhibition_lock = int(getattr(self, "_b73_inhibition_lock", 0))
        decision_label = "preserve_b72"

        if corridor_map:
            if inhibition_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_inhibition_lock"
                reason = "b73_continue_inhibition_lock"
            elif (
                near_shelter > 0.0
                and surround_suppression >= float(params["b73_hold_threshold"])
                and inhibitory_tone > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                inhibition_lock = max(inhibition_lock, int(params["b73_inhibition_lock_ticks"]))
                decision_label = "reticular_surround_hold"
                reason = "b73_reticular_surround_hold"
            elif (
                release_drive >= float(params["b73_release_threshold"])
                and b72_filter >= float(params["b73_release_threshold"])
                and surround_suppression < float(params["b73_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "reticular_focus_release"
                reason = "b73_reticular_focus_release"
            elif inhibition_context > 0.0 and release_drive > 0.0:
                decision_label = "suppress_distractor_stride"

        trace_payload.update(
            {
                "b73_controller_profile": profile,
                "b73_inhibitory_tone": round(float(inhibitory_tone), 6),
                "b73_surround_suppression": round(float(surround_suppression), 6),
                "b73_release_drive": round(float(release_drive), 6),
                "b73_inhibition_lock": int(inhibition_lock),
                "b73_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b73_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b73_genetic_candidate"] = int(params["ga_candidate"])

        self._b73_inhibitory_tone = float(inhibitory_tone)
        self._b73_surround_suppression = float(surround_suppression)
        self._b73_release_drive = float(release_drive)
        self._b73_inhibition_lock = max(0, int(inhibition_lock) - 1)
        self._b73_last_tick = int(tick)
        return (
            semantic_action,
            B73_RETICULAR_INHIBITION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
