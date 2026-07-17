from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8kMixin:
    def _b74_controller_params(self) -> dict[str, float]:
        params = self._b73_controller_params()
        defaults = {
            "b74_rebound_decay": 0.90,
            "b74_rebound_gain": 0.34,
            "b74_aftereffect_gain": 0.32,
            "b74_release_gain": 0.30,
            "b74_hold_threshold": 0.18,
            "b74_release_threshold": 0.30,
            "b74_rebound_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "thalamic_rebound_gate"
        )
        if profile == "rebound_release_pacing":
            defaults.update({"b74_release_gain": 0.34, "b74_release_threshold": 0.28})
        elif profile == "post_inhibition_recovery":
            defaults.update({"b74_aftereffect_gain": 0.35, "b74_rebound_lock_ticks": 5.0})
        elif profile == "thalamic_rebound_gate_h56":
            defaults.update({"b74_rebound_decay": 0.92, "b74_rebound_lock_ticks": 5.0})
        elif profile == "genetic_thalamic_rebound":
            defaults.update({"b74_rebound_gain": 0.35, "b74_release_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b74_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b74_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b74_rebound_potential = 0.0
        self._b74_inhibition_aftereffect = 0.0
        self._b74_release_window = 0.0
        self._b74_rebound_lock = 0
        self._b74_last_tick = int(tick)

    def _b74_thalamic_rebound_semantic_action(
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
        ) = self._b73_reticular_inhibition_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b74_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "thalamic_rebound_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b74_reset_state_if_needed(tick)

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
        b72_focus = float(trace_payload.get("b72_focus_signal", 0.0) or 0.0)
        b72_filter = float(trace_payload.get("b72_filter_gain", 0.0) or 0.0)
        b73_tone = float(trace_payload.get("b73_inhibitory_tone", 0.0) or 0.0)
        b73_suppression = float(trace_payload.get("b73_surround_suppression", 0.0) or 0.0)
        b73_release = float(trace_payload.get("b73_release_drive", 0.0) or 0.0)
        b73_lock = int(trace_payload.get("b73_inhibition_lock", 0) or 0)
        rebound_context = 1.0 if (
            b73_lock > 0
            or b73_tone > 0.0
            or b73_suppression > 0.0
            or b73_release > 0.0
        ) else 0.0

        decay = float(params["b74_rebound_decay"])
        previous_rebound = float(getattr(self, "_b74_rebound_potential", 0.0))
        previous_aftereffect = float(getattr(self, "_b74_inhibition_aftereffect", 0.0))
        previous_release = float(getattr(self, "_b74_release_window", 0.0))
        inhibition_aftereffect = float(
            np.clip(
                previous_aftereffect * decay
                + b73_suppression * float(params["b74_aftereffect_gain"])
                + b73_tone * 0.10
                + predator_motion * 0.04
                + predator_smell * 0.03
                + recent_contact * 0.05
                + sleep_debt * 0.02,
                0.0,
                1.0,
            )
        )
        rebound_potential = float(
            np.clip(
                previous_rebound * decay
                + inhibition_aftereffect * float(params["b74_rebound_gain"])
                + b73_release * 0.12
                + b72_focus * 0.05
                + rebound_context * 0.03,
                0.0,
                1.0,
            )
        )
        release_window = float(
            np.clip(
                previous_release * decay
                + rebound_potential * float(params["b74_release_gain"])
                + b73_release * 0.12
                + b72_filter * 0.08
                + max(0.0, b72_focus - inhibition_aftereffect) * 0.08,
                0.0,
                1.0,
            )
        )
        rebound_lock = int(getattr(self, "_b74_rebound_lock", 0))
        decision_label = "preserve_b73"

        if corridor_map:
            if rebound_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_rebound_lock"
                reason = "b74_continue_rebound_lock"
            elif (
                near_shelter > 0.0
                and inhibition_aftereffect >= float(params["b74_hold_threshold"])
                and rebound_potential > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                rebound_lock = max(rebound_lock, int(params["b74_rebound_lock_ticks"]))
                decision_label = "thalamic_rebound_hold"
                reason = "b74_thalamic_rebound_hold"
            elif (
                release_window >= float(params["b74_release_threshold"])
                and b73_release >= float(params["b74_release_threshold"])
                and inhibition_aftereffect < float(params["b74_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "rebound_release_stride"
                reason = "b74_rebound_release_stride"
            elif rebound_context > 0.0 and release_window > 0.0:
                decision_label = "post_inhibition_stride"

        trace_payload.update(
            {
                "b74_controller_profile": profile,
                "b74_rebound_potential": round(float(rebound_potential), 6),
                "b74_inhibition_aftereffect": round(float(inhibition_aftereffect), 6),
                "b74_release_window": round(float(release_window), 6),
                "b74_rebound_lock": int(rebound_lock),
                "b74_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b74_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b74_genetic_candidate"] = int(params["ga_candidate"])

        self._b74_rebound_potential = float(rebound_potential)
        self._b74_inhibition_aftereffect = float(inhibition_aftereffect)
        self._b74_release_window = float(release_window)
        self._b74_rebound_lock = max(0, int(rebound_lock) - 1)
        self._b74_last_tick = int(tick)
        return (
            semantic_action,
            B74_THALAMIC_REBOUND_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
