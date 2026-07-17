from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8lMixin:
    def _b75_controller_params(self) -> dict[str, float]:
        params = self._b74_controller_params()
        defaults = {
            "b75_timing_decay": 0.90,
            "b75_go_gain": 0.34,
            "b75_nogo_gain": 0.32,
            "b75_burst_gain": 0.30,
            "b75_hold_threshold": 0.18,
            "b75_release_threshold": 0.30,
            "b75_release_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "basal_thalamic_release"
        )
        if profile == "release_burst_pacing":
            defaults.update({"b75_burst_gain": 0.34, "b75_release_threshold": 0.28})
        elif profile == "go_nogo_rebound_timing":
            defaults.update(
                {"b75_go_gain": 0.35, "b75_nogo_gain": 0.35, "b75_release_lock_ticks": 5.0}
            )
        elif profile == "basal_thalamic_release_h56":
            defaults.update({"b75_timing_decay": 0.92, "b75_release_lock_ticks": 5.0})
        elif profile == "genetic_basal_thalamic_release":
            defaults.update({"b75_go_gain": 0.35, "b75_burst_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b75_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b75_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b75_release_timing_drive = 0.0
        self._b75_go_disinhibition = 0.0
        self._b75_nogo_brake = 0.0
        self._b75_burst_window = 0.0
        self._b75_release_lock = 0
        self._b75_last_tick = int(tick)

    def _b75_basal_thalamic_release_semantic_action(
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
        ) = self._b74_thalamic_rebound_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b75_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "basal_thalamic_release"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b75_reset_state_if_needed(tick)

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
        b73_tone = float(trace_payload.get("b73_inhibitory_tone", 0.0) or 0.0)
        b73_release = float(trace_payload.get("b73_release_drive", 0.0) or 0.0)
        b74_rebound = float(trace_payload.get("b74_rebound_potential", 0.0) or 0.0)
        b74_aftereffect = float(
            trace_payload.get("b74_inhibition_aftereffect", 0.0) or 0.0
        )
        b74_release = float(trace_payload.get("b74_release_window", 0.0) or 0.0)
        b74_lock = int(trace_payload.get("b74_rebound_lock", 0) or 0)
        release_context = 1.0 if (
            b74_lock > 0
            or b74_rebound > 0.0
            or b74_aftereffect > 0.0
            or b74_release > 0.0
        ) else 0.0

        decay = float(params["b75_timing_decay"])
        previous_timing = float(getattr(self, "_b75_release_timing_drive", 0.0))
        previous_go = float(getattr(self, "_b75_go_disinhibition", 0.0))
        previous_nogo = float(getattr(self, "_b75_nogo_brake", 0.0))
        previous_burst = float(getattr(self, "_b75_burst_window", 0.0))
        release_timing_drive = float(
            np.clip(
                previous_timing * decay
                + b74_release * 0.22
                + b74_rebound * 0.18
                + max(0.0, b74_rebound - b74_aftereffect) * 0.10
                + b73_release * 0.08
                + release_context * 0.03,
                0.0,
                1.0,
            )
        )
        go_disinhibition = float(
            np.clip(
                previous_go * decay
                + release_timing_drive * float(params["b75_go_gain"])
                + b74_release * 0.12
                + max(0.0, hunger - 0.50) * 0.05,
                0.0,
                1.0,
            )
        )
        nogo_brake = float(
            np.clip(
                previous_nogo * decay
                + b74_aftereffect * float(params["b75_nogo_gain"])
                + b73_tone * 0.07
                + predator_motion * 0.05
                + predator_smell * 0.04
                + recent_contact * 0.05
                + sleep_debt * 0.03,
                0.0,
                1.0,
            )
        )
        burst_window = float(
            np.clip(
                previous_burst * decay
                + go_disinhibition * float(params["b75_burst_gain"])
                + release_timing_drive * 0.10
                + max(0.0, go_disinhibition - nogo_brake) * 0.08
                + release_context * 0.03,
                0.0,
                1.0,
            )
        )
        release_lock = int(getattr(self, "_b75_release_lock", 0))
        decision_label = "preserve_b74"

        if corridor_map:
            if release_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_release_lock"
                reason = "b75_continue_release_lock"
            elif (
                near_shelter > 0.0
                and nogo_brake >= float(params["b75_hold_threshold"])
                and release_timing_drive > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                release_lock = max(release_lock, int(params["b75_release_lock_ticks"]))
                decision_label = "basal_thalamic_hold"
                reason = "b75_basal_thalamic_hold"
            elif (
                burst_window >= float(params["b75_release_threshold"])
                and go_disinhibition >= float(params["b75_release_threshold"])
                and nogo_brake < float(params["b75_hold_threshold"])
                and hunger >= 0.72
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "timed_rebound_stride"
                reason = "b75_timed_rebound_stride"
            elif go_disinhibition > 0.0 and release_timing_drive > 0.0:
                if hunger >= 0.68 and near_shelter > 0.0:
                    semantic_action = "MOVE_TO_FOOD"
                decision_label = "go_disinhibition_stride"
                reason = "b75_go_disinhibition_stride"

        trace_payload.update(
            {
                "b75_controller_profile": profile,
                "b75_release_timing_drive": round(float(release_timing_drive), 6),
                "b75_go_disinhibition": round(float(go_disinhibition), 6),
                "b75_nogo_brake": round(float(nogo_brake), 6),
                "b75_burst_window": round(float(burst_window), 6),
                "b75_release_lock": int(release_lock),
                "b75_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b75_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b75_genetic_candidate"] = int(params["ga_candidate"])

        self._b75_release_timing_drive = float(release_timing_drive)
        self._b75_go_disinhibition = float(go_disinhibition)
        self._b75_nogo_brake = float(nogo_brake)
        self._b75_burst_window = float(burst_window)
        self._b75_release_lock = max(0, int(release_lock) - 1)
        self._b75_last_tick = int(tick)
        return (
            semantic_action,
            B75_BASAL_THALAMIC_RELEASE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
