from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8dMixin:
    def _b67_controller_params(self) -> dict[str, float]:
        params = self._b66_controller_params()
        defaults = {
            "b67_glial_decay": 0.90,
            "b67_energy_reserve_gain": 0.32,
            "b67_lactate_support_gain": 0.30,
            "b67_fatigue_gain": 0.34,
            "b67_hold_threshold": 0.18,
            "b67_release_threshold": 0.30,
            "b67_recovery_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "glial_energy_gate"
        )
        if profile == "lactate_recovery_support":
            defaults.update({"b67_lactate_support_gain": 0.34, "b67_hold_threshold": 0.16})
        elif profile == "fatigue_aware_release":
            defaults.update({"b67_fatigue_gain": 0.30, "b67_release_threshold": 0.26})
        elif profile == "glial_energy_gate_h56":
            defaults.update({"b67_glial_decay": 0.92, "b67_recovery_lock_ticks": 5.0})
        elif profile == "genetic_glial_energy":
            defaults.update({"b67_energy_reserve_gain": 0.34, "b67_fatigue_gain": 0.32})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b67_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b67_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b67_glial_reserve = 0.0
        self._b67_lactate_support = 0.0
        self._b67_fatigue_pressure = 0.0
        self._b67_recovery_lock = 0
        self._b67_last_tick = int(tick)

    def _b67_glial_energy_semantic_action(
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
        ) = self._b66_immune_malaise_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b67_controller_params()
        profile = str(getattr(self.config, "b_controller_profile", None) or "glial_energy_gate")
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b67_reset_state_if_needed(tick)

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
            if shelter_dist <= 1.0 or shelter_role in {"at_shelter", "deep_shelter"}
            else 0.0
        )
        b65_readiness = float(trace_payload.get("b65_forage_readiness", 0.0) or 0.0)
        b65_enteric_tone = float(trace_payload.get("b65_enteric_tone", 0.0) or 0.0)
        b66_immune_tone = float(trace_payload.get("b66_immune_tone", 0.0) or 0.0)
        b66_malaise = float(trace_payload.get("b66_malaise_drive", 0.0) or 0.0)
        b66_veto = float(trace_payload.get("b66_recovery_veto", 0.0) or 0.0)
        b66_lock = int(trace_payload.get("b66_inflammation_lock", 0) or 0)
        immune_context = 1.0 if (
            b66_lock > 0
            or b66_immune_tone > 0.0
            or b66_malaise > 0.0
        ) else 0.0

        decay = float(params["b67_glial_decay"])
        previous_reserve = float(getattr(self, "_b67_glial_reserve", 0.0))
        previous_lactate = float(getattr(self, "_b67_lactate_support", 0.0))
        previous_fatigue = float(getattr(self, "_b67_fatigue_pressure", 0.0))
        glial_reserve = float(
            np.clip(
                previous_reserve * decay
                + max(0.0, health) * float(params["b67_energy_reserve_gain"])
                + b65_readiness * 0.08
                + near_shelter * 0.03
                - sleep_debt * 0.04
                - b66_malaise * 0.03,
                0.0,
                1.0,
            )
        )
        lactate_support = float(
            np.clip(
                previous_lactate * decay
                + b65_enteric_tone * 0.08
                + b66_immune_tone * float(params["b67_lactate_support_gain"])
                + immune_context * 0.05,
                0.0,
                1.0,
            )
        )
        fatigue_pressure = float(
            np.clip(
                previous_fatigue * decay
                + b66_malaise * float(params["b67_fatigue_gain"])
                + b66_veto * 0.08
                + sleep_debt * 0.08
                + max(0.0, 1.0 - health) * 0.10
                - glial_reserve * 0.03,
                0.0,
                1.0,
            )
        )
        recovery_lock = int(getattr(self, "_b67_recovery_lock", 0))
        decision_label = "preserve_b66"

        if corridor_map:
            if recovery_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_glial_recovery_lock"
                reason = "b67_continue_glial_recovery_lock"
            elif (
                near_shelter > 0.0
                and fatigue_pressure >= float(params["b67_hold_threshold"])
                and lactate_support > 0.0
                and hunger < 0.88
            ):
                semantic_action = "SLEEP"
                recovery_lock = max(
                    recovery_lock,
                    int(params["b67_recovery_lock_ticks"]),
                )
                decision_label = "glial_recovery_support"
                reason = "b67_glial_recovery_support"
            elif (
                glial_reserve >= float(params["b67_release_threshold"])
                and fatigue_pressure < float(params["b67_hold_threshold"])
                and hunger >= 0.72
                and b66_malaise < 0.30
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "glial_energy_release"
                reason = "b67_glial_energy_release"
            elif immune_context > 0.0:
                decision_label = "glial_energy_release"

        trace_payload.update(
            {
                "b67_controller_profile": profile,
                "b67_glial_reserve": round(float(glial_reserve), 6),
                "b67_lactate_support": round(float(lactate_support), 6),
                "b67_fatigue_pressure": round(float(fatigue_pressure), 6),
                "b67_recovery_lock": int(recovery_lock),
                "b67_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b67_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b67_genetic_candidate"] = int(params["ga_candidate"])

        self._b67_glial_reserve = float(glial_reserve)
        self._b67_lactate_support = float(lactate_support)
        self._b67_fatigue_pressure = float(fatigue_pressure)
        self._b67_recovery_lock = max(0, int(recovery_lock) - 1)
        self._b67_last_tick = int(tick)
        return (
            semantic_action,
            B67_GLIAL_ENERGY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
