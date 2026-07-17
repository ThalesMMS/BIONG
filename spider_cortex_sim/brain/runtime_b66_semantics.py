from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8cMixin:
    def _b66_controller_params(self) -> dict[str, float]:
        params = self._b65_controller_params()
        defaults = {
            "b66_immune_decay": 0.90,
            "b66_immune_tone_gain": 0.34,
            "b66_malaise_gain": 0.32,
            "b66_recovery_veto_gain": 0.30,
            "b66_hold_threshold": 0.18,
            "b66_release_threshold": 0.30,
            "b66_inflammation_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "immune_malaise_gate"
        )
        if profile == "inflammatory_recovery_hold":
            defaults.update({"b66_malaise_gain": 0.36, "b66_hold_threshold": 0.16})
        elif profile == "damage_aware_forage_release":
            defaults.update({"b66_recovery_veto_gain": 0.26, "b66_release_threshold": 0.26})
        elif profile == "immune_malaise_gate_h56":
            defaults.update({"b66_immune_decay": 0.92, "b66_inflammation_lock_ticks": 5.0})
        elif profile == "genetic_immune_malaise":
            defaults.update({"b66_immune_tone_gain": 0.36, "b66_malaise_gain": 0.34})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b66_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b66_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b66_immune_tone = 0.0
        self._b66_malaise_drive = 0.0
        self._b66_recovery_veto = 0.0
        self._b66_inflammation_lock = 0
        self._b66_last_tick = int(tick)

    def _b66_immune_malaise_semantic_action(
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
        ) = self._b65_enteric_assimilation_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b66_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "immune_malaise_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b66_reset_state_if_needed(tick)

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
        current_threat = max(
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
        )
        damage_signal = max(
            self._b_series_float(meta, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
            max(0.0, 1.0 - health),
        )
        b65_decision = str(trace_payload.get("b65_decision", "preserve_b64"))
        b65_assimilation = float(trace_payload.get("b65_assimilation_drive", 0.0) or 0.0)
        b65_enteric_tone = float(trace_payload.get("b65_enteric_tone", 0.0) or 0.0)
        b65_readiness = float(trace_payload.get("b65_forage_readiness", 0.0) or 0.0)
        b65_lock = int(trace_payload.get("b65_digestive_lock", 0) or 0)
        recovery_context = 1.0 if (
            b65_decision
            in {"enteric_digestive_hold", "continue_digestive_lock", "enteric_safe_release"}
            or b65_lock > 0
            or b65_assimilation > 0.0
        ) else 0.0

        decay = float(params["b66_immune_decay"])
        previous_tone = float(getattr(self, "_b66_immune_tone", 0.0))
        previous_malaise = float(getattr(self, "_b66_malaise_drive", 0.0))
        previous_veto = float(getattr(self, "_b66_recovery_veto", 0.0))
        immune_tone = float(
            np.clip(
                previous_tone * decay
                + damage_signal * float(params["b66_immune_tone_gain"])
                + current_threat * 0.08
                + recovery_context * 0.05,
                0.0,
                1.0,
            )
        )
        malaise_drive = float(
            np.clip(
                previous_malaise * decay
                + immune_tone * float(params["b66_malaise_gain"])
                + sleep_debt * 0.08
                + b65_assimilation * 0.08,
                0.0,
                1.0,
            )
        )
        recovery_veto = float(
            np.clip(
                previous_veto * decay
                + malaise_drive * float(params["b66_recovery_veto_gain"])
                + b65_enteric_tone * 0.06
                - b65_readiness * 0.04,
                0.0,
                1.0,
            )
        )
        inflammation_lock = int(getattr(self, "_b66_inflammation_lock", 0))
        decision_label = "preserve_b65"

        if corridor_map:
            if inflammation_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_inflammation_lock"
                reason = "b66_continue_inflammation_lock"
            elif (
                near_shelter > 0.0
                and recovery_veto >= float(params["b66_hold_threshold"])
                and hunger < 0.88
                and recovery_context > 0.0
            ):
                semantic_action = "SLEEP"
                inflammation_lock = max(
                    inflammation_lock,
                    int(params["b66_inflammation_lock_ticks"]),
                )
                decision_label = "immune_recovery_hold"
                reason = "b66_immune_recovery_hold"
            elif (
                b65_readiness >= float(params["b66_release_threshold"])
                and recovery_veto < float(params["b66_hold_threshold"])
                and hunger >= 0.72
                and current_threat < 0.35
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "immune_safe_release"
                reason = "b66_immune_safe_release"
            elif recovery_context > 0.0:
                decision_label = "immune_safe_release"

        trace_payload.update(
            {
                "b66_controller_profile": profile,
                "b66_immune_tone": round(float(immune_tone), 6),
                "b66_malaise_drive": round(float(malaise_drive), 6),
                "b66_recovery_veto": round(float(recovery_veto), 6),
                "b66_inflammation_lock": int(inflammation_lock),
                "b66_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b66_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b66_genetic_candidate"] = int(params["ga_candidate"])

        self._b66_immune_tone = float(immune_tone)
        self._b66_malaise_drive = float(malaise_drive)
        self._b66_recovery_veto = float(recovery_veto)
        self._b66_inflammation_lock = max(0, int(inflammation_lock) - 1)
        self._b66_last_tick = int(tick)
        return (
            semantic_action,
            B66_IMMUNE_MALAISE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
