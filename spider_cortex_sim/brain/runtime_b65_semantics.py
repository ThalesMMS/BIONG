from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart8bMixin:
    def _b65_controller_params(self) -> dict[str, float]:
        params = self._b64_controller_params()
        defaults = {
            "b65_enteric_decay": 0.90,
            "b65_assimilation_gain": 0.34,
            "b65_forage_readiness_gain": 0.30,
            "b65_recovery_coupling_gain": 0.30,
            "b65_digestive_threshold": 0.18,
            "b65_release_threshold": 0.34,
            "b65_digestive_lock_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "enteric_assimilation_gate"
        )
        if profile == "post_recovery_forage_release":
            defaults.update(
                {"b65_forage_readiness_gain": 0.36, "b65_release_threshold": 0.30}
            )
        elif profile == "shelter_digestive_coupling":
            defaults.update(
                {"b65_recovery_coupling_gain": 0.36, "b65_digestive_threshold": 0.16}
            )
        elif profile == "enteric_assimilation_gate_h56":
            defaults.update({"b65_enteric_decay": 0.92, "b65_digestive_lock_ticks": 5.0})
        elif profile == "genetic_enteric_assimilation":
            defaults.update({"b65_assimilation_gain": 0.36, "b65_recovery_coupling_gain": 0.34})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b65_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b65_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b65_enteric_tone = 0.0
        self._b65_assimilation_drive = 0.0
        self._b65_forage_readiness = 0.0
        self._b65_digestive_lock = 0
        self._b65_last_tick = int(tick)

    def _b65_enteric_assimilation_semantic_action(
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
        ) = self._b64_vagal_recovery_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b65_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "enteric_assimilation_gate"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b65_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        shelter_role = str(meta.get("shelter_role", "outside"))
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
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
        b64_decision = str(trace_payload.get("b64_decision", "preserve_b63"))
        b64_recovery_tone = float(trace_payload.get("b64_recovery_tone", 0.0) or 0.0)
        b64_vagal_brake = float(trace_payload.get("b64_vagal_brake", 0.0) or 0.0)
        b64_post_escape_bias = float(
            trace_payload.get("b64_post_escape_bias", 0.0) or 0.0
        )
        b64_lock = int(trace_payload.get("b64_recovery_lock", 0) or 0)
        recovery_context = 1.0 if (
            b64_decision
            in {"vagal_recovery_hold", "continue_recovery_brake", "vagal_safe_release"}
            or b64_lock > 0
            or b64_post_escape_bias > 0.0
        ) else 0.0
        satiety_signal = max(0.0, 1.0 - hunger)
        recovery_need = float(
            np.clip((1.0 - health) * 0.34 + sleep_debt * 0.28 + b64_vagal_brake * 0.20, 0.0, 1.0)
        )

        decay = float(params["b65_enteric_decay"])
        previous_tone = float(getattr(self, "_b65_enteric_tone", 0.0))
        previous_drive = float(getattr(self, "_b65_assimilation_drive", 0.0))
        previous_readiness = float(getattr(self, "_b65_forage_readiness", 0.0))
        enteric_tone = float(
            np.clip(
                previous_tone * decay
                + satiety_signal * 0.14
                + near_shelter * 0.08
                + recovery_context * 0.08,
                0.0,
                1.0,
            )
        )
        assimilation_drive = float(
            np.clip(
                previous_drive * decay
                + enteric_tone * float(params["b65_assimilation_gain"])
                + b64_recovery_tone * float(params["b65_recovery_coupling_gain"])
                + b64_post_escape_bias * 0.12
                + max(0.0, 1.0 - current_threat) * 0.05,
                0.0,
                1.0,
            )
        )
        forage_readiness = float(
            np.clip(
                previous_readiness * decay
                + hunger * float(params["b65_forage_readiness_gain"])
                + health * 0.08
                + max(0.0, 1.0 - sleep_debt) * 0.04
                - assimilation_drive * 0.05,
                0.0,
                1.0,
            )
        )
        digestive_lock = int(getattr(self, "_b65_digestive_lock", 0))
        decision_label = "preserve_b64"

        if corridor_map:
            if digestive_lock > 0 and near_shelter > 0.0 and hunger < 0.90:
                semantic_action = "SLEEP"
                decision_label = "continue_digestive_lock"
                reason = "b65_continue_digestive_lock"
            elif (
                near_shelter > 0.0
                and assimilation_drive >= float(params["b65_digestive_threshold"])
                and hunger < 0.86
                and recovery_context > 0.0
            ):
                semantic_action = "SLEEP"
                digestive_lock = max(digestive_lock, int(params["b65_digestive_lock_ticks"]))
                decision_label = "enteric_digestive_hold"
                reason = "b65_enteric_digestive_hold"
            elif (
                forage_readiness >= float(params["b65_release_threshold"])
                and hunger >= 0.72
                and current_threat < 0.35
            ):
                semantic_action = "MOVE_TO_FOOD"
                decision_label = "enteric_safe_release"
                reason = "b65_enteric_safe_release"
            elif recovery_context > 0.0:
                decision_label = "enteric_safe_release"

        trace_payload.update(
            {
                "b65_controller_profile": profile,
                "b65_enteric_tone": round(float(enteric_tone), 6),
                "b65_assimilation_drive": round(float(assimilation_drive), 6),
                "b65_forage_readiness": round(float(forage_readiness), 6),
                "b65_digestive_lock": int(digestive_lock),
                "b65_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b65_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b65_genetic_candidate"] = int(params["ga_candidate"])

        self._b65_enteric_tone = float(enteric_tone)
        self._b65_assimilation_drive = float(assimilation_drive)
        self._b65_forage_readiness = float(forage_readiness)
        self._b65_digestive_lock = max(0, int(digestive_lock) - 1)
        self._b65_last_tick = int(tick)
        return (
            semantic_action,
            B65_ENTERIC_ASSIMILATION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )
