from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart1Mixin:
    @staticmethod
    def _b_series_float(mapping: MappingProxyType | Dict[str, float], key: str) -> float:
        try:
            value = float(mapping.get(key, 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    def _b0_current_simple_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int]:
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        hunger = self._b_series_float(hunger_obs, "hunger")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        health = self._b_series_float(sleep_obs, "health")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        on_shelter = self._b_series_float(sleep_obs, "on_shelter") > 0.5 or bool(
            meta.get("on_shelter", False)
        )
        night = self._b_series_float(sleep_obs, "night") > 0.5 or bool(
            meta.get("night", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        shelter_role_level = max(
            self._b_series_float(sleep_obs, "shelter_role_level"),
            self._b_series_float(meta, "shelter_role_level"),
        )
        food_memory_signal = (
            1.0 - self._b_series_float(hunger_obs, "food_memory_age")
            if (
                abs(float(hunger_obs.get("food_memory_dx", 0.0)))
                + abs(float(hunger_obs.get("food_memory_dy", 0.0)))
            )
            > 0.05
            else 0.0
        )
        food_signal = max(
            self._b_series_float(hunger_obs, "food_visible"),
            self._b_series_float(hunger_obs, "food_certainty"),
            self._b_series_float(hunger_obs, "food_smell_strength"),
            self._b_series_float(hunger_obs, "food_trace_strength"),
            food_memory_signal,
        )
        acute_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
        )
        threat_pressure = max(
            acute_threat,
            self._b_series_float(threat_obs, "predator_smell_strength"),
        )

        if on_food and hunger >= 0.10:
            semantic_action = "EAT"
            reason = "b0_current_eat_on_food"
        elif on_shelter:
            rest_pressure = bool(night or fatigue >= 0.25 or sleep_debt >= 0.25)
            if health <= 0.65 and hunger < 0.55:
                if shelter_role_level < 0.75:
                    semantic_action = "MOVE_TO_SHELTER"
                    reason = "b0_current_low_health_deepen"
                elif rest_pressure:
                    semantic_action = "SLEEP"
                    reason = "b0_current_low_health_rest"
                else:
                    semantic_action = "STAY"
                    reason = "b0_current_low_health_hold"
            elif threat_pressure >= 0.55 and hunger < 0.48:
                if shelter_role_level < 0.75:
                    semantic_action = "MOVE_TO_SHELTER"
                    reason = "b0_current_threat_hold_deepen"
                elif rest_pressure:
                    semantic_action = "SLEEP"
                    reason = "b0_current_threat_hold_rest"
                else:
                    semantic_action = "STAY"
                    reason = "b0_current_threat_hold_shelter"
            elif rest_pressure and hunger < 0.55:
                if shelter_role_level < 0.75:
                    semantic_action = "MOVE_TO_SHELTER"
                    reason = "b0_current_deepen_before_rest"
                else:
                    semantic_action = "SLEEP"
                    reason = "b0_current_rest_in_shelter"
            elif hunger >= 0.50 or (food_signal >= 0.35 and not rest_pressure):
                semantic_action = "MOVE_TO_FOOD"
                reason = "b0_current_forage_from_shelter"
            else:
                semantic_action = "STAY"
                reason = "b0_current_shelter_hold"
        elif (
            (hunger < 0.40 and (night or fatigue >= 0.25 or sleep_debt >= 0.25))
            or acute_threat >= 0.85
            or (threat_pressure >= 0.55 and hunger < 0.55)
            or (health <= 0.65 and hunger < 0.55)
            or health <= 0.35
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b0_current_recover_return"
        elif hunger >= 0.45 or food_signal >= 0.15:
            semantic_action = "MOVE_TO_FOOD"
            reason = "b0_current_forage"
        elif night or fatigue >= 0.52 or sleep_debt >= 0.52:
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b0_current_rest_return"
        else:
            semantic_action = "EXPLORE"
            reason = "b0_current_explore"

        return (
            semantic_action,
            B_CURRENT_BRIDGE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
        )

    def _b1_threat_guard_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
        ) = self._b0_current_simple_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        hunger = self._b_series_float(hunger_obs, "hunger")
        on_shelter = self._b_series_float(sleep_obs, "on_shelter") > 0.5 or bool(
            meta.get("on_shelter", False)
        )
        acute_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
        )
        threat_pressure = max(
            acute_threat,
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(meta, "predator_smell_strength"),
        )
        if not on_shelter and threat_pressure >= 0.60 and hunger < 0.80:
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b1_threat_guard_return_under_threat"
        else:
            reason = f"b1_threat_guard_{reason}"
        return (
            semantic_action,
            B1_THREAT_GUARD_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
        )

    def _b2_temporal_threat_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
        ) = self._b1_threat_guard_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        def _mapping(value: object) -> dict[str, object]:
            return value if isinstance(value, dict) else {}

        def _signed_float(value: object) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return 0.0
            return numeric if np.isfinite(numeric) else 0.0

        hunger = self._b_series_float(hunger_obs, "hunger")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        health = self._b_series_float(sleep_obs, "health")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or shelter_role != "outside"
        )
        night = self._b_series_float(sleep_obs, "night") > 0.5 or bool(
            meta.get("night", False)
        )
        shelter_role_level = max(
            self._b_series_float(sleep_obs, "shelter_role_level"),
            self._b_series_float(meta, "shelter_role_level"),
        )
        rest_pressure = bool(night or fatigue >= 0.25 or sleep_debt >= 0.25)
        recent_contact_pressure = max(
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "recent_contact"),
        )
        current_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "predator_smell_strength"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
        )
        memory_vectors = _mapping(meta.get("memory_vectors"))
        predator_memory = _mapping(memory_vectors.get("predator"))
        predator_memory_pressure = (
            1.0 - self._b_series_float(predator_memory, "age")
            if abs(_signed_float(predator_memory.get("dx", 0.0)))
            + abs(_signed_float(predator_memory.get("dy", 0.0)))
            > 0.05
            else 0.0
        )
        predator_trace = _mapping(_mapping(meta.get("percept_traces")).get("predator"))
        predator_trace_pressure = max(
            self._b_series_float(predator_trace, "strength"),
            self._b_series_float(predator_trace, "freshness"),
            self._b_series_float(predator_trace, "certainty"),
        )
        temporal_threat = max(
            current_threat,
            0.85 * predator_memory_pressure,
            0.75 * predator_trace_pressure,
        )

        if on_food and hunger >= 0.10:
            semantic_action = "EAT"
            reason = "b2_temporal_threat_eat_on_food"
        elif on_shelter and health < 0.45 and hunger < 0.65:
            if shelter_role != "deep" or shelter_role_level < 0.95:
                semantic_action = "MOVE_TO_SHELTER"
                reason = "b2_temporal_threat_low_health_deepen"
            elif rest_pressure or fatigue >= 0.15 or sleep_debt >= 0.15:
                semantic_action = "SLEEP"
                reason = "b2_temporal_threat_low_health_rest"
            else:
                semantic_action = "STAY"
                reason = "b2_temporal_threat_low_health_hold"
        elif on_shelter and (
            (current_threat >= 0.70 and hunger < 0.62 and health < 0.70)
            or predator_trace_pressure >= 0.75
            or (predator_memory_pressure >= 0.85 and hunger < 0.55)
        ):
            if hunger >= 0.62 and not (
                predator_trace_pressure >= 0.75 and hunger < 0.62
            ):
                semantic_action = "MOVE_TO_FOOD"
                reason = "b2_temporal_threat_safe_hunger_release"
            elif shelter_role != "deep" or shelter_role_level < 0.95:
                semantic_action = "MOVE_TO_SHELTER"
                reason = "b2_temporal_threat_deepen_shelter"
            elif rest_pressure and hunger < 0.70:
                semantic_action = "SLEEP"
                reason = "b2_temporal_threat_rest_deep"
            else:
                semantic_action = "STAY"
                reason = "b2_temporal_threat_hold_deep"
        elif (
            on_shelter
            and semantic_action == "MOVE_TO_SHELTER"
            and hunger >= 0.70
            and current_threat < 0.90
            and predator_trace_pressure < 0.75
        ):
            semantic_action = "MOVE_TO_FOOD"
            reason = "b2_temporal_threat_shelter_role_hunger_release"
        elif (
            not on_shelter
            and hunger >= 0.74
            and health <= 0.35
            and health >= 0.08
            and current_threat < 0.90
            and recent_contact_pressure < 0.50
            and predator_trace_pressure < 0.75
        ):
            semantic_action = "MOVE_TO_FOOD"
            reason = "b2_temporal_threat_emergency_food_over_recover"
        elif not on_shelter and hunger < 0.80 and (
            predator_trace_pressure >= 0.70
            or (predator_memory_pressure >= 0.85 and hunger < 0.60)
            or (current_threat >= 0.50 and health < 0.70 and hunger < 0.75)
            or (current_threat >= 0.50 and 0.45 <= health < 0.70 and hunger < 0.90)
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b2_temporal_threat_return_from_recent_threat"
        else:
            reason = f"b2_temporal_threat_{reason}"

        trace_metrics = {
            "b_current_threat_pressure": round(float(current_threat), 6),
            "b_temporal_threat_pressure": round(float(temporal_threat), 6),
            "b_predator_memory_pressure": round(float(predator_memory_pressure), 6),
            "b_predator_trace_pressure": round(float(predator_trace_pressure), 6),
        }
        return (
            semantic_action,
            B2_TEMPORAL_THREAT_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_metrics,
        )

    def _b3_reset_contact_memory_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b3_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b3_contact_cooldown = 0
        self._b3_post_food_cooldown = 0
        self._b3_last_hunger = None
        self._b3_last_on_food = False

    def _b3_contact_memory_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float | int | str]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_metrics,
        ) = self._b2_temporal_threat_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        profile = (
            str(getattr(self, "_b3_contact_memory_profile_override"))
            if getattr(self, "_b3_contact_memory_profile_override", None) is not None
            else (
                "strict"
                if str(getattr(self.config, "name", ""))
                == B3_CONTACT_MEMORY_STRICT_H48_POLICY_NAME
                else "standard"
            )
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b3_reset_contact_memory_if_needed(tick)

        def _mapping(value: object) -> dict[str, object]:
            return value if isinstance(value, dict) else {}

        hunger = self._b_series_float(hunger_obs, "hunger")
        health = self._b_series_float(sleep_obs, "health")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or shelter_role != "outside"
        )
        shelter_role_level = max(
            self._b_series_float(sleep_obs, "shelter_role_level"),
            self._b_series_float(meta, "shelter_role_level"),
        )
        current_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "predator_smell_strength"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
        )
        recent_contact_pressure = max(
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "recent_contact"),
        )
        recent_pain = max(
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(meta, "recent_pain"),
        )
        predator_trace = _mapping(_mapping(meta.get("percept_traces")).get("predator"))
        predator_trace_pressure = max(
            self._b_series_float(predator_trace, "strength"),
            self._b_series_float(predator_trace, "freshness"),
            self._b_series_float(predator_trace, "certainty"),
        )
        temporal_threat = float(trace_metrics.get("b_temporal_threat_pressure", 0.0))
        previous_hunger = getattr(self, "_b3_last_hunger", None)
        hunger_drop = (
            max(0.0, float(previous_hunger) - float(hunger))
            if previous_hunger is not None
            else 0.0
        )
        contact_cooldown = int(getattr(self, "_b3_contact_cooldown", 0))
        post_food_cooldown = int(getattr(self, "_b3_post_food_cooldown", 0))
        if recent_contact_pressure >= 0.35 or recent_pain >= 0.30:
            contact_cooldown = max(contact_cooldown, 14 if profile == "strict" else 10)
        if bool(getattr(self, "_b3_last_on_food", False)) and not on_food:
            post_food_cooldown = max(
                post_food_cooldown,
                10 if profile == "strict" else 7,
            )
        if hunger_drop >= 0.18:
            post_food_cooldown = max(
                post_food_cooldown,
                12 if profile == "strict" else 8,
            )

        contact_active = contact_cooldown > 0 or recent_contact_pressure >= 0.25
        post_food_active = post_food_cooldown > 0 or hunger_drop >= 0.18
        rest_pressure = bool(
            self._b_series_float(sleep_obs, "night") > 0.5
            or bool(meta.get("night", False))
            or fatigue >= 0.25
            or sleep_debt >= 0.25
        )
        extreme_forage_release = (
            hunger >= (0.92 if profile == "strict" else 0.90)
            and current_threat < 0.45
            and predator_trace_pressure < 0.50
            and recent_contact_pressure < 0.20
        )

        if semantic_action == "EAT" and on_food:
            reason = "b3_contact_memory_eat_on_food"
        elif on_shelter and contact_active and not extreme_forage_release:
            if shelter_role != "deep" or shelter_role_level < 0.95:
                semantic_action = "MOVE_TO_SHELTER"
                reason = "b3_contact_memory_deepen_contact_cooldown"
            elif rest_pressure and hunger < 0.72:
                semantic_action = "SLEEP"
                reason = "b3_contact_memory_rest_contact_cooldown"
            else:
                semantic_action = "STAY"
                reason = "b3_contact_memory_hold_contact_cooldown"
        elif (
            not on_shelter
            and contact_active
            and not extreme_forage_release
            and hunger < (0.88 if profile == "strict" else 0.86)
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b3_contact_memory_return_contact_cooldown"
        elif (
            not on_shelter
            and post_food_active
            and current_threat >= (0.38 if profile == "strict" else 0.45)
            and hunger < (0.88 if profile == "strict" else 0.86)
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b3_contact_memory_return_after_food"
        elif (
            semantic_action == "MOVE_TO_FOOD"
            and not on_shelter
            and current_threat >= (0.55 if profile == "strict" else 0.62)
            and hunger < (0.86 if profile == "strict" else 0.82)
            and health >= 0.12
        ):
            semantic_action = "MOVE_TO_SHELTER"
            reason = "b3_contact_memory_cancel_forage_under_threat"
        else:
            reason = f"b3_contact_memory_{reason}"

        trace_payload: dict[str, float | int | str] = {
            **trace_metrics,
            "b3_contact_cooldown": int(contact_cooldown),
            "b3_post_food_cooldown": int(post_food_cooldown),
            "b3_hunger_drop": round(float(hunger_drop), 6),
            "b3_controller_profile": profile,
        }
        self._b3_last_hunger = float(hunger)
        self._b3_last_on_food = bool(on_food)
        self._b3_contact_cooldown = max(0, int(contact_cooldown) - 1)
        self._b3_post_food_cooldown = max(0, int(post_food_cooldown) - 1)
        self._b3_last_tick = int(tick)
        return (
            semantic_action,
            B3_CONTACT_MEMORY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b3_recurrent_guard_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float | int | str]]:
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        if getattr(self, "_b3_recurrent_guard_last_tick", None) is None or tick <= int(
            getattr(self, "_b3_recurrent_guard_last_tick", 0)
        ):
            hunger_obs = self._bound_observation("hunger_center", observation)
            initial_hunger = self._b_series_float(hunger_obs, "hunger")
            self._b3_recurrent_guard_profile = (
                "easy_like_b2" if initial_hunger < 0.80 else "canonical_guard"
            )
        profile = str(getattr(self, "_b3_recurrent_guard_profile", "canonical_guard"))
        self._b3_recurrent_guard_last_tick = int(tick)
        if profile == "easy_like_b2":
            (
                semantic_action,
                _source,
                reason,
                _override_count,
                trace_payload,
            ) = self._b2_temporal_threat_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
            trace_payload = dict(trace_payload)
            trace_payload.update(
                {
                    "b3_contact_cooldown": 0,
                    "b3_post_food_cooldown": 0,
                    "b3_hunger_drop": 0.0,
                    "b3_controller_profile": "recurrent_guard_easy_b2",
                }
            )
            return (
                semantic_action,
                B3_RECURRENT_GUARD_SELECTION_SOURCE,
                f"b3_recurrent_guard_easy_b2_{reason}",
                int(semantic_action != learned_semantic_action),
                trace_payload,
            )

        previous_override = getattr(self, "_b3_contact_memory_profile_override", None)
        self._b3_contact_memory_profile_override = "strict" if tick < 60 else "standard"
        try:
            (
                semantic_action,
                _source,
                reason,
                _override_count,
                trace_payload,
            ) = self._b3_contact_memory_semantic_action(
                observation,
                learned_semantic_action=learned_semantic_action,
            )
        finally:
            self._b3_contact_memory_profile_override = previous_override
        trace_payload = dict(trace_payload)
        phase = "strict_until_60" if tick < 60 else "standard_after_60"
        trace_payload["b3_controller_profile"] = f"recurrent_guard_{phase}"
        return (
            semantic_action,
            B3_RECURRENT_GUARD_SELECTION_SOURCE,
            f"b3_recurrent_guard_{phase}_{reason}",
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b4_controller_params(self) -> dict[str, float]:
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "recovery_balance"
        )
        defaults: dict[str, float] = {
            "recovery_pressure_threshold": 0.62,
            "sleep_hunger_max": 0.72,
            "sleep_threat_max": 0.50,
            "exit_health_min": 0.26,
            "exit_threat_max": 0.55,
            "hunger_release": 0.88,
            "emergency_hunger_release": 0.94,
            "contact_hold_hunger_max": 0.86,
            "return_threat_min": 0.62,
            "return_hunger_max": 0.90,
            "deep_shelter_level": 0.95,
        }
        if profile == "predator_exit_memory":
            defaults.update(
                {
                    "recovery_pressure_threshold": 0.56,
                    "sleep_threat_max": 0.42,
                    "exit_health_min": 0.38,
                    "exit_threat_max": 0.38,
                    "hunger_release": 0.90,
                    "emergency_hunger_release": 0.96,
                    "contact_hold_hunger_max": 0.90,
                    "return_threat_min": 0.50,
                }
            )
        elif profile == "recovery_balance_h56":
            defaults.update(
                {
                    "recovery_pressure_threshold": 0.58,
                    "sleep_hunger_max": 0.74,
                    "exit_health_min": 0.30,
                    "exit_threat_max": 0.50,
                }
            )
        elif profile == "genetic_recovery":
            defaults.update(
                {
                    "recovery_pressure_threshold": 0.58,
                    "sleep_hunger_max": 0.73,
                    "sleep_threat_max": 0.55,
                    "exit_health_min": 0.32,
                    "exit_threat_max": 0.46,
                    "hunger_release": 0.88,
                    "emergency_hunger_release": 0.95,
                    "contact_hold_hunger_max": 0.88,
                    "return_threat_min": 0.54,
                    "return_hunger_max": 0.91,
                }
            )
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            defaults[str(key)] = float(value)
        return defaults

    def _b4_recovery_balance_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float | int | str | bool]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b3_recurrent_guard_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b4_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "recovery_balance"
        )
        source = (
            B4_GENETIC_RECOVERY_SELECTION_SOURCE
            if str(getattr(self.config, "name", ""))
            == B4_GENETIC_RECOVERY_H48_POLICY_NAME
            or profile == "genetic_recovery"
            else B4_RECOVERY_BALANCE_SELECTION_SOURCE
        )
        if trace_payload.get("b3_controller_profile") == "recurrent_guard_easy_b2":
            trace_payload.update(
                {
                    "b4_controller_profile": profile,
                    "b4_recovery_pressure": 0.0,
                    "b4_sleep_hold": False,
                    "b4_exit_blocked": False,
                    "b4_hunger_release": round(float(params["hunger_release"]), 6),
                }
            )
            if "ga_generation" in params:
                trace_payload["b4_genetic_generation"] = int(params["ga_generation"])
            if "ga_candidate" in params:
                trace_payload["b4_genetic_candidate"] = int(params["ga_candidate"])
            return (
                semantic_action,
                source,
                f"b4_recovery_balance_easy_passthrough_{reason}",
                int(semantic_action != learned_semantic_action),
                trace_payload,
            )
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        hunger = self._b_series_float(hunger_obs, "hunger")
        health = self._b_series_float(sleep_obs, "health")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or shelter_role != "outside"
        )
        shelter_role_level = max(
            self._b_series_float(sleep_obs, "shelter_role_level"),
            self._b_series_float(meta, "shelter_role_level"),
        )
        current_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "predator_smell_strength"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
            float(trace_payload.get("b_current_threat_pressure", 0.0) or 0.0),
        )
        temporal_threat = max(
            float(trace_payload.get("b_temporal_threat_pressure", 0.0) or 0.0),
            current_threat,
        )
        predator_memory_pressure = float(
            trace_payload.get("b_predator_memory_pressure", 0.0) or 0.0
        )
        predator_trace_pressure = float(
            trace_payload.get("b_predator_trace_pressure", 0.0) or 0.0
        )
        contact_cooldown = int(trace_payload.get("b3_contact_cooldown", 0) or 0)
        recovery_pressure = max(
            float(fatigue),
            float(sleep_debt),
            1.0 - float(health),
            min(1.0, float(contact_cooldown) / 14.0),
        )
        threat_pressure = max(
            current_threat,
            temporal_threat,
            predator_memory_pressure,
            predator_trace_pressure,
        )
        sleep_hold = False
        exit_blocked = False
        hunger_release = float(params["hunger_release"])
        emergency_release = float(params["emergency_hunger_release"])
        deep_enough = (
            shelter_role == "deep"
            or shelter_role_level >= float(params["deep_shelter_level"])
        )

        if semantic_action == "EAT" and on_food:
            reason = "b4_recovery_balance_eat_on_food"
        elif (
            on_shelter
            and recovery_pressure >= float(params["recovery_pressure_threshold"])
            and hunger < float(params["sleep_hunger_max"])
            and threat_pressure <= float(params["sleep_threat_max"])
        ):
            semantic_action = "SLEEP" if deep_enough else "MOVE_TO_SHELTER"
            sleep_hold = True
            reason = "b4_recovery_balance_sleep_recovery"
        elif (
            on_shelter
            and contact_cooldown > 0
            and hunger < float(params["contact_hold_hunger_max"])
        ):
            semantic_action = "STAY" if deep_enough else "MOVE_TO_SHELTER"
            exit_blocked = True
            reason = "b4_recovery_balance_hold_recent_contact"
        elif (
            on_shelter
            and semantic_action == "MOVE_TO_FOOD"
            and hunger < emergency_release
            and (
                health < float(params["exit_health_min"])
                or threat_pressure > float(params["exit_threat_max"])
            )
        ):
            semantic_action = "STAY" if deep_enough else "MOVE_TO_SHELTER"
            exit_blocked = True
            reason = "b4_recovery_balance_block_unsafe_exit"
        elif (
            not on_shelter
            and semantic_action == "MOVE_TO_FOOD"
            and hunger < float(params["return_hunger_max"])
            and threat_pressure >= float(params["return_threat_min"])
        ):
            semantic_action = "MOVE_TO_SHELTER"
            exit_blocked = True
            reason = "b4_recovery_balance_return_under_threat"
        elif (
            on_shelter
            and hunger >= hunger_release
            and threat_pressure <= float(params["exit_threat_max"])
        ):
            reason = f"b4_recovery_balance_release_forage_{reason}"
        else:
            reason = f"b4_recovery_balance_{reason}"

        trace_payload.update(
            {
                "b4_controller_profile": profile,
                "b4_recovery_pressure": round(float(recovery_pressure), 6),
                "b4_sleep_hold": bool(sleep_hold),
                "b4_exit_blocked": bool(exit_blocked),
                "b4_hunger_release": round(float(hunger_release), 6),
            }
        )
        if "ga_generation" in params:
            trace_payload["b4_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b4_genetic_candidate"] = int(params["ga_candidate"])
        return (
            semantic_action,
            source,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b5_controller_params(self) -> dict[str, float]:
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "homeostatic_arbiter"
        )
        defaults: dict[str, float] = {
            "hunger_release": 0.86,
            "emergency_hunger_release": 0.94,
            "forage_threat_max": 0.56,
            "forage_lock_ticks": 8.0,
            "sleep_pressure_threshold": 0.70,
            "sleep_hunger_max": 0.78,
            "sleep_threat_max": 0.58,
            "sleep_lock_ticks": 7.0,
            "exit_sleep_pressure_max": 0.58,
            "exit_recovery_debt_max": 0.66,
            "exit_threat_max": 0.62,
            "return_sleep_pressure_min": 0.84,
            "return_recovery_debt_min": 0.82,
            "return_hunger_max": 0.82,
        }
        if profile == "circadian_recovery":
            defaults.update(
                {
                    "hunger_release": 0.88,
                    "emergency_hunger_release": 0.96,
                    "forage_threat_max": 0.52,
                    "sleep_pressure_threshold": 0.58,
                    "sleep_hunger_max": 0.82,
                    "sleep_threat_max": 0.62,
                    "sleep_lock_ticks": 9.0,
                    "exit_sleep_pressure_max": 0.46,
                    "exit_recovery_debt_max": 0.56,
                }
            )
        elif profile == "homeostatic_arbiter_h56":
            defaults.update(
                {
                    "hunger_release": 0.85,
                    "forage_threat_max": 0.58,
                    "sleep_pressure_threshold": 0.66,
                    "sleep_hunger_max": 0.80,
                    "exit_sleep_pressure_max": 0.54,
                }
            )
        elif profile == "genetic_homeostasis":
            defaults.update(
                {
                    "hunger_release": 0.86,
                    "emergency_hunger_release": 0.95,
                    "forage_threat_max": 0.58,
                    "forage_lock_ticks": 9.0,
                    "sleep_pressure_threshold": 0.66,
                    "sleep_hunger_max": 0.80,
                    "sleep_threat_max": 0.60,
                    "sleep_lock_ticks": 8.0,
                    "exit_sleep_pressure_max": 0.54,
                    "exit_recovery_debt_max": 0.62,
                    "exit_threat_max": 0.60,
                }
            )
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            defaults[str(key)] = float(value)
        return defaults

    def _b5_reset_homeostatic_locks_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b5_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b5_sleep_bout_lock = 0
        self._b5_forage_commitment_lock = 0

    def _b5_homeostatic_arbiter_semantic_action(
        self,
        observation: Dict[str, np.ndarray],
        *,
        learned_semantic_action: str,
    ) -> tuple[str, str, str, int, dict[str, float | int | str | bool]]:
        (
            semantic_action,
            _source,
            reason,
            _override_count,
            trace_payload,
        ) = self._b4_recovery_balance_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b5_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "homeostatic_arbiter"
        )
        source = (
            B5_GENETIC_HOMEOSTASIS_SELECTION_SOURCE
            if str(getattr(self.config, "name", ""))
            == B5_GENETIC_HOMEOSTASIS_H48_POLICY_NAME
            or profile == "genetic_homeostasis"
            else B5_HOMEOSTATIC_ARBITER_SELECTION_SOURCE
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b5_reset_homeostatic_locks_if_needed(tick)
        if trace_payload.get("b3_controller_profile") == "recurrent_guard_easy_b2":
            trace_payload.update(
                {
                    "b5_controller_profile": profile,
                    "b5_hunger_urgency": 0.0,
                    "b5_sleep_pressure": 0.0,
                    "b5_recovery_debt": 0.0,
                    "b5_threat_gate": 0.0,
                    "b5_sleep_bout_lock": 0,
                    "b5_forage_commitment_lock": 0,
                    "b5_homeostatic_decision": "easy_passthrough",
                }
            )
            if "ga_generation" in params:
                trace_payload["b5_genetic_generation"] = int(params["ga_generation"])
            if "ga_candidate" in params:
                trace_payload["b5_genetic_candidate"] = int(params["ga_candidate"])
            return (
                semantic_action,
                source,
                f"b5_homeostatic_easy_passthrough_{reason}",
                int(semantic_action != learned_semantic_action),
                trace_payload,
            )

        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        threat_obs = self._bound_observation("threat_center", observation)
        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}

        hunger = self._b_series_float(hunger_obs, "hunger")
        health = self._b_series_float(sleep_obs, "health")
        fatigue = self._b_series_float(sleep_obs, "fatigue")
        sleep_debt = self._b_series_float(sleep_obs, "sleep_debt")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        shelter_role = str(meta.get("shelter_role", "outside"))
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or shelter_role != "outside"
        )
        current_threat = max(
            self._b_series_float(threat_obs, "predator_visible"),
            self._b_series_float(threat_obs, "predator_certainty"),
            self._b_series_float(threat_obs, "predator_motion_salience"),
            self._b_series_float(threat_obs, "visual_predator_threat"),
            self._b_series_float(threat_obs, "olfactory_predator_threat"),
            self._b_series_float(threat_obs, "predator_smell_strength"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            self._b_series_float(meta, "predator_smell_strength"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
            float(trace_payload.get("b_temporal_threat_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_predator_memory_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_predator_trace_pressure", 0.0) or 0.0),
        )
        sleep_pressure = max(float(fatigue), float(sleep_debt))
        recovery_debt = max(
            1.0 - float(health),
            float(sleep_debt) * 0.90,
            float(fatigue) * 0.85,
            float(trace_payload.get("b4_recovery_pressure", 0.0) or 0.0),
        )
        threat_gate = float(current_threat)
        sleep_lock = int(getattr(self, "_b5_sleep_bout_lock", 0))
        forage_lock = int(getattr(self, "_b5_forage_commitment_lock", 0))
        decision_label = "preserve_b4"

        if (
            on_shelter
            and sleep_pressure >= float(params["sleep_pressure_threshold"])
            and hunger < float(params["sleep_hunger_max"])
            and threat_gate <= float(params["sleep_threat_max"])
        ):
            sleep_lock = max(sleep_lock, int(params["sleep_lock_ticks"]))
        if (
            hunger >= float(params["hunger_release"])
            and threat_gate <= float(params["forage_threat_max"])
            and not on_food
        ):
            forage_lock = max(forage_lock, int(params["forage_lock_ticks"]))

        if semantic_action == "EAT" and on_food:
            forage_lock = 0
            decision_label = "eat_on_food"
            reason = "b5_homeostatic_eat_on_food"
        elif (
            on_shelter
            and sleep_lock > 0
            and hunger < float(params["emergency_hunger_release"])
            and threat_gate <= float(params["sleep_threat_max"])
        ):
            semantic_action = "SLEEP"
            decision_label = "sleep_bout_hold"
            reason = "b5_homeostatic_sleep_bout_hold"
        elif (
            on_shelter
            and semantic_action == "MOVE_TO_FOOD"
            and hunger < float(params["emergency_hunger_release"])
            and (
                sleep_pressure > float(params["exit_sleep_pressure_max"])
                or recovery_debt > float(params["exit_recovery_debt_max"])
                or threat_gate > float(params["exit_threat_max"])
            )
        ):
            semantic_action = "STAY"
            decision_label = "block_premature_exit"
            reason = "b5_homeostatic_block_premature_exit"
        elif (
            not on_shelter
            and semantic_action == "MOVE_TO_FOOD"
            and hunger < float(params["return_hunger_max"])
            and (
                sleep_pressure >= float(params["return_sleep_pressure_min"])
                or recovery_debt >= float(params["return_recovery_debt_min"])
            )
            and threat_gate <= float(params["sleep_threat_max"])
        ):
            semantic_action = "MOVE_TO_SHELTER"
            decision_label = "return_for_recovery"
            reason = "b5_homeostatic_return_for_recovery"
        elif (
            forage_lock > 0
            and hunger >= float(params["hunger_release"])
            and threat_gate <= float(params["forage_threat_max"])
        ):
            semantic_action = "MOVE_TO_FOOD"
            decision_label = "forage_commitment"
            reason = "b5_homeostatic_forage_commitment"
        else:
            reason = f"b5_homeostatic_{reason}"

        trace_payload.update(
            {
                "b5_controller_profile": profile,
                "b5_hunger_urgency": round(float(hunger), 6),
                "b5_sleep_pressure": round(float(sleep_pressure), 6),
                "b5_recovery_debt": round(float(recovery_debt), 6),
                "b5_threat_gate": round(float(threat_gate), 6),
                "b5_sleep_bout_lock": int(sleep_lock),
                "b5_forage_commitment_lock": int(forage_lock),
                "b5_homeostatic_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b5_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b5_genetic_candidate"] = int(params["ga_candidate"])

        self._b5_sleep_bout_lock = max(0, int(sleep_lock) - 1)
        self._b5_forage_commitment_lock = max(0, int(forage_lock) - 1)
        self._b5_last_tick = int(tick)
        return (
            semantic_action,
            source,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b6_controller_params(self) -> dict[str, float]:
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "risk_forage_arbiter"
        )
        defaults: dict[str, float] = {
            "b6_family": 1.0,
            "b6_risk_threshold": 0.35,
            "b6_corridor_hunger": 0.86,
            "b6_corridor_lock_ticks": 10.0,
            "b6_threat_memory_ticks": 8.0,
            "b6_return_lock_ticks": 6.0,
            "b6_recurrent_decay": 0.65,
        }
        if profile == "corridor_survival_guard":
            defaults.update(
                {
                    "b6_risk_threshold": 0.40,
                    "b6_corridor_hunger": 0.82,
                    "b6_corridor_lock_ticks": 14.0,
                }
            )
        elif profile == "threat_priority_memory":
            defaults.update(
                {
                    "b6_risk_threshold": 0.22,
                    "b6_threat_memory_ticks": 10.0,
                    "b6_return_lock_ticks": 8.0,
                }
            )
        elif profile == "risk_corridor_h56":
            defaults.update(
                {
                    "b6_risk_threshold": 0.32,
                    "b6_corridor_hunger": 0.84,
                    "b6_corridor_lock_ticks": 12.0,
                }
            )
        elif profile == "genetic_risk_corridor":
            defaults.update(
                {
                    "b6_risk_threshold": 0.30,
                    "b6_corridor_hunger": 0.83,
                    "b6_corridor_lock_ticks": 14.0,
                    "b6_threat_memory_ticks": 12.0,
                }
            )
        elif profile == "recurrent_context":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_recurrent_decay": 0.70,
                    "b6_return_lock_ticks": 8.0,
                }
            )
        elif profile == "recurrent_threat_homeostasis":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_risk_threshold": 0.28,
                    "b6_threat_memory_ticks": 12.0,
                    "b6_return_lock_ticks": 10.0,
                    "b6_recurrent_decay": 0.78,
                }
            )
        elif profile == "recurrent_corridor_guard":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_corridor_hunger": 0.82,
                    "b6_corridor_lock_ticks": 16.0,
                    "b6_recurrent_decay": 0.80,
                }
            )
        elif profile == "recurrent_context_h56":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_corridor_hunger": 0.84,
                    "b6_recurrent_decay": 0.75,
                    "b6_return_lock_ticks": 8.0,
                }
            )
        elif profile == "genetic_recurrent_memory":
            defaults.update(
                {
                    "b6_family": 2.0,
                    "b6_risk_threshold": 0.30,
                    "b6_corridor_hunger": 0.83,
                    "b6_corridor_lock_ticks": 14.0,
                    "b6_threat_memory_ticks": 12.0,
                    "b6_return_lock_ticks": 10.0,
                    "b6_recurrent_decay": 0.78,
                }
            )
        elif profile == "fused_risk_recurrent":
            defaults.update(
                {
                    "b6_family": 3.0,
                    "b6_risk_threshold": 0.28,
                    "b6_corridor_hunger": 0.82,
                    "b6_corridor_lock_ticks": 16.0,
                    "b6_threat_memory_ticks": 12.0,
                    "b6_return_lock_ticks": 10.0,
                    "b6_recurrent_decay": 0.80,
                }
            )
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            defaults[str(key)] = float(value)
        return defaults

    def _b6_controller_family(self, profile: str, params: dict[str, float]) -> str:
        name = str(getattr(self.config, "name", ""))
        if name == B6_FUSED_RISK_RECURRENT_H48_POLICY_NAME or profile == "fused_risk_recurrent":
            return "fused_risk_recurrent"
        if name in {
            B6_RECURRENT_CONTEXT_H48_POLICY_NAME,
            B6_RECURRENT_THREAT_HOMEOSTASIS_H48_POLICY_NAME,
            B6_RECURRENT_CORRIDOR_GUARD_H48_POLICY_NAME,
            B6_RECURRENT_CONTEXT_H56_POLICY_NAME,
            B6_GENETIC_RECURRENT_MEMORY_H48_POLICY_NAME,
        } or profile.startswith("recurrent") or profile == "genetic_recurrent_memory":
            return "recurrent_memory"
        if int(round(float(params.get("b6_family", 1.0)))) == 2:
            return "recurrent_memory"
        if int(round(float(params.get("b6_family", 1.0)))) == 3:
            return "fused_risk_recurrent"
        return "risk_corridor"

    def _b6_reset_recurrent_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b6_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b6_corridor_commitment = 0
        self._b6_corridor_progress_memory = 0.0
        self._b6_recurrent_threat_memory = 0.0
        self._b6_return_lock = 0
        self._b6_last_tick = int(tick)

    def _b6_local_food_progress_signal(self, meta: dict[str, object]) -> float:
        transitions = meta.get("local_transition_consequences")
        transitions = transitions if isinstance(transitions, dict) else {}
        best = 0.0
        for transition in transitions.values():
            if not isinstance(transition, dict):
                continue
            best = max(
                best,
                self._b_series_float(transition, "food_dist_delta"),
                1.0 if bool(transition.get("next_cell_has_food", False)) else 0.0,
            )
        return float(best)
