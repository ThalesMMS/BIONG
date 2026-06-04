from __future__ import annotations

from .runtime_shared import *


class _BrainRuntimePart2Mixin:
    def _b6_risk_corridor_semantic_action(
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
        ) = self._b5_homeostatic_arbiter_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b6_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "risk_forage_arbiter"
        )
        family = self._b6_controller_family(profile, params)
        if family == "fused_risk_recurrent":
            source = B6_FUSED_RISK_RECURRENT_SELECTION_SOURCE
        elif family == "recurrent_memory":
            source = B6_RECURRENT_MEMORY_SELECTION_SOURCE
        else:
            source = B6_RISK_CORRIDOR_SELECTION_SOURCE

        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b6_reset_recurrent_state_if_needed(tick)
        if trace_payload.get("b5_homeostatic_decision") == "easy_passthrough":
            trace_payload.update(
                {
                    "b6_controller_family": family,
                    "b6_controller_profile": profile,
                    "b6_risk_pressure": 0.0,
                    "b6_threat_priority": 0.0,
                    "b6_forage_suppressed": 0.0,
                    "b6_corridor_commitment": 0,
                    "b6_corridor_progress_memory": 0.0,
                    "b6_recurrent_state": "easy_passthrough",
                    "b6_return_lock": 0,
                    "b6_decision": "easy_passthrough",
                    "b6_emit_action_center_payload": True,
                    "b6_action_center_payload": {
                        "winning_valence": "exploration",
                        "evidence": {
                            "threat": {
                                "predator_visible": 0.0,
                                "predator_proximity": 0.0,
                                "predator_certainty": 0.0,
                            }
                        },
                        "module_gates": {"hunger_center": 1.0},
                    },
                }
            )
            if "ga_generation" in params:
                trace_payload["b6_genetic_generation"] = int(params["ga_generation"])
            if "ga_candidate" in params:
                trace_payload["b6_genetic_candidate"] = int(params["ga_candidate"])
            self._b6_last_tick = int(tick)
            return (
                semantic_action,
                source,
                f"b6_easy_passthrough_{reason}",
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
        )
        risk_pressure = max(
            current_threat,
            float(trace_payload.get("b5_threat_gate", 0.0) or 0.0),
            float(trace_payload.get("b_temporal_threat_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_predator_memory_pressure", 0.0) or 0.0),
            float(trace_payload.get("b_predator_trace_pressure", 0.0) or 0.0),
        )
        recent_contact = max(
            self._b_series_float(threat_obs, "recent_contact"),
            self._b_series_float(threat_obs, "recent_pain"),
            self._b_series_float(meta, "recent_contact"),
            self._b_series_float(meta, "recent_pain"),
        )
        map_template = str(meta.get("map_template", ""))
        central_retention_map = map_template == "central_burrow"
        corridor_map = map_template == "corridor_escape"
        food_progress_signal = self._b6_local_food_progress_signal(meta)
        corridor_commitment = int(getattr(self, "_b6_corridor_commitment", 0))
        corridor_memory = float(getattr(self, "_b6_corridor_progress_memory", 0.0))
        recurrent_threat = float(getattr(self, "_b6_recurrent_threat_memory", 0.0))
        return_lock = int(getattr(self, "_b6_return_lock", 0))
        decay = float(params["b6_recurrent_decay"])
        if family in {"recurrent_memory", "fused_risk_recurrent"}:
            recurrent_threat = max(recurrent_threat * decay, risk_pressure)
            corridor_memory = max(corridor_memory * decay, food_progress_signal)
            if risk_pressure >= float(params["b6_risk_threshold"]) or recent_contact > 0.0:
                return_lock = max(return_lock, int(params["b6_return_lock_ticks"]))
        else:
            recurrent_threat = max(recurrent_threat * 0.50, risk_pressure)
            corridor_memory = max(corridor_memory * 0.50, food_progress_signal)

        threat_priority = float(
            max(risk_pressure, recurrent_threat)
            >= float(params["b6_risk_threshold"])
        )
        forage_suppressed = 0.0
        decision_label = "preserve_b5"
        emergency_hunger = float(params.get("emergency_hunger_release", 0.95))
        shelter_block_hunger = min(0.70, emergency_hunger)
        corridor_hunger = float(params["b6_corridor_hunger"])

        if on_food and hunger >= 0.10:
            semantic_action = "EAT"
            corridor_commitment = 0
            return_lock = 0
            decision_label = "eat_on_food"
            reason = "b6_eat_on_food"
        elif (
            threat_priority > 0.0
            and not on_shelter
            and hunger < shelter_block_hunger
            and not central_retention_map
            and family in {"risk_corridor", "fused_risk_recurrent"}
        ):
            semantic_action = "MOVE_TO_SHELTER"
            forage_suppressed = 1.0
            return_lock = max(return_lock, int(params["b6_return_lock_ticks"]))
            decision_label = "risk_shelter_return"
            reason = "b6_risk_corridor_return_under_threat"
        elif (
            threat_priority > 0.0
            and not on_shelter
            and hunger < shelter_block_hunger
            and not central_retention_map
            and family == "recurrent_memory"
            and recurrent_threat >= float(params["b6_risk_threshold"])
        ):
            semantic_action = "MOVE_TO_SHELTER"
            forage_suppressed = 1.0
            return_lock = max(return_lock, int(params["b6_return_lock_ticks"]))
            decision_label = "recurrent_risk_return"
            reason = "b6_recurrent_memory_return_lock"
        elif (
            return_lock > 0
            and not on_shelter
            and hunger < shelter_block_hunger
            and not central_retention_map
        ):
            semantic_action = "MOVE_TO_SHELTER"
            forage_suppressed = 1.0
            decision_label = "return_lock"
            reason = "b6_return_lock_shelter"
        elif (
            corridor_map
            and hunger >= corridor_hunger
            and risk_pressure < 0.80
            and recent_contact <= 0.0
            and not on_food
            and food_progress_signal > 0.0
        ):
            semantic_action = "MOVE_TO_FOOD"
            corridor_commitment = max(
                corridor_commitment,
                int(params["b6_corridor_lock_ticks"]),
            )
            decision_label = "corridor_commitment"
            reason = "b6_corridor_commitment_food_progress"
        elif (
            corridor_map
            and corridor_commitment > 0
            and not on_food
            and risk_pressure < 0.75
        ):
            semantic_action = "MOVE_TO_FOOD"
            decision_label = "corridor_commitment_hold"
            reason = "b6_corridor_commitment_hold"
        else:
            reason = f"b6_{family}_{reason}"

        action_center_threat = max(
            risk_pressure,
            recurrent_threat,
            0.55 if threat_priority > 0.0 else 0.0,
        )
        if threat_priority > 0.0 and hunger < emergency_hunger:
            forage_suppressed = max(float(forage_suppressed), 1.0)
        winning_valence = "threat" if threat_priority > 0.0 else (
            "hunger" if hunger >= corridor_hunger else "exploration"
        )
        hunger_gate = 0.20 if threat_priority > 0.0 else 1.0
        trace_payload.update(
            {
                "b6_controller_family": family,
                "b6_controller_profile": profile,
                "b6_risk_pressure": round(float(risk_pressure), 6),
                "b6_threat_priority": round(float(threat_priority), 6),
                "b6_forage_suppressed": round(float(forage_suppressed), 6),
                "b6_corridor_commitment": int(corridor_commitment),
                "b6_corridor_progress_memory": round(float(corridor_memory), 6),
                "b6_recurrent_state": (
                    "active"
                    if family in {"recurrent_memory", "fused_risk_recurrent"}
                    and (recurrent_threat > 0.0 or corridor_memory > 0.0)
                    else "risk_only"
                ),
                "b6_return_lock": int(return_lock),
                "b6_decision": decision_label,
                "b6_emit_action_center_payload": True,
                "b6_action_center_payload": {
                    "winning_valence": winning_valence,
                    "evidence": {
                        "threat": {
                            "predator_visible": round(
                                float(1.0 if action_center_threat >= 0.30 else 0.0),
                                6,
                            ),
                            "predator_proximity": round(
                                float(max(action_center_threat, 0.35 if threat_priority > 0.0 else 0.0)),
                                6,
                            ),
                            "predator_certainty": round(
                                float(max(action_center_threat, 0.45 if threat_priority > 0.0 else 0.0)),
                                6,
                            ),
                        }
                    },
                    "module_gates": {"hunger_center": round(float(hunger_gate), 6)},
                },
            }
        )
        if "ga_generation" in params:
            trace_payload["b6_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b6_genetic_candidate"] = int(params["ga_candidate"])

        self._b6_corridor_commitment = max(0, int(corridor_commitment) - 1)
        self._b6_corridor_progress_memory = float(corridor_memory)
        self._b6_recurrent_threat_memory = float(recurrent_threat)
        self._b6_return_lock = max(0, int(return_lock) - 1)
        self._b6_last_tick = int(tick)
        return (
            semantic_action,
            source,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b7_controller_params(self) -> dict[str, float]:
        params = self._b6_controller_params()
        defaults = {
            "b7_budget_step_cost": 0.085,
            "b7_viability_margin": -0.08,
            "b7_abort_health": 0.36,
            "b7_recover_health": 0.42,
            "b7_food_commit_distance": 13.0,
            "b7_commitment_ticks": 8.0,
            "b7_recurrent_decay": 0.72,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "affordance_budget"
        )
        if profile == "energy_budget_corridor":
            defaults.update({"b7_viability_margin": -0.02, "b7_abort_health": 0.42})
        elif profile == "recurrent_affordance":
            defaults.update({"b7_recurrent_decay": 0.84, "b7_commitment_ticks": 10.0})
        elif profile == "affordance_budget_h56":
            defaults.update({"b7_budget_step_cost": 0.080, "b7_commitment_ticks": 10.0})
        elif profile == "genetic_affordance_budget":
            defaults.update({"b7_abort_health": 0.38, "b7_recurrent_decay": 0.80})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b7_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b7_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b7_commitment_lock = 0
        self._b7_last_food_distance = None
        self._b7_progress_memory = 0.0
        self._b7_last_tick = int(tick)

    def _b7_affordance_budget_semantic_action(
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
        ) = self._b6_risk_corridor_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b7_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "affordance_budget"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b7_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        map_template = str(meta.get("map_template", ""))
        hunger_obs = self._bound_observation("hunger_center", observation)
        sleep_obs = self._bound_observation("sleep_center", observation)
        hunger = self._b_series_float(hunger_obs, "hunger")
        health = self._b_series_float(sleep_obs, "health")
        on_food = self._b_series_float(hunger_obs, "on_food") > 0.5 or bool(
            meta.get("on_food", False)
        )
        on_shelter = (
            self._b_series_float(sleep_obs, "on_shelter") > 0.5
            or bool(meta.get("on_shelter", False))
            or str(meta.get("shelter_role", "outside")) != "outside"
        )

        def _raw_float(key: str, default: float = 0.0) -> float:
            try:
                value = float(meta.get(key, default))
            except (TypeError, ValueError):
                return float(default)
            return float(value) if np.isfinite(value) else float(default)

        food_steps = max(0.0, _raw_float("food_dist", 0.0))
        return_steps = max(0.0, _raw_float("shelter_dist", 0.0))
        threat = max(
            float(trace_payload.get("b6_risk_pressure", 0.0) or 0.0),
            self._b_series_float(meta, "predator_smell_strength"),
            self._b_series_float(meta, "visual_predator_threat"),
            self._b_series_float(meta, "olfactory_predator_threat"),
            self._b_series_float(meta, "predator_motion_salience"),
            1.0 if bool(meta.get("predator_visible", False)) else 0.0,
        )
        step_cost = float(params["b7_budget_step_cost"])
        energy_budget = health - min(1.0, hunger * 0.08) - min(0.25, threat * 0.12)
        budget_margin = energy_budget - (min(food_steps, return_steps + 1.0) * step_cost)

        last_food_distance = getattr(self, "_b7_last_food_distance", None)
        recent_progress = 0.0
        if last_food_distance is not None:
            recent_progress = max(0.0, float(last_food_distance) - food_steps)
        progress_memory = max(
            float(getattr(self, "_b7_progress_memory", 0.0))
            * float(params["b7_recurrent_decay"]),
            recent_progress,
            float(trace_payload.get("b6_corridor_progress_memory", 0.0) or 0.0),
        )
        commitment_lock = int(getattr(self, "_b7_commitment_lock", 0))
        corridor_map = map_template == "corridor_escape"
        decision_label = "preserve_b6"
        affordance_state = "non_corridor"
        abort_return = False
        viability = 1.0

        if on_food:
            semantic_action = "EAT"
            decision_label = "eat_on_food"
            affordance_state = "food_reached"
            commitment_lock = 0
            reason = "b7_eat_on_food"
        elif corridor_map:
            affordance_state = "corridor_open"
            explicit_budget_risk = (
                health <= float(params["b7_abort_health"])
                and food_steps <= float(params["b7_food_commit_distance"])
            )
            if on_shelter and health < float(params["b7_recover_health"]):
                semantic_action = "STAY"
                decision_label = "recover_before_crossing"
                affordance_state = "recover_in_shelter"
                reason = "b7_recover_before_crossing"
            elif explicit_budget_risk:
                abort_return = True
                decision_label = "abort_return_unviable"
                affordance_state = "budget_unviable_commitment"
                reason = "b7_abort_return_unviable"
            elif (
                budget_margin >= float(params["b7_viability_margin"])
                or semantic_action == "MOVE_TO_FOOD"
                or commitment_lock > 0
            ):
                semantic_action = "MOVE_TO_FOOD"
                commitment_lock = max(commitment_lock, int(params["b7_commitment_ticks"]))
                decision_label = "continue_viable"
                affordance_state = "corridor_commitment"
                reason = "b7_continue_viable"
            else:
                abort_return = True
                semantic_action = "MOVE_TO_SHELTER"
                decision_label = "abort_return_unviable"
                affordance_state = "budget_unviable_return"
                reason = "b7_abort_return_unviable"
            viability = float(np.clip(0.5 + budget_margin, 0.0, 1.0))

        trace_payload.update(
            {
                "b7_controller_profile": profile,
                "b7_affordance_state": affordance_state,
                "b7_energy_budget": round(float(energy_budget), 6),
                "b7_budget_margin": round(float(budget_margin), 6),
                "b7_food_steps_estimate": round(float(food_steps), 6),
                "b7_return_steps_estimate": round(float(return_steps), 6),
                "b7_corridor_viability": round(float(viability), 6),
                "b7_abort_return": bool(abort_return),
                "b7_commitment_lock": int(commitment_lock),
                "b7_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b7_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b7_genetic_candidate"] = int(params["ga_candidate"])

        self._b7_commitment_lock = max(0, int(commitment_lock) - 1)
        self._b7_last_food_distance = float(food_steps)
        self._b7_progress_memory = float(progress_memory)
        self._b7_last_tick = int(tick)
        return (
            semantic_action,
            B7_AFFORDANCE_BUDGET_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b8_controller_params(self) -> dict[str, float]:
        params = self._b7_controller_params()
        defaults = {
            "b8_place_memory_decay": 0.78,
            "b8_dead_end_risk_threshold": 0.62,
            "b8_return_vector_threshold": 0.18,
            "b8_abort_health": 0.18,
            "b8_hold_health": 0.10,
            "b8_food_progress_floor": 13.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "spatial_affordance_map"
        )
        if profile == "return_vector":
            defaults.update({"b8_return_vector_threshold": 0.10, "b8_abort_health": 0.24})
        elif profile == "corridor_place_memory":
            defaults.update({"b8_place_memory_decay": 0.88, "b8_dead_end_risk_threshold": 0.55})
        elif profile == "spatial_affordance_map_h56":
            defaults.update({"b8_place_memory_decay": 0.82, "b8_food_progress_floor": 12.5})
        elif profile == "genetic_spatial_affordance":
            defaults.update({"b8_place_memory_decay": 0.84, "b8_abort_health": 0.20})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b8_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b8_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b8_place_memory = 0.0
        self._b8_last_food_distance = None
        self._b8_last_tick = int(tick)

    def _b8_spatial_affordance_semantic_action(
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
        ) = self._b7_affordance_budget_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b8_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "spatial_affordance_map"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b8_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        map_template = str(meta.get("map_template", ""))
        sleep_obs = self._bound_observation("sleep_center", observation)
        health = self._b_series_float(sleep_obs, "health")

        def _raw_float(mapping: dict[str, object], key: str, default: float = 0.0) -> float:
            try:
                value = float(mapping.get(key, default))
            except (TypeError, ValueError):
                return float(default)
            return float(value) if np.isfinite(value) else float(default)

        def _unblocked_delta(action: str, field: str) -> float:
            affordances = meta.get("local_affordances")
            affordances = affordances if isinstance(affordances, dict) else {}
            transitions = meta.get("local_transition_consequences")
            transitions = transitions if isinstance(transitions, dict) else {}
            action_affordance = affordances.get(action)
            action_affordance = (
                action_affordance if isinstance(action_affordance, dict) else {}
            )
            if bool(action_affordance.get("blocked", False)):
                return -1.0
            action_transition = transitions.get(action)
            action_transition = (
                action_transition if isinstance(action_transition, dict) else {}
            )
            return _raw_float(action_transition, field, 0.0)

        food_deltas = [
            _unblocked_delta(action, "food_dist_delta")
            for action in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT")
        ]
        shelter_deltas = [
            _unblocked_delta(action, "shelter_dist_delta")
            for action in ("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT")
        ]
        best_food_delta = max(food_deltas) if food_deltas else 0.0
        best_shelter_delta = max(shelter_deltas) if shelter_deltas else 0.0
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        last_food_distance = getattr(self, "_b8_last_food_distance", None)
        recent_progress = (
            max(0.0, float(last_food_distance) - food_steps)
            if last_food_distance is not None
            else 0.0
        )
        place_memory = max(
            float(getattr(self, "_b8_place_memory", 0.0))
            * float(params["b8_place_memory_decay"]),
            recent_progress,
            float(trace_payload.get("b7_corridor_viability", 0.0) or 0.0) * 0.25,
        )
        threat = float(trace_payload.get("b6_risk_pressure", 0.0) or 0.0)
        local_affordance_score = best_food_delta + 0.25 * best_shelter_delta - 0.5 * threat
        return_vector_strength = max(0.0, best_shelter_delta)
        dead_end_risk = float(np.clip(1.0 - max(best_food_delta, 0.0) + threat * 0.25, 0.0, 1.0))
        corridor_map = map_template == "corridor_escape"
        b7_decision = str(trace_payload.get("b7_decision", "preserve_b7"))
        decision_label = "preserve_b7"
        spatial_map_state = "non_corridor"
        abort_executed = False

        if corridor_map:
            spatial_map_state = "corridor_place_field"
            if b7_decision == "abort_return_unviable":
                decision_label = "corridor_abort_signal"
                spatial_map_state = "return_vector_available"
                if (
                    profile == "return_vector"
                    and health <= float(params["b8_abort_health"])
                    and return_vector_strength >= float(params["b8_return_vector_threshold"])
                ):
                    semantic_action = "MOVE_TO_SHELTER"
                    abort_executed = True
                    reason = "b8_return_vector_abort"
            elif (
                dead_end_risk >= float(params["b8_dead_end_risk_threshold"])
                and health <= float(params["b8_hold_health"])
            ):
                decision_label = "corridor_hold_unviable"
                semantic_action = "STAY"
                spatial_map_state = "dead_end_hold"
                reason = "b8_spatial_hold"
            elif (
                best_food_delta > 0.0
                or food_steps <= float(params["b8_food_progress_floor"])
                or b7_decision == "continue_viable"
            ):
                decision_label = "corridor_continue_mapped"
                semantic_action = "MOVE_TO_FOOD"
                spatial_map_state = "food_vector_available"
                reason = "b8_spatial_continue"

        trace_payload.update(
            {
                "b8_controller_profile": profile,
                "b8_spatial_map_state": spatial_map_state,
                "b8_local_affordance_score": round(float(local_affordance_score), 6),
                "b8_return_vector_strength": round(float(return_vector_strength), 6),
                "b8_corridor_dead_end_risk": round(float(dead_end_risk), 6),
                "b8_abort_executed": bool(abort_executed),
                "b8_place_memory": round(float(place_memory), 6),
                "b8_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b8_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b8_genetic_candidate"] = int(params["ga_candidate"])

        self._b8_place_memory = float(place_memory)
        self._b8_last_food_distance = float(food_steps)
        self._b8_last_tick = int(tick)
        return (
            semantic_action,
            B8_SPATIAL_AFFORDANCE_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b9_controller_params(self) -> dict[str, float]:
        params = self._b8_controller_params()
        defaults = {
            "b9_route_memory_decay": 0.82,
            "b9_waypoint_commit_ticks": 6.0,
            "b9_route_confidence_threshold": 0.18,
            "b9_path_integrator_gain": 0.50,
            "b9_replan_dead_end_threshold": 0.72,
            "b9_progress_floor": 13.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "waypoint_planner"
        )
        if profile == "path_integration":
            defaults.update({"b9_path_integrator_gain": 0.70, "b9_route_confidence_threshold": 0.12})
        elif profile == "route_memory":
            defaults.update({"b9_route_memory_decay": 0.90, "b9_waypoint_commit_ticks": 8.0})
        elif profile == "waypoint_planner_h56":
            defaults.update({"b9_route_memory_decay": 0.86, "b9_waypoint_commit_ticks": 7.0})
        elif profile == "genetic_waypoint_planner":
            defaults.update({"b9_route_memory_decay": 0.88, "b9_path_integrator_gain": 0.62})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b9_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b9_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b9_route_memory = 0.0
        self._b9_waypoint_lock = 0
        self._b9_last_tick = int(tick)

    def _b9_waypoint_planner_semantic_action(
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
        ) = self._b8_spatial_affordance_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b9_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "waypoint_planner"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b9_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        local_score = float(trace_payload.get("b8_local_affordance_score", 0.0) or 0.0)
        place_memory = float(trace_payload.get("b8_place_memory", 0.0) or 0.0)
        dead_end_risk = float(trace_payload.get("b8_corridor_dead_end_risk", 0.0) or 0.0)
        b8_decision = str(trace_payload.get("b8_decision", "preserve_b8"))
        route_memory = max(
            float(getattr(self, "_b9_route_memory", 0.0))
            * float(params["b9_route_memory_decay"]),
            place_memory,
            max(0.0, local_score) * float(params["b9_path_integrator_gain"]),
        )
        route_confidence = float(
            np.clip(route_memory + max(0.0, 16.0 - food_steps) / 16.0, 0.0, 1.0)
        )
        replan_signal = float(np.clip(dead_end_risk - route_confidence, 0.0, 1.0))
        waypoint_lock = int(getattr(self, "_b9_waypoint_lock", 0))
        route_state = "non_corridor"
        decision_label = "preserve_b8"

        if corridor_map:
            if (
                b8_decision == "corridor_continue_mapped"
                and route_confidence >= float(params["b9_route_confidence_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                waypoint_lock = max(waypoint_lock, int(params["b9_waypoint_commit_ticks"]))
                route_state = "food_waypoint_locked"
                decision_label = "commit_food_waypoint"
                reason = "b9_commit_food_waypoint"
            elif replan_signal >= float(params["b9_replan_dead_end_threshold"]):
                route_state = "replan_return_vector"
                decision_label = "replan_return"
                reason = "b9_replan_return"
            elif waypoint_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                route_state = "waypoint_lock_continues"
                decision_label = "continue_locked_waypoint"
                reason = "b9_continue_locked_waypoint"

        trace_payload.update(
            {
                "b9_controller_profile": profile,
                "b9_route_state": route_state,
                "b9_route_confidence": round(float(route_confidence), 6),
                "b9_waypoint_lock": int(waypoint_lock),
                "b9_path_integrator": round(float(route_memory), 6),
                "b9_replan_signal": round(float(replan_signal), 6),
                "b9_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b9_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b9_genetic_candidate"] = int(params["ga_candidate"])

        self._b9_route_memory = float(route_memory)
        self._b9_waypoint_lock = max(0, int(waypoint_lock) - 1)
        self._b9_last_tick = int(tick)
        return (
            semantic_action,
            B9_WAYPOINT_PLANNER_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b10_controller_params(self) -> dict[str, float]:
        params = self._b9_controller_params()
        defaults = {
            "b10_replay_memory_decay": 0.84,
            "b10_rollout_gain": 0.55,
            "b10_value_threshold": 0.20,
            "b10_replay_commit_ticks": 5.0,
            "b10_abort_threshold": 0.64,
            "b10_progress_floor": 13.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "prospective_replay"
        )
        if profile == "value_route_evaluator":
            defaults.update({"b10_rollout_gain": 0.68, "b10_value_threshold": 0.16})
        elif profile == "replay_planner":
            defaults.update({"b10_replay_memory_decay": 0.90, "b10_replay_commit_ticks": 7.0})
        elif profile == "prospective_replay_h56":
            defaults.update({"b10_replay_memory_decay": 0.87, "b10_replay_commit_ticks": 6.0})
        elif profile == "genetic_replay_planner":
            defaults.update({"b10_replay_memory_decay": 0.88, "b10_rollout_gain": 0.64})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b10_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b10_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b10_replay_memory = 0.0
        self._b10_plan_commitment = 0
        self._b10_last_tick = int(tick)

    def _b10_prospective_replay_semantic_action(
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
        ) = self._b9_waypoint_planner_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b10_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "prospective_replay"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b10_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b9_decision = str(trace_payload.get("b9_decision", "preserve_b9"))
        route_confidence = float(trace_payload.get("b9_route_confidence", 0.0) or 0.0)
        replan_signal = float(trace_payload.get("b9_replan_signal", 0.0) or 0.0)
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        dead_end_risk = float(trace_payload.get("b8_corridor_dead_end_risk", 0.0) or 0.0)
        replay_memory = max(
            float(getattr(self, "_b10_replay_memory", 0.0))
            * float(params["b10_replay_memory_decay"]),
            route_confidence * float(params["b10_rollout_gain"]),
        )
        rollout_depth = int(max(1, min(5, round(max(0.0, 16.0 - food_steps) / 4.0) + 1)))
        prospective_value = float(
            np.clip(
                replay_memory
                + route_confidence
                + max(0.0, 16.0 - food_steps) / 20.0
                - dead_end_risk * 0.35,
                0.0,
                1.0,
            )
        )
        abort_signal = float(np.clip(max(dead_end_risk, replan_signal) - prospective_value, 0.0, 1.0))
        plan_commitment = int(getattr(self, "_b10_plan_commitment", 0))
        replay_state = "non_corridor"
        decision_label = "preserve_b9"

        if corridor_map:
            if (
                b9_decision in {"commit_food_waypoint", "continue_locked_waypoint"}
                and prospective_value >= float(params["b10_value_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                plan_commitment = max(plan_commitment, int(params["b10_replay_commit_ticks"]))
                replay_state = "prospective_food_plan"
                decision_label = "commit_replayed_route"
                reason = "b10_commit_replayed_route"
            elif abort_signal >= float(params["b10_abort_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                replay_state = "prospective_abort_return"
                decision_label = "abort_failed_rollout"
                reason = "b10_abort_failed_rollout"
            elif plan_commitment > 0:
                semantic_action = "MOVE_TO_FOOD"
                replay_state = "plan_commitment_continues"
                decision_label = "continue_replay_commitment"
                reason = "b10_continue_replay_commitment"

        trace_payload.update(
            {
                "b10_controller_profile": profile,
                "b10_replay_state": replay_state,
                "b10_prospective_value": round(float(prospective_value), 6),
                "b10_rollout_depth": int(rollout_depth),
                "b10_replay_memory": round(float(replay_memory), 6),
                "b10_plan_commitment": int(plan_commitment),
                "b10_abort_signal": round(float(abort_signal), 6),
                "b10_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b10_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b10_genetic_candidate"] = int(params["ga_candidate"])

        self._b10_replay_memory = float(replay_memory)
        self._b10_plan_commitment = max(0, int(plan_commitment) - 1)
        self._b10_last_tick = int(tick)
        return (
            semantic_action,
            B10_PROSPECTIVE_REPLAY_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b11_controller_params(self) -> dict[str, float]:
        params = self._b10_controller_params()
        defaults = {
            "b11_confidence_decay": 0.86,
            "b11_confidence_threshold": 0.24,
            "b11_uncertainty_threshold": 0.70,
            "b11_neuromod_gain": 0.50,
            "b11_confidence_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "confidence_arbiter"
        )
        if profile == "uncertainty_gate":
            defaults.update({"b11_uncertainty_threshold": 0.58, "b11_confidence_threshold": 0.18})
        elif profile == "neuromodulated_replay":
            defaults.update({"b11_neuromod_gain": 0.70, "b11_confidence_decay": 0.90})
        elif profile == "confidence_arbiter_h56":
            defaults.update({"b11_confidence_decay": 0.88, "b11_confidence_commit_ticks": 6.0})
        elif profile == "genetic_confidence_gate":
            defaults.update({"b11_confidence_threshold": 0.20, "b11_neuromod_gain": 0.62})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b11_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b11_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b11_confidence_memory = 0.0
        self._b11_confidence_lock = 0
        self._b11_last_tick = int(tick)

    def _b11_confidence_arbiter_semantic_action(
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
        ) = self._b10_prospective_replay_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b11_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "confidence_arbiter"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b11_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        prospective_value = float(trace_payload.get("b10_prospective_value", 0.0) or 0.0)
        abort_signal = float(trace_payload.get("b10_abort_signal", 0.0) or 0.0)
        route_confidence = float(trace_payload.get("b9_route_confidence", 0.0) or 0.0)
        b10_decision = str(trace_payload.get("b10_decision", "preserve_b10"))
        confidence_memory = max(
            float(getattr(self, "_b11_confidence_memory", 0.0))
            * float(params["b11_confidence_decay"]),
            prospective_value,
            route_confidence * float(params["b11_neuromod_gain"]),
        )
        uncertainty = float(np.clip(abort_signal + max(0.0, 0.50 - route_confidence), 0.0, 1.0))
        neuromod_signal = float(np.clip(confidence_memory - uncertainty, 0.0, 1.0))
        confidence_lock = int(getattr(self, "_b11_confidence_lock", 0))
        confidence_state = "non_corridor"
        decision_label = "preserve_b10"

        if corridor_map:
            if (
                b10_decision in {"commit_replayed_route", "continue_replay_commitment"}
                and neuromod_signal >= float(params["b11_confidence_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                confidence_lock = max(confidence_lock, int(params["b11_confidence_commit_ticks"]))
                confidence_state = "high_confidence_plan"
                decision_label = "commit_confident_plan"
                reason = "b11_commit_confident_plan"
            elif uncertainty >= float(params["b11_uncertainty_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                confidence_state = "high_uncertainty_return"
                decision_label = "gate_uncertain_plan"
                reason = "b11_gate_uncertain_plan"
            elif confidence_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                confidence_state = "confidence_lock_continues"
                decision_label = "continue_confidence_lock"
                reason = "b11_continue_confidence_lock"

        trace_payload.update(
            {
                "b11_controller_profile": profile,
                "b11_confidence_state": confidence_state,
                "b11_plan_confidence": round(float(confidence_memory), 6),
                "b11_uncertainty": round(float(uncertainty), 6),
                "b11_neuromod_signal": round(float(neuromod_signal), 6),
                "b11_confidence_lock": int(confidence_lock),
                "b11_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b11_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b11_genetic_candidate"] = int(params["ga_candidate"])

        self._b11_confidence_memory = float(confidence_memory)
        self._b11_confidence_lock = max(0, int(confidence_lock) - 1)
        self._b11_last_tick = int(tick)
        return (
            semantic_action,
            B11_CONFIDENCE_ARBITER_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b12_controller_params(self) -> dict[str, float]:
        params = self._b11_controller_params()
        defaults = {
            "b12_attention_decay": 0.84,
            "b12_attention_threshold": 0.18,
            "b12_prediction_error_threshold": 0.66,
            "b12_affordance_gain": 0.45,
            "b12_search_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "predictive_attention"
        )
        if profile == "active_inference_gate":
            defaults.update({"b12_prediction_error_threshold": 0.58, "b12_attention_threshold": 0.14})
        elif profile == "affordance_attention":
            defaults.update({"b12_affordance_gain": 0.66, "b12_attention_decay": 0.88})
        elif profile == "predictive_attention_h56":
            defaults.update({"b12_attention_decay": 0.87, "b12_search_commit_ticks": 6.0})
        elif profile == "genetic_attention_gate":
            defaults.update({"b12_attention_threshold": 0.16, "b12_affordance_gain": 0.58})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b12_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b12_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b12_attention_memory = 0.0
        self._b12_search_lock = 0
        self._b12_last_tick = int(tick)

    def _b12_predictive_attention_semantic_action(
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
        ) = self._b11_confidence_arbiter_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b12_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "predictive_attention"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b12_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b11_decision = str(trace_payload.get("b11_decision", "preserve_b11"))
        plan_confidence = float(trace_payload.get("b11_plan_confidence", 0.0) or 0.0)
        uncertainty = float(trace_payload.get("b11_uncertainty", 0.0) or 0.0)
        neuromod_signal = float(trace_payload.get("b11_neuromod_signal", 0.0) or 0.0)
        local_affordance = float(trace_payload.get("b8_local_affordance_score", 0.0) or 0.0)
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        expected_progress = float(np.clip(max(0.0, 16.0 - food_steps) / 16.0, 0.0, 1.0))
        prediction_error = float(
            np.clip(abs(plan_confidence - expected_progress) + uncertainty * 0.25, 0.0, 1.0)
        )
        attention_memory = max(
            float(getattr(self, "_b12_attention_memory", 0.0))
            * float(params["b12_attention_decay"]),
            neuromod_signal,
            max(0.0, local_affordance) * float(params["b12_affordance_gain"]),
        )
        attention_gain = float(np.clip(attention_memory + expected_progress * 0.35 - prediction_error * 0.20, 0.0, 1.0))
        search_lock = int(getattr(self, "_b12_search_lock", 0))
        attention_state = "non_corridor"
        decision_label = "preserve_b11"

        if corridor_map:
            if (
                b11_decision in {"commit_confident_plan", "continue_confidence_lock"}
                and attention_gain >= float(params["b12_attention_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                search_lock = max(search_lock, int(params["b12_search_commit_ticks"]))
                attention_state = "attended_food_affordance"
                decision_label = "commit_attended_affordance"
                reason = "b12_commit_attended_affordance"
            elif prediction_error >= float(params["b12_prediction_error_threshold"]):
                semantic_action = "MOVE_TO_SHELTER"
                attention_state = "prediction_error_return_check"
                decision_label = "gate_prediction_error"
                reason = "b12_gate_prediction_error"
            elif search_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                attention_state = "attention_lock_continues"
                decision_label = "continue_attention_lock"
                reason = "b12_continue_attention_lock"

        trace_payload.update(
            {
                "b12_controller_profile": profile,
                "b12_attention_state": attention_state,
                "b12_prediction_error": round(float(prediction_error), 6),
                "b12_attention_gain": round(float(attention_gain), 6),
                "b12_expected_progress": round(float(expected_progress), 6),
                "b12_search_lock": int(search_lock),
                "b12_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b12_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b12_genetic_candidate"] = int(params["ga_candidate"])

        self._b12_attention_memory = float(attention_memory)
        self._b12_search_lock = max(0, int(search_lock) - 1)
        self._b12_last_tick = int(tick)
        return (
            semantic_action,
            B12_PREDICTIVE_ATTENTION_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b13_controller_params(self) -> dict[str, float]:
        params = self._b12_controller_params()
        defaults = {
            "b13_search_memory_decay": 0.86,
            "b13_candidate_gain": 0.50,
            "b13_search_threshold": 0.20,
            "b13_dead_end_threshold": 0.68,
            "b13_local_commit_ticks": 5.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "local_affordance_search"
        )
        if profile == "counterfactual_route":
            defaults.update({"b13_candidate_gain": 0.64, "b13_search_threshold": 0.16})
        elif profile == "affordance_sampler":
            defaults.update({"b13_search_memory_decay": 0.90, "b13_local_commit_ticks": 7.0})
        elif profile == "local_affordance_search_h56":
            defaults.update({"b13_search_memory_decay": 0.88, "b13_local_commit_ticks": 6.0})
        elif profile == "genetic_local_search":
            defaults.update({"b13_candidate_gain": 0.58, "b13_search_threshold": 0.18})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b13_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b13_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b13_search_memory = 0.0
        self._b13_search_lock = 0
        self._b13_last_tick = int(tick)

    def _b13_local_affordance_search_semantic_action(
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
        ) = self._b12_predictive_attention_semantic_action(
            observation,
            learned_semantic_action=learned_semantic_action,
        )
        trace_payload = dict(trace_payload)
        params = self._b13_controller_params()
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "local_affordance_search"
        )
        tick = int(getattr(self, "_direct_policy_event_clock", -1))
        self._b13_reset_state_if_needed(tick)

        meta = observation.get("meta")
        meta = meta if isinstance(meta, dict) else {}
        corridor_map = str(meta.get("map_template", "")) == "corridor_escape"
        b12_decision = str(trace_payload.get("b12_decision", "preserve_b12"))
        attention_gain = float(trace_payload.get("b12_attention_gain", 0.0) or 0.0)
        prediction_error = float(trace_payload.get("b12_prediction_error", 0.0) or 0.0)
        expected_progress = float(trace_payload.get("b12_expected_progress", 0.0) or 0.0)
        local_affordance = float(trace_payload.get("b8_local_affordance_score", 0.0) or 0.0)
        dead_end_risk = float(trace_payload.get("b8_corridor_dead_end_risk", 0.0) or 0.0)
        food_steps = float(trace_payload.get("b7_food_steps_estimate", 0.0) or 0.0)
        food_step_bonus = float(np.clip(max(0.0, 18.0 - food_steps) / 18.0, 0.0, 1.0))
        candidate_score = float(
            np.clip(
                attention_gain * 0.45
                + expected_progress * 0.25
                + max(0.0, local_affordance) * float(params["b13_candidate_gain"])
                + food_step_bonus * 0.20,
                0.0,
                1.0,
            )
        )
        search_memory = max(
            float(getattr(self, "_b13_search_memory", 0.0))
            * float(params["b13_search_memory_decay"]),
            candidate_score,
        )
        dead_end_score = float(
            np.clip(
                prediction_error * 0.55
                + max(0.0, dead_end_risk) * 0.35
                + max(0.0, 1.0 - expected_progress) * 0.10,
                0.0,
                1.0,
            )
        )
        search_lock = int(getattr(self, "_b13_search_lock", 0))
        search_state = "non_corridor"
        decision_label = "preserve_b12"

        if corridor_map:
            if (
                b12_decision in {"commit_attended_affordance", "continue_attention_lock"}
                and search_memory >= float(params["b13_search_threshold"])
            ):
                semantic_action = "MOVE_TO_FOOD"
                search_lock = max(search_lock, int(params["b13_local_commit_ticks"]))
                search_state = "local_route_viable"
                decision_label = "commit_local_affordance_search"
                reason = "b13_commit_local_affordance_search"
            elif (
                dead_end_score >= float(params["b13_dead_end_threshold"])
                and b12_decision == "gate_prediction_error"
            ):
                semantic_action = "MOVE_TO_SHELTER"
                search_state = "local_dead_end_return"
                decision_label = "abort_local_dead_end"
                reason = "b13_abort_local_dead_end"
            elif search_lock > 0:
                semantic_action = "MOVE_TO_FOOD"
                search_state = "local_search_lock_continues"
                decision_label = "continue_local_search_lock"
                reason = "b13_continue_local_search_lock"

        trace_payload.update(
            {
                "b13_controller_profile": profile,
                "b13_search_state": search_state,
                "b13_local_route_score": round(float(search_memory), 6),
                "b13_affordance_samples": round(float(candidate_score), 6),
                "b13_search_memory": round(float(search_memory), 6),
                "b13_dead_end_score": round(float(dead_end_score), 6),
                "b13_search_lock": int(search_lock),
                "b13_decision": decision_label,
            }
        )
        if "ga_generation" in params:
            trace_payload["b13_genetic_generation"] = int(params["ga_generation"])
        if "ga_candidate" in params:
            trace_payload["b13_genetic_candidate"] = int(params["ga_candidate"])

        self._b13_search_memory = float(search_memory)
        self._b13_search_lock = max(0, int(search_lock) - 1)
        self._b13_last_tick = int(tick)
        return (
            semantic_action,
            B13_LOCAL_SEARCH_SELECTION_SOURCE,
            reason,
            int(semantic_action != learned_semantic_action),
            trace_payload,
        )

    def _b14_controller_params(self) -> dict[str, float]:
        params = self._b13_controller_params()
        defaults = {
            "b14_uncertainty_decay": 0.82,
            "b14_confidence_threshold": 0.42,
            "b14_uncertainty_threshold": 0.58,
            "b14_risk_gain": 0.40,
            "b14_commit_ticks": 4.0,
        }
        profile = str(
            getattr(self.config, "b_controller_profile", None)
            or "affordance_uncertainty"
        )
        if profile == "risk_calibrated_search":
            defaults.update({"b14_risk_gain": 0.56, "b14_uncertainty_threshold": 0.52})
        elif profile == "confidence_weighted_route":
            defaults.update({"b14_confidence_threshold": 0.34, "b14_commit_ticks": 5.0})
        elif profile == "affordance_uncertainty_h56":
            defaults.update({"b14_uncertainty_decay": 0.86, "b14_commit_ticks": 5.0})
        elif profile == "genetic_uncertainty_search":
            defaults.update({"b14_confidence_threshold": 0.38, "b14_risk_gain": 0.48})
        params.update(defaults)
        for key, value in dict(getattr(self.config, "b_controller_params", {})).items():
            params[str(key)] = float(value)
        return params

    def _b14_reset_state_if_needed(self, tick: int) -> None:
        last_tick = getattr(self, "_b14_last_tick", None)
        if last_tick is not None and int(tick) > int(last_tick):
            return
        self._b14_uncertainty_memory = 0.0
        self._b14_commitment_lock = 0
        self._b14_last_tick = int(tick)
