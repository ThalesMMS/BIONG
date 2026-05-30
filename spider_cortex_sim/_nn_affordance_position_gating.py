from __future__ import annotations

from ._nn_affordance_geometry import *


class _RecurrentOptionAffordancePositionGatingMixin:
    def _should_cooldown_terminated_option(self, termination_reason: str | None) -> bool:
        if not self.option_termination_cooldown or self.current_option_idx < 0:
            return False
        current_option = OPTION_NAMES[int(self.current_option_idx)]
        return (
            (current_option == "REST" and termination_reason == "recovery_completed")
            or (
                current_option == "RETURN_TO_SHELTER"
                and termination_reason == "shelter_exited"
            )
            or (current_option == "FORAGE" and termination_reason == "food_reached")
            or (
                current_option == "DEEPEN_IN_SHELTER"
                and termination_reason == "deep_shelter_reached"
            )
        )

    @staticmethod
    def _bounded_input_signal(x: Array, index: int) -> float:
        if index < 0 or index >= x.shape[0]:
            return 0.0
        return float(np.clip(x[index], -1.0, 1.0))

    def _executive_state_signals(self, x: Array) -> dict[str, float]:
        predator_visible = self._bounded_input_signal(
            x, _ALERT_PREDATOR_VISIBLE_IDX
        )
        predator_certainty = self._bounded_input_signal(
            x, _ALERT_PREDATOR_CERTAINTY_IDX
        )
        predator_smell_strength = self._bounded_input_signal(
            x, _ALERT_PREDATOR_SMELL_STRENGTH_IDX
        )
        predator_motion_salience = self._bounded_input_signal(
            x, _ALERT_PREDATOR_MOTION_SALIENCE_IDX
        )
        visual_predator_threat = self._bounded_input_signal(
            x, _ALERT_VISUAL_PREDATOR_THREAT_IDX
        )
        olfactory_predator_threat = self._bounded_input_signal(
            x, _ALERT_OLFACTORY_PREDATOR_THREAT_IDX
        )
        recent_pain = self._bounded_input_signal(x, _ALERT_RECENT_PAIN_IDX)
        recent_contact = self._bounded_input_signal(x, _ALERT_RECENT_CONTACT_IDX)
        predator_trace_strength = self._bounded_input_signal(
            x, _ALERT_PREDATOR_TRACE_STRENGTH_IDX
        )
        acute_threat = max(
            predator_visible * predator_certainty,
            predator_smell_strength,
            predator_motion_salience,
            visual_predator_threat,
            olfactory_predator_threat,
            recent_pain,
            recent_contact,
            predator_trace_strength,
        )
        night = self._bounded_input_signal(x, _SLEEP_NIGHT_IDX)
        fatigue = self._bounded_input_signal(x, _SLEEP_FATIGUE_IDX)
        sleep_debt = self._bounded_input_signal(x, _SLEEP_DEBT_IDX)
        return {
            "on_shelter": self._bounded_input_signal(x, _SLEEP_ON_SHELTER_IDX),
            "hunger": self._bounded_input_signal(x, _SLEEP_HUNGER_IDX),
            "fatigue": fatigue,
            "night": night,
            "sleep_phase_level": self._bounded_input_signal(
                x, _SLEEP_PHASE_LEVEL_IDX
            ),
            "rest_streak": self._bounded_input_signal(
                x, _SLEEP_REST_STREAK_IDX
            ),
            "sleep_debt": sleep_debt,
            "shelter_role_level": self._bounded_input_signal(
                x, _SLEEP_SHELTER_ROLE_LEVEL_IDX
            ),
            "shelter_memory_age": self._bounded_input_signal(
                x, _SLEEP_SHELTER_MEMORY_AGE_IDX
            ),
            "acute_threat": acute_threat,
            "rest_pressure": max(night, fatigue, sleep_debt),
            "food_visible": self._bounded_input_signal(x, _HUNGER_FOOD_VISIBLE_IDX),
            "food_certainty": self._bounded_input_signal(x, _HUNGER_FOOD_CERTAINTY_IDX),
            "food_dx": self._bounded_input_signal(x, _HUNGER_FOOD_DX_IDX),
            "food_dy": self._bounded_input_signal(x, _HUNGER_FOOD_DY_IDX),
            "food_smell_strength": self._bounded_input_signal(
                x, _HUNGER_FOOD_SMELL_STRENGTH_IDX
            ),
            "food_smell_dx": self._bounded_input_signal(x, _HUNGER_FOOD_SMELL_DX_IDX),
            "food_smell_dy": self._bounded_input_signal(x, _HUNGER_FOOD_SMELL_DY_IDX),
            "food_memory_dx": self._bounded_input_signal(x, _HUNGER_FOOD_MEMORY_DX_IDX),
            "food_memory_dy": self._bounded_input_signal(x, _HUNGER_FOOD_MEMORY_DY_IDX),
            "food_memory_age": self._bounded_input_signal(
                x, _HUNGER_FOOD_MEMORY_AGE_IDX
            ),
        }

    def _apply_executive_physiology_option_gating(
        self,
        x: Array,
        selection_option_logits: Array,
    ) -> Array:
        if not self.executive_physiology_option_gating:
            return selection_option_logits
        signals = self._executive_state_signals(x)
        on_shelter = signals["on_shelter"]
        hunger = signals["hunger"]
        sleep_phase_level = signals["sleep_phase_level"]
        rest_streak = signals["rest_streak"]
        shelter_role_level = signals["shelter_role_level"]
        shelter_memory_age = signals["shelter_memory_age"]
        acute_threat = signals["acute_threat"]
        rest_pressure = signals["rest_pressure"]
        sheltered = on_shelter > 0.5
        fresh_shelter_memory = shelter_memory_age < 0.95
        gated_logits = selection_option_logits.copy()

        if self.executive_post_food_return and self.executive_post_food_return_steps_remaining > 0:
            if sheltered:
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] += 8.0
                gated_logits[OPTION_TO_INDEX["REST"]] += 6.0
                gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 8.0
                gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 8.0
            else:
                gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 10.0
                gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 8.0
                gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 6.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 4.0
            return gated_logits

        if sheltered:
            if acute_threat >= 0.2:
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] += 4.0
                gated_logits[OPTION_TO_INDEX["REST"]] += 2.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 6.0
            elif rest_pressure >= 0.18 and hunger <= 0.22:
                gated_logits[OPTION_TO_INDEX["REST"]] += 6.0
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] += 3.0
                gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 6.0
                gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 4.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 8.0
            elif (
                hunger >= 0.16
                and rest_pressure <= 0.12
                and sleep_phase_level <= 0.2
                and rest_streak >= 0.1
            ):
                gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] += 6.0
                gated_logits[OPTION_TO_INDEX["FORAGE"]] += 2.0
                gated_logits[OPTION_TO_INDEX["REST"]] -= 4.0
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 3.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 8.0
            else:
                gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] += 2.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 6.0
            return gated_logits

        if acute_threat >= 0.2:
            gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 6.0
            gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 6.0
            gated_logits[OPTION_TO_INDEX["REST"]] -= 8.0
            gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 8.0
            if fresh_shelter_memory:
                gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 6.0
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 2.0
            else:
                gated_logits[OPTION_TO_INDEX["ESCAPE"]] += 4.0
                gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 2.0
            return gated_logits

        if rest_pressure >= 0.18 and fresh_shelter_memory:
            gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] += 5.0
            gated_logits[OPTION_TO_INDEX["FORAGE"]] -= 3.0
            gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] -= 2.0
            return gated_logits

        if hunger >= 0.16:
            gated_logits[OPTION_TO_INDEX["FORAGE"]] += 4.0
            gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] += 3.0
            gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 4.0
            gated_logits[OPTION_TO_INDEX["REST"]] -= 6.0
            gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 6.0
            if fresh_shelter_memory:
                gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] -= 1.0

        if shelter_role_level < 0.25:
            gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 4.0
        return gated_logits

    def _prime_executive_release_latch(
        self,
        x: Array,
        termination_reason: str | None,
    ) -> None:
        if not self.executive_event_release_latching:
            return
        signals = self._executive_state_signals(x)
        sheltered = signals["on_shelter"] > 0.5
        hunger = signals["hunger"]
        acute_threat = signals["acute_threat"]
        role_level = signals["shelter_role_level"]
        if (
            not sheltered
            or acute_threat >= 0.2
            or hunger < 0.12
            or termination_reason in {"shelter_exited", "acute_predator_threat"}
        ):
            self.executive_release_steps_remaining = 0
            return
        if termination_reason == "recovery_completed":
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                2,
            )
            return
        if (
            termination_reason in {"deep_shelter_reached", "no_progress"}
            and role_level >= 0.95
        ):
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                2,
            )

    def _prime_executive_release_phase_state(
        self,
        x: Array,
        termination_reason: str | None,
    ) -> None:
        if not self.executive_release_phase_state:
            return
        signals = self._executive_state_signals(x)
        sheltered = signals["on_shelter"] > 0.5
        hunger = signals["hunger"]
        acute_threat = signals["acute_threat"]
        rest_pressure = signals["rest_pressure"]
        role_level = signals["shelter_role_level"]
        if not sheltered or acute_threat >= 0.2:
            self.executive_release_steps_remaining = 0
            return
        if self.executive_release_steps_remaining > 0:
            return
        if (
            self.current_option_idx < 0
            and hunger >= 0.08
            and rest_pressure <= 0.32
        ):
            self.executive_release_steps_remaining = 3
            return
        if (
            termination_reason in {"recovery_completed", "deep_shelter_reached"}
            and hunger >= 0.12
        ):
            self.executive_release_steps_remaining = 3
            return
        if (
            termination_reason == "no_progress"
            and role_level >= 0.95
            and hunger >= 0.12
        ):
            self.executive_release_steps_remaining = 3

    def _apply_executive_event_release_latch(
        self,
        x: Array,
        selection_option_logits: Array,
    ) -> Array:
        if (
            not self.executive_event_release_latching
            or self.executive_release_steps_remaining <= 0
        ):
            return selection_option_logits
        signals = self._executive_state_signals(x)
        if signals["on_shelter"] <= 0.5 or signals["acute_threat"] >= 0.2:
            self.executive_release_steps_remaining = 0
            return selection_option_logits
        gated_logits = selection_option_logits.copy()
        gated_logits[OPTION_TO_INDEX["POST_REST_REACTIVATE"]] += 10.0
        gated_logits[OPTION_TO_INDEX["FORAGE"]] += 3.0
        gated_logits[OPTION_TO_INDEX["REST"]] -= 8.0
        gated_logits[OPTION_TO_INDEX["DEEPEN_IN_SHELTER"]] -= 8.0
        gated_logits[OPTION_TO_INDEX["RETURN_TO_SHELTER"]] -= 4.0
        gated_logits[OPTION_TO_INDEX["ESCAPE"]] -= 8.0
        return gated_logits

    def _apply_executive_release_substate_progression(
        self,
        x: Array,
        selected_option_idx: int,
    ) -> None:
        if (
            not self.executive_release_substate_progression
            or self.executive_release_steps_remaining <= 0
            or OPTION_NAMES[selected_option_idx] != "POST_REST_REACTIVATE"
        ):
            return
        signals = self._executive_state_signals(x)
        if signals["on_shelter"] <= 0.5 or signals["acute_threat"] >= 0.2:
            self.executive_release_steps_remaining = 0
            return
        if signals["hunger"] < 0.12:
            return
        role_level = signals["shelter_role_level"]
        if role_level >= 0.95:
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                3,
            )
        elif role_level >= 0.55:
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                2,
            )
        elif role_level > 0.05:
            self.executive_release_steps_remaining = max(
                self.executive_release_steps_remaining,
                2,
            )

    def _apply_executive_affordance_action_gating(
        self,
        x: Array,
        selected_option_idx: int,
        policy_logits: Array,
        blocked_probs: Array,
        geometry_probs: Array,
        shelter_position_probs: Array,
    ) -> Array:
        if not self.executive_affordance_action_gating:
            return policy_logits
        signals = self._executive_state_signals(x)
        sheltered = signals["on_shelter"] > 0.5
        hunger = signals["hunger"]
        acute_threat = signals["acute_threat"]
        rest_pressure = signals["rest_pressure"]
        role_level = signals["shelter_role_level"]
        option_name = OPTION_NAMES[selected_option_idx]
        gated_logits = policy_logits.copy()
        orientation_indices = _POLICY_ORIENTATION_ACTION_INDICES
        locomotion_indices = tuple(_LOCAL_ACTION_TO_POLICY_INDEX.values())
        geometry_matrix = geometry_probs.reshape(
            self.output_dim,
            len(AFFORDANCE_GEOMETRY_TARGET_NAMES),
        )

        def action_score(action_name: str) -> float:
            action_idx = _LOCAL_ACTION_TO_POLICY_INDEX[action_name]
            blocked_penalty = 8.0 * float(blocked_probs[action_idx])
            pos_probs = shelter_position_probs[action_idx]
            deep_prob = float(sum(pos_probs[idx] for idx in _DEEP_SHELTER_POSITION_INDICES))
            inside_prob = float(
                sum(pos_probs[idx] for idx in _INSIDE_SHELTER_POSITION_INDICES)
            )
            entrance_prob = float(
                sum(pos_probs[idx] for idx in _ENTRANCE_POSITION_INDICES)
            )
            outside_prob = float(pos_probs[_OUTSIDE_POSITION_INDEX])
            deepen_prob = float(
                geometry_matrix[action_idx, _GEOMETRY_DEEPEN_INDEX]
            )
            outside_geom = float(
                geometry_matrix[action_idx, _GEOMETRY_OUTSIDE_INDEX]
            )
            if option_name == "POST_REST_REACTIVATE":
                return outside_prob * 8.0 + entrance_prob * 3.0 + outside_geom * 4.0 - blocked_penalty
            if option_name == "FORAGE":
                return outside_prob * 6.0 + entrance_prob * 2.0 + outside_geom * 3.0 - blocked_penalty
            if option_name == "RETURN_TO_SHELTER":
                return deep_prob * 5.0 + inside_prob * 3.0 + entrance_prob * 2.0 + deepen_prob * 3.0 - blocked_penalty
            if option_name == "DEEPEN_IN_SHELTER":
                return deep_prob * 8.0 + inside_prob * 2.0 + deepen_prob * 5.0 - blocked_penalty
            if option_name == "REST":
                return 5.0 if action_name == "STAY" else -blocked_penalty
            return -blocked_penalty

        def promote_local_action(action_name: str, bonus: float) -> None:
            action_idx = _LOCAL_ACTION_TO_POLICY_INDEX[action_name]
            gated_logits[action_idx] += bonus

        def guided_food_action() -> str | None:
            if not self.executive_post_exit_food_guidance:
                return None
            dx = 0.0
            dy = 0.0
            weight = 0.0
            if signals["food_visible"] > 0.1 and signals["food_certainty"] > 0.1:
                dx = signals["food_dx"]
                dy = signals["food_dy"]
                weight = signals["food_certainty"]
            elif signals["food_smell_strength"] > 0.1:
                dx = signals["food_smell_dx"]
                dy = signals["food_smell_dy"]
                weight = signals["food_smell_strength"]
            elif signals["food_memory_age"] < 0.75:
                dx = signals["food_memory_dx"]
                dy = signals["food_memory_dy"]
                weight = 1.0 - max(0.0, signals["food_memory_age"])
            if weight <= 0.05:
                return None
            if abs(dx) >= abs(dy):
                if dx > 0.05:
                    return "MOVE_RIGHT"
                if dx < -0.05:
                    return "MOVE_LEFT"
            if dy > 0.05:
                return "MOVE_DOWN"
            if dy < -0.05:
                return "MOVE_UP"
            return None

        def guided_food_action_with_source() -> tuple[str | None, str | None]:
            if not self.executive_post_exit_food_guidance:
                return None, None
            if signals["food_visible"] > 0.1 and signals["food_certainty"] > 0.1:
                dx = signals["food_dx"]
                dy = signals["food_dy"]
                source = "visible"
            elif signals["food_smell_strength"] > 0.1:
                dx = signals["food_smell_dx"]
                dy = signals["food_smell_dy"]
                source = "smell"
            elif signals["food_memory_age"] < 0.75:
                dx = signals["food_memory_dx"]
                dy = signals["food_memory_dy"]
                source = "memory"
            else:
                return None, None
            action = guided_food_action()
            return action, source

        def has_live_food_guidance() -> bool:
            return guided_food_action() is not None

        def guided_shelter_return_action() -> str | None:
            if not self.executive_post_food_vector_return:
                return None
            meta = self.executive_runtime_meta if isinstance(self.executive_runtime_meta, dict) else {}
            memory_vectors = meta.get("memory_vectors", {})
            shelter_vector = (
                memory_vectors.get("shelter", {})
                if isinstance(memory_vectors, dict)
                else {}
            )
            dx = float(shelter_vector.get("dx", 0.0) or 0.0)
            dy = float(shelter_vector.get("dy", 0.0) or 0.0)
            if abs(dx) >= abs(dy):
                if dx > 0.05:
                    action_name = "MOVE_RIGHT"
                elif dx < -0.05:
                    action_name = "MOVE_LEFT"
                else:
                    action_name = None
            else:
                if dy > 0.05:
                    action_name = "MOVE_DOWN"
                elif dy < -0.05:
                    action_name = "MOVE_UP"
                else:
                    action_name = None
            if action_name is None:
                return None
            if float(blocked_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]]) > 0.5:
                return None
            return action_name

        def guided_shelter_path_return_action() -> str | None:
            if not self.executive_post_food_path_return:
                return None
            while self.executive_post_food_return_queue:
                action_idx = int(self.executive_post_food_return_queue[0])
                if action_idx not in _LOCAL_ACTION_TO_POLICY_INDEX.values():
                    self.executive_post_food_return_queue.pop(0)
                    continue
                blocked = float(blocked_probs[action_idx]) > 0.5
                if blocked:
                    return None
                action_name = next(
                    name
                    for name, idx in _LOCAL_ACTION_TO_POLICY_INDEX.items()
                    if idx == action_idx
                )
                return action_name
            return None

        if (
            self.executive_release_exit_contract
            and self.executive_release_steps_remaining > 0
            and sheltered
            and acute_threat < 0.2
            and hunger >= 0.12
        ):
            if self.executive_option_action_masking:
                gated_logits[:] = -20.0
            if role_level <= 0.4:
                promote_local_action("MOVE_LEFT", 16.0)
                promote_local_action("MOVE_UP", 10.0)
            elif role_level >= 0.95:
                promote_local_action("MOVE_UP", 14.0)
                promote_local_action("MOVE_LEFT", 5.0)
            else:
                promote_local_action("MOVE_UP", 12.0)
                promote_local_action("MOVE_LEFT", 8.0)
            for idx in orientation_indices:
                gated_logits[idx] -= 10.0
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_RIGHT"]] -= 10.0
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]] -= 8.0
            return gated_logits

        if (
            self.executive_release_progression
            and self.executive_release_steps_remaining > 0
            and sheltered
            and acute_threat < 0.2
            and hunger >= 0.12
        ):
            pos_scores = {
                name: float(
                    shelter_position_probs[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]][idx]
                )
                for idx, name in enumerate(AFFORDANCE_SHELTER_POSITION_NAMES)
            }
            if self.executive_option_action_masking:
                gated_logits[:] = -20.0
            entrance_prob = max(
                pos_scores.get("entrance_left", 0.0),
                pos_scores.get("entrance_center", 0.0),
                pos_scores.get("entrance_right", 0.0),
            )
            if entrance_prob >= 0.3 or role_level <= 0.4:
                promote_local_action("MOVE_RIGHT", 14.0)
                promote_local_action("MOVE_UP", 4.0)
            elif role_level >= 0.95:
                promote_local_action("MOVE_UP", 14.0)
                promote_local_action("MOVE_RIGHT", 3.0)
            else:
                promote_local_action("MOVE_UP", 12.0)
                promote_local_action("MOVE_RIGHT", 6.0)
            for idx in orientation_indices:
                gated_logits[idx] -= 10.0
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]] -= 8.0
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_LEFT"]] -= 8.0
            return gated_logits

        if (
            self.executive_event_release_action_commitment
            and self.executive_release_steps_remaining > 0
            and sheltered
            and acute_threat < 0.2
            and hunger >= 0.12
        ):
            if self.executive_option_action_masking:
                gated_logits[:] = -20.0
            best_action = max(
                DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                key=lambda action_name: (
                    action_score(action_name)
                    + (1.5 if action_name == "MOVE_UP" else 0.0)
                ),
            )
            promote_local_action(best_action, 12.0)
            for idx in orientation_indices:
                gated_logits[idx] -= 8.0
            return gated_logits

        if (
            self.executive_post_exit_continuation
            and self.executive_post_exit_steps_remaining > 0
            and not sheltered
            and acute_threat < 0.2
            and hunger >= 0.12
            and option_name == "POST_REST_REACTIVATE"
        ):
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
                gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] = -20.0
            best_action, guidance_source = guided_food_action_with_source()
            if (
                self.executive_post_exit_corridor_progression
                and self.executive_post_exit_corridor_steps_remaining > 0
                and best_action is None
            ):
                if self.executive_post_exit_corridor_affordance_progression:
                    move_down_idx = _LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]
                    move_right_idx = _LOCAL_ACTION_TO_POLICY_INDEX["MOVE_RIGHT"]
                    move_down_blocked = float(blocked_probs[move_down_idx]) > 0.5
                    move_right_blocked = float(blocked_probs[move_right_idx]) > 0.5
                    move_down_pos = shelter_position_probs[move_down_idx]
                    move_down_sheltered = float(
                        sum(move_down_pos[idx] for idx in _DEEP_SHELTER_POSITION_INDICES)
                        + sum(move_down_pos[idx] for idx in _INSIDE_SHELTER_POSITION_INDICES)
                        + sum(move_down_pos[idx] for idx in _ENTRANCE_POSITION_INDICES)
                    ) > 0.5
                    best_action = (
                        "MOVE_RIGHT"
                        if (not move_right_blocked and (move_down_blocked or move_down_sheltered))
                        else "MOVE_DOWN"
                    )
                else:
                    best_action = (
                        "MOVE_RIGHT"
                        if self.executive_post_exit_corridor_steps_remaining > 3
                        else "MOVE_DOWN"
                    )
            elif best_action is None:
                best_action = max(
                    DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                    key=action_score,
                )
            elif (
                self.executive_post_exit_food_heading_progression
                and guidance_source == "memory"
                and self.previous_action_idx == ACTION_TO_INDEX["MOVE_RIGHT"]
                and best_action == "MOVE_RIGHT"
            ):
                best_action = "MOVE_UP"
            elif (
                self.executive_post_exit_smell_progression
                and guidance_source == "smell"
                and self.previous_action_idx == ACTION_TO_INDEX["MOVE_RIGHT"]
                and best_action == "MOVE_RIGHT"
            ):
                best_action = "MOVE_UP"
            elif (
                self.executive_post_exit_food_progression
                and guidance_source == "memory"
                and self.previous_action_idx >= 0
            ):
                previous_action = DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES[
                    list(_LOCAL_ACTION_TO_POLICY_INDEX.values()).index(self.previous_action_idx)
                ] if self.previous_action_idx in _LOCAL_ACTION_TO_POLICY_INDEX.values() else None
                if previous_action == "MOVE_LEFT" and best_action == "MOVE_RIGHT":
                    best_action = "MOVE_UP"
                elif previous_action == "MOVE_RIGHT" and best_action == "MOVE_LEFT":
                    best_action = "MOVE_UP"
            promote_local_action(best_action, 10.0)
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] -= 6.0
            if self.executive_post_exit_food_commitment and has_live_food_guidance():
                gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]] -= 6.0
            if self.executive_post_exit_food_progression and guidance_source == "memory":
                gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["MOVE_DOWN"]] -= 4.0
            for idx in orientation_indices:
                gated_logits[idx] -= 8.0
            return gated_logits

        if (
            self.executive_post_food_return
            and self.executive_post_food_return_steps_remaining > 0
            and acute_threat < 0.2
            and option_name in {"RETURN_TO_SHELTER", "DEEPEN_IN_SHELTER", "REST"}
        ):
            if sheltered and signals["shelter_role_level"] >= 0.95:
                if self.executive_option_action_masking:
                    gated_logits[:] = -20.0
                promote_local_action("STAY", 12.0)
                for idx in orientation_indices:
                    gated_logits[idx] -= 8.0
                return gated_logits
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
            if not sheltered:
                best_action = guided_shelter_path_return_action()
                if best_action is None:
                    best_action = guided_shelter_return_action()
                if best_action is None:
                    best_action = max(
                        DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                        key=lambda action_name: (
                            float(
                                sum(
                                    shelter_position_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]][idx]
                                    for idx in _DEEP_SHELTER_POSITION_INDICES
                                )
                            ) * 5.0
                            + float(
                                sum(
                                    shelter_position_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]][idx]
                                    for idx in _INSIDE_SHELTER_POSITION_INDICES
                                )
                            ) * 3.0
                            + float(
                                sum(
                                    shelter_position_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]][idx]
                                    for idx in _ENTRANCE_POSITION_INDICES
                                )
                            ) * 2.0
                            - 8.0 * float(blocked_probs[_LOCAL_ACTION_TO_POLICY_INDEX[action_name]])
                        ),
                    )
            else:
                best_action = max(
                    DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                    key=action_score,
                )
            promote_local_action(best_action, 10.0)
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] -= 4.0
            for idx in orientation_indices:
                gated_logits[idx] -= 8.0
            return gated_logits

        if option_name == "REST" and sheltered:
            if self.executive_option_action_masking:
                gated_logits[:] = -20.0
            promote_local_action("STAY", 10.0)
            for idx in locomotion_indices:
                if idx != _LOCAL_ACTION_TO_POLICY_INDEX["STAY"]:
                    gated_logits[idx] -= 4.0
            for idx in orientation_indices:
                gated_logits[idx] -= 8.0
            return gated_logits

        if (
            option_name == "POST_REST_REACTIVATE"
            and sheltered
            and acute_threat < 0.2
            and hunger >= 0.14
            and rest_pressure <= 0.18
        ):
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
                gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] = -20.0
            best_action = max(
                DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                key=action_score,
            )
            promote_local_action(best_action, 8.0)
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] -= 5.0
            for idx in orientation_indices:
                gated_logits[idx] -= 6.0
            return gated_logits

        if option_name in {"RETURN_TO_SHELTER", "DEEPEN_IN_SHELTER"} and (
            not sheltered or role_level < 0.95
        ):
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
            best_action = max(
                DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                key=action_score,
            )
            promote_local_action(best_action, 7.0)
            for idx in orientation_indices:
                gated_logits[idx] -= 5.0
            return gated_logits

        if option_name == "FORAGE" and sheltered and acute_threat < 0.2:
            if self.executive_option_action_masking:
                for idx in orientation_indices:
                    gated_logits[idx] = -20.0
            best_action = max(
                DIRECT_POLICY_LOCAL_AFFORDANCE_ACTION_NAMES,
                key=action_score,
            )
            promote_local_action(best_action, 5.0)
            gated_logits[_LOCAL_ACTION_TO_POLICY_INDEX["STAY"]] -= 2.0
            for idx in orientation_indices:
                gated_logits[idx] -= 4.0
        return gated_logits
