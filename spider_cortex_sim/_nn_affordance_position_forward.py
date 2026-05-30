from __future__ import annotations

from ._nn_affordance_geometry import *


class _RecurrentOptionAffordancePositionForwardMixin:
    def forward(
        self,
        x: Array,
        *,
        store_cache: bool = True,
    ) -> tuple[Array, float, Array] | tuple[Array, float, Array, Array]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.input_dim,):
            raise ValueError(
                f"{self.name}: x expected shape {(self.input_dim,)}, received {x.shape}"
            )
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        h_prev = self.hidden_state.copy()
        (
            event_context,
            query_input,
            query,
            slot_raws,
            keys,
            values,
            attention_weights,
            event_type_indices,
        ) = self._attention_context(x, h_prev)
        previous_option_vector = self._current_option_vector()
        previous_option_idx = (
            int(np.argmax(previous_option_vector))
            if previous_option_vector.sum() > 0.0
            else -1
        )
        x_aug = np.concatenate([x, event_context, previous_option_vector], axis=0)
        recurrent_pre = self.W_xh @ x_aug + self.W_hh @ h_prev + self.b_h
        if self.option_recurrent_dynamics and previous_option_idx >= 0:
            recurrent_pre = (
                recurrent_pre
                + self.W_option_recurrent_dynamics[previous_option_idx] @ h_prev
                + self.b_option_recurrent_dynamics[previous_option_idx]
            )
        h_new = np.tanh(recurrent_pre)
        blocked_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_blocked @ h_new + self.b2_affordance_blocked,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        role_logits = np.clip(
            np.nan_to_num(
                self.W2_affordance_role @ h_new + self.b2_affordance_role,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        geometry_logits = np.clip(
            np.nan_to_num(
                self.W2_geometry @ h_new + self.b2_geometry,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        shelter_position_logits = np.clip(
            np.nan_to_num(
                self.W2_shelter_position @ h_new + self.b2_shelter_position,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        blocked_probs = _sigmoid(blocked_logits)
        role_logits_matrix = role_logits.reshape(self.output_dim, self.affordance_role_dim)
        role_probs = np.vstack(
            [softmax(role_logits_matrix[action_idx]) for action_idx in range(self.output_dim)]
        )
        geometry_probs = _sigmoid(geometry_logits)
        shelter_position_logits_matrix = shelter_position_logits.reshape(
            self.output_dim,
            self.shelter_position_dim,
        )
        shelter_position_probs = np.vstack(
            [
                softmax(shelter_position_logits_matrix[action_idx])
                for action_idx in range(self.output_dim)
            ]
        )
        transition_prediction_logits = np.tanh(
            np.nan_to_num(
                self.W2_transition_prediction @ h_new + self.b2_transition_prediction,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        transition_rollout_prediction_logits = np.tanh(
            np.nan_to_num(
                self.W2_transition_rollout_prediction @ h_new
                + self.b2_transition_rollout_prediction,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            )
        )
        affordance_features = np.concatenate(
            [
                blocked_probs,
                role_probs.reshape(-1),
                geometry_probs,
                shelter_position_probs.reshape(-1),
            ],
            axis=0,
        )
        affordance_feedback = np.tanh(
            self.W_affordance_feedback @ affordance_features
            + self.b_affordance_feedback
        )
        transition_prediction_feedback = np.zeros(self.hidden_dim, dtype=float)
        if self.transition_prediction_feedback:
            transition_prediction_feedback = np.tanh(
                self.W_transition_prediction_feedback @ transition_prediction_logits
                + self.b_transition_prediction_feedback
            )
        transition_rollout_prediction_feedback = np.zeros(
            self.hidden_dim,
            dtype=float,
        )
        if self.transition_rollout_prediction_feedback:
            transition_rollout_prediction_feedback = np.tanh(
                self.W_transition_rollout_prediction_feedback
                @ transition_rollout_prediction_logits
                + self.b_transition_rollout_prediction_feedback
            )
        combined_feedback = (
            affordance_feedback
            + transition_prediction_feedback
            + transition_rollout_prediction_feedback
        )
        base_policy_logits = np.clip(
            np.nan_to_num(
                self.W2_policy @ h_new + self.b2_policy,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        option_logits = np.clip(
            np.nan_to_num(
                self.W2_option @ h_new
                + self.b2_option
                + self.W2_option_feedback @ combined_feedback
                + self.b2_option_feedback,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        phase_logits: Array | None = None
        phase_probs = np.zeros(self.phase_output_dim, dtype=float)
        if self.phase_output_dim > 0:
            phase_logits = np.clip(
                np.nan_to_num(
                    self.W2_phase @ h_new + self.b2_phase,
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ),
                -20.0,
                20.0,
            )
            phase_probs = softmax(phase_logits)
        if self.phase_option_feedback:
            option_logits = np.clip(
                np.nan_to_num(
                    option_logits
                    + self.W2_phase_option_feedback @ phase_probs
                    + self.b2_phase_option_feedback,
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ),
                -20.0,
                20.0,
            )
        if self.option_transition_feedback:
            option_logits = np.clip(
                np.nan_to_num(
                    option_logits
                    + self.W2_option_transition_feedback @ previous_option_vector
                    + self.b2_option_transition_feedback,
                    nan=0.0,
                    posinf=20.0,
                    neginf=-20.0,
                ),
                -20.0,
                20.0,
            )
        termination_reason = self._termination_reason()
        termination_reason = self._apply_executive_post_exit_continuation(
            x,
            termination_reason,
        )
        self._prime_executive_post_food_return(x, termination_reason)
        if self._should_cooldown_terminated_option(termination_reason):
            self.option_cooldowns[int(self.current_option_idx)] = self.option_ttl
        self._prime_executive_release_phase_state(x, termination_reason)
        self._prime_executive_release_latch(x, termination_reason)
        selection_option_logits = option_logits.copy()
        if self.option_termination_cooldown:
            cooldown_mask = self.option_cooldowns > 0
            if np.any(cooldown_mask):
                selection_option_logits[cooldown_mask] = -20.0
        selection_option_logits = self._apply_executive_physiology_option_gating(
            x,
            selection_option_logits,
        )
        selection_option_logits = self._apply_executive_event_release_latch(
            x,
            selection_option_logits,
        )
        option_probs = softmax(selection_option_logits)
        started_new_option = self.current_option_idx < 0 or termination_reason is not None
        if started_new_option:
            selected_option_idx = int(np.argmax(option_probs))
            self.current_option_idx = selected_option_idx
            self.current_option_age = 0
            self.current_option_steps_remaining = self.option_ttl
        else:
            selected_option_idx = int(self.current_option_idx)
            self.current_option_age += 1
        self._apply_executive_release_substate_progression(x, selected_option_idx)
        self.current_option_steps_remaining = max(
            0,
            int(self.current_option_steps_remaining) - 1,
        )
        if self.executive_event_release_latching:
            if (
                OPTION_NAMES[selected_option_idx] == "POST_REST_REACTIVATE"
                and self.executive_release_steps_remaining > 0
            ):
                self.executive_release_steps_remaining = max(
                    0,
                    int(self.executive_release_steps_remaining) - 1,
                )
            elif termination_reason == "shelter_exited":
                self.executive_release_steps_remaining = 0
        if self.executive_post_exit_continuation:
            signals = self._executive_state_signals(x)
            if (
                OPTION_NAMES[selected_option_idx] == "POST_REST_REACTIVATE"
                and self.executive_post_exit_steps_remaining > 0
                and signals["on_shelter"] <= 0.5
                and signals["acute_threat"] < 0.2
            ):
                self.executive_post_exit_steps_remaining = max(
                    0,
                    int(self.executive_post_exit_steps_remaining) - 1,
                )
                self.executive_post_exit_corridor_steps_remaining = max(
                    0,
                    int(self.executive_post_exit_corridor_steps_remaining) - 1,
                )
            elif signals["on_shelter"] > 0.5 or termination_reason == "food_reached":
                self.executive_post_exit_steps_remaining = 0
                self.executive_post_exit_corridor_steps_remaining = 0
        if self.executive_post_food_return:
            signals = self._executive_state_signals(x)
            if (
                self.executive_post_food_return_steps_remaining > 0
                or self.executive_post_food_return_queue
            ) and (
                signals["acute_threat"] >= 0.2
                or termination_reason == "recovery_completed"
            ):
                self.executive_post_food_return_steps_remaining = 0
                self.executive_post_food_return_queue = []
                self.executive_post_food_path_history = []
            elif self.executive_post_food_return_steps_remaining > 0:
                if signals["on_shelter"] > 0.5 and signals["shelter_role_level"] >= 0.95:
                    self.executive_post_food_return_steps_remaining = max(
                        self.executive_post_food_return_steps_remaining,
                        2,
                    )
                else:
                    self.executive_post_food_return_steps_remaining = max(
                        0,
                        int(self.executive_post_food_return_steps_remaining) - 1,
                    )
                if self.executive_post_food_return_steps_remaining <= 0:
                    self.executive_post_food_return_queue = []
                    self.executive_post_food_path_history = []
        if self.option_termination_cooldown:
            self.option_cooldowns = np.maximum(0, self.option_cooldowns - 1)
        option_vector = one_hot(selected_option_idx, self.option_dim)
        selected_option_age_bucket = min(
            max(int(self.current_option_age), 0),
            max(self.option_ttl - 1, 0),
        )
        previous_action_vector = (
            np.zeros(self.output_dim, dtype=float)
            if started_new_option or self.previous_action_idx < 0
            else one_hot(int(self.previous_action_idx), self.output_dim)
        )
        previous_decoder_action_state = (
            np.zeros(self.hidden_dim, dtype=float)
            if started_new_option
            else self.decoder_action_state.copy()
        )
        previous_action_backbone_state = self.action_backbone_state.copy()
        previous_action_policy_state = self.action_policy_state.copy()
        previous_action_controller_state = self.action_controller_state.copy()
        previous_action_token_state = self.action_token_state.copy()
        decoder_hidden_pre = h_new.copy()
        if self.option_decoder_state:
            decoder_hidden_pre = (
                h_new
                + self.W_option_decoder_state[selected_option_idx] @ h_new
                + self.b_option_decoder_state[selected_option_idx]
            )
        if self.option_decoder_recurrent_state:
            decoder_hidden_pre = (
                decoder_hidden_pre
                + self.W_option_decoder_recurrent_state[selected_option_idx]
                @ previous_decoder_action_state
                + self.b_option_decoder_recurrent_state[selected_option_idx]
            )
        if self.option_action_transition_state:
            decoder_hidden_pre = (
                decoder_hidden_pre
                + self.W_option_action_transition_state[selected_option_idx]
                @ previous_action_vector
                + self.b_option_action_transition_state[selected_option_idx]
            )
        decoder_hidden = np.tanh(decoder_hidden_pre)
        action_backbone_pre = np.zeros(self.hidden_dim, dtype=float)
        action_backbone_state = np.zeros(self.hidden_dim, dtype=float)
        if self.option_action_separate_backbone:
            action_backbone_pre = (
                self.W_action_backbone_input @ x
                + self.W_action_backbone_prev @ previous_action_backbone_state
                + self.W_action_backbone_action @ previous_action_vector
                + self.b_action_backbone
            )
            action_backbone_state = np.tanh(action_backbone_pre)
        action_policy_source = (
            h_new
            if (
                self.option_action_separate_recurrent_head
                or self.option_action_separate_policy_path
            )
            else decoder_hidden
        )
        action_policy_pre = action_policy_source.copy()
        if self.option_action_separate_policy_path:
            action_policy_pre = (
                self.W_action_policy_path_input @ action_policy_source
                + self.W_action_policy_path_prev @ previous_action_policy_state
                + self.W_action_policy_path_action @ previous_action_vector
                + self.b_action_policy_path
            )
        elif self.option_action_recurrent_core:
            action_policy_pre = (
                self.W_option_action_policy_decoder[selected_option_idx]
                @ action_policy_source
                + self.W_option_action_policy_prev[selected_option_idx]
                @ previous_action_policy_state
                + self.W_option_action_policy_action[selected_option_idx]
                @ previous_action_vector
                + self.b_option_action_policy[selected_option_idx]
            )
        action_policy_state = np.tanh(action_policy_pre)
        action_token_pre = decoder_hidden.copy()
        if self.option_action_token_decoder:
            action_token_pre = (
                self.W_option_action_token_decoder[selected_option_idx]
                @ decoder_hidden
                + self.W_option_action_token_prev[selected_option_idx]
                @ previous_action_token_state
                + self.W_option_action_token_action[selected_option_idx]
                @ previous_action_vector
                + self.b_option_action_token[selected_option_idx]
            )
        action_token_state = np.tanh(action_token_pre)
        policy_core = (
            action_token_state
            if self.option_action_token_decoder
            else (
                action_backbone_state
                if self.option_action_separate_backbone
                else (
                action_policy_state
                if self.option_action_recurrent_core
                else decoder_hidden
                )
            )
        )
        action_controller_pre = decoder_hidden.copy()
        if self.option_action_controller_state:
            action_controller_pre = (
                self.W_option_action_controller_decoder[selected_option_idx]
                @ decoder_hidden
                + self.W_option_action_controller_prev[selected_option_idx]
                @ previous_action_controller_state
                + self.W_option_action_controller_action[selected_option_idx]
                @ previous_action_vector
                + self.b_option_action_controller[selected_option_idx]
            )
        action_controller_state = np.tanh(action_controller_pre)
        policy_logits = np.clip(
            np.nan_to_num(
                (
                    self.W2_action_backbone @ action_backbone_state
                    + self.b2_action_backbone
                    if self.option_action_separate_backbone
                    else
                    (
                    self.W2_action_policy_path @ action_policy_state
                    + self.b2_action_policy_path
                    if self.option_action_separate_policy_path
                    else
                    (
                    self.W2_action_policy_core @ action_policy_state
                    + self.b2_action_policy_core
                    if self.option_action_recurrent_core
                    else self.W2_policy @ policy_core + self.b2_policy
                    )
                    )
                )
                + (
                    0.0
                    if self.option_action_separate_policy_path
                    or self.option_action_separate_backbone
                    else self.option_action_bias[selected_option_idx]
                )
                + (
                    self.W2_option_action_head[selected_option_idx] @ policy_core
                    + self.b2_option_action_head[selected_option_idx]
                    if self.option_action_head
                    else 0.0
                )
                + (
                    self.W2_option_sequence_head[
                        selected_option_idx,
                        selected_option_age_bucket,
                    ]
                    @ policy_core
                    + self.b2_option_sequence_head[
                        selected_option_idx,
                        selected_option_age_bucket,
                    ]
                    if self.option_sequence_head
                    else 0.0
                )
                + (
                    self.W2_option_action_controller_head[selected_option_idx]
                    @ action_controller_state
                    + self.b2_option_action_controller_head[selected_option_idx]
                    if self.option_action_controller_state
                    else 0.0
                )
                + self.W2_policy_feedback @ combined_feedback
                + self.b2_policy_feedback,
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ),
            -20.0,
            20.0,
        )
        policy_logits = self._apply_executive_affordance_action_gating(
            x,
            selected_option_idx,
            policy_logits,
            blocked_probs,
            geometry_probs,
            shelter_position_probs,
        )
        value = float(
            np.clip(
                np.nan_to_num(
                    self.W2_value @ h_new + self.b2_value,
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6,
                )[0],
                -1e6,
                1e6,
            )
        )
        self.hidden_state = h_new.copy()
        self.current_option_vector = option_vector.copy()
        self.decoder_action_state = decoder_hidden.copy()
        self.action_backbone_state = action_backbone_state.copy()
        self.action_policy_state = action_policy_state.copy()
        self.action_controller_state = action_controller_state.copy()
        self.action_token_state = action_token_state.copy()
        self.last_option_summary = {
                "selected_option": OPTION_NAMES[selected_option_idx],
                "option_age": int(self.current_option_age),
                "option_termination_reason": (
                    "initial_selection"
                if termination_reason is None and previous_option_vector.sum() <= 0.0
                else ("active" if termination_reason is None else termination_reason)
            ),
            "option_logits": selection_option_logits.round(6).tolist(),
            "option_cooldowns": self.option_cooldowns.astype(int).tolist(),
            "executive_release_steps_remaining": int(
                self.executive_release_steps_remaining
            ),
        }
        self.last_affordance_summary = {
            "blocked_logits": blocked_logits.round(6).tolist(),
            "role_logits": role_logits.round(6).tolist(),
            "geometry_logits": geometry_logits.round(6).tolist(),
            "shelter_position_logits": shelter_position_logits.round(6).tolist(),
            "transition_prediction_logits": transition_prediction_logits.round(6).tolist(),
            "transition_rollout_prediction_logits": transition_rollout_prediction_logits.round(6).tolist(),
        }
        if store_cache:
            self.cache = OptionAffordancePositionFeedbackCache(
                x=x,
                x_aug=x_aug,
                h_prev=h_prev,
                h_new=h_new,
                query_input=query_input,
                query=query,
                slot_raws=slot_raws,
                keys=keys,
                values=values,
                attention_weights=attention_weights,
                valid_event_type_indices=event_type_indices,
                option_probs=option_probs,
                selected_option_idx=selected_option_idx,
                blocked_probs=blocked_probs,
                role_probs=role_probs,
                affordance_features=affordance_features,
                affordance_feedback=affordance_feedback,
                geometry_probs=geometry_probs,
                shelter_position_probs=shelter_position_probs,
                transition_prediction_values=transition_prediction_logits,
                transition_prediction_feedback=transition_prediction_feedback,
                transition_rollout_prediction_values=transition_rollout_prediction_logits,
                transition_rollout_prediction_feedback=transition_rollout_prediction_feedback,
                combined_feedback=combined_feedback,
                phase_probs=phase_probs,
                previous_option_vector=previous_option_vector,
                previous_option_idx=previous_option_idx,
                previous_action_vector=previous_action_vector,
                selected_option_age_bucket=selected_option_age_bucket,
                previous_decoder_action_state=previous_decoder_action_state,
                previous_action_backbone_state=previous_action_backbone_state,
                action_backbone_state=action_backbone_state,
                action_backbone_pre=action_backbone_pre,
                previous_action_policy_state=previous_action_policy_state,
                action_policy_state=action_policy_state,
                action_policy_pre=action_policy_pre,
                previous_action_controller_state=previous_action_controller_state,
                action_controller_state=action_controller_state,
                action_controller_pre=action_controller_pre,
                previous_action_token_state=previous_action_token_state,
                action_token_state=action_token_state,
                action_token_pre=action_token_pre,
                decoder_hidden=decoder_hidden,
                decoder_hidden_pre=decoder_hidden_pre,
            )
        if phase_logits is not None:
            return policy_logits, value, option_logits, phase_logits
        return policy_logits, value, option_logits
