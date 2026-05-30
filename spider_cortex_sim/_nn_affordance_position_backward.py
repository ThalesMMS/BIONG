from __future__ import annotations

from ._nn_affordance_geometry import *


class _RecurrentOptionAffordancePositionBackwardMixin:
    def backward(
        self,
        grad_policy_logits: Array,
        grad_value: float,
        lr: float,
        grad_clip: float = 5.0,
        grad_option_logits: Array | None = None,
        grad_phase_logits: Array | None = None,
        grad_affordance_blocked_logits: Array | None = None,
        grad_affordance_role_logits: Array | None = None,
        grad_geometry_logits: Array | None = None,
        grad_shelter_position_logits: Array | None = None,
        grad_transition_prediction_logits: Array | None = None,
        grad_transition_rollout_prediction_logits: Array | None = None,
    ) -> Array:
        if not isinstance(self.cache, OptionAffordancePositionFeedbackCache):
            raise RuntimeError(
                "Recurrent option affordance position feedback network backward called without position cache."
            )
        grad_policy_logits = _clip_grad_logits(grad_policy_logits, grad_clip)
        grad_value = float(np.clip(grad_value, -grad_clip, grad_clip))
        if grad_option_logits is None:
            grad_option_logits = np.zeros(self.option_dim, dtype=float)
        else:
            grad_option_logits = _clip_grad_logits(
                np.asarray(grad_option_logits, dtype=float),
                grad_clip,
            )
        if grad_phase_logits is None or self.phase_output_dim <= 0:
            grad_phase_logits = np.zeros(self.phase_output_dim, dtype=float)
        else:
            grad_phase_logits = _clip_grad_logits(
                np.asarray(grad_phase_logits, dtype=float),
                grad_clip,
            )
        if grad_affordance_blocked_logits is None:
            grad_affordance_blocked_logits = np.zeros(self.output_dim, dtype=float)
        else:
            grad_affordance_blocked_logits = _clip_grad_logits(
                grad_affordance_blocked_logits,
                grad_clip,
            )
        affordance_role_output_dim = self.output_dim * self.affordance_role_dim
        if grad_affordance_role_logits is None:
            grad_affordance_role_logits = np.zeros(
                affordance_role_output_dim,
                dtype=float,
            )
        else:
            grad_affordance_role_logits = _clip_grad_logits(
                np.asarray(grad_affordance_role_logits, dtype=float),
                grad_clip,
            )
        if grad_geometry_logits is None:
            grad_geometry_logits = np.zeros(self.geometry_feature_dim, dtype=float)
        else:
            grad_geometry_logits = _clip_grad_logits(
                np.asarray(grad_geometry_logits, dtype=float),
                grad_clip,
            )
        shelter_position_output_dim = self.output_dim * self.shelter_position_dim
        if grad_shelter_position_logits is None:
            grad_shelter_position_logits = np.zeros(
                shelter_position_output_dim,
                dtype=float,
            )
        else:
            grad_shelter_position_logits = _clip_grad_logits(
                np.asarray(grad_shelter_position_logits, dtype=float),
                grad_clip,
            )
        if grad_transition_prediction_logits is None or not self.transition_prediction_head:
            grad_transition_prediction_logits = np.zeros(0, dtype=float)
        else:
            grad_transition_prediction_logits = _clip_grad_logits(
                np.asarray(grad_transition_prediction_logits, dtype=float),
                grad_clip,
            )
        if (
            grad_transition_rollout_prediction_logits is None
            or not self.transition_rollout_prediction_head
        ):
            grad_transition_rollout_prediction_logits = np.zeros(0, dtype=float)
        else:
            grad_transition_rollout_prediction_logits = _clip_grad_logits(
                np.asarray(
                    grad_transition_rollout_prediction_logits,
                    dtype=float,
                ),
                grad_clip,
            )
        x_aug = self.cache.x_aug
        h_prev = self.cache.h_prev
        h_new = self.cache.h_new
        decoder_hidden = self.cache.decoder_hidden
        decoder_hidden_pre = self.cache.decoder_hidden_pre
        policy_core = (
            self.cache.action_token_state
            if self.option_action_token_decoder
            else (
                self.cache.action_backbone_state
                if self.option_action_separate_backbone
                else (
                self.cache.action_policy_state
                if self.option_action_recurrent_core
                else decoder_hidden
                )
            )
        )
        affordance_feedback = self.cache.affordance_feedback
        transition_prediction_feedback = self.cache.transition_prediction_feedback
        transition_rollout_prediction_feedback = (
            self.cache.transition_rollout_prediction_feedback
        )
        combined_feedback = self.cache.combined_feedback
        grad_W2_policy = (
            np.outer(grad_policy_logits, policy_core)
            if not self.option_action_recurrent_core
            and not self.option_action_separate_policy_path
            and not self.option_action_separate_backbone
            else np.zeros_like(self.W2_policy)
        )
        grad_b2_policy = (
            grad_policy_logits
            if not self.option_action_recurrent_core
            and not self.option_action_separate_policy_path
            and not self.option_action_separate_backbone
            else np.zeros_like(self.b2_policy)
        )
        grad_W2_action_backbone = (
            np.outer(grad_policy_logits, self.cache.action_backbone_state)
            if self.option_action_separate_backbone
            else np.zeros_like(self.W2_action_backbone)
        )
        grad_b2_action_backbone = (
            grad_policy_logits.copy()
            if self.option_action_separate_backbone
            else np.zeros_like(self.b2_action_backbone)
        )
        grad_W2_action_policy_core = (
            np.outer(grad_policy_logits, self.cache.action_policy_state)
            if self.option_action_recurrent_core
            else np.zeros_like(self.W2_action_policy_core)
        )
        grad_b2_action_policy_core = (
            grad_policy_logits.copy()
            if self.option_action_recurrent_core
            else np.zeros_like(self.b2_action_policy_core)
        )
        grad_W2_action_policy_path = (
            np.outer(grad_policy_logits, self.cache.action_policy_state)
            if self.option_action_separate_policy_path
            else np.zeros_like(self.W2_action_policy_path)
        )
        grad_b2_action_policy_path = (
            grad_policy_logits.copy()
            if self.option_action_separate_policy_path
            else np.zeros_like(self.b2_action_policy_path)
        )
        grad_W2_value = grad_value * h_new.reshape(1, -1)
        grad_b2_value = np.array([grad_value], dtype=float)
        grad_option_action_bias = np.zeros_like(self.option_action_bias)
        if (
            not self.option_action_separate_policy_path
            and not self.option_action_separate_backbone
        ):
            grad_option_action_bias[self.cache.selected_option_idx] += grad_policy_logits
        grad_W2_option_action_head = (
            np.zeros_like(self.W2_option_action_head)
            if self.option_action_head
            else np.zeros((0, self.output_dim, self.hidden_dim), dtype=float)
        )
        grad_b2_option_action_head = (
            np.zeros_like(self.b2_option_action_head)
            if self.option_action_head
            else np.zeros((0, self.output_dim), dtype=float)
        )
        grad_W2_option_sequence_head = (
            np.zeros_like(self.W2_option_sequence_head)
            if self.option_sequence_head
            else np.zeros((0, 0, self.output_dim, self.hidden_dim), dtype=float)
        )
        grad_b2_option_sequence_head = (
            np.zeros_like(self.b2_option_sequence_head)
            if self.option_sequence_head
            else np.zeros((0, 0, self.output_dim), dtype=float)
        )
        if self.option_action_head:
            grad_W2_option_action_head[self.cache.selected_option_idx] = np.outer(
                grad_policy_logits,
                policy_core,
            )
            grad_b2_option_action_head[self.cache.selected_option_idx] = (
                grad_policy_logits
            )
        if self.option_sequence_head:
            grad_W2_option_sequence_head[
                self.cache.selected_option_idx,
                self.cache.selected_option_age_bucket,
            ] = np.outer(
                grad_policy_logits,
                policy_core,
            )
            grad_b2_option_sequence_head[
                self.cache.selected_option_idx,
                self.cache.selected_option_age_bucket,
            ] = grad_policy_logits
        option_target = one_hot(self.cache.selected_option_idx, self.option_dim)
        option_advantage = -grad_value
        grad_option_logits = grad_option_logits + 0.2 * option_advantage * (
            self.cache.option_probs - option_target
        )
        grad_W2_option = np.outer(grad_option_logits, h_new)
        grad_b2_option = grad_option_logits
        grad_W2_phase = (
            np.outer(grad_phase_logits, h_new)
            if self.phase_output_dim > 0
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_b2_phase = np.asarray(grad_phase_logits, dtype=float)
        grad_W2_policy_feedback = np.outer(grad_policy_logits, combined_feedback)
        grad_b2_policy_feedback = grad_policy_logits
        grad_W2_option_feedback = np.outer(grad_option_logits, combined_feedback)
        grad_b2_option_feedback = grad_option_logits
        grad_W2_phase_option_feedback = (
            np.outer(grad_option_logits, self.cache.phase_probs)
            if self.phase_option_feedback
            else np.zeros_like(self.W2_phase_option_feedback)
        )
        grad_b2_phase_option_feedback = (
            grad_option_logits.copy()
            if self.phase_option_feedback
            else np.zeros_like(self.b2_phase_option_feedback)
        )
        grad_W2_option_transition_feedback = (
            np.outer(grad_option_logits, self.cache.previous_option_vector)
            if self.option_transition_feedback
            else np.zeros_like(self.W2_option_transition_feedback)
        )
        grad_b2_option_transition_feedback = (
            grad_option_logits.copy()
            if self.option_transition_feedback
            else np.zeros_like(self.b2_option_transition_feedback)
        )
        grad_feedback = (
            self.W2_policy_feedback.T @ grad_policy_logits
            + self.W2_option_feedback.T @ grad_option_logits
        )
        dz_feedback = grad_feedback * (1.0 - affordance_feedback**2)
        grad_W_affordance_feedback = np.outer(
            dz_feedback,
            self.cache.affordance_features,
        )
        grad_b_affordance_feedback = dz_feedback
        grad_affordance_features = self.W_affordance_feedback.T @ dz_feedback
        grad_W_transition_prediction_feedback = (
            np.zeros_like(self.W_transition_prediction_feedback)
            if self.transition_prediction_feedback
            else np.zeros((0, self.transition_prediction_feature_dim), dtype=float)
        )
        grad_b_transition_prediction_feedback = (
            np.zeros_like(self.b_transition_prediction_feedback)
            if self.transition_prediction_feedback
            else np.zeros(0, dtype=float)
        )
        grad_W_transition_rollout_prediction_feedback = (
            np.zeros_like(self.W_transition_rollout_prediction_feedback)
            if self.transition_rollout_prediction_feedback
            else np.zeros(
                (0, self.transition_rollout_prediction_feature_dim),
                dtype=float,
            )
        )
        grad_b_transition_rollout_prediction_feedback = (
            np.zeros_like(self.b_transition_rollout_prediction_feedback)
            if self.transition_rollout_prediction_feedback
            else np.zeros(0, dtype=float)
        )
        grad_transition_prediction_from_feedback = np.zeros(0, dtype=float)
        grad_transition_rollout_prediction_from_feedback = np.zeros(0, dtype=float)
        if self.transition_prediction_feedback:
            grad_transition_prediction_from_feedback = np.zeros(
                self.transition_prediction_feature_dim,
                dtype=float,
            )
            dz_transition_feedback = grad_feedback * (
                1.0 - transition_prediction_feedback**2
            )
            grad_W_transition_prediction_feedback = np.outer(
                dz_transition_feedback,
                self.cache.transition_prediction_values,
            )
            grad_b_transition_prediction_feedback = dz_transition_feedback
            grad_transition_prediction_from_feedback = (
                self.W_transition_prediction_feedback.T @ dz_transition_feedback
            )
        if self.transition_rollout_prediction_feedback:
            grad_transition_rollout_prediction_from_feedback = np.zeros(
                self.transition_rollout_prediction_feature_dim,
                dtype=float,
            )
            dz_transition_rollout_feedback = grad_feedback * (
                1.0 - transition_rollout_prediction_feedback**2
            )
            grad_W_transition_rollout_prediction_feedback = np.outer(
                dz_transition_rollout_feedback,
                self.cache.transition_rollout_prediction_values,
            )
            grad_b_transition_rollout_prediction_feedback = (
                dz_transition_rollout_feedback
            )
            grad_transition_rollout_prediction_from_feedback = (
                self.W_transition_rollout_prediction_feedback.T
                @ dz_transition_rollout_feedback
            )
        grad_blocked_probs = grad_affordance_features[: self.output_dim]
        role_prob_end = self.output_dim + affordance_role_output_dim
        geometry_prob_end = role_prob_end + self.geometry_feature_dim
        grad_role_probs = grad_affordance_features[
            self.output_dim : role_prob_end
        ].reshape(self.output_dim, self.affordance_role_dim)
        grad_geometry_probs = grad_affordance_features[role_prob_end:geometry_prob_end]
        grad_shelter_position_probs = grad_affordance_features[
            geometry_prob_end:
        ].reshape(self.output_dim, self.shelter_position_dim)
        grad_phase_probs = (
            self.W2_phase_option_feedback.T @ grad_option_logits
            if self.phase_option_feedback
            else np.zeros(self.phase_output_dim, dtype=float)
        )
        if self.phase_output_dim > 0 and grad_phase_probs.size > 0:
            phase_probs = self.cache.phase_probs
            grad_phase_logits = grad_phase_logits + phase_probs * (
                grad_phase_probs - float(np.dot(grad_phase_probs, phase_probs))
            )
            grad_W2_phase = np.outer(grad_phase_logits, h_new)
            grad_b2_phase = np.asarray(grad_phase_logits, dtype=float)
        grad_affordance_blocked_logits = np.asarray(
            grad_affordance_blocked_logits,
            dtype=float,
        ) + grad_blocked_probs * (
            self.cache.blocked_probs * (1.0 - self.cache.blocked_probs)
        )
        grad_affordance_role_matrix = np.asarray(
            grad_affordance_role_logits,
            dtype=float,
        ).reshape(self.output_dim, self.affordance_role_dim)
        feedback_role_grad_matrix = np.zeros_like(grad_affordance_role_matrix)
        for action_idx in range(self.output_dim):
            role_probs = self.cache.role_probs[action_idx]
            role_prob_grad = grad_role_probs[action_idx]
            feedback_role_grad_matrix[action_idx] = role_probs * (
                role_prob_grad - float(np.dot(role_prob_grad, role_probs))
            )
        grad_affordance_role_logits = (
            grad_affordance_role_matrix + feedback_role_grad_matrix
        ).reshape(-1)
        grad_geometry_logits = np.asarray(grad_geometry_logits, dtype=float) + (
            grad_geometry_probs * self.cache.geometry_probs * (1.0 - self.cache.geometry_probs)
        )
        grad_shelter_position_matrix = np.asarray(
            grad_shelter_position_logits,
            dtype=float,
        ).reshape(self.output_dim, self.shelter_position_dim)
        feedback_position_grad_matrix = np.zeros_like(grad_shelter_position_matrix)
        for action_idx in range(self.output_dim):
            position_probs = self.cache.shelter_position_probs[action_idx]
            position_prob_grad = grad_shelter_position_probs[action_idx]
            feedback_position_grad_matrix[action_idx] = position_probs * (
                position_prob_grad
                - float(np.dot(position_prob_grad, position_probs))
            )
        grad_shelter_position_logits = (
            grad_shelter_position_matrix + feedback_position_grad_matrix
        ).reshape(-1)
        grad_W2_affordance_blocked = np.outer(
            grad_affordance_blocked_logits,
            h_new,
        )
        grad_b2_affordance_blocked = grad_affordance_blocked_logits
        grad_W2_affordance_role = np.outer(
            grad_affordance_role_logits,
            h_new,
        )
        grad_b2_affordance_role = grad_affordance_role_logits
        grad_W2_geometry = np.outer(grad_geometry_logits, h_new)
        grad_b2_geometry = grad_geometry_logits
        grad_W2_shelter_position = np.outer(grad_shelter_position_logits, h_new)
        grad_b2_shelter_position = grad_shelter_position_logits
        if self.transition_prediction_head:
            grad_transition_prediction_pre = (
                np.asarray(grad_transition_prediction_logits, dtype=float)
                + grad_transition_prediction_from_feedback
            ) * (1.0 - self.cache.transition_prediction_values**2)
        else:
            grad_transition_prediction_pre = np.zeros(0, dtype=float)
        if self.transition_rollout_prediction_head:
            grad_transition_rollout_prediction_pre = (
                np.asarray(
                    grad_transition_rollout_prediction_logits,
                    dtype=float,
                )
                + grad_transition_rollout_prediction_from_feedback
            ) * (1.0 - self.cache.transition_rollout_prediction_values**2)
        else:
            grad_transition_rollout_prediction_pre = np.zeros(0, dtype=float)
        grad_W2_transition_prediction = (
            np.outer(grad_transition_prediction_pre, h_new)
            if self.transition_prediction_head
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_b2_transition_prediction = (
            grad_transition_prediction_pre
            if self.transition_prediction_head
            else np.zeros(0, dtype=float)
        )
        grad_W2_transition_rollout_prediction = (
            np.outer(grad_transition_rollout_prediction_pre, h_new)
            if self.transition_rollout_prediction_head
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_b2_transition_rollout_prediction = (
            grad_transition_rollout_prediction_pre
            if self.transition_rollout_prediction_head
            else np.zeros(0, dtype=float)
        )
        grad_policy_core = (
            self.W2_action_backbone.T @ grad_policy_logits
            if self.option_action_separate_backbone
            else
            (
            self.W2_action_policy_path.T @ grad_policy_logits
            if self.option_action_separate_policy_path
            else
            (
            self.W2_action_policy_core.T @ grad_policy_logits
            if self.option_action_recurrent_core
            else self.W2_policy.T @ grad_policy_logits
            )
            )
        )
        if self.option_action_head:
            grad_policy_core = (
                grad_policy_core
                + self.W2_option_action_head[self.cache.selected_option_idx].T
                @ grad_policy_logits
            )
        if self.option_sequence_head:
            grad_policy_core = (
                grad_policy_core
                + self.W2_option_sequence_head[
                    self.cache.selected_option_idx,
                    self.cache.selected_option_age_bucket,
                ].T
                @ grad_policy_logits
            )
        grad_W_option_action_policy_decoder = (
            np.zeros_like(self.W_option_action_policy_decoder)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_policy_prev = (
            np.zeros_like(self.W_option_action_policy_prev)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_policy_action = (
            np.zeros_like(self.W_option_action_policy_action)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        grad_b_option_action_policy = (
            np.zeros_like(self.b_option_action_policy)
            if self.option_action_recurrent_core
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_W_action_backbone_input = (
            np.zeros_like(self.W_action_backbone_input)
            if self.option_action_separate_backbone
            else np.zeros((0, self.input_dim), dtype=float)
        )
        grad_W_action_backbone_prev = (
            np.zeros_like(self.W_action_backbone_prev)
            if self.option_action_separate_backbone
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_W_action_backbone_action = (
            np.zeros_like(self.W_action_backbone_action)
            if self.option_action_separate_backbone
            else np.zeros((0, self.output_dim), dtype=float)
        )
        grad_b_action_backbone = (
            np.zeros_like(self.b_action_backbone)
            if self.option_action_separate_backbone
            else np.zeros(0, dtype=float)
        )
        grad_x_from_action_backbone = np.zeros(self.input_dim, dtype=float)
        grad_W_action_policy_path_input = (
            np.zeros_like(self.W_action_policy_path_input)
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_W_action_policy_path_prev = (
            np.zeros_like(self.W_action_policy_path_prev)
            if self.option_action_separate_policy_path
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        grad_W_action_policy_path_action = (
            np.zeros_like(self.W_action_policy_path_action)
            if self.option_action_separate_policy_path
            else np.zeros((0, self.output_dim), dtype=float)
        )
        grad_b_action_policy_path = (
            np.zeros_like(self.b_action_policy_path)
            if self.option_action_separate_policy_path
            else np.zeros(0, dtype=float)
        )
        grad_h_from_action_policy = np.zeros(self.hidden_dim, dtype=float)
        grad_W_option_action_token_decoder = (
            np.zeros_like(self.W_option_action_token_decoder)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_token_prev = (
            np.zeros_like(self.W_option_action_token_prev)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_token_action = (
            np.zeros_like(self.W_option_action_token_action)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        grad_b_option_action_token = (
            np.zeros_like(self.b_option_action_token)
            if self.option_action_token_decoder
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_action_token_decoder:
            grad_action_token_pre = grad_policy_core * (
                1.0 - self.cache.action_token_state**2
            )
            grad_W_option_action_token_decoder[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_token_pre,
                decoder_hidden,
            )
            grad_W_option_action_token_prev[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_token_pre,
                self.cache.previous_action_token_state,
            )
            grad_W_option_action_token_action[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_token_pre,
                self.cache.previous_action_vector,
            )
            grad_b_option_action_token[
                self.cache.selected_option_idx
            ] = grad_action_token_pre
            grad_decoder_hidden = (
                self.W_option_action_token_decoder[
                    self.cache.selected_option_idx
                ].T
                @ grad_action_token_pre
            )
        elif self.option_action_separate_backbone:
            grad_action_backbone_pre = grad_policy_core * (
                1.0 - self.cache.action_backbone_state**2
            )
            grad_W_action_backbone_input = np.outer(
                grad_action_backbone_pre,
                self.cache.x,
            )
            grad_W_action_backbone_prev = np.outer(
                grad_action_backbone_pre,
                self.cache.previous_action_backbone_state,
            )
            grad_W_action_backbone_action = np.outer(
                grad_action_backbone_pre,
                self.cache.previous_action_vector,
            )
            grad_b_action_backbone = grad_action_backbone_pre
            grad_x_from_action_backbone = (
                self.W_action_backbone_input.T @ grad_action_backbone_pre
            )
            grad_decoder_hidden = np.zeros(self.hidden_dim, dtype=float)
        elif self.option_action_separate_policy_path:
            grad_action_policy_pre = grad_policy_core * (
                1.0 - self.cache.action_policy_state**2
            )
            grad_W_action_policy_path_input = np.outer(
                grad_action_policy_pre,
                h_new,
            )
            grad_W_action_policy_path_prev = np.outer(
                grad_action_policy_pre,
                self.cache.previous_action_policy_state,
            )
            grad_W_action_policy_path_action = np.outer(
                grad_action_policy_pre,
                self.cache.previous_action_vector,
            )
            grad_b_action_policy_path = grad_action_policy_pre
            grad_h_from_action_policy = (
                self.W_action_policy_path_input.T @ grad_action_policy_pre
            )
            grad_decoder_hidden = np.zeros(self.hidden_dim, dtype=float)
        elif self.option_action_recurrent_core:
            grad_action_policy_pre = grad_policy_core * (
                1.0 - self.cache.action_policy_state**2
            )
            action_policy_source = (
                h_new if self.option_action_separate_recurrent_head else decoder_hidden
            )
            grad_W_option_action_policy_decoder[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_policy_pre,
                action_policy_source,
            )
            grad_W_option_action_policy_prev[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_policy_pre,
                self.cache.previous_action_policy_state,
            )
            grad_W_option_action_policy_action[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_policy_pre,
                self.cache.previous_action_vector,
            )
            grad_b_option_action_policy[
                self.cache.selected_option_idx
            ] = grad_action_policy_pre
            grad_action_policy_source = (
                self.W_option_action_policy_decoder[
                    self.cache.selected_option_idx
                ].T
                @ grad_action_policy_pre
            )
            if self.option_action_separate_recurrent_head:
                grad_h_from_action_policy = grad_action_policy_source
                grad_decoder_hidden = np.zeros(self.hidden_dim, dtype=float)
            else:
                grad_decoder_hidden = grad_action_policy_source
        else:
            grad_decoder_hidden = grad_policy_core
        grad_action_controller_state = (
            self.W2_option_action_controller_head[self.cache.selected_option_idx].T
            @ grad_policy_logits
            if self.option_action_controller_state
            else np.zeros(self.hidden_dim, dtype=float)
        )
        grad_decoder_hidden_pre = grad_decoder_hidden * (1.0 - decoder_hidden**2)
        grad_W2_option_action_controller_head = (
            np.zeros_like(self.W2_option_action_controller_head)
            if self.option_action_controller_state
            else np.zeros((0, self.output_dim, self.hidden_dim), dtype=float)
        )
        grad_b2_option_action_controller_head = (
            np.zeros_like(self.b2_option_action_controller_head)
            if self.option_action_controller_state
            else np.zeros((0, self.output_dim), dtype=float)
        )
        grad_action_controller_pre = np.zeros(self.hidden_dim, dtype=float)
        grad_W_option_action_controller_decoder = (
            np.zeros_like(self.W_option_action_controller_decoder)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_controller_prev = (
            np.zeros_like(self.W_option_action_controller_prev)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_W_option_action_controller_action = (
            np.zeros_like(self.W_option_action_controller_action)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        grad_b_option_action_controller = (
            np.zeros_like(self.b_option_action_controller)
            if self.option_action_controller_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_action_controller_state:
            grad_W2_option_action_controller_head[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_policy_logits,
                self.cache.action_controller_state,
            )
            grad_b2_option_action_controller_head[
                self.cache.selected_option_idx
            ] = grad_policy_logits
            grad_action_controller_pre = (
                grad_action_controller_state
                * (1.0 - self.cache.action_controller_state**2)
            )
            grad_W_option_action_controller_decoder[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_controller_pre,
                self.cache.decoder_hidden,
            )
            grad_W_option_action_controller_prev[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_controller_pre,
                self.cache.previous_action_controller_state,
            )
            grad_W_option_action_controller_action[
                self.cache.selected_option_idx
            ] = np.outer(
                grad_action_controller_pre,
                self.cache.previous_action_vector,
            )
            grad_b_option_action_controller[
                self.cache.selected_option_idx
            ] = grad_action_controller_pre
            grad_decoder_hidden_pre = (
                grad_decoder_hidden_pre
                + self.W_option_action_controller_decoder[
                    self.cache.selected_option_idx
                ].T
                @ grad_action_controller_pre
            )
        grad_W_option_decoder_recurrent_state = (
            np.zeros_like(self.W_option_decoder_recurrent_state)
            if self.option_decoder_recurrent_state
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_b_option_decoder_recurrent_state = (
            np.zeros_like(self.b_option_decoder_recurrent_state)
            if self.option_decoder_recurrent_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_decoder_recurrent_state:
            grad_W_option_decoder_recurrent_state[self.cache.selected_option_idx] = (
                np.outer(
                    grad_decoder_hidden_pre,
                    self.cache.previous_decoder_action_state,
                )
            )
            grad_b_option_decoder_recurrent_state[self.cache.selected_option_idx] = (
                grad_decoder_hidden_pre
            )
        grad_W_option_action_transition_state = (
            np.zeros_like(self.W_option_action_transition_state)
            if self.option_action_transition_state
            else np.zeros((0, self.hidden_dim, self.output_dim), dtype=float)
        )
        grad_b_option_action_transition_state = (
            np.zeros_like(self.b_option_action_transition_state)
            if self.option_action_transition_state
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_action_transition_state:
            grad_W_option_action_transition_state[self.cache.selected_option_idx] = (
                np.outer(
                    grad_decoder_hidden_pre,
                    self.cache.previous_action_vector,
                )
            )
            grad_b_option_action_transition_state[self.cache.selected_option_idx] = (
                grad_decoder_hidden_pre
            )
        if self.option_decoder_state:
            grad_W_option_decoder_state = np.zeros_like(self.W_option_decoder_state)
            grad_b_option_decoder_state = np.zeros_like(self.b_option_decoder_state)
            grad_W_option_decoder_state[self.cache.selected_option_idx] = np.outer(
                grad_decoder_hidden_pre,
                h_new,
            )
            grad_b_option_decoder_state[self.cache.selected_option_idx] = (
                grad_decoder_hidden_pre
            )
            dh = (
                grad_decoder_hidden_pre
                + self.W_option_decoder_state[self.cache.selected_option_idx].T
                @ grad_decoder_hidden_pre
            )
        else:
            grad_W_option_decoder_state = np.zeros(
                (0, self.hidden_dim, self.hidden_dim),
                dtype=float,
            )
            grad_b_option_decoder_state = np.zeros((0, self.hidden_dim), dtype=float)
            dh = grad_decoder_hidden
        dh = (
            dh
            + self.W2_value.T[:, 0] * grad_value
            + self.W2_option.T @ grad_option_logits
            + grad_h_from_action_policy
            + (
                self.W2_phase.T @ grad_phase_logits
                if self.phase_output_dim > 0
                else 0.0
            )
            + self.W2_affordance_blocked.T @ grad_affordance_blocked_logits
            + self.W2_affordance_role.T @ grad_affordance_role_logits
            + self.W2_geometry.T @ grad_geometry_logits
            + self.W2_shelter_position.T @ grad_shelter_position_logits
            + (
                self.W2_transition_prediction.T @ grad_transition_prediction_pre
                if self.transition_prediction_head
                else 0.0
            )
            + (
                self.W2_transition_rollout_prediction.T
                @ grad_transition_rollout_prediction_pre
                if self.transition_rollout_prediction_head
                else 0.0
            )
        )
        dz = dh * (1.0 - h_new**2)
        grad_inputs_aug = self.W_xh.T @ dz
        grad_x = grad_inputs_aug[: self.input_dim].copy()
        grad_event_context = grad_inputs_aug[
            self.input_dim : self.input_dim + self.event_context_dim
        ].copy()
        grad_W_xh = np.outer(dz, x_aug)
        grad_W_hh = np.outer(dz, h_prev)
        grad_b_h = dz
        grad_W_option_recurrent_dynamics = (
            np.zeros_like(self.W_option_recurrent_dynamics)
            if self.option_recurrent_dynamics
            else np.zeros((0, self.hidden_dim, self.hidden_dim), dtype=float)
        )
        grad_b_option_recurrent_dynamics = (
            np.zeros_like(self.b_option_recurrent_dynamics)
            if self.option_recurrent_dynamics
            else np.zeros((0, self.hidden_dim), dtype=float)
        )
        if self.option_recurrent_dynamics and self.cache.previous_option_idx >= 0:
            grad_W_option_recurrent_dynamics[self.cache.previous_option_idx] = np.outer(
                dz,
                h_prev,
            )
            grad_b_option_recurrent_dynamics[self.cache.previous_option_idx] = dz
            grad_b_h = grad_b_h
        grad_W_query = np.zeros_like(self.W_query)
        grad_b_query = np.zeros_like(self.b_query)
        grad_W_key = np.zeros_like(self.W_key)
        grad_b_key = np.zeros_like(self.b_key)
        grad_W_value = np.zeros_like(self.W_value)
        grad_b_value_attn = np.zeros_like(self.b_value)
        grad_event_type_embeddings = np.zeros_like(self.event_type_embeddings)
        if self.cache.valid_event_type_indices.size > 0:
            slot_raws = self.cache.slot_raws
            keys = self.cache.keys
            values = self.cache.values
            attention_weights = self.cache.attention_weights
            query = self.cache.query
            scale = float(np.sqrt(max(1, self.event_context_dim)))
            grad_values = attention_weights.reshape(-1, 1) * grad_event_context.reshape(
                1,
                -1,
            )
            grad_attention = values @ grad_event_context
            grad_scores = attention_weights * (
                grad_attention - float(np.dot(attention_weights, grad_attention))
            )
            grad_query = np.sum(
                (grad_scores.reshape(-1, 1) * keys) / scale,
                axis=0,
            )
            grad_keys = (grad_scores.reshape(-1, 1) * query.reshape(1, -1)) / scale
            for slot_index, event_type_index in enumerate(
                self.cache.valid_event_type_indices
            ):
                dz_value = grad_values[slot_index] * (1.0 - values[slot_index] ** 2)
                grad_W_value += np.outer(dz_value, slot_raws[slot_index])
                grad_b_value_attn += dz_value
                grad_slot = self.W_value.T @ dz_value
                dz_key = grad_keys[slot_index] * (1.0 - keys[slot_index] ** 2)
                grad_W_key += np.outer(dz_key, slot_raws[slot_index])
                grad_b_key += dz_key
                grad_slot += self.W_key.T @ dz_key
                grad_event_type_embeddings[event_type_index] += grad_slot[
                    : self.event_embedding_dim
                ]
            dz_query = grad_query * (1.0 - self.cache.query**2)
            grad_W_query += np.outer(dz_query, self.cache.query_input)
            grad_b_query += dz_query
        self.W2_policy -= lr * grad_W2_policy
        self.b2_policy -= lr * grad_b2_policy
        if self.option_action_recurrent_core:
            self.W2_action_policy_core -= lr * grad_W2_action_policy_core
            self.b2_action_policy_core -= lr * grad_b2_action_policy_core
            self.W_option_action_policy_decoder -= (
                lr * grad_W_option_action_policy_decoder
            )
            self.W_option_action_policy_prev -= lr * grad_W_option_action_policy_prev
            self.W_option_action_policy_action -= (
                lr * grad_W_option_action_policy_action
            )
            self.b_option_action_policy -= lr * grad_b_option_action_policy
        if self.option_action_separate_backbone:
            self.W2_action_backbone -= lr * grad_W2_action_backbone
            self.b2_action_backbone -= lr * grad_b2_action_backbone
            self.W_action_backbone_input -= lr * grad_W_action_backbone_input
            self.W_action_backbone_prev -= lr * grad_W_action_backbone_prev
            self.W_action_backbone_action -= lr * grad_W_action_backbone_action
            self.b_action_backbone -= lr * grad_b_action_backbone
        if self.option_action_separate_policy_path:
            self.W2_action_policy_path -= lr * grad_W2_action_policy_path
            self.b2_action_policy_path -= lr * grad_b2_action_policy_path
            self.W_action_policy_path_input -= lr * grad_W_action_policy_path_input
            self.W_action_policy_path_prev -= lr * grad_W_action_policy_path_prev
            self.W_action_policy_path_action -= lr * grad_W_action_policy_path_action
            self.b_action_policy_path -= lr * grad_b_action_policy_path
        self.W2_value -= lr * grad_W2_value
        self.b2_value -= lr * grad_b2_value
        self.W2_option -= lr * grad_W2_option
        self.b2_option -= lr * grad_b2_option
        if self.phase_output_dim > 0:
            self.W2_phase -= lr * grad_W2_phase
            self.b2_phase -= lr * grad_b2_phase
        self.option_action_bias -= lr * grad_option_action_bias
        self.W2_policy_feedback -= lr * grad_W2_policy_feedback
        self.b2_policy_feedback -= lr * grad_b2_policy_feedback
        self.W2_option_feedback -= lr * grad_W2_option_feedback
        self.b2_option_feedback -= lr * grad_b2_option_feedback
        if self.phase_option_feedback:
            self.W2_phase_option_feedback -= lr * grad_W2_phase_option_feedback
            self.b2_phase_option_feedback -= lr * grad_b2_phase_option_feedback
        if self.option_transition_feedback:
            self.W2_option_transition_feedback -= (
                lr * grad_W2_option_transition_feedback
            )
            self.b2_option_transition_feedback -= (
                lr * grad_b2_option_transition_feedback
            )
        if self.option_action_head:
            self.W2_option_action_head -= lr * grad_W2_option_action_head
            self.b2_option_action_head -= lr * grad_b2_option_action_head
        if self.option_sequence_head:
            self.W2_option_sequence_head -= lr * grad_W2_option_sequence_head
            self.b2_option_sequence_head -= lr * grad_b2_option_sequence_head
        if self.option_action_controller_state:
            self.W2_option_action_controller_head -= (
                lr * grad_W2_option_action_controller_head
            )
            self.b2_option_action_controller_head -= (
                lr * grad_b2_option_action_controller_head
            )
            self.W_option_action_controller_decoder -= (
                lr * grad_W_option_action_controller_decoder
            )
            self.W_option_action_controller_prev -= (
                lr * grad_W_option_action_controller_prev
            )
            self.W_option_action_controller_action -= (
                lr * grad_W_option_action_controller_action
            )
            self.b_option_action_controller -= lr * grad_b_option_action_controller
        if self.option_action_token_decoder:
            self.W_option_action_token_decoder -= lr * grad_W_option_action_token_decoder
            self.W_option_action_token_prev -= lr * grad_W_option_action_token_prev
            self.W_option_action_token_action -= lr * grad_W_option_action_token_action
            self.b_option_action_token -= lr * grad_b_option_action_token
        if self.option_decoder_recurrent_state:
            self.W_option_decoder_recurrent_state -= (
                lr * grad_W_option_decoder_recurrent_state
            )
            self.b_option_decoder_recurrent_state -= (
                lr * grad_b_option_decoder_recurrent_state
            )
        if self.option_action_transition_state:
            self.W_option_action_transition_state -= (
                lr * grad_W_option_action_transition_state
            )
            self.b_option_action_transition_state -= (
                lr * grad_b_option_action_transition_state
            )
        if self.option_decoder_state:
            self.W_option_decoder_state -= lr * grad_W_option_decoder_state
            self.b_option_decoder_state -= lr * grad_b_option_decoder_state
        if self.option_recurrent_dynamics:
            self.W_option_recurrent_dynamics -= lr * grad_W_option_recurrent_dynamics
            self.b_option_recurrent_dynamics -= lr * grad_b_option_recurrent_dynamics
        self.W_affordance_feedback -= lr * grad_W_affordance_feedback
        self.b_affordance_feedback -= lr * grad_b_affordance_feedback
        self.W2_affordance_blocked -= lr * grad_W2_affordance_blocked
        self.b2_affordance_blocked -= lr * grad_b2_affordance_blocked
        self.W2_affordance_role -= lr * grad_W2_affordance_role
        self.b2_affordance_role -= lr * grad_b2_affordance_role
        self.W2_geometry -= lr * grad_W2_geometry
        self.b2_geometry -= lr * grad_b2_geometry
        self.W2_shelter_position -= lr * grad_W2_shelter_position
        self.b2_shelter_position -= lr * grad_b2_shelter_position
        if self.transition_prediction_head:
            self.W2_transition_prediction -= lr * grad_W2_transition_prediction
            self.b2_transition_prediction -= lr * grad_b2_transition_prediction
        if self.transition_prediction_feedback:
            self.W_transition_prediction_feedback -= (
                lr * grad_W_transition_prediction_feedback
            )
            self.b_transition_prediction_feedback -= (
                lr * grad_b_transition_prediction_feedback
            )
        if self.transition_rollout_prediction_head:
            self.W2_transition_rollout_prediction -= (
                lr * grad_W2_transition_rollout_prediction
            )
            self.b2_transition_rollout_prediction -= (
                lr * grad_b2_transition_rollout_prediction
            )
        if self.transition_rollout_prediction_feedback:
            self.W_transition_rollout_prediction_feedback -= (
                lr * grad_W_transition_rollout_prediction_feedback
            )
            self.b_transition_rollout_prediction_feedback -= (
                lr * grad_b_transition_rollout_prediction_feedback
            )
        self.W_xh -= lr * grad_W_xh
        self.W_hh -= lr * grad_W_hh
        self.b_h -= lr * grad_b_h
        self.W_query -= lr * grad_W_query
        self.b_query -= lr * grad_b_query
        self.W_key -= lr * grad_W_key
        self.b_key -= lr * grad_b_key
        self.W_value -= lr * grad_W_value
        self.b_value -= lr * grad_b_value_attn
        self.event_type_embeddings -= lr * grad_event_type_embeddings
        grad_x = grad_x + grad_x_from_action_backbone
        return grad_x
