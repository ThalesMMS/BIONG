from __future__ import annotations

from .learning_shared import *


class _BrainLearningLearnMixin:
    def learn(self, decision: BrainStep, reward: float, next_observation: Dict[str, np.ndarray], done: bool) -> Dict[str, object]:
        """
        Perform a TD policy-gradient training step that updates value, policy (module proposals or monolithic), motor cortex, and optional arbitration parameters.
        
        This applies a clipped TD advantage to compute policy-logit and value gradients, selects and applies per-module crediting (including counterfactual or local-only modes), incorporates auxiliary/reflex targets when enabled, backpropagates through the action-center and either the module bank or monolithic policy, updates the motor cortex, and — when present and enabled — trains the arbitration network including valence and gate adjustments. Diagnostics include TD quantities, entropy, arbitration losses/grad norms, credit weights, and per-module gradient norms.
        
        Parameters:
            decision (BrainStep): Recorded decision containing selected action, policy, value estimate, module results, and optional arbitration decision; used as the learning target and for required cached tensors.
            reward (float): Observed scalar reward following the decision.
            next_observation (Dict[str, np.ndarray]): Observation after the action used to estimate the next state's value (ignored if done is True).
            done (bool): Whether the episode terminated after the action; if True the next-state value is treated as 0.0.
        
        Returns:
            Dict[str, object]: Training diagnostics including (at minimum):
                - "reward": provided reward.
                - "td_target": computed TD target (reward + gamma * next_value).
                - "td_error": clipped advantage used for policy gradients.
                - "value": value estimate from the decision.
                - "next_value": estimated next-state value (0.0 if done).
                - "entropy": policy entropy computed from decision.policy.
                - "aux_modules": number of modules that produced auxiliary gradients.
                - arbitration diagnostics (e.g. "arbitration_value", "arbitration_loss", grad norms).
                - credit diagnostics ("credit_strategy", "module_credit_weights", "counterfactual_credit_weights").
                - "module_gradient_norms": per-module gradient L2 norms (or monolithic entry).
        """
        if decision.policy_mode != "normal":
            raise ValueError(
                "learn() only supports decisions produced with policy_mode='normal'."
            )
        next_value = 0.0 if done else self.estimate_value(next_observation)
        td_target = reward + self.gamma * next_value
        advantage = float(np.clip(td_target - decision.value, -4.0, 4.0))
        if self.config.is_b_series:
            if self.b_series_policy is None:
                raise RuntimeError(
                    "B-series policy unavailable for the configured architecture."
                )
            if decision.semantic_policy.size != len(B_SEMANTIC_ACTIONS):
                raise ValueError(
                    "B-series learning requires a semantic policy over the six B0 actions."
                )
            semantic_action_idx = int(decision.semantic_action_idx)
            if not 0 <= semantic_action_idx < len(B_SEMANTIC_ACTIONS):
                raise ValueError(
                    f"Invalid B-series semantic_action_idx: {semantic_action_idx}."
                )
            grad_policy_logits = advantage * (
                decision.semantic_policy
                - one_hot(semantic_action_idx, len(B_SEMANTIC_ACTIONS))
            )
            value_grad = decision.value - td_target
            if B_SERIES_POLICY_NAME not in self._frozen_modules:
                self.b_series_policy.backward(
                    grad_policy_logits=grad_policy_logits,
                    grad_value=value_grad,
                    lr=self.module_lr,
                )
            entropy = -float(
                np.sum(
                    decision.semantic_policy
                    * np.log(decision.semantic_policy + 1e-8)
                )
            )
            return {
                "reward": float(reward),
                "td_target": float(td_target),
                "td_error": float(advantage),
                "value": float(decision.value),
                "next_value": float(next_value),
                "entropy": entropy,
                "credit_strategy": "b_series_semantic_policy",
                "module_gradient_norms": {
                    B_SERIES_POLICY_NAME: float(np.linalg.norm(grad_policy_logits))
                },
                "b_level": int(self.config.b_level),
                "b_mode": str(self.config.b_mode),
                "semantic_action": decision.semantic_action,
                "bridge_primitive_action": decision.bridge_primitive_action,
                "bridge_reason": decision.bridge_reason,
                "external_override_count": int(decision.external_override_count),
            }
        grad_policy_logits = advantage * (decision.policy - one_hot(decision.action_idx, self.action_dim))
        effective_credit_strategy = self.config.credit_strategy
        uses_counterfactual_credit = self.config.uses_counterfactual_credit
        uses_local_credit_only = self.config.uses_local_credit_only
        uses_route_mask_credit = self.config.uses_route_mask_credit
        if self.config.is_true_monolithic and (
            uses_counterfactual_credit or uses_local_credit_only or uses_route_mask_credit
        ):
            # A direct policy has no downstream proposal path, so all non-broadcast
            # credit modes collapse to the applied broadcast update before any
            # per-module credit logic runs.
            effective_credit_strategy = "broadcast"
            uses_counterfactual_credit = False
            uses_local_credit_only = False
            uses_route_mask_credit = False
        elif self.config.is_monolithic and (
            uses_counterfactual_credit or uses_local_credit_only or uses_route_mask_credit
        ):
            # A monolithic proposer has no per-module proposal path, so diagnostics
            # follow the broadcast-style update that is actually applied below.
            effective_credit_strategy = "broadcast"
            uses_counterfactual_credit = False
            uses_local_credit_only = False
            uses_route_mask_credit = False
        module_credit_weights = {
            result.name: (1.0 if result.active else 0.0)
            for result in decision.module_results
        }
        counterfactual_credit_weights: Dict[str, float] = {}
        route_mask_threshold = float(self.config.route_mask_threshold)
        route_mask_enabled = bool(uses_route_mask_credit and self.config.is_modular)
        route_active_modules: list[str] = []
        route_credit_weights: Dict[str, float] = {
            result.name: 0.0 for result in decision.module_results
        }
        if uses_counterfactual_credit:
            if not decision.observation:
                raise ValueError(
                    "counterfactual credit requires BrainStep.observation to be populated."
                )
            counterfactual_credit_weights = self._compute_counterfactual_credit(
                decision.module_results,
                decision.observation,
                decision.action_idx,
            )
            module_credit_weights = dict(counterfactual_credit_weights)
        elif uses_local_credit_only:
            module_credit_weights = {
                result.name: 0.0
                for result in decision.module_results
            }
        elif uses_route_mask_credit:
            route_active_modules, route_credit_weights = self._compute_route_mask_credit(
                decision.module_results,
                threshold=route_mask_threshold,
                dominant_module=(
                    ""
                    if decision.arbitration_decision is None
                    else str(decision.arbitration_decision.dominant_module)
                ),
            )
            active_name_set = set(route_active_modules)
            module_credit_weights = {
                result.name: (
                    1.0
                    if result.active and result.name in active_name_set
                    else 0.0
                )
                for result in decision.module_results
            }
        reflex_aux_grads = {
            name: np.asarray(grad, dtype=float).copy()
            for name, grad in self._auxiliary_module_gradients(decision.module_results).items()
        }
        if self.config.is_true_monolithic:
            if self.true_monolithic_policy is None:
                raise RuntimeError(
                    "True monolithic network unavailable for the configured architecture."
                )
            effective_credit_strategy = "broadcast"
            value_grad = decision.value - td_target
            module_credit_weights = {self.TRUE_MONOLITHIC_POLICY_NAME: 1.0}
            entropy = -float(np.sum(decision.policy * np.log(decision.policy + 1e-8)))
            handoff_teacher_loss = 0.0
            handoff_teacher_grad_norm = 0.0
            handoff_teacher_weight = 0.0
            handoff_teacher_grad_logits = np.zeros(self.action_dim, dtype=float)
            handoff_option_teacher_loss = 0.0
            handoff_option_teacher_grad_norm = 0.0
            handoff_option_teacher_weight = 0.0
            handoff_option_teacher_grad_logits = np.zeros_like(
                decision.option_logits,
                dtype=float,
            )
            continuation_weights = self._continuation_auxiliary_weights(decision)
            post_rest_sequence_replay_boost_active = bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_replay_boost",
                    False,
                )
                and self._post_rest_release_sequence_active(decision)
            )
            if post_rest_sequence_replay_boost_active:
                continuation_weights = dict(continuation_weights)
                continuation_weights["teacher_action"] = max(
                    float(continuation_weights["teacher_action"]),
                    6.0,
                )
                continuation_weights["teacher_option"] = max(
                    float(continuation_weights["teacher_option"]),
                    4.0,
                )
                continuation_weights["phase"] = max(
                    float(continuation_weights["phase"]),
                    0.8,
                )
            post_rest_sequence_distill_active = bool(
                getattr(
                    self.config,
                    "direct_policy_post_rest_release_sequence_distill",
                    False,
                )
                and self._post_rest_release_sequence_active(decision)
            )
            post_rest_sequence_distill_loss = 0.0
            post_rest_sequence_distill_action_loss = 0.0
            post_rest_sequence_distill_option_loss = 0.0
            post_rest_sequence_distill_phase_loss = 0.0
            post_rest_sequence_distill_action_weight = 2.0
            post_rest_sequence_distill_option_weight = 1.5
            post_rest_sequence_distill_phase_weight = 1.0
            sequence_distill_targets = (
                self._post_rest_release_sequence_distillation_targets(decision)
                if post_rest_sequence_distill_active
                else {}
            )
            continuation_margin_weight = float(
                getattr(self.config, "direct_policy_continuation_margin_weight", 0.0)
            )
            continuation_margin_loss = 0.0
            continuation_margin_grad_norm = 0.0
            stay_action_idx = int(ACTION_TO_INDEX["STAY"])
            return_option_idx = int(OPTION_NAMES.index("RETURN_TO_SHELTER"))
            initial_forage_phase_idx = int(PHASE_LABELS.index("INITIAL_FORAGE"))
            if (
                self.config.direct_policy_handoff_teacher
                and 0 <= int(decision.teacher_action_target_idx) < self.action_dim
                and decision.total_logits.size == self.action_dim
            ):
                teacher_target = one_hot(
                    int(decision.teacher_action_target_idx),
                    self.action_dim,
                )
                handoff_teacher_weight = float(continuation_weights["teacher_action"])
                handoff_teacher_loss = handoff_teacher_weight * cross_entropy_loss(
                    decision.total_logits,
                    teacher_target,
                )
                handoff_teacher_grad_logits = handoff_teacher_weight * (
                    softmax(decision.total_logits) - teacher_target
                )
                if continuation_margin_weight > 0.0:
                    margin_loss, margin_grad = self._continuation_margin_loss_and_grad(
                        decision.total_logits,
                        target_idx=int(decision.teacher_action_target_idx),
                        competitor_idx=stay_action_idx,
                        margin=self.CONTINUATION_MARGIN,
                        scale=continuation_margin_weight * handoff_teacher_weight,
                    )
                    continuation_margin_loss += float(margin_loss)
                    handoff_teacher_grad_logits += margin_grad
            if (
                post_rest_sequence_distill_active
                and decision.total_logits.size == self.action_dim
            ):
                action_distill_target = np.asarray(
                    sequence_distill_targets.get("action", ()),
                    dtype=float,
                )
                if action_distill_target.size == self.action_dim:
                    post_rest_sequence_distill_action_loss = (
                        post_rest_sequence_distill_action_weight
                        * cross_entropy_loss(
                            decision.total_logits,
                            action_distill_target,
                        )
                    )
                    post_rest_sequence_distill_loss += (
                        post_rest_sequence_distill_action_loss
                    )
                    handoff_teacher_grad_logits += (
                        post_rest_sequence_distill_action_weight
                        * (softmax(decision.total_logits) - action_distill_target)
                    )
                handoff_teacher_grad_norm = float(
                    np.linalg.norm(handoff_teacher_grad_logits)
                )
            if (
                self.config.direct_policy_handoff_option_teacher
                and 0 <= int(decision.teacher_option_target_idx) < len(OPTION_NAMES)
                and decision.option_logits.size == len(OPTION_NAMES)
            ):
                option_teacher_target = one_hot(
                    int(decision.teacher_option_target_idx),
                    len(OPTION_NAMES),
                )
                handoff_option_teacher_weight = float(
                    continuation_weights["teacher_option"]
                )
                handoff_option_teacher_loss = (
                    handoff_option_teacher_weight
                    * cross_entropy_loss(
                        decision.option_logits,
                        option_teacher_target,
                    )
                )
                handoff_option_teacher_grad_logits = (
                    handoff_option_teacher_weight
                    * (softmax(decision.option_logits) - option_teacher_target)
                )
                if continuation_margin_weight > 0.0:
                    margin_loss, margin_grad = self._continuation_margin_loss_and_grad(
                        decision.option_logits,
                        target_idx=int(decision.teacher_option_target_idx),
                        competitor_idx=return_option_idx,
                        margin=self.CONTINUATION_MARGIN,
                        scale=continuation_margin_weight
                        * handoff_option_teacher_weight,
                    )
                    continuation_margin_loss += float(margin_loss)
                    handoff_option_teacher_grad_logits += margin_grad
            if (
                post_rest_sequence_distill_active
                and decision.option_logits.size == len(OPTION_NAMES)
            ):
                option_distill_target = np.asarray(
                    sequence_distill_targets.get("option", ()),
                    dtype=float,
                )
                if option_distill_target.size == len(OPTION_NAMES):
                    post_rest_sequence_distill_option_loss = (
                        post_rest_sequence_distill_option_weight
                        * cross_entropy_loss(
                            decision.option_logits,
                            option_distill_target,
                        )
                    )
                    post_rest_sequence_distill_loss += (
                        post_rest_sequence_distill_option_loss
                    )
                    handoff_option_teacher_grad_logits += (
                        post_rest_sequence_distill_option_weight
                        * (softmax(decision.option_logits) - option_distill_target)
                    )
                handoff_option_teacher_grad_norm = float(
                    np.linalg.norm(handoff_option_teacher_grad_logits)
                )
            phase_loss = 0.0
            phase_grad_norm = 0.0
            phase_grad_logits = np.zeros_like(decision.phase_logits, dtype=float)
            if (
                self.config.direct_policy_phase_head
                and decision.phase_target_idx >= 0
                and decision.phase_logits.size == len(PHASE_LABELS)
            ):
                phase_target = one_hot(decision.phase_target_idx, len(PHASE_LABELS))
                phase_weight = float(continuation_weights["phase"])
                phase_loss = phase_weight * cross_entropy_loss(
                    decision.phase_logits,
                    phase_target,
                )
                phase_grad_logits = phase_weight * (
                    softmax(decision.phase_logits) - phase_target
                )
                if continuation_margin_weight > 0.0:
                    margin_loss, margin_grad = self._continuation_margin_loss_and_grad(
                        decision.phase_logits,
                        target_idx=int(decision.phase_target_idx),
                        competitor_idx=initial_forage_phase_idx,
                        margin=self.CONTINUATION_MARGIN,
                        scale=continuation_margin_weight * phase_weight,
                    )
                    continuation_margin_loss += float(margin_loss)
                    phase_grad_logits += margin_grad
            if (
                post_rest_sequence_distill_active
                and decision.phase_logits.size == len(PHASE_LABELS)
            ):
                phase_distill_target = np.asarray(
                    sequence_distill_targets.get("phase", ()),
                    dtype=float,
                )
                if phase_distill_target.size == len(PHASE_LABELS):
                    post_rest_sequence_distill_phase_loss = (
                        post_rest_sequence_distill_phase_weight
                        * cross_entropy_loss(
                            decision.phase_logits,
                            phase_distill_target,
                        )
                    )
                    post_rest_sequence_distill_loss += (
                        post_rest_sequence_distill_phase_loss
                    )
                    phase_grad_logits += (
                        post_rest_sequence_distill_phase_weight
                        * (softmax(decision.phase_logits) - phase_distill_target)
                    )
                phase_grad_norm = float(np.linalg.norm(phase_grad_logits))
            continuation_margin_grad_norm = float(
                np.linalg.norm(
                    np.concatenate(
                        [
                            np.asarray(handoff_teacher_grad_logits, dtype=float),
                            np.asarray(handoff_option_teacher_grad_logits, dtype=float),
                            np.asarray(phase_grad_logits, dtype=float),
                        ]
                    )
                )
            )
            affordance_blocked_loss = 0.0
            affordance_blocked_grad_norm = 0.0
            affordance_blocked_grad_logits = np.zeros(
                self.action_dim,
                dtype=float,
            )
            if (
                self.config.direct_policy_affordance_head
                and decision.affordance_blocked_logits.size == self.action_dim
                and decision.affordance_blocked_targets.size == self.action_dim
            ):
                affordance_blocked_target = np.clip(
                    np.asarray(decision.affordance_blocked_targets, dtype=float),
                    0.0,
                    1.0,
                )
                blocked_probs = 1.0 / (
                    1.0 + np.exp(-np.asarray(decision.affordance_blocked_logits, dtype=float))
                )
                blocked_weight = float(continuation_weights["affordance_blocked"])
                affordance_blocked_loss = float(
                    blocked_weight
                    * np.mean(
                        -(
                            affordance_blocked_target * np.log(blocked_probs + 1e-8)
                            + (1.0 - affordance_blocked_target)
                            * np.log(1.0 - blocked_probs + 1e-8)
                        )
                    )
                )
                affordance_blocked_grad_logits = blocked_weight * (
                    blocked_probs - affordance_blocked_target
                ) / max(1, self.action_dim)
                affordance_blocked_grad_norm = float(
                    np.linalg.norm(affordance_blocked_grad_logits)
                )
            affordance_role_loss = 0.0
            affordance_role_grad_norm = 0.0
            affordance_role_grad_logits = np.zeros(
                0,
                dtype=float,
            )
            affordance_role_dim = len(AFFORDANCE_SHELTER_ROLE_NAMES)
            expected_affordance_role_size = self.action_dim * affordance_role_dim
            if (
                self.config.direct_policy_affordance_head
                and decision.affordance_role_logits.size
                == expected_affordance_role_size
                and decision.affordance_role_targets.size == self.action_dim
            ):
                role_weight = float(continuation_weights["affordance_role"])
                affordance_role_grad_matrix = np.zeros(
                    (self.action_dim, affordance_role_dim),
                    dtype=float,
                )
                role_logits_matrix = np.asarray(
                    decision.affordance_role_logits,
                    dtype=float,
                ).reshape(self.action_dim, affordance_role_dim)
                role_targets = np.asarray(
                    decision.affordance_role_targets,
                    dtype=int,
                )
                for action_idx, role_target_idx in enumerate(role_targets.tolist()):
                    role_target = one_hot(role_target_idx, affordance_role_dim)
                    affordance_role_loss += cross_entropy_loss(
                        role_logits_matrix[action_idx],
                        role_target,
                    )
                    affordance_role_grad_matrix[action_idx] = (
                        softmax(role_logits_matrix[action_idx]) - role_target
                    )
                affordance_role_loss = float(
                    role_weight * affordance_role_loss / max(1, self.action_dim)
                )
                affordance_role_grad_logits = (
                    role_weight
                    * affordance_role_grad_matrix.reshape(-1)
                    / max(1, self.action_dim)
                )
                affordance_role_grad_norm = float(
                    np.linalg.norm(affordance_role_grad_logits)
                )
            geometry_loss = 0.0
            geometry_grad_norm = 0.0
            geometry_grad_logits = np.zeros(0, dtype=float)
            geometry_dim = len(AFFORDANCE_GEOMETRY_TARGET_NAMES)
            expected_geometry_size = self.action_dim * geometry_dim
            if (
                self.config.direct_policy_geometry_head
                and decision.geometry_logits.size == expected_geometry_size
                and decision.geometry_targets.size == expected_geometry_size
            ):
                geometry_targets = np.clip(
                    np.asarray(decision.geometry_targets, dtype=float),
                    0.0,
                    1.0,
                )
                geometry_probs = 1.0 / (
                    1.0 + np.exp(-np.asarray(decision.geometry_logits, dtype=float))
                )
                geometry_weight = float(continuation_weights["geometry"])
                geometry_loss = float(
                    geometry_weight
                    * np.mean(
                        -(
                            geometry_targets * np.log(geometry_probs + 1e-8)
                            + (1.0 - geometry_targets)
                            * np.log(1.0 - geometry_probs + 1e-8)
                        )
                    )
                )
                geometry_grad_logits = geometry_weight * (
                    geometry_probs - geometry_targets
                ) / max(1, expected_geometry_size)
                geometry_grad_norm = float(np.linalg.norm(geometry_grad_logits))
            shelter_column_loss = 0.0
            shelter_column_grad_norm = 0.0
            shelter_column_grad_logits = np.zeros(0, dtype=float)
            shelter_column_dim = len(AFFORDANCE_SHELTER_COLUMN_NAMES)
            expected_shelter_column_size = self.action_dim * shelter_column_dim
            if (
                self.config.direct_policy_shelter_column_head
                and decision.shelter_column_logits.size
                == expected_shelter_column_size
                and decision.shelter_column_targets.size == self.action_dim
            ):
                shelter_column_weight = float(
                    continuation_weights["shelter_column"]
                )
                shelter_column_grad_matrix = np.zeros(
                    (self.action_dim, shelter_column_dim),
                    dtype=float,
                )
                shelter_column_logits_matrix = np.asarray(
                    decision.shelter_column_logits,
                    dtype=float,
                ).reshape(self.action_dim, shelter_column_dim)
                shelter_column_targets = np.asarray(
                    decision.shelter_column_targets,
                    dtype=int,
                )
                for action_idx, column_target_idx in enumerate(
                    shelter_column_targets.tolist()
                ):
                    shelter_column_target = one_hot(
                        column_target_idx,
                        shelter_column_dim,
                    )
                    shelter_column_loss += cross_entropy_loss(
                        shelter_column_logits_matrix[action_idx],
                        shelter_column_target,
                    )
                    shelter_column_grad_matrix[action_idx] = (
                        softmax(shelter_column_logits_matrix[action_idx])
                        - shelter_column_target
                    )
                shelter_column_loss = float(
                    shelter_column_weight
                    * shelter_column_loss
                    / max(1, self.action_dim)
                )
                shelter_column_grad_logits = (
                    shelter_column_weight
                    * shelter_column_grad_matrix.reshape(-1)
                    / max(1, self.action_dim)
                )
                shelter_column_grad_norm = float(
                    np.linalg.norm(shelter_column_grad_logits)
                )
            shelter_position_loss = 0.0
            shelter_position_grad_norm = 0.0
            shelter_position_grad_logits = np.zeros(0, dtype=float)
            shelter_position_dim = len(AFFORDANCE_SHELTER_POSITION_NAMES)
            expected_shelter_position_size = self.action_dim * shelter_position_dim
            if (
                self.config.direct_policy_shelter_position_head
                and decision.shelter_position_logits.size
                == expected_shelter_position_size
                and decision.shelter_position_targets.size == self.action_dim
            ):
                shelter_position_weight = float(
                    continuation_weights["shelter_position"]
                )
                shelter_position_grad_matrix = np.zeros(
                    (self.action_dim, shelter_position_dim),
                    dtype=float,
                )
                shelter_position_logits_matrix = np.asarray(
                    decision.shelter_position_logits,
                    dtype=float,
                ).reshape(self.action_dim, shelter_position_dim)
                shelter_position_targets = np.asarray(
                    decision.shelter_position_targets,
                    dtype=int,
                )
                for action_idx, position_target_idx in enumerate(
                    shelter_position_targets.tolist()
                ):
                    shelter_position_target = one_hot(
                        position_target_idx,
                        shelter_position_dim,
                    )
                    shelter_position_loss += cross_entropy_loss(
                        shelter_position_logits_matrix[action_idx],
                        shelter_position_target,
                    )
                    shelter_position_grad_matrix[action_idx] = (
                        softmax(shelter_position_logits_matrix[action_idx])
                        - shelter_position_target
                    )
                shelter_position_loss = float(
                    shelter_position_weight
                    * shelter_position_loss
                    / max(1, self.action_dim)
                )
                shelter_position_grad_logits = (
                    shelter_position_weight
                    * shelter_position_grad_matrix.reshape(-1)
                    / max(1, self.action_dim)
                )
                shelter_position_grad_norm = float(
                    np.linalg.norm(shelter_position_grad_logits)
                )
            transition_prediction_loss = 0.0
            transition_prediction_grad_norm = 0.0
            transition_prediction_grad_logits = np.zeros(0, dtype=float)
            transition_rollout_prediction_loss = 0.0
            transition_rollout_prediction_grad_norm = 0.0
            transition_rollout_prediction_grad_logits = np.zeros(0, dtype=float)
            if (
                getattr(self.config, "direct_policy_transition_prediction_head", False)
                and decision.transition_prediction_logits.size
                == decision.transition_prediction_targets.size
                and decision.transition_prediction_logits.size > 0
            ):
                transition_prediction_targets = np.clip(
                    np.asarray(decision.transition_prediction_targets, dtype=float),
                    -1.0,
                    1.0,
                )
                transition_prediction_values = np.clip(
                    np.asarray(decision.transition_prediction_logits, dtype=float),
                    -1.0,
                    1.0,
                )
                transition_prediction_weight = float(
                    continuation_weights["transition_prediction"]
                )
                transition_prediction_loss = float(
                    transition_prediction_weight
                    * np.mean(
                        (transition_prediction_values - transition_prediction_targets)
                        ** 2
                    )
                )
                transition_prediction_grad_logits = (
                    2.0
                    * transition_prediction_weight
                    * (transition_prediction_values - transition_prediction_targets)
                    / max(1, transition_prediction_values.size)
                )
                transition_prediction_grad_norm = float(
                    np.linalg.norm(transition_prediction_grad_logits)
                )
            if (
                getattr(
                    self.config,
                    "direct_policy_transition_rollout_prediction_head",
                    False,
                )
                and decision.transition_rollout_prediction_logits.size
                == decision.transition_rollout_prediction_targets.size
                and decision.transition_rollout_prediction_logits.size > 0
            ):
                transition_rollout_prediction_targets = np.clip(
                    np.asarray(
                        decision.transition_rollout_prediction_targets,
                        dtype=float,
                    ),
                    -1.0,
                    1.0,
                )
                transition_rollout_prediction_values = np.clip(
                    np.asarray(
                        decision.transition_rollout_prediction_logits,
                        dtype=float,
                    ),
                    -1.0,
                    1.0,
                )
                transition_rollout_prediction_weight = float(
                    continuation_weights["transition_rollout_prediction"]
                )
                transition_rollout_prediction_loss = float(
                    transition_rollout_prediction_weight
                    * np.mean(
                        (
                            transition_rollout_prediction_values
                            - transition_rollout_prediction_targets
                        )
                        ** 2
                    )
                )
                transition_rollout_prediction_grad_logits = (
                    2.0
                    * transition_rollout_prediction_weight
                    * (
                        transition_rollout_prediction_values
                        - transition_rollout_prediction_targets
                    )
                    / max(1, transition_rollout_prediction_values.size)
                )
                transition_rollout_prediction_grad_norm = float(
                    np.linalg.norm(transition_rollout_prediction_grad_logits)
                )
            true_monolithic_grad = np.concatenate(
                [
                    np.asarray(
                        grad_policy_logits + handoff_teacher_grad_logits,
                        dtype=float,
                    ),
                    np.array([value_grad], dtype=float),
                    np.asarray(phase_grad_logits, dtype=float),
                    np.asarray(affordance_blocked_grad_logits, dtype=float),
                    np.asarray(affordance_role_grad_logits, dtype=float),
                    np.asarray(geometry_grad_logits, dtype=float),
                    np.asarray(shelter_column_grad_logits, dtype=float),
                    np.asarray(shelter_position_grad_logits, dtype=float),
                    np.asarray(transition_prediction_grad_logits, dtype=float),
                    np.asarray(
                        transition_rollout_prediction_grad_logits,
                        dtype=float,
                    ),
                ]
            )
            if self.TRUE_MONOLITHIC_POLICY_NAME not in self._frozen_modules:
                backward_kwargs = {
                    "grad_policy_logits": (
                        np.asarray(grad_policy_logits, dtype=float)
                        + handoff_teacher_grad_logits
                    ),
                    "grad_value": value_grad,
                    "lr": self.module_lr,
                }
                if (
                    self.config.direct_policy_handoff_option_teacher
                    and decision.option_logits.size == len(OPTION_NAMES)
                ):
                    backward_kwargs["grad_option_logits"] = (
                        handoff_option_teacher_grad_logits
                    )
                if (
                    hasattr(self.true_monolithic_policy, "phase_output_dim")
                    and phase_grad_logits.size > 0
                ):
                    backward_kwargs["grad_phase_logits"] = phase_grad_logits
                if hasattr(self.true_monolithic_policy, "affordance_role_dim"):
                    backward_kwargs["grad_affordance_blocked_logits"] = (
                        affordance_blocked_grad_logits
                    )
                    if affordance_role_grad_logits.size > 0:
                        backward_kwargs["grad_affordance_role_logits"] = (
                            affordance_role_grad_logits
                        )
                if (
                    hasattr(self.true_monolithic_policy, "geometry_dim")
                    and geometry_grad_logits.size > 0
                ):
                    backward_kwargs["grad_geometry_logits"] = geometry_grad_logits
                if (
                    hasattr(self.true_monolithic_policy, "shelter_column_dim")
                    and shelter_column_grad_logits.size > 0
                ):
                    backward_kwargs["grad_shelter_column_logits"] = (
                        shelter_column_grad_logits
                    )
                if (
                    hasattr(self.true_monolithic_policy, "shelter_position_dim")
                    and shelter_position_grad_logits.size > 0
                ):
                    backward_kwargs["grad_shelter_position_logits"] = (
                        shelter_position_grad_logits
                    )
                if transition_prediction_grad_logits.size > 0:
                    backward_kwargs["grad_transition_prediction_logits"] = (
                        transition_prediction_grad_logits
                    )
                if transition_rollout_prediction_grad_logits.size > 0:
                    backward_kwargs["grad_transition_rollout_prediction_logits"] = (
                        transition_rollout_prediction_grad_logits
                    )
                self.true_monolithic_policy.backward(**backward_kwargs)
            continuation_replay_passes_applied = 0
            continuation_replay_total_loss = 0.0
            continuation_replay_total_grad_norm = 0.0
            continuation_replay_passes = int(
                getattr(
                    self.config,
                    "direct_policy_continuation_replay_passes",
                    0,
                )
            )
            continuation_replay_lr_scale = float(
                getattr(
                    self.config,
                    "direct_policy_continuation_replay_lr_scale",
                    0.0,
                )
            )
            if post_rest_sequence_replay_boost_active:
                continuation_replay_passes = max(
                    continuation_replay_passes,
                    self.POST_REST_SEQUENCE_REPLAY_PASSES,
                )
                continuation_replay_lr_scale = max(
                    continuation_replay_lr_scale,
                    self.POST_REST_SEQUENCE_REPLAY_LR_SCALE,
                )
            if (
                continuation_replay_passes > 0
                and continuation_replay_lr_scale > 0.0
                and self._continuation_replay_focus_active(decision)
            ):
                for _ in range(continuation_replay_passes):
                    replay_stats = self._true_monolithic_continuation_replay_step(
                        decision,
                        lr_scale=continuation_replay_lr_scale,
                    )
                    if not bool(replay_stats.get("active", False)):
                        continue
                    continuation_replay_passes_applied += 1
                    continuation_replay_total_loss += float(
                        replay_stats.get("loss", 0.0)
                    )
                    continuation_replay_total_grad_norm += float(
                        replay_stats.get("grad_norm", 0.0)
                    )
            module_gradient_norms = {
                self.TRUE_MONOLITHIC_POLICY_NAME: float(
                    0.0
                    if self.TRUE_MONOLITHIC_POLICY_NAME in self._frozen_modules
                    else np.linalg.norm(true_monolithic_grad)
                )
            }
            return {
                "reward": float(reward),
                "td_target": float(td_target),
                "td_error": float(advantage),
                "value": float(decision.value),
                "next_value": float(next_value),
                "entropy": entropy,
                "handoff_teacher_loss": float(handoff_teacher_loss),
                "handoff_teacher_grad_norm": float(handoff_teacher_grad_norm),
                "handoff_teacher_weight": float(handoff_teacher_weight),
                "handoff_teacher_target_idx": int(decision.teacher_action_target_idx),
                "handoff_teacher_active": bool(
                    decision.teacher_action_target_idx >= 0
                ),
                "handoff_option_teacher_loss": float(
                    handoff_option_teacher_loss
                ),
                "handoff_option_teacher_grad_norm": float(
                    handoff_option_teacher_grad_norm
                ),
                "handoff_option_teacher_weight": float(
                    handoff_option_teacher_weight
                ),
                "handoff_option_teacher_target_idx": int(
                    decision.teacher_option_target_idx
                ),
                "handoff_option_teacher_active": bool(
                    decision.teacher_option_target_idx >= 0
                ),
                "phase_loss": float(phase_loss),
                "phase_weight": float(continuation_weights["phase"]),
                "phase_grad_norm": float(phase_grad_norm),
                "phase_target": (
                    ""
                    if decision.phase_target is None
                    else str(decision.phase_target)
                ),
                "phase_prediction": (
                    ""
                    if decision.phase_prediction is None
                    else str(decision.phase_prediction)
                ),
                "phase_prediction_confidence": float(
                    decision.phase_prediction_confidence
                ),
                "continuation_weight_profile": dict(continuation_weights),
                "post_rest_sequence_replay_boost_active": bool(
                    post_rest_sequence_replay_boost_active
                ),
                "post_rest_sequence_distill_active": bool(
                    post_rest_sequence_distill_active
                ),
                "post_rest_sequence_distill_loss": float(
                    post_rest_sequence_distill_loss
                ),
                "post_rest_sequence_distill_action_loss": float(
                    post_rest_sequence_distill_action_loss
                ),
                "post_rest_sequence_distill_option_loss": float(
                    post_rest_sequence_distill_option_loss
                ),
                "post_rest_sequence_distill_phase_loss": float(
                    post_rest_sequence_distill_phase_loss
                ),
                "continuation_margin_weight": float(continuation_margin_weight),
                "continuation_margin_loss": float(continuation_margin_loss),
                "continuation_margin_grad_norm": float(
                    continuation_margin_grad_norm
                ),
                "continuation_replay_passes_configured": int(
                    continuation_replay_passes
                ),
                "continuation_replay_passes_applied": int(
                    continuation_replay_passes_applied
                ),
                "continuation_replay_lr_scale": float(
                    continuation_replay_lr_scale
                ),
                "continuation_replay_total_loss": float(
                    continuation_replay_total_loss
                ),
                "continuation_replay_total_grad_norm": float(
                    continuation_replay_total_grad_norm
                ),
                "affordance_blocked_loss": float(affordance_blocked_loss),
                "affordance_blocked_weight": float(
                    continuation_weights["affordance_blocked"]
                ),
                "affordance_blocked_grad_norm": float(
                    affordance_blocked_grad_norm
                ),
                "affordance_role_loss": float(affordance_role_loss),
                "affordance_role_weight": float(
                    continuation_weights["affordance_role"]
                ),
                "affordance_role_grad_norm": float(affordance_role_grad_norm),
                "geometry_loss": float(geometry_loss),
                "geometry_weight": float(continuation_weights["geometry"]),
                "geometry_grad_norm": float(geometry_grad_norm),
                "shelter_column_loss": float(shelter_column_loss),
                "shelter_column_weight": float(
                    continuation_weights["shelter_column"]
                ),
                "shelter_column_grad_norm": float(shelter_column_grad_norm),
                "shelter_position_loss": float(shelter_position_loss),
                "shelter_position_weight": float(
                    continuation_weights["shelter_position"]
                ),
                "shelter_position_grad_norm": float(shelter_position_grad_norm),
                "transition_prediction_loss": float(
                    transition_prediction_loss
                ),
                "transition_prediction_weight": float(
                    continuation_weights["transition_prediction"]
                ),
                "transition_prediction_grad_norm": float(
                    transition_prediction_grad_norm
                ),
                "transition_rollout_prediction_loss": float(
                    transition_rollout_prediction_loss
                ),
                "transition_rollout_prediction_weight": float(
                    continuation_weights["transition_rollout_prediction"]
                ),
                "transition_rollout_prediction_grad_norm": float(
                    transition_rollout_prediction_grad_norm
                ),
                "aux_modules": 0.0,
                "arbitration_value": 0.0,
                "arbitration_value_grad": 0.0,
                "arbitration_grad_valence_norm": 0.0,
                "arbitration_grad_gate_norm": 0.0,
                "arbitration_gate_regularization_norm": 0.0,
                "arbitration_valence_regularization_norm": 0.0,
                "arbitration_loss": 0.0,
                "gate_adjustment_magnitude": 0.0,
                "regularization_loss": 0.0,
                "credit_strategy": effective_credit_strategy,
                "module_credit_weights": {
                    name: float(value)
                    for name, value in sorted(module_credit_weights.items())
                },
                "module_gradient_norms": module_gradient_norms,
                "counterfactual_credit_weights": {},
                "route_mask_enabled": False,
                "route_mask_threshold": route_mask_threshold,
                "route_active_modules": [],
                "route_credit_weights": {},
            }
        action_center_value_grad = decision.value - td_target
        if self.action_center is None:
            raise RuntimeError("Action center unavailable for the configured architecture.")
        action_center_input_grads = self.action_center.backward(
            grad_policy_logits=grad_policy_logits,
            grad_value=action_center_value_grad,
            lr=self.motor_lr,
        )
        proposal_grad_width = self.action_dim * len(decision.module_results)
        proposal_input_grads = np.asarray(
            action_center_input_grads[:proposal_grad_width],
            dtype=float,
        )
        per_result_input_grads = proposal_input_grads.reshape(
            len(decision.module_results),
            self.action_dim,
        )

        arbitration = decision.arbitration_decision
        arbitration_grad_valence_norm = 0.0
        arbitration_grad_gate_norm = 0.0
        arbitration_gate_regularization_norm = 0.0
        arbitration_valence_regularization_norm = 0.0
        arbitration_value_grad = 0.0
        arbitration_loss = 0.0
        regularization_loss = 0.0
        gate_adjustment_magnitude = 0.0
        if arbitration is not None and arbitration.gate_adjustments:
            gate_adjustment_magnitude = float(
                np.mean(
                    [
                        abs(float(adjustment) - 1.0)
                        for adjustment in arbitration.gate_adjustments.values()
                    ]
                )
            )
        if (
            arbitration is not None
            and arbitration.learned_adjustment
            and self.config.use_learned_arbitration
            and self.arbitration_network.cache is not None
        ):
            valence_probs = np.array(
                [
                    float(arbitration.valence_scores.get(name, 0.0))
                    for name in self.VALENCE_ORDER
                ],
                dtype=float,
            )
            winning_valence_idx = self.VALENCE_ORDER.index(arbitration.winning_valence)
            grad_valence = advantage * (
                valence_probs - one_hot(winning_valence_idx, len(self.VALENCE_ORDER))
            )

            valence_reg = np.zeros_like(grad_valence)
            valence_regularization_loss = 0.0
            if self.arbitration_valence_regularization_weight > 0.0:
                fixed_valence_targets = self._fixed_formula_valence_scores_from_evidence(
                    arbitration.evidence,
                )
                valence_reg = (
                    self.arbitration_valence_regularization_weight
                    * (valence_probs - fixed_valence_targets)
                )
                grad_valence += valence_reg
                valence_regularization_loss = float(
                    0.5
                    * self.arbitration_valence_regularization_weight
                    * np.sum((valence_probs - fixed_valence_targets) ** 2)
                )

            grad_final_gates = np.zeros(
                len(self.ARBITRATION_GATE_MODULE_ORDER),
                dtype=float,
            )
            gate_regularization = np.zeros_like(grad_final_gates)
            gate_deviation = np.zeros_like(grad_final_gates)
            base_gate_by_index = np.zeros_like(grad_final_gates)
            module_gate_indices = {
                name: index
                for index, name in enumerate(self.ARBITRATION_GATE_MODULE_ORDER)
            }
            for result, extra_grad in zip(
                decision.module_results,
                per_result_input_grads,
                strict=True,
            ):
                gate_index = module_gate_indices.get(result.name)
                if gate_index is None:
                    continue
                pre_gate_logits = (
                    result.post_reflex_logits
                    if result.post_reflex_logits is not None
                    else result.gated_logits
                )
                if pre_gate_logits is None:
                    continue
                base_gate = float(arbitration.base_gates.get(result.name, 0.0))
                learned_gate = float(arbitration.module_gates.get(result.name, base_gate))
                base_gate_by_index[gate_index] = base_gate
                gate_deviation[gate_index] = learned_gate - base_gate
                grad_final_gates[gate_index] += float(
                    np.dot(
                        np.asarray(extra_grad, dtype=float),
                        np.asarray(pre_gate_logits, dtype=float),
                    )
                )
                gate_regularization[gate_index] += float(
                    self.arbitration_regularization_weight
                    * (learned_gate - base_gate)
                )

            # action_center_input_grads are gradients with respect to gated
            # logits. Project through gated_logits_i = gate_i * raw_logits_i,
            # then through gate_i ~= base_gate_i * learned_adjustment_i. This
            # uses the requested first-order approximation and ignores only the
            # final diagnostic [0, 1] clamp.
            grad_gates = base_gate_by_index * (grad_final_gates + gate_regularization)
            arbitration_value_grad = float(arbitration.arbitration_value - td_target)
            gate_regularization_loss = float(
                0.5
                * self.arbitration_regularization_weight
                * np.sum(gate_deviation**2)
            )
            regularization_loss = gate_regularization_loss + valence_regularization_loss
            selected_valence_prob = max(float(valence_probs[winning_valence_idx]), 1e-8)
            arbitration_policy_loss = float(-advantage * np.log(selected_valence_prob))
            arbitration_value_loss = float(0.5 * arbitration_value_grad * arbitration_value_grad)
            arbitration_loss = arbitration_policy_loss + arbitration_value_loss + regularization_loss
            # Keep the warm-start valence and gate behavior stable in proportion
            # to the baseline regularizer without fully freezing the default.
            anchor = float(np.clip(self.arbitration_regularization_weight, 0.0, 1.0))
            _anchored_param_names = ("W1", "b1", "W2_valence", "b2_valence", "W2_gate", "b2_gate")
            anchored_params: dict[str, np.ndarray] = (
                {name: getattr(self.arbitration_network, name).copy() for name in _anchored_param_names}
                if anchor > 0.0
                else {}
            )
            self.arbitration_network.backward(
                grad_valence_logits=grad_valence,
                grad_gate_adjustments=grad_gates,
                grad_value=arbitration_value_grad,
                lr=self.arbitration_lr,
            )
            for param_name, pre_update in anchored_params.items():
                post_update = getattr(self.arbitration_network, param_name)
                setattr(
                    self.arbitration_network,
                    param_name,
                    anchor * pre_update + (1.0 - anchor) * post_update,
                )
            arbitration_grad_valence_norm = float(np.linalg.norm(grad_valence))
            arbitration_grad_gate_norm = float(np.linalg.norm(grad_gates))
            arbitration_gate_regularization_norm = float(np.linalg.norm(gate_regularization))
            arbitration_valence_regularization_norm = float(np.linalg.norm(valence_reg))

        module_total_grads: Dict[str, np.ndarray] = {}
        module_gradient_norms: Dict[str, float] = {}
        if self.config.is_modular:
            if self.module_bank is None:
                raise RuntimeError("Module bank unavailable for modular architecture.")
            trainable_module_names: list[str] = []
            route_active_name_set = set(route_active_modules)
            for result, extra_grad in zip(
                decision.module_results,
                per_result_input_grads,
                strict=True,
            ):
                if not result.active:
                    continue
                if uses_route_mask_credit and result.name not in route_active_name_set:
                    module_gradient_norms[result.name] = 0.0
                    continue
                if result.name in self._frozen_modules:
                    module_gradient_norms[result.name] = 0.0
                    continue
                gate_weight = float(result.gate_weight)
                total_grad = gate_weight * np.asarray(
                    reflex_aux_grads.get(
                        result.name,
                        np.zeros(self.action_dim, dtype=float),
                    ),
                    dtype=float,
                )
                if uses_counterfactual_credit:
                    cf_weight = float(counterfactual_credit_weights.get(result.name, 0.0))
                    total_grad += gate_weight * cf_weight * grad_policy_logits
                elif not uses_local_credit_only:
                    total_grad += gate_weight * grad_policy_logits
                total_grad += gate_weight * np.asarray(extra_grad, dtype=float)
                module_total_grads[result.name] = module_total_grads.get(
                    result.name,
                    np.zeros(self.action_dim, dtype=float),
                ) + total_grad
                module_gradient_norms[result.name] = float(np.linalg.norm(total_grad))
                trainable_module_names.append(result.name)
            if trainable_module_names:
                original_active_names = list(self.module_bank._active_names)
                try:
                    self.module_bank._active_names = trainable_module_names
                    self.module_bank.backward(
                        np.zeros(self.action_dim, dtype=float),
                        lr=self.module_lr,
                        aux_grads=module_total_grads,
                    )
                finally:
                    self.module_bank._active_names = original_active_names
        else:
            if self.monolithic_policy is None:
                raise RuntimeError("Monolithic network unavailable for the configured architecture.")
            grad_for_monolithic = grad_policy_logits + proposal_input_grads
            if self.MONOLITHIC_POLICY_NAME in self._frozen_modules:
                module_gradient_norms[self.MONOLITHIC_POLICY_NAME] = 0.0
            else:
                module_gradient_norms[self.MONOLITHIC_POLICY_NAME] = float(
                    np.linalg.norm(grad_for_monolithic)
                )
                self.monolithic_policy.backward(grad_for_monolithic, lr=self.module_lr)
        if self.motor_cortex is None:
            raise RuntimeError("Motor cortex unavailable for the configured architecture.")
        self.motor_cortex.backward(grad_policy_logits, lr=self.motor_lr)

        entropy = -float(np.sum(decision.policy * np.log(decision.policy + 1e-8)))
        return {
            "reward": float(reward),
            "td_target": float(td_target),
            "td_error": float(advantage),
            "value": float(decision.value),
            "next_value": float(next_value),
            "entropy": entropy,
            "aux_modules": float(len(reflex_aux_grads)),
            "arbitration_value": float(arbitration.arbitration_value) if arbitration is not None else 0.0,
            "arbitration_value_grad": float(arbitration_value_grad),
            "arbitration_grad_valence_norm": arbitration_grad_valence_norm,
            "arbitration_grad_gate_norm": arbitration_grad_gate_norm,
            "arbitration_gate_regularization_norm": arbitration_gate_regularization_norm,
            "arbitration_valence_regularization_norm": arbitration_valence_regularization_norm,
            "arbitration_loss": float(arbitration_loss),
            "gate_adjustment_magnitude": float(gate_adjustment_magnitude),
            "regularization_loss": float(regularization_loss),
            "credit_strategy": effective_credit_strategy,
            "module_credit_weights": {
                name: float(value)
                for name, value in sorted(module_credit_weights.items())
            },
            "module_gradient_norms": {
                name: float(value)
                for name, value in sorted(module_gradient_norms.items())
            },
            "counterfactual_credit_weights": {
                name: float(value)
                for name, value in sorted(counterfactual_credit_weights.items())
            },
            "route_mask_enabled": bool(route_mask_enabled),
            "route_mask_threshold": float(route_mask_threshold),
            "route_active_modules": list(route_active_modules),
            "route_credit_weights": {
                name: float(value)
                for name, value in sorted(route_credit_weights.items())
            },
        }
